import csv
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from llm_numpy.data.dataloader import DataLoader
from llm_numpy.backend import array_module, to_cpu, to_device
from llm_numpy.nn.model import TinyLLM, perplexity
from llm_numpy.optim.adamw import AdamW
from llm_numpy.optim.clip import clip_grad_norm_
from llm_numpy.tensor import no_grad
from llm_numpy.training.checkpoint import save_checkpoint
from llm_numpy.training.config import TrainingConfig
from llm_numpy.training.metrics import finite_or_raise, parameter_finiteness_or_raise, token_accuracy
from llm_numpy.training.schedules import learning_rate
from llm_numpy.training.profiler import StepProfiler
from llm_numpy.training.progress import LiveProgress


class Trainer:
    """An explicit NumPy optimization loop with metrics and safe checkpoints."""

    def __init__(self, model: TinyLLM, optimizer: AdamW, train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None, config: Optional[TrainingConfig] = None):
        self.model, self.optimizer = model, optimizer
        self.train_loader, self.val_loader = train_loader, val_loader
        self.config = config or TrainingConfig()
        dtype = {"float16": np.float16, "float32": np.float32, "float64": np.float64}[self.config.dtype]
        self.model.to(self.config.device, dtype=dtype)
        self.optimizer.to(self.model.device, dtype=dtype)
        self.step = 0
        self.epoch = 0
        self.tokens_processed = 0
        self.profiler: Optional[StepProfiler] = None
        self.best_val_loss = float("inf")
        self._consecutive_small_gradient_steps = 0

    def _set_learning_rate(self) -> float:
        value = learning_rate(self.step, self.config.learning_rate, self.config.warmup_steps,
                              self.config.cosine_decay_steps, self.config.min_learning_rate)
        for group in self.optimizer.param_groups:
            group["lr"] = value
        return value

    def train_accumulated_step(self, microbatches) -> Dict[str, float]:
        """Optimize a token-weighted mean over microbatches, then step once.

        If microbatch i contains n_i target tokens and N is the total, its
        already-mean loss is backpropagated with weight n_i / N. This gives
        the same gradient as one mean loss over the concatenated targets.
        """
        batches = list(microbatches)
        if not batches:
            raise ValueError("at least one microbatch is required")
        total_tokens = sum(int(np.prod(targets.shape)) for _, targets in batches)
        self.optimizer.zero_grad()
        total_nll = 0.0
        correct = 0
        for inputs, targets in batches:
            device = self.model.device
            inputs = to_device(inputs, device)
            targets = to_device(targets, device)
            logits, loss = self.model(inputs, targets=targets)
            count = int(np.prod(targets.shape))
            value = float(loss.data.item())
            finite_or_raise("loss", value, self.step)
            total_nll += value * count
            correct += int(to_cpu(array_module(logits.data).argmax(logits.data, axis=-1) == targets).sum())
            (loss * (count / total_tokens) * self.config.loss_scale).backward()
        if self.config.loss_scale != 1.0:
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    parameter.grad /= self.config.loss_scale
        grad_norm = clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        finite_or_raise("gradient norm", grad_norm, self.step)
        if grad_norm <= self.config.min_gradient_norm:
            self._consecutive_small_gradient_steps += 1
            if (self.config.max_consecutive_small_gradient_steps
                    and self._consecutive_small_gradient_steps
                    >= self.config.max_consecutive_small_gradient_steps):
                raise FloatingPointError(
                    "gradient norm remained at or below "
                    f"{self.config.min_gradient_norm} for "
                    f"{self._consecutive_small_gradient_steps} consecutive steps "
                    f"ending at step {self.step}"
                )
        else:
            self._consecutive_small_gradient_steps = 0
        learning_rate_value = self._set_learning_rate()
        self.optimizer.step()
        parameter_finiteness_or_raise(self.model.parameters(), self.step)
        self.optimizer.zero_grad()
        self.tokens_processed += total_tokens
        loss_value = total_nll / total_tokens
        return {"loss": loss_value, "perplexity": perplexity(loss_value),
                "accuracy": correct / total_tokens, "gradient_norm": grad_norm,
                "learning_rate": learning_rate_value, "tokens": total_tokens}

    def train_step(self, inputs: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        return self.train_accumulated_step([(inputs, targets)])

    def evaluate(self) -> Dict[str, float]:
        if self.val_loader is None or len(self.val_loader) == 0:
            return {"loss": 0.0, "perplexity": 0.0, "accuracy": 0.0, "tokens": 0}
        total_nll = 0.0
        total_tokens = 0
        correct = 0
        with no_grad():
            for batch_index, (inputs, targets) in enumerate(self.val_loader):
                if self.config.eval_max_batches and batch_index >= self.config.eval_max_batches:
                    break
                inputs = to_device(inputs, self.model.device)
                targets = to_device(targets, self.model.device)
                logits, loss = self.model(inputs, targets=targets)
                count = int(np.prod(targets.shape))
                total_nll += float(loss.data.item()) * count
                total_tokens += count
                correct += int(to_cpu((array_module(logits.data).argmax(logits.data, axis=-1)) == targets).sum())
        mean_loss = total_nll / max(total_tokens, 1)
        return {"loss": mean_loss, "perplexity": perplexity(mean_loss),
                "accuracy": correct / max(total_tokens, 1), "tokens": total_tokens}

    def train(self) -> Dict[str, List[float]]:
        history: Dict[str, List[float]] = {key: [] for key in (
            "step", "epoch", "tokens_processed", "train_loss", "train_perplexity",
            "train_accuracy", "val_loss", "val_perplexity", "val_accuracy",
            "learning_rate", "gradient_norm", "tokens_per_second", "step_time")}
        if self.config.log_csv:
            directory = os.path.dirname(self.config.log_csv)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(self.config.log_csv, "w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(["step", "epoch", "tokens", "train_loss", "train_perplexity",
                                             "train_accuracy", "val_loss", "val_perplexity", "val_accuracy",
                                             "learning_rate", "gradient_norm", "tokens_per_second", "step_time"])

        started = time.perf_counter()
        progress = LiveProgress(self.config.run_name, self.config.target_tokens) if self.config.progress else None
        last_val = self.evaluate() if self.val_loader is not None else {"loss": 0.0, "perplexity": 0.0, "accuracy": 0.0}
        while self.step < self.config.max_steps:
            self.train_loader.set_epoch(self.epoch)
            batch_iterator = iter(self.train_loader)
            for inputs, targets in batch_iterator:
                if self.step >= self.config.max_steps:
                    break
                step_started = time.perf_counter()
                microbatches = [(inputs, targets)]
                while len(microbatches) < self.config.accumulation_steps:
                    try:
                        microbatches.append(next(batch_iterator))
                    except StopIteration:
                        break
                metrics = self.train_accumulated_step(microbatches)
                self.step += 1
                elapsed = time.perf_counter() - step_started
                if self.step % self.config.eval_interval == 0 or self.step == self.config.max_steps:
                    last_val = self.evaluate()
                    self.best_val_loss = min(self.best_val_loss, float(last_val.get("loss", float("inf"))))
                val = last_val
                row = {
                    "step": self.step, "epoch": self.epoch, "tokens_processed": self.tokens_processed,
                    "train_loss": metrics["loss"], "train_perplexity": metrics["perplexity"],
                    "train_accuracy": metrics["accuracy"], "val_loss": val.get("loss", 0.0),
                    "val_perplexity": val.get("perplexity", 0.0), "val_accuracy": val.get("accuracy", 0.0),
                    "learning_rate": metrics["learning_rate"], "gradient_norm": metrics["gradient_norm"],
                    "tokens_per_second": metrics["tokens"] / max(elapsed, 1e-12), "step_time": elapsed,
                }
                for key, value in row.items():
                    history[key].append(value)
                if self.config.log_csv:
                    with open(self.config.log_csv, "a", newline="", encoding="utf-8") as handle:
                        csv.DictWriter(handle, fieldnames=list(row)).writerow(row)
                if self.config.checkpoint_interval and self.step % self.config.checkpoint_interval == 0:
                    checkpoint_dir = self.config.checkpoint_dir or "checkpoints"
                    save_checkpoint(os.path.join(checkpoint_dir, f"checkpoint_step_{self.step:06d}.npz"), self.model,
                                    self.optimizer, self.step, self.epoch, self.tokens_processed,
                                    getattr(self.model, "config", None), self.config,
                                    tokenizer_path=self.config.tokenizer_path,
                                    best_val_loss=self.best_val_loss,
                                    dataset_manifest_sha256=self.config.dataset_manifest_sha256 or None)
                if progress:
                    progress.update(row)
            self.epoch += 1
        if progress:
            progress.close()
        return history
