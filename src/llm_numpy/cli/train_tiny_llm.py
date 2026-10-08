"""Train from a local UTF-8 text file without downloading data."""
from llm_numpy.assets import workspace_root, normalize_arguments, resolve_input_path, resolve_output_path
import sys
from pathlib import Path

import argparse

from llm_numpy.config import LLMConfig
from llm_numpy.data import DataLoader, LanguageModelDataset, TextCorpus
from llm_numpy.nn.model import TinyLLM
from llm_numpy.optim.adamw import AdamW
from llm_numpy.tokenization import ByteLevelBPETokenizer
from llm_numpy.training.config import TrainingConfig
from llm_numpy.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text_file", type=Path)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=100)
    args = normalize_arguments(parser.parse_args())
    corpus = TextCorpus.from_file(args.text_file, drop_empty=True)
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(corpus.documents, args.vocab_size)
    ids = corpus.token_ids(tokenizer, tokenizer.EOS_ID)
    train_config = TrainingConfig(context_length=32, max_steps=args.steps, log_csv="logs/train.csv")
    dataset = LanguageModelDataset(ids, train_config.context_length)
    loader = DataLoader(dataset, train_config.batch_size, seed=train_config.seed)
    model_config = LLMConfig(vocab_size=len(tokenizer.vocab), max_seq_len=train_config.context_length,
                             dim=64, num_layers=2, num_heads=4, hidden_dim=176)
    model = TinyLLM(model_config)
    optimizer = AdamW(model.parameters(), lr=train_config.learning_rate,
                      betas=(train_config.beta1, train_config.beta2), eps=train_config.adam_eps,
                      weight_decay=train_config.weight_decay)
    history = Trainer(model, optimizer, loader, config=train_config).train()
    print(f"documents={len(corpus)} tokens={len(ids)} vocab={len(tokenizer.vocab)}")
    print(f"final train loss={history['train_loss'][-1]:.6f}")


if __name__ == "__main__":
    main()
