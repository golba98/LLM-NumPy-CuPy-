"""Bounded real-corpus validation; corpus acquisition stays outside the framework."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_numpy.config import LLMConfig
from llm_numpy.data import DataLoader, LanguageModelDataset, TextCorpus
from llm_numpy.nn.model import TinyLLM
from llm_numpy.optim.adamw import AdamW
from llm_numpy.tokenization import ByteLevelBPETokenizer
from llm_numpy.training.config import TrainingConfig
from llm_numpy.training.metrics import token_accuracy
from llm_numpy.training.trainer import Trainer
from llm_numpy.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    from llm_numpy.assets import normalize_arguments
    args = normalize_arguments(parser.parse_args())
    set_seed(args.seed)
    corpus = TextCorpus.from_file(args.input)
    train_corpus, val_corpus = corpus.split(0.9)
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(train_corpus.documents, vocab_size=512)
    train_ids = train_corpus.token_ids(tokenizer, tokenizer.EOS_ID)
    val_ids = val_corpus.token_ids(tokenizer, tokenizer.EOS_ID)
    train_dataset = LanguageModelDataset(train_ids, args.context)
    val_dataset = LanguageModelDataset(val_ids, args.context)
    train_loader = DataLoader(train_dataset, args.batch_size, seed=args.seed)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
    model_config = LLMConfig(vocab_size=len(tokenizer.vocab), max_seq_len=args.context,
                             dim=64, num_layers=2, num_heads=4, hidden_dim=176)
    model = TinyLLM(model_config)
    training_config = TrainingConfig(batch_size=args.batch_size, context_length=args.context,
                                     learning_rate=args.learning_rate, weight_decay=0.0,
                                     max_steps=args.steps, accumulation_steps=args.gradient_accumulation,
                                     eval_interval=max(1, args.steps // 5), checkpoint_interval=0,
                                     seed=args.seed, log_csv="logs/real_corpus.csv")
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    initial = Trainer(model, optimizer, train_loader, val_loader, training_config)
    initial_val = initial.evaluate()
    prompt_ids = tokenizer.encode("That")
    before_text = tokenizer.decode(model.generate(prompt_ids, max_new_tokens=12, greedy=True)[0])
    history = initial.train()
    after_text = tokenizer.decode(model.generate(prompt_ids, max_new_tokens=12, greedy=True)[0])
    best = min(zip(history["step"], history["val_loss"]), key=lambda item: item[1])
    print("NumPy TinyLLM - Real Corpus Validation")
    print(f"characters={len(corpus.documents[0])} train_tokens={len(train_ids)} val_tokens={len(val_ids)}")
    print(f"vocab={len(tokenizer.vocab)} merges={len(tokenizer.merges)} parameters={sum(p.data.size for p in model.parameters())}")
    print(f"initial_val_loss={initial_val['loss']:.6f} best_val_loss={best[1]:.6f} best_step={best[0]}")
    print(f"final_train_loss={history['train_loss'][-1]:.6f} final_train_accuracy={history['train_accuracy'][-1]:.2%}")
    print("generation_before:", before_text)
    print("generation_after:", after_text)


if __name__ == "__main__":
    main()
