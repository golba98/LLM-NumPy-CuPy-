"""Train from a local UTF-8 text file without downloading data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from src.config import LLMConfig
from src.data import DataLoader, LanguageModelDataset, TextCorpus
from src.nn.model import TinyLLM
from src.optim.adamw import AdamW
from src.tokenization import ByteLevelBPETokenizer
from src.training.config import TrainingConfig
from src.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text_file", type=Path)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
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
