"""Existing tier configurations used by training and inference."""
from llm_numpy.config import LLMConfig

def architecture(tier: str, vocab_size: int, context: int) -> LLMConfig:
    if tier == "tiny":
        return LLMConfig(vocab_size=vocab_size, max_seq_len=context, dim=128, num_layers=2, num_heads=4, hidden_dim=352)
    if tier == "small":
        return LLMConfig(vocab_size=vocab_size, max_seq_len=context, dim=320, num_layers=8, num_heads=8, hidden_dim=896)
    if tier == "medium":
        return LLMConfig(vocab_size=vocab_size, max_seq_len=context, dim=512, num_layers=14, num_heads=8, hidden_dim=1408)
    if tier == "large":
        return LLMConfig(vocab_size=vocab_size, max_seq_len=context, dim=640, num_layers=18, num_heads=10, hidden_dim=1760)
    if tier == "target":
        return LLMConfig(vocab_size=vocab_size, max_seq_len=context, dim=768, num_layers=34, num_heads=12, hidden_dim=2048)
    raise ValueError(f"unknown tier: {tier}")
