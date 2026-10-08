# Next training set

Current checkpoint: `runs/pretraining/pretrain_250m_fineweb_followup_10m/target_final.npz`

It has processed about 18.4M tokens and reaches validation loss 4.585, but the
Codexa-dev smoke test shows that it is still a base language model, not a
usable chat model. The next run should therefore be chosen by outcome:

## Option A — conversational SFT (recommended)

Teach the existing checkpoint to answer user messages.

- Source: the prepared PyTorch corpus at
  `31-LLM (PyTorch)/data/processed/chat-sft-v1/`.
- Available data: 196,882 UltraChat training conversations and 30,404 OASST1
  training conversations, with held-out validation splits.
- Required preparation: reserialize the conversations with the NumPy
  byte-level BPE tokenizer; the PyTorch tokenizer and token IDs are not
  compatible.
- Required implementation: target-architecture assistant-only loss masking,
  checkpoint resume, validation loss, and a chat behavior evaluation gate.
- First bounded run: 5,000–10,000 conversations, context 128, then evaluate
  before expanding to the full cleaned set.

Expected result: a meaningful improvement in instruction following and basic
conversation. It will not automatically produce strong coding or reasoning
quality.

## Option B — another general-language continuation

Improve continuation quality before SFT by extracting the next non-overlapping
FineWeb-Edu slice.

- Build the next 30M-character slice after the two existing FineWeb slices.
- Resume from `target_final.npz`.
- Target: another 10M tokens, with the same tokenizer and architecture.
- Approximate training time: 1.5–2 hours on the previous measured throughput,
  excluding checkpoint writes.

Expected result: lower language-model validation loss, but it will not fix the
gibberish response behavior by itself.

## Option C — small diagnostic SFT pilot

Use a tiny, hand-reviewed set to verify the SFT pipeline before spending GPU
time on thousands of conversations.

- 100–500 selected conversations.
- Assistant-only loss mask.
- 100–500 optimizer steps.
- Required gate: loss decreases, checkpoint reload works, and fixed prompts
  produce readable assistant text.

Expected result: pipeline validation only. It is not a production model.

## Recommendation

Choose Option C first if the target SFT trainer has not yet been implemented;
otherwise choose Option A with a 5,000-conversation bounded run. Do not use the
existing five-example `examples/train_sft.py` as the next model run: it is a
small pilot and is not wired for the 253M-parameter target checkpoint.

No training was started by preparing this document.
