# Production pretraining workflow

The base model is trained on general-language text first; conversational SFT
is a later phase. Shakespeare remains smoke/regression data only.

Before any long run, execute:

```powershell
python scripts/preflight_training.py --config configs/pretrain_250m.json
```

The checked-in configuration is intentionally a bounded 10M-token calibration
plan. It uses the known-good 240,897,792-parameter model, FP16 activations,
FP32 AdamW moments, and static loss scale 128. The current local token cache is
marked non-production until its source provenance is rebuilt; preflight must
therefore fail until a cleared manifest is supplied.

Each real run must have its own directory under `runs/pretraining/`, retain
`config.json`, environment and manifests, and use atomic checkpoint replacement.
Resume must use the same tokenizer, dataset manifest, and model configuration.
