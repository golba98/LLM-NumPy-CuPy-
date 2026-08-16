# Base-model evaluation

`evaluation/prompts_general.json` contains deterministic qualitative probes for
continuation, factual language, science, code, and proto-conversation. These
are base-model probes, not evidence of chat quality. Keep seed, temperature,
top-k, top-p, and maximum generation length fixed when comparing checkpoints.

Primary quantitative metrics are held-out general-language cross entropy and
perplexity. Shakespeare is not the production validation set.
