# InferenceX Playground

A zero-dependency Python CLI for interacting with and comparing InferenceX models.
No npm, no frontend build, no extra pip installs — runs with `python3` on any WSL2 setup.

---

## Prerequisites

1. The InferenceX server must be running:
   ```bash
   INFERENCE_X_DEFAULT_MODEL=qwen2.5-0.5b ./scripts/dev.sh serve
   ```
2. Python 3.8+ (already available in your WSL2 environment).

---

## Quick start

```bash
# From the repo root

# Single prompt to the default model
python3 playground/client.py "What is the capital of France?"

# Target a specific model
python3 playground/client.py --model tinyllama-chat "Explain transformers in one sentence."

# Compare two models side-by-side
python3 playground/client.py --compare qwen2.5-0.5b tinyllama-chat \
  "Write a haiku about a GPU running out of memory."

# Run all sample prompts against the default model
python3 playground/client.py --prompts-file playground/prompts/sample_prompts.json

# Compare all sample prompts across two models
python3 playground/client.py --compare qwen2.5-0.5b tinyllama-chat \
  --prompts-file playground/prompts/sample_prompts.json

# Check server health
python3 playground/client.py --health

# List registered models
python3 playground/client.py --list-models
```

---

## Flags

| Flag | Default | Description |
|---|---|---|
| `prompt` (positional) | — | Inline prompt text |
| `--model NAME` | `qwen2.5-0.5b` | Model to query |
| `--compare A B` | — | Compare two models side-by-side |
| `--prompts-file PATH` | — | JSON file of prompts (overrides inline) |
| `--base-url URL` | `http://localhost:8000` | Server URL |
| `--temperature FLOAT` | `0.7` | Sampling temperature (0–2) |
| `--max-tokens N` | `512` | Max tokens to generate |
| `--list-models` | — | Print available models and exit |
| `--health` | — | Check server health and exit |

---

## Sample prompts

`playground/prompts/sample_prompts.json` contains 8 prompts designed for comparison:

| Label | Use |
|---|---|
| `basic-factual` | Sanity check: correct factual answer expected |
| `explain-concept` | Tests instruction-following and explanation quality |
| `haiku-gpu` | Creative; compare personality between models |
| `code-snippet` | Tests code generation with memoisation |
| `summarise-text` | Tests precision under a system prompt |
| `compare-models` | Tests structured output (numbered bullets) |
| `creative-story` | Creative; compare style |
| `reasoning` | Tests arithmetic reasoning |

Each entry can include a `system` prompt and a `label` for display.

---

## Compare output format

```
Prompt: Write a haiku about a GPU running out of memory.
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 qwen2.5-0.5b (842ms)                                        │  tinyllama-chat (631ms)
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 Silicon dreams fade,                                        │  Memory fills up fast,
 Out of memory errors,                                       │  GPU struggles to keep pace,
 Restart and retry.                                          │  Error, reload now.
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
tokens: 11 prompt + 18 completion = 29 total                 │ tokens: 11 prompt + 17 completion = 28 total
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

---

## Prompts file format

```json
[
  {
    "label": "my-prompt",
    "text": "Your prompt here.",
    "system": "Optional system message."
  },
  "A bare string prompt also works."
]
```

---

## Running against a different server

```bash
python3 playground/client.py --base-url http://my-gpu-server:8000 \
  --model llama3-8b "Summarise the Transformer paper."
```

---

## Demo tips for the article

1. Start with `--health` to confirm the server is live.
2. Run `--list-models` to show the registry.
3. Pick two models with different sizes for `--compare` — the latency and style differences make for good screenshots.
4. Use `--max-tokens 100` for quicker responses during live demos.
5. The compare output is designed to be pasted directly into a terminal screenshot tool.
