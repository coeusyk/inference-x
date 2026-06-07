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

# Compare two models side-by-side (requires two server instances — one model per GPU)
# Terminal 1:
INFERENCE_X_DEFAULT_MODEL=qwen2.5-0.5b ./scripts/dev.sh serve
# Terminal 2:
INFERENCE_X_DEFAULT_MODEL=tinyllama-chat uv run uvicorn inference_x.api.main:app --host 0.0.0.0 --port 8001
# Terminal 3:
python3 playground/client.py --compare qwen2.5-0.5b tinyllama-chat \
  --base-url-a http://localhost:8000 --base-url-b http://localhost:8001 \
  "Write a haiku about a GPU running out of memory."

# Compare all sample prompts across two models
python3 playground/client.py --compare qwen2.5-0.5b tinyllama-chat \
  --base-url-a http://localhost:8000 --base-url-b http://localhost:8001 \
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
| `--compare A B` | — | Compare two models side-by-side (two servers if A≠B) |
| `--prompts-file PATH` | — | JSON file of prompts (overrides inline) |
| `--base-url URL` | `http://localhost:8000` | Server URL (single-model mode) |
| `--base-url-a URL` | `--base-url` | Server URL for MODEL_A in compare mode |
| `--base-url-b URL` | `--base-url` | Server URL for MODEL_B in compare mode |
| `--temperature FLOAT` | `0.7` | Sampling temperature (0–2) |
| `--max-tokens N` | `512` | Max tokens to generate |
| `--list-models` | — | Print available models and exit |
| `--health` | — | Check server health and exit |

---

## Compare mode (single GPU — recommended)

On a single GPU only one model loads at a time. Use `--sequential` to compare without running two servers:

```bash
# Terminal 1 — start with model A
INFERENCE_X_DEFAULT_MODEL=qwen2.5-0.5b ./scripts/dev.sh serve

# Terminal 2 — run sequential compare
python3 playground/client.py --compare qwen2.5-0.5b tinyllama-chat --sequential \
  --prompts-file playground/prompts/sample_prompts.json
```

The client will:
1. Run all prompts against `qwen2.5-0.5b`
2. Tell you to restart the server with `tinyllama-chat`
3. Wait until the new model is healthy
4. Run all prompts against `tinyllama-chat` and print side-by-side results

Without `--sequential`, compare on a single server fails fast with setup instructions.

---

## Compare mode (two GPUs / dual-server)

InferenceX loads **one model per server process** (single GPU). To compare two different models you need two server instances on different ports:

```bash
# Terminal 1 — model A
INFERENCE_X_DEFAULT_MODEL=qwen2.5-0.5b ./scripts/dev.sh serve

# Terminal 2 — model B
INFERENCE_X_DEFAULT_MODEL=tinyllama-chat \
  uv run uvicorn inference_x.api.main:app --host 0.0.0.0 --port 8001

# Terminal 3 — compare
python3 playground/client.py --compare qwen2.5-0.5b tinyllama-chat \
  --base-url-a http://localhost:8000 --base-url-b http://localhost:8001 \
  "Your prompt here"
```

If you run `--compare` with two different models on the same `--base-url`, the client fails fast with setup instructions instead of repeating HTTP 400 errors for every prompt.

---

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
