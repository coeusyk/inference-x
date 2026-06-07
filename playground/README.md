# InferenceX Playground

Python playground tools for interacting with and comparing InferenceX models.
The batch CLI uses [rich](https://github.com/Textualize/rich) for polished
terminal output. The interactive TUI uses
[Textual](https://github.com/Textualize/textual) and streams tokens live over
SSE.

---

## Prerequisites

1. The InferenceX server must be running:
   ```bash
   INFERENCE_X_DEFAULT_MODEL=qwen2.5-0.5b ./scripts/dev.sh serve
   ```
2. Python 3.8+ and the project venv (`uv sync` installs everything).

---

## Quick start

```bash
# From the repo root

# Interactive Textual playground
uv run python playground/app.py

# Interactive playground with a specific model
uv run python playground/app.py --model llama3-8b

# Interactive side-by-side compare mode
uv run python playground/app.py --compare qwen2.5-0.5b tinyllama-chat

# Single prompt to the default model
uv run python playground/client.py "What is the capital of France?"

# Target a specific model
uv run python playground/client.py --model tinyllama-chat "Explain transformers in one sentence."

# Compare two models side-by-side on a single server (both loaded)
uv run python playground/client.py --compare qwen2.5-0.5b tinyllama-chat \
  "Write a haiku about a GPU running out of memory."

# Compare across two server instances
# Terminal 1:
INFERENCE_X_DEFAULT_MODEL=qwen2.5-0.5b ./scripts/dev.sh serve
# Terminal 2:
INFERENCE_X_DEFAULT_MODEL=tinyllama-chat uv run uvicorn inference_x.api.main:app --host 0.0.0.0 --port 8001
# Terminal 3:
uv run python playground/client.py --compare qwen2.5-0.5b tinyllama-chat \
  --base-url-a http://localhost:8000 --base-url-b http://localhost:8001 \
  "Write a haiku about a GPU running out of memory."

# Check server health
uv run python playground/client.py --health

# List registered models
uv run python playground/client.py --list-models
```

---

## Interactive Textual app

Launch:

```bash
uv run python playground/app.py
uv run python playground/app.py --model llama3-8b
uv run python playground/app.py --compare qwen2.5-0.5b tinyllama-chat
```

The top bar shows the server URL and health status. The response panel streams
tokens as SSE chunks arrive from `POST /v1/chat/completions` with
`stream=true`. In compare mode, both models stream into side-by-side panels.

Shortcuts:

- `Ctrl+Enter` submits the prompt.
- `Enter` inserts a newline.
- `Ctrl+M` or `Tab` cycles models in single-model mode.
- `Ctrl+L` clears response panels.
- `F1` toggles the help overlay.
- `q` or `Ctrl+C` quits.

---

## Visual output examples

### `--health`

```
● healthy  http://localhost:8000
```

Coloured green when healthy, red when unreachable. No extra noise.

---

### `--list-models`

```
         Available Models
 ───────────────────────────────
  Name
  qwen2.5-0.5b
  tinyllama-chat
```

Alternating dim/normal rows.

---

### Single-model response

```
╭─ qwen2.5-0.5b  628ms ──────────────────────────────────────────╮
│ Silicon dreams fade,                                            │
│ Out of memory errors,                                           │
│ Restart and retry.                                              │
╰─────────────────────────────────────────────────────────────────╯
tokens: 36 prompt + 18 completion = 54 total
```

- Title: **bold** model name + dim latency
- Border: dim blue
- Content: bright white
- Footer: dim token usage

---

### Compare mode

```
─────────────────────────── haiku-gpu ───────────────────────────
╭─ qwen2.5-0.5b  95ms ───────────────╮╭─ tinyllama-chat  324ms ──────────────╮
│ GPU struggles to run,               ││ Without light, CPU dances on GPU's   │
│ Memory leaks, it crashes,           ││ edge                                 │
│ Powering down.                      ││ Strokes of memory, never enough      │
╰─────────────────────────────────────╯╰──────────────────────────────────────╯

 Model             Prompt tokens  Completion tokens  Total tokens  Latency
 qwen2.5-0.5b                36                 18            54    95ms
 tinyllama-chat               48                 40            88   324ms
```

- Prompt label displayed as a `Rule` above each pair
- Left panel: dim blue border; right panel: dim cyan border
- Stats table below with right-aligned numbers

---

### Error output

```
╭─ Error ─────────────────────────────────────────────────────────╮
│ Cannot compare 'qwen2.5-0.5b' and 'tinyllama-chat' on a        │
│ single server.                                                   │
│   Server at http://localhost:8000 has 'qwen2.5-0.5b' loaded … │
╰─────────────────────────────────────────────────────────────────╯
```

Red bordered panel to stderr — never a bare `print()`.

---

## Flags

| Flag | Default | Description |
|---|---|---|
| `prompt` (positional) | — | Inline prompt text |
| `--model NAME` | `qwen2.5-0.5b` | Model to query |
| `--compare A B` | — | Compare two models side-by-side |
| `--prompts-file PATH` | — | JSON file of prompts (overrides inline) |
| `--base-url URL` | `http://localhost:8000` | Server URL (single-model mode) |
| `--base-url-a URL` | `--base-url` | Server URL for MODEL_A in compare mode |
| `--base-url-b URL` | `--base-url` | Server URL for MODEL_B in compare mode |
| `--sequential` | — | Run model A, wait for B restart, then compare (single GPU) |
| `--temperature FLOAT` | `0.7` | Sampling temperature (0–2) |
| `--max-tokens N` | `512` | Max tokens to generate |
| `--list-models` | — | Print available models and exit |
| `--health` | — | Check server health and exit |

---

## Compare mode (single GPU — recommended)

On a single GPU with `INFERENCE_X_LOADED_MODELS=qwen2.5-0.5b,tinyllama-chat` both models
load into the same server process and compare works directly:

```bash
INFERENCE_X_LOADED_MODELS=qwen2.5-0.5b,tinyllama-chat ./scripts/dev.sh serve

# In another terminal:
uv run python playground/client.py --compare qwen2.5-0.5b tinyllama-chat \
  --prompts-file playground/prompts/sample_prompts.json
```

For sequential compare on a single-model server (restart between models):

```bash
# Terminal 1 — start with model A
INFERENCE_X_DEFAULT_MODEL=qwen2.5-0.5b ./scripts/dev.sh serve

# Terminal 2 — run sequential compare
uv run python playground/client.py --compare qwen2.5-0.5b tinyllama-chat --sequential \
  --prompts-file playground/prompts/sample_prompts.json
```

The client will:
1. Run all prompts against `qwen2.5-0.5b`
2. Show a spinner while waiting for tinyllama-chat to become ready
3. Print a green ✓ checkmark when the model is loaded
4. Run all prompts against `tinyllama-chat` and render side-by-side results

---

## Compare mode (two GPUs / dual-server)

```bash
# Terminal 1 — model A
INFERENCE_X_DEFAULT_MODEL=qwen2.5-0.5b ./scripts/dev.sh serve

# Terminal 2 — model B
INFERENCE_X_DEFAULT_MODEL=tinyllama-chat \
  uv run uvicorn inference_x.api.main:app --host 0.0.0.0 --port 8001

# Terminal 3 — compare
uv run python playground/client.py --compare qwen2.5-0.5b tinyllama-chat \
  --base-url-a http://localhost:8000 --base-url-b http://localhost:8001 \
  "Your prompt here"
```

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
uv run python playground/client.py --base-url http://my-gpu-server:8000 \
  --model llama3-8b "Summarise the Transformer paper."
```

---

## Demo tips for the article

1. Start with `--health` to confirm the server is live — single green line, no noise.
2. Run `--list-models` to show the model registry in a clean table.
3. Use `--compare` with two models of different sizes — latency differences in the stats table make for compelling screenshots.
4. Use `--max-tokens 100` for quicker responses during live demos.
5. The per-prompt `Rule` separator clearly labels each comparison section in screenshots.
