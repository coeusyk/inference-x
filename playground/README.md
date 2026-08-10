# InferenceX Playground

Python playground tools for interacting with, comparing, and benchmarking
InferenceX models. The batch CLI uses [rich](https://github.com/Textualize/rich)
for polished terminal output. Two Textual TUIs are available:

- **`playground/chat.py`** — Claude-style daily-driver chat (multi-turn history, `make chat`)
- **`playground/app.py`** — compare-only tool (side-by-side compare, `make playground`)

Both stream tokens live over SSE.

Run `make help` from the repo root for all available commands.

---

## Which tool to use?

| Goal | Command |
|------|---------|
| Chat with a model | `make chat` |
| Compare two models | `make playground` |
| Run benchmarks | `make benchmark` |
| Batch prompts / scripted runs | `python playground/client.py` |

---

## Prerequisites

1. The InferenceX server must be running:
   ```bash
   INFERENCE_X_DEFAULT_MODEL=qwen2.5-0.5b ./scripts/dev.sh serve
   ```
2. Python 3.13+ and the project venv (`uv sync` installs everything).
3. For localhost URLs, pass `--allow-internal` (included automatically in `make playground`).

---

## Quick start

```bash
# Daily-driver chat CLI (recommended)
make chat

# Compare two models side-by-side
make playground

# Or manually:
uv run python playground/chat.py --allow-internal
uv run python playground/app.py --allow-internal

# Pre-select two models (skips the compare picker)
uv run python playground/app.py --allow-internal --compare qwen2.5-0.5b tinyllama-chat
make playground-compare MODEL_A=qwen2.5-0.5b MODEL_B=tinyllama-chat

# Single prompt to the default model
uv run python playground/client.py --allow-internal "What is the capital of France?"

# Target a specific model
uv run python playground/client.py --model tinyllama-chat "Explain transformers in one sentence."

# Compare two models side-by-side (each on its own server process)
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

## Chat CLI (`playground/chat.py`)

Daily-driver Claude-style interface with scrollable multi-turn history and a bottom input bar.

```bash
make chat
# or
uv run python playground/chat.py --allow-internal
uv run python playground/chat.py --allow-internal --model qwen2.5-0.5b
```

### Shortcuts

- `Enter` — send message
- `Ctrl+N` — new conversation (clears history)
- `Ctrl+C` — quit

Server logs from `make chat` are written to `logs/playground-server.log` so they do not overlap the TUI.

---

## Interactive compare app (`playground/app.py`)

Compare-only TUI: two response panels, single-line prompt `Input`, and a one-line status
bar. For single-model chat, use `make chat` instead. For benchmarks, use
`make benchmark` / `make advise` from the CLI.

Launch:

```bash
make playground
# or
uv run python playground/app.py --allow-internal
```

### Startup flow

1. **Model selection screen** — pick two models from `config/models.yaml` (or the
   server registry when it is already running). Continue starts one server process
   per model.
2. **Loading screen** — phase title, step indicators (Server → Model → Ready), and a
   live tail of both processes' logs (`logs/playground-server.log` for the first
   model, `logs/playground-server-1.log` for the second) while they start, each line
   prefixed with its model name. On failure, an error banner shows a parsed summary
   from the failing process's log (VRAM OOM, vLLM errors, etc.).
3. **Compare UI** — stream the same prompt to both models side-by-side with Markdown
   rendering; per-panel titles show elapsed time and state; usage footers show token
   counts on completion.

The header shows server URL and health (`● healthy` / `● offline`).

### Compare shortcuts

- `Enter` submits the prompt to both models.
- `Ctrl+L` clears response panels.
- `F1` toggles the help overlay.
- `q` or `Ctrl+C` quits.

### Benchmarks (CLI only)

Benchmarks are not in the compare TUI. Run from a second terminal while the server
is up (or after `make playground` has started it):

```bash
make benchmark MODEL=qwen2.5-0.5b
make advise    # prints WARNING: lines for skipped or legacy results
```

Each benchmark JSON includes a `hardware` snapshot. Re-run benchmarks after changing GPU
or moving result files between machines so `make advise` ranks against matching hardware.

---

## CLI flags (app.py)

| Flag | Default | Description |
|---|---|---|
| `--base-url URL` | `http://localhost:8000` | Server base URL |
| `--compare A B` | — | Pre-select two models; skips compare picker |
| `--allow-internal` | off | Allow loopback/private URLs (required for localhost) |

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
| `--allow-internal` | off | Allow loopback/private URLs (required for localhost) |

---

## Compare mode (dual-server — recommended)

Each server process serves exactly one model, so comparing two models means running
two processes — one per model, each on its own port. `make playground` /
`make playground-compare` do this automatically (see "Interactive compare app" below).
To drive it manually with the batch client:

```bash
# Terminal 1 — model A
INFERENCE_X_DEFAULT_MODEL=qwen2.5-0.5b ./scripts/dev.sh serve

# Terminal 2 — model B
INFERENCE_X_DEFAULT_MODEL=tinyllama-chat \
  uv run uvicorn inference_x.api.main:app --host 127.0.0.1 --port 8001

# Terminal 3 — compare
uv run python playground/client.py --compare qwen2.5-0.5b tinyllama-chat \
  --base-url-a http://localhost:8000 --base-url-b http://localhost:8001 \
  --prompts-file playground/prompts/sample_prompts.json
```

This works on one GPU as well as two — two independent single-model processes fit
comfortably on a single card for real model pairs (verified on an 8 GiB card,
docs/DECISIONS.md DEC-059).

For sequential compare on a single server instead (restart between models — slower,
but only ever runs one model process at a time):

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
  --model qwen2.5-1.5b "Summarise the Transformer paper."
```

---

## Demo tips for the article

1. Start with `--health` to confirm the server is live — single green line, no noise.
2. Run `--list-models` to show the model registry in a clean table.
3. Use `--compare` with two models of different sizes — latency differences in the stats table make for compelling screenshots.
4. Use `--max-tokens 100` for quicker responses during live demos.
5. The per-prompt `Rule` separator clearly labels each comparison section in screenshots.
