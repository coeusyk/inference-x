# Proposal: add-streaming-tool-calling

## Why

Continue CLI (`@continuedev/cli` 1.5.47) is the second v1.0 acceptance client after Aider. It cannot edit anything through InferenceX today. Every request it sends carries an OpenAI `tools` array, and V1-0 rejects that field with HTTP 400 because the request schema uses `extra="forbid"` (DEC-063). A proxy that stripped `tools` let chat-only traffic through, which confirmed the missing API surface is the only blocker.

Tools were deferred at v1.0 planning on the rule "add them only when a real acceptance client proves they are required." Continue is that evidence. The traffic was captured again on 2026-09-27 against a fake OpenAI server so the exact wire shapes are known rather than assumed:

- Every request is `{model, messages, stream: true, stream_options: {include_usage: true}, tools}`. `tool_choice` and `parallel_tool_calls` are never sent. Non-streaming tool calls are never used.
- Each tool is `{type: "function", function: {name, description, parameters}}`. There is no `strict` key. Continue 1.5.47 sent 12 tools (6,725 bytes of JSON).
- An assistant tool-call message is `{role: "assistant", content: "", tool_calls: [{id, type: "function", function: {name, arguments}}]}`. `content` is an empty string when the model emitted no text, never null. `arguments` is a JSON string.
- A tool result is `{role: "tool", content, tool_call_id}`.

The protocol was also run against vLLM 0.22.1's own OpenAI server with its Hermes tool parser. Three models were tried. Qwen2.5-Coder-1.5B produced no parser-compatible calls (Markdown JSON instead of `<tool_call>`), 0/6 edits. Qwen2.5-7B-AWQ produced valid calls but made 1/6 correct edits. Qwen3-4B-Instruct-2507-FP8 produced valid calls and 3/3 correct edits, with the largest request around 4,589 tokens, which fits 8192 x 1 under DEC-064.

## What changes

- **Request.** `tools` is accepted as a list of OpenAI function tools, typed as exactly the shape Continue sends. `tools` requires `stream: true`; a non-streaming request with `tools` is rejected with 400.
- **Messages.** `role: "tool"` with a required `tool_call_id`, and assistant messages with `tool_calls`, are accepted and strongly typed. Every other combination stays rejected.
- **Capability is explicit.** A model entry declares `tool_call_parser: hermes` in `config/models.yaml`. A request that uses tools (a `tools` array or tool-call messages) against a model without a declared parser is rejected before admission with HTTP 400 and `code: tool_calling_unsupported`. The model is never run with the tools dropped.
- **Rendering and parsing stay inside `VLLMEngine`.** Tools and tool-call history go through the tokenizer's own chat template. vLLM's `Hermes2ProToolParser` extracts calls. Nothing Hermes-specific leaves the adapter (DEC-047).
- **Transported, never repaired or invented.** Tool calls are emitted only after the model finishes. A call the parser can read is emitted as parsed. A call whose framing is unambiguous but whose arguments are not valid JSON is emitted with the model's raw argument text, so the client can report the error and the model can retry (owner decision after the first live runs, design D6). Output that is not unambiguously a call is returned as ordinary content with the normal finish reason.
- **Streaming.** A tool call is streamed as OpenAI `delta.tool_calls` events (identity and name first, then the arguments), followed by `finish_reason: "tool_calls"`, then the usage event when requested. Text the model emits before the call streams as ordinary content.
- **Provenance.** Tool definitions and tool-call history are part of run identity: `request.tools_sha256` (canonical hash of the tools array) and tool-call fields inside `prompt_sha256`.
- **Acceptance model.** New `qwen3-4b-fp8` entry (`Qwen/Qwen3-4B-Instruct-2507-FP8`, 8192 context, 1 sequence, `tool_call_parser: hermes`) with the alias `local-4b`.
- **Model aliases.** `ModelEntry.aliases` routes another client-facing name to an entry without changing what `resolved` and the manifest report. Continue picks its edit tool from the model name, and the alias lets it offer the flat `Edit` tool without renaming the registry entry (design D10).
- **DEC-065** records the contract.

## Out of scope

`tool_choice`, `parallel_tool_calls`, non-streaming tool calls, the `strict` function flag, any parser other than Hermes, heuristic or regex recovery of malformed calls, server-side tool execution, MCP, the Responses API, and validating the contents of `parameters` beyond "is a JSON object". `tool_choice` and the other unsent fields remain rejected by `extra="forbid"`.

## Impact

- `schemas/chat.py`, `schemas/model.py`, `engines/base.py`, `engines/vllm_engine.py`, `services/chat_service.py`, `services/model_service.py`, `routing/policies.py`, `utils/ids.py`, `api/errors.py`, `api/main.py`, `api/routes/chat_completions.py`
- `scripts/continue_acceptance.py`, `scripts/log_proxy.py`
- `config/models.yaml`, `docs/DECISIONS.md`, `CHANGELOG.md`, `README.md`
- Non-tool requests are unchanged on the wire and in `prompt_sha256`. `run_id` gains the `tools_sha256` key (null when absent) in its preimage; see design D8.
