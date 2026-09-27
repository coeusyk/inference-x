# Tasks: add-streaming-tool-calling

## 1. Schema
- [x] 1.1 `ToolDefinition`, `FunctionDefinition`, `ToolCall`, `FunctionCall` in `schemas/chat.py` (D2, D3)
- [x] 1.2 `ChatMessage` gains `role: "tool"`, `tool_calls`, `tool_call_id` with a model validator; request-side arguments must be a JSON object (D3)
- [x] 1.3 `ChatCompletionRequest.tools` with the streaming-only validator; `tools` excluded from `resolved` (D2, D9)
- [x] 1.4 `ChatStreamChunk.tool_calls` and `finish_reason: "tool_calls"` (D7)
- [x] 1.5 `ManifestRequestInfo.tools_sha256`; tool-call fields in `compute_prompt_sha256`; `compute_tools_sha256` (D8)
- [x] 1.6 `ModelEntry.tool_call_parser` and `ModelEntry.aliases` (D4, D10)

## 2. Boundary, routing and service
- [x] 2.1 `BaseEngine.supports_tools` non-abstract capability (D4)
- [x] 2.2 `ChatService` rejects tool requests on unsupported engines before admission; `ToolCallingUnsupportedError` mapped to 400 `tool_calling_unsupported` (D4)
- [x] 2.3 Streaming route advances to the pre-generation event before the response starts, so pre-generation rejections are HTTP errors (D4)
- [x] 2.4 SSE formatting of terminal tool calls: identity event, arguments event, terminal `tool_calls` (D7)
- [x] 2.5 Registry alias index with collision check; `ExplicitModelPolicy` resolves aliases; `resolved.model` reports the canonical name (D10)

## 3. vLLM adapter
- [x] 3.1 Resolve the parser class at startup from `tool_call_parser` (D4)
- [x] 3.2 Render tools and tool-call history through the chat template, no plain-text fallback for tool requests (D5)
- [x] 3.3 `skip_special_tokens=False` when tools are present (D5)
- [x] 3.4 Stream content up to `<tool_call>`, buffer the rest with heartbeats, classify the complete output: parsed call, recognized call with raw arguments, or verbatim text (D6)

## 4. Config and docs
- [x] 4.1 `qwen3-4b-fp8` entry in `config/models.yaml` with `gpu_memory_utilization: 0.85` and alias `local-4b` (D10)
- [x] 4.2 DEC-065 in `docs/DECISIONS.md`
- [x] 4.3 CHANGELOG `[Unreleased]`, README Continue quickstart

## 5. Validation
- [x] 5.1 Unit tests (`tests/unit/test_tool_calling.py`): schema accept/reject cases, capability rejection (service and HTTP envelope), SSE event order with tool calls and usage, real Hermes parser on valid, raw-invalid and unrecognizable output, raw arguments unchanged on the wire, prompt/tools hashing, aliases, non-tool streams unchanged
- [x] 5.2 `ruff`, `mypy`, full unit suite, `openspec validate --strict`
- [x] 5.3 Continue CLI 1.5.47 acceptance against InferenceX serving `qwen3-4b-fp8` on an RTX 4060 8 GiB, traffic captured with `scripts/log_proxy.py`:
  - Addressed as `local-4b` (Continue offers `Edit`): **9/10** correct edits. 52 requests, all HTTP 200, each exactly `{model, messages, stream, stream_options, tools}`; 42 tool calls, every one with `finish_reason: "tool_calls"`; usage event on every stream; `resolved.model` always `qwen3-4b-fp8`.
  - Addressed as `qwen3-4b-fp8` (Continue offers `MultiEdit`): 2/5, the same as vLLM's reference server (2/5). Malformed MultiEdit calls reached Continue with raw arguments, Continue echoed them back as `"{}"` with the parse error, and one trial recovered; two looped until the timeout, as on the reference. 250 requests, all HTTP 200.
- [x] 5.4 VS Code smoke, first attempt: the extension sends `parallel_tool_calls: false` (and `max_tokens: 4096`), which was rejected with 400. Owner decision D11.
- [x] 5.5 `parallel_tool_calls` accepted and echoed in `resolved`; truncation to the first call under `false` with a `parallel_tool_calls_truncated` warning on the terminal event; raw argument span used whenever the envelope is intact (D6, D11)
- [x] 5.6 VS Code smoke, second attempt (Continue VS Code extension, Agent mode, `local-4b`, config on the Windows side reaching the WSL server): **pass**. `add()` fixed in a scratch file, test file unchanged, test passes. 5 requests, all HTTP 200: 4 agent turns `{model, max_tokens: 4096, stream, parallel_tool_calls: false, stream_options, tools (12)}` and one no-tools title request. Loop `file_glob_search` -> `read_file` -> `edit_existing_file`, one call per turn (no truncation warning), each echoed back byte-identical; `resolved.model` `qwen3-4b-fp8`.
  - Limitation found, not a protocol issue: the extension's agent prompt is about 7,800 tokens, so at 8,192 context admission clamped `max_tokens` to 395/322/267/100 with `max_tokens_clamped_to_context` warnings, and the final summary turn ended with `finish_reason: "length"`. Longer VS Code sessions will reach `context_length_exceeded` quickly at this context size.
