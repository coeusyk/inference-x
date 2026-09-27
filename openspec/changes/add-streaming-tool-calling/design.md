# Design: add-streaming-tool-calling

## Context

V1-0 made Aider work. Continue is the next acceptance client and it only edits through OpenAI tool calling. This change adds the narrowest tool-calling surface that Continue CLI 1.5.47 actually uses, and keeps three questions separate:

1. Does the API accept tool requests? (the schema)
2. Does the backend have a parser for the model's tool format? (the vLLM adapter)
3. Can the selected model produce that format? (the model entry's declared capability)

Answering yes to (1) says nothing about (2) or (3). A model that cannot produce tool calls must be refused, not run with the tools quietly ignored.

## D1: Evidence and scope

The acceptance client is `@continuedev/cli` 1.5.47. Its traffic was captured on 2026-09-27 by pointing it at a fake OpenAI server (a 40-line stdlib HTTP server) that answered the first turn with a canned streamed tool call and the second with text. That gave both the first request and a follow-up containing the assistant tool call and the tool result, without needing a GPU. Findings are in the proposal. The scope of this change is exactly what that capture showed; anything Continue did not send is out of scope and stays rejected.

## D2: Tool request schema

```
ToolDefinition     {type: "function", function: FunctionDefinition}
FunctionDefinition {name: str, description?: str, parameters?: object}
```

Both models use `extra="forbid"`, so `strict`, or a non-function tool type, is a 400 with a named `param`. `parameters` is typed as a JSON object and nothing more. InferenceX does not validate JSON Schema contents, because it never executes tools and the chat template only serializes them. `tools` is limited to 1..128 entries as a body-size guard, the same role the message-count bound plays.

`tools` requires `stream: true` (D9). The check is a field validator on `tools`, so the error `param` is `tools`. That mirrors how `include_manifest` rejects `stream: true`.

`tools` does not appear in `resolved`. It is content, like `messages`, and echoing ~7 KB of tool schema back on every stream would bloat the pre-generation event for no gain. The `ResolvedRequest` derivability test gets `tools` added to its explicit exclusion set, so the decision is enforced rather than implied.

## D3: Message representation

`ChatMessage` keeps one class and gains `role: "tool"` plus two optional fields:

- `tool_calls: list[ToolCall] | None`: allowed only on `assistant`, and must be non-empty when present.
- `tool_call_id: str | None`: required on `tool`, forbidden elsewhere.

A model validator enforces both rules. `content` stays a required `str`, because Continue sends `""` rather than null for an assistant message that only calls tools. I looked at a discriminated union of four message classes. It types more precisely, but every consumer reads only `role` and `content`, and the union would touch every call site and test fixture for two optional fields. One class with a validator is the smaller correct change.

`ToolCall` is `{id, type: "function", function: {name, arguments}}`, shared by request messages and the response stream. `FunctionCall.arguments` is a plain string so a response can carry the model's raw text (D6). On a request message, `ChatMessage`'s validator additionally requires each `arguments` to parse as a JSON object: the engine has to turn it into a dict for the chat template (D5), and a bad string there would otherwise surface as a 500 from inside the engine instead of a 400. This does not block Continue's retry loop. When Continue cannot parse a call's arguments it echoes the call back with `arguments: "{}"` and puts the parse error in the tool result (captured against a fake server on 2026-09-27).

## D4: Capability declaration

`ModelEntry` gains `tool_call_parser: Literal["hermes"] | None = None`. The name mirrors vLLM's `--tool-call-parser` flag, and the value is a vLLM parser registry name. That is backend-specific, but so are `quantization`, `max_num_seqs`, and `max_num_batched_tokens` in the same entry. It is operator config, not public API. The public API never names a parser.

The Engine Boundary gets one non-abstract capability, `BaseEngine.supports_tools -> bool` (default `False`), following the pattern DEC-047 section 3 set for `count_prompt_tokens`. `VLLMEngine` returns true only when a parser was resolved at startup and the tokenizer has a chat template. `ChatService` checks it after routing and before admission. A request "uses tools" if it has a `tools` array or any message with `tool_calls` or `role: "tool"`. If the engine does not support tools, the service raises `ToolCallingUnsupportedError`, which maps to HTTP 400, `type: invalid_request_error`, `param: tools`, `code: tool_calling_unsupported`. Nothing reaches admission or the engine.

Continue only streams, and until now every pre-generation rejection on a streaming request (context overflow included) was raised inside the SSE generator after the 200 had started, so the client saw a broken stream instead of a 400. The route now advances the generator to its pre-generation event before returning the `StreamingResponse`, which makes these rejections ordinary HTTP errors. The generator's single `finally` still releases the admission reservation exactly once.

The parser class is resolved once in `VLLMEngine.__init__` through `ToolParserManager.get_tool_parser(name)`. An unknown name fails startup instead of failing the first request. V1-1 (llama.cpp) will implement `supports_tools` its own way; nothing here says OpenAI tools means Hermes.

## D5: Rendering and parser ownership

All of this lives in `VLLMEngine`:

- **Rendering.** Messages become dicts with `tool_calls` (arguments decoded to a dict) and `tool_call_id` when present, and the tools list is passed as `tools=` to `tokenizer.apply_chat_template`. This is the same thing vLLM's server does (`chat_utils._postprocess_messages` decodes arguments for the same reason: HF templates expect a dict). Checked offline against the Qwen3-4B-2507 template: tools render into the system block, the assistant call renders as `<tool_call>{"name":..., "arguments":{...}}</tool_call>`, and the result renders as `<tool_response>`. When a request uses tools, a template failure raises. The existing plain-text fallback is not used, because it would drop the tools silently.
- **Sampling.** With tools present, `skip_special_tokens=False`, copied from `Hermes2ProToolParser.adjust_request`. For Qwen3 the `<tool_call>` tokens are not special, so this does not change Qwen output. It keeps the parser correct for Hermes models where they are special.
- **Parsing.** One `Hermes2ProToolParser(tokenizer)` per request, and only its non-streaming `extract_tool_calls(text, request=None)` is called. The method does not read the request. No vLLM request object and no parser object ever leaves the adapter; the adapter maps the parser's `ToolCall`s to InferenceX's own `ToolCall` schema.

`vllm.tool_parsers` is not a documented stable API. The coupling is kept small on purpose: one registry lookup, one constructor, one method, the parser's `tool_call_start_token`, and two result attributes (`tools_called`, `tool_calls`). A unit test runs the real installed parser on a valid and several malformed outputs, so an upstream change breaks a test instead of production.

## D6: Recognized calls are transported, never repaired or invented

This rule was revised by the owner after the first live Continue runs (see "Acceptance history" below). The invariant is:

> InferenceX may faithfully transport malformed model-generated tool arguments when the tool-call structure itself is unambiguous. It must never repair or invent tool-call structure or arguments.

vLLM's reference streaming path (`extract_tool_calls_streaming`) starts emitting a call as soon as a regex finds `"name": "..."` and streams argument text as it is generated. It sets `finish_reason: "tool_calls"` whenever it has seen a name, before it can know whether the block will ever be complete. InferenceX does not use the streaming parser. While generating, the adapter streams text up to the first `<tool_call>` (holding back any suffix that could be the start of the tag) and buffers the rest. When the model finishes, the complete output is classified in two steps:

1. **Parsed call.** `Hermes2ProToolParser.extract_tool_calls` runs first. It `json.loads` every block (closed, or an unclosed final block whose JSON is complete) and requires `name` and `arguments`. If it returns calls, they are emitted as it produced them.
2. **Recognized call with raw arguments.** If step 1 finds nothing, the output is still a call when its framing is unambiguous: every `<tool_call>` is closed by `</tool_call>`, and every block's body fully matches `{"name": "<string>", "arguments": <text>}`, meaning the name slot comes first, the `arguments` key follows, and the block ends with the envelope's closing brace. Each call's arguments are then the model's text between `"arguments":` and that final brace, byte for byte (surrounding whitespace aside), even when it is not valid JSON. The regex only locates that span. It never rewrites it.

The same raw span is used in step 1 whenever a block's envelope is intact, instead of the Hermes parser's `json.dumps` re-serialization (which re-spaces and can re-escape). A call is therefore carried identically whether its neighbours parsed or not. Only a block Hermes parses but the envelope cannot frame (keys in another order), or whose span does not decode to exactly the arguments Hermes parsed (an extra key after `arguments` would otherwise be swept into the span), keeps Hermes's form.

Anything else, such as an unclosed block that is not valid JSON, a missing name or `arguments` key, one unrecognizable block among several, or JSON in a Markdown fence, is not a call. The held-back text is emitted verbatim as ordinary content with the engine's own finish reason (`stop` or `length`).

InferenceX never repairs quotes, never runs `ast.literal_eval`, never rebalances brackets, never infers a missing brace, never invents a name, and never rewrites arguments. When the last `}` could belong either to the arguments or to the envelope, it is attributed to the envelope; nothing is added.

Why transport instead of refusing: a malformed call forwarded as a call lets the client do what it does for any bad call. Continue sends the parse error back as the tool result and the model gets another try. Returned as text, the same output ends the session. Measured on the median task with Continue's MultiEdit tool, vLLM's reference (which forwards such calls) recovered in 2/5 trials, while the earlier strict InferenceX rule recovered in 0/6.

Checked against the installed parser: a well-formed call parses in step 1. Python-style quotes and an unbalanced nested array (both seen live) are transported raw in step 2. An unterminated envelope, a call without `arguments`, and Markdown-fenced JSON (the Qwen2.5-Coder-1.5B failure) produce no call.

Two consequences, both accepted:

- Tool arguments arrive at the end of generation instead of token by token. Continue executes a call only after `finish_reason` anyway, so the edit loop is not slower. Only the incremental display of arguments is lost.
- While a call is being buffered the engine yields empty heartbeat chunks, so the per-token stream timeout (`INFERENCE_X_STREAM_TIMEOUT_S`) still means "the engine stalled" and not "the tool call was long". The service emits no SSE event for an empty chunk.

Parity note: like vLLM, text after `</tool_call>` in a response that did produce calls is not returned. Qwen templates put nothing there in practice.

## D7: Streaming contract

`ChatStreamChunk` gains `tool_calls: list[ToolCall] | None`, and its `finish_reason` gains `"tool_calls"`. Tool calls only ever ride the terminal chunk, which is the only chunk that can know they are complete. This extends the existing boundary type instead of adding a second one (DEC-049).

`ChatService.stream_response` formats a terminal chunk with tool calls as, for each call `i`:

1. `{"delta": {"tool_calls": [{"index": i, "id", "type": "function", "function": {"name", "arguments": ""}}]}, "finish_reason": null}`
2. `{"delta": {"tool_calls": [{"index": i, "function": {"arguments": "<the call's arguments string, as produced>"}}]}, "finish_reason": null}`

followed by the usual terminal event (empty delta, `finish_reason: "tool_calls"`), the usage event when `include_usage` was requested, and `[DONE]`. Shape (1) then (2) is what Continue's captured stream used and what OpenAI sends. The existing event-order requirement holds: tool-call events sit where content events do, before the terminal event.

The HTTP-boundary TTFT is anchored to the first content delta (existing requirement). A response that is only a tool call has no content delta, so its TTFT is recorded as absent rather than the time the buffered call was flushed.

## D8: Provenance identity

Tools change what the model generates, so two runs with different tool sets are different configurations. The manifest `request` block gains `tools_sha256`: SHA-256 of the canonical JSON (sorted keys, compact separators) of the validated `tools` list, or null when there are no tools. The ~7 KB schema is not inlined; the hash is enough for comparability, and the client already holds the tools it sent. `prompt_sha256` now also covers `tool_calls` and `tool_call_id`, which are added to a message's hashed form only when present. Plain messages hash to the same `prompt_sha256` as before this change.

Two facts about the current system matter here:

- `run_id` is computed only on non-streaming responses today, and `tools` requires streaming. So no request that carries `tools` gets a `run_id` yet. A non-streaming request with tool-call history but no `tools` does get one, and its `prompt_sha256` now distinguishes it. The rule is defined now so the preimage is already correct if streaming manifests or non-streaming tool calls arrive later.
- `tools_sha256` extends identity only when the request actually uses tools. The `request` block enters the `run_id` preimage in exactly its pre-DEC-065 shape, and `tools_sha256` is added to it only when it is non-null (`chat_service._request_preimage`). The returned manifest still shows `tools_sha256: null` for a no-tools request, but that key is not hashed. So a no-tools request keeps its historical `run_id`, and a regression test pins the value computed by the pre-change code on `develop`. Only this key is conditional; the block's other nullable fields keep their nulls in the preimage, as before. The first version of this change hashed the null key into every preimage and so changed every no-tools `run_id`. The owner rejected that before merge.

## D9: Unsupported features stay rejected

- `tool_choice`, `functions`, `function_call`: rejected by `extra="forbid"` (unknown field), unchanged. `parallel_tool_calls` was in this list until the VS Code extension proved it required (D11).
- `tools` with `stream: false`: 400, `param: tools`. Mechanically the non-streaming path could return calls (it is derived from the streaming path), but no client has asked for it, and the non-streaming response schema would need `tool_calls` and nullable content. That is left for when a client needs it.
- A model without a declared parser: 400, `code: tool_calling_unsupported` (D4).
- Function `strict`, non-function tool types, `role: "function"`: rejected by the schema.

## D11: `parallel_tool_calls` (added after the VS Code smoke test)

Continue for VS Code (unlike the CLI) sends `parallel_tool_calls: false` on every request, alongside `max_tokens: 4096`, which was already supported. InferenceX rejected it as an unknown field, so the extension could not run at all. vLLM's reference accepts the field and implements `false` by silently keeping only the first call after generation (`openai/utils.py::maybe_filter_parallel_tool_calls`).

Owner decision: accept `parallel_tool_calls: bool | None`, echoed in `resolved`.

- Absent or `true`: unchanged, every call the model produced is returned.
- `false` with 0 or 1 calls: unchanged, no warning.
- `false` with more than one call: only the first call is returned, and the truncation is disclosed. Nothing is dropped silently, rewritten, merged, or replaced, and the session is not ended.

The disclosure reuses `ResponseWarning`: `type: "substituted"`, `code: "parallel_tool_calls_truncated"`, `field: "parallel_tool_calls"`, message "The model produced N tool calls while parallel_tool_calls=false; only the first call was returned." It can only be known after generation, so it cannot ride the pre-generation event. The narrowest additive extension is a top-level `warnings` list on the terminal event (the same key the pre-generation event uses), present only when a post-generation warning exists. Every stream has a terminal event, unlike the optional usage event, so that is where it goes. The produced and returned call counts are also logged server-side. `strict` cannot turn this into a rejection because it is only knowable after the response has started; strict's contract (DEC-052) covers substitutions decided before dispatch.

The truncation is a request-parameter policy, not parser behavior, so it lives in `ChatService`. The engine still returns every call it recognized.

## D10: Acceptance model

`qwen3-4b-fp8` maps to `Qwen/Qwen3-4B-Instruct-2507-FP8` (the repo id of the cached snapshot), `quantization: fp8`, `max_model_len: 8192`, `max_num_seqs: 1`, `tool_call_parser: hermes`. 8192 x 1 is inside the 6gb tier's 2048 x 4 envelope (DEC-064). It was chosen on measured results: 3/3 Continue edits against the vLLM reference server, where the 1.5B coder model made 0/6 and the 7B AWQ model 1/6.

**Alias.** Continue CLI 1.5.47 decides which edit tool to offer from the model name: any name matching `/gemini|claude|gpt|o\d|kimi|qwen|llama|nemotron|grok|mistral/` gets `MultiEdit` (a nested `edits` array), anything else gets the flat `Edit`. Qwen3-4B malformed the MultiEdit JSON on its first attempt in every run that reached an edit, through InferenceX and through the reference alike. Under a neutral name every `Edit` call it made was valid JSON.

The owner ruled out renaming the entry to steer a client heuristic, since that would make the registry less truthful. Instead `ModelEntry` gains `aliases: list[str]`, and the entry keeps its canonical name `qwen3-4b-fp8` with alias `local-4b`. `ModelRegistry` indexes aliases at load and rejects an alias that collides with a model name or another alias. `ExplicitModelPolicy` resolves an alias to its canonical entry. `ChatService` then sets the effective request's `model` to the canonical name, so `resolved.model` names the entry that served the request. The manifest's `model.registry_name` and `hf_repo` come from the routed entry as before, so provenance still identifies the Qwen3-4B FP8 checkpoint. Nothing in the registry knows about Continue; the alias is plain config with a comment that explains why it exists.

Unrelated pre-existing behavior noticed along the way and left alone: an unregistered `model` falls back to the default model without a warning, and `resolved.model` then echoes the unregistered name. That is a separate truthfulness gap for its own change.

**Memory.** `gpu_memory_utilization` is set to 0.85 explicitly. Footprint auto-sizing asked for 0.911 (7.28 GiB) on the 8 GiB RTX 4060 from free VRAM measured before the engine process's CUDA context existed, and vLLM then refused to start with 6.93 GiB free.

**Sampling defaults.** Continue sends no `temperature` or `top_p`. vLLM's server then uses the model's `generation_config.json` (Qwen3: temperature 0.7, top_p 0.8, top_k 20), while InferenceX applies its schema defaults (0.7, 0.95, no top_k). This affects output quality in both directions and is out of scope here. It is recorded as a follow-up question, not changed.

Protocol correctness is what InferenceX guarantees. Model quality at tool use is not. The 1.5B coder model stays the Aider model and gets no parser: it cannot emit the format, so declaring a parser for it would claim a capability it lacks.

## Exit test

Continue CLI 1.5.47 against InferenceX itself (not the vLLM reference server), serving `qwen3-4b-fp8`, on a mechanically verified edit task, several trials, with traffic captured through a logging proxy to confirm the request fields and the tool-call events. VS Code is a manual smoke test afterwards and does not block the change.

## Acceptance history

1. First runs through InferenceX with the model named `qwen3-4b-fp8` and the strict "parse or refuse" rule: 0/6. Every trial listed, read, and then emitted a malformed MultiEdit call that came back as text, which ended the session.
2. vLLM reference server, same name and task: 2/5. Prompt token counts matched InferenceX exactly on identical turns (2455 / 2544 / 2614 / 2720), which confirms rendering parity. The reference forwarded the malformed MultiEdit calls, and the model sometimes recovered after Continue's parse error. Other runs looped 20 to 40 times.
3. vLLM reference under a neutral name (so Continue offered Edit): 4/5, every call valid JSON.
4. InferenceX under a neutral name, strict rule: 3/5. One failure was a Python-quoted `new_string` refused as text, the other a wrong `old_string` the model then claimed had worked.
5. Owner decisions: keep the canonical name and add an alias; transport recognized calls with raw arguments (D6). Final results are in the tasks file.

## Risks

- **vLLM parser API drift.** Contained to one call site and covered by a test that runs the real parser (D5).
- **Buffered arguments.** Long edit calls do not render incrementally in the client (D6). Accepted for correctness.
- **Model quality.** A declared parser means the model *can* emit the format, not that it uses tools well. The 7B AWQ result shows the difference. Documentation says only the narrow claim.
