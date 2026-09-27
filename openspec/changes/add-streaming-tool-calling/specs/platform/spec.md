## ADDED Requirements

### Requirement: Chat requests accept OpenAI function tools on streaming requests
The platform SHALL accept a `tools` array of OpenAI function tools, each `{type: "function", function: {name, description?, parameters?}}`, on streaming chat completion requests, and SHALL reject any other tool shape, any unknown key inside a tool, and `tools` on a non-streaming request with HTTP 400 naming the offending `param`. `tools` SHALL NOT appear in the `resolved` block.

#### Scenario: Continue-shaped tool request
- **WHEN** a request sets `stream: true`, `stream_options.include_usage: true`, and a `tools` array of function tools with `name`, `description`, and object `parameters`
- **THEN** the request passes validation

#### Scenario: Tools on a non-streaming request
- **WHEN** a request sets `tools` and `stream: false`
- **THEN** the response is HTTP 400 with `param` `tools`

#### Scenario: Unsupported tool options stay rejected
- **WHEN** a request sets `tool_choice`, or a function definition carries `strict`
- **THEN** the response is HTTP 400 and the field is not silently discarded

### Requirement: Tool-call conversation messages are strongly typed
The platform SHALL accept `role: "tool"` messages that carry a `tool_call_id`, and assistant messages that carry a non-empty `tool_calls` list of `{id, type: "function", function: {name, arguments}}` whose `arguments` is a JSON object encoded as a string (a client echoing back a call it could not parse sends `"{}"`). It SHALL reject `tool_calls` on any non-assistant role, a `tool` message without `tool_call_id`, `tool_call_id` on any other role, and `arguments` that is not a JSON-object string, with HTTP 400.

#### Scenario: Round trip of a tool call and its result
- **WHEN** a request contains an assistant message with `content: ""` and one tool call, followed by a `tool` message whose `tool_call_id` matches it
- **THEN** the request passes validation and both messages reach the model's chat template

#### Scenario: Tool message without an id
- **WHEN** a `tool` message omits `tool_call_id`
- **THEN** the response is HTTP 400

#### Scenario: Arguments that are not JSON
- **WHEN** an assistant tool call's `arguments` is not a JSON object string
- **THEN** the response is HTTP 400

### Requirement: Tool calling requires a declared model capability
The platform SHALL serve a request that uses tools (a `tools` array, or any message with `tool_calls` or `role: "tool"`) only when the routed model declares a supported tool-call parser. Otherwise it SHALL reject the request before admission and before generation with HTTP 400, `type: invalid_request_error`, `param: tools`, and `code: tool_calling_unsupported`. The platform MUST NOT run the model with the tools removed.

#### Scenario: Model without a declared parser
- **WHEN** a tool request is routed to a model entry with no `tool_call_parser`
- **THEN** the response is HTTP 400 with code `tool_calling_unsupported`
- **AND** on a streaming request this is the HTTP response itself, not an event inside a started stream
- **AND** no admission reservation is made and the engine is not called

#### Scenario: Model with a declared parser
- **WHEN** a tool request is routed to a model entry declaring `tool_call_parser: hermes`
- **THEN** the tools and the tool-call history are rendered by the model's chat template

#### Scenario: Unknown parser name
- **WHEN** a model entry declares a parser name the backend does not provide
- **THEN** the model fails to load at startup rather than at the first request

### Requirement: Tool calls are transported from model output, never repaired or invented
The platform SHALL emit a structured tool call only when the model's complete output is unambiguously a tool call: either the backend's parser extracted it with a JSON-valid body carrying a name and arguments, or every tool-call block is closed and its envelope carries the name slot first, then the `arguments` key, then the envelope's closing brace. In the second case the call's `arguments` SHALL be the model's own text for that value, byte for byte, even when it is not valid JSON. The platform MUST NOT repair, normalize, or re-quote arguments, infer missing brackets or braces, or invent a function name. When the output is not unambiguously a tool call, the platform SHALL return the model's text verbatim as content with the engine's own finish reason.

#### Scenario: Well-formed call
- **WHEN** the model emits `<tool_call>{"name": "Read", "arguments": {"filepath": "a.py"}}</tool_call>`
- **THEN** the stream carries one tool call named `Read` with arguments `{"filepath": "a.py"}` and `finish_reason` `tool_calls`

#### Scenario: Recognized call with invalid arguments
- **WHEN** the model emits a closed `<tool_call>` block `{"name": "Edit", "arguments": <text>}` where `<text>` is not valid JSON (for example Python-style quotes, or an unbalanced array)
- **THEN** the stream carries one tool call named `Edit` whose `arguments` is exactly `<text>`, and `finish_reason` is `tool_calls`

#### Scenario: Unrecognizable framing
- **WHEN** the model emits a block without a name or `arguments` key, an unclosed block that is not valid JSON, a block whose envelope is not closed, or JSON in a Markdown fence instead of a tool-call block
- **THEN** no tool-call event is emitted
- **AND** the text is returned as content and `finish_reason` is `stop` or `length` as the engine reported

### Requirement: Models may be addressed by declared aliases without changing provenance
A model entry MAY declare `aliases`. A request whose `model` is an alias SHALL be routed to that entry, and its `resolved.model` and run manifest SHALL report the entry's canonical name and checkpoint. The registry SHALL reject at load an alias that equals another model's name or another model's alias.

#### Scenario: Continue addresses the model by alias
- **WHEN** a request sets `model` to `local-4b`, an alias of `qwen3-4b-fp8`
- **THEN** it is served by `qwen3-4b-fp8`
- **AND** `resolved.model` is `qwen3-4b-fp8`

#### Scenario: Colliding alias
- **WHEN** a model entry declares an alias equal to another entry's name or alias
- **THEN** the registry fails to load

### Requirement: Tool calls stream in the OpenAI delta format
For a streamed completion that ends in tool calls, the platform SHALL emit, for each call in order, one event whose `delta.tool_calls` carries the call's `index`, `id`, `type`, `function.name`, and an empty `function.arguments`, then one event whose `delta.tool_calls` carries the same `index` and the call's `function.arguments`. These events SHALL precede the terminal event, whose `finish_reason` SHALL be `tool_calls`. The usage event, when requested, and `[DONE]` SHALL follow in the existing order. Text emitted before the call SHALL stream as ordinary content events.

#### Scenario: Continue edit turn
- **WHEN** a streamed tool request with `include_usage: true` ends in one tool call
- **THEN** the client receives the pre-generation event, any content events, the identity event, the arguments event, a terminal event with an empty delta and `finish_reason: "tool_calls"`, the usage event, and `[DONE]`, in that order

#### Scenario: Stream timeout while a call is buffered
- **WHEN** the model is still generating a tool call
- **THEN** the engine keeps yielding heartbeat chunks that produce no SSE event, so the per-token timeout measures an engine stall rather than the call's length

### Requirement: Tool definitions and tool-call history participate in run identity
The run manifest `request` block SHALL carry `tools_sha256`, the SHA-256 of the canonical JSON of the request's tools, or null when the request has none. `prompt_sha256` SHALL cover each message's `tool_calls` and `tool_call_id` when present, and SHALL hash a message without them exactly as before. The full tool schemas SHALL NOT be inlined into the manifest. `tools_sha256` SHALL enter the `run_id` preimage only when it is non-null; the rest of the `request` block SHALL enter it unchanged.

#### Scenario: Different tools, different identity
- **WHEN** two otherwise identical requests carry different `tools`
- **THEN** their `tools_sha256` values differ, and so would their `run_id`s

#### Scenario: A request without tools keeps its run identity
- **WHEN** a request carries no `tools`
- **THEN** `tools_sha256` does not enter the `run_id` preimage, the `request` block is hashed in its pre-existing shape, and the `run_id` equals the value computed before tool calling existed
- **AND** the returned manifest may still show `tools_sha256` as null

#### Scenario: Plain messages keep their hash
- **WHEN** a request contains only system, user, and assistant messages without tool calls
- **THEN** its `prompt_sha256` equals the value computed before tool-call fields existed
