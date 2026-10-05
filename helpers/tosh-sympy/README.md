# tosh-sympy

Symbolic math tools for the engine, backed by [SymPy](https://www.sympy.org). It is an MCP
server over stdio, so `llama-server` lists its tools on `/tools` next to its own and any
OpenAI-compatible client of the server can call them.

```
llama-server --mcp-servers-json '{"mcpServers":{"sympy":{...}}}'
    └── tosh_sympy/server.py     supervisor, never imports SymPy (about 8 MB)
            └── tosh_sympy/worker.py     SymPy, started on the first call
```

`scripts/build-sympy.sh` builds the runtime in `vendor/tosh-sympy` from a pinned CPython,
SymPy and mpmath, and `make-app.sh` copies it to `Contents/Resources/tosh-sympy`. No Python
from the system is used, to build or to run. In the app it is off until **Symbolic math
(SymPy)** is switched on in the chat settings, under Agents.

The same runtime carries NumPy and SciPy for the [numerical tools](../tosh-scientific/README.md).
Those run as a second server, `server.py scientific`, with a worker of their own: this one
never loads either library.

## Tools

| Tool | Operations |
|---|---|
| `sympy_expression` | simplify, expand, factor, cancel, together, apart, evaluate, differentiate, integrate, limit, series, summation, product, laplace_transform, inverse_laplace_transform, fourier_transform, inverse_fourier_transform |
| `sympy_solve` | solve (equations, systems, inequalities), solveset, nsolve, dsolve |
| `sympy_matrix` | determinant, inverse, transpose, rank, rref, nullspace, eigenvalues, eigenvectors, multiply, linear_solve |
| `sympy_verify` | equivalent (two expressions), solution (values against equations) |

With the six [numerical tools](../tosh-scientific/README.md) that makes ten tools and 70
operations. The JSON schemas live in `tosh_sympy/schema.py` and nowhere else. They are kept short on
purpose: every conversation that offers the tools carries them, about 1000 tokens in all. A
few arguments are accepted without being advertised (`assumptions`, `precision` and
`substitutions` on every tool, `transform_variable` on transforms).

A result is one JSON object:

```json
{"success": true, "operation": "integrate", "exact": "sqrt(pi)", "latex": "\\sqrt{\\pi}",
 "numeric": "1.77245385090552", "warnings": []}
{"success": false, "operation": "factor", "error": {"code": "invalid_expression", "message": "..."}}
```

Error codes: `invalid_arguments`, `invalid_expression`, `unknown_function`, `input_too_large`,
`too_large`, `no_result`, `not_supported`, `math_error`, `no_closed_form`, `timeout`,
`memory_limit`, `output_too_large`, `worker_crashed`, `runtime_unavailable`.

### When there is no exact answer

An exact result and an approximation are never mixed up. `exact` is `null` whenever the
value below it is not a closed form.

```json
{"success": true, "operation": "integrate", "exact": null,
 "unevaluated": "Integral(exp(sin(x)), (x, 0, 1))", "numeric": "1.63186960841805",
 "method": "numerical_integration", "error_estimate": "1.0e-51", "precision": 15,
 "timed_out_symbolic": false, "warnings": ["no closed form was obtained; ..."]}
{"success": false, "operation": "integrate", "exact": null, "timed_out": true,
 "unevaluated": "Integral(1/(x**3 + sin(x) + 1), x)",
 "error": {"code": "timeout", "message": "No closed form was obtained within the 15 s computation budget."}}
```

- A definite integral with finite numeric bounds and no free parameters falls back to
  numerical quadrature when SymPy leaves it open or runs out of time. The value is kept only
  if two different subdivisions of the interval agree, so a divergent or singular integral
  gets `no_closed_form`, not a number.
- An indefinite integral, an integral with a parameter or an infinite bound, a sum, a
  product, a limit or a transform SymPy cannot close returns `no_closed_form` with the
  `unevaluated` form.
- A request stopped by the time budget returns `timed_out: true`. For a series it also
  carries the expansion to the highest lower order that finished, under `partial`. For an
  integral it tries the numerical fallback, and one exact retry with the trigonometric
  functions rewritten as exponentials.
- `nsolve` never invents a starting point: without `initial_guess` it says so.
- `e` is Euler's number, like `E`, unless the request names it as a variable (`variable`,
  `variables`, or a key of `assumptions`, `substitutions` or `solution`).

## What the model can and cannot send

The model sends math, never code. `tosh_sympy/mathlang.py` tokenizes and parses each
expression itself and builds SymPy objects from a fixed table of functions and constants.
No text from the model reaches `sympify`, `parse_expr`, `eval` or `exec`, and `nsolve` runs on closures
instead of `lambdify`. SymPy does compile code of its own while it works (polynomial kernels,
its integral tables); that text is SymPy's, and a variable name can only be letters, digits
and underscores, so it stays inert wherever it is printed.

On top of that the worker:

- runs under a macOS sandbox profile with no network, no file writes, no new processes and no
  reads under `/Users` or `/Volumes` outside the runtime;
- has a Python audit hook that refuses the same things;
- is killed by the supervisor after `TOSH_SYMPY_TIMEOUT_MS` (default 15000, at most 20000) or
  above `TOSH_SYMPY_MEMORY_MB` (default 1024). After a timeout a fresh worker gets 5 s to
  report what it still can about the request;
- runs with a fixed hash seed. SymPy's heuristics walk sets, and with Python's random string
  hashing the same integral took 0.6 s in one process and never finished in the next;
- exits after `TOSH_SYMPY_IDLE_SECONDS` (default 300) without use.

Inputs are capped at 4000 characters per expression and 64 KB per request, results at 6000
characters per field.

## Checking the call against the request

A model sometimes writes a call that is not the user's problem: it drops a denominator, writes
8 for infinity, adds an initial condition or loses a sample. The helper would compute that call
correctly. So the agent passes the user's own message next to the arguments, as `_source`
(`{"request": ..., "context": ...}`); the model's own `_` arguments are dropped first.
`tosh_sympy/anchor.py` reads the formulas, numbers, lists, matrices, conditions and ranges out
of that text with the same restricted grammar and compares them with the call before it runs:

- **inconsistent**: the call is not run and the reply says why (`transcription_mismatch`).
  A formula that is only part of, one side of, or different from the one in the request; a
  finite limit where the request goes to infinity; swapped or changed limits; an initial
  condition the request does not state; a list with a value dropped or changed; a scalar taken
  from inside a list; a matrix with a row or column missing; the opposite operation
  (low-pass for high-pass); a series order that stops before the power asked for.
- **uncertain**: nothing in the request to compare with, as in a word problem. The reply is
  `needs_review`; the agent asks the model, in a separate request that sees only the user's text
  and what the helper read, whether the two state the same problem, and runs the call only on
  "consistent".
- **consistent**: the call runs. A number of the request the call leaves out is reported in
  `warnings`.

Every reply carries `interpreted_input`, short lines with what was read (`limits: [0, +oo)`,
`initial conditions: y(0) = 1`, `x: n=5 [...]`), and `result_kind`, `exact` or `approximate`.
The chat card shows those lines under the result. A call without `_source`, as from another
client, runs as before.

`_source` and `_reviewed` are honoured only next to `_trust` carrying the key the server was
started with (`TOSH_TRUST_KEY` in the server's environment). The app makes a new key on every
launch. A client or a model that sets them without it gets them dropped, the call runs as a plain
call, and `warnings` says the fields were ignored. A client cannot vouch for its own call.

## Four ways to reach the tools

| Path | Who runs the loop | Checks of this page |
|---|---|---|
| Tosh agent, `/v1/chat/completions` | the engine (`server.py agent`) | all of them |
| Native chat of the app | the same engine agent | all of them |
| Raw inference, `/v1/chat/completions` with `tools` or `X-Tosh-Agent: off` | the client | none |
| Direct, `POST /tools` | nobody: one call | the call itself (a plain call, no `_source`) |

Only the agent guarantees that a computed number in the answer came from a validated tool
result. Raw inference is the model as it is, whatever tools the client gives it.

### The Tosh agent

Started with `--mcp-agent agent` and a third MCP server, `server.py agent`, the engine answers
`/v1/chat/completions` the way the app's chat does: a request for a computed result has to go
through a tool or a question back; every call is checked against the user's message; a turn
keeps the results it validated when a later call is refused; and the answer may only state
numbers from the user's messages or from a validated result. Without the math servers it
answers as the model does.

Which requests it takes, per request:

| Request | Answered by |
|---|---|
| `X-Tosh-Agent: off` (or `raw`) | the model, always |
| `X-Tosh-Agent: on` | the agent; 400 if the engine runs none or the request brings tools |
| no header, from a page in a browser (`Origin` or `Sec-Fetch-Site`) | the agent |
| no header, any other client | the agent, unless the engine was started with `--mcp-agent-explicit` |
| `tools`, `functions`, `grammar`, `json_schema` or `response_format`, without `on` | the model |

The app starts the engine with `--mcp-agent-explicit` unless **Answer with the tools over the
API** is on in the chat settings, so an API client that does not ask keeps the model as it is.
Its own chat and the bundled web chat always get the agent.

Messages with `tool_calls` or the `tool` role are refused with 400: the tools run inside, and a
result written by a client could otherwise pass for one the engine computed. Earlier messages, the
client's or the model's, are context: only a call the agent ran in this turn can appear in
`validated_results`. Turns of different clients run in parallel. The sampling
fields of the request (`temperature`, `top_p`, `top_k`, `min_p`, `seed`, the penalties,
`samplers`, `max_tokens`, `chat_template_kwargs`, `model`...) go to every model pass of the
turn; in router mode every pass goes to the requested model. `max_tokens` applies to each pass,
as the client set it, never lowered. A turn takes at most 10 passes, or what the request sets in
`"tosh": {"max_rounds": n}` (1 to 100); the app sends its **Maximum agent turns**.

### Streaming

A streaming client gets ordinary `chat.completion.chunk` events. While the turn runs it gets
`: keep-alive` comments and chunks with an empty `delta` and a `tosh.event`; the answer comes in
one chunk at the end, once it is checked, never as a draft. Clients that ignore unknown fields
see a normal stream with a slow first token.

```
data: {"object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":null}],
       "tosh":{"version":1,"event":{"type":"tool_call","pass":1,"call":{"tool":"scientific_compute","status":"ok",...}}}}
data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"The integral is 6.49393940227..."}}],...}
data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{...},"tosh":{"version":1,...}}
data: [DONE]
```

Event types: `intent`, `pass` (`state` `start` or `end`, with that pass's tokens) and
`tool_call` (the call as below, with its `id`, `state` and `result`).

Closing the connection stops the turn: the engine tells the agent, the agent drops the model
pass in flight, and no further pass or call runs.

### The `tosh` object

Every agent answer carries it, next to the usual fields. `version` changes only when a field
below changes meaning or goes away; new fields may appear.

| Field | Meaning |
|---|---|
| `version` | `1` |
| `intent` | `no_math`, `conceptual`, `computational`, `ambiguous` |
| `outcome` | `answered`, `validated_results_only`, `clarification_required`, `unresolved`, `turn_limit` |
| `calls[]` | `tool`, `arguments`, `status` (`ok` or an error code such as `transcription_mismatch`, `needs_review`, `no_closed_form`, `timeout`), `interpreted_input`, `result_kind` |
| `validated_results[]` | `tool`, `operation`, `result_kind`, `result` (the tool's own fields), `interpreted_input` |
| `passes` | model passes the turn took |
| `seconds` | wall time of the turn |

`usage` adds up every pass. Status codes and the fields above never change language; the texts
the agent writes itself (a question back, the list of validated results) follow the language of
the conversation, Spanish or English, and English when in doubt.

### Direct calls

`GET /tools` lists the ten tools with their schemas; `POST /tools`
`{"tool": "scientific_compute", "params": {...}}` runs one. No model is involved, and the reply
is the JSON above. An infinite limit is `"oo"` or `"-oo"` (JSON has no infinity). A page in a
browser gets neither: its tool loop would bypass the agent.

### Reserved fields

`_source`, `_reviewed` and `_trust` belong to the engine. They are honoured only with the key
the helpers were started with, which no client sees; the agent drops whatever the model writes
in them and adds its own. A forged `_trust` gets the fields dropped with a warning.

### Example

```
llama-server -m model.gguf --jinja --mcp-agent agent --mcp-servers-json '{"mcpServers":{
  "sympy":      {"command": "python3", "args": ["-I", "-B", "tosh_sympy/server.py"],               "env": {"TOSH_TRUST_KEY": "<key>"}},
  "scientific": {"command": "python3", "args": ["-I", "-B", "tosh_sympy/server.py", "scientific"], "env": {"TOSH_TRUST_KEY": "<key>"}},
  "agent":      {"command": "python3", "args": ["-I", "-B", "tosh_sympy/server.py", "agent"],      "env": {"TOSH_TRUST_KEY": "<key>"},
                 "timeout_ms": 900000}}}'

curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -H 'X-Tosh-Agent: on' \
  -d '{"messages":[{"role":"user","content":"Integrate x^3/(e^x-1) from 0 to infinity."}]}'

curl http://127.0.0.1:8080/tools -H 'Content-Type: application/json' \
  -d '{"tool":"scientific_compute","params":{"operation":"integrate","expression":"x**3/(exp(x)-1)","lower":0,"upper":"oo"}}'
```

Any OpenAI client works with the base URL `http://host:8080/v1`; to send the header, use its
extra-headers option, or start the engine without `--mcp-agent-explicit`.

## Tests

```sh
./scripts/build-sympy.sh
vendor/tosh-sympy/python/bin/python3 -I helpers/tosh-sympy/tests/test_helper.py
vendor/tosh-sympy/python/bin/python3 -I helpers/tosh-sympy/tests/test_agent.py
TOSH_ENGINE_URL=http://127.0.0.1:8080 TOSH_ENGINE_AGENT=1 python3 helpers/tosh-sympy/tests/test_engine_api.py
TOSH_SYMPY_E2E_MODEL=~/models/Qwen3-4B-Q4_K_M.gguf ./scripts/test.sh --filter SymPyEngine
```
