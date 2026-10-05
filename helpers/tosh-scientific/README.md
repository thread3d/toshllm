# tosh-scientific

Numerical tools for the engine, backed by [NumPy](https://numpy.org) and
[SciPy](https://scipy.org). It is the same MCP server as [tosh-sympy](../tosh-sympy/README.md)
started with another tool set, so it has its own supervisor and its own worker:

```
llama-server --mcp-servers-json '{"mcpServers":{"sympy":{...},"scientific":{...}}}'
    ├── tosh_sympy/server.py                 SymPy tools, never loads NumPy or SciPy
    └── tosh_sympy/server.py scientific      supervisor, imports no math library
            └── tosh_sympy/worker.py scientific     NumPy on the first call, SciPy per operation
```

`scripts/build-sympy.sh` adds pinned NumPy and SciPy wheels to the runtime in
`vendor/tosh-sympy`, next to the CPython and SymPy already there. There is one interpreter.
BLAS and LAPACK come from the Accelerate framework of macOS, so no OpenBLAS is bundled. In
the app the tools are off until **Scientific computing (NumPy and SciPy)** is switched on in
the chat settings, under Agents, independently of the SymPy switch.

SymPy answers exactly; these tools answer in floating point. Use SymPy for algebra, closed
forms and verification, and these for integrals without a closed form, optimization, data,
signals and matrices of decimal numbers.

When both switches are on, the chat adds one rule to the system text
(`ScientificToolsService.routingRule`, 85 tokens): SymPy is the default, the numerical tools
are for requests that ask for a numerical or approximate answer or bring data, and an exact
or unspecified request is never answered with an approximation. Without it a model takes
`scientific_compute` for a plain "integrate x^2 from 0 to 3". The rule is not repeated in
the tool definitions.

## Tools

| Tool | Operations |
|---|---|
| `scientific_compute` | integrate (definite, up to 3 variables), root (one equation or a system), interpolate, evaluate |
| `scientific_linalg` | solve, inverse, determinant, rank, condition, eigen, svd, qr, least_squares, multiply, norm |
| `scientific_optimize` | minimize, maximize (bounds and constraints), curve_fit, fit_residuals |
| `scientific_signal` | fft, ifft, psd, lowpass, highpass, bandpass, bandstop, peaks, convolve, correlate |
| `scientific_ode` | solve_ivp (systems, and higher orders reduced to first order) |
| `scientific_stats` | describe, percentile, correlation, regression, ttest, confidence_interval, distribution |

The JSON schemas live in `tosh_scientific/schema.py`, about 1530 tokens in all. Some
arguments are accepted without being advertised (`max_values`, `parameters`, `method`,
`tolerance`, `bounds`, filter `order`, peak `height`, `distance` and `prominence`, and others
listed under `accepted` there).

A result is one JSON object:

```json
{"success": true, "operation": "integrate", "value": 0.58367089993, "error_estimate": 2.7e-11,
 "method": "adaptive quadrature (QUADPACK)", "warnings": []}
{"success": false, "operation": "solve", "error": {"code": "singular_matrix", "message": "..."}}
```

Error codes: `invalid_arguments`, `invalid_dimensions`, `invalid_expression`,
`undefined_symbol`, `limit_exceeded`, `non_finite_input`, `non_finite_result`, `not_converged`,
`missing_initial_guess`, `invalid_bracket`, `invalid_bounds`, `out_of_range`,
`singular_matrix`, `invalid_initial_conditions`, `integration_failed`, `not_supported`,
`numerical_error`, `timeout`, `memory_limit`, `worker_crashed`.

A call is read generously where its meaning is not in doubt: empty placeholder arguments are
dropped, interpolation points may come as pairs, a second-order equation may arrive already
split into first-order form, a bound of 1e15 or more is infinity. Where the meaning is in
doubt the call is refused or the reply carries a warning: values given for a variable the
expression does not have, a spectrum asked for with a cutoff and no sampling rate, an
optimum that ended on a bound.

Calls are checked against the user's request before they run, as described in [tosh-sympy](../tosh-sympy/README.md#checking-the-call-against-the-request).

A failure is never reported as a value. A divergent integral, a solver that did not converge,
a singular matrix, a root search without a sign change or a result holding NaN or infinity
comes back with `success: false` and its code. An ill-conditioned matrix or a fit whose
covariance could not be estimated is answered with a warning.

## Data in, data out

Numbers go in as JSON lists, or as a generator when they are too many to type:

```json
{"linspace": [0, 1, 1000]}   {"arange": [0, 10, 0.5]}   {"random": [500, 500], "seed": 1}
{"zeros": [3, 3]}   {"ones": [100]}   {"identity": 4}
```

A signal can be given as a formula in `t` with `sample_rate` and `duration`, so the model
never types the samples.

A result with more than 32 values comes back as its shape, minimum, maximum, mean, standard
deviation and both ends; `max_values` raises that up to 1000.

| Limit | Value |
|---|---|
| Values in one array | 1 000 000 |
| Matrix side | 1000 |
| FFT length | 1 048 576 |
| ODE points reported | 2000 |
| Optimizer iterations | 2000 |
| Integration variables | 3 |
| Unknowns of a system or a fit | 32 |
| Request size | 1 MB |

## What the model can and cannot send

The model sends an operation name, numbers and options. Each operation is a Python function
in `tosh_scientific/ops.py` reached through a fixed table; nothing in a request names a
module, a function of NumPy or SciPy, an attribute or a path.

A formula such as `a*exp(-b*x)` is parsed by the restricted grammar of the SymPy helper
(`tosh_sympy/mathlang.py`) and the resulting tree is walked into closures over a fixed table
of NumPy functions (`tosh_scientific/numeric.py`). No code is generated or evaluated, and
`lambdify` is not used. Text like `np.sin(x)`, `scipy.optimize.minimize(...)` or
`__import__('os')` is rejected by the parser.

The worker has the same confinement as the SymPy one: macOS sandbox profile and Python audit
hook with no network, no file writes, no new processes and no reads under `/Users` or
`/Volumes` outside the runtime. Nothing is unpickled and no file is loaded.

- `TOSH_SYMPY_TIMEOUT_MS` (default 15000) and `TOSH_SYMPY_MEMORY_MB` (default 1024) apply to
  this worker too: the supervisor kills it past either one and answers `timeout` or
  `memory_limit`. The next call starts a fresh worker.
- `TOSH_SCIENTIFIC_THREADS` (default 1, at most 16) caps the threads of Accelerate. One
  thread keeps a long decomposition from competing with the model for cores.
- The worker exits after `TOSH_SYMPY_IDLE_SECONDS` (default 300) without use.

## Tests

```sh
./scripts/build-sympy.sh
vendor/tosh-sympy/python/bin/python3 -I helpers/tosh-scientific/tests/test_scientific.py
TOSH_SYMPY_E2E_MODEL=~/models/Qwen3-4B-Q4_K_M.gguf ./scripts/test.sh --filter SymPyEngine
```
