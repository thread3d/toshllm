# ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
# Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tool definitions for the numerical tools. Nothing here may import NumPy: the supervisor
reads this module and must stay small.

Every word is paid for in each conversation that offers the tools. "accepted" lists
arguments the helper takes without advertising them.
"""

_TEXT = {"type": "string"}
_NUMBER = {"type": "number"}
_BOUND = {"type": ["number", "string"], "description": "Number, or \"oo\" / \"-oo\" for infinity."}
_ARRAY = {"type": "array"}
_OBJECT = {"type": "object"}
_COMMON = ("max_values", "parameters")

TOOLS = {
    "compute": {
        "description": "Numerical (floating point) calculus: definite integral, root of an equation, "
                       "interpolation or spline through data points, evaluation.",
        "operations": ["integrate", "root", "interpolate", "evaluate"],
        "properties": {
            "expression": {**_TEXT, "description": "Function, sin(x**2), or for root the whole equation, "
                                                   "cos(x) = x."},
            "variable": _TEXT,
            "lower": _BOUND,
            "upper": _BOUND,
            "bracket": {**_ARRAY, "description": "root: [a, b] with a sign change."},
            "initial_guess": {**_ARRAY, "description": "root: start value."},
            "x": _ARRAY,
            "y": _ARRAY,
            "at": {**_ARRAY, "description": "interpolate: x values to estimate."},
            "kind": {"type": "string", "enum": ["linear", "cubic", "pchip", "akima"]},
        },
        "accepted": ("equations", "variables", "bounds", "tolerance", "data", "dx", "method", "extrapolate"),
        "required": ["operation"],
    },
    "linalg": {
        "description": "Numerical linear algebra on a matrix of decimal numbers given as rows. solve and "
                       "least_squares use matrix * x = other.",
        "operations": ["solve", "inverse", "determinant", "rank", "condition", "eigen", "svd", "qr",
                       "least_squares", "multiply", "norm"],
        "properties": {
            "matrix": {**_ARRAY, "description": "[[1, 2], [3, 4]]"},
            "other": _ARRAY,
        },
        "accepted": ("kind",),
        "required": ["operation", "matrix"],
    },
    "optimize": {
        "description": "Numerical optimization: minimum or maximum of a function, curve fitting of data to a "
                       "model, nonlinear least squares.",
        "operations": ["minimize", "maximize", "curve_fit", "fit_residuals"],
        "properties": {
            "expression": {**_TEXT, "description": "Objective, or the model for curve_fit, e.g. a*exp(-b*x)."},
            "initial_guess": {**_OBJECT, "description": "{\"x\": 1, \"y\": 0}; for curve_fit the parameters."},
            "bounds": {**_OBJECT, "description": "{\"x\": [0, 10]}"},
            "constraints": {**_ARRAY, "description": "[\"x + y <= 1\"]"},
            "x": _ARRAY,
            "y": _ARRAY,
        },
        "accepted": ("variables", "variable", "method", "max_iterations", "residuals"),
        "required": ["operation"],
    },
    "signal": {
        "description": "Processing of a signal sampled in time: FFT and dominant frequencies, power spectral "
                       "density, Butterworth filters, peak detection, convolution, correlation. The signal is a "
                       "formula in t or samples.",
        "operations": ["fft", "psd", "lowpass", "highpass", "bandpass", "bandstop", "peaks", "convolve",
                       "correlate", "ifft"],
        "properties": {
            "expression": {**_TEXT, "description": "Formula of the signal in t, e.g. sin(2*pi*50*t), with "
                                                   "sample_rate and duration. Never type out its samples."},
            "sample_rate": {**_NUMBER, "description": "Hz."},
            "duration": {**_NUMBER, "description": "Seconds."},
            "values": {**_ARRAY, "description": "Measured samples."},
            "cutoff": {"description": "Hz; [low, high] for a band."},
            "other": _ARRAY,
        },
        "accepted": ("order", "top", "window", "segment", "height", "distance", "prominence", "threshold",
                     "mode", "real", "imag", "variable"),
        "required": ["operation"],
    },
    "ode": {
        "description": "Solve an initial value problem for ordinary differential equations numerically, "
                       "over an interval, from given initial values.",
        "operations": ["solve_ivp"],
        "properties": {
            "equations": {**_ARRAY, "description": "Each equation copied from the request with its own left "
                                                    "side. One per unknown."},
            "initial_conditions": {**_OBJECT, "description": "Start value of every unknown: {\"y\": 1}; "
                                                             "add \"y'\" when the equation has y''."},
            "interval": {**_ARRAY, "description": "[start, end]"},
            "at": {**_ARRAY, "description": "Times to report."},
        },
        "accepted": ("variable", "method", "tolerance", "points", "derivatives"),
        "required": ["operation", "equations", "initial_conditions", "interval"],
    },
    "stats": {
        "description": "Statistics on numeric data: descriptive summary, percentiles, correlation, linear "
                       "regression, t-tests, confidence interval, probability distributions.",
        "operations": ["describe", "percentile", "correlation", "regression", "ttest", "confidence_interval",
                       "distribution"],
        "properties": {
            "values": _ARRAY,
            "x": _ARRAY,
            "y": _ARRAY,
            "mean": {**_NUMBER, "description": "ttest: value to test against."},
            "q": {"description": "percentile: 0-100."},
            "name": {"type": "string", "enum": ["normal", "t", "chi2", "exponential", "uniform", "binomial", "poisson"]},
            "parameters": {**_OBJECT, "description": "distribution: mean, std, df, n, p or rate."},
            "at": {**_NUMBER, "description": "distribution: point for the density and cumulative probability."},
            "p": {**_NUMBER, "description": "distribution: probability whose quantile is wanted."},
            "between": _ARRAY,
            "level": {**_NUMBER, "description": "confidence_interval: default 0.95."},
        },
        "accepted": ("method", "paired", "equal_variance", "alternative"),
        "required": ["operation"],
    },
}


def definitions():
    """The tools in MCP tools/list form."""
    listed = []
    for name, tool in TOOLS.items():
        properties = {"operation": {"type": "string", "enum": tool["operations"]}}
        properties.update(tool["properties"])
        listed.append({
            "name": name,
            "description": tool["description"],
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": tool["required"],
                "additionalProperties": False,
            },
            "annotations": {"readOnlyHint": True, "openWorldHint": False},
        })
    return listed


def check(name, arguments):
    """Returns the operation, or raises ValueError naming what is wrong with the call."""
    tool = TOOLS.get(name)
    if tool is None:
        raise ValueError(f"unknown tool '{str(name)[:40]}'")
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be an object")
    operation = arguments.get("operation")
    if operation not in tool["operations"]:
        raise ValueError("'operation' must be one of: " + ", ".join(tool["operations"]))
    allowed = set(tool["properties"]) | set(tool["accepted"]) | set(_COMMON)
    unknown = [key for key in arguments if key != "operation" and key not in allowed]
    if unknown:
        raise ValueError(f"unknown argument '{str(unknown[0])[:40]}'; allowed: " + ", ".join(tool["properties"]))
    return operation
