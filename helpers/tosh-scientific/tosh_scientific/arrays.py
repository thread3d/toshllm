# ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
# Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Numeric input and output: what a request may carry in, and how much of a result goes back."""

import math

import numpy as np

MAX_ELEMENTS = 1_000_000
MAX_LISTED_VALUES = 32
MAX_REQUESTED_VALUES = 1000
MAX_MATRIX_SIDE = 1000
DIGITS = 12


class SciError(Exception):
    def __init__(self, code, message, **extra):
        super().__init__(message)
        self.code = code
        self.message = message
        self.extra = extra


def _size(shape, name):
    if not isinstance(shape, (list, tuple)):
        shape = [shape]
    if not shape or len(shape) > 3 or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in shape):
        raise SciError("invalid_arguments", f"'{name}': a shape is one to three positive integers")
    if math.prod(shape) > MAX_ELEMENTS:
        raise SciError("limit_exceeded", f"'{name}' would hold {math.prod(shape)} values; the limit is {MAX_ELEMENTS}",
                       limit=MAX_ELEMENTS)
    return tuple(shape)


def _generated(spec, name):
    """Data too long to type out: {"linspace": [0, 1, 1000]}, {"random": [500, 500], "seed": 1}, ..."""
    if "linspace" in spec or "arange" in spec:
        key = "linspace" if "linspace" in spec else "arange"
        args = spec[key]
        if not isinstance(args, list) or len(args) != 3 or any(isinstance(a, bool) or not isinstance(a, (int, float)) for a in args):
            raise SciError("invalid_arguments", f"'{name}': {key} takes [start, stop, {'count' if key == 'linspace' else 'step'}]")
        if key == "linspace":
            count = _size(int(args[2]), name)[0]
            return np.linspace(float(args[0]), float(args[1]), count)
        if args[2] == 0 or (args[1] - args[0]) / args[2] > MAX_ELEMENTS:
            raise SciError("limit_exceeded", f"'{name}': that range holds more than {MAX_ELEMENTS} values", limit=MAX_ELEMENTS)
        return np.arange(float(args[0]), float(args[1]), float(args[2]))
    if "random" in spec:
        shape = _size(spec["random"], name)
        seed = spec.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise SciError("invalid_arguments", f"'{name}': seed must be a non-negative integer")
        generator = np.random.default_rng(seed)
        kind = spec.get("distribution", "normal")
        if kind == "normal":
            return generator.standard_normal(shape)
        if kind == "uniform":
            return generator.random(shape)
        raise SciError("invalid_arguments", f"'{name}': distribution must be normal or uniform")
    for key, make in (("zeros", np.zeros), ("ones", np.ones)):
        if key in spec:
            return make(_size(spec[key], name))
    if "identity" in spec:
        return np.eye(_size(spec["identity"], name)[0])
    raise SciError("invalid_arguments", f"'{name}': expected numbers, or one of linspace, arange, random, zeros, ones, identity")


def _number(value, name):
    if isinstance(value, bool):
        raise SciError("invalid_arguments", f"'{name}' must hold numbers")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            from . import numeric
            return numeric.constant(value, name)
    raise SciError("invalid_arguments", f"'{name}' must hold numbers")


def array(value, name, dimensions=None, required=True):
    """A float array from a JSON list or a generator object. dimensions limits how many axes it may have."""
    if value is None:
        if required:
            raise SciError("invalid_arguments", f"'{name}' is required")
        return None
    if isinstance(value, dict):
        result = _generated(value, name)
    elif isinstance(value, list):
        if not value:
            raise SciError("invalid_arguments", f"'{name}' is empty")
        nested = isinstance(value[0], list)
        if nested:
            width = len(value[0])
            if width == 0 or any(not isinstance(row, list) or len(row) != width for row in value):
                raise SciError("invalid_dimensions", f"every row of '{name}' must have the same length")
            if len(value) * width > MAX_ELEMENTS:
                raise SciError("limit_exceeded", f"'{name}' holds more than {MAX_ELEMENTS} values", limit=MAX_ELEMENTS)
            result = np.array([[_number(v, name) for v in row] for row in value], dtype=float)
        else:
            if len(value) > MAX_ELEMENTS:
                raise SciError("limit_exceeded", f"'{name}' holds more than {MAX_ELEMENTS} values", limit=MAX_ELEMENTS)
            result = np.array([_number(v, name) for v in value], dtype=float)
    else:
        raise SciError("invalid_arguments", f"'{name}' must be a list of numbers")
    if not np.all(np.isfinite(result)):
        raise SciError("non_finite_input", f"'{name}' contains NaN or infinity")
    if dimensions == (1,) and result.ndim == 2 and 1 in result.shape:
        # a vector written as one row or one column
        result = result.ravel()
    if dimensions is not None and result.ndim not in dimensions:
        wanted = " or ".join({1: "a list of numbers", 2: "a list of rows"}[d] for d in dimensions)
        raise SciError("invalid_dimensions", f"'{name}' must be {wanted}")
    return result


def matrix(value, name, required=True):
    result = array(value, name, (2,), required)
    if result is not None and max(result.shape) > MAX_MATRIX_SIDE:
        raise SciError("limit_exceeded", f"'{name}' is {result.shape[0]} x {result.shape[1]}; "
                       f"the limit is {MAX_MATRIX_SIDE} per side", limit=MAX_MATRIX_SIDE)
    return result


def number(value):
    """A finite float rounded for output, or SciError: a NaN is never reported as a result."""
    if isinstance(value, (complex, np.complexfloating)):
        if abs(value.imag) <= 1e-12 * max(1.0, abs(value.real)):
            return number(value.real)
        return {"re": number(value.real), "im": number(value.imag)}
    value = float(value)
    if not math.isfinite(value):
        raise SciError("non_finite_result", "the computation produced NaN or infinity")
    if value == 0:
        return 0.0
    return float(f"{value:.{DIGITS}g}")


def values(data, limit=None):
    """A nested list when it is short, otherwise its shape, statistics and both ends."""
    data = np.asarray(data)
    limit = MAX_LISTED_VALUES if limit is None else limit
    if data.ndim == 0:
        return number(data.item())
    if np.iscomplexobj(data):
        if np.all(np.abs(data.imag) <= 1e-12 * np.maximum(1.0, np.abs(data.real))):
            return values(data.real, limit)
        return {"re": values(data.real, limit), "im": values(data.imag, limit)}
    if not np.all(np.isfinite(data)):
        raise SciError("non_finite_result", "the computation produced NaN or infinity")
    if data.size <= limit:
        return _nested(data)
    flat = data.ravel()
    return {"shape": list(data.shape), "min": number(flat.min()), "max": number(flat.max()),
            "mean": number(flat.mean()), "std": number(flat.std()),
            "first": _nested(flat[:8]), "last": _nested(flat[-8:]),
            "summarized": True}


def _nested(data):
    if data.ndim == 1:
        return [number(v) for v in data]
    return [_nested(row) for row in data]


def listed_limit(args):
    """How many values the caller is willing to read back in full."""
    limit = args.get("max_values")
    if limit is None:
        return MAX_LISTED_VALUES
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_REQUESTED_VALUES:
        raise SciError("invalid_arguments", f"'max_values' must be between 1 and {MAX_REQUESTED_VALUES}")
    return limit


def integer(args, name, default, low, high):
    value = args.get(name)
    if value is None:
        return default
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SciError("invalid_arguments", f"'{name}' must be an integer")
    if not low <= value <= high:
        raise SciError("limit_exceeded" if value > high else "invalid_arguments",
                       f"'{name}' must be between {low} and {high}", limit=high)
    return value


def real(args, name, default=None, required=False, low=None, high=None):
    value = args.get(name)
    if value is None:
        if required:
            raise SciError("invalid_arguments", f"'{name}' is required")
        return default
    value = _number(value, name)
    if not math.isfinite(value):
        raise SciError("invalid_arguments", f"'{name}' must be finite")
    if (low is not None and value < low) or (high is not None and value > high):
        raise SciError("invalid_arguments", f"'{name}' is out of range")
    return value


def choice(args, name, options, default):
    value = args.get(name)
    if value is None:
        return default
    if value not in options:
        raise SciError("invalid_arguments", f"'{name}' must be one of: {', '.join(options)}")
    return value
