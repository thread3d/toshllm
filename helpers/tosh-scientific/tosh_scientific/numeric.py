# ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
# Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Turns a parsed mathematical expression into a NumPy function.

The text is parsed by the same restricted grammar the symbolic tools use. The resulting
tree is walked once and each node becomes a closure over a fixed table of NumPy and SciPy
functions. No code is generated, compiled or evaluated.
"""

import numpy as np

from .arrays import SciError

_sympy = None
_FUNCTIONS = None


def sympy():
    """SymPy is only needed for requests that carry an expression, so it loads on the first one."""
    global _sympy, _FUNCTIONS
    if _sympy is None:
        import sympy as sp
        from scipy import special
        _sympy = sp
        _FUNCTIONS = {
            sp.sin: np.sin, sp.cos: np.cos, sp.tan: np.tan,
            sp.cot: lambda x: 1 / np.tan(x), sp.sec: lambda x: 1 / np.cos(x), sp.csc: lambda x: 1 / np.sin(x),
            sp.asin: np.arcsin, sp.acos: np.arccos, sp.atan: np.arctan,
            sp.acot: lambda x: np.arctan(1 / x), sp.atan2: np.arctan2,
            sp.sinh: np.sinh, sp.cosh: np.cosh, sp.tanh: np.tanh,
            sp.coth: lambda x: 1 / np.tanh(x), sp.sech: lambda x: 1 / np.cosh(x), sp.csch: lambda x: 1 / np.sinh(x),
            sp.asinh: np.arcsinh, sp.acosh: np.arccosh, sp.atanh: np.arctanh,
            sp.exp: np.exp, sp.log: np.log, sp.Abs: np.abs, sp.sign: np.sign,
            sp.floor: np.floor, sp.ceiling: np.ceil,
            sp.erf: special.erf, sp.erfc: special.erfc, sp.gamma: special.gamma,
            sp.factorial: lambda x: special.gamma(x + 1),
            sp.Heaviside: lambda x, *_: np.heaviside(x, 0.5),
            sp.LambertW: lambda x: np.real(special.lambertw(x)),
            sp.Mod: np.mod,
        }
    return _sympy


def parse(text, declared=(), functions=None):
    sympy()
    from tosh_sympy.mathlang import Context, MathError, parse_expression
    try:
        return parse_expression(_text(text), Context(None, functions, declared))
    except MathError as error:
        raise SciError(error.code, error.message)


def parse_equation(text, declared=(), functions=None):
    """left - right of "left = right"; a bare expression counts as "expression = 0"."""
    sp = sympy()
    from tosh_sympy.mathlang import Context, MathError, parse_relation
    try:
        relation = parse_relation(_text(text), Context(None, functions, declared))
    except MathError as error:
        raise SciError(error.code, error.message)
    if not isinstance(relation, sp.Eq):
        raise SciError("invalid_arguments", "expected an equation, not an inequality")
    return relation.lhs - relation.rhs


def parse_relation(text, declared=()):
    sympy()
    from tosh_sympy.mathlang import Context, MathError
    from tosh_sympy.mathlang import parse_relation as parse_any
    try:
        return parse_any(_text(text), Context(None, None, declared))
    except MathError as error:
        raise SciError(error.code, error.message)


def _text(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise SciError("invalid_arguments", "an expression must be a string")
    return value if isinstance(value, str) else repr(value)


def constant(value, name):
    """A number given as a JSON number or as constant math text such as pi/2 or -oo."""
    if isinstance(value, bool):
        raise SciError("invalid_arguments", f"'{name}' must be a number")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
        expr = parse(value)
        if expr.free_symbols:
            raise SciError("invalid_arguments", f"'{name}' must be a number, not an expression with variables")
        sp = sympy()
        if expr in (sp.oo, -sp.oo):
            return float("inf") if expr == sp.oo else float("-inf")
        number = complex(expr.evalf(17))
        if number.imag:
            raise SciError("invalid_arguments", f"'{name}' must be a real number")
        return number.real
    raise SciError("invalid_arguments", f"'{name}' must be a number")


def compile_expression(expr, symbols):
    """A function of len(symbols) positional arguments, scalars or arrays, for expr."""
    sp = sympy()
    position = {symbol: index for index, symbol in enumerate(symbols)}
    unknown = sorted(s.name for s in expr.free_symbols if s not in position)
    if unknown:
        raise SciError("undefined_symbol",
                       f"'{unknown[0]}' has no value; name it as a variable or give it in 'parameters'")

    def build(node):
        if node.is_Symbol:
            index = position[node]
            return lambda values: values[index]
        if node.is_Number or node.is_NumberSymbol:
            if node in (sp.oo, -sp.oo, sp.zoo, sp.nan):
                raise SciError("invalid_expression", "the expression is not finite")
            number = float(node)
            return lambda values: number
        if node is sp.I:
            raise SciError("not_supported", "complex expressions are not supported numerically")
        if isinstance(node, sp.Add):
            parts = [build(arg) for arg in node.args]

            def add(values):
                total = parts[0](values)
                for part in parts[1:]:
                    total = total + part(values)
                return total
            return add
        if isinstance(node, sp.Mul):
            parts = [build(arg) for arg in node.args]

            def multiply(values):
                total = parts[0](values)
                for part in parts[1:]:
                    total = total * part(values)
                return total
            return multiply
        if isinstance(node, sp.Pow):
            base, exponent = build(node.base), node.exp
            if exponent == sp.Rational(1, 2):
                return lambda values: np.sqrt(base(values))
            if exponent == -1:
                return lambda values: 1.0 / base(values)
            if exponent.is_Integer and 0 < int(exponent) <= 64:
                power = int(exponent)
                return lambda values: base(values) ** power
            if exponent.is_Number:
                power = float(exponent)
                return lambda values: np.power(np.asarray(base(values), dtype=float), power)
            if node.base is sp.E:
                inner = build(exponent)
                return lambda values: np.exp(inner(values))
            inner = build(exponent)
            return lambda values: np.power(np.asarray(base(values), dtype=float), inner(values))
        if isinstance(node, (sp.Min, sp.Max)):
            parts = [build(arg) for arg in node.args]
            reduce = np.minimum if isinstance(node, sp.Min) else np.maximum

            def extreme(values):
                total = parts[0](values)
                for part in parts[1:]:
                    total = reduce(total, part(values))
                return total
            return extreme
        function = _FUNCTIONS.get(node.func)
        if function is None:
            raise SciError("not_supported", f"'{node.func.__name__}' cannot be evaluated numerically here")
        parts = [build(arg) for arg in node.args]
        return lambda values: function(*[part(values) for part in parts])

    compiled = build(expr)

    def call(*values):
        # NumPy scalars give inf and nan where Python floats would raise
        values = [np.float64(v) if isinstance(v, (int, float)) else v for v in values]
        with np.errstate(all="ignore"):
            return compiled(values)
    return call


def symbols(names, what="variables"):
    sp = sympy()
    from tosh_sympy.mathlang import MathError, check_identifier
    result = []
    for name in names:
        if not isinstance(name, str):
            raise SciError("invalid_arguments", f"'{what}' must be names")
        try:
            result.append(sp.Symbol(check_identifier(name.strip())))
        except MathError as error:
            raise SciError(error.code, error.message)
    return result
