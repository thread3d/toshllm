# ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
# Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Restricted mathematical expression language.

Text from the model is tokenized and parsed here into SymPy objects built from an
explicit whitelist. Nothing is ever handed to sympify(), parse_expr(), eval() or exec().
"""

import keyword
import re

import sympy as sp

MAX_SOURCE_CHARS = 4000
MAX_TOKENS = 2000
MAX_DEPTH = 60
MAX_INT_DIGITS = 300
MAX_IDENT_CHARS = 32
MAX_POWER_BITS = 20000
MAX_FACTORIAL_ARG = 2000


class MathError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


_TOKEN_RE = re.compile(r"""
    (?P<ws>\s+)
  | (?P<num>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
  | (?P<id>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<op>\*\*|==|<=|>=|!=|[-+*/^(),!'=<>])
""", re.VERBOSE)

# typographic characters models paste from prose
_TRANSLATE = str.maketrans({
    "−": "-", "–": "-", "×": "*", "·": "*", "⋅": "*",
    "÷": "/", "≤": "<=", "≥": ">=", "≠": "!=",
    "π": " pi ", "∞": " oo ", "’": "'", "′": "'",
})

CONSTANTS = {
    "pi": sp.pi, "E": sp.E, "e": sp.E, "I": sp.I,
    "oo": sp.oo, "inf": sp.oo, "infinity": sp.oo, "Infinity": sp.oo,
    "EulerGamma": sp.EulerGamma, "GoldenRatio": sp.GoldenRatio, "Catalan": sp.Catalan,
}

# names a request may claim back as plain variables by declaring them
DECLARABLE = {"e"}

ASSUMPTIONS = (
    "real", "positive", "negative", "nonnegative", "nonpositive", "nonzero",
    "integer", "rational", "complex", "even", "odd", "prime",
)

RELATIONS = {
    "=": sp.Eq, "==": sp.Eq, "!=": sp.Ne,
    "<": sp.Lt, "<=": sp.Le, ">": sp.Gt, ">=": sp.Ge,
}


def _is_big_integer(value, limit):
    return getattr(value, "is_Integer", False) and abs(int(value)) > limit


def _power(base, exponent):
    # an integer tower such as 9**9**9 would be evaluated eagerly and never return
    if base.is_Rational and exponent.is_Rational and base not in (0, 1, -1):
        size = max(abs(base.p), abs(base.q)).bit_length()
        if size * abs(exponent.p) > MAX_POWER_BITS * exponent.q:
            raise MathError("too_large", "power is too large to evaluate exactly")
    return sp.Pow(base, exponent)


def _bounded(function, limit=MAX_FACTORIAL_ARG):
    def call(*args):
        if any(_is_big_integer(a, limit) for a in args):
            raise MathError("too_large", f"argument is too large (limit {limit})")
        return function(*args)
    return call


def _log(x, base=None):
    return sp.log(x) if base is None else sp.log(x, base)


def _need_symbol(value, what):
    if not isinstance(value, sp.Symbol):
        raise MathError("invalid_expression", f"{what} must be a plain variable")
    return value


def _range(args, start):
    """(var, a, b) written either as a tuple or as three flat arguments."""
    rest = args[start:]
    if len(rest) == 1 and isinstance(rest[0], tuple):
        rest = rest[0]
    if len(rest) == 1:
        return (_need_symbol(rest[0], "the variable"),)
    if len(rest) == 3:
        return (_need_symbol(rest[0], "the variable"), rest[1], rest[2])
    raise MathError("invalid_expression", "expected a variable, or a variable with two bounds")


def _diff(expr, *rest):
    if not rest:
        raise MathError("invalid_expression", "diff needs a variable")
    spec = []
    for item in rest:
        if isinstance(item, sp.Symbol):
            spec.append(item)
        elif getattr(item, "is_Integer", False) and 0 <= int(item) <= 20 and spec:
            spec.append(int(item))
        else:
            raise MathError("invalid_expression", "diff takes variables and small orders")
    return sp.diff(expr, *spec)


def _integrate(*args):
    return sp.integrate(args[0], _range(args, 1))


def _summation(*args):
    return sp.summation(args[0], _range(args, 1))


def _product(*args):
    return sp.product(args[0], _range(args, 1))


def _limit(expr, var, point):
    return sp.limit(expr, _need_symbol(var, "the variable"), point, "+-")


# name -> (callable, minimum arguments, maximum arguments)
FUNCTIONS = {
    "sqrt": (sp.sqrt, 1, 1), "cbrt": (sp.cbrt, 1, 1), "root": (sp.root, 2, 2),
    "exp": (sp.exp, 1, 1), "log": (_log, 1, 2), "ln": (sp.log, 1, 1),
    "log2": (lambda x: sp.log(x, 2), 1, 1), "log10": (lambda x: sp.log(x, 10), 1, 1),
    "sin": (sp.sin, 1, 1), "cos": (sp.cos, 1, 1), "tan": (sp.tan, 1, 1),
    "cot": (sp.cot, 1, 1), "sec": (sp.sec, 1, 1), "csc": (sp.csc, 1, 1),
    "asin": (sp.asin, 1, 1), "acos": (sp.acos, 1, 1), "atan": (sp.atan, 1, 1),
    "acot": (sp.acot, 1, 1), "asec": (sp.asec, 1, 1), "acsc": (sp.acsc, 1, 1),
    "arcsin": (sp.asin, 1, 1), "arccos": (sp.acos, 1, 1), "arctan": (sp.atan, 1, 1),
    "atan2": (sp.atan2, 2, 2),
    "sinh": (sp.sinh, 1, 1), "cosh": (sp.cosh, 1, 1), "tanh": (sp.tanh, 1, 1),
    "coth": (sp.coth, 1, 1), "sech": (sp.sech, 1, 1), "csch": (sp.csch, 1, 1),
    "asinh": (sp.asinh, 1, 1), "acosh": (sp.acosh, 1, 1), "atanh": (sp.atanh, 1, 1),
    "abs": (sp.Abs, 1, 1), "Abs": (sp.Abs, 1, 1), "sign": (sp.sign, 1, 1),
    "floor": (sp.floor, 1, 1), "ceiling": (sp.ceiling, 1, 1), "ceil": (sp.ceiling, 1, 1),
    "re": (sp.re, 1, 1), "im": (sp.im, 1, 1), "arg": (sp.arg, 1, 1),
    "conjugate": (sp.conjugate, 1, 1), "conj": (sp.conjugate, 1, 1),
    "factorial": (_bounded(sp.factorial), 1, 1),
    "binomial": (_bounded(sp.binomial), 2, 2),
    "gamma": (_bounded(sp.gamma), 1, 1),
    "beta": (sp.beta, 2, 2), "zeta": (sp.zeta, 1, 1),
    "erf": (sp.erf, 1, 1), "erfc": (sp.erfc, 1, 1), "LambertW": (sp.LambertW, 1, 1),
    "Heaviside": (sp.Heaviside, 1, 1), "DiracDelta": (sp.DiracDelta, 1, 1),
    "Min": (sp.Min, 1, 8), "Max": (sp.Max, 1, 8), "min": (sp.Min, 1, 8), "max": (sp.Max, 1, 8),
    "gcd": (sp.gcd, 2, 2), "lcm": (sp.lcm, 2, 2), "Mod": (sp.Mod, 2, 2),
    "diff": (_diff, 2, 8), "Derivative": (_diff, 2, 8),
    "integrate": (_integrate, 2, 4), "Integral": (_integrate, 2, 4),
    "summation": (_summation, 2, 4), "Sum": (_summation, 2, 4),
    "product": (_product, 2, 4), "Product": (_product, 2, 4),
    "limit": (_limit, 3, 3),
}

# only these read a parenthesised (var, a, b) group
_TUPLE_FUNCTIONS = {"integrate", "Integral", "summation", "Sum", "product", "Product"}


def check_identifier(name):
    if (len(name) > MAX_IDENT_CHARS or name.startswith("_") or "__" in name
            or keyword.iskeyword(name) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name)):
        raise MathError("invalid_expression", f"'{name[:40]}' is not a valid variable name")
    return name


class Context:
    """Symbols and undefined functions shared by every expression of one request."""

    def __init__(self, assumptions=None, functions=None, declared=()):
        self.declared = DECLARABLE.intersection(declared)
        self.assumptions = {}
        for name, flags in (assumptions or {}).items():
            check_identifier(name)
            if isinstance(flags, str):
                flags = [flags]
            if not isinstance(flags, list) or not all(f in ASSUMPTIONS for f in flags):
                raise MathError("invalid_arguments",
                                f"assumptions for '{name}' must come from: {', '.join(ASSUMPTIONS)}")
            self.assumptions[name] = {flag: True for flag in flags}
        self.symbols = {}
        self.functions = {}
        self.function_variables = {}
        for name, variable in (functions or {}).items():
            check_identifier(name)
            if name in FUNCTIONS or name in CONSTANTS:
                raise MathError("invalid_arguments", f"'{name}' is a reserved name")
            self.functions[name] = sp.Function(name)
            self.function_variables[name] = self.symbol(variable)

    def symbol(self, name):
        check_identifier(name)
        if name in self.declared:
            pass
        elif name in DECLARABLE:
            raise MathError("invalid_arguments",
                            f"'{name}' is Euler's number here; to use it as a variable, name it in "
                            "'variable' or 'variables'")
        elif name in CONSTANTS or name in FUNCTIONS or name in self.functions:
            raise MathError("invalid_arguments", f"'{name}' cannot be used as a variable")
        if name not in self.symbols:
            self.symbols[name] = sp.Symbol(name, **self.assumptions.get(name, {}))
        return self.symbols[name]


class _Parser:
    def __init__(self, source, context):
        if not isinstance(source, str):
            raise MathError("invalid_arguments", "an expression must be a string")
        if len(source) > MAX_SOURCE_CHARS:
            raise MathError("input_too_large", f"expression is longer than {MAX_SOURCE_CHARS} characters")
        self.context = context
        self.tokens = self._tokenize(source.translate(_TRANSLATE))
        self.position = 0
        self.depth = 0

    @staticmethod
    def _tokenize(source):
        tokens = []
        position = 0
        while position < len(source):
            match = _TOKEN_RE.match(source, position)
            if match is None:
                raise MathError("invalid_expression",
                                f"unexpected character {source[position]!r} at position {position}")
            position = match.end()
            if match.lastgroup == "ws":
                continue
            tokens.append((match.lastgroup, match.group()))
            if len(tokens) > MAX_TOKENS:
                raise MathError("input_too_large", "expression has too many terms")
        if not tokens:
            raise MathError("invalid_expression", "expression is empty")
        return tokens

    def peek(self):
        return self.tokens[self.position] if self.position < len(self.tokens) else (None, None)

    def take(self, text=None):
        kind, value = self.peek()
        if kind is None or (text is not None and value != text):
            found = "end of expression" if kind is None else repr(value)
            raise MathError("invalid_expression",
                            f"expected {text!r} but found {found}" if text else f"unexpected {found}")
        self.position += 1
        return kind, value

    def at(self, *texts):
        kind, value = self.peek()
        return kind == "op" and value in texts

    def enter(self):
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise MathError("input_too_large", "expression is nested too deeply")

    def finish(self):
        if self.position != len(self.tokens):
            raise MathError("invalid_expression", f"unexpected {self.peek()[1]!r}")

    def sum(self):
        self.enter()
        value = self.term()
        while self.at("+", "-"):
            _, op = self.take()
            right = self.term()
            value = self._value(value) + self._value(right) if op == "+" else self._value(value) - self._value(right)
        self.depth -= 1
        return value

    def term(self):
        value = self.unary()
        while True:
            kind, text = self.peek()
            if kind == "op" and text in ("*", "/"):
                self.take()
                right = self._value(self.unary())
                value = self._value(value) * right if text == "*" else self._value(value) / right
            elif kind in ("num", "id") or (kind == "op" and text == "("):
                # implicit multiplication: 2x, 2(x + 1), (x + 1)(x - 1)
                value = self._value(value) * self._value(self.power())
            else:
                return value

    def unary(self):
        if not self.at("-", "+"):
            return self.power()
        _, sign = self.take()
        self.enter()
        value = self._value(self.unary())
        self.depth -= 1
        return -value if sign == "-" else value

    def power(self):
        base = self.postfix()
        if self.at("**", "^"):
            self.take()
            self.enter()
            exponent = self._value(self.unary())
            self.depth -= 1
            return _power(self._value(base), exponent)
        return base

    def postfix(self):
        value = self.atom()
        while self.at("!"):
            self.take()
            value = FUNCTIONS["factorial"][0](self._value(value))
        return value

    @staticmethod
    def _value(value):
        if isinstance(value, tuple):
            raise MathError("invalid_expression", "a parenthesised list is not allowed here")
        if not isinstance(value, sp.Basic):
            raise MathError("invalid_expression", "not a mathematical expression")
        return value

    def number(self, text):
        mantissa, _, exponent = text.lower().partition("e")
        whole, _, fraction = mantissa.partition(".")
        if len(whole) + len(fraction) > MAX_INT_DIGITS or len(exponent.lstrip("+-")) > 3:
            raise MathError("too_large", "number literal is too long")
        # decimals are read as exact fractions so the algebra stays exact
        value = sp.Rational(int((whole or "0") + fraction), 10 ** len(fraction))
        if exponent:
            value *= sp.Integer(10) ** int(exponent)
        return value

    def arguments(self, allow_tuple):
        args = []
        self.take("(")
        if not self.at(")"):
            while True:
                value = self.sum()
                if isinstance(value, tuple) and not allow_tuple:
                    self._value(value)
                args.append(value)
                if not self.at(","):
                    break
                self.take()
        self.take(")")
        return args

    def primes(self):
        count = 0
        while self.at("'"):
            self.take()
            count += 1
        return count

    def atom(self):
        kind, text = self.take()
        if kind == "num":
            return self.number(text)
        if kind == "id":
            return self.identifier(text)
        if text == "(":
            self.enter()
            items = [self.sum()]
            while self.at(","):
                self.take()
                items.append(self.sum())
            self.take(")")
            self.depth -= 1
            if len(items) == 1:
                return items[0]
            return tuple(self._value(item) for item in items)
        raise MathError("invalid_expression", f"unexpected {text!r}")

    def identifier(self, name):
        if name in self.context.functions:
            function = self.context.functions[name]
            order = self.primes()
            # a bare y or y'' stands for the function of its own variable
            if self.at("("):
                args = [self._value(a) for a in self.arguments(False)]
            else:
                args = [self.context.function_variables[name]]
            if len(args) != 1:
                raise MathError("invalid_expression", f"function '{name}' takes one argument")
            applied = function(args[0])
            if not order:
                return applied
            # y'(0) is the derivative evaluated at 0, not the derivative of a constant
            if isinstance(args[0], sp.Symbol):
                return sp.Derivative(applied, (args[0], order))
            variable = self.context.function_variables[name]
            return sp.Subs(sp.Derivative(function(variable), (variable, order)), variable, args[0])
        if name in FUNCTIONS:
            function, low, high = FUNCTIONS[name]
            if not self.at("("):
                raise MathError("invalid_expression", f"function '{name}' needs arguments")
            self.enter()
            args = self.arguments(name in _TUPLE_FUNCTIONS)
            self.depth -= 1
            if not low <= len(args) <= high:
                raise MathError("invalid_expression", f"'{name}' takes {low} to {high} arguments")
            if name not in _TUPLE_FUNCTIONS:
                args = [self._value(a) for a in args]
            return self._call(name, function, args)
        if name in CONSTANTS and name not in self.context.declared:
            return CONSTANTS[name]
        leibniz = self.leibniz(name)
        if leibniz is not None:
            return leibniz
        if self.at("(") and len(name) > 1:
            raise MathError("unknown_function", f"'{name[:40]}' is not a supported function")
        return self.context.symbol(name)

    def leibniz(self, name):
        """dy/dx and d2y/dx2, for a function the request declared."""
        match = re.fullmatch(r"d([2-9]?)([A-Za-z]\w*)", name)
        if match is None or match.group(2) not in self.context.functions or not self.at("/"):
            return None
        order, function = match.group(1), match.group(2)
        variable = self.context.function_variables[function]
        following = self.tokens[self.position + 1] if self.position + 1 < len(self.tokens) else (None, None)
        if following != ("id", f"d{variable.name}{order}"):
            return None
        self.position += 2
        return sp.Derivative(self.context.functions[function](variable), (variable, int(order or 1)))

    @staticmethod
    def _call(name, function, args):
        try:
            return function(*args)
        except MathError:
            raise
        except (TypeError, ValueError, ArithmeticError, NotImplementedError) as error:
            raise MathError("invalid_expression", f"cannot evaluate {name}: {str(error)[:200]}")


def parse_expression(source, context):
    parser = _Parser(source, context)
    value = parser._value(parser.sum())
    parser.finish()
    return value


def parse_relation(source, context):
    """An equation or inequality. A bare expression means expression = 0."""
    parser = _Parser(source, context)
    left = parser._value(parser.sum())
    kind, text = parser.peek()
    if kind is None:
        return sp.Eq(left, 0, evaluate=False)
    if text not in RELATIONS:
        raise MathError("invalid_expression", f"unexpected {text!r}")
    parser.take()
    right = parser._value(parser.sum())
    parser.finish()
    if text in ("=", "=="):
        return sp.Eq(left, right, evaluate=False)
    return RELATIONS[text](left, right)
