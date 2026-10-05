# ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
# Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Checks a structured call against the request it was written from.

The model turns the user's words into arguments, and sometimes drops a denominator, invents
an initial condition or loses a sample. The helper then computes the wrong problem correctly.
The app passes the user's own text next to the arguments; this module reads the formulas,
numbers, lists and conditions out of that text and compares them with the call before it runs.

Nothing here solves anything, and nothing from the text is executed: formulas go through the
same restricted grammar as the arguments.
"""

import math
import re

MAX_TEXT = 8000
CONSISTENT, INCONSISTENT, UNCERTAIN, NOTE = "consistent", "inconsistent", "uncertain", "note"
_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
          "nine": 9, "ten": 10, "half": 0.5, "fair": 0.5, "twice": 2, "double": 2}

_TRANSLATE = str.maketrans({
    "−": "-", "–": "-", "—": " ", "×": "*", "·": "*", "⋅": "*", "÷": "/",
    "π": " pi ", "∞": " oo ", "’": "'", "′": "'", "²": "^2", "³": "^3", "√": " sqrt",
    " ": " ",
})
_NUMBER = r"-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?"
# function and constant names of the grammar, listed here so that reading a request does not load SymPy
_KNOWN = frozenset("""
sqrt cbrt root exp log ln log2 log10 sin cos tan cot sec csc asin acos atan acot asec acsc arcsin arccos arctan atan2
sinh cosh tanh coth sech csch asinh acosh atanh abs Abs sign floor ceiling ceil re im arg conjugate conj factorial
binomial gamma beta zeta erf erfc LambertW Heaviside DiracDelta Min Max min max gcd lcm Mod diff Derivative integrate
Integral summation Sum product Product limit pi E e I oo inf infinity Infinity EulerGamma GoldenRatio Catalan
""".split())
_CONSTANT_NAMES = frozenset("pi E e I oo inf infinity Infinity EulerGamma GoldenRatio Catalan".split())
# y' is read as the plain name yDrv1, so a derivative compares like any other symbol
_D = "Drv"


def _float(text):
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _same(a, b):
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def _within(value, numbers):
    return any(_same(value, n) for n in numbers)


# reading the request

_TEX = re.compile(r"\\[A-Za-z]+|\\[\[\]()]|[_^]\{|\$")
_TEX_HEAD = re.compile(r"\\[dt]?frac(?![A-Za-z])|\\sqrt(?![A-Za-z])|\\(?:int|sum|prod|lim)(?![A-Za-z])|[_^](?=\{)")
_TEX_TOKEN = re.compile(r"-?(?:oo|pi|\d+(?:\.\d+)?|[A-Za-z])")
_TEX_WORDS = {"int": "integral", "sum": "summation", "prod": "product"}
_TEX_SYMBOLS = {"cdot": "*", "times": "*", "div": "/", "infty": "oo", "pi": "pi", "to": " approaches ",
                "rightarrow": " approaches "}


def _tex_group(text, start):
    """(content, index after it) of the {...} group or the single token at start, or None."""
    if start < len(text) and text[start] == "{":
        depth = 0
        for index in range(start, len(text)):
            depth += (text[index] == "{") - (text[index] == "}")
            if depth == 0:
                return text[start + 1:index], index + 1
        return None
    match = _TEX_TOKEN.match(text, start)
    return (match.group(0), match.end()) if match else None


def _tex_structure(text):
    out, index = [], 0
    while index < len(text):
        match = _TEX_HEAD.match(text, index)
        if match is None:
            out.append(text[index])
            index += 1
            continue
        head, index = match.group(0).lstrip("\\"), match.end()
        if head in ("_", "^"):
            inner, index = _tex_group(text, index) or ("", index + 1)
            inner = _tex_structure(inner)
            if head == "^":
                out.append(f"^({inner})")
            elif re.fullmatch(r"\w+", inner.strip()):
                out.append("_" + inner.strip())
        elif head.endswith("frac"):
            top = _tex_group(text, len(text) - len(text[index:].lstrip()))
            bottom = top and _tex_group(text, len(text) - len(text[top[1]:].lstrip()))
            if not bottom:
                out.append(" ")
                continue
            out.append(f"(({_tex_structure(top[0])})/({_tex_structure(bottom[0])}))")
            index = bottom[1]
        elif head == "sqrt":
            inner = _tex_group(text, index) if text[index:index + 1] == "{" else None
            if inner is None:
                out.append(" sqrt")
                continue
            out.append(f"sqrt({_tex_structure(inner[0])})")
            index = inner[1]
        else:
            ends = {}
            while text[index:index + 1] in ("_", "^") and text[index] not in ends:
                group = _tex_group(text, index + 1)
                if group is None:
                    break
                ends[text[index]] = re.sub(r"\s+", "", _tex_structure(group[0])) if head != "lim" else _tex_structure(group[0])
                index = group[1]
            if head == "lim":
                out.append(" limit as " + " ".join(ends.get("_", "").split()) + " of ")
            elif "_" in ends and "^" in ends:
                out.append(f" {_TEX_WORDS[head]} from {ends['_']} to {ends['^']} of ")
            else:
                out.append(f" {_TEX_WORDS[head]} of ")
    return "".join(out)


def _tex(text):
    """LaTeX markup as the plain math text the rest of this module reads. Not a TeX parser:
    fractions, roots, powers, infinity and the limits of an integral, sum or product."""
    if not _TEX.search(text):
        return text
    text = re.sub(r"\\[\[\]()]|\$|\\[{}]", " ", text)
    text = re.sub(r"\\(?:mathrm|operatorname|text|textrm|mathit|mbox)\s*\{d\}\s*([A-Za-z])(?![A-Za-z])", r" d\1 ", text)
    text = re.sub(r"\\(?:mathrm|operatorname|text|textrm|mathit|mathbf|mbox)\s*\{([^{}]*)\}", r" \1 ", text)
    text = re.sub(r"\\[,;:!]\s*d\s*([A-Za-z])(?![A-Za-z])", r" d\1 ", text)
    text = re.sub(r"\\[,;:! ]|\\q?quad(?![A-Za-z])|\\(?:left|right|displaystyle|big|Big)(?![A-Za-z])", " ", text)

    def symbol(match):
        value = _TEX_SYMBOLS.get(match.group(1))
        if value is None:
            return match.group(0)
        before, after = match.string[:match.start()][-1:], match.string[match.end():][:1]
        return (" " if before.isalnum() else "") + value + (" " if after.isalnum() else "")
    text = re.sub(r"\\([A-Za-z]+)", symbol, text)
    text = _tex_structure(text).replace("{", "(").replace("}", ")")
    return re.sub(r"\\([A-Za-z]+)", lambda m: m.group(1) if m.group(1) in _KNOWN else f" {m.group(1)} ", text)


_PRECISION = (r"(?:decimal\s+(?:places?|digits?)|decimals?|significant\s+(?:figures?|digits?)|correct\s+digits?"
              r"|cifras\s+(?:decimales|significativas)|decimales|d[ií]gitos\s+(?:decimales|significativos)"
              r"|lugares\s+decimales)")
_PRECISION_AFTER = r"(?:to|with|within|at|of|con|a|en|hasta)\s+\d+\s+(?:digits?|figures?|places?|cifras|d[ií]gitos)"
_TOLERANCE = r"(?:tolerance|accuracy|precision|tolerancia|precisi[oó]n|exactitud)"


def _format_numbers(text):
    """Cuts out the numbers that say how to answer rather than what to compute: the markers of a
    numbered list and a count of digits go; a tolerance is returned apart, since some requests use
    that word for a quantity of the problem."""
    markers = list(re.finditer(r"(?m)^[ \t]*\(?(\d{1,2})[.)][ \t]+(?=\S)", text))
    if len(markers) >= 2 and all(int(m.group(1)) == index + 1 for index, m in enumerate(markers)):
        for match in reversed(markers):
            text = text[:match.start()] + " ; " + text[match.end():]
    text = re.sub(rf"\b\d+\s+{_PRECISION}\b", " ", text, flags=re.I)
    text = re.sub(rf"\b{_PRECISION_AFTER}\b", " ", text, flags=re.I)
    meta = []

    def take(match):
        meta.append(float(match.group(1)))
        return " ; "
    text = re.sub(rf"(?<![\w.])({_NUMBER})\s+(?:absolute\s+|relative\s+)?{_TOLERANCE}\b", take, text, flags=re.I)
    text = re.sub(rf"\b{_TOLERANCE}\s*(?:of|de|=|:)?\s*({_NUMBER})", take, text, flags=re.I)
    return text, meta


class Source:
    """What a piece of text states: formulas, numbers, lists, matrices and conditions."""

    def __init__(self, text):
        text, self.meta = _format_numbers(_tex(str(text or "")[:MAX_TEXT].translate(_TRANSLATE)))
        self.text = text
        self.matrices, self.lists, self.named, self.conditions, self.points = [], [], {}, {}, []
        self.hertz, self.seconds = [], []
        self.infinity = bool(re.search(r"\b(?:infinity|infinite|inf|oo|without bound|unbounded|improper|real line"
                                       r"|all real|grows? large|arbitrarily large)\b", text, re.I))
        self.mentions_condition = bool(re.search(r"\binitial|\bcondition|\bpassing through|\bstarting (?:at|from|with)",
                                                 text, re.I))
        value = r"(?:[A-Za-z]\s*=\s*)?([^\s,;:?!]+?)"
        self.ranges = self._pairs(rf"\bfrom\s+{value}\s+to\s+{value}(?=[\s,;:?!)]|\.(?:\s|$)|$)", text)
        self.ranges += self._pairs(rf"\b[A-Za-z]\s*=\s*([^\s,;:?!]+?)\s+to\s+{value}(?=[\s,;:?!)]|\.(?:\s|$)|$)", text)
        self.between = self._pairs(rf"\bbetween\s+{value}\s+and\s+{value}(?=[\s,;:?!)]|\.(?:\s|$)|$)", text)
        text = self._matrices(text)
        text = self._lists(text)
        text = self._points(text)
        text = self._conditions(text)
        text = re.sub(r"\b\d+\s*[xX]\s*\d+\b(?=\s+[A-Za-z])", " ", text)
        text = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", r"\1 ", text)
        self.hertz = [float(n) for n in re.findall(rf"({_NUMBER})\s*(?:Hz|hertz)\b", text, re.I)]
        self.hertz += [1000 * float(n) for n in re.findall(rf"({_NUMBER})\s*kHz\b", text, re.I)]
        self.seconds = [float(n) for n in re.findall(rf"({_NUMBER})\s*(?:seconds?|secs?)\b", text, re.I)]
        self.formulas, self.assignments = [], {}
        prose = self._formulas(text)
        self.standalone = [float(n) for n in re.findall(rf"(?<![\w.]){_NUMBER}", prose)]
        self.standalone += [v for point, v in self.conditions.values()] + [p for p, _ in self.conditions.values()]
        self.standalone += list(self.assignments.values()) + self.points
        self.standalone += [float(v) for w, v in _WORDS.items() if re.search(rf"\b{w}\b", text, re.I)]
        if re.search(r"\bstandard normal\b", text, re.I):
            self.standalone += [0.0, 1.0]
        self.constants = [name for name in ("pi", "e") if re.search(rf"(?<![A-Za-z]){name}(?![A-Za-z])", text)]
        self.inside = []     # numbers that only occur inside a formula of the problem
        for formula in self.formulas:
            self.inside += [float(n) for n in re.findall(rf"(?<![\w.])(?:\d+\.\d*|\.\d+|\d+)", formula)]

    @staticmethod
    def _pairs(pattern, text):
        found = []
        for first, second in re.findall(pattern, text, re.I):
            pair = (_bound(first.rstrip(".")), _bound(second.rstrip(".")))
            if None not in pair and pair[0] != pair[1]:
                found.append(pair)
        return found

    def _matrices(self, text):
        def take(match):
            rows = [[e.strip() for e in row.split(",")] for row in re.findall(r"\[([^\[\]]*)\]", match.group(0))]
            if rows and all(rows) and len({len(r) for r in rows}) == 1:
                self.matrices.append(rows)
                return " ; "
            return match.group(0)
        return re.sub(r"\[\s*\[[^\[\]]*\](?:\s*,\s*\[[^\[\]]*\])*\s*\]", take, text)

    def _lists(self, text):
        def take(match):
            values = [_float(e.strip()) for e in match.group(2).split(",")] if match.group(2).strip() else []
            if len(values) < 2 or any(v is None for v in values):
                return match.group(0)
            self.lists.append(values)
            if match.group(1):
                self.named[match.group(1)] = values
            return " ; "
        return re.sub(r"(?:\b([A-Za-z]\w*)\s*=\s*)?\[([^\[\]]*)\]", take, text)

    def _points(self, text):
        pair = rf"\(\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\)"

        def take(match):
            found = re.findall(pair, match.group(0))
            self.lists.append([float(a) for a, _ in found])
            self.lists.append([float(b) for _, b in found])
            self.named.setdefault("x", self.lists[-2])
            self.named.setdefault("y", self.lists[-1])
            return " ; "
        return re.sub(rf"{pair}(?:\s*(?:,|and|,\s*and)?\s*{pair})+", take, text)

    def _conditions(self, text):
        def take(match):
            name, primes, point, value = match.groups()
            self.conditions[(name, len(primes))] = (float(point), float(value))
            return " ; "
        text = re.sub(rf"\b([A-Za-z]\w*)\s*('*)\s*\(\s*({_NUMBER})\s*\)\s*=\s*({_NUMBER})(?![\w.*/^(])", take, text)

        def point(match):
            self.points.append(float(match.group(1)))
            return " ; "
        return re.sub(rf"(?<![A-Za-z])[A-Za-z]'*\(\s*({_NUMBER})\s*\)(?!\s*=)", point, text)

    def _formulas(self, text):
        """Cuts the maximal runs of math out of a sentence. Returns what is left."""
        known = _KNOWN
        tokens = re.findall(rf"\s+|(?:\d+\.\d+|\.\d+|\d+)(?:[eE][+-]?\d+)?|[A-Za-z_]\w*|\*\*|[-+*/^=()'!<>,]|.", text)

        def neighbours(index):
            before = next((t for t in reversed(tokens[:index]) if not t.isspace()), "")
            after = next((t for t in tokens[index + 1:] if not t.isspace()), "")
            return before, after

        def mathy(index):
            token = tokens[index]
            if re.fullmatch(r"(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?", token) or token in ("**", "+", "-", "*", "/", "^", "=", "(", ")",
                                                                              "'", "!", "<", ">"):
                return True
            if not re.fullmatch(r"[A-Za-z_]\w*", token):
                return False
            before, after = neighbours(index)
            if token not in known and re.fullmatch(r"[A-Za-z]{2,3}", token) and index > 0 and tokens[index - 1][-1:].isdigit():
                tokens[index] = "*".join(token)
                return True
            if token in known:
                if token in _CONSTANT_NAMES:
                    # "e" and "I" are words as often as constants
                    return len(token) > 1 or (before != "" and before in "+-*/^=(") or (after != "" and after in "+-*/^=)")
                # "integrate x^2" has a verb in it, not a function
                return after == "("
            if re.fullmatch(r"d[2-9]?[A-Za-z]\w?", token) and after == "/":
                return True
            if re.fullmatch(r"d[A-Za-z][2-9]?", token) and before == "/":
                return True
            if re.fullmatch(r"[A-Za-z]\d?", token):
                if token in ("a", "A", "I"):
                    glued = index > 0 and tokens[index - 1][-1:].isdigit()
                    return glued or (before != "" and before in "+-*/^=(") or (after != "" and after in "+-*/^=)'")
                return True
            return False

        spans, current, depth = [], [], 0
        for index, token in enumerate(tokens):
            if token.isspace():
                if current:
                    current.append(" ")
                continue
            if token == "," and depth > 0 and current:
                current.append(token)
                continue
            if mathy(index):
                depth += token == "("
                depth -= token == ")" and depth > 0
                current.append(tokens[index])
            else:
                spans.append(current)
                spans.append(None)
                current, depth = [], 0
        spans.append(current)

        prose = []
        for span in spans:
            if span is None:
                prose.append(" ; ")
                continue
            raw = "".join(span).strip()
            piece = _trim(raw)
            if piece and _is_formula(piece):
                assigned = re.fullmatch(rf"([A-Za-z]\w*)\s*=\s*({_NUMBER})", piece)
                if assigned:
                    self.assignments[assigned.group(1)] = float(assigned.group(2))
                else:
                    self.formulas.append(piece)
                    continue
            prose.append(" " + raw + " ")
        return "".join(prose)


def _unmatched(text):
    """Index of the first parenthesis without a partner, or None."""
    opened = []
    for index, character in enumerate(text):
        if character == "(":
            opened.append(index)
        elif character == ")":
            if not opened:
                return index
            opened.pop()
    return opened[0] if opened else None


def _trim(text):
    """A span without the operators left at its ends, cut at a parenthesis that has no partner."""
    text = text.strip()
    while True:
        before = text
        text = re.sub(r"^[*/^=)'!<>,\s]+", "", text)
        text = re.sub(r"[-+*/^=(<>,\s]+$", "", text)
        index = _unmatched(text)
        if index is not None:
            left, right = text[:index], text[index + 1:]
            text = left if len(left) >= len(right) else right
        if text == before:
            return text


def _is_formula(text):
    if not re.search(r"[-+*/^=]|\w\(|'", text):
        return False
    # a signed number is a number, not a formula
    return not re.fullmatch(rf"[-+]?{_NUMBER}", text.strip())


# comparing formulas

def _normal(text):
    """Derivative notation as plain names, so y'' + y and d2y/dt2 + y read the same."""
    text = str(text).translate(_TRANSLATE)
    text = re.sub(r"\b(?:Derivative|diff)\(\s*([A-Za-z]\w*)(?:\([^()]*\))?\s*,\s*\(?\s*[A-Za-z]\w*\s*,\s*([2-9])\s*\)?\s*\)",
                  lambda m: f"{m.group(1)}{_D}{m.group(2)}", text)
    text = re.sub(r"\b(?:Derivative|diff)\(\s*([A-Za-z]\w*)(?:\([^()]*\))?\s*,\s*[A-Za-z]\w*\s*,\s*[A-Za-z]\w*\s*\)",
                  lambda m: f"{m.group(1)}{_D}2", text)
    text = re.sub(r"\b(?:Derivative|diff)\(\s*([A-Za-z]\w*)(?:\([^()]*\))?\s*,\s*[A-Za-z]\w*\s*\)",
                  lambda m: f"{m.group(1)}{_D}1", text)
    text = re.sub(r"(?<![A-Za-z_])d([2-9]?)([A-Za-z]\w*?)/d[A-Za-z]\1(?![A-Za-z0-9_])",
                  lambda m: f"{m.group(2)}{_D}{m.group(1) or 1}", text)
    text = re.sub(r"(?<![A-Za-z_])([A-Za-z]\w*?)('+)", lambda m: f"{m.group(1)}{_D}{len(m.group(2))}", text)
    # y(x) for the unknown of an equation that also carries its derivative
    for name in set(re.findall(rf"\b([A-Za-z]\w*?){_D}\d", text)):
        text = re.sub(rf"\b{re.escape(name)}({_D}\d)?\(\s*[A-Za-z]\s*\)", lambda m: name + (m.group(1) or ""), text)
    return text


_PARSED = {}


def _parse(text):
    """A SymPy relation or expression for a piece of math text, or None when it is not one."""
    if not isinstance(text, (str, int, float)) or isinstance(text, bool):
        return None
    text = text if isinstance(text, str) else repr(text)
    if text in _PARSED:
        return _PARSED[text]
    from . import mathlang
    import sympy as sp
    value = None
    try:
        context = mathlang.Context(None, None, ())
        relation = mathlang.parse_relation(_normal(text), context)
        if isinstance(relation, sp.Eq) and "=" not in text:
            value = relation.lhs
        else:
            value = relation
    except Exception:
        value = None
    if len(_PARSED) > 512:
        _PARSED.clear()
    _PARSED[text] = value
    return value


_POINTS = (0.3719538, 1.1340271, 1.7093827, 0.5946213, 2.2871639, 0.8312957, 1.4706183, 1.9257341)


def _samples(expressions):
    """The expressions evaluated at fixed points, or None when they cannot be."""
    import sympy as sp
    symbols = sorted(set().union(*[e.free_symbols for e in expressions]), key=lambda s: s.name)
    rows = []
    for shift in range(len(_POINTS)):
        values = {s: sp.Float(_POINTS[(i + shift) % len(_POINTS)] + 0.0117311 * i) for i, s in enumerate(symbols)}
        try:
            row = [complex(e.evalf(20, subs=values)) for e in expressions]
        except Exception:
            continue
        if all(math.isfinite(v.real) and math.isfinite(v.imag) for v in row):
            rows.append(row)
        if len(rows) == 4:
            break
    return rows if len(rows) >= 2 else None


def _close(a, b):
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def _equal(a, b):
    rows = _samples([a, b])
    return rows is not None and all(_close(x, y) for x, y in rows)


def _proportional(a, b):
    """a = k*b for a constant k that is not zero: the two say the same equation."""
    if not (a.free_symbols or b.free_symbols):
        return _equal(a, b)
    rows = _samples([a, b])
    if rows is None:
        return False
    ratios = []
    for x, y in rows:
        if abs(y) < 1e-12 and abs(x) < 1e-12:
            continue
        if abs(y) < 1e-12 or abs(x) < 1e-12:
            return False
        ratios.append(x / y)
    return all(_close(r, ratios[0]) for r in ratios)


def _sides(value):
    import sympy as sp
    return (value.lhs, value.rhs) if isinstance(value, sp.Eq) else None


def _match(call, given, sides_allowed=False):
    """How a formula of the call stands to one of the request: same, side (one side of its
    equation), part (a piece of it), whole (it contains it), other (it shares its symbols), or None."""
    import sympy as sp
    inequality = lambda value: isinstance(value, sp.Rel) and not isinstance(value, sp.Eq)
    if inequality(call) or inequality(given):
        if type(call) is type(given) and _equal(call.lhs - call.rhs, given.lhs - given.rhs):
            return "same"
        if inequality(call) and not isinstance(given, sp.Rel) and (call.lhs.has(given) or call.rhs.has(given)):
            return "whole"
        return "other" if call.free_symbols & given.free_symbols else None
    c, g = _sides(call), _sides(given)
    if c and g:
        identity = _equal(c[0], c[1])
        if identity and (_equal(c[0], g[0]) and _equal(c[1], g[1]) or _equal(c[0], g[1]) and _equal(c[1], g[0])):
            return "same"
        if not identity and _proportional(c[0] - c[1], g[0] - g[1]):
            return "same"
    elif c:
        if _proportional(c[0] - c[1], given):
            return "same"
        if c[0].is_Symbol and c[0] not in c[1].free_symbols and _equal(c[1], given):
            return "same"
    elif g:
        if _proportional(call, g[0] - g[1]):
            return "same"
        # y = a*x + b names a model: the call may carry its right side alone
        if g[0].is_Symbol and g[0] not in g[1].free_symbols and _equal(call, g[1]):
            return "same"
        if _equal(call, g[0]) or _equal(call, g[1]):
            return "same" if sides_allowed else "side"
    elif _equal(call, given):
        return "same"
    call_parts, given_parts = c or (call,), g or (given,)
    if any(gp.has(cp) for cp in call_parts for gp in given_parts if not cp.is_Atom and gp != cp):
        return "part"
    if not g and not given.is_Symbol and any(cp.has(given) for cp in call_parts if cp != given):
        return "whole"
    shared = call.free_symbols & given.free_symbols
    if shared or (not call.free_symbols and not given.free_symbols and call.atoms(sp.Number) & given.atoms(sp.Number)):
        return "other"
    return None


# reading the call

_FORMULA_KEYS = ("expression", "left", "right")
_FORMULA_LISTS = ("equations", "residuals", "constraints")
_LIST_KEYS = ("values", "x", "y", "other")
_SIGNALS = ("fft", "psd", "lowpass", "highpass", "bandpass", "bandstop", "peaks", "ifft", "convolve", "correlate")
_SYSTEMS = {"solve", "solveset", "nsolve", "dsolve", "solve_ivp", "linear_solve"}
_INFINITE = {"integrate", "limit", "summation", "product"}


# the scalar arguments each operation reads; the rest is filler the helper ignores
_SCALARS = {
    "integrate": ("lower", "upper"), "summation": ("lower", "upper"), "product": ("lower", "upper"),
    "limit": ("point",), "evaluate": ("at",), "interpolate": ("at",),
    "fft": ("sample_rate", "duration"), "psd": ("sample_rate", "duration"), "peaks": ("sample_rate", "duration"),
    "ifft": ("sample_rate",),
    "lowpass": ("sample_rate", "duration", "cutoff"), "highpass": ("sample_rate", "duration", "cutoff"),
    "bandpass": ("sample_rate", "duration", "cutoff"), "bandstop": ("sample_rate", "duration", "cutoff"),
    "ttest": ("mean",), "confidence_interval": ("level",), "percentile": ("q",),
    "distribution": ("at", "p", "between"),
}
_FROM_LIST = ("mean", "cutoff", "sample_rate", "level", "q", "p")


class _Empty:
    free_symbols = frozenset()


_OPPOSITES = (
    (("lowpass", ("low-pass", "lowpass", "low pass")), ("highpass", ("high-pass", "highpass", "high pass"))),
    (("bandpass", ("band-pass", "bandpass", "band pass")), ("bandstop", ("band-stop", "bandstop", "band stop", "notch"))),
    (("minimize", ("minimum", "minimi", "smallest", "lowest")), ("maximize", ("maximum", "maximi", "largest", "highest"))),
    (("integrate", ("integral", "integrate", "antiderivative", "area under")), ("summation", ("sum of", "summation"))),
    (("integrate", ("integral", "integrate", "antiderivative", "area under")),
     ("differentiate", ("derivative", "differentiate"))),
    (("fft", ("fft", "fourier transform", "spectrum", "frequenc")), ("ifft", ("inverse fft", "ifft", "inverse fourier"))),
)


def _constant(value):
    """A number for a scalar argument, with oo for an infinite one, or None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return math.copysign(math.inf, value) if abs(value) >= 1e15 else float(value)
    if not isinstance(value, str):
        return None
    number = _float(value)
    if number is not None:
        return math.copysign(math.inf, number) if abs(number) >= 1e15 else number
    parsed = _parse(value)
    if parsed is None or _sides(parsed) or parsed.free_symbols:
        return None
    import sympy as sp
    if parsed in (sp.oo, -sp.oo):
        return math.inf if parsed == sp.oo else -math.inf
    try:
        number = complex(parsed.evalf(17))
    except Exception:
        return None
    return number.real if abs(number.imag) < 1e-12 and math.isfinite(number.real) else None


def _bound(value):
    """An end of a range: a number, or the name of a symbolic end such as n."""
    number = _constant(value)
    if number is not None:
        return number
    text = str(value).strip()
    return text if re.fullmatch(r"[A-Za-z]\w?", text) else None


def _numbers_in(node, out):
    if isinstance(node, bool):
        return
    if isinstance(node, (int, float)):
        out.append(float(node))
    elif isinstance(node, str):
        out += [float(n) for n in re.findall(r"(?<![\w.])(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?", node)]
    elif isinstance(node, dict):
        for key, value in node.items():
            _numbers_in(key, out)
            _numbers_in(value, out)
    elif isinstance(node, list):
        for value in node:
            _numbers_in(value, out)


def _flat(value):
    """A list argument as floats, or None when it is not a plain list of numbers."""
    if not isinstance(value, list) or not value:
        return None
    numbers = [_constant(v) for v in value]
    return None if any(n is None for n in numbers) else numbers


def _condition_key(key):
    key = str(key).strip()
    match = re.fullmatch(r"d([2-9]?)([A-Za-z]\w*?)/d[A-Za-z]\w*?\1?(?:\([^)]*\))?", key)
    if match:
        return match.group(2), int(match.group(1) or 1)
    match = re.fullmatch(r"([A-Za-z]\w*?)\s*('*)\s*(?:\([^)]*\))?", key)
    return (match.group(1), len(match.group(2))) if match else None


class _Check:
    def __init__(self, request, context, operation, arguments):
        self.request, self.context = request, context
        self.operation, self.arguments = operation, arguments
        self.findings = []

    def no(self, text):
        self.findings.append((INCONSISTENT, text))

    def unsure(self, text):
        self.findings.append((UNCERTAIN, text))

    # formulas

    def call_formulas(self):
        args, found = self.arguments, []
        sampled = self.operation in _SIGNALS and args.get("values") is not None
        if self.operation != "interpolate" and not sampled:
            for key in _FORMULA_KEYS:
                if isinstance(args.get(key), str) and args[key].strip():
                    found.append((key, args[key]))
        for key in _FORMULA_LISTS:
            values = args.get(key)
            for value in values if isinstance(values, list) else [values] if isinstance(values, str) else []:
                if isinstance(value, str) and value.strip():
                    found.append((key, value))
        for key in ("solution", "substitutions"):
            if isinstance(args.get(key), dict):
                for name, value in args[key].items():
                    if _constant(value) is None and isinstance(value, str):
                        found.append((key, f"{name} = {value}"))
        return found

    def formulas(self):
        given = [(text, _parse(text)) for text in self.request.formulas]
        given = [(text, value) for text, value in given if value is not None]
        earlier = [value for value in (_parse(text) for text in self.context.formulas) if value is not None]
        used = set()
        # rewriting one side of an identity is a way of checking it
        sides = self.operation in ("equivalent", "solution", "simplify", "expand", "factor", "cancel", "together", "apart")
        calls = self.call_formulas()
        for key, text in calls:
            if not _is_formula(text):
                continue
            call = _parse(text)
            if call is None:
                self.unsure(f"'{key}' is {_show(text)}, which could not be compared with the request")
                continue
            if _sides(call) and _equal(*_sides(call)):
                continue
            relations = [(_match(call, value, sides), index) for index, (_, value) in enumerate(given)]
            same = [index for relation, index in relations if relation == "same"]
            if same:
                used.update(same)
                continue
            if any(_match(call, value, True) == "same" for value in earlier):
                continue
            if not given:
                verdict = self.composed(call)
                if verdict is False:
                    self.no(f"'{key}' is {_show(text)}, which does not combine the quantities the way the request says")
                elif verdict is None:
                    self.unsure(f"'{key}' is {_show(text)}; the request does not spell that formula out")
                continue
            whole = [index for relation, index in relations if relation == "whole"]
            if whole:
                # built from pieces of the request by its words: "the difference between A and B"
                used.update(whole)
                extra, inside = [], []
                _numbers_in(text, extra)
                for index in whole:
                    _numbers_in(given[index][0], inside)
                for number in inside:
                    if _within(number, extra):
                        extra.remove(next(n for n in extra if _same(n, number)))
                if all(_within(number, self.grounded()) for number in extra):
                    verdict = _combination(call, self.request.text, [given[index][0] for index in whole] + extra)
                    if verdict is True:
                        continue
                    if verdict is False:
                        self.no(f"'{key}' is {_show(text)}, which does not combine the quantities the way the request says")
                    else:
                        self.unsure(f"'{key}' is {_show(text)}, which the request does not write as one formula")
                else:
                    self.no(f"'{key}' is {_show(text)}, which differs from {_show(given[whole[0]][0])} in the request")
                continue
            for wanted, verdict in (("side", "is one side only of"), ("part", "is only a part of"), ("other", "differs from")):
                index = next((i for relation, i in relations if relation == wanted), None)
                if index is not None:
                    used.add(index)
                    self.no(f"'{key}' is {_show(text)}, which {verdict} {_show(given[index][0])} in the request")
                    break
            else:
                self.unsure(f"'{key}' is {_show(text)}; the request does not spell that formula out")
        unused = [text for index, (text, value) in enumerate(given)
                  if index not in used and value.free_symbols and not (value.is_Pow or value.is_Symbol)]
        if unused and calls:
            equations = [t for t in unused if "=" in t]
            if equations and self.operation in _SYSTEMS and any(key == "equations" for key, _ in calls):
                self.no(f"the request also has {_show(equations[0])}, which the call leaves out")
            else:
                self.unsure(f"the request also has {_show(unused[0])}, which the call does not use")

    def composed(self, call):
        """A formula with no variables built from numbers and constants the request names, joined the way its
        words say ("the error of approximating e by 2.718"): True, False, or None when that cannot be told."""
        import sympy as sp
        if call.free_symbols or _sides(call):
            return None
        grounded = self.grounded()
        numbers = [abs(float(n)) for n in call.atoms(sp.Number) if n not in (sp.S.One, sp.S.NegativeOne)]
        constants = {"pi": sp.pi, "e": sp.E}
        if not all(_within(n, grounded) for n in numbers):
            return None
        if any(call.has(value) and name not in self.request.constants for name, value in constants.items()):
            return None
        named = [name for name, value in constants.items() if call.has(value)]
        return _combination(call, self.request.text, numbers + named)

    # numbers

    def grounded(self):
        numbers = list(self.request.standalone) + list(self.context.standalone) + list(self.context.inside)
        numbers += self.request.meta + self.context.meta
        for source in (self.request, self.context):
            numbers += [math.pi for name in source.constants if name == "pi"]
            numbers += [math.e for name in source.constants if name == "e"]
            for text in source.formulas:
                value = _constant(text)
                if value is not None:
                    numbers.append(value)
        return numbers

    def scalar(self, name, value, grounded, strict):
        number = _constant(value)
        if number is None or math.isinf(number):
            return
        if _within(number, grounded):
            return
        if name in ("level", "q", "p") and (_within(number * 100, grounded) or _within(number / 100, grounded)):
            return
        if name in _FROM_LIST and any(_within(number, values) for values in self.request.lists):
            self.no(f"'{name}' is {_number(number)}, which the request gives as one value of a list, not on its own")
        elif self.operation in _INFINITE and _within(number, self.request.inside):
            self.no(f"'{name}' is {_number(number)}, which the request only has inside a formula")
        elif strict:
            self.no(f"'{name}' is {_number(number)}, which is not in the request")
        elif self.unused():
            self.no(f"'{name}' is {_number(number)}; the request has {_number(self.unused()[0])}, which the call does not use")
        else:
            self.unsure(f"'{name}' is {_number(number)}, which is not in the request")

    def unused(self):
        """Numbers the request states on their own that appear nowhere in the call."""
        used = []
        _numbers_in(self.arguments, used)
        return [n for n in self.request.standalone
                if not _within(n, used) and not _within(abs(n), used) and not _within(n / 100, used)]

    def ranges(self):
        """A range the request gives as "from A to B" is the range of the call, in that order."""
        args, op = self.arguments, self.operation
        if op in ("integrate", "summation", "product") and args.get("lower") is not None and args.get("upper") is not None:
            pair, stated = (_bound(args["lower"]), _bound(args["upper"])), self.request.ranges + self.request.between
        elif op == "solve_ivp" and isinstance(args.get("interval"), list) and len(args["interval"]) == 2:
            pair, stated = (_bound(args["interval"][0]), _bound(args["interval"][1])), self.request.ranges
        elif op == "root" and isinstance(args.get("bracket"), list) and len(args["bracket"]) == 2:
            pair = tuple(sorted(c for c in map(_constant, args["bracket"]) if c is not None))
            stated = [tuple(sorted(r)) for r in self.request.between + self.request.ranges
                      if all(isinstance(v, float) for v in r)]
        else:
            return
        if None in pair or len(pair) != 2 or not stated:
            return

        def same(a, b):
            if isinstance(a, str) or isinstance(b, str):
                return a == b
            return a == b or (not math.isinf(a) and not math.isinf(b) and _same(a, b))
        if any(same(pair[0], lo) and same(pair[1], hi) for lo, hi in stated):
            return
        lo, hi = stated[0]
        show = lambda v: v if isinstance(v, str) else ("oo" if v > 0 else "-oo") if math.isinf(v) else _number(v)
        if same(pair[0], hi) and same(pair[1], lo):
            self.no(f"the limits are swapped: the request goes from {show(lo)} to {show(hi)}")
        else:
            self.no(f"the call goes from {show(pair[0])} to {show(pair[1])}; the request says {show(lo)} to {show(hi)}")

    def scalars(self):
        args, op, grounded = self.arguments, self.operation, self.grounded()
        for key in _SCALARS.get(op, ()):
            value = args.get(key)
            if key == "duration" and args.get("values") is not None:
                continue
            if key == "at" and op == "evaluate" and not (_parse(args.get("expression")) or _Empty).free_symbols:
                continue
            items = value if isinstance(value, list) else [value]
            if len(items) > 3:
                continue
            for item in items:
                if item is not None:
                    self.scalar(key, item, grounded, strict=key == "mean")
        bounds = args.get("bounds")
        if isinstance(bounds, dict) and op in ("integrate", "minimize", "maximize"):
            for name, limits in bounds.items():
                for item in limits if isinstance(limits, list) else [limits]:
                    if item is not None:
                        self.scalar(f"bounds of {name}", item, grounded, strict=op != "integrate")
        interval = args.get("interval")
        if op == "solve_ivp" and isinstance(interval, list) and len(interval) == 2:
            if (_constant(interval[0]) or 0) != 0:
                self.scalar("interval start", interval[0], grounded, strict=False)
            self.scalar("interval end", interval[1], grounded, strict=False)
        if isinstance(args.get("parameters"), dict) and op in ("distribution", "percentile"):
            for name, value in args["parameters"].items():
                self.scalar(f"parameter {name}", value, grounded, strict=False)

    def infinity(self):
        if self.operation not in _INFINITE:
            return
        args = self.arguments
        limits = [args.get(k) for k in ("lower", "upper", "point")]
        if isinstance(args.get("bounds"), dict):
            limits += [v for pair in args["bounds"].values() if isinstance(pair, list) for v in pair]
        infinite = [v for v in limits if v is not None and (_constant(v) or 0) in (math.inf, -math.inf)]
        if self.request.infinity and not infinite and not self.context.infinity:
            finite = [v for v in limits if v is not None]
            self.no("the request goes to infinity; the call has " + (f"the finite limits {finite}" if finite else "no infinite limit")
                    + '; write an infinite limit as "oo" or "-oo"')
        elif infinite and not self.request.infinity and not self.context.infinity:
            self.unsure("the call has an infinite limit that the request does not mention")

    def units(self):
        args = self.arguments
        if self.operation not in _SIGNALS[:8]:
            return
        given = []
        for key in ("sample_rate", "cutoff"):
            value = args.get(key)
            given += [n for n in (_constant(v) for v in (value if isinstance(value, list) else [value])) if n is not None]
        for rate in self.request.hertz:
            if not _within(rate, given):
                self.no(f"the request gives {_number(rate)} Hz, which the call does not use")

    # data

    def lists(self):
        args = self.arguments
        known = self.request.lists + self.context.lists
        used = []
        candidates = []
        for key in _LIST_KEYS:
            value = args.get(key)
            if isinstance(value, list) and value and all(isinstance(v, list) for v in value):
                rows = [[str(e) for e in row] for row in value]
                if any(_same_matrix(rows, given) for given in self.request.matrices + self.context.matrices):
                    continue
                if len(value) == 1 or all(len(v) == 1 for v in value):
                    candidates.append((key, _flat([e for v in value for e in v])))
                elif key != "x":
                    candidates += [(key, _flat(v)) for v in value]
            else:
                candidates.append((key, _flat(value)))
        if isinstance(args.get("data"), dict):
            candidates += [(f"data {k}", _flat(v)) for k, v in args["data"].items()]
        for key, numbers in candidates:
            if numbers is None or len(numbers) < 2:
                continue
            exact = next((given for given in known if len(given) == len(numbers) and all(map(_same, given, numbers))), None)
            if exact is not None:
                used.append(exact)
                named = self.request.named
                if key in ("x", "y") and key in named and named[key] is not exact and exact in named.values():
                    other = next(n for n, v in named.items() if v is exact)
                    self.no(f"'{key}' carries the values the request gives for {other}")
                continue
            if not self.request.lists:
                if all(_within(n, self.grounded()) for n in numbers):
                    continue
                self.unsure(f"'{key}' has {len(numbers)} values that the request does not list")
                continue
            given = min(self.request.lists, key=lambda g: _distance(g, numbers))
            distance = _distance(given, numbers)
            if distance <= max(2, len(given) // 5):
                used.append(given)
                if len(given) != len(numbers):
                    self.no(f"'{key}' has {len(numbers)} values; the request lists {len(given)}")
                else:
                    index = next(i for i, (a, b) in enumerate(zip(given, numbers)) if not _same(a, b))
                    self.no(f"'{key}' has {_number(numbers[index])} where the request has {_number(given[index])}")
            else:
                self.unsure(f"'{key}' has {len(numbers)} values that are not a list of the request")
        unused = [given for given in self.request.lists if not any(given is u for u in used)]
        if unused and candidates and any(numbers for _, numbers in candidates):
            self.unsure(f"the request also lists {len(unused[0])} values that the call does not use")

    def matrices(self):
        value = self.arguments.get("matrix")
        if not isinstance(value, list) or not value or not all(isinstance(row, list) for row in value):
            return
        rows = [[str(e) for e in row] for row in value]
        known = self.request.matrices + self.context.matrices
        if any(_same_matrix(rows, given) for given in known):
            return
        if not self.request.matrices:
            numbers = []
            _numbers_in(value, numbers)
            if not all(_within(n, self.grounded() + self.request.inside) for n in numbers):
                self.unsure(f"the matrix is {len(rows)} x {len(rows[0])}; the request does not write one out")
            return
        given = self.request.matrices[0]
        if (len(rows), len(rows[0])) != (len(given), len(given[0])):
            self.no(f"the matrix is {len(rows)} x {len(rows[0])}; the request has {len(given)} x {len(given[0])}")
        else:
            i, j = next((i, j) for i in range(len(rows)) for j in range(len(rows[0])) if not _same_entry(rows[i][j], given[i][j]))
            self.no(f"the matrix has {rows[i][j]} at row {i + 1}, column {j + 1}; the request has {given[i][j]}")

    def conditions(self):
        given = self.arguments.get("initial_conditions")
        if given is None:
            given = self.arguments.get("ics")
        if not isinstance(given, dict) or not given:
            return
        stated = {**self.context.conditions, **self.request.conditions}
        grounded = self.grounded()
        for key, value in given.items():
            where, number = _condition_key(key), _constant(value)
            label = f"{str(key).strip()} = {value}"
            if where is None or number is None:
                continue
            if where in stated:
                if not _same(stated[where][1], number):
                    self.no(f"the call starts from {label}; the request says {_number(stated[where][1])}")
            elif stated:
                self.no(f"the call starts from {label}, which the request does not state")
            elif self.operation != "solve_ivp" and not self.request.mentions_condition:
                self.no(f"the call adds the condition {label}; the request has none")
            elif not _within(number, grounded):
                self.no(f"the call starts from {label}, and {_number(number)} is not in the request")

    def leftovers(self):
        missing = self.unused()
        if missing:
            self.findings.append((NOTE, f"the request mentions {_number(missing[0])}, which the call does not use"))

    def named_operation(self):
        """The request names one of two opposite operations and the call runs the other."""
        text = self.request.text.lower()
        for first, second in _OPPOSITES:
            for mine, other in ((first, second), (second, first)):
                if self.operation == mine[0] and any(w in text for w in other[1]) and not any(w in text for w in mine[1]):
                    self.no(f"the request asks for {other[1][0]}; the call runs {self.operation}")

    def series_order(self):
        """"Up to x^4" needs the x^4 term; an order of 4 stops before it."""
        if self.operation != "series":
            return
        wanted = re.search(r"\b(?:up to|through|to)\s+(?:and including\s+)?(?:the\s+)?(?:term\s+)?(?:in\s+)?"
                           r"[A-Za-z]\s*\^\s*(\d+)", self.request.text)
        order = _constant(self.arguments.get("order"))
        if wanted and order is not None and order <= int(wanted.group(1)):
            self.no(f"the request wants the terms up to power {wanted.group(1)}; an order of {_number(order)} stops "
                    f"before that power, so the order has to be {int(wanted.group(1)) + 1}")

    def method(self):
        """A way of interpolating changes the answer, so it has to come from the request."""
        kind = self.arguments.get("kind")
        if self.operation != "interpolate" or not isinstance(kind, str) or kind.strip().lower() == "linear":
            return
        words = {"cubic": ("cubic", "spline"), "pchip": ("pchip", "monoton", "shape-preserving"), "akima": ("akima",)}
        wanted = words.get(kind.strip().lower(), (kind.strip().lower(),))
        text = self.request.text.lower() + " " + self.context.text.lower()
        if not any(word in text for word in wanted + ("smooth",)):
            self.no(f"the call interpolates with {kind}, which the request does not ask for; leave 'kind' out for linear")

    def run(self):
        for step in (self.named_operation, self.series_order, self.method, self.formulas, self.infinity, self.ranges, self.conditions, self.lists, self.matrices, self.units,
                     self.scalars, self.leftovers):
            step()
        severities = {severity for severity, _ in self.findings}
        status = INCONSISTENT if INCONSISTENT in severities else UNCERTAIN if UNCERTAIN in severities else CONSISTENT
        wanted = NOTE if status == CONSISTENT else status
        return status, [text for severity, text in self.findings if severity == wanted][:4]


_COMBINATIONS = (
    ("subtract", r"\b(?:difference|differ|error|how far|minus|deviation|subtract)"),
    ("add", r"\b(?:sum of|plus|total|add)\b"),
    ("multiply", r"\b(?:times|product of|multiplied)\b"),
    ("divide", r"\b(?:ratio|divided|quotient)\b"),
)


def _beside(text, quantities):
    """The sentences of the text that state one of the quantities: formulas and names as written, numbers by value."""
    found = []
    for sentence in re.split(r"(?<=[.;?!])\s+|\n+", text):
        packed = re.sub(r"\s+", "", sentence)
        numbers = [float(n) for n in re.findall(rf"(?<![\w.])(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?", sentence)]
        for quantity in quantities:
            if isinstance(quantity, float):
                here = _within(quantity, numbers)
            elif re.fullmatch(r"[A-Za-z]+", quantity):
                here = bool(re.search(rf"(?<![A-Za-z]){quantity}(?![A-Za-z])", sentence))
            else:
                here = re.sub(r"\s+", "", quantity) in packed
            if here:
                found.append(sentence)
                break
    return " ".join(found)


def _combination(call, text, quantities):
    """Whether the call joins the quantities of the request the way the words next to them say: True,
    False, or None when no single way is named there. A word elsewhere in the request is not evidence."""
    import sympy as sp
    text = _beside(text, quantities)
    named = [kind for kind, pattern in _COMBINATIONS if re.search(pattern, text, re.I)]
    if len(named) != 1:
        return None
    value = call.args[0] if isinstance(call, sp.Abs) else call
    if isinstance(value, sp.Add) and len(value.args) == 2:
        negative = [a.could_extract_minus_sign() for a in value.args]
        found = "subtract" if any(negative) and not all(negative) else "add"
    elif isinstance(value, sp.Mul) and len(value.args) == 2:
        found = "divide" if any(a.is_Pow and a.exp.is_negative for a in value.args) else "multiply"
    else:
        return None
    return found == named[0]


def _distance(given, numbers):
    """How many values have to be dropped, added or changed to turn one list into the other."""
    if abs(len(given) - len(numbers)) > max(2, len(given) // 5):
        return len(given) + len(numbers)
    previous = list(range(len(numbers) + 1))
    for i, a in enumerate(given, 1):
        current = [i]
        for j, b in enumerate(numbers, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (0 if _same(a, b) else 1)))
        previous = current
    return previous[-1]


def _same_entry(a, b):
    x, y = _constant(a), _constant(b)
    if x is not None and y is not None:
        return _same(x, y)
    p, q = _parse(a), _parse(b)
    return p is not None and q is not None and not _sides(p) and not _sides(q) and _equal(p, q)


def _same_matrix(rows, given):
    return len(rows) == len(given) and all(len(r) == len(g) for r, g in zip(rows, given)) and \
        all(_same_entry(a, b) for r, g in zip(rows, given) for a, b in zip(r, g))


def _number(value):
    return f"{value:.12g}"


def _show(text):
    return "`" + " ".join(str(text).split())[:120] + "`"


def check(request, context, operation, arguments):
    """(status, reasons) for a call against the user's request and the earlier text of the conversation.
    For a consistent call the reasons are remarks worth showing, such as a number the call leaves out."""
    try:
        return _Check(Source(request), Source(context), operation, arguments).run()
    except RecursionError:
        return UNCERTAIN, ["the call is too large to compare with the request"]


# what the call says, for the reply

def _canonical(text):
    value = _parse(text)
    if value is None:
        return " ".join(str(text).split())[:200]
    import sympy as sp
    if isinstance(value, sp.Eq):
        shown = f"{sp.sstr(value.lhs)} = {sp.sstr(value.rhs)}"
    elif isinstance(value, sp.Rel):
        shown = f"{sp.sstr(value.lhs)} {value.rel_op} {sp.sstr(value.rhs)}"
    else:
        shown = sp.sstr(value)
    return re.sub(rf"\b([A-Za-z]\w*?){_D}(\d)", lambda m: m.group(1) + "'" * int(m.group(2)), shown)[:200]


def _limit(value):
    number = _constant(value)
    if number is None:
        return " ".join(str(value).split())[:40]
    if math.isinf(number):
        return "+oo" if number > 0 else "-oo"
    return str(value) if isinstance(value, str) else _number(number)


def _data(value):
    if isinstance(value, dict):
        return " ".join(f"{k} {_short(v)}" for k, v in value.items())[:80]
    numbers = _flat(value)
    if numbers is None:
        if isinstance(value, list) and value and all(isinstance(v, list) for v in value):
            return "; ".join(_data(v) for v in value[:4])
        return _short(value)
    if len(numbers) <= 8:
        return f"n={len(numbers)} [" + ", ".join(_number(n) for n in numbers) + "]"
    return (f"n={len(numbers)} [" + ", ".join(_number(n) for n in numbers[:3]) + ", ..., "
            + ", ".join(_number(n) for n in numbers[-2:]) + "]")


def _short(value):
    text = str(value)
    return text if len(text) <= 60 else text[:57] + "..."


def interpret(operation, arguments):
    """Short lines saying what will be computed, in canonical form."""
    args, lines = arguments, []
    try:
        for key in _FORMULA_KEYS:
            if isinstance(args.get(key), (str, int, float)) and not isinstance(args.get(key), bool) and operation != "interpolate":
                lines.append(f"{key}: {_canonical(args[key])}")
        for key in _FORMULA_LISTS:
            values = args.get(key)
            values = values if isinstance(values, list) else [values] if isinstance(values, str) else []
            if values:
                lines.append(f"{key}: " + "; ".join(_canonical(v) for v in values[:8]))
        variables = args.get("variables") or ([args["variable"]] if args.get("variable") else [])
        if variables and isinstance(variables, list):
            lines.append("variable: " + ", ".join(str(v) for v in variables[:8]))
        if args.get("lower") is not None or args.get("upper") is not None:
            low, high = _limit(args.get("lower", "?")), _limit(args.get("upper", "?"))
            lines.append(f"limits: {'(' if low == '-oo' else '['}{low}, {high}{')' if high == '+oo' else ']'}")
        if isinstance(args.get("bounds"), dict):
            lines.append("bounds: " + ", ".join(
                f"{name} in [{', '.join(_limit(v) for v in pair)}]" if isinstance(pair, list) else f"{name} {pair}"
                for name, pair in args["bounds"].items()))
        if args.get("point") is not None:
            lines.append(f"point: {_limit(args['point'])}")
        for key in ("bracket", "interval"):
            if isinstance(args.get(key), list):
                lines.append(f"{key}: [" + ", ".join(_limit(v) for v in args[key]) + "]")
        for key in ("initial_conditions", "initial_guess", "solution", "substitutions", "parameters"):
            if isinstance(args.get(key), dict) and args[key]:
                lines.append(key.replace("_", " ") + ": " + ", ".join(
                    f"{k} = {_canonical(v) if isinstance(v, str) else v}" for k, v in list(args[key].items())[:8]))
            elif isinstance(args.get(key), list) and args[key]:
                lines.append(key.replace("_", " ") + ": " + ", ".join(str(v) for v in args[key][:8]))
        matrix = args.get("matrix")
        if isinstance(matrix, list) and matrix and all(isinstance(row, list) for row in matrix):
            shape = f"{len(matrix)} x {len(matrix[0])}"
            entries = sum(len(row) for row in matrix)
            lines.append(f"matrix: {shape}" + (" " + str(matrix).replace("'", "") if entries <= 16 else f", {entries} entries"))
        elif isinstance(matrix, dict):
            lines.append("matrix: " + _short(matrix))
        for key in _LIST_KEYS + ("at",):
            if args.get(key) is not None and not (key == "at" and not isinstance(args[key], (list, dict))):
                lines.append(f"{key}: {_data(args[key])}")
        for key in ("at", "mean", "q", "p", "level", "sample_rate", "duration", "cutoff", "kind", "name", "between",
                    "order", "method", "domain"):
            value = args.get(key)
            if value is not None and not isinstance(value, (list, dict)) or key in ("cutoff", "between") and isinstance(value, list):
                lines.append(f"{key}: {value}")
    except Exception:
        pass
    return [line[:240] for line in lines][:14]
