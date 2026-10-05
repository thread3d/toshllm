# ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
# Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The mathematical operations behind the tools. Every input goes through mathlang."""

import re

import mpmath
import sympy as sp
from sympy.printing.str import StrPrinter

from .mathlang import Context, MathError, check_identifier, parse_expression, parse_relation

MAX_FIELD_CHARS = 6000
MAX_ITEMS = 64
MAX_MATRIX_SIDE = 16
MAX_EQUATIONS = 16
MAX_PRECISION = 1000
DEFAULT_PRECISION = 15
MAX_QUADRATURE_DIGITS = 50
QUADRATURE_TOLERANCE = 8

# remarks gathered while a request is parsed, before its reply exists
_notes = []


class _Printer(StrPrinter):
    # "a = b" reads back through mathlang, "Eq(a, b)" does not
    def _print_Equality(self, expr):
        return f"{self._print(expr.lhs)} = {self._print(expr.rhs)}"


_printer = _Printer()


class Reply:
    def __init__(self, operation):
        self.fields = {"success": True, "operation": operation}
        self.warnings = []

    def text(self, value):
        text = value if isinstance(value, str) else _printer.doprint(value)
        if len(text) > MAX_FIELD_CHARS:
            self.fields["truncated"] = True
            if "result truncated" not in " ".join(self.warnings):
                self.warnings.append(f"result truncated to {MAX_FIELD_CHARS} characters per field")
            return text[:MAX_FIELD_CHARS]
        return text

    def items(self, values):
        values = list(values)
        if len(values) > MAX_ITEMS:
            self.fields["truncated"] = True
            self.warnings.append(f"only the first {MAX_ITEMS} of {len(values)} results are listed")
        return values[:MAX_ITEMS]

    def value(self, value, precision, numeric=True):
        """The main result: exact text, LaTeX, and a decimal value when it is a number."""
        if value is sp.nan:
            raise MathError("no_result", "SymPy returned an undefined value (nan); "
                            "the quantity may diverge or not exist")
        self.fields["exact"] = self.text(value)
        try:
            latex = sp.latex(value)
            if len(latex) <= MAX_FIELD_CHARS:
                self.fields["latex"] = latex
        except Exception:
            pass
        if numeric:
            decimal = _decimal(value, precision)
            if decimal is not None:
                self.fields["numeric"] = self.text(decimal)

    def done(self, **extra):
        self.fields.update(extra)
        self.fields["warnings"] = self.warnings + _notes
        return self.fields

    def open(self, unevaluated, what):
        """A valid request SymPy could not close: says so instead of passing the input back as a result."""
        self.fields.update(success=False, exact=None, unevaluated=self.text(unevaluated), timed_out=False,
                           error={"code": "no_closed_form",
                                  "message": f"SymPy found no closed form for this {what}."})
        return self.done()


def _decimal(value, precision):
    if not isinstance(value, sp.Expr) or value.free_symbols or value.is_Integer:
        return None
    if value.has(sp.oo, sp.zoo, sp.nan) or all(part.is_Integer for part in value.as_real_imag()):
        return None
    try:
        number = sp.N(value, precision)
    except Exception:
        return None
    return _printer.doprint(number) if number.is_number and not number.has(sp.Integral, sp.Sum) else None


def _take(args, name, kind, default=None, required=False):
    value = args.get(name)
    if value is None:
        if required:
            raise MathError("invalid_arguments", f"'{name}' is required")
        return default
    if kind == "math":
        # models send small numbers as JSON numbers as often as strings
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise MathError("invalid_arguments", f"'{name}' must be a string")
        return value if isinstance(value, str) else repr(value)
    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise MathError("invalid_arguments", f"'{name}' must be an integer")
        return value
    if kind == "list":
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            value = [value]
        if not isinstance(value, list) or not value:
            raise MathError("invalid_arguments", f"'{name}' must be a non-empty list")
        if len(value) > MAX_EQUATIONS:
            raise MathError("input_too_large", f"'{name}' has more than {MAX_EQUATIONS} entries")
        return [_take({name: item}, name, "math") for item in value]
    if kind == "dict":
        if not isinstance(value, dict) or len(value) > MAX_EQUATIONS:
            raise MathError("invalid_arguments", f"'{name}' must be an object with at most {MAX_EQUATIONS} entries")
        return {key: _take({name: item}, name, "math") for key, item in value.items()}
    raise AssertionError(kind)


def _precision(args):
    digits = _take(args, "precision", "int", DEFAULT_PRECISION)
    if not 1 <= digits <= MAX_PRECISION:
        raise MathError("invalid_arguments", f"'precision' must be between 1 and {MAX_PRECISION}")
    return digits


def _declared(args):
    """Names the request itself calls variables."""
    names = []
    for key in ("variable", "transform_variable"):
        if isinstance(args.get(key), str):
            names.append(args[key].strip())
    if isinstance(args.get("variables"), list):
        names += [name.strip() for name in args["variables"] if isinstance(name, str)]
    for key in ("assumptions", "substitutions", "solution"):
        if isinstance(args.get(key), dict):
            names += [name.strip() for name in args[key] if isinstance(name, str)]
    return names


def _context(args, functions=None):
    assumptions = args.get("assumptions")
    if assumptions is not None and not isinstance(assumptions, dict):
        raise MathError("invalid_arguments", "'assumptions' must be an object")
    context = Context(assumptions, functions, _declared(args))
    if context.declared:
        _notes.append("e is a variable in this request; write E or exp(1) for Euler's number")
    return context


def _expression(args, context, name="expression"):
    expr = parse_expression(_take(args, name, "math", required=True), context)
    for key, value in (_take(args, "substitutions", "dict") or {}).items():
        expr = expr.subs(context.symbol(key), parse_expression(value, context))
    return expr


def _variable(args, context, expr, name="variable"):
    given = _take(args, name, "math")
    if given is not None:
        return context.symbol(given.strip())
    free = sorted(expr.free_symbols, key=lambda s: s.name)
    if len(free) != 1:
        raise MathError("invalid_arguments", f"'{name}' is required when the expression has {len(free)} variables")
    return free[0]


def _is_open(value, *kinds):
    return isinstance(value, sp.Basic) and value.has(*kinds)


def _quadrature(expr, variable, lower, upper, digits):
    """A definite integral by tanh-sinh quadrature: (value, error estimate, good digits).

    None unless two different subdivisions of the interval agree, which is what a
    singularity inside it breaks.
    """
    work = min(digits, MAX_QUADRATURE_DIGITS) + 15
    with mpmath.workdps(work):
        def integrand(point):
            value = expr.evalf(work, subs={variable: sp.Float(point, work)})
            if not value.is_number or value.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
                raise ValueError("the integrand is not finite")
            return value._to_mpmath(mpmath.mp.prec)

        try:
            a, b = (bound.evalf(work)._to_mpmath(mpmath.mp.prec) for bound in (lower, upper))
            whole, _ = mpmath.quad(integrand, [a, b], error=True)
            cuts = [a + (b - a) * mpmath.mpf(c) for c in ("0.31", "0.5", "0.73")]
            split, estimate = mpmath.quad(integrand, [a, *cuts, b], error=True)
        except (ValueError, ZeroDivisionError, TypeError, OverflowError, AttributeError):
            return None
        estimate = max(abs(whole - split), estimate)
        scale = max(1, abs(split))
        if not mpmath.isfinite(split) or estimate > scale * mpmath.mpf(10) ** -QUADRATURE_TOLERANCE:
            return None
        good = min(digits, MAX_QUADRATURE_DIGITS)
        if estimate:
            good = max(6, min(good, int(-mpmath.log10(estimate / scale)) - 1))
        return sp.sympify(split).evalf(good), mpmath.nstr(estimate, 2), good


def _numeric_integral(reply, expr, variable, bounds, precision, timed_out):
    """Fills the reply with a numerical value when the integral is one a number can stand for."""
    if not bounds or expr.free_symbols - {variable}:
        return False
    if not all(bound.is_number and bound.is_finite and bound.is_real for bound in bounds):
        return False
    outcome = _quadrature(expr, variable, *bounds, precision)
    if outcome is None:
        return False
    value, estimate, digits = outcome
    reply.fields.update(success=True, exact=None, unevaluated=reply.text(sp.Integral(expr, (variable, *bounds))),
                        numeric=reply.text(value), method="numerical_integration", error_estimate=estimate,
                        precision=digits, timed_out_symbolic=timed_out)
    reply.fields.pop("error", None)
    reply.fields.pop("timed_out", None)
    reply.warnings.append("no closed form was obtained; the value is a numerical approximation, not an exact result")
    return True


# expression tool

def _rewrite(function):
    def run(args, reply):
        context = _context(args)
        reply.value(function(_expression(args, context)), _precision(args))
        return reply.done()
    return run


def _apart(args, reply):
    context = _context(args)
    expr = _expression(args, context)
    given = _take(args, "variable", "math")
    result = sp.apart(expr, context.symbol(given)) if given else sp.apart(expr)
    reply.value(result, _precision(args))
    return reply.done()


def _evaluate(args, reply):
    context = _context(args)
    expr = _expression(args, context)
    precision = _precision(args)
    reply.value(sp.nsimplify(expr) if expr.is_Rational else expr, precision, numeric=False)
    reply.fields["numeric"] = reply.text(sp.N(expr, precision))
    if expr.free_symbols:
        reply.warnings.append("the expression still has variables; give them values in 'substitutions'")
    return reply.done()


def _differentiate(args, reply):
    context = _context(args)
    expr = _expression(args, context)
    order = _take(args, "order", "int", 1)
    if not 1 <= order <= 20:
        raise MathError("invalid_arguments", "'order' must be between 1 and 20")
    reply.value(sp.diff(expr, _variable(args, context, expr), order), _precision(args))
    return reply.done()


def _bounds(args, context):
    lower, upper = _take(args, "lower", "math"), _take(args, "upper", "math")
    if (lower is None) != (upper is None):
        raise MathError("invalid_arguments", "'lower' and 'upper' must be given together")
    if lower is None:
        return None
    return parse_expression(lower, context), parse_expression(upper, context)


def _integrate(args, reply):
    context = _context(args)
    expr = _expression(args, context)
    variable = _variable(args, context, expr)
    bounds = _bounds(args, context)
    precision = _precision(args)
    result = sp.integrate(expr, (variable, *bounds) if bounds else variable)
    if _is_open(result, sp.Integral):
        if _numeric_integral(reply, expr, variable, bounds, precision, False):
            return reply.done()
        return reply.open(result, "integral")
    reply.value(result, precision)
    if not bounds:
        reply.fields["note"] = "antiderivative, constant of integration omitted"
    return reply.done()


def _ranged(function, kinds, what):
    def run(args, reply):
        context = _context(args)
        expr = _expression(args, context)
        variable = _variable(args, context, expr)
        bounds = _bounds(args, context)
        if bounds is None:
            raise MathError("invalid_arguments", "'lower' and 'upper' are required")
        result = function(expr, (variable, *bounds))
        if _is_open(result, *kinds):
            return reply.open(result, what)
        if result.has(sp.RisingFactorial, sp.FallingFactorial):
            # products come back as ratios of factorials that usually collapse, e.g. to n + 1
            result = sp.simplify(result)
        reply.value(result, _precision(args))
        return reply.done()
    return run


def _limit(args, reply):
    context = _context(args)
    expr = _expression(args, context)
    variable = _variable(args, context, expr)
    point = parse_expression(_take(args, "point", "math", required=True), context)
    direction = args.get("direction") or "both"
    if direction not in ("both", "+", "-"):
        raise MathError("invalid_arguments", "'direction' must be \"both\", \"+\" or \"-\"")
    if direction == "both" and point in (sp.oo, -sp.oo):
        direction = "-" if point == sp.oo else "+"
    try:
        result = sp.limit(expr, variable, point, "+-" if direction == "both" else direction)
    except ValueError as error:
        # SymPy raises when the one-sided limits differ
        raise MathError("no_result", str(error)[:300])
    if _is_open(result, sp.Limit):
        return reply.open(result, "limit")
    reply.value(result, _precision(args))
    return reply.done()


def _ring_series(expr, variable, order):
    """Expansion at 0 through SymPy's ring series, far quicker on composed functions.

    Its result is accepted only if the remainder shrinks like variable**order at two
    points; anything else returns None and the general routine takes over.
    """
    if expr.free_symbols != {variable}:
        return None
    try:
        from sympy.polys.ring_series import rs_series
        raw = rs_series(expr, variable, order).as_expr()
        if not raw.is_polynomial(variable):
            return None
        poly = sp.Poly(raw, variable)
        kept = sum(coefficient * variable**power for (power,), coefficient in poly.terms() if power < order)
        step = sp.Rational(1, 32)
        far, near = (abs(sp.N((expr - kept).subs(variable, h), 30)) for h in (step, step / 2))
    except Exception:
        return None
    if not (far.is_number and near.is_number and far.is_finite and near.is_finite):
        return None
    if far == 0 and near == 0:
        return kept
    if near == 0 or far / near < sp.Rational(3, 4) * 2**order:
        return None
    return kept


def _series_parts(args):
    context = _context(args)
    expr = _expression(args, context)
    variable = _variable(args, context, expr)
    point = parse_expression(_take(args, "point", "math", "0"), context)
    order = _take(args, "order", "int", 6)
    if not 1 <= order <= 30:
        raise MathError("invalid_arguments", "'order' must be between 1 and 30")
    return expr, variable, point, order


def _series(args, reply):
    expr, variable, point, order = _series_parts(args)
    fast = _ring_series(expr, variable, order) if point == 0 else None
    result = sp.series(expr, variable, point, order) if fast is None else fast + sp.O(variable**order)
    reply.value(result, _precision(args), numeric=False)
    reply.fields["polynomial"] = reply.text(result.removeO())
    return reply.done()


def _transform(function, kind, default_from, default_to):
    def run(args, reply):
        context = _context(args)
        expr = _expression(args, context)
        given = _take(args, "variable", "math")
        source = context.symbol(given or default_from)
        target = context.symbol(_take(args, "transform_variable", "math", default_to))
        result = function(expr, source, target)
        if _is_open(result, kind):
            return reply.open(result, "transform")
        reply.value(result, _precision(args), numeric=False)
        return reply.done()
    return run


# solve tool

def _relations(args, context):
    return [parse_relation(text, context) for text in _take(args, "equations", "list", required=True)]


def _unknowns(args, context, relations):
    given = _take(args, "variables", "list")
    if given:
        return [context.symbol(name.strip()) for name in given]
    free = sorted(set().union(*(r.free_symbols for r in relations)), key=lambda s: s.name)
    if not free:
        raise MathError("invalid_arguments", "the equations have no variables; name the unknowns in "
                        "'variables' (that also makes e a variable instead of Euler's number)")
    return free


def _solution_rows(reply, solutions, precision):
    rows, decimals = [], []
    for solution in reply.items(solutions):
        rows.append({symbol.name: reply.text(value) for symbol, value in solution.items()})
        decimal = {symbol.name: _decimal(value, precision) for symbol, value in solution.items()}
        decimals.append({name: reply.text(text) for name, text in decimal.items() if text is not None})
    reply.fields["solutions"] = rows
    if any(decimals):
        reply.fields["numeric"] = decimals


def _solve(args, reply):
    context = _context(args)
    relations = _relations(args, context)
    unknowns = _unknowns(args, context, relations)
    precision = _precision(args)
    if any(not isinstance(r, sp.Eq) for r in relations):
        if len(unknowns) != 1:
            raise MathError("not_supported", "inequalities are solved for one variable at a time")
        result = sp.reduce_inequalities(relations, unknowns)
        reply.value(result, precision, numeric=False)
        try:
            reply.fields["set"] = reply.text(result.as_set())
        except Exception:
            pass
        return reply.done()
    try:
        solutions = sp.solve(relations, unknowns, dict=True)
    except NotImplementedError as error:
        raise MathError("not_supported", f"SymPy cannot solve this exactly: {str(error)[:200]}. "
                        "For a numerical root use nsolve with an initial_guess.")
    reply.value(solutions, precision, numeric=False)
    _solution_rows(reply, solutions, precision)
    if not solutions:
        reply.warnings.append("no solution found; this does not prove that none exists. "
                              "For a numerical root use nsolve with an initial_guess")
    return reply.done(count=len(solutions))


def _solveset(args, reply):
    context = _context(args)
    relations = _relations(args, context)
    unknowns = _unknowns(args, context, relations)
    if len(relations) != 1 or len(unknowns) != 1:
        raise MathError("not_supported", "solveset takes one equation or inequality in one variable")
    domain = args.get("domain") or "complex"
    if domain not in ("complex", "real"):
        raise MathError("invalid_arguments", "'domain' must be \"complex\" or \"real\"")
    relation = relations[0]
    if not isinstance(relation, sp.Eq):
        domain = "real"
    target = relation.lhs - relation.rhs if isinstance(relation, sp.Eq) else relation
    result = sp.solveset(target, unknowns[0], sp.S.Reals if domain == "real" else sp.S.Complexes)
    precision = _precision(args)
    reply.value(result, precision, numeric=False)
    if isinstance(result, sp.FiniteSet):
        _solution_rows(reply, [{unknowns[0]: v} for v in sorted(result, key=sp.default_sort_key)], precision)
    elif isinstance(result, sp.ConditionSet):
        reply.warnings.append("the solution set could not be written explicitly")
    return reply.done(domain=domain)


def _nsolve(args, reply):
    context = _context(args)
    relations = _relations(args, context)
    unknowns = _unknowns(args, context, relations)
    if any(not isinstance(r, sp.Eq) for r in relations) or len(relations) != len(unknowns):
        raise MathError("invalid_arguments", "nsolve needs as many equations as variables")
    if args.get("initial_guess") is None:
        raise MathError("invalid_arguments", "nsolve needs 'initial_guess': one starting value per "
                        "variable; it is never guessed for you")
    guesses = [parse_expression(g, context) for g in _take(args, "initial_guess", "list", required=True)]
    if len(guesses) != len(unknowns) or any(g.free_symbols for g in guesses):
        raise MathError("invalid_arguments", "'initial_guess' needs one number per variable")
    precision = _precision(args)
    residuals = [r.lhs - r.rhs for r in relations]
    if set().union(*(r.free_symbols for r in residuals)) - set(unknowns):
        raise MathError("invalid_arguments", "every symbol must be one of 'variables' for a numerical solve")

    # findroot gets plain closures, never generated code
    def closure(residual):
        def evaluate(*values):
            point = {u: sp.sympify(v) for u, v in zip(unknowns, values)}
            return residual.evalf(precision + 10, subs=point)._to_mpmath(mpmath.mp.prec)
        return evaluate

    with mpmath.workdps(precision + 10):
        start = [g.evalf(precision + 10)._to_mpmath(mpmath.mp.prec) for g in guesses]
        functions = [closure(r) for r in residuals]
        try:
            root = mpmath.findroot(functions[0] if len(functions) == 1 else functions,
                                   start[0] if len(start) == 1 else start)
        except (ValueError, ZeroDivisionError, TypeError) as error:
            raise MathError("no_result", f"the numerical solver did not converge: {str(error)[:200]}")
        values = [root] if len(unknowns) == 1 else list(root)
        solution = {u: sp.sympify(v).evalf(precision) for u, v in zip(unknowns, values)}
    reply.value([solution], precision, numeric=False)
    reply.fields["numeric"] = [{u.name: reply.text(v) for u, v in solution.items()}]
    reply.warnings.append("numerical result near the initial guess; other solutions may exist")
    return reply.done()


def _ode_context(args):
    name = _take(args, "function", "math", "y").strip()
    variable = _take(args, "variable", "math")
    # "y(x)" names both the function and its variable
    written = re.fullmatch(r"(\w+)\((\w+)\)", name)
    if written:
        name, variable = written.group(1), variable or written.group(2)
    name = check_identifier(name)
    variable = check_identifier((variable or "x").strip())
    context = _context(args, {name: variable})
    return context, context.functions[name](context.function_variables[name])


def _dsolve_other_methods(reply, equation, function, conditions):
    """SymPy's first choice of method crashed: try the others it lists for this equation."""
    for hint in sp.classify_ode(equation, function)[1:]:
        if hint.endswith("_Integral"):
            continue
        try:
            result = sp.dsolve(equation, function, hint=hint, ics=conditions or None)
        except Exception:
            continue
        reply.fields["method"] = hint
        if "series" in hint:
            reply.warnings.append("no closed form was found; this is a power series solution")
        return result
    raise MathError("not_supported", "SymPy could not solve this differential equation")


def _dsolve(args, reply):
    context, function = _ode_context(args)
    relations = _relations(args, context)
    if len(relations) != 1 or not isinstance(relations[0], sp.Eq):
        raise MathError("not_supported", "dsolve takes one differential equation")
    conditions = {}
    for key, value in (_take(args, "initial_conditions", "dict") or {}).items():
        conditions[parse_expression(key, context)] = parse_expression(value, context)
    try:
        result = sp.dsolve(relations[0], function, ics=conditions or None)
    except (TypeError, AttributeError, IndexError, KeyError):
        result = _dsolve_other_methods(reply, relations[0], function, conditions)
    solutions = result if isinstance(result, list) else [result]
    reply.value(solutions[0] if len(solutions) == 1 else solutions, _precision(args), numeric=False)
    reply.fields["solutions"] = [reply.text(s) for s in reply.items(solutions)]
    return reply.done()


# matrix tool

def _matrix(args, context, name, required=True):
    rows = args.get(name)
    if rows is None:
        if required:
            raise MathError("invalid_arguments", f"'{name}' is required")
        return None
    if not isinstance(rows, list) or not rows:
        raise MathError("invalid_arguments", f"'{name}' must be a list of rows")
    if not isinstance(rows[0], list):
        rows = [[entry] for entry in rows]
    width = len(rows[0])
    if len(rows) > MAX_MATRIX_SIDE or width > MAX_MATRIX_SIDE:
        raise MathError("input_too_large", f"matrices are limited to {MAX_MATRIX_SIDE} by {MAX_MATRIX_SIDE}")
    if width == 0 or any(not isinstance(row, list) or len(row) != width for row in rows):
        raise MathError("invalid_arguments", f"every row of '{name}' must have the same length")
    return sp.Matrix([[parse_expression(_take({name: entry}, name, "math", required=True), context)
                       for entry in row] for row in rows])


def _matrix_rows(reply, matrix):
    return [[reply.text(entry) for entry in matrix.row(i)] for i in range(matrix.rows)]


def _matrix_value(reply, matrix, precision):
    reply.value(matrix, precision, numeric=False)
    reply.fields["matrix"] = _matrix_rows(reply, matrix)


def _square(matrix):
    if matrix.rows != matrix.cols:
        raise MathError("invalid_arguments", "the matrix must be square")


def _matrix_operation(operation):
    def run(args, reply):
        context = _context(args)
        matrix = _matrix(args, context, "matrix")
        precision = _precision(args)
        if operation == "determinant":
            _square(matrix)
            reply.value(sp.simplify(matrix.det()), precision)
        elif operation == "inverse":
            _square(matrix)
            if sp.simplify(matrix.det()) == 0:
                raise MathError("no_result", "the matrix is singular and has no inverse")
            _matrix_value(reply, matrix.inv().applyfunc(sp.simplify), precision)
        elif operation == "transpose":
            _matrix_value(reply, matrix.T, precision)
        elif operation == "rank":
            reply.value(sp.Integer(matrix.rank()), precision)
        elif operation == "rref":
            reduced, pivots = matrix.rref()
            _matrix_value(reply, reduced, precision)
            reply.fields["pivot_columns"] = list(pivots)
        elif operation == "nullspace":
            basis = matrix.nullspace()
            reply.value(basis, precision, numeric=False)
            reply.fields["vectors"] = [[reply.text(e) for e in vector] for vector in reply.items(basis)]
        elif operation == "eigenvalues":
            _square(matrix)
            values = matrix.eigenvals()
            ordered = sorted(values, key=sp.default_sort_key)
            reply.value(ordered, precision, numeric=False)
            reply.fields["eigenvalues"] = [
                {"value": reply.text(v), "multiplicity": int(values[v]),
                 **({"numeric": reply.text(d)} if (d := _decimal(v, precision)) else {})}
                for v in reply.items(ordered)]
        elif operation == "eigenvectors":
            _square(matrix)
            triples = sorted(matrix.eigenvects(), key=lambda t: sp.default_sort_key(t[0]))
            reply.value([t[0] for t in triples], precision, numeric=False)
            reply.fields["eigenvectors"] = [
                {"eigenvalue": reply.text(value), "multiplicity": int(multiplicity),
                 "vectors": [[reply.text(sp.simplify(e)) for e in vector] for vector in vectors]}
                for value, multiplicity, vectors in reply.items(triples)]
        elif operation == "multiply":
            other = _matrix(args, context, "other")
            if matrix.cols != other.rows:
                raise MathError("invalid_arguments", "the column count of 'matrix' must equal the row count of 'other'")
            _matrix_value(reply, (matrix * other).applyfunc(sp.expand), precision)
        elif operation == "solve":
            other = _matrix(args, context, "other")
            if other.rows != matrix.rows:
                raise MathError("invalid_arguments", "'other' needs one row per row of 'matrix'")
            unknowns = sp.symbols(f"x1:{matrix.cols + 1}")
            solutions = sp.linsolve((matrix, other), *unknowns)
            if not solutions:
                raise MathError("no_result", "the linear system has no solution")
            solution = next(iter(solutions))
            reply.value(sp.Matrix(solution), precision, numeric=False)
            reply.fields["solution"] = [reply.text(e) for e in solution]
            if any(e.free_symbols & set(unknowns) for e in solution):
                reply.warnings.append("infinitely many solutions; x1, x2, ... are free parameters")
        return reply.done()
    return run


# verify tool

_PROBES = (sp.Rational(-7, 3), sp.Rational(5, 4), sp.Rational(-1, 2), sp.Integer(3), sp.Rational(11, 7))


def _counterexample(expr):
    """Values of the variables at which expr is clearly not zero, if a few trials find any."""
    symbols = sorted(expr.free_symbols, key=lambda s: s.name)
    for shift in range(len(_PROBES)):
        point = {}
        for index, symbol in enumerate(symbols):
            value = _PROBES[(index + shift) % len(_PROBES)]
            if symbol.is_integer:
                value = sp.Integer(sp.floor(value * 3))
            if symbol.is_positive or symbol.is_nonnegative:
                value = abs(value)
            elif symbol.is_negative or symbol.is_nonpositive:
                value = -abs(value)
            point[symbol] = value
        try:
            value = sp.N(expr.subs(point), 30)
        except Exception:
            continue
        if value.is_number and value.is_finite and abs(value) > sp.Float("1e-15"):
            return point
    return None


def _is_zero(expr):
    """True, False, or None when SymPy cannot decide."""
    simplified = sp.simplify(expr)
    if simplified == 0:
        return True, simplified
    verdict = simplified.equals(0)
    if verdict is None and not simplified.free_symbols:
        verdict = bool(abs(sp.N(simplified, 50)) < sp.Float("1e-40"))
    if verdict is None and _counterexample(simplified) is not None:
        # one point where the difference is not zero settles it
        verdict = False
    return verdict, simplified


def _equivalent(args, reply):
    # values to test were given: the request is a solution check under the wrong name
    if args.get("solution") is not None and (args.get("equations") is not None or args.get("right") is not None):
        reply.fields["operation"] = "solution"
        reply.warnings.append("treated as a solution check because 'solution' was given")
        if args.get("equations") is not None:
            args = {k: v for k, v in args.items() if k not in ("left", "right")}
        return _check_solution(args, reply)
    if args.get("equations") is not None:
        # left = right is the equation and each "name = value" beside it the candidate
        equations = args["equations"] if isinstance(args["equations"], list) else [args["equations"]]
        named = [re.fullmatch(r"\s*([A-Za-z]\w*)\s*=(?!=)\s*(.+)", e) if isinstance(e, str) else None for e in equations]
        if not all(named) or args.get("left") is None or args.get("right") is None:
            raise MathError("invalid_arguments", "'equations' belongs to operation solution, with the values to "
                            "test in 'solution'; equivalent compares 'left' and 'right' only")
        reply.fields["operation"] = "solution"
        reply.warnings.append("treated as a solution check of left = right with the values given in 'equations'")
        args = {k: v for k, v in args.items() if k not in ("left", "right", "equations")} | {
            "equations": [f"{_take(args, 'left', 'math')} = {_take(args, 'right', 'math')}"],
            "solution": {m.group(1): m.group(2).strip() for m in named}}
        return _check_solution(args, reply)
    context = _context(args)
    left = _expression(args, context, "left")
    right = _expression(args, context, "right")
    verdict, difference = _is_zero(left - right)
    reply.fields["equivalent"] = verdict
    reply.fields["difference"] = reply.text(difference)
    if verdict is False and difference.free_symbols:
        point = _counterexample(difference)
        if point:
            reply.fields["differs_at"] = {symbol.name: reply.text(value) for symbol, value in point.items()}
    try:
        reply.fields["difference_latex"] = reply.text(sp.latex(difference))
    except Exception:
        pass
    if verdict is None:
        reply.warnings.append("could not decide; the difference did not simplify to zero")
    return reply.done()


def _check_solution(args, reply):
    # a model checking one equation tends to fill left and right, as it does for equivalent
    if args.get("equations") is None and args.get("left") is not None and args.get("right") is not None:
        args = {**args, "equations": [f"{_take(args, 'left', 'math')} = {_take(args, 'right', 'math')}"]}
    # or it writes the candidate there, as name and value
    elif args.get("solution") is None and isinstance(args.get("left"), str) and args.get("right") is not None \
            and args["left"].strip().isidentifier():
        args = {**args, "solution": {args["left"].strip(): args["right"]}}
    # an equation with derivatives names its unknown function; the candidate is that function
    candidate = args.get("solution")
    text = " ".join(str(e) for e in args.get("equations") or [] if isinstance(e, str))
    named = re.search(r"([A-Za-z]\w*)\s*'|\bd\d?([A-Za-z]\w*)/d\w|(?:diff|Derivative)\(\s*([A-Za-z]\w*)", text)
    if args.get("function") is None and isinstance(candidate, dict) and named:
        function = next(group for group in named.groups() if group)
        offered = {k: v for k, v in candidate.items() if str(v).strip() != k.strip()}
        if function not in offered and len(offered) == 1:
            offered = {function: next(iter(offered.values()))}
        args = {**args, "function": function, "solution": offered}
    functions = None
    if args.get("function") is not None:
        context, function = _ode_context(args)
        functions = {function.func.__name__: function}
    else:
        context = _context(args)
    relations = _relations(args, context)
    solution = _take(args, "solution", "dict", required=True)
    replacements = {}
    for key, value in solution.items():
        key = key.strip()
        target = functions[key] if functions and key in functions else context.symbol(key)
        replacements[target] = parse_expression(value, context)

    checks, satisfied = [], True
    for relation in relations:
        if isinstance(relation, sp.Eq):
            # substituting into the Eq itself would collapse it to a bare True or False
            verdict, residual = _is_zero((relation.lhs - relation.rhs).subs(replacements).doit())
            checks.append({"equation": reply.text(relation), "satisfied": verdict,
                           "residual": reply.text(residual)})
        else:
            outcome = sp.simplify(relation.subs(replacements).doit())
            verdict = True if outcome == sp.true else False if outcome == sp.false else None
            checks.append({"equation": reply.text(relation), "satisfied": verdict,
                           "residual": reply.text(outcome)})
        if verdict is None:
            reply.warnings.append("could not decide one of the checks")
        satisfied = None if (verdict is None and satisfied) else (satisfied and verdict)
    return reply.done(satisfied=satisfied, checks=checks)


OPERATIONS = {
    "simplify": _rewrite(sp.simplify),
    "expand": _rewrite(sp.expand),
    "factor": _rewrite(sp.factor),
    "cancel": _rewrite(sp.cancel),
    "together": _rewrite(sp.together),
    "apart": _apart,
    "evaluate": _evaluate,
    "differentiate": _differentiate,
    "integrate": _integrate,
    "limit": _limit,
    "series": _series,
    "summation": _ranged(sp.summation, (sp.Sum,), "sum"),
    "product": _ranged(sp.product, (sp.Product,), "product"),
    "laplace_transform": _transform(
        lambda f, t, s: sp.laplace_transform(f, t, s, noconds=True), sp.LaplaceTransform, "t", "s"),
    "inverse_laplace_transform": _transform(
        sp.inverse_laplace_transform, sp.InverseLaplaceTransform, "s", "t"),
    "fourier_transform": _transform(sp.fourier_transform, sp.FourierTransform, "x", "k"),
    "inverse_fourier_transform": _transform(
        sp.inverse_fourier_transform, sp.InverseFourierTransform, "k", "x"),
    "solve": _solve,
    "solveset": _solveset,
    "nsolve": _nsolve,
    "dsolve": _dsolve,
    **{name: _matrix_operation(name) for name in (
        "determinant", "inverse", "transpose", "rank", "rref", "nullspace",
        "eigenvalues", "eigenvectors", "multiply")},
    "linear_solve": _matrix_operation("solve"),
    "equivalent": _equivalent,
    "solution": _check_solution,
}


def failure(operation, code, message):
    return {"success": False, "operation": operation, "error": {"code": code, "message": message}}


# after a timeout

def _timed_out(operation, seconds, what="The calculation did not finish"):
    return {"success": False, "operation": operation, "timed_out": True,
            "error": {"code": "timeout",
                      "message": f"{what} within the {seconds:g} s computation budget."}, "warnings": []}


def _recover_integral(args, base, emit):
    context = _context(args)
    expr = _expression(args, context)
    variable = _variable(args, context, expr)
    bounds = _bounds(args, context)
    reply = Reply("integrate")
    base.update(exact=None, unevaluated=reply.text(sp.Integral(expr, (variable, *bounds) if bounds else variable)))
    emit(dict(base))
    reply.fields = base
    precision = _precision(args)
    if _numeric_integral(reply, expr, variable, bounds, precision, True):
        emit(dict(reply.done()))
    if not bounds or not expr.has(sp.sin, sp.cos, sp.tan, sp.cot, sp.sinh, sp.cosh, sp.tanh):
        return reply.done()
    # products of trigonometric functions and exponentials often close once written as exponentials
    result = sp.simplify(sp.integrate(sp.expand(expr.rewrite(sp.exp)), (variable, *bounds)))
    if _is_open(result, sp.Integral) or result.has(sp.nan, sp.zoo):
        return reply.done()
    numeric = base.get("numeric") if base.get("method") == "numerical_integration" else None
    if numeric is not None and abs(sp.N(result, 20) - sp.Float(numeric, 20)) > sp.Float("1e-6") * max(1, abs(sp.Float(numeric, 20))):
        return reply.done()
    exact = Reply("integrate")
    exact.value(result, precision)
    exact.warnings.append("the first attempt ran out of time; this was found after rewriting the "
                          "trigonometric functions as exponentials")
    return exact.done(method="exponential_rewrite", timed_out_symbolic=True)


def _recover_series(args, base, emit):
    expr, variable, point, order = _series_parts(args)
    reply = Reply("series")
    base.update(exact=None, expression=reply.text(expr), variable=variable.name, point=reply.text(point), order=order)
    emit(dict(base))
    for lower in range(4, order, 4):
        result = sp.series(expr, variable, point, lower)
        base["partial"] = {"order": lower, "exact": reply.text(result), "polynomial": reply.text(result.removeO())}
        base["warnings"] = [f"only the expansion to order {lower} finished in time; no higher terms are known"]
        emit(dict(base))
    return base


def _recover_dsolve(args, base, emit):
    context, _ = _ode_context(args)
    base.update(exact=None, unevaluated=[Reply("dsolve").text(r) for r in _relations(args, context)])
    return base


_RECOVERY = {"integrate": _recover_integral, "series": _recover_series, "dsolve": _recover_dsolve}


def after_timeout(operation, arguments, seconds, emit):
    """Runs in a fresh worker once a request has been stopped: what can still be said about it."""
    what = "No closed form was obtained" if operation == "integrate" else "The calculation did not finish"
    base = _timed_out(operation, seconds, what)
    recover = _RECOVERY.get(operation)
    if recover is None or not isinstance(arguments, dict):
        return base
    del _notes[:]
    emit(dict(base))
    try:
        return recover(arguments, base, emit)
    except Exception:
        return base


def run(operation, arguments):
    if not isinstance(arguments, dict):
        return failure(operation, "invalid_arguments", "arguments must be an object")
    handler = OPERATIONS.get(operation) if isinstance(operation, str) else None
    if handler is None:
        return failure(str(operation)[:40], "unknown_operation",
                       "'operation' must be one of: " + ", ".join(OPERATIONS))
    del _notes[:]
    if operation == "expand" and (arguments.get("order") is not None or arguments.get("point") is not None):
        # only a series has an order or a point
        operation, handler = "series", OPERATIONS["series"]
        _notes.append("treated as a series expansion because an order or a point was given")
    try:
        return handler(arguments, Reply(operation))
    except MathError as error:
        return failure(operation, error.code, error.message)
    except RecursionError:
        return failure(operation, "too_large", "the expression is too complex to process")
    except NotImplementedError as error:
        return failure(operation, "not_supported", f"SymPy cannot do this: {str(error)[:300]}")
    except (ValueError, TypeError, ArithmeticError, AttributeError, KeyError, IndexError) as error:
        return failure(operation, "math_error", f"{type(error).__name__}: {str(error)[:300]}")
