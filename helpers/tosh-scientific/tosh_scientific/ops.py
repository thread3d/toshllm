# ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
# Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The numerical operations behind the scientific tools.

A request names an operation and carries numbers, options and, where a function is needed,
math text for the restricted grammar. Each operation is an entry in a fixed table. NumPy and
the SciPy modules load the first time an operation needs them.
"""

import math
import re
import warnings

import numpy as np

from . import numeric
from .arrays import (SciError, array, choice, integer, listed_limit, matrix, number, real, values)

MAX_FFT = 1 << 20
MAX_ODE_POINTS = 2000
MAX_ITERATIONS = 2000
MAX_DIMENSIONS = 3
MAX_VARIABLES = 32
MAX_PEAKS = 50
MAX_ELEMENTS_SMALL = 64


def _variables(args, expressions, name="variables", single="variable"):
    """The symbols a request treats as unknowns, in the order it gave them."""
    given = args.get(name)
    if given is None and args.get(single) is not None:
        given = [args[single]]
    if given is not None:
        if isinstance(given, str):
            given = [given]
        if not isinstance(given, list) or not given or len(given) > MAX_VARIABLES:
            raise SciError("invalid_arguments", f"'{name}' must name between 1 and {MAX_VARIABLES} variables")
        return numeric.symbols(given, name)
    free = set()
    for expr in expressions:
        free |= expr.free_symbols
    free -= set(_parameters(args))
    if not free:
        raise SciError("invalid_arguments", "the expression has no variable")
    return sorted(free, key=lambda s: s.name)


def _declared(args):
    names = []
    for key in ("variable",):
        if isinstance(args.get(key), str):
            names.append(args[key])
    for key in ("variables",):
        if isinstance(args.get(key), list):
            names += [n for n in args[key] if isinstance(n, str)]
    for key in ("parameters", "initial_guess", "bounds", "data"):
        if isinstance(args.get(key), dict):
            names += list(args[key])
    return [n.strip() for n in names]


def _parameters(args):
    given = args.get("parameters")
    if given is None:
        return {}
    if not isinstance(given, dict):
        raise SciError("invalid_arguments", "'parameters' must be an object of name: number")
    return {symbol: numeric.constant(value, "parameters")
            for symbol, value in zip(numeric.symbols(given, "parameters"), given.values())}


def _parsed(args, text, equation=False):
    expr = (numeric.parse_equation if equation else numeric.parse)(text, _declared(args))
    constants = _parameters(args)
    return expr.subs(constants) if constants else expr


def _finite(value, what):
    if not np.all(np.isfinite(value)):
        raise SciError("non_finite_result", f"{what} is not finite")
    return value


def _guess(args, variables, name="initial_guess", required=True):
    given = args.get(name)
    if given is None:
        if required:
            raise SciError("missing_initial_guess",
                           f"'{name}' is required: one starting value per variable. It is never guessed for you")
        return None
    if isinstance(given, dict):
        missing = [v.name for v in variables if v.name not in given]
        if missing:
            raise SciError("invalid_arguments", f"'{name}' has no value for '{missing[0]}'")
        return np.array([numeric.constant(given[v.name], name) for v in variables])
    if isinstance(given, (int, float)) and not isinstance(given, bool):
        given = [given]
    if not isinstance(given, list) or len(given) != len(variables):
        raise SciError("invalid_arguments", f"'{name}' needs one value per variable ({len(variables)})")
    return np.array([numeric.constant(v, name) for v in given])


def _named(variables, point):
    return {v.name: number(x) for v, x in zip(variables, np.atleast_1d(point))}


# compute

def _integrate(args):
    from scipy import integrate
    if args.get("y") is not None:
        y = array(args["y"], "y", (1,))
        x = array(args.get("x"), "x", (1,), required=False)
        if x is not None and (x.shape != y.shape or np.any(np.diff(x) <= 0)):
            raise SciError("invalid_dimensions", "'x' must be increasing and as long as 'y'")
        if y.size < 2:
            raise SciError("invalid_arguments", "at least two samples are needed")
        step = real(args, "dx", 1.0)
        method = choice(args, "method", ("simpson", "trapezoid"), "simpson")
        rule = integrate.simpson if method == "simpson" else integrate.trapezoid
        value = rule(y, x=x) if x is not None else rule(y, dx=step)
        return {"value": number(value), "method": method, "samples": int(y.size)}

    expr = _parsed(args, args.get("expression"))
    bounds = args.get("bounds")
    if bounds is not None:
        if not isinstance(bounds, dict) or not 1 <= len(bounds) <= MAX_DIMENSIONS:
            raise SciError("invalid_arguments", f"'bounds' maps 1 to {MAX_DIMENSIONS} variables to [lower, upper]")
        variables = numeric.symbols(bounds, "bounds")
        ranges = []
        for limits in bounds.values():
            if not isinstance(limits, list) or len(limits) != 2:
                raise SciError("invalid_arguments", "each entry of 'bounds' is [lower, upper]")
            ranges.append([numeric.constant(limits[0], "bounds"), numeric.constant(limits[1], "bounds")])
    else:
        variables = _variables(args, [expr])
        if len(variables) != 1:
            raise SciError("invalid_arguments", "name the integration variable, or give 'bounds' for several")
        if args.get("lower") is None or args.get("upper") is None:
            raise SciError("invalid_arguments", "'lower' and 'upper' are required: this integrates numerically "
                           "over an interval and cannot return an antiderivative")
        ranges = [[numeric.constant(args["lower"], "lower"), numeric.constant(args["upper"], "upper")]]
    notes = []
    for limits in ranges:
        for index, bound in enumerate(limits):
            if math.isfinite(bound) and abs(bound) >= 1e15:
                # quadrature over a range that wide never samples where the function lives
                limits[index] = math.copysign(math.inf, bound)
                notes = ["a bound of 1e15 or more was taken as infinity"]
    function = numeric.compile_expression(expr, variables)
    tolerance = real(args, "tolerance", 1e-10, low=1e-14, high=1e-2)

    def integrand(*point):
        value = float(function(*point))
        if not math.isfinite(value):
            raise SciError("non_finite_result", "the integrand is not finite inside the interval; "
                           "the integral may diverge or the function is undefined there")
        return value

    if len(variables) == 1:
        # with full_output quad reports trouble as a fourth element instead of a warning
        outcome = integrate.quad(integrand, *ranges[0], epsabs=tolerance, epsrel=tolerance, limit=200, full_output=1)
        value, estimate, evaluations = outcome[0], outcome[1], outcome[2]["neval"]
        method = "adaptive quadrature (QUADPACK)"
        if len(outcome) > 3:
            raise SciError("not_converged", "the integral did not converge to the requested tolerance: "
                           + str(outcome[3]).split(".")[0].strip().lower(),
                           estimate=number(value) if math.isfinite(value) else None)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error", integrate.IntegrationWarning)
            try:
                value, estimate, info = integrate.nquad(lambda *p: integrand(*p[::-1]), ranges[::-1],
                                                        opts={"epsabs": tolerance, "epsrel": tolerance, "limit": 50},
                                                        full_output=True)
            except integrate.IntegrationWarning as warning:
                raise SciError("not_converged", "the integral did not converge to the requested tolerance: "
                               + str(warning).split(".")[0].strip().lower())
        evaluations, method = info["neval"], "nested adaptive quadrature"
    if estimate > max(1e-6, 1e-6 * abs(value)):
        raise SciError("not_converged", "the error estimate is too large to trust the value",
                       estimate=number(value), error_estimate=number(estimate))
    return {"value": number(value), "error_estimate": number(estimate), "method": method,
            "tolerance": tolerance, "evaluations": int(evaluations), "warnings": notes}


def _root(args):
    from scipy import optimize
    texts = args.get("equations")
    if texts is None:
        texts = [args.get("expression")]
    if isinstance(texts, str):
        texts = [texts]
    if not isinstance(texts, list) or not texts or len(texts) > MAX_VARIABLES or texts[0] is None:
        raise SciError("invalid_arguments", "give 'expression', or 'equations' for a system")
    residuals = [_parsed(args, text, equation=True) for text in texts]
    variables = _variables(args, residuals)
    if len(variables) != len(residuals):
        raise SciError("invalid_arguments", f"{len(residuals)} equations need {len(residuals)} variables, "
                       f"not {len(variables)}")
    sp = numeric.sympy()
    functions = [numeric.compile_expression(r, variables) for r in residuals]

    if len(variables) == 1:
        function = lambda x: float(functions[0](x))
        bracket = args.get("bracket")
        if bracket is None and args.get("lower") is not None and args.get("upper") is not None:
            # the interval written as for an integral
            bracket = [args["lower"], args["upper"]]
        if bracket is not None:
            if not isinstance(bracket, list) or len(bracket) != 2:
                raise SciError("invalid_arguments", "'bracket' is [a, b] with the function changing sign between them")
            a, b = (numeric.constant(v, "bracket") for v in bracket)
            fa, fb = function(a), function(b)
            if not (math.isfinite(fa) and math.isfinite(fb)):
                raise SciError("non_finite_result", "the function is not finite at the ends of the bracket")
            if fa * fb > 0:
                raise SciError("invalid_bracket", "the function has the same sign at both ends of the bracket, "
                               "so a root between them is not guaranteed; to solve f(x) = c write the whole "
                               "equation in 'expression'", values=[number(fa), number(fb)])
            root, result = optimize.brentq(function, a, b, xtol=1e-14, rtol=1e-14, maxiter=500, full_output=True)
            method, iterations, converged = "brentq", result.iterations, result.converged
        else:
            start = _guess(args, variables)
            derivative = numeric.compile_expression(sp.diff(residuals[0], variables[0]), variables)
            result = optimize.root_scalar(function, x0=float(start[0]), fprime=lambda x: float(derivative(x)),
                                          method="newton", maxiter=200, xtol=1e-14)
            root, method, iterations, converged = result.root, "newton", result.iterations, result.converged
        residual = function(root) if math.isfinite(root) else math.inf
        if not converged or not math.isfinite(residual) or abs(residual) > 1e-7 * max(1.0, abs(root)):
            raise SciError("not_converged", "no root was found from that starting point")
        return {"root": _named(variables, root), "residual": number(residual), "method": method,
                "iterations": int(iterations)}

    start = _guess(args, variables)
    jacobian = [[numeric.compile_expression(sp.diff(r, v), variables) for v in variables] for r in residuals]
    system = lambda p: np.array([float(f(*p)) for f in functions])
    result = optimize.root(system, start, jac=lambda p: np.array([[float(d(*p)) for d in row] for row in jacobian]),
                           method="hybr")
    residual = np.abs(system(result.x)).max() if np.all(np.isfinite(result.x)) else math.inf
    if not result.success or not math.isfinite(residual) or residual > 1e-7:
        raise SciError("not_converged", "no solution was found from that starting point: " + str(result.message)[:160])
    return {"root": _named(variables, result.x), "residual": number(residual), "method": "hybrid Powell",
            "evaluations": int(result.nfev)}


def _interpolate(args):
    from scipy import interpolate
    points = args.get("x")
    if args.get("y") is None and isinstance(points, list) and points and all(isinstance(p, list) and len(p) == 2 for p in points):
        # the points came as [x, y] pairs
        args = {**args, "x": [p[0] for p in points], "y": [p[1] for p in points]}
    text = args.get("expression")
    if args.get("y") is None and isinstance(text, str):
        # the points written out as text, "(0, 1), (1, 3)"; x is then where to estimate
        pair = r"[\(\[]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\)\]]"
        if re.fullmatch(rf"\s*[\[\(]?\s*(?:{pair}\s*,?\s*){{2,}}[\]\)]?\s*", text):
            found = re.findall(pair, text)
            args = {**args, "x": [float(a) for a, _ in found], "y": [float(b) for _, b in found],
                    "at": args.get("at") if args.get("at") is not None else args.get("x")}
    for name, what in (("x", "the positions of the data points"), ("y", "the value at each x"),
                       ("at", "the positions to estimate")):
        if args.get(name) is None:
            raise SciError("invalid_arguments", f"'{name}' is required: {what}, as a list of numbers")
    x, y = array(args.get("x"), "x", (1,)), array(args.get("y"), "y", (1,))
    at = args["at"]
    at = array(at if isinstance(at, (list, dict)) else [at], "at", (1,))
    if x.shape != y.shape:
        raise SciError("invalid_dimensions", "'x' and 'y' must have the same length")
    order = np.argsort(x)
    x, y = x[order], y[order]
    if np.any(np.diff(x) == 0):
        raise SciError("invalid_arguments", "'x' has repeated values")
    kinds = ("linear", "cubic", "pchip", "akima")
    named = args["expression"].lower() if isinstance(args.get("expression"), str) else ""
    for word in kinds + ("spline",):
        # the kind written where the formula goes
        if word in named:
            args = {**args, "kind": "cubic" if word == "spline" else word}
            break
    kind = choice(args, "kind", kinds, "linear")
    needed = {"linear": 2, "cubic": 4, "pchip": 2, "akima": 5}[kind]
    if x.size < needed:
        raise SciError("invalid_arguments", f"{kind} interpolation needs at least {needed} points")
    outside = (at < x[0]) | (at > x[-1])
    if np.any(outside) and args.get("extrapolate") is not True:
        raise SciError("out_of_range", f"{int(outside.sum())} of the requested points lie outside the data "
                       f"({number(x[0])} to {number(x[-1])}); set 'extrapolate' to true to allow it")
    if kind == "linear":
        result = interpolate.interp1d(x, y, kind="linear", fill_value="extrapolate")(at)
    elif kind == "cubic":
        result = interpolate.CubicSpline(x, y, extrapolate=True)(at)
    elif kind == "pchip":
        result = interpolate.PchipInterpolator(x, y, extrapolate=True)(at)
    else:
        result = interpolate.Akima1DInterpolator(x, y, extrapolate=True)(at)
    reply = {"values": values(result, listed_limit(args)), "kind": kind, "points": int(x.size)}
    if np.any(outside):
        reply["extrapolated"] = int(outside.sum())
    return reply


def _evaluate(args):
    expr = _parsed(args, args.get("expression"))
    data = args.get("data") or {}
    if not isinstance(data, dict) or len(data) > MAX_VARIABLES:
        raise SciError("invalid_arguments", "'data' maps variable names to numbers or lists")
    names = {symbol.name for symbol in expr.free_symbols}
    if not data and args.get("at") is not None:
        # the points written as variable and at
        variable = args.get("variable")
        if isinstance(variable, str) and variable in names:
            data = {variable: args["at"]}
        elif len(names) == 1:
            data = {next(iter(names)): args["at"]}
    ignored = sorted(set(data) - names) + (["at"] if args.get("at") is not None and not data else [])
    data = {name: value for name, value in data.items() if name in names}
    variables = numeric.symbols(data, "data")
    arrays = [array(v if isinstance(v, (list, dict)) else [v], "data") for v in data.values()]
    try:
        result = np.asarray(numeric.compile_expression(expr, variables)(*arrays), dtype=float)
        result = np.broadcast_to(result, np.broadcast_shapes(*[a.shape for a in arrays])) if arrays else result
    except ValueError:
        raise SciError("invalid_dimensions", "the arrays in 'data' do not have compatible shapes")
    notes = [f"the expression has no '{ignored[0]}', so the values given for it were not used"] if ignored else []
    if not arrays or all(a.size == 1 for a in arrays):
        return {"value": number(result.reshape(-1)[0]), "warnings": notes}
    return {"values": values(result, listed_limit(args)), "count": int(result.size), "warnings": notes}


# linear algebra

def _square(a):
    if a.shape[0] != a.shape[1]:
        raise SciError("invalid_dimensions", f"the matrix must be square, not {a.shape[0]} x {a.shape[1]}")


def _condition(a):
    return float(np.linalg.cond(a))


def _linalg(operation):
    def run(args):
        a = matrix(args.get("matrix"), "matrix")
        limit = listed_limit(args)
        reply = {"shape": list(a.shape)}
        try:
            if operation in ("solve", "least_squares", "multiply"):
                other = array(args.get("other"), "other", (1, 2))
            if operation == "solve":
                _square(a)
                if other.shape[0] != a.shape[0]:
                    raise SciError("invalid_dimensions", f"'other' needs {a.shape[0]} rows to match the matrix")
                condition = _condition(a)
                if not math.isfinite(condition) or condition > 1e15:
                    raise SciError("singular_matrix", "the matrix is singular to working precision; "
                                   "the system has no unique solution")
                solution = np.linalg.solve(a, other)
                reply.update(solution=values(solution, limit), condition=number(condition),
                             residual=number(np.abs(a @ solution - other).max()))
                if condition > 1e10:
                    reply["warnings"] = ["the matrix is ill-conditioned; several digits of the solution are unreliable"]
            elif operation == "inverse":
                _square(a)
                condition = _condition(a)
                if not math.isfinite(condition) or condition > 1e15:
                    raise SciError("singular_matrix", "the matrix is singular to working precision and has no inverse")
                reply.update(inverse=values(np.linalg.inv(a), limit), condition=number(condition))
            elif operation == "determinant":
                _square(a)
                sign, logarithm = np.linalg.slogdet(a)
                if sign != 0 and logarithm > 700:
                    reply.update(sign=number(sign), log_abs_determinant=number(logarithm),
                                 warnings=["the determinant overflows a float; its sign and logarithm are given"])
                else:
                    reply["determinant"] = number(sign * math.exp(logarithm)) if sign != 0 else 0.0
            elif operation == "rank":
                reply["rank"] = int(np.linalg.matrix_rank(a))
            elif operation == "condition":
                condition = _condition(a)
                reply["condition"] = number(condition) if math.isfinite(condition) else None
                if not math.isfinite(condition):
                    reply["warnings"] = ["the matrix is singular"]
            elif operation == "eigen":
                _square(a)
                if np.allclose(a, a.T, rtol=1e-12, atol=1e-12):
                    eigenvalues, vectors = np.linalg.eigh(a)
                    reply["symmetric"] = True
                else:
                    eigenvalues, vectors = np.linalg.eig(a)
                    order = np.argsort(-np.abs(eigenvalues), kind="stable")
                    eigenvalues, vectors = eigenvalues[order], vectors[:, order]
                reply["eigenvalues"] = values(eigenvalues, limit)
                if a.shape[0] <= 8:
                    reply["eigenvectors"] = values(vectors.T, MAX_ELEMENTS_SMALL)
                    reply["note"] = "eigenvectors are listed in the same order as the eigenvalues"
            elif operation == "svd":
                small = max(a.shape) <= 8
                if small:
                    u, s, vt = np.linalg.svd(a, full_matrices=False)
                    reply.update(singular_values=values(s, limit), u=values(u, MAX_ELEMENTS_SMALL),
                                 vt=values(vt, MAX_ELEMENTS_SMALL))
                else:
                    reply["singular_values"] = values(np.linalg.svd(a, compute_uv=False), limit)
                reply["rank"] = int(np.linalg.matrix_rank(a))
            elif operation == "qr":
                q, r = np.linalg.qr(a)
                reply.update(q=values(q, limit), r=values(r, limit))
            elif operation == "least_squares":
                if other.shape[0] != a.shape[0]:
                    raise SciError("invalid_dimensions", f"'other' needs {a.shape[0]} rows to match the matrix")
                solution, _, rank, _ = np.linalg.lstsq(a, other, rcond=None)
                reply.update(solution=values(solution, limit), rank=int(rank),
                             residual_norm=number(np.linalg.norm(a @ solution - other)))
                if rank < min(a.shape):
                    reply["warnings"] = ["the matrix is rank deficient; this is the minimum-norm solution"]
            elif operation == "multiply":
                if a.shape[1] != other.shape[0]:
                    raise SciError("invalid_dimensions", f"cannot multiply {a.shape[0]} x {a.shape[1]} by "
                                   f"{' x '.join(str(n) for n in other.shape)}")
                reply["product"] = values(a @ other, limit)
            elif operation == "norm":
                kind = choice(args, "kind", ("frobenius", "2", "1", "inf"), "frobenius")
                order = {"frobenius": "fro", "2": 2, "1": 1, "inf": np.inf}[kind]
                reply.update(norm=number(np.linalg.norm(a, order)), kind=kind)
        except np.linalg.LinAlgError as error:
            raise SciError("singular_matrix" if "ingular" in str(error) else "not_converged", str(error)[:200])
        return reply
    return run


# optimization

def _bounds(args, variables):
    given = args.get("bounds")
    if given is None:
        return None
    if not isinstance(given, dict):
        raise SciError("invalid_arguments", "'bounds' maps a variable to [lower, upper]; null leaves a side open")
    known = {v.name for v in variables}
    pairs = []
    for name in given:
        if name not in known:
            raise SciError("invalid_arguments", f"'bounds' names '{name}', which is not a variable")
    for variable in variables:
        limits = given.get(variable.name, [None, None])
        if not isinstance(limits, list) or len(limits) != 2:
            raise SciError("invalid_arguments", "each entry of 'bounds' is [lower, upper]")
        low, high = (None if v is None else numeric.constant(v, "bounds") for v in limits)
        low = None if low is not None and math.isinf(low) else low
        high = None if high is not None and math.isinf(high) else high
        if low is not None and high is not None and low > high:
            raise SciError("invalid_bounds", f"the lower bound of '{variable.name}' is above its upper bound")
        pairs.append((low, high))
    return pairs


def _minimize(sign):
    def run(args):
        from scipy import optimize
        sp = numeric.sympy()
        expr = _parsed(args, args.get("expression"))
        variables = _variables(args, [expr])
        if args.get("bounds") is None:
            # bounds written as "x": [0, 10] beside the other arguments
            loose = {v.name: args[v.name] for v in variables
                     if isinstance(args.get(v.name), list) and len(args[v.name]) == 2}
            if loose:
                args = {**args, "bounds": loose}
        objective = numeric.compile_expression(sign * expr, variables)
        gradient = [numeric.compile_expression(sign * sp.diff(expr, v), variables) for v in variables]
        bounds = _bounds(args, variables)
        iterations = integer(args, "max_iterations", 500, 1, MAX_ITERATIONS)
        constraints = []
        for text in args.get("constraints") or []:
            relation = numeric.parse_relation(text, _declared(args) + [v.name for v in variables])
            relation = relation.subs(_parameters(args))
            if isinstance(relation, sp.Eq):
                kind, margin = "eq", relation.lhs - relation.rhs
            elif isinstance(relation, (sp.Le, sp.Lt)):
                kind, margin = "ineq", relation.rhs - relation.lhs
            elif isinstance(relation, (sp.Ge, sp.Gt)):
                kind, margin = "ineq", relation.lhs - relation.rhs
            else:
                raise SciError("invalid_arguments", "a constraint is an equation or an inequality")
            compiled = numeric.compile_expression(margin, variables)
            constraints.append({"type": kind, "fun": lambda p, f=compiled: float(f(*p))})
        value = lambda p: float(objective(*p))
        slope = lambda p: np.array([float(g(*p)) for g in gradient])

        start = _guess(args, variables, required=False)
        scalar = len(variables) == 1 and start is None
        if scalar:
            if not bounds or bounds[0][0] is None or bounds[0][1] is None:
                raise SciError("missing_initial_guess", "give 'initial_guess', or 'bounds' with both ends for the "
                               "single variable. A starting point is never guessed for you")
            if constraints:
                raise SciError("invalid_arguments", "constraints need an 'initial_guess'")
            result = optimize.minimize_scalar(lambda x: value([x]), bounds=bounds[0], method="bounded",
                                              options={"maxiter": iterations, "xatol": 1e-10})
            point, method = np.array([result.x]), "bounded Brent"
        else:
            if start is None:
                raise SciError("missing_initial_guess", "'initial_guess' is required: one starting value per "
                               "variable. It is never guessed for you")
            allowed = ("auto", "BFGS", "L-BFGS-B", "Nelder-Mead", "Powell", "SLSQP", "trust-constr")
            method = choice(args, "method", allowed, "auto")
            if method == "auto":
                method = "SLSQP" if constraints else "L-BFGS-B" if bounds else "BFGS"
            if constraints and method not in ("SLSQP", "trust-constr"):
                raise SciError("invalid_arguments", "constraints need the SLSQP or trust-constr method")
            if bounds and method == "BFGS":
                raise SciError("invalid_arguments", "BFGS does not take bounds; use L-BFGS-B")
            if bounds and any((lo is not None and x < lo) or (hi is not None and x > hi)
                              for x, (lo, hi) in zip(start, bounds)):
                raise SciError("invalid_bounds", "the initial guess lies outside the bounds")
            options = {"maxiter": iterations}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = optimize.minimize(value, start, jac=None if method in ("Nelder-Mead", "Powell") else slope,
                                           bounds=bounds, constraints=constraints, method=method, options=options)
            point = result.x
        best = value(point) if np.all(np.isfinite(point)) else math.nan
        if not math.isfinite(best):
            raise SciError("not_converged", "the search ran into values where the function is not finite")
        reply = {"solution": _named(variables, point), "objective": number(sign * best), "method": method,
                 "iterations": int(getattr(result, "nit", 0)), "evaluations": int(result.nfev),
                 "termination": str(result.message)[:160]}
        if not result.success:
            raise SciError("not_converged", "the optimizer stopped without converging: " + str(result.message)[:160],
                           best=reply)
        if not scalar and not constraints:
            norm = float(np.abs(slope(point)).max())
            interior = bounds is None or all((lo is None or x > lo + 1e-9) and (hi is None or x < hi - 1e-9)
                                             for x, (lo, hi) in zip(point, bounds))
            if interior and norm > 1e-4 * max(1.0, abs(best)):
                reply["warnings"] = ["the gradient is not zero here; this may not be a true optimum"]
        if bounds is not None:
            held = [v.name for v, x, (lo, hi) in zip(variables, point, bounds)
                    if (lo is not None and x <= lo + 1e-9) or (hi is not None and x >= hi - 1e-9)]
            if held:
                reply.setdefault("warnings", []).append(
                    f"'{held[0]}' ended on one of its bounds; without that bound the optimum may be elsewhere")
        reply["note"] = "a local optimum near the start; other optima may exist"
        return reply
    return run


def _curve_fit(args):
    from scipy import optimize
    x, y = array(args.get("x"), "x", (1,)), array(args.get("y"), "y", (1,))
    if x.shape != y.shape:
        raise SciError("invalid_dimensions", "'x' and 'y' must have the same length")
    guess = args.get("initial_guess")
    notes = []
    if not isinstance(guess, dict) or not guess:
        sp = numeric.sympy()
        model = _parsed(args, args.get("expression"))
        data = numeric.symbols([args.get("variable") or "x"])[0]
        free = sorted(model.free_symbols - {data}, key=lambda symbol: symbol.name)
        linear = free and data in model.free_symbols and all(
            sp.diff(model, a, b) == 0 for a in free for b in free)
        if not linear:
            raise SciError("missing_initial_guess", "'initial_guess' is required: an object naming each parameter "
                           "of the model with a starting value, e.g. {\"a\": 1, \"b\": 0.1}")
        # one least-squares solution, whatever the start
        guess = {symbol.name: 1 for symbol in free}
        notes = ["the model is linear in its parameters, so no initial guess was needed"]
    parameters = numeric.symbols(guess, "initial_guess")
    start = np.array([numeric.constant(v, "initial_guess") for v in guess.values()])
    if x.size <= len(parameters):
        raise SciError("invalid_arguments", f"{len(parameters)} parameters need more than {x.size} data points")
    expr = _parsed(args, args.get("expression"))
    independent = sorted(expr.free_symbols - set(parameters), key=lambda s: s.name)
    if args.get("variable") is not None:
        independent = numeric.symbols([args["variable"]], "variable")
    if len(independent) != 1:
        raise SciError("invalid_arguments", "the model needs exactly one independent variable besides its parameters")
    model = numeric.compile_expression(expr, independent + parameters)
    function = lambda data, *p: np.asarray(model(data, *p), dtype=float) + np.zeros_like(data)
    with warnings.catch_warnings():
        warnings.simplefilter("error", optimize.OptimizeWarning)
        try:
            fitted, covariance = optimize.curve_fit(function, x, y, p0=start, maxfev=MAX_ITERATIONS * len(parameters))
        except (RuntimeError, optimize.OptimizeWarning, ValueError) as error:
            raise SciError("not_converged", "the fit did not converge: " + str(error)[:160])
    residual = y - function(x, *fitted)
    _finite(fitted, "the fit")
    total = float(((y - y.mean()) ** 2).sum())
    errors = np.sqrt(np.diag(covariance))
    reply = {"parameters": _named(parameters, fitted), "r_squared": number(1 - float((residual ** 2).sum()) / total)
             if total > 0 else None, "rms_residual": number(math.sqrt(float((residual ** 2).mean()))),
             "points": int(x.size)}
    if np.all(np.isfinite(errors)):
        reply["standard_errors"] = _named(parameters, errors)
    else:
        notes.append("the parameter uncertainties could not be estimated")
    reply["warnings"] = notes
    return reply


def _least_squares(args):
    from scipy import optimize
    sp = numeric.sympy()
    texts = args.get("residuals")
    if not isinstance(texts, list) or not texts or len(texts) > 256:
        raise SciError("invalid_arguments", "'residuals' is a list of expressions to drive to zero")
    residuals = [_parsed(args, text, equation=True) for text in texts]
    variables = _variables(args, residuals)
    start = _guess(args, variables)
    functions = [numeric.compile_expression(r, variables) for r in residuals]
    jacobian = [[numeric.compile_expression(sp.diff(r, v), variables) for v in variables] for r in residuals]
    bounds = _bounds(args, variables)
    limits = (-np.inf, np.inf) if bounds is None else (
        [-np.inf if lo is None else lo for lo, _ in bounds], [np.inf if hi is None else hi for _, hi in bounds])
    try:
        result = optimize.least_squares(lambda p: np.array([float(f(*p)) for f in functions]), start,
                                        jac=lambda p: np.array([[float(d(*p)) for d in row] for row in jacobian]),
                                        bounds=limits, max_nfev=integer(args, "max_iterations", 500, 1, MAX_ITERATIONS))
    except ValueError as error:
        raise SciError("invalid_bounds" if "bound" in str(error) or "feasible" in str(error) else "not_converged",
                       str(error)[:200])
    if not result.success:
        raise SciError("not_converged", "the solver stopped without converging: " + str(result.message)[:160])
    return {"solution": _named(variables, result.x), "cost": number(result.cost),
            "residual_norm": number(np.linalg.norm(result.fun)), "evaluations": int(result.nfev),
            "termination": str(result.message)[:160]}


# signals

def _signal(args, name="values"):
    """Samples given outright, or generated from an expression in t over a duration."""
    rate = real(args, "sample_rate", low=1e-12)
    given = args.get(name)
    if isinstance(given, list) and len(given) == 1 and isinstance(given[0], str):
        given = given[0]
    if isinstance(given, str):
        # a formula written where the samples go
        args, given = {**args, "expression": given}, None
    if given is not None:
        return array(given, name, (1,)), rate
    if args.get("expression") is None:
        raise SciError("invalid_arguments", f"'{name}' is required, or an 'expression' in t with 'sample_rate' and 'duration'")
    if rate is None or args.get("duration") is None:
        raise SciError("invalid_arguments", "sampling an expression needs 'sample_rate' and 'duration'")
    count = int(round(rate * real(args, "duration", low=0)))
    if not 2 <= count <= MAX_FFT:
        raise SciError("limit_exceeded", f"that is {count} samples; the limit is {MAX_FFT}", limit=MAX_FFT)
    expr = _parsed(args, args["expression"])
    variable = numeric.symbols([args.get("variable") or "t"])
    samples = np.asarray(numeric.compile_expression(expr, variable)(np.arange(count) / rate), dtype=float)
    return _finite(samples + np.zeros(count), "the sampled signal"), rate


def _spectrum_input(args):
    if args.get("cutoff") is not None and args.get("sample_rate") is None:
        raise SciError("invalid_arguments", "a spectrum takes 'sample_rate' in Hz, not 'cutoff'")
    return _signal(args)


def _fft(args):
    from scipy import fft, signal
    samples, rate = _spectrum_input(args)
    if samples.size > MAX_FFT:
        raise SciError("limit_exceeded", f"{samples.size} samples; the FFT limit is {MAX_FFT}", limit=MAX_FFT)
    if samples.size < 2:
        raise SciError("invalid_arguments", "at least two samples are needed")
    window = choice(args, "window", ("none", "hann", "hamming", "blackman"), "none")
    weights = np.ones(samples.size) if window == "none" else signal.get_window(window, samples.size)
    spectrum = fft.rfft((samples - samples.mean()) * weights)
    amplitude = np.abs(spectrum) * 2 / weights.sum()
    frequencies = fft.rfftfreq(samples.size, d=1 / rate if rate else 1.0)
    top = integer(args, "top", 5, 1, MAX_PEAKS)
    candidates, _ = signal.find_peaks(np.concatenate(([0.0], amplitude, [0.0])))
    candidates = candidates - 1
    if amplitude.max() > 0:
        candidates = candidates[amplitude[candidates] >= 1e-6 * amplitude.max()]
    strongest = candidates[np.argsort(-amplitude[candidates], kind="stable")][:top]
    unit = "hz" if rate else "cycles_per_sample"
    reply = {"n": int(samples.size), "resolution": number(frequencies[1]), "mean": number(samples.mean()),
             "dominant_frequencies": [{unit: number(frequencies[i]), "amplitude": number(amplitude[i]),
                                       "phase": number(np.angle(spectrum[i]))} for i in strongest]}
    if rate:
        reply["sample_rate"] = number(rate)
    else:
        reply["note"] = "no sample_rate was given; frequencies are in cycles per sample"
    if window != "none":
        reply["window"] = window
    if args.get("max_values") is not None:
        reply["frequencies"] = values(frequencies, listed_limit(args))
        reply["amplitudes"] = values(amplitude, listed_limit(args))
    return reply


def _ifft(args):
    from scipy import fft
    real_part = array(args.get("real"), "real", (1,))
    imaginary = array(args.get("imag"), "imag", (1,), required=False)
    if imaginary is not None and imaginary.shape != real_part.shape:
        raise SciError("invalid_dimensions", "'real' and 'imag' must have the same length")
    if real_part.size > MAX_FFT:
        raise SciError("limit_exceeded", f"{real_part.size} bins; the FFT limit is {MAX_FFT}", limit=MAX_FFT)
    spectrum = real_part + 1j * (imaginary if imaginary is not None else 0)
    return {"n": int(spectrum.size), "values": values(fft.ifft(spectrum), listed_limit(args))}


def _psd(args):
    from scipy import signal
    samples, rate = _signal(args)
    if rate is None:
        raise SciError("invalid_arguments", "'sample_rate' is required for a power spectral density")
    if samples.size < 8:
        raise SciError("invalid_arguments", "at least 8 samples are needed")
    segment = integer(args, "segment", min(256, samples.size), 8, samples.size)
    frequencies, power = signal.welch(samples, fs=rate, nperseg=segment)
    top = integer(args, "top", 5, 1, MAX_PEAKS)
    candidates, _ = signal.find_peaks(np.concatenate(([0.0], power, [0.0])))
    candidates = candidates - 1
    strongest = candidates[np.argsort(-power[candidates], kind="stable")][:top]
    return {"n": int(samples.size), "sample_rate": number(rate), "segment": segment, "method": "welch",
            "resolution": number(frequencies[1]), "total_power": number(np.trapezoid(power, frequencies)),
            "peaks": [{"hz": number(frequencies[i]), "power_density": number(power[i])} for i in strongest]}


def _filter(kind):
    return lambda args: _apply_filter(args, kind)


def _apply_filter(args, kind):
    from scipy import signal
    rate = real(args, "sample_rate", required=True, low=1e-12)
    cutoff = args.get("cutoff")
    band = kind in ("bandpass", "bandstop")
    if band:
        if not isinstance(cutoff, list) or len(cutoff) != 2:
            raise SciError("invalid_arguments", f"a {kind} filter needs 'cutoff' as [low, high] in Hz")
        edges = sorted(numeric.constant(c, "cutoff") for c in cutoff)
    else:
        if isinstance(cutoff, list) and len(cutoff) == 1:
            cutoff = cutoff[0]
        if cutoff is None or isinstance(cutoff, list):
            raise SciError("invalid_arguments", f"a {kind} filter needs one 'cutoff' frequency in Hz")
        edges = [numeric.constant(cutoff, "cutoff")]
    if any(not 0 < edge < rate / 2 for edge in edges) or (band and edges[0] == edges[1]):
        raise SciError("invalid_arguments", f"cutoff frequencies must lie between 0 and the Nyquist frequency "
                       f"({number(rate / 2)} Hz)")
    order = integer(args, "order", 4, 1, 12)
    sections = signal.butter(order, edges if band else edges[0], btype=kind, fs=rate, output="sos")
    probe = sorted({rate / 2 * 1e-3, *edges, min(rate / 2 * 0.999, edges[-1] * 2), edges[0] / 2})
    _, response = signal.sosfreqz(sections, worN=np.array(probe), fs=rate)
    reply = {"filter": f"Butterworth {kind}, order {order}", "cutoff_hz": [number(e) for e in edges],
             "sample_rate": number(rate),
             "gain": [{"hz": number(f), "gain": number(abs(g))} for f, g in zip(probe, response)]}
    if args.get("values") is None and args.get("expression") is None:
        reply["note"] = "filter design only; give 'values' to apply it"
        return reply
    samples, _ = _signal(args)
    if samples.size <= 3 * (2 * len(sections) + 1):
        raise SciError("invalid_arguments", f"the signal is too short for an order {order} filter")
    filtered = signal.sosfiltfilt(sections, samples)
    reply.update(n=int(samples.size), values=values(filtered, listed_limit(args)),
                 rms_before=number(math.sqrt(float((samples ** 2).mean()))),
                 rms_after=number(math.sqrt(float((filtered ** 2).mean()))),
                 note="applied forward and backward (zero phase), so the gains above act twice")
    return reply


def _peaks(args):
    from scipy import signal
    samples, rate = _signal(args)
    options = {}
    for key in ("height", "prominence", "threshold"):
        if args.get(key) is not None:
            options[key] = real(args, key)
    if args.get("distance") is not None:
        options["distance"] = integer(args, "distance", 1, 1, samples.size)
    indices, _ = signal.find_peaks(samples, **options)
    order = indices[np.argsort(-samples[indices], kind="stable")][:MAX_PEAKS]
    order = np.sort(order)
    reply = {"n": int(samples.size), "count": int(indices.size),
             "peaks": [{"index": int(i), "value": number(samples[i]),
                        **({"time": number(i / rate)} if rate else {})} for i in order]}
    if indices.size > MAX_PEAKS:
        reply["note"] = f"the {MAX_PEAKS} highest of {indices.size} peaks are listed"
    return reply


def _combine(operation):
    def run(args):
        from scipy import signal
        both = args.get("values")
        if args.get("other") is None and isinstance(both, list) and len(both) == 2 and all(isinstance(v, list) for v in both):
            args = {**args, "values": both[0], "other": both[1]}
        first, second = array(args.get("values"), "values", (1,)), array(args.get("other"), "other", (1,))
        if first.size * second.size > 1 << 34:
            raise SciError("limit_exceeded", "those two signals are too long to combine")
        mode = choice(args, "mode", ("full", "same", "valid"), "full")
        function = signal.convolve if operation == "convolve" else signal.correlate
        result = function(first, second, mode=mode)
        reply = {"n": int(result.size), "mode": mode, "values": values(result, listed_limit(args))}
        if operation == "correlate":
            lags = signal.correlation_lags(first.size, second.size, mode=mode)
            best = int(np.argmax(np.abs(result)))
            reply.update(best_lag=int(lags[best]), value_at_best_lag=number(result[best]))
        return reply
    return run


# ordinary differential equations

_CONDITION = re.compile(r"^\s*([A-Za-z]\w*)\s*('*)\s*(?:\(\s*([^)]*)\s*\))?\s*$")
_LEIBNIZ_CONDITION = re.compile(r"^\s*d(\d?)([A-Za-z]\w*?)/d[A-Za-z]\w*?\d?\s*(?:\(\s*([^)]*)\s*\))?\s*$")
_UNKNOWN = re.compile(r"\bd\d?([A-Za-z]\w*?)/d[A-Za-z]|\b([A-Za-z]\w*)\s*'")


def _ode(args):
    from scipy import integrate
    sp = numeric.sympy()
    texts = args.get("equations")
    if isinstance(texts, str):
        texts = [texts]
    conditions = args.get("initial_conditions")
    if not isinstance(texts, list) or not texts or len(texts) > 16:
        raise SciError("invalid_arguments", "'equations' is a list such as [\"dy/dt = -0.5*y\"]")
    if conditions is None:
        conditions = {}
    if not isinstance(conditions, dict):
        raise SciError("invalid_initial_conditions", "'initial_conditions' is an object such as {\"y\": 1}")
    interval = args.get("interval")
    if not isinstance(interval, list) or len(interval) != 2:
        raise SciError("invalid_arguments", "'interval' is [start, end] of the independent variable")
    start, end = (numeric.constant(v, "interval") for v in interval)
    if not (math.isfinite(start) and math.isfinite(end)) or start == end:
        raise SciError("invalid_arguments", "'interval' needs two different finite ends")

    # conditions written among the equations, as y(0) = 1
    kept = []
    for text in texts:
        left, _, right = str(text).partition("=")
        if _CONDITION.match(left) and "(" in left and right and "=" not in right:
            try:
                conditions = {**conditions, left.strip(): float(right)}
                continue
            except ValueError:
                pass
        kept.append(text)
    texts = kept or texts
    # a model reducing the order itself writes dy'/dt for y''
    texts = [re.sub(r"\bd([A-Za-z]\w*)('+)/d[A-Za-z]\w*", r"\1\2'", text) if isinstance(text, str) else text
             for text in texts]
    if not conditions:
        raise SciError("invalid_initial_conditions", "'initial_conditions' is required, e.g. {\"y\": 1}; a "
                       "numerical solution needs a starting value for every unknown")
    given = {}
    for key, value in conditions.items():
        leibniz = _LEIBNIZ_CONDITION.match(str(key))
        match = _CONDITION.match(str(key))
        if leibniz is not None:
            name, primes, point = leibniz.group(2), "'" * int(leibniz.group(1) or 1), leibniz.group(3)
        elif match is not None:
            name, primes, point = match.groups()
        else:
            raise SciError("invalid_initial_conditions", f"cannot read the condition '{str(key)[:40]}'")
        if point not in (None, "") and abs(numeric.constant(point, "initial_conditions") - start) > 1e-12 * max(1, abs(start)):
            raise SciError("invalid_initial_conditions", f"'{key}' is not at the start of the interval ({number(start)}); "
                           "only initial value problems are solved")
        given[(name, len(primes))] = numeric.constant(value, "initial_conditions")
    written = " ".join(str(t) for t in texts)
    names = sorted({name for name, _ in given} | {a or b for a, b in _UNKNOWN.findall(written)})
    variable_name = args.get("variable")
    if variable_name is None:
        leibniz = re.search(r"\bd\d?[A-Za-z]\w*/d([A-Za-z]\w*?)\d?\b", " ".join(str(t) for t in texts))
        variable_name = leibniz.group(1) if leibniz else "t"
    if variable_name in names:
        raise SciError("invalid_arguments", f"'{variable_name}' is both the variable and an unknown")
    functions = {name: variable_name for name in names}
    variable = numeric.symbols([variable_name])[0]
    constants = _parameters(args)
    equations = [numeric.parse_equation(text, _declared(args), functions).subs(constants) for text in texts]
    applied = {name: sp.Function(name)(variable) for name in names}

    restated = [equation for equation in equations if sp.simplify(equation) == 0]
    if len(restated) == len(equations):
        raise SciError("invalid_arguments", "the equations only restate a derivative; write the differential "
                       "equation itself, such as y'' + y = 0")
    equations = [equation for equation in equations if equation not in restated]
    orders = {name: 0 for name in names}
    for equation in equations:
        for derivative in equation.atoms(sp.Derivative):
            function = derivative.expr
            if function not in applied.values() or derivative.variables != (variable,) * len(derivative.variables):
                raise SciError("not_supported", "only ordinary derivatives of the declared unknowns are supported")
            orders[function.func.__name__] = max(orders[function.func.__name__], len(derivative.variables))
    if any(order == 0 for order in orders.values()):
        missing = [name for name, order in orders.items() if order == 0][0]
        raise SciError("invalid_arguments", f"no equation contains a derivative of '{missing}'")
    highest = [sp.Derivative(applied[name], (variable, orders[name])) for name in names]
    placeholders = sp.symbols(f"tosh_h0:{len(names)}")
    isolated = [equation.subs(dict(zip(highest, placeholders))) for equation in equations]
    solved = sp.solve(isolated, placeholders, dict=True)
    if len(solved) != 1:
        raise SciError("not_supported", "the equations cannot be solved uniquely for the highest derivatives")

    state, replace = [], {}
    for name in names:
        for order in range(orders[name]):
            symbol = sp.Symbol(f"tosh_s_{name}_{order}")
            state.append((name, order, symbol))
            replace[sp.Derivative(applied[name], (variable, order)) if order else applied[name]] = symbol
    lower_first = sorted(replace.items(), key=lambda item: -len(getattr(item[0], "variables", ())))
    symbols = [variable] + [symbol for _, _, symbol in state]
    right = []
    for name, order, _ in state:
        if order + 1 < orders[name]:
            right.append(sp.Symbol(f"tosh_s_{name}_{order + 1}"))
        else:
            right.append(solved[0][placeholders[names.index(name)]].subs(lower_first))
    for expr in right:
        if expr.atoms(sp.Derivative) or any(f in expr.atoms(sp.Function) for f in applied.values()):
            raise SciError("not_supported", "the equations could not be reduced to a first order system")
    compiled = [numeric.compile_expression(expr, symbols) for expr in right]
    missing = [(name, order) for name, order, _ in state if (name, order) not in given]
    if missing:
        name, order = missing[0]
        raise SciError("invalid_initial_conditions", f"an initial value for {name}{chr(39) * order} is missing; "
                       f"an equation of order {orders[name]} in {name} needs {orders[name]} of them")
    extra = [key for key in given if key not in {(name, order) for name, order, _ in state}]
    if extra:
        raise SciError("invalid_initial_conditions", f"the condition on {extra[0][0]}{chr(39) * extra[0][1]} does "
                       "not match the order of the equations")
    initial = np.array([given[(name, order)] for name, order, _ in state])

    method = choice(args, "method", ("RK45", "DOP853", "Radau", "BDF", "LSODA"), "RK45")
    tolerance = real(args, "tolerance", 1e-8, low=1e-13, high=1e-2)
    at = args.get("at")
    if at is not None:
        times = array(at, "at", (1,))
        if times.size > MAX_ODE_POINTS:
            raise SciError("limit_exceeded", f"'at' has {times.size} points; the limit is {MAX_ODE_POINTS}", limit=MAX_ODE_POINTS)
        low, high = min(start, end), max(start, end)
        if np.any((times < low) | (times > high)):
            raise SciError("out_of_range", "'at' has points outside the interval")
        times = np.sort(times) if end > start else np.sort(times)[::-1]
    else:
        times = np.linspace(start, end, integer(args, "points", 11, 2, MAX_ODE_POINTS))

    def derivative(t, y):
        result = np.array([float(f(t, *y)) for f in compiled])
        if not np.all(np.isfinite(result)):
            raise SciError("non_finite_result", f"the equations are not finite at {variable_name} = {number(t)}")
        return result

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solution = integrate.solve_ivp(derivative, (start, end), initial, method=method, t_eval=times,
                                       rtol=tolerance, atol=tolerance * 1e-3)
    if not solution.success:
        raise SciError("integration_failed", "the solver stopped before the end of the interval: "
                       + str(solution.message)[:160],
                       reached=number(solution.t[-1]) if solution.t.size else number(start))
    limit = listed_limit(args)
    reply = {variable_name: values(solution.t, limit), "solution": {}, "final": {},
             "method": method, "tolerance": tolerance, "evaluations": int(solution.nfev)}
    for row, (name, order, _) in zip(solution.y, state):
        label = name + "'" * order
        if order == 0 or args.get("derivatives") is True:
            reply["solution"][label] = values(row, limit)
        reply["final"][label] = number(row[-1])
    return reply


# statistics

def _sample(args, name="values", minimum=2):
    data = array(args.get(name), name, (1,))
    if data.size < minimum:
        raise SciError("invalid_arguments", f"'{name}' needs at least {minimum} values")
    return data


def _describe(args):
    from scipy import stats
    data = _sample(args, minimum=1)
    reply = {"n": int(data.size), "mean": number(data.mean()), "median": number(np.median(data)),
             "min": number(data.min()), "max": number(data.max()),
             "q1": number(np.percentile(data, 25)), "q3": number(np.percentile(data, 75)),
             "sum": number(data.sum())}
    if data.size > 1:
        reply.update(std=number(data.std(ddof=1)), variance=number(data.var(ddof=1)),
                     note="std and variance are the sample values (n - 1)")
    if data.size > 2 and data.std() > 0:
        reply.update(skewness=number(stats.skew(data)), kurtosis=number(stats.kurtosis(data)))
    return reply


def _percentile(args):
    if args.get("name") is not None and args.get("q") is not None:
        # a percentile of a named distribution is its quantile
        return _distribution({**args, "p": real(args, "q", low=0, high=100) / 100})
    data = _sample(args, minimum=1)
    q = args.get("q")
    wanted = array(q if isinstance(q, list) else [q], "q", (1,))
    if np.any((wanted < 0) | (wanted > 100)):
        raise SciError("invalid_arguments", "'q' is between 0 and 100")
    result = np.percentile(data, wanted)
    return {"n": int(data.size), "percentiles": [{"q": number(a), "value": number(b)} for a, b in zip(wanted, result)]}


def _pair(args):
    x, y = _sample(args, "x", 3), _sample(args, "y", 3)
    if x.shape != y.shape:
        raise SciError("invalid_dimensions", "'x' and 'y' must have the same length")
    if x.std() == 0 or y.std() == 0:
        raise SciError("invalid_arguments", "one of the series is constant, so the result is undefined")
    return x, y


def _correlation(args):
    from scipy import stats
    x, y = _pair(args)
    method = choice(args, "method", ("pearson", "spearman"), "pearson")
    result = stats.pearsonr(x, y) if method == "pearson" else stats.spearmanr(x, y)
    return {"r": number(result.statistic), "p_value": number(result.pvalue), "n": int(x.size), "method": method}


def _regression(args):
    from scipy import stats
    x, y = _pair(args)
    fit = stats.linregress(x, y)
    return {"slope": number(fit.slope), "intercept": number(fit.intercept), "r": number(fit.rvalue),
            "r_squared": number(fit.rvalue ** 2), "p_value": number(fit.pvalue),
            "slope_standard_error": number(fit.stderr), "n": int(x.size)}


def _ttest(args):
    from scipy import stats
    both = args.get("values")
    if args.get("x") is None and isinstance(both, list) and len(both) == 2 and all(isinstance(g, list) for g in both):
        # two samples written as one list of two
        args = {**args, "x": both[0], "y": both[1], "values": None}
    x = _sample(args, "x" if args.get("x") is not None else "values")
    alternative = choice(args, "alternative", ("two-sided", "less", "greater"), "two-sided")
    if args.get("y") is not None:
        y = _sample(args, "y")
        if args.get("paired") is True:
            if x.shape != y.shape:
                raise SciError("invalid_dimensions", "a paired test needs 'x' and 'y' of the same length")
            result, kind = stats.ttest_rel(x, y, alternative=alternative), "paired"
        else:
            equal = args.get("equal_variance") is True
            result = stats.ttest_ind(x, y, equal_var=equal, alternative=alternative)
            kind = "two-sample, equal variances" if equal else "two-sample, Welch"
    else:
        if args.get("mean") is None:
            raise SciError("invalid_arguments", "a one-sample test needs 'mean', the value to test against; "
                           "or give 'y' for two samples")
        result = stats.ttest_1samp(x, real(args, "mean"), alternative=alternative)
        kind = "one-sample"
    if not math.isfinite(float(result.statistic)):
        raise SciError("invalid_arguments", "the samples have no variation, so the test is undefined")
    return {"statistic": number(result.statistic), "p_value": number(result.pvalue),
            "degrees_of_freedom": number(result.df), "test": kind, "alternative": alternative}


def _confidence_interval(args):
    from scipy import stats
    data = _sample(args)
    level = real(args, "level", 0.95, low=0.5, high=0.9999)
    error = stats.sem(data)
    if error == 0:
        raise SciError("invalid_arguments", "the sample has no variation")
    low, high = stats.t.interval(level, data.size - 1, loc=data.mean(), scale=error)
    return {"mean": number(data.mean()), "lower": number(low), "upper": number(high), "level": level,
            "n": int(data.size), "method": "Student t"}


def _distribution(args):
    from scipy import stats
    name = choice(args, "name", ("normal", "t", "chi2", "exponential", "uniform", "binomial", "poisson"), None)
    if name is None:
        raise SciError("invalid_arguments", "'name' is required: normal, t, chi2, exponential, uniform, "
                       "binomial or poisson")
    given = args.get("parameters") or {}
    if not isinstance(given, dict):
        raise SciError("invalid_arguments", "'parameters' is an object such as {\"mean\": 0, \"std\": 1}")
    p = {key: numeric.constant(value, "parameters") for key, value in given.items()}

    def need(key, default=None, positive=False):
        if key not in p and default is None:
            raise SciError("invalid_arguments", f"the {name} distribution needs '{key}' in 'parameters'")
        value = p.get(key, default)
        if positive and value <= 0:
            raise SciError("invalid_arguments", f"'{key}' must be positive")
        return value

    if name == "normal":
        dist = stats.norm(need("mean", 0.0), need("std", 1.0, positive=True))
    elif name == "t":
        dist = stats.t(need("df", positive=True))
    elif name == "chi2":
        dist = stats.chi2(need("df", positive=True))
    elif name == "exponential":
        dist = stats.expon(scale=1 / need("rate", positive=True))
    elif name == "uniform":
        low, high = need("low", 0.0), need("high", 1.0)
        if high <= low:
            raise SciError("invalid_arguments", "'high' must be above 'low'")
        dist = stats.uniform(low, high - low)
    elif name == "binomial":
        n, probability = need("n"), need("p")
        if n != int(n) or n < 0 or not 0 <= probability <= 1:
            raise SciError("invalid_arguments", "'n' is a non-negative integer and 'p' is between 0 and 1")
        dist = stats.binom(int(n), probability)
    else:
        dist = stats.poisson(need("rate", positive=True))
    discrete = name in ("binomial", "poisson")
    reply = {"distribution": name, "mean": number(dist.mean()), "std": number(dist.std())}
    asked = False
    point = args.get("at") if args.get("at") is not None else args.get("x")
    if isinstance(point, list) and len(point) == 1:
        point = point[0]
    if point is not None:
        x = real({"at": point}, "at")
        reply.update({"pmf" if discrete else "pdf": number(dist.pmf(x) if discrete else dist.pdf(x)),
                      "cdf": number(dist.cdf(x)), "upper_tail": number(dist.sf(x)), "x": number(x)})
        asked = True
    if args.get("p") is not None:
        probability = real(args, "p", low=0, high=1)
        reply.update(quantile=number(dist.ppf(probability)), p=probability)
        asked = True
    if args.get("between") is not None:
        between = args["between"]
        if not isinstance(between, list) or len(between) != 2:
            raise SciError("invalid_arguments", "'between' is [a, b]")
        a, b = sorted(numeric.constant(v, "between") for v in between)
        lower = dist.cdf(a - 1) if discrete else dist.cdf(a)
        reply.update(probability_between=number(dist.cdf(b) - lower), between=[number(a), number(b)])
        asked = True
    if not asked:
        raise SciError("invalid_arguments", "give 'at' for the density and cumulative probability, 'p' for a "
                       "quantile, or 'between' for the probability of an interval")
    return reply


OPERATIONS = {
    "integrate": _integrate, "root": _root, "interpolate": _interpolate, "evaluate": _evaluate,
    **{name: _linalg(name) for name in ("solve", "inverse", "determinant", "rank", "condition", "eigen", "svd",
                                        "qr", "least_squares", "multiply", "norm")},
    "minimize": _minimize(1), "maximize": _minimize(-1), "curve_fit": _curve_fit, "fit_residuals": _least_squares,
    "fft": _fft, "ifft": _ifft, "psd": _psd, "peaks": _peaks,
    **{kind: _filter(kind) for kind in ("lowpass", "highpass", "bandpass", "bandstop")},
    "convolve": _combine("convolve"), "correlate": _combine("correlate"),
    "solve_ivp": _ode,
    "describe": _describe, "percentile": _percentile, "correlation": _correlation, "regression": _regression,
    "ttest": _ttest, "confidence_interval": _confidence_interval, "distribution": _distribution,
}


def failure(operation, code, message, **extra):
    return {"success": False, "operation": operation, "error": {"code": code, "message": message}, **extra}


def run(operation, arguments):
    handler = OPERATIONS.get(operation) if isinstance(operation, str) else None
    if handler is None:
        return failure(str(operation)[:40], "unknown_operation", "'operation' must be one of: " + ", ".join(OPERATIONS))
    if not isinstance(arguments, dict):
        return failure(operation, "invalid_arguments", "arguments must be an object")
    # models fill the arguments they do not need with empty placeholders
    arguments = {key: value for key, value in arguments.items() if value not in ([], {}, "")}
    try:
        result = handler(arguments)
        warned = result.pop("warnings", [])
        return {"success": True, "operation": operation, **result, "warnings": warned}
    except SciError as error:
        return failure(operation, error.code, error.message, **error.extra)
    except MemoryError:
        raise
    except RecursionError:
        return failure(operation, "too_large", "the expression is too complex to process")
    except (ValueError, TypeError, ArithmeticError, IndexError, KeyError, np.linalg.LinAlgError) as error:
        return failure(operation, "numerical_error", f"{type(error).__name__}: {str(error)[:300]}")


def after_timeout(operation, arguments, seconds, emit):
    return {"success": False, "operation": operation, "timed_out": True, "warnings": [],
            "error": {"code": "timeout", "message":
                      f"The calculation did not finish within the {seconds:g} s computation budget."}}
