# ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
# Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the SymPy helper, driven over its MCP stdio protocol.

Run with the bundled interpreter, which has no unittest:
    vendor/tosh-sympy/python/bin/python3 -I helpers/tosh-sympy/tests/test_helper.py [runtime dir]
"""

import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUNTIME = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(ROOT, "vendor", "tosh-sympy")
PYTHON = os.path.join(RUNTIME, "python", "bin", "python3")


# the key the app gives the helpers, so a test can vouch for a request the way the app does
KEY = "test-trust-key"
# CI runners are virtual and run this x86 runtime translated: wall-clock limits get three times the room
SLOW = 3 if os.environ.get("CI") else 1
SHARED = {"TOSH_SYMPY_TIMEOUT_MS": 20000} if SLOW > 1 else {}
# a budget the next small calculation fits in; the heavy ones take seconds even on a fast machine
SHORT = 100*SLOW


class Helper:
    def __init__(self, **environment):
        self.process = subprocess.Popen(
            [PYTHON, "-I", "-B", os.path.join(RUNTIME, "tosh_sympy", "server.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            env={"TOSH_TRUST_KEY": KEY, **{key: str(value) for key, value in environment.items()}})
        self.next_id = 0
        self.rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}})

    def rpc(self, method, params=None):
        self.next_id += 1
        message = {"jsonrpc": "2.0", "id": self.next_id, "method": method}
        if params is not None:
            message["params"] = params
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()
        reply = json.loads(self.process.stdout.readline())
        assert reply["id"] == self.next_id, reply
        return reply

    def call(self, tool, **arguments):
        result = self.rpc("tools/call", {"name": tool, "arguments": arguments})["result"]
        reply = json.loads(result["content"][0]["text"])
        assert result["isError"] == (not reply["success"] and "timed_out" not in reply), result
        return reply

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=10)


def expression(helper, operation, text, **arguments):
    return helper.call("expression", operation=operation, expression=text, **arguments)


def error_code(reply):
    assert reply["success"] is False, reply
    assert {"success", "operation", "error"} <= set(reply), reply
    assert set(reply["error"]) == {"code", "message"}, reply
    return reply["error"]["code"]


def test_simplify(h):
    assert expression(h, "simplify", "(x**2 - 1)/(x - 1)")["exact"] == "x + 1"


def test_factor(h):
    reply = expression(h, "factor", "x**2 - 5*x + 6")
    assert reply["exact"] == "(x - 3)*(x - 2)", reply
    assert reply["latex"] and reply["warnings"] == [] and reply["operation"] == "factor"


def test_expand_cancel_apart_together(h):
    assert expression(h, "expand", "(x + 1)**2")["exact"] == "x**2 + 2*x + 1"
    assert expression(h, "cancel", "(x**2 - 1)/(x - 1)")["exact"] == "x + 1"
    assert expression(h, "apart", "1/(x**2 - 1)")["exact"] == "-1/(2*(x + 1)) + 1/(2*(x - 1))"
    assert expression(h, "together", "1/x + 1/y")["exact"] == "(x + y)/(x*y)"


def test_solve(h):
    reply = h.call("solve", operation="solve", equations=["x**2 - 5*x + 6 = 0"])
    assert reply["solutions"] == [{"x": "2"}, {"x": "3"}], reply


def test_solve_exact_roots(h):
    reply = h.call("solve", operation="solve", equations=["x**2 - 2 = 0"])
    assert reply["solutions"] == [{"x": "-sqrt(2)"}, {"x": "sqrt(2)"}], reply
    assert reply["numeric"][1]["x"].startswith("1.41421356"), reply


def test_solve_system(h):
    reply = h.call("solve", operation="solve", equations=["x + y = 3", "x - y = 1"])
    assert reply["solutions"] == [{"x": "2", "y": "1"}], reply


def test_solve_inequality(h):
    reply = h.call("solve", operation="solve", equations=["x**2 < 4"])
    assert reply["set"] == "Interval.open(-2, 2)", reply


def test_solveset(h):
    reply = h.call("solve", operation="solveset", equations=["x**2 + 1 = 0"], domain="real")
    assert reply["exact"] == "EmptySet", reply
    reply = h.call("solve", operation="solveset", equations=["x**2 + 1 = 0"])
    assert reply["solutions"] == [{"x": "-I"}, {"x": "I"}], reply


def test_derivative(h):
    assert expression(h, "differentiate", "x**2*sin(x)")["exact"] == "x**2*cos(x) + 2*x*sin(x)"
    assert expression(h, "differentiate", "x**5", order=2)["exact"] == "20*x**3"


def test_integral(h):
    reply = expression(h, "integrate", "x**2*sin(x)", variable="x")
    assert reply["exact"] == "-x**2*cos(x) + 2*x*sin(x) + 2*cos(x)", reply
    reply = expression(h, "integrate", "exp(-x**2)", lower="-oo", upper="oo")
    assert reply["exact"] == "sqrt(pi)" and reply["numeric"].startswith("1.7724538509"), reply


def test_limit(h):
    assert expression(h, "limit", "sin(x)/x", point="0")["exact"] == "1"
    assert expression(h, "limit", "(1 + 1/n)**n", point="oo")["exact"] == "E"
    assert expression(h, "limit", "1/x", point=0, direction="+")["exact"] == "oo"


def test_series(h):
    reply = expression(h, "series", "cos(x)", order=6)
    assert reply["exact"] == "1 - x**2/2 + x**4/24 + O(x**6)", reply


def test_sum_and_product(h):
    reply = expression(h, "summation", "1/n**2", variable="n", lower="1", upper="oo")
    assert reply["exact"] == "pi**2/6", reply
    reply = expression(h, "product", "k", variable="k", lower="1", upper="5")
    assert reply["exact"] == "120", reply


def test_matrix_determinant_and_inverse(h):
    matrix = [["1", "2"], ["3", "4"]]
    assert h.call("matrix", operation="determinant", matrix=matrix)["exact"] == "-2"
    assert h.call("matrix", operation="inverse", matrix=matrix)["matrix"] == [["-2", "1"], ["3/2", "-1/2"]]
    assert error_code(h.call("matrix", operation="inverse", matrix=[[1, 2], [2, 4]])) == "no_result"


def test_matrix_rank_and_linear_system(h):
    assert h.call("matrix", operation="rank", matrix=[[1, 2], [2, 4]])["exact"] == "1"
    reply = h.call("matrix", operation="linear_solve", matrix=[[1, 1], [1, -1]], other=[[3], [1]])
    assert reply["solution"] == ["2", "1"], reply
    reply = h.call("matrix", operation="multiply", matrix=[[1, 2], [3, 4]], other=[[0, 1], [1, 0]])
    assert reply["matrix"] == [["2", "1"], ["4", "3"]], reply


def test_eigenvalues(h):
    reply = h.call("matrix", operation="eigenvalues", matrix=[[2, 1], [1, 2]])
    assert [e["value"] for e in reply["eigenvalues"]] == ["1", "3"], reply
    reply = h.call("matrix", operation="eigenvectors", matrix=[[2, 1], [1, 2]])
    assert reply["eigenvectors"][1] == {"eigenvalue": "3", "multiplicity": 1, "vectors": [["1", "1"]]}, reply


def test_ode(h):
    reply = h.call("solve", operation="dsolve", equations=["y' = y"])
    assert reply["exact"] == "y(x) = C1*exp(x)", reply
    reply = h.call("solve", operation="dsolve", equations=["y'' + y = 0"],
                   initial_conditions={"y(0)": "1", "y'(0)": "0"})
    assert reply["exact"] == "y(x) = cos(x)", reply
    reply = h.call("solve", operation="dsolve", equations=["diff(f(t), t) = -2*f(t)"],
                   function="f", variable="t")
    assert reply["exact"] == "f(t) = C1*exp(-2*t)", reply


def test_laplace(h):
    assert expression(h, "laplace_transform", "exp(-a*t)")["exact"] == "1/(a + s)"
    assert expression(h, "inverse_laplace_transform", "1/(s**2 + 1)")["exact"] == "sin(t)*Heaviside(t)"


def test_fourier(h):
    assert expression(h, "fourier_transform", "exp(-x**2)")["exact"] == "sqrt(pi)*exp(-pi**2*k**2)"
    reply = expression(h, "inverse_fourier_transform", "sqrt(pi)*exp(-pi**2*k**2)")
    assert reply["exact"] == "exp(-x**2)", reply


def test_complex_arithmetic(h):
    assert expression(h, "simplify", "(1 + 2*I)*(3 - I)")["exact"] == "5 + 5*I"
    assert expression(h, "simplify", "exp(I*pi)")["exact"] == "-1"


def test_numeric_precision(h):
    reply = expression(h, "evaluate", "pi", precision=50)
    assert reply["numeric"] == "3.1415926535897932384626433832795028841971693993751", reply
    reply = h.call("solve", operation="nsolve", equations=["cos(x) = x"], initial_guess=["1"], precision=20)
    assert reply["numeric"][0]["x"].startswith("0.7390851332151606"), reply


def test_equivalent(h):
    reply = h.call("verify", operation="equivalent", left="(x + 1)**2", right="x**2 + 2*x + 1")
    assert reply["equivalent"] is True and reply["difference"] == "0", reply
    reply = h.call("verify", operation="equivalent", left="sin(x)**2 + cos(x)**2", right="1")
    assert reply["equivalent"] is True, reply


def test_not_equivalent(h):
    reply = h.call("verify", operation="equivalent", left="(x + 1)**2", right="x**2 + 2*x")
    assert reply["success"] is True and reply["equivalent"] is False and reply["difference"] == "1", reply


def test_verify_solution(h):
    equations = ["x**2 - 5*x + 6 = 0"]
    reply = h.call("verify", operation="solution", equations=equations, solution={"x": "2"})
    assert reply["satisfied"] is True and reply["checks"][0]["residual"] == "0", reply
    reply = h.call("verify", operation="solution", equations=equations, solution={"x": "4"})
    assert reply["satisfied"] is False and reply["checks"][0]["residual"] == "2", reply
    reply = h.call("verify", operation="solution", equations=["y'' + y = 0"], function="y",
                   solution={"y": "exp(x)"})
    assert reply["satisfied"] is False and reply["checks"][0]["residual"] == "2*exp(x)", reply


def test_assumptions(h):
    assert expression(h, "simplify", "sqrt(x**2)")["exact"] == "sqrt(x**2)"
    assert expression(h, "simplify", "sqrt(x**2)", assumptions={"x": ["positive"]})["exact"] == "x"
    assert error_code(expression(h, "simplify", "x", assumptions={"x": ["evil"]})) == "invalid_arguments"


def test_invalid_expression(h):
    for text in ("x +* 2", "(x + 1", "", "sin", "2 +", "x $ y", "foo(x)"):
        assert error_code(expression(h, "simplify", text)) in ("invalid_expression", "unknown_function"), text
    assert error_code(h.call("expression", operation="nope", expression="x")) == "invalid_arguments"
    assert error_code(h.call("expression", operation="simplify")) == "invalid_arguments"
    assert error_code(h.call("expression", operation="simplify", expression="x", extra=1)) == "invalid_arguments"
    assert error_code(h.call("nothing", operation="simplify")) == "invalid_arguments"


def test_code_injection(h):
    marker = os.path.join(os.environ.get("TMPDIR", "/tmp"), "tosh-sympy-injection-marker")
    attempts = [
        "__import__('os').system('touch %s')" % marker,
        "__import__(\"os\").system(\"touch %s\")" % marker,
        "exec('import os')", "eval('1+1')", "open('/etc/passwd').read()",
        "x.__class__.__mro__", "().__class__.__bases__[0].__subclasses__()",
        "lambda: 1", "[x for x in (1,2)]", "import os", "x; y", "getattr(x, 'y')",
        "globals()", "compile('1','a','eval')", "breakpoint()", "__builtins__",
        "sympify('1')", "S('1')", "Symbol('x')", "parse_expr('1')", "lambdify(x, x)",
        "f\"{x}\"", "x if x else y", "x := 2", "print(1)", "os.system", "\\x41",
    ]
    for text in attempts:
        code = error_code(expression(h, "simplify", text))
        assert code in ("invalid_expression", "unknown_function"), (text, code)
        for where in ("left", "right"):
            assert h.call("verify", operation="equivalent", **{"left": "x", "right": "x", where: text})["success"] is False
        assert h.call("solve", operation="solve", equations=[text])["success"] is False
        assert h.call("matrix", operation="determinant", matrix=[[text]])["success"] is False
    assert error_code(h.call("solve", operation="dsolve", equations=["y' = y"], function="__import__")) == "invalid_expression"
    assert error_code(h.call("solve", operation="solve", equations=["x = 1"], variables=["x); import os; ("])) == "invalid_expression"
    assert not os.path.exists(marker)


def test_size_limits(h):
    assert error_code(expression(h, "simplify", "x + " * 1500 + "x")) == "input_too_large"
    assert error_code(expression(h, "simplify", "(" * 200 + "x" + ")" * 200)) == "input_too_large"
    assert error_code(expression(h, "simplify", "9**9**9")) == "too_large"
    assert error_code(expression(h, "simplify", "factorial(10**6)")) == "too_large"
    assert error_code(expression(h, "simplify", "1" * 400)) == "too_large"
    assert error_code(h.call("matrix", operation="rank", matrix=[["1"] * 17] * 17)) == "input_too_large"
    assert error_code(expression(h, "evaluate", "pi", precision=100000)) == "invalid_arguments"
    assert error_code(expression(h, "simplify", "x", assumptions={"x" * 70000: ["real"]})) == "input_too_large"


def test_output_is_bounded(h):
    reply = expression(h, "expand", "(x + y + z + w)**16")
    assert reply["success"] and reply["truncated"] is True, reply
    assert len(reply["exact"]) == 6000 and "latex" not in reply, len(reply["exact"])
    assert len(json.dumps(reply)) < 8000


def test_timeout_and_recovery(_):
    helper = Helper(TOSH_SYMPY_TIMEOUT_MS=1500)
    try:
        started = time.monotonic()
        reply = expression(helper, "expand", "(a + b + c + d + f + g)**90")
        elapsed = time.monotonic() - started
        assert error_code(reply) == "timeout", reply
        assert elapsed < 4*SLOW, elapsed
        assert expression(helper, "factor", "x**2 - 1")["exact"] == "(x - 1)*(x + 1)"
    finally:
        helper.close()


def test_memory_limit(_):
    helper = Helper(TOSH_SYMPY_MEMORY_MB=200, TOSH_SYMPY_TIMEOUT_MS=25000)
    try:
        assert error_code(expression(helper, "expand", "(a + b + c + d + f + g)**90")) == "memory_limit"
        assert expression(helper, "factor", "x**2 - 1")["success"] is True
    finally:
        helper.close()


def test_e_is_a_variable_only_when_declared(h):
    reply = h.call("solve", operation="solve", equations=["e**2 - 4 = 0"], variables=["e"])
    assert reply["solutions"] == [{"e": "-2"}, {"e": "2"}], reply
    assert "e is a variable" in reply["warnings"][0], reply
    reply = expression(h, "differentiate", "e**3 + E*e", variable="e")
    assert reply["exact"] == "3*e**2 + E", reply
    assert error_code(h.call("solve", operation="solve", equations=["e**2 - 4 = 0"])) == "invalid_arguments"


def test_euler_constant(h):
    assert expression(h, "evaluate", "E")["numeric"] == "2.71828182845905"
    assert expression(h, "evaluate", "e")["numeric"] == "2.71828182845905"
    assert expression(h, "simplify", "log(E) + log(e) + log(exp(1))")["exact"] == "3"
    assert expression(h, "evaluate", "E", variable="E")["exact"] == "E"


def test_definite_integral_falls_back_to_a_number(h):
    reply = expression(h, "integrate", "exp(sin(x))", variable="x", lower="0", upper="1")
    assert reply["success"] is True and reply["exact"] is None, reply
    assert reply["method"] == "numerical_integration" and reply["timed_out_symbolic"] is False
    assert reply["numeric"] == "1.63186960841805" and reply["precision"] == 15, reply
    assert reply["unevaluated"] == "Integral(exp(sin(x)), (x, 0, 1))" and reply["warnings"], reply
    reply = expression(h, "integrate", "x**x", lower="0", upper="1", precision=30)
    assert reply["numeric"] == "0.783430510712134407059264386527", reply


def test_no_number_for_a_divergent_integral(h):
    for text in ("exp(sin(x))/x", "x**x/(x - 1/2)", "exp(sin(x))/(x - 0.3)"):
        reply = expression(h, "integrate", text, variable="x", lower="0", upper="1")
        assert error_code(reply) == "no_closed_form" and "numeric" not in reply, reply
        assert reply["timed_out"] is False and reply["exact"] is None and "Integral" in reply["unevaluated"]
    # SymPy takes seconds to give up on a double pole: finished or out of time, it gets no number
    reply = expression(h, "integrate", "x**x/(x - 0.3)**2", variable="x", lower="0", upper="1")
    assert error_code(reply) in ("no_closed_form", "timeout") and "numeric" not in reply, reply
    assert reply["exact"] is None and "Integral" in reply["unevaluated"], reply
    assert error_code(expression(h, "integrate", "1/(sin(x) - 1/2)", lower="0", upper="1")) == "no_result"


def test_no_number_without_numeric_bounds(h):
    reply = expression(h, "integrate", "exp(sin(a*x))", variable="x", lower="0", upper="1")
    assert error_code(reply) == "no_closed_form" and "numeric" not in reply, reply
    reply = expression(h, "integrate", "sin(sin(x))", variable="x")
    assert error_code(reply) == "no_closed_form" and reply["unevaluated"] == "Integral(sin(sin(x)), x)", reply


def test_open_sum_is_not_a_result(h):
    reply = expression(h, "summation", "1/(k**3 + 1)", variable="k", lower="1", upper="oo")
    assert error_code(reply) == "no_closed_form" and reply["unevaluated"].startswith("Sum("), reply


def test_series_fast_path_matches(h):
    started = time.monotonic()
    reply = expression(h, "series", "exp(sin(x))", order=20)
    assert time.monotonic() - started < 5*SLOW, "the composed series should not need the slow routine"
    assert reply["exact"].startswith("1 + x + x**2/2 - x**4/8 - x**5/15 - x**6/240 + x**7/90 + 31*x**8/5760"), reply
    assert reply["exact"].endswith("O(x**20)"), reply
    assert expression(h, "series", "1/(1 - x)", order=4)["exact"] == "1 + x + x**2 + x**3 + O(x**4)"
    assert expression(h, "series", "sin(x)/x", order=6)["exact"] == "1 - x**2/6 + x**4/120 + O(x**6)"
    assert expression(h, "series", "sin(a*x)", variable="x", order=4)["polynomial"] == "-a**3*x**3/6 + a*x"


def test_dsolve_survives_a_failing_method(h):
    reply = h.call("solve", operation="dsolve", equations=["y' = x**2 + y**2"])
    assert reply["success"] and reply["method"] == "1st_power_series" and reply["warnings"], reply


def test_nsolve_never_guesses(h):
    reply = h.call("solve", operation="nsolve", equations=["cos(x) = x"])
    assert error_code(reply) == "invalid_arguments" and "initial_guess" in reply["error"]["message"]
    reply = h.call("solve", operation="solve", equations=["x*exp(x) + sin(x) = 2"])
    assert error_code(reply) == "not_supported" and "nsolve" in reply["error"]["message"]


def test_timed_out_indefinite_integral(_):
    helper = Helper(TOSH_SYMPY_TIMEOUT_MS=SHORT)
    try:
        # slow under every hash order: the worker runs with -I, so its seed is random and some orders
        # let SymPy give up on simpler integrands in milliseconds
        reply = expression(helper, "integrate", "exp(sin(x))/(1 + x + x**5)", variable="x")
        assert error_code(reply) == "timeout" and reply["timed_out"] is True, reply
        assert reply["exact"] is None and "numeric" not in reply, reply
        assert reply["unevaluated"] == "Integral(exp(sin(x))/(x**5 + x + 1), x)", reply
        assert expression(helper, "factor", "x**2 - 1")["exact"] == "(x - 1)*(x + 1)"
    finally:
        helper.close()


def test_timed_out_definite_integral_gets_a_number(_):
    helper = Helper(TOSH_SYMPY_TIMEOUT_MS=SHORT)
    try:
        reply = expression(helper, "integrate", "exp(sin(x)**2)/(1 + x**3)", variable="x", lower="0", upper="1")
        assert reply["success"] is True and reply["exact"] is None, reply
        assert reply["timed_out_symbolic"] is True and reply["method"] == "numerical_integration", reply
        assert reply["numeric"] == "1.07619861679014", reply
        assert expression(helper, "expand", "(x + 1)**2")["exact"] == "x**2 + 2*x + 1"
        # trigonometric functions get a second symbolic attempt after the number, which a slow machine
        # stops as well, worker included: nothing may be asked of this helper after it
        reply = expression(helper, "integrate", "exp(sin(x))", variable="x", lower="0", upper="1")
        assert reply["success"] is True and reply["exact"] is None, reply
        assert reply["method"] == "numerical_integration" and reply["numeric"] == "1.63186960841805", reply
    finally:
        helper.close()


def test_timed_out_series_reports_what_finished(_):
    helper = Helper(TOSH_SYMPY_TIMEOUT_MS=SHORT)
    try:
        reply = expression(helper, "series", "exp(exp(sin(x)))*log(x)", point="1", order=8)
        assert error_code(reply) == "timeout" and reply["timed_out"] is True and reply["exact"] is None, reply
        assert (reply["expression"], reply["variable"], reply["point"], reply["order"]) == \
            ("exp(exp(sin(x)))*log(x)", "x", "1", 8), reply
        assert reply["partial"]["order"] == 4 and "O((x - 1)**4" in reply["partial"]["exact"], reply
        assert expression(helper, "factor", "x**2 - 1")["success"] is True
    finally:
        helper.close()


def test_timed_out_ode(_):
    helper = Helper(TOSH_SYMPY_TIMEOUT_MS=SHORT)
    try:
        reply = helper.call("solve", operation="dsolve", equations=["y''' + y = tan(x)"])
        assert error_code(reply) == "timeout" and reply["timed_out"] is True, reply
        assert reply["unevaluated"] == ["y(x) + Derivative(y(x), (x, 3)) = tan(x)"], reply
        assert helper.call("solve", operation="solve", equations=["x = 1"])["success"] is True
    finally:
        helper.close()


def test_same_answer_in_every_worker(_):
    seeds = set()
    for _ in range(3):
        probe = subprocess.run(
            [PYTHON, "-P", "-s", "-B", "-c", "print(hash('x'))"], capture_output=True, text=True,
            env={"PYTHONHASHSEED": "1"}).stdout.strip()
        seeds.add(probe)
    assert len(seeds) == 1, seeds
    helper = Helper()
    try:
        started = time.monotonic()
        reply = expression(helper, "integrate", "1/(1 + x**3 + sin(x))", variable="x")
        assert error_code(reply) == "no_closed_form" and time.monotonic() - started < 10*SLOW, reply
    finally:
        helper.close()


def test_the_check_knows_every_name_of_the_grammar(_):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tosh_sympy import anchor, mathlang
    assert anchor._KNOWN == set(mathlang.FUNCTIONS) | set(mathlang.CONSTANTS)


def test_a_call_that_does_not_say_what_the_request_says_is_not_run(h):
    def ask(text, tool, **arguments):
        return h.call(tool, _trust=KEY, _source={"request": text}, **arguments)

    def refused(reply):
        return reply["success"] is False and reply["error"]["code"] == "transcription_mismatch" and reply["interpreted_input"]

    text = "Find the limit of (sin(x) - x)/x^3 as x approaches 0."
    assert refused(ask(text, "expression", operation="limit", expression="sin(x) - x", variable="x", point="0"))
    reply = ask(text, "expression", operation="limit", expression="(sin(x) - x)/x**3", variable="x", point="0")
    assert reply["exact"] == "-1/6" and reply["result_kind"] == "exact", reply
    assert "expression: (-x + sin(x))/x**3" in reply["interpreted_input"], reply
    text = "Integrate exp(-2x) from 0 to infinity."
    assert refused(ask(text, "expression", operation="integrate", expression="exp(-2*x)", lower="0", upper="8"))
    assert ask(text, "expression", operation="integrate", expression="exp(-2*x)", lower="0", upper="oo")["exact"] == "1/2"
    text = "Integrate x^3 from 1 to 2."
    assert refused(ask(text, "expression", operation="integrate", expression="x**3", lower="2", upper="1"))
    assert refused(ask("Solve y' = 2y.", "solve", operation="dsolve", equations=["y' = 2*y"],
                       initial_conditions={"y(0)": "1"}))
    assert refused(ask("Factor x^2 + 5x + 6.", "expression", operation="factor", expression="x**2 - 5*x + 6"))
    assert refused(ask("Solve the system x + y = 10, x - y = 2.", "solve", operation="solve", equations=["x + y = 10"]))
    text = "Find the determinant of [[1, 2, 0], [3, 1, 4], [0, 5, 2]]."
    assert refused(ask(text, "matrix", operation="determinant", matrix=[["1", "2", "0"], ["3", "1", "4"]]))
    assert refused(ask(text, "matrix", operation="determinant", matrix=[["1", "2"], ["3", "1"], ["0", "5"]]))
    assert ask(text, "matrix", operation="determinant", matrix=[["1", "2", "0"], ["3", "1", "4"], ["0", "5", "2"]])["exact"] == "-30"
    # the request names no formula: the call waits for a review instead of running on trust
    text = "The sum of two numbers is 9 and their product is 20. What are they?"
    reply = ask(text, "solve", operation="solve", equations=["x + y = 9", "x*y = 20"])
    assert reply["error"]["code"] == "needs_review" and reply["reasons"], reply
    reply = h.call("solve", _trust=KEY, _source={"request": text}, _reviewed="consistent", operation="solve", equations=["x + y = 9", "x*y = 20"])
    assert reply["success"], reply
    # without the request, as from another client, the call runs as before
    assert h.call("expression", operation="factor", expression="x**2 - 5*x + 6")["exact"] == "(x - 3)*(x - 2)"


def _check(text, operation, **arguments):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tosh_sympy import anchor
    return anchor.check(text, "", operation, arguments)


def test_latex_requests_are_read_as_math(_):
    text = r"Simplifica \[ \frac{\frac{1}{x} + 1}{x^{2} - 1} \]"
    assert _check(text, "simplify", expression="(1/x + 1)/(x**2 - 1)")[0] == "consistent"
    assert _check(text, "simplify", expression="(1/x + 1)/(x**2 + 1)")[0] == "inconsistent"
    assert _check(text, "simplify", expression="1/x + 1")[0] == "inconsistent"
    text = r"Factoriza $x^{3} - 2 \cdot x^{2} + x$"
    assert _check(text, "factor", expression="x**3 - 2*x**2 + x")[0] == "consistent"
    assert _check(text, "factor", expression="x**3 - 2*x + x")[0] == "inconsistent"
    text = r"Calcula \(\int_{-\infty}^{0} e^{2x}\,dx\)"
    assert _check(text, "integrate", expression="exp(2*x)", lower="-oo", upper="0")[0] == "consistent"
    status, reasons = _check(text, "integrate", expression="exp(2*x)", lower="-100", upper="0")
    assert status == "inconsistent" and "goes to infinity" in reasons[0], reasons
    text = r"Evalúa \int_0^1 \frac{\ln(1+x)}{1+x^{2}} \mathrm{d}x"
    assert _check(text, "integrate", expression="log(1+x)/(1+x**2)", lower="0", upper="1")[0] == "consistent"
    assert _check(text, "integrate", expression="log(1+x)/(1+x**2)", lower="0", upper="2")[0] == "inconsistent"
    assert _check(text, "integrate", expression="log(1+x)", lower="0", upper="1")[0] == "inconsistent"
    text = r"Suma \sum_{n=1}^{\infty} \frac{1}{n^{2}}"
    assert _check(text, "summation", expression="1/n**2", variable="n", lower="1", upper="oo")[0] == "consistent"
    assert _check(text, "summation", expression="1/n**2", variable="n", lower="1", upper="100")[0] == "inconsistent"
    text = r"Halla \lim_{x \to 0} \frac{\sin(x)}{x}"
    assert _check(text, "limit", expression="sin(x)/x", variable="x", point="0")[0] == "consistent"
    assert _check(text, "limit", expression="sin(x)", variable="x", point="0")[0] == "inconsistent"
    text = r"Calcula \sqrt{x^{2} + 1} \times \left( x - 1 \right) en x = 2"
    assert _check(text, "evaluate", expression="sqrt(x**2 + 1)*(x - 1)", at={"x": 2})[0] == "consistent"


def test_a_word_elsewhere_does_not_set_the_operator(_):
    status, reasons = _check("Take x^3 and e^x - 1.\nIndica el error absoluto del resultado.", "simplify",
                             expression="x**3/(exp(x)-1)")
    assert status == "uncertain", (status, reasons)
    assert _check("What is the difference between x^3 and e^x - 1?", "simplify",
                  expression="x**3/(exp(x)-1)")[0] == "inconsistent"
    assert _check("What is the error of approximating e by 2.718?", "evaluate", expression="E - 2.718")[0] == "consistent"
    assert _check("What is the error of approximating e by 2.718?", "evaluate", expression="E/2.718")[0] != "consistent"


def test_format_numbers_are_not_data(_):
    text = ("Integrate x^2 from 0 to 1.\n1. Give the exact value.\n2. Give it to 10 decimal places.\n"
            "3. Check it within 1e-9 tolerance.")
    assert _check(text, "integrate", expression="x**2", lower="0", upper="1") == ("consistent", [])
    status, reasons = _check(text, "integrate", expression="x**2", lower="0", upper="3")
    assert status == "inconsistent" and not any("has 2" in r or "has 10" in r for r in reasons), reasons
    status, reasons = _check("Calcula la integral de x^2 from 0 to 1 con 10 cifras decimales.", "integrate",
                             expression="x**2", lower="0", upper="10")
    assert status == "inconsistent" and "the call goes from 0 to 10" in reasons[0], reasons
    # a tolerance is not data the call must use, but a call may still use it
    assert _check("Integrate x^2 from 0 to 1 with a tolerance of 0.001.", "integrate",
                  expression="x**2", lower="0", upper="1") == ("consistent", [])


def test_a_client_cannot_vouch_for_its_own_call(h):
    text = "Factor x^2 + 5x + 6."
    # without the key the fields are dropped: the call runs as a plain call, and says so
    for extra in ({}, {"_trust": "guess"}, {"_trust": ""}):
        reply = h.call("expression", _source={"request": text}, _reviewed="consistent", operation="factor",
                       expression="x**2 - 5*x + 6", **extra)
        assert reply["success"] and any("reserved" in w for w in reply["warnings"]), reply
    reply = h.call("expression", _trust=KEY, _source={"request": text}, operation="factor", expression="x**2 - 5*x + 6")
    assert reply["error"]["code"] == "transcription_mismatch", reply
    # a forged review never passes a call the request does not support
    reply = h.call("solve", _trust="guess", _reviewed="consistent", _source={"request": "Solve x + y = 9."},
                   operation="solve", equations=["x + y = 9", "x*y = 20"])
    assert reply["success"] and any("reserved" in w for w in reply["warnings"]), reply


def test_calls_in_the_shapes_models_use(h):
    reply = h.call("verify", operation="solution", left="x**2 - 5*x + 6", right="0", solution={"x": "4"})
    assert reply["satisfied"] is False and reply["checks"][0]["residual"] == "2", reply
    reply = h.call("verify", operation="equivalent", left="x**2 - 2*x - 3", right="0",
                   equations=["x**2 - 2*x - 3 = 0"], solution={"x": "3"})
    assert reply["operation"] == "solution" and reply["satisfied"] is True and reply["warnings"], reply
    reply = h.call("verify", operation="solution", equations=["y' = 2*x"], solution={"x": "x", "y": "x**2"})
    assert reply["satisfied"] is True, reply
    reply = h.call("verify", operation="solution", equations=["y'' + 4*y = 0"], solution={"x": "sin(2*x)"})
    assert reply["satisfied"] is True, reply
    reply = h.call("verify", operation="solution", left="y", right="exp(3*x)", equations=["y' - 3*y = 0"], function="y")
    assert reply["satisfied"] is True, reply
    # equivalent never ignores 'equations': they are the candidate, or the call is refused
    reply = h.call("verify", operation="equivalent", left="Derivative(y, x) + 2*y", right="0",
                   equations=["y = exp(-2*x)"])
    assert reply["operation"] == "solution" and reply["satisfied"] is True and reply["warnings"], reply
    reply = h.call("verify", operation="equivalent", left="x**2 - 9", right="0", equations=["x = 4"])
    assert reply["satisfied"] is False, reply
    reply = h.call("verify", operation="equivalent", left="x**3 - 27", right="0", equations=["x**3 - 27 = 0"])
    assert reply["success"] is False and reply["error"]["code"] == "invalid_arguments", reply
    reply = expression(h, "expand", "log(1 + x)", order=5)
    assert reply["operation"] == "series" and reply["exact"] == "x - x**2/2 + x**3/3 - x**4/4 + O(x**5)", reply
    assert expression(h, "expand", "(x + 1)**2")["operation"] == "expand"


def test_leibniz_notation(h):
    reply = h.call("solve", operation="dsolve", equations=["dy/dx = 3x**2"], function="y")
    assert reply["exact"] == "y(x) = C1 + x**3", reply
    reply = h.call("solve", operation="dsolve", equations=["d2y/dx2 + y = 0"])
    assert reply["exact"] == "y(x) = C1*sin(x) + C2*cos(x)", reply
    reply = h.call("solve", operation="dsolve", equations=["diff(y(x), x) = 3*x**2"], function="y(x)")
    assert reply["exact"] == "y(x) = C1 + x**3", reply
    assert expression(h, "simplify", "dy/dx")["exact"] == "dy/dx"


def test_counterexample_settles_non_equivalence(h):
    reply = h.call("verify", operation="equivalent", left="sqrt(x**2)", right="x")
    assert reply["equivalent"] is False and reply["differs_at"] == {"x": "-7/3"}, reply
    reply = h.call("verify", operation="equivalent", left="sqrt(x**2)", right="x", assumptions={"x": ["positive"]})
    assert reply["equivalent"] is True, reply


def test_product_is_simplified(h):
    reply = expression(h, "product", "1 + 1/k", variable="k", lower="1", upper="n")
    assert reply["exact"] == "n + 1", reply


def test_definitions_stay_small(h):
    tools = h.rpc("tools/list")["result"]["tools"]
    assert len(json.dumps(tools)) < 4200, len(json.dumps(tools))
    assert expression(h, "laplace_transform", "exp(-a*t)", transform_variable="s")["exact"] == "1/(a + s)"
    assert h.call("matrix", operation="determinant", matrix=[["x", "1"], ["1", "x"]],
                  assumptions={"x": ["real"]})["exact"] == "x**2 - 1"


def test_idle_worker_is_released(_):
    helper = Helper(TOSH_SYMPY_IDLE_SECONDS=1)
    try:
        expression(helper, "factor", "x**2 - 1")
        children = lambda: subprocess.run(["/usr/bin/pgrep", "-P", str(helper.process.pid)],
                                          capture_output=True, text=True).stdout.split()
        assert len(children()) == 1
        time.sleep(7)
        assert children() == []
        assert expression(helper, "factor", "x**2 - 1")["success"] is True
    finally:
        helper.close()


def test_worker_is_confined(_):
    probe = r'''
import os, sys
sys.path.insert(0, sys.argv[1])
from tosh_sympy import worker
import sympy
print(worker._sandbox())
sys.addaudithook(worker._audit)
def attempt(action):
    try:
        action()
        print("allowed")
    except Exception as error:
        print("denied")
attempt(lambda: open(sys.argv[2], "w"))
attempt(lambda: os.open(sys.argv[2], os.O_WRONLY | os.O_CREAT))
attempt(lambda: __import__("subprocess").run(["/usr/bin/true"]))
attempt(lambda: os.fork())
attempt(lambda: __import__("socket").create_connection(("127.0.0.1", 9), 1))
attempt(lambda: os.listdir(os.path.expanduser("~")))
attempt(lambda: sympy.integrate(sympy.Symbol("x") ** 2))
'''
    marker = os.path.join(os.environ.get("TMPDIR", "/tmp"), "tosh-sympy-write-marker")
    lines = subprocess.run([PYTHON, "-I", "-B", "-c", probe, RUNTIME, marker],
                           capture_output=True, text=True).stdout.split()
    home_is_inside_runtime = os.path.expanduser("~").startswith(RUNTIME)
    expected = ["True"] + ["denied"] * 5 + ["allowed" if home_is_inside_runtime else "denied", "allowed"]
    assert lines == expected, lines
    assert not os.path.exists(marker)


def test_protocol(h):
    tools = h.rpc("tools/list")["result"]["tools"]
    assert [t["name"] for t in tools] == ["expression", "solve", "matrix", "verify"]
    for tool in tools:
        schema = tool["inputSchema"]
        assert schema["additionalProperties"] is False and "operation" in schema["required"]
        assert schema["properties"]["operation"]["enum"]
        assert tool["annotations"]["readOnlyHint"] is True
    assert "error" in h.rpc("resources/list")
    assert h.rpc("ping")["result"] == {}


def measure():
    started = time.monotonic()
    helper = Helper()
    ready = time.monotonic() - started
    started = time.monotonic()
    expression(helper, "factor", "x**2 - 5*x + 6")
    first = time.monotonic() - started
    warm = []
    for _ in range(20):
        started = time.monotonic()
        expression(helper, "factor", "x**2 - 5*x + 6")
        warm.append(time.monotonic() - started)
    helper.close()
    warm.sort()
    print(f"supervisor start {ready * 1000:.0f} ms, first call {first * 1000:.0f} ms, "
          f"warm call median {warm[len(warm) // 2] * 1000:.1f} ms")


def main():
    tests = [(name, function) for name, function in globals().items() if name.startswith("test_")]
    helper = Helper(**SHARED)
    failed = 0
    for name, function in tests:
        try:
            function(helper)
            print(f"ok    {name}")
        except Exception as error:
            failed += 1
            print(f"FAIL  {name}: {type(error).__name__}: {str(error)[:500]}")
            if not helper.process.poll() is None:
                helper = Helper(**SHARED)
    helper.close()
    measure()
    print(f"{len(tests) - failed} of {len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
