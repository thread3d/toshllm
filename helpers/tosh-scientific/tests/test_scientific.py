# ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
# Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the scientific helper, driven over its MCP stdio protocol against known answers.

    vendor/tosh-sympy/python/bin/python3 -I helpers/tosh-scientific/tests/test_scientific.py [runtime dir]
"""

import json
import math
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


class Helper:
    def __init__(self, toolset="scientific", **environment):
        self.process = subprocess.Popen(
            [PYTHON, "-I", "-B", os.path.join(RUNTIME, "tosh_sympy", "server.py"), toolset],
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

    def call(self, tool, operation, **arguments):
        result = self.rpc("tools/call", {"name": tool, "arguments": {"operation": operation, **arguments}})["result"]
        reply = json.loads(result["content"][0]["text"])
        assert result["isError"] == (not reply["success"] and "timed_out" not in reply), result
        return reply

    def children(self):
        return subprocess.run(["/usr/bin/pgrep", "-P", str(self.process.pid)],
                              capture_output=True, text=True).stdout.split()

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=10)


def near(value, expected, tolerance=1e-8):
    return abs(value - expected) <= tolerance * max(1.0, abs(expected))


def code(reply):
    assert reply["success"] is False, reply
    assert set(reply["error"]) == {"code", "message"}, reply
    return reply["error"]["code"]


# NumPy

def test_dot_and_matrix_product(h):
    reply = h.call("linalg", "multiply", matrix=[[1, 2, 3]], other=[4, 5, 6])
    assert reply["product"] == [32.0], reply
    reply = h.call("linalg", "multiply", matrix=[[1, 2], [3, 4]], other=[[0, 1], [1, 0]])
    assert reply["product"] == [[2.0, 1.0], [4.0, 3.0]], reply


def test_solve(h):
    reply = h.call("linalg", "solve", matrix=[[2, 1], [1, 3]], other=[3, 5])
    assert reply["solution"] == [0.8, 1.4] and reply["residual"] < 1e-12, reply


def test_determinant_and_inverse(h):
    assert h.call("linalg", "determinant", matrix=[[1, 2], [3, 4]])["determinant"] == -2.0
    assert h.call("linalg", "inverse", matrix=[[1, 2], [3, 4]])["inverse"] == [[-2.0, 1.0], [1.5, -0.5]]
    assert h.call("linalg", "rank", matrix=[[1, 2], [2, 4]])["rank"] == 1
    assert near(h.call("linalg", "norm", matrix=[[3, 4]])["norm"], 5.0)
    assert near(h.call("linalg", "condition", matrix=[[1, 0], [0, 100]])["condition"], 100.0)


def test_eigenvalues(h):
    reply = h.call("linalg", "eigen", matrix=[[2, 1], [1, 2]])
    assert reply["eigenvalues"] == [1.0, 3.0] and reply["symmetric"] is True, reply
    reply = h.call("linalg", "eigen", matrix=[[0, 1], [-1, 0]])
    assert reply["eigenvalues"] == {"re": [0.0, 0.0], "im": [1.0, -1.0]}, reply


def test_svd_reconstructs(h):
    matrix = [[1, 2], [3, 4], [5, 6]]
    reply = h.call("linalg", "svd", matrix=matrix)
    u, s, vt = reply["u"], reply["singular_values"], reply["vt"]
    for i in range(3):
        for j in range(2):
            assert near(sum(u[i][k] * s[k] * vt[k][j] for k in range(2)), matrix[i][j], 1e-9), reply
    reply = h.call("linalg", "qr", matrix=[[1, 2], [3, 4]])
    q, r = reply["q"], reply["r"]
    assert near(sum(q[0][k] * r[k][1] for k in range(2)), 2.0, 1e-9), reply


def test_least_squares(h):
    reply = h.call("linalg", "least_squares", matrix=[[1, 0], [1, 1], [1, 2]], other=[1, 3, 5])
    assert near(reply["solution"][0], 1.0) and near(reply["solution"][1], 2.0) and reply["rank"] == 2, reply


def test_fft_of_a_known_sinusoid(h):
    reply = h.call("signal", "fft", expression="3*sin(2*pi*50*t) + sin(2*pi*120*t)", sample_rate=1000, duration=1)
    peaks = reply["dominant_frequencies"]
    assert reply["n"] == 1000 and near(peaks[0]["hz"], 50) and near(peaks[0]["amplitude"], 3, 1e-9), reply
    assert near(peaks[1]["hz"], 120) and near(peaks[1]["amplitude"], 1, 1e-9), reply
    values = [math.sin(2 * math.pi * 5 * n / 64) for n in range(64)]
    reply = h.call("signal", "fft", values=values)
    assert near(reply["dominant_frequencies"][0]["cycles_per_sample"], 5 / 64) and "note" in reply, reply


def test_inverse_fft(h):
    reply = h.call("signal", "ifft", real=[4, 0, 0, 0])
    assert reply["values"] == [1.0, 1.0, 1.0, 1.0], reply


# SciPy

def test_integral_with_known_answer(h):
    reply = h.call("compute", "integrate", expression="exp(-x**2)", lower="-oo", upper="oo")
    assert near(reply["value"], math.sqrt(math.pi)) and reply["error_estimate"] < 1e-8, reply
    reply = h.call("compute", "integrate", expression="sin(x)", variable="x", lower=0, upper="pi")
    assert near(reply["value"], 2.0) and "method" in reply and "tolerance" in reply, reply
    reply = h.call("compute", "integrate", expression="x*y", bounds={"x": [0, 1], "y": [0, 2]})
    assert near(reply["value"], 1.0), reply
    reply = h.call("compute", "integrate", y=[0, 1, 4, 9, 16], dx=1)
    assert near(reply["value"], 64 / 3) and reply["method"] == "simpson", reply


def test_divergent_integral_is_not_a_result(h):
    assert code(h.call("compute", "integrate", expression="1/x", lower=-1, upper=1)) in ("non_finite_result", "not_converged")
    assert code(h.call("compute", "integrate", expression="1/(x - 0.3)", lower=0, upper=1)) in ("non_finite_result", "not_converged")
    assert code(h.call("compute", "integrate", expression="1/x", lower=0, upper=1)) in ("non_finite_result", "not_converged")
    assert code(h.call("compute", "integrate", expression="sqrt(x)", lower=-1, upper=1)) == "non_finite_result"
    assert code(h.call("compute", "integrate", expression="sin(x)", variable="x")) == "invalid_arguments"


def test_root_with_known_root(h):
    reply = h.call("compute", "root", expression="x**2 - 2", bracket=[0, 2])
    assert near(reply["root"]["x"], math.sqrt(2)) and reply["method"] == "brentq", reply
    reply = h.call("compute", "root", expression="cos(x) = x", initial_guess=[1])
    assert near(reply["root"]["x"], 0.7390851332151607), reply
    reply = h.call("compute", "root", equations=["x**2 + y**2 = 4", "x - y = 0"], initial_guess=[1, 1])
    assert near(reply["root"]["x"], math.sqrt(2)) and near(reply["root"]["y"], math.sqrt(2)), reply


def test_root_never_guesses(h):
    assert code(h.call("compute", "root", expression="cos(x) - x")) == "missing_initial_guess"
    assert code(h.call("compute", "root", expression="x**2 + 1", bracket=[-1, 1])) == "invalid_bracket"
    assert code(h.call("compute", "root", expression="x**2 + 1", initial_guess=[1])) == "not_converged"


def test_optimization_with_known_minimum(h):
    reply = h.call("optimize", "minimize", expression="(x - 3)**2 + (y + 2)**2", initial_guess={"x": 0, "y": 0})
    assert near(reply["solution"]["x"], 3, 1e-6) and near(reply["solution"]["y"], -2, 1e-6), reply
    assert reply["objective"] < 1e-10 and reply["iterations"] > 0 and reply["termination"], reply
    reply = h.call("optimize", "minimize", expression="x**2 + y**2", initial_guess={"x": 1, "y": 1},
                   constraints=["x + y = 1"])
    assert near(reply["solution"]["x"], 0.5, 1e-6) and near(reply["objective"], 0.5, 1e-6), reply
    reply = h.call("optimize", "maximize", expression="x*(10 - x)", bounds={"x": [0, 10]})
    assert near(reply["solution"]["x"], 5, 1e-6) and near(reply["objective"], 25, 1e-6), reply
    reply = h.call("optimize", "minimize", expression="(1 - x)**2 + 100*(y - x**2)**2", initial_guess=[-1.2, 1])
    assert near(reply["solution"]["x"], 1, 1e-4) and near(reply["solution"]["y"], 1, 1e-4), reply


def test_optimization_limits_and_bounds(h):
    assert code(h.call("optimize", "minimize", expression="(x - 3)**2 + y**2")) == "missing_initial_guess"
    assert code(h.call("optimize", "minimize", expression="x**2", initial_guess={"x": 1},
                       bounds={"x": [5, 1]})) == "invalid_bounds"
    assert code(h.call("optimize", "minimize", expression="x**2", initial_guess={"x": 9},
                       bounds={"x": [0, 1]})) == "invalid_bounds"
    assert code(h.call("optimize", "minimize", expression="x", initial_guess={"x": 0})) == "not_converged"
    assert code(h.call("optimize", "minimize", expression="x**2", initial_guess={"x": 1},
                       max_iterations=1000000)) == "limit_exceeded"


def test_curve_fit(h):
    xs = [0, 1, 2, 3, 4, 5]
    ys = [2.0 * math.exp(-0.7 * x) + 0.5 for x in xs]
    reply = h.call("optimize", "curve_fit", expression="a*exp(-b*x) + c", x=xs, y=ys,
                   initial_guess={"a": 1, "b": 1, "c": 0})
    fitted = reply["parameters"]
    assert near(fitted["a"], 2, 1e-6) and near(fitted["b"], 0.7, 1e-6) and near(fitted["c"], 0.5, 1e-6), reply
    assert near(reply["r_squared"], 1.0), reply
    assert code(h.call("optimize", "curve_fit", expression="a*exp(b*x)", x=xs, y=ys)) == "missing_initial_guess"
    reply = h.call("optimize", "fit_residuals", residuals=["a + b - 3", "a - b - 1"], initial_guess={"a": 0, "b": 0})
    assert near(reply["solution"]["a"], 2, 1e-6) and near(reply["solution"]["b"], 1, 1e-6), reply


def test_interpolation(h):
    reply = h.call("compute", "interpolate", x=[0, 1, 2, 3], y=[0, 1, 4, 9], at=[1.5], kind="cubic")
    assert near(reply["values"][0], 2.25), reply
    assert h.call("compute", "interpolate", x=[0, 2], y=[0, 4], at=[1])["values"] == [2.0]
    assert h.call("compute", "interpolate", x=[[0, 2, 4]], y=[[10, 20, 15]], at=[[3]])["values"] == [17.5]
    assert code(h.call("compute", "interpolate", x=[0, 1, 2, 3], y=[0, 1, 4, 9], at=[5])) == "out_of_range"
    assert code(h.call("compute", "interpolate", x=[0, 1, 1], y=[0, 1, 2], at=[0.5])) == "invalid_arguments"
    assert code(h.call("compute", "interpolate", x=[0, 1], y=[0, 1], at=[0.5], kind="cubic")) == "invalid_arguments"
    assert code(h.call("compute", "interpolate", x=[0, 1, 2], y=[0, 1], at=[0.5])) == "invalid_dimensions"
    assert h.call("compute", "interpolate", x=[[0, 0], [2, 4]], at=1)["values"] == [2.0]
    reply = h.call("compute", "interpolate", x=[0, 2], at=[1])
    assert code(reply) == "invalid_arguments" and "value at each x" in reply["error"]["message"], reply
    reply = h.call("compute", "interpolate", expression="[(0, 1), (2, 5)]", x=["1"])
    assert reply["values"] == [3.0], reply
    assert code(h.call("compute", "interpolate", expression="x**2", x=[1])) == "invalid_arguments"
    assert h.call("compute", "interpolate", expression="cubic spline", x=[0, 1, 2, 3], y=[0, 1, 8, 27], at=[1.5])["kind"] == "cubic"
    # empty placeholders for arguments the operation does not take are ignored
    reply = h.call("compute", "evaluate", expression="pi**2/6", x=[], y=[], at=[], bracket=[], data={})
    assert near(reply["value"], 1.6449340668482), reply


def test_ode_with_known_solution(h):
    reply = h.call("ode", "solve_ivp", equations=["dy/dt = -0.5*y"], initial_conditions={"y": 1}, interval=[0, 2])
    assert near(reply["final"]["y"], math.exp(-1), 1e-6) and len(reply["t"]) == 11, reply
    reply = h.call("ode", "solve_ivp", equations=["y'' + y = 0"], initial_conditions={"y(0)": 1, "y'(0)": 0},
                   interval=[0, math.pi], at=[math.pi])
    assert near(reply["final"]["y"], -1, 1e-6) and abs(reply["final"]["y'"]) < 1e-6, reply
    reply = h.call("ode", "solve_ivp", equations=["dx/dt = y", "dy/dt = -x"],
                   initial_conditions={"x": 1, "y": 0}, interval=[0, 1])
    assert near(reply["final"]["x"], math.cos(1), 1e-6) and near(reply["final"]["y"], -math.sin(1), 1e-6), reply
    reply = h.call("ode", "solve_ivp", equations=["y' = -1000*(y - cos(t))"], initial_conditions={"y": 0},
                   interval=[0, 1], method="Radau")
    assert near(reply["final"]["y"], math.cos(1), 2e-3) and reply["method"] == "Radau", reply
    # the second order equation already split by the caller
    reply = h.call("ode", "solve_ivp", equations=["dy/dt = y'", "dy'/dt = -y"],
                   initial_conditions={"y": 0, "y'": 1}, interval=[0, 1])
    assert near(reply["final"]["y"], math.sin(1), 1e-6), reply
    assert code(h.call("ode", "solve_ivp", equations=["dy/dt = y'"], initial_conditions={"y": 0, "y'": 1},
                       interval=[0, 1])) == "invalid_arguments"


def test_ode_accepts_the_notations_models_use(h):
    reply = h.call("ode", "solve_ivp", equations=["d2y/dt2 + y = 0"], initial_conditions={"y": 1, "dy/dt": 0},
                   interval=[0, math.pi])
    assert near(reply["final"]["y"], -1, 1e-6), reply
    reply = h.call("ode", "solve_ivp", equations=["dx/dt = y", "dy/dt = -x"], initial_conditions={"x": 1},
                   interval=[0, 1])
    assert code(reply) == "invalid_initial_conditions" and "y" in reply["error"]["message"], reply
    reply = h.call("ode", "solve_ivp", equations=["y'' + y = 0", "y(0) = 1", "y'(0) = 0"],
                   initial_conditions={}, interval=[0, math.pi])
    assert near(reply["final"]["y"], -1, 1e-6), reply
    reply = h.call("stats", "ttest", values=[[5.2, 4.8, 5.5, 5.9, 5.1], [4.1, 4.4, 3.9, 4.6, 4.2]])
    assert reply["test"].startswith("two-sample") and reply["p_value"] < 0.01, reply
    reply = h.call("stats", "percentile", name="normal", parameters={"mean": 100, "std": 15}, q=90)
    assert near(reply["quantile"], 119.22327348316901), reply
    reply = h.call("compute", "evaluate", expression="sqrt(x) + x**2", variable="x", at=[1, 4])
    assert reply["values"] == [2.0, 18.0], reply
    reply = h.call("stats", "confidence_interval", values=[10.2, 9.8, 10.5, 10.1, 9.9, 10.3], level=0.99)
    assert reply["level"] == 0.99 and reply["lower"] < 9.8, reply
    reply = h.call("compute", "root", expression="cos(x) - x", lower=0, upper=1)
    assert near(reply["root"]["x"], 0.7390851332151607), reply
    reply = h.call("optimize", "maximize", expression="x*(10 - x)", x=[0, 10])
    assert near(reply["solution"]["x"], 5, 1e-6), reply


def test_ode_needs_proper_conditions(h):
    assert code(h.call("ode", "solve_ivp", equations=["y'' + y = 0"], initial_conditions={"y": 1},
                       interval=[0, 1])) == "invalid_initial_conditions"
    assert code(h.call("ode", "solve_ivp", equations=["y' = y"], initial_conditions={"y(2)": 1},
                       interval=[0, 1])) == "invalid_initial_conditions"
    assert code(h.call("ode", "solve_ivp", equations=["y' = y"], interval=[0, 1])) == "invalid_initial_conditions"
    assert code(h.call("ode", "solve_ivp", equations=["y' = y**2"], initial_conditions={"y": 1},
                       interval=[0, 2])) == "integration_failed"


def test_filter_response(h):
    reply = h.call("signal", "lowpass", expression="sin(2*pi*5*t) + sin(2*pi*200*t)", sample_rate=1000, duration=1,
                   cutoff=20)
    gains = {round(g["hz"]): g["gain"] for g in reply["gain"]}
    assert near(gains[20], 1 / math.sqrt(2), 1e-6) and gains[40] < 0.1 and gains[10] > 0.99, reply
    assert near(reply["rms_before"], 1.0, 1e-3) and near(reply["rms_after"], 1 / math.sqrt(2), 5e-2), reply
    reply = h.call("signal", "bandpass", sample_rate=1000, cutoff=[40, 60])
    assert reply["note"].startswith("filter design only"), reply
    assert code(h.call("signal", "lowpass", sample_rate=1000, cutoff=600)) == "invalid_arguments"
    assert code(h.call("signal", "lowpass", values=[1, 2, 3], cutoff=1)) == "invalid_arguments"
    assert code(h.call("signal", "bandpass", sample_rate=1000, cutoff=50)) == "invalid_arguments"
    reply = h.call("signal", "highpass", values=["sin(2*pi*5*t) + sin(2*pi*200*t)"], sample_rate=1000, duration=1, cutoff=50)
    assert reply["success"] and near(reply["rms_after"], 1 / math.sqrt(2), 5e-2), reply


def test_peak_detection(h):
    reply = h.call("signal", "peaks", values=[0, 1, 0, 2, 0, 3, 0, 1, 0])
    assert [p["index"] for p in reply["peaks"]] == [1, 3, 5, 7] and reply["count"] == 4, reply
    reply = h.call("signal", "peaks", values=[0, 1, 0, 2, 0, 3, 0, 1, 0], height=1.5, sample_rate=10)
    assert [(p["index"], p["time"]) for p in reply["peaks"]] == [(3, 0.3), (5, 0.5)], reply
    reply = h.call("signal", "psd", expression="sin(2*pi*60*t)", sample_rate=1024, duration=2)
    assert near(reply["peaks"][0]["hz"], 60, 0.05) and near(reply["total_power"], 0.5, 0.02), reply


def test_convolution_and_correlation(h):
    assert h.call("signal", "convolve", values=[1, 2, 3], other=[0, 1, 0.5])["values"] == [0.0, 1.0, 2.5, 4.0, 1.5]
    reply = h.call("signal", "correlate", values=[0, 0, 1, 2, 1, 0], other=[1, 2, 1])
    assert reply["best_lag"] == 2 and reply["value_at_best_lag"] == 6.0, reply


def test_statistics(h):
    reply = h.call("stats", "describe", values=[2, 4, 4, 4, 5, 5, 7, 9])
    assert reply["mean"] == 5.0 and reply["median"] == 4.5 and near(reply["std"], 2.138089935299395), reply
    reply = h.call("stats", "ttest", x=[5.1, 4.9, 5.6, 5.8, 6.0], mean=5)
    assert near(reply["statistic"], 2.3040737315, 1e-8) and near(reply["p_value"], 0.0825682967, 1e-8), reply
    reply = h.call("stats", "correlation", x=[1, 2, 3, 4, 5], y=[2, 4, 6, 8, 10.5])
    assert reply["r"] > 0.99 and reply["method"] == "pearson", reply
    reply = h.call("stats", "regression", x=[1, 2, 3, 4], y=[3, 5, 7, 9])
    assert near(reply["slope"], 2) and near(reply["intercept"], 1) and near(reply["r_squared"], 1), reply
    reply = h.call("stats", "distribution", name="normal", parameters={"mean": 0, "std": 1}, at=1.96)
    assert near(reply["cdf"], 0.9750021048517795), reply
    assert near(h.call("stats", "distribution", name="normal", x=[1.96])["cdf"], 0.9750021048517795)
    reply = h.call("stats", "distribution", name="normal", p=0.975)
    assert near(reply["quantile"], 1.959963984540054), reply
    reply = h.call("stats", "distribution", name="binomial", parameters={"n": 10, "p": 0.5}, between=[4, 6])
    assert near(reply["probability_between"], 0.65625), reply
    reply = h.call("stats", "confidence_interval", values=[5.1, 4.9, 5.6, 5.8, 6.0])
    assert near(reply["lower"], 4.901592446, 1e-8) and near(reply["upper"], 6.058407554, 1e-8), reply
    reply = h.call("stats", "percentile", values=[1, 2, 3, 4, 5], q=[50, 100])
    assert [p["value"] for p in reply["percentiles"]] == [3.0, 5.0], reply
    reply = h.call("stats", "ttest", x=[1, 2, 3, 4], y=[2, 3, 4, 6], paired=True)
    assert reply["test"] == "paired" and reply["p_value"] < 0.05, reply


def test_statistics_reject_undefined_cases(h):
    assert code(h.call("stats", "correlation", x=[1, 1, 1], y=[1, 2, 3])) == "invalid_arguments"
    assert code(h.call("stats", "ttest", x=[1, 1, 1], mean=1)) == "invalid_arguments"
    assert code(h.call("stats", "ttest", x=[1, 2, 3])) == "invalid_arguments"
    assert code(h.call("stats", "describe", values=[])) == "invalid_arguments"
    assert code(h.call("stats", "distribution", name="normal", parameters={"std": -1}, x=0)) == "invalid_arguments"
    assert code(h.call("stats", "distribution", name="t", x=0)) == "invalid_arguments"


# failures that must not look like results

def test_nan_and_infinity(h):
    assert code(h.call("stats", "describe", values=[1, "nan", 3])) == "non_finite_input"
    assert code(h.call("linalg", "determinant", matrix=[[1, "inf"], [0, 1]])) == "non_finite_input"
    assert code(h.call("compute", "evaluate", expression="log(x)", data={"x": [1, 0, -1]})) == "non_finite_result"
    assert code(h.call("compute", "evaluate", expression="1/x", data={"x": 0})) == "non_finite_result"


def test_singular_matrices(h):
    assert code(h.call("linalg", "solve", matrix=[[1, 2], [2, 4]], other=[1, 2])) == "singular_matrix"
    assert code(h.call("linalg", "inverse", matrix=[[1, 2], [2, 4]])) == "singular_matrix"
    assert h.call("linalg", "determinant", matrix=[[1, 2], [2, 4]])["determinant"] == 0.0
    reply = h.call("linalg", "solve", matrix=[[1, 1], [1, 1.00000000001]], other=[2, 2])
    assert reply["success"] and "ill-conditioned" in reply["warnings"][0], reply


def test_invalid_dimensions(h):
    assert code(h.call("linalg", "solve", matrix=[[1, 2, 3], [4, 5, 6]], other=[1, 2])) == "invalid_dimensions"
    assert code(h.call("linalg", "solve", matrix=[[1, 2], [3, 4]], other=[1, 2, 3])) == "invalid_dimensions"
    assert code(h.call("linalg", "multiply", matrix=[[1, 2]], other=[[1, 2]])) == "invalid_dimensions"
    assert code(h.call("linalg", "determinant", matrix=[[1, 2], [3]])) == "invalid_dimensions"
    assert code(h.call("linalg", "determinant", matrix=[1, 2, 3])) == "invalid_dimensions"
    assert code(h.call("linalg", "eigen", matrix=[[1, 2, 3], [4, 5, 6]])) == "invalid_dimensions"


def test_limits(h):
    assert code(h.call("linalg", "rank", matrix={"random": [3000, 3000]})) == "limit_exceeded"
    assert code(h.call("linalg", "rank", matrix={"random": [1001, 5]})) == "limit_exceeded"
    assert code(h.call("stats", "describe", values={"linspace": [0, 1, 5000000]})) == "limit_exceeded"
    assert code(h.call("signal", "fft", expression="sin(t)", sample_rate=1000000, duration=10)) == "limit_exceeded"
    assert code(h.call("ode", "solve_ivp", equations=["y' = y"], initial_conditions={"y": 1}, interval=[0, 1],
                       points=1000000)) == "limit_exceeded"
    assert code(h.call("compute", "integrate", expression="x*y*z*w",
                       bounds={"x": [0, 1], "y": [0, 1], "z": [0, 1], "w": [0, 1]})) == "invalid_arguments"
    reply = h.call("stats", "describe", values={"linspace": [0, 1, 100001]})
    assert reply["n"] == 100001 and near(reply["mean"], 0.5), reply


def test_large_results_are_summarized(h):
    reply = h.call("compute", "evaluate", expression="x**2", data={"x": {"linspace": [0, 1, 10001]}})
    summary = reply["values"]
    assert summary["summarized"] is True and summary["shape"] == [10001] and len(summary["first"]) == 8, reply
    assert len(json.dumps(reply)) < 1200
    reply = h.call("compute", "evaluate", expression="x**2", data={"x": {"linspace": [0, 1, 101]}}, max_values=200)
    assert len(reply["values"]) == 101, reply
    reply = h.call("linalg", "eigen", matrix={"random": [300, 300], "seed": 3})
    assert reply["shape"] == [300, 300] and "eigenvectors" not in reply and len(json.dumps(reply)) < 2500, reply


def test_code_is_not_accepted(h):
    attempts = ["__import__('os').system('id')", "np.sin(x)", "scipy.optimize.minimize", "numpy.fft.fft(x)",
                "exec('1')", "eval('1')", "open('/etc/passwd')", "x.__class__", "lambda x: x", "import os",
                "[i for i in x]", "getattr(x, 'y')", "pickle.loads(x)", "ctypes.CDLL('x')"]
    for text in attempts:
        for reply in (h.call("compute", "evaluate", expression=text),
                      h.call("compute", "integrate", expression=text, lower=0, upper=1),
                      h.call("compute", "root", expression=text, bracket=[0, 1]),
                      h.call("optimize", "minimize", expression=text, initial_guess={"x": 1}),
                      h.call("signal", "fft", expression=text, sample_rate=10, duration=1),
                      h.call("ode", "solve_ivp", equations=["y' = " + text], initial_conditions={"y": 1}, interval=[0, 1])):
            assert code(reply) in ("invalid_expression", "unknown_function", "invalid_arguments", "undefined_symbol"), (text, reply)
    assert code(h.call("linalg", "determinant", matrix=[["__import__('os')", 1], [1, 1]])) in ("invalid_expression", "unknown_function")
    assert code(h.call("compute", "evaluate", expression="x", code="import os")) == "invalid_arguments"
    assert code(h.call("compute", "run_python", expression="1")) == "invalid_arguments"
    assert code(h.call("stats", "describe", values={"pickle": "gASV"})) == "invalid_arguments"


def test_definitions(h):
    tools = h.rpc("tools/list")["result"]["tools"]
    assert [t["name"] for t in tools] == ["compute", "linalg", "optimize", "signal", "ode", "stats"]
    for tool in tools:
        schema = tool["inputSchema"]
        assert schema["additionalProperties"] is False and "operation" in schema["required"]
        assert "code" not in schema["properties"] and tool["annotations"]["readOnlyHint"] is True
    assert len(json.dumps(tools)) < 5600, len(json.dumps(tools))


def test_worker_starts_only_when_used(_):
    helper = Helper()
    try:
        helper.rpc("tools/list")
        assert helper.children() == [], "the worker must not start before the first call"
        assert helper.call("linalg", "determinant", matrix=[[1, 2], [3, 4]])["determinant"] == -2.0
        assert len(helper.children()) == 1
    finally:
        helper.close()


def test_sympy_worker_never_loads_numpy(_):
    helper = Helper("sympy")
    try:
        reply = helper.call("expression", "integrate", expression="exp(sin(x))", variable="x", lower="0", upper="1")
        assert reply["method"] == "numerical_integration", reply
        worker = helper.children()[0]
        mapped = subprocess.run(["/usr/bin/vmmap", worker], capture_output=True, text=True).stdout
        assert "site-packages" not in mapped or ("numpy" not in mapped and "scipy" not in mapped), "numpy is mapped"
        assert "Accelerate" not in mapped or "numpy" not in mapped
    finally:
        helper.close()


def test_threads_are_capped(_):
    helper = Helper()
    try:
        helper.call("linalg", "multiply", matrix={"random": [600, 600]}, other={"random": [600, 600], "seed": 2})
        worker = helper.children()[0]
        environment = subprocess.run(["/bin/ps", "eww", "-p", worker], capture_output=True, text=True).stdout
        assert "VECLIB_MAXIMUM_THREADS=1" in environment and "OMP_NUM_THREADS=1" in environment, environment[-400:]
        started, cpu_before = time.monotonic(), float(subprocess.run(
            ["/bin/ps", "-o", "time=", "-p", worker], capture_output=True, text=True).stdout.strip().split(":")[-1])
        for _ in range(3):
            helper.call("linalg", "svd", matrix={"random": [600, 600]})
        cpu_after = float(subprocess.run(["/bin/ps", "-o", "time=", "-p", worker], capture_output=True,
                                         text=True).stdout.strip().split(":")[-1])
        assert (cpu_after - cpu_before) / (time.monotonic() - started) < 1.3, "the worker used more than one core"
    finally:
        helper.close()


def test_timeout_and_recovery(_):
    helper = Helper(TOSH_SYMPY_TIMEOUT_MS=400)
    try:
        started = time.monotonic()
        # a Python callback per solver step over ~10^7 periods: seconds on any machine, unlike linear algebra
        reply = helper.call("ode", "solve_ivp", equations=["dy/dt = cos(1000*t)"], initial_conditions={"y": 0},
                            interval=[0, 100000])
        assert code(reply) == "timeout" and reply["timed_out"] is True and time.monotonic() - started < 8*SLOW, reply
        assert helper.call("linalg", "determinant", matrix=[[1, 2], [3, 4]])["determinant"] == -2.0
        workers = helper.children()
        assert len(workers) == 1
    finally:
        helper.close()
    alive = subprocess.run(["/bin/ps", "-p", workers[0]], capture_output=True, text=True).stdout
    assert workers[0] not in alive, "the worker outlived its supervisor"


def test_memory_limit(_):
    helper = Helper(TOSH_SYMPY_MEMORY_MB=64, TOSH_SYMPY_TIMEOUT_MS=20000)
    try:
        big = {name: {"linspace": [0, 1, 1000000]} for name in "abcdefgh"}
        # the server samples the footprint every 50 ms, which a plain sum can finish under: this holds the
        # arrays for about a second
        reply = helper.call("compute", "evaluate", data=big,
                            expression="LambertW(a) + LambertW(b) + LambertW(c) + LambertW(d) + LambertW(f) + LambertW(g) + LambertW(h)")
        assert code(reply) == "memory_limit", reply
        assert helper.call("linalg", "determinant", matrix=[[1, 2], [3, 4]])["determinant"] == -2.0
    finally:
        helper.close()


def test_idle_worker_is_released(_):
    helper = Helper(TOSH_SYMPY_IDLE_SECONDS=1)
    try:
        helper.call("stats", "describe", values=[1, 2, 3])
        assert len(helper.children()) == 1
        time.sleep(7)
        assert helper.children() == []
        assert helper.call("stats", "describe", values=[1, 2, 3])["success"] is True
    finally:
        helper.close()


def test_worker_is_confined(_):
    probe = r'''
import os, sys
sys.path.insert(0, sys.argv[1])
from tosh_sympy import worker
print(worker._sandbox())
sys.addaudithook(worker._audit)
import numpy, scipy.linalg, scipy.optimize, scipy.signal, scipy.stats, scipy.integrate, scipy.interpolate
def attempt(action):
    try:
        action()
        print("allowed")
    except Exception as error:
        print("denied")
attempt(lambda: open(sys.argv[2], "w"))
attempt(lambda: numpy.save(sys.argv[2], numpy.ones(3)))
attempt(lambda: numpy.load(os.path.expanduser("~/x.npy"), allow_pickle=True))
attempt(lambda: __import__("subprocess").run(["/usr/bin/true"]))
attempt(lambda: __import__("socket").create_connection(("127.0.0.1", 9), 1))
attempt(lambda: os.listdir(os.path.expanduser("~")))
attempt(lambda: __import__("ctypes").CDLL("/usr/lib/libz.dylib"))
attempt(lambda: scipy.linalg.solve(numpy.eye(3), numpy.ones(3)))
'''
    marker = os.path.join(os.environ.get("TMPDIR", "/tmp"), "tosh-scientific-write-marker")
    lines = subprocess.run([PYTHON, "-P", "-s", "-B", "-c", probe, RUNTIME, marker], capture_output=True, text=True,
                           env={"OPENBLAS_MAIN_FREE": "1", "VECLIB_MAXIMUM_THREADS": "2"}).stdout.split()
    inside = os.path.expanduser("~").startswith(RUNTIME)
    assert lines == ["True"] + ["denied"] * 5 + ["allowed" if inside else "denied", "denied", "allowed"], lines
    assert not os.path.exists(marker) and not os.path.exists(marker + ".npy")


def measure():
    started = time.monotonic()
    helper = Helper()
    ready = time.monotonic() - started
    started = time.monotonic()
    helper.call("linalg", "determinant", matrix=[[1, 2], [3, 4]])
    first = time.monotonic() - started
    warm = []
    for _ in range(20):
        started = time.monotonic()
        helper.call("linalg", "determinant", matrix=[[1, 2], [3, 4]])
        warm.append(time.monotonic() - started)
    helper.close()
    warm.sort()
    print(f"supervisor start {ready * 1000:.0f} ms, first call {first * 1000:.0f} ms, "
          f"warm call median {warm[len(warm) // 2] * 1000:.1f} ms")


def test_a_call_that_cannot_mean_what_it_says_is_refused(h):
    # a huge bound stands for infinity instead of hiding the function from the quadrature
    reply = h.call("compute", "integrate", expression="exp(-x**2)", lower=0, upper=1e300)
    assert near(reply["value"], math.sqrt(math.pi) / 2) and reply["warnings"], reply
    # values for a variable the expression does not have are reported, not silently dropped
    reply = h.call("compute", "evaluate", expression="exp(1)", variable="x", at=[2.718])
    assert near(reply["value"], math.e) and "not used" in reply["warnings"][0], reply
    assert near(h.call("compute", "evaluate", expression="pi**2/6", variable="pi", at=[3.14])["value"], math.pi ** 2 / 6)
    assert near(h.call("compute", "evaluate", expression="x**2", at=[3])["value"], 9.0)
    assert near(h.call("compute", "evaluate", expression="exp(1)", variable="x")["value"], math.e)
    assert code(h.call("signal", "fft", values=[1, 0, -1, 0, 1, 0, -1, 0], cutoff=8)) == "invalid_arguments"
    reply = h.call("signal", "convolve", values=[[1, 2, 3], [0, 1, 0.5]])
    assert reply["values"] == [0.0, 1.0, 2.5, 4.0, 1.5], reply
    reply = h.call("optimize", "minimize", expression="x**2 + 3*x + 5", initial_guess={"x": 0}, bounds={"x": [0, 10]})
    assert reply["solution"]["x"] == 0.0 and "bound" in reply["warnings"][0], reply
    # only a model linear in its parameters is fitted without a start
    reply = h.call("optimize", "curve_fit", expression="a*x + b", x=[0, 1, 2, 3], y=[1, 3, 5, 7])
    assert near(reply["parameters"]["a"], 2.0, 1e-6) and near(reply["parameters"]["b"], 1.0, 1e-6) and reply["warnings"], reply
    assert code(h.call("optimize", "curve_fit", expression="a*exp(b*x)", x=[0, 1, 2, 3], y=[1, 2, 4, 8])) == "missing_initial_guess"


def test_a_call_that_does_not_say_what_the_request_says_is_not_run(h):
    def ask(text, tool, operation, **arguments):
        return h.call(tool, operation, _trust=KEY, _source={"request": text}, **arguments)

    def refused(reply):
        return reply["success"] is False and reply["error"]["code"] == "transcription_mismatch" and reply["interpreted_input"]

    a, b = [3.1, 2.9, 3.4, 3.0], [3.6, 3.8, 3.5, 3.9]
    text = f"Run a two-sample t-test on {a} and {b}."
    assert refused(ask(text, "stats", "ttest", values=a, mean=3.6))
    assert refused(ask(text, "stats", "ttest", x=a[:3], y=b))
    assert refused(ask(text, "stats", "ttest", x=[3.1, 29, 3.4, 3.0], y=b))
    reply = ask(text, "stats", "ttest", x=a, y=b)
    assert reply["success"] and reply["result_kind"] == "approximate", reply
    assert "x: n=4 [3.1, 2.9, 3.4, 3]" in reply["interpreted_input"], reply
    text = "Low-pass filter sin(2*pi*3*t) + sin(2*pi*80*t) at 20 Hz, sampled at 500 Hz for 1 second."
    signal = dict(expression="sin(2*pi*3*t) + sin(2*pi*80*t)", sample_rate=500, duration=1)
    assert refused(ask(text, "signal", "lowpass", cutoff=200, **signal))
    assert refused(ask(text, "signal", "highpass", cutoff=20, **signal))
    assert ask(text, "signal", "lowpass", cutoff=20, **signal)["success"]
    text = "Simulate dx/dt = -2*y, dy/dt = x with x(0) = 1 and y(0) = 0 up to t = 2."
    system = dict(initial_conditions={"x": 1, "y": 0}, interval=[0, 2])
    assert refused(ask(text, "ode", "solve_ivp", equations=["dy/dt = -2*y", "dx/dt = x"], **system))
    assert refused(ask(text, "ode", "solve_ivp", equations=["dx/dt = -2*y"], **system))
    assert refused(ask(text, "ode", "solve_ivp", equations=["dx/dt = -2*y", "dy/dt = x"],
                       initial_conditions={"x": 1, "y": 1}, interval=[0, 2]))
    assert ask(text, "ode", "solve_ivp", equations=["dx/dt = -2*y", "dy/dt = x"], **system)["success"]
    text = "Numerically integrate 1/(1 + x^2) from 0 to infinity."
    assert refused(ask(text, "compute", "integrate", expression="1/(1 + x**2)", lower=0, upper=8))
    assert refused(ask(text, "compute", "integrate", expression="1/(1 - x**2)", lower=0, upper="oo"))
    assert near(ask(text, "compute", "integrate", expression="1/(1 + x**2)", lower=0, upper="oo")["value"], math.pi / 2)
    text = "Interpolate between the points x = [2, 4, 8], y = [10, 30, 20] at x = 5."
    assert refused(ask(text, "compute", "interpolate", x=[2, 4, 8], y=[10, 30, 20], at=[5], kind="pchip"))
    assert ask(text, "compute", "interpolate", x=[2, 4, 8], y=[10, 30, 20], at=[5])["values"] == [27.5]
    rows = [[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]]
    text = f"Solve {rows} x = [1.0, 2.0, 3.0]."
    assert refused(ask(text, "linalg", "solve", matrix=[r[:2] for r in rows], other=[1.0, 2.0, 3.0]))
    assert refused(ask(text, "linalg", "solve", matrix=rows[:2], other=[1.0, 2.0]))
    reply = ask(text, "linalg", "solve", matrix=rows, other=[1.0, 2.0, 3.0])
    assert reply["success"] and any(line.startswith("matrix: 3 x 3") for line in reply["interpreted_input"]), reply


REPORTED = r"""Calcula exactamente
\[
I=\int_{0}^{\infty}\frac{x^3}{e^x-1}\,dx
\]
Después:
1. expresa el resultado en forma exacta;
2. dame su valor decimal con 10 cifras decimales;
3. verifica el resultado mediante integración numérica independiente;
4. indica el error absoluto entre el valor exacto evaluado numéricamente y la integración numérica.
Usa las herramientas matemáticas/científicas disponibles cuando corresponda. No hagas el cálculo solo de memoria."""


def test_an_improper_integral_written_in_latex(h):
    def ask(context="", **bounds):
        return h.call("compute", "integrate", _trust=KEY, _source={"request": REPORTED, "context": context},
                      expression="x**3/(exp(x)-1)", variable="x", **bounds)

    for upper in ("oo", "inf"):
        reply = ask(lower=0, upper=upper)
        assert reply["success"] and abs(reply["value"] - math.pi ** 4 / 15) <= max(reply["error_estimate"], 1e-9), reply
        assert "limits: [0, +oo)" in reply["interpreted_input"] and not reply["warnings"], reply
    first = ask(lower=0, upper=100)
    assert first["error"]["code"] == "transcription_mismatch", first
    assert "goes to infinity" in first["reasons"][0] and not any("combine" in r or "has 2" in r for r in first["reasons"]), first
    # the same call again stays refused, also if its own refusal were taken for context
    assert ask(lower=0, upper=100)["error"]["code"] == "transcription_mismatch"
    assert ask(context=json.dumps(first), lower=0, upper=100)["error"]["code"] == "transcription_mismatch"
    assert near(h.call("compute", "integrate", expression="exp(x)", lower="-oo", upper=0)["value"], 1.0)


def test_infinite_limits_are_in_the_definition(h):
    compute = next(t for t in h.rpc("tools/list")["result"]["tools"] if t["name"] == "compute")
    for key in ("lower", "upper"):
        bound = compute["inputSchema"]["properties"][key]
        assert set(bound["type"]) == {"number", "string"} and '"oo"' in bound["description"], bound


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
            if helper.process.poll() is not None:
                helper = Helper(**SHARED)
    helper.close()
    measure()
    print(f"{len(tests) - failed} of {len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
