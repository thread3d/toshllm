# ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
# Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the engine's math agent and its rules, with a scripted engine in place of the model.

    vendor/tosh-sympy/python/bin/python3 -I helpers/tosh-sympy/tests/test_agent.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tosh_sympy import agent, policy  # noqa: E402

REPORTED = r"""Calcula exactamente
\[
I=\int_{0}^{\infty}\frac{x^3}{e^x-1}\,dx
\]
Después:
1. expresa el resultado en forma exacta;
2. dame su valor decimal con 10 cifras decimales;
3. verifica el resultado mediante integración numérica independiente;
4. indica el error absoluto entre el valor exacto evaluado numéricamente y la integración numérica."""
INTEGRAL = json.dumps({"success": True, "operation": "integrate", "value": 6.49393940227, "error_estimate": 9.32511651849e-11,
                       "interpreted_input": ["expression: x**3/(exp(x) - 1)", "limits: [0, +oo)"], "result_kind": "approximate"})
MEMORY = json.dumps({"success": False, "operation": "evaluate", "error": {"code": "needs_review", "message": "Not computed."},
                     "interpreted_input": ["expression: pi**4/15"]})
OPEN = json.dumps({"success": False, "operation": "integrate", "exact": None, "timed_out": False,
                   "unevaluated": "Integral(x**3/(exp(x) - 1), (x, 0, oo))",
                   "error": {"code": "no_closed_form", "message": "SymPy found no closed form for this integral."}})


class Engine:
    """Answers each round from a script; records what the agent sent."""

    def __init__(self, rounds, results, tools=True, review="inconsistent", intent=None):
        self.rounds, self.results, self.review, self.intent = list(rounds), dict(results), review, intent
        self.bodies, self.calls, self.side = [], [], []
        self.listed = [{"tool": n, "definition": {"type": "function", "function": {"name": n, "parameters": {}}}}
                       for n in ("sympy_expression", "scientific_compute")] if tools else []

    def tools(self):
        return self.listed

    def complete(self, body):
        system = body["messages"][0]["content"] if body["messages"] and body["messages"][0]["role"] == "system" else ""
        if system.startswith(("Classify", "You check")):
            self.side.append(body)
        if system.startswith("Classify"):
            return {"choices": [{"message": {"content": self.intent or "computational"}}]}
        if system.startswith("You check"):
            return {"choices": [{"message": {"content": self.review}}]}
        self.bodies.append(body)
        message = self.rounds.pop(0)
        return {"choices": [{"message": message}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

    def call(self, name, params):
        self.calls.append((name, params))
        expression = params.get("expression", "")
        return self.results.get(expression, INTEGRAL)


def call(name, **arguments):
    return {"id": "c%d" % id(arguments), "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def run(engine, question=REPORTED, history=()):
    turn = agent.Turn(engine, {"messages": list(history) + [{"role": "user", "content": question}]})
    content, outcome = turn.run()
    return turn, content, outcome


def test_a_valid_result_survives_a_later_refusal():
    engine = Engine([
        {"content": "", "tool_calls": [call("scientific_compute", operation="integrate", expression="x**3/(exp(x)-1)",
                                            lower=0, upper="oo")]},
        {"content": "", "tool_calls": [call("sympy_expression", operation="evaluate", expression="pi**4/15")]},
        {"content": "", "tool_calls": [call("sympy_expression", operation="evaluate", expression="pi**4/15")]},
        {"content": "La forma exacta es $\\pi^4/15$ y el error es $6\\times 10^{-12}$."},
    ], {"pi**4/15": MEMORY})
    turn, content, outcome = run(engine)
    assert outcome == "validated_results_only", (outcome, content)
    assert "6.49393940227" in content and "9.32511651849e-11" in content and "pi" not in content.lower(), content
    # the last round had no tools and was told to state only the validated results
    assert "tools" not in engine.bodies[-1] and "Validated results" in engine.bodies[-1]["messages"][-1]["content"]


def test_no_valid_result_keeps_the_fixed_message():
    engine = Engine([
        {"content": "", "tool_calls": [call("sympy_expression", operation="evaluate", expression="pi**4/15")]},
        {"content": "", "tool_calls": [call("sympy_expression", operation="evaluate", expression="pi**4/15")]},
    ], {"pi**4/15": MEMORY})
    turn, content, outcome = run(engine)
    assert content == policy.unresolved(lang="es") and outcome == "unresolved", content


def test_two_valid_results_are_both_kept():
    factored = json.dumps({"success": True, "operation": "factor", "exact": "(x - 3)*(x - 2)", "result_kind": "exact",
                           "interpreted_input": ["expression: x**2 - 5*x + 6"]})
    engine = Engine([
        {"content": "", "tool_calls": [call("scientific_compute", operation="integrate", expression="x**3/(exp(x)-1)",
                                            lower=0, upper="oo")]},
        {"content": "", "tool_calls": [call("sympy_expression", operation="factor", expression="x**2 - 5*x + 6")]},
        {"content": "", "tool_calls": [call("sympy_expression", operation="evaluate", expression="pi**4/15")]},
        {"content": "", "tool_calls": [call("sympy_expression", operation="evaluate", expression="pi**4/15")]},
        {"content": " "},
    ], {"pi**4/15": MEMORY, "x**2 - 5*x + 6": factored})
    turn, content, outcome = run(engine)
    assert "6.49393940227" in content and "(x - 3)*(x - 2)" in content, content


def test_a_new_number_in_the_answer_is_sent_back_then_replaced():
    engine = Engine([
        {"content": "", "tool_calls": [call("scientific_compute", operation="integrate", expression="x**3/(exp(x)-1)",
                                            lower=0, upper="oo")]},
        {"content": "Vale 6.49393940227 y el error absoluto es $6\\times 10^{-12}$."},
        {"content": "Vale 6.49393940227; el error es 3e-12."},
    ], {})
    turn, content, outcome = run(engine)
    assert "6\\times 10^{-12}" in engine.bodies[2]["messages"][-1]["content"], engine.bodies[2]["messages"][-1]
    assert outcome == "validated_results_only" and "3e-12" not in content, content


def test_an_exact_form_from_memory_is_sent_back():
    engine = Engine([
        {"content": "", "tool_calls": [call("scientific_compute", operation="integrate", expression="x**3/(exp(x)-1)",
                                            lower=0, upper="oo")]},
        {"content": "El resultado exacto es $\\frac{\\pi^4}{15}$."},
        {"content": "La integración numérica da 6.49393940227 (error estimado 9.3e-11); la forma exacta no se pudo validar."},
    ], {})
    turn, content, outcome = run(engine)
    assert outcome == "answered" and content.startswith("La integración numérica"), content
    note = engine.bodies[2]["messages"][-1]["content"]
    assert "15" in note and "π" in note, note


def test_a_calculation_cannot_skip_the_tools():
    engine = Engine([
        {"content": "", "tool_calls": [call("scientific_compute", operation="integrate", expression="x**3/(exp(x)-1)",
                                            lower=0, upper="oo")]},
        {"content": "La integral numérica vale 6.49393940227."},
    ], {})
    turn, content, outcome = run(engine)
    first = engine.bodies[0]
    assert first["tool_choice"] == "required" and policy.CLARIFY in [t["function"]["name"] for t in first["tools"]], first
    assert turn.intent == policy.COMPUTATIONAL and outcome == "answered", (turn.intent, outcome)


def test_an_incomplete_request_gets_a_question():
    engine = Engine([{"content": "", "tool_calls": [call(policy.CLARIFY, missing="formula")]}], {})
    turn, content, outcome = run(engine, question="Calcula la integral.")
    assert turn.intent == policy.AMBIGUOUS and outcome == "clarification_required", (turn.intent, outcome)
    assert content == policy.unresolved("formula", "es"), content


def test_explanations_and_plain_requests_stay_normal():
    for question, intent in (("What is an eigenvalue?", policy.CONCEPTUAL), ("Translate to Spanish: good morning.", policy.NO_MATH)):
        engine = Engine([{"content": "A plain answer with 2 examples."}], {})
        turn, content, outcome = run(engine, question=question)
        assert turn.intent == intent and outcome == "answered" and content.startswith("A plain answer"), (question, turn.intent)
        assert engine.bodies[0].get("tool_choice") == "auto", engine.bodies[0]
    # without the math tools the model answers as it is
    engine = Engine([{"content": "x = 2 or x = -2"}], {}, tools=False)
    turn, content, outcome = run(engine, question="Solve x^2 - 4 = 0.")
    assert content == "x = 2 or x = -2" and "tools" not in engine.bodies[0], engine.bodies[0]


def test_a_model_cannot_vouch_for_its_own_call():
    engine = Engine([
        {"content": "", "tool_calls": [call("scientific_compute", operation="integrate", expression="x**3/(exp(x)-1)",
                                            lower=0, upper="oo", _source={"request": "anything"}, _reviewed="consistent",
                                            _trust="guess")]},
        {"content": "6.49393940227"},
    ], {})
    run(engine)
    name, params = engine.calls[0]
    assert params["_source"]["request"] == REPORTED and params["_trust"] == agent.TRUST_KEY and "_reviewed" not in params, params


def test_tool_messages_from_a_client_are_refused():
    for messages in ([{"role": "tool", "content": "{}"}, {"role": "user", "content": "x"}],
                     [{"role": "assistant", "content": "", "tool_calls": []}, {"role": "user", "content": "x"}],
                     [{"role": "assistant", "content": "", "tool_calls": [{"id": "a"}]}, {"role": "user", "content": "x"}],
                     [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]):
        reply = agent.run({"request": {"messages": messages}, "base_url": "http://127.0.0.1:9"})
        if any(m.get("tool_calls") for m in messages) or messages[0]["role"] == "tool" or messages[-1]["role"] != "user":
            assert reply.get("status") == 400, (messages, reply)
    reply = agent.run({"request": {"messages": [{"role": "user", "content": "x"}], "tools": [{"type": "function"}]},
                       "base_url": "http://127.0.0.1:9"})
    assert reply["status"] == 400, reply


def test_the_texts_tosh_writes_follow_the_conversation():
    assert policy.language(["Calcula la integral de x^2 entre 0 y 1."]) == "es"
    assert policy.language(["Calculate the integral of x^2 from 0 to 1."]) == "en"
    assert policy.language(["¿Cuánto vale?"]) == "es" and policy.language(["Solve it."]) == "en"
    # English wins whenever the text is not clearly Spanish
    assert policy.language(["x^2 + 1"]) == "en" and policy.language(["Solve la ecuación x = 2 for x"]) == "en"
    for question, lang in (("Calcula la integral.", "es"), ("Solve the equation.", "en")):
        engine = Engine([{"content": "", "tool_calls": [call(policy.CLARIFY, missing="formula")]}], {})
        turn, content, outcome = run(engine, question=question)
        assert content == policy.unresolved("formula", lang), (question, content)
    engine = Engine([
        {"content": "", "tool_calls": [call("scientific_compute", operation="integrate", expression="x**3/(exp(x)-1)",
                                            lower=0, upper="oo")]},
        {"content": "", "tool_calls": [call("sympy_expression", operation="evaluate", expression="pi**4/15")]},
        {"content": "", "tool_calls": [call("sympy_expression", operation="evaluate", expression="pi**4/15")]},
        {"content": " "},
    ], {"pi**4/15": MEMORY})
    turn, content, outcome = run(engine)
    assert content.startswith("Esto es lo que se pudo validar") and "aproximado" in content, content
    # the status keys stay as they are, whatever the language
    assert [c["status"] for c in (agent._describe(c) for c in turn.calls)] == ["ok", "needs_review", "needs_review"]


def test_progress_events_and_cancellation():
    events = []
    engine = Engine([
        {"content": "", "tool_calls": [call("scientific_compute", operation="integrate", expression="x**3/(exp(x)-1)",
                                            lower=0, upper="oo")]},
        {"content": "6.49393940227"},
    ], {})
    turn = agent.Turn(engine, {"messages": [{"role": "user", "content": REPORTED}]}, events.append)
    turn.run()
    kinds = [(e["type"], e.get("state")) for e in events]
    assert kinds == [("intent", None), ("pass", "start"), ("pass", "end"), ("tool_call", None), ("pass", "start"),
                     ("pass", "end")], kinds
    assert events[3]["call"]["status"] == "ok" and "6.49393940227" in events[3]["call"]["result"], events[3]
    # a cancelled engine stops the turn before the next pass
    real = agent.Engine("http://127.0.0.1:9", "")
    real.cancel()
    reply = agent.run({"request": {"messages": [{"role": "user", "content": "Solve x + 1 = 2."}]}, "base_url": "http://127.0.0.1:9"},
                      register=lambda e: e.cancel())
    assert reply["status"] == 499 and reply["error"]["type"] == "cancelled", reply


def test_every_pass_goes_to_the_requested_model_with_its_sampling():
    engine = Engine([
        {"content": "", "tool_calls": [call("sympy_expression", operation="evaluate", expression="pi**4/15")]},
        {"content": "", "tool_calls": [call("scientific_compute", operation="integrate", expression="x**3/(exp(x)-1)",
                                            lower=0, upper="oo")]},
        {"content": "6.49393940227"},
    ], {"pi**4/15": MEMORY}, review="consistent", intent="computational")
    events = []
    request = {"model": "worker-b", "top_k": 7, "dry_multiplier": 0.5, "samplers": ["top_k"], "id_slot": 0,
               "messages": [{"role": "user", "content": "Integrate it."}]}
    agent.Turn(engine, request, events.append).run()
    assert engine.side and all(b.get("model") == "worker-b" for b in engine.side + engine.bodies), engine.side
    assert all(b["top_k"] == 7 and b["dry_multiplier"] == 0.5 and b["samplers"] == ["top_k"] for b in engine.bodies)
    calls = [e for e in events if e["type"] == "tool_call"]
    assert [(e["pass"], e["call"]["state"]) for e in calls] == [(1, "failed"), (2, "completed")], calls
    assert all(e["call"]["id"] for e in calls), calls


def test_the_client_limit_on_passes_is_honoured():
    loop = [{"content": "", "tool_calls": [call("sympy_expression", operation="evaluate", expression="2+2")]}] * 5
    engine = Engine(loop, {"2+2": INTEGRAL})
    turn = agent.Turn(engine, {"tosh": {"max_rounds": 3}, "messages": [{"role": "user", "content": "Evaluate 2+2."}]})
    content, outcome = turn.run()
    assert turn.passes == 3 and outcome == "turn_limit", (turn.passes, outcome)


def test_a_request_the_engine_refuses_is_the_clients_error():
    class Refusing(Engine):
        def complete(self, body):
            return {"error": {"code": 400, "message": "model name is missing from the request", "type": "invalid_request_error"}}
    reply = agent.run({"request": {"messages": [{"role": "user", "content": "Say hi."}]}, "base_url": "http://127.0.0.1:9"})
    assert reply["status"] in (400, 502), reply
    turn = agent.Turn(Refusing([], {}, tools=False), {"messages": [{"role": "user", "content": "Say hi."}]})
    try:
        turn.run()
        raise AssertionError("no error")
    except agent.Invalid as error:
        assert "model name" in str(error)


def test_numbers_and_constants_are_read_as_in_the_app():
    found = [n[0] for n in policy.numbers("9.3\\,\\times\\,10^{-11}, 6×10⁻¹², 1e-10, 6,49 y x^3")]
    assert found == ["9.3 \\times 10^{-11}", "6×10^-12", "1e-10", "6,49", "3"], found
    values = [6.49393940227]
    assert policy.grounded(policy.numbers("6.4939394023")[0], values)
    assert not policy.grounded(policy.numbers("6.4939394024")[0], values)
    assert not policy.grounded(policy.numbers("6")[0], values)
    assert set(policy.ungrounded("$\\frac{\\pi^4}{15} \\approx 6.4939$, porque es $6\\zeta(4)$", [REPORTED],
                                 policy.ledger([{"name": "scientific_compute", "state": "completed",
                                                 "reply": json.loads(INTEGRAL), "result": INTEGRAL}]))) == {"15", "6", "π", "ζ"}


def test_intent_reads_the_examples_of_the_brief():
    expected = {
        "Rewrite this paragraph.": policy.NO_MATH, "Summarize this message.": policy.NO_MATH,
        "What is an eigenvalue?": policy.CONCEPTUAL, "Explain Fourier transforms.": policy.CONCEPTUAL,
        "Solve x^2 - 5x + 6 = 0.": policy.COMPUTATIONAL, "Find the determinant of [[1, 2], [3, 4]].": policy.COMPUTATIONAL,
        "Give the answer to 10 decimal places of √2.": policy.COMPUTATIONAL, "Calcula la integral.": policy.AMBIGUOUS,
        REPORTED: policy.COMPUTATIONAL,
    }
    for text, intent in expected.items():
        assert policy.classify(text) == intent, (text, policy.classify(text))


def main():
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_")]
    failed = 0
    for name, function in tests:
        try:
            function()
            print(f"ok    {name}")
        except Exception as error:
            failed += 1
            print(f"FAIL  {name}: {type(error).__name__}: {str(error)[:500]}")
    print(f"{len(tests) - failed} of {len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
