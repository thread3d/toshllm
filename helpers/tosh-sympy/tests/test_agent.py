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
        self.bodies, self.calls, self.side, self.live = [], [], [], []
        self.listed = [{"tool": n, "definition": {"type": "function", "function": {"name": n, "parameters": {}}}}
                       for n in ("sympy_expression", "scientific_compute")] if tools else []

    def tools(self):
        return self.listed

    def context(self, model=None):
        return None

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

    def stream(self, body, on_delta):
        self.live.append(len(self.bodies))
        reply = self.complete(body)
        if not reply.get("choices"):
            return reply
        message = reply["choices"][0]["message"]
        for field in ("reasoning_content", "content"):
            if message.get(field):
                on_delta(field, message[field])
        return reply

    def call(self, name, params):
        self.calls.append((name, params))
        expression = params.get("expression", "")
        return self.results.get(expression, INTEGRAL)

    def call_checked(self, name, params, timeout=120):
        self.calls.append((name, params))
        return self.results.get(name, "unknown tool"), name in self.results


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


def test_an_attachment_the_agent_cannot_read_goes_back_to_the_client():
    for content in ([{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}],
                    [{"type": "text", "text": "Solve the equation in the photo."},
                     {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}]):
        # the app, which says it is one with its options, takes the turn back to its own loop
        reply = agent.run({"request": {"messages": [{"role": "user", "content": content}], "tosh": {"locale": "en"}},
                           "base_url": "http://127.0.0.1:9"})
        assert reply["tosh"]["outcome"] == "declined" and reply["choices"][0]["message"]["content"] == "", reply
        # any other client, as the web chat, gets the model's answer to what it sent, never an empty one
        engine = Engine([{"content": "A cat on a sofa."}], {}, tools=False)
        original = agent.Engine
        agent.Engine = lambda *args: engine
        events = []
        try:
            reply = agent.run({"request": {"messages": [{"role": "user", "content": content}], "max_tokens": 50},
                               "base_url": "http://127.0.0.1:9"}, events.append)
        finally:
            agent.Engine = original
        assert reply["tosh"]["outcome"] == "answered" and reply["choices"][0]["message"]["content"] == "A cat on a sofa."
        assert engine.bodies[0]["messages"][0]["content"] == content and engine.bodies[0]["max_tokens"] == 50
        assert {"type": "delta", "content": "A cat on a sofa."} in events, events


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


def _events(engine, question):
    events = []
    turn = agent.Turn(engine, {"messages": [{"role": "user", "content": question}]}, events.append)
    content, outcome = turn.run()
    return events, content, outcome


def test_a_turn_that_needs_no_tools_streams_its_answer():
    engine = Engine([{"content": "An eigenvalue is a scale factor."}], {})
    events, content, outcome = _events(engine, "What is an eigenvalue?")
    # the tools were offered, so the text is a draft until the pass ends without calls
    drafts = [e for e in events if e["type"] == "draft"]
    assert "".join(e.get("content", "") for e in drafts) == content and outcome == "answered", events
    assert not any(e["type"] == "delta" and e.get("content") for e in events), events
    assert [e["live"] for e in events if e["type"] == "pass" and e["state"] == "start"] == [True], events
    assert not any(e["type"] == "retract" for e in events), events
    # without the math tools every answer goes out as it is generated
    engine = Engine([{"content": "x = 2 or x = -2"}], {}, tools=False)
    events, content, outcome = _events(engine, "Solve x^2 - 4 = 0.")
    assert [e.get("content") for e in events if e["type"] == "delta"] == ["x = 2 or x = -2"], events


def test_a_calculation_is_never_streamed():
    engine = Engine([
        {"content": "", "tool_calls": [call("scientific_compute", operation="integrate", expression="x**3/(exp(x)-1)",
                                            lower=0, upper="oo")]},
        {"content": "6.49393940227"},
    ], {})
    events, content, outcome = _events(engine, REPORTED)
    assert engine.live == [] and not any(e["type"] == "delta" for e in events), events


def test_text_streamed_before_a_math_call_is_retracted():
    engine = Engine([
        {"content": "Let me check with 3 terms.", "tool_calls": [call("sympy_expression", operation="factor",
                                                                       expression="x**2 - 5*x + 6")]},
        {"content": "It factors as (x - 3)*(x - 2)."},
    ], {"x**2 - 5*x + 6": json.dumps({"success": True, "operation": "factor", "exact": "(x - 3)*(x - 2)",
                                      "result_kind": "exact", "interpreted_input": ["expression: x**2 - 5*x + 6"]})})
    events, content, outcome = _events(engine, "What is an eigenvalue?")
    kinds = [e["type"] for e in events]
    assert kinds.index("retract") == kinds.index("tool_call") - 1, kinds
    # the retracted text was only ever a draft, and once a math tool has run the answer is not streamed at all
    assert engine.live == [0] and kinds.count("draft") == 1 and "delta" not in kinds, (engine.live, kinds)
    assert content == "It factors as (x - 3)*(x - 2).", content


def test_a_streamed_reply_is_read_like_a_complete_one():
    import http.server
    import threading
    chunks = [
        {"choices": [{"delta": {"role": "assistant", "content": "Hi"}}]},
        {"choices": [{"delta": {"reasoning_content": "hm"}}]},
        {"choices": [{"delta": {"content": " there"}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "a", "function": {"name": "sympy_", "arguments": "{\"x\""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "expression", "arguments": ": 1}"}}]}}]},
        {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}},
    ]
    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in chunks:
                self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.handle_request, daemon=True).start()
    try:
        pieces = []
        reply = agent.Engine(f"http://127.0.0.1:{server.server_port}", "").stream({"messages": []},
                                                                                   lambda f, t: pieces.append((f, t)))
    finally:
        server.server_close()
    message = reply["choices"][0]["message"]
    assert seen[0]["stream"] is True and seen[0]["stream_options"] == {"include_usage": True}, seen
    assert message["content"] == "Hi there" and message["reasoning_content"] == "hm", message
    assert message["tool_calls"] == [{"id": "a", "type": "function",
                                      "function": {"name": "sympy_expression", "arguments": '{"x": 1}'}}], message
    assert reply["usage"] == {"prompt_tokens": 7, "completion_tokens": 3}, reply
    assert pieces == [("content", "Hi"), ("reasoning_content", "hm"), ("content", " there")], pieces
    # the reason it stopped comes back too, or an answer streamed until its tokens ran out passes for a whole one
    chunks[-1:] = [{"choices": [{"delta": {}, "finish_reason": "length"}]}, chunks[-1]]
    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.handle_request, daemon=True).start()
    try:
        reply = agent.Engine(f"http://127.0.0.1:{server.server_port}", "").stream({"messages": []}, lambda f, t: None)
    finally:
        server.server_close()
    assert reply["choices"][0]["finish_reason"] == "length", reply


def test_a_cancel_stops_every_request_in_flight():
    import http.server
    import socketserver
    import threading
    import time
    arrived = threading.Semaphore(0)
    release = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            arrived.release()
            release.wait(10)
            try:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")
            except OSError:
                pass

        def log_message(self, *args):
            pass

    class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True

    server = Server(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    engine = agent.Engine(f"http://127.0.0.1:{server.server_port}", "")
    outcomes = []

    def ask():
        try:
            engine.complete({"messages": []})
            outcomes.append("answered")
        except agent.Cancelled:
            outcomes.append("cancelled")

    threads = [threading.Thread(target=ask) for _ in range(2)]
    try:
        for t in threads:
            t.start()
        assert arrived.acquire(timeout=5) and arrived.acquire(timeout=5), "both requests should be in flight"
        start = time.monotonic()
        engine.cancel()
        for t in threads:
            t.join(5)
        assert outcomes == ["cancelled", "cancelled"], outcomes
        assert time.monotonic() - start < 2, "a cancelled request waited for its answer"
    finally:
        release.set()
        server.shutdown()
        server.server_close()


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


def _run_with(engine, question):
    """agent.run with `engine` in place of the HTTP one."""
    original = agent.Engine
    agent.Engine = lambda *args: engine
    try:
        return agent.run({"request": {"messages": [{"role": "user", "content": question}]}, "base_url": "http://x:1"})
    finally:
        agent.Engine = original


def test_an_engine_failure_is_said_as_such_and_never_as_an_unvalidated_answer():
    class TooBig(Engine):
        def complete(self, body):
            return {"error": {"code": 400, "type": "exceed_context_size_error", "n_prompt_tokens": 22300, "n_ctx": 16384,
                              "message": "request (22300 tokens) exceeds the available context size (16384 tokens)"}}
    reply = _run_with(TooBig([], {}, tools=False), "Say hi.")
    assert (reply["status"], reply["error"]["type"], reply["error"]["n_ctx"]) == (400, "exceed_context_size_error", 16384)
    assert "22300 tokens" in reply["error"]["message"], reply

    class Broken(Engine):
        def complete(self, body):
            if body["messages"][0]["content"].startswith("You check"):
                return {"error": {"code": 500, "type": "server_error", "message": "Compute error."}}
            return super().complete(body)
    reply = _run_with(Broken([{"content": "", "tool_calls": [call("sympy_expression", expression="pi**4/15")]}],
                             {"pi**4/15": MEMORY}), REPORTED)
    assert (reply["status"], reply["error"]["type"]) == (503, "engine_gpu_error"), reply
    # in the user's language, with the engine's own words beside it
    assert "no terminó" in reply["error"]["message"] and reply["error"]["engine_message"] == "Compute error.", reply


def test_an_answer_cut_at_its_token_limit_says_it_is_incomplete():
    class Cut(Engine):
        def complete(self, body):
            reply = super().complete(body)
            reply["choices"][0]["finish_reason"] = "length"
            return reply
    reply = _run_with(Cut([{"content": "Once upon a time"}], {}, tools=False), "Tell me a long story.")
    assert reply["choices"][0]["finish_reason"] == "length", reply
    assert reply["choices"][0]["message"]["content"].endswith("so it is incomplete._"), reply
    assert reply["tosh"]["outcome"] == "cut_at_limit"


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


def with_approved(servers, test):
    saved = agent.APPROVED
    agent.APPROVED = tuple(name + "_" for name in servers)
    try:
        test()
    finally:
        agent.APPROVED = saved


def gis_engine(rounds, math_tools):
    engine = Engine(rounds, {"gis_count": "42 features"}, tools=math_tools, intent=policy.NO_MATH)
    engine.listed.append({"tool": "gis_count",
                          "definition": {"type": "function", "function": {"name": "gis_count", "parameters": {}}}})
    return engine


def test_an_approved_server_answers_with_the_arguments_it_was_given():
    def check():
        engine = gis_engine([{"content": "", "tool_calls": [call("gis_count", layer="parks", _trust="x")]},
                             {"content": "The layer has 42 parks."}], math_tools=False)
        turn, content, outcome = run(engine, "How many parks are in the city layer?")
        assert (content, outcome) == ("The layer has 42 parks.", "answered"), (content, outcome)
        assert engine.calls == [("gis_count", {"layer": "parks"})], engine.calls
        assert [agent._describe(c)["status"] for c in turn.calls] == ["ok"], turn.calls
        assert engine.bodies[0]["tool_choice"] == "auto" and "gis_count" in json.dumps(engine.bodies[0]["tools"])
    with_approved(["gis"], check)


def test_a_server_nobody_approved_stays_out():
    def check():
        engine = gis_engine([{"content": "", "tool_calls": [call("gis_count", layer="parks")]},
                             {"content": "I could not count them."}], math_tools=True)
        turn, _, _ = run(engine, "How many parks are in the city layer?")
        assert engine.calls == [], engine.calls
        assert "no such tool" in turn.calls[0]["result"], turn.calls[0]
    with_approved([], check)


def test_numbers_from_an_approved_server_count_as_sources_beside_math():
    def check():
        engine = gis_engine([
            {"content": "", "tool_calls": [call("gis_count", layer="parks")]},
            {"content": "", "tool_calls": [call("scientific_compute", operation="integrate", expression="x**3/(exp(x)-1)",
                                                lower=0, upper="oo")]},
            {"content": "There are 42 parks and the integral is 6.49393940227."}], math_tools=True)
        engine.intent = policy.COMPUTATIONAL
        _, content, outcome = run(engine, REPORTED + "\nTambién: ¿cuántos features tiene la capa parks?")
        assert outcome == "answered" and "42 parks" in content, (content, outcome)
    with_approved(["gis"], check)


def test_a_missing_input_for_an_approved_tool_gets_a_question():
    def check():
        engine = gis_engine([{"content": "", "tool_calls": [call(policy.CLARIFY, missing=["the layer"])]}], math_tools=False)
        turn, content, outcome = run(engine, "How many features does the layer have?")
        assert outcome == "clarification_required" and engine.calls == [], (outcome, engine.calls)
        assert policy.CLARIFY in json.dumps(engine.bodies[0]["tools"])
    with_approved(["gis"], check)


def test_malformed_or_unknown_calls_never_run():
    def check():
        broken = {"id": "x1", "type": "function", "function": {"name": "gis_count", "arguments": "{\"layer\": \"par"}}
        engine = gis_engine([{"content": "", "tool_calls": [broken, call("gis_delete", layer="parks"),
                                                            call("exec_shell_command", command="ls")]},
                             {"content": "I could not run them."}], math_tools=False)
        turn, _, _ = run(engine, "How many parks are in the city layer?")
        assert engine.calls == [], engine.calls
        assert [c["state"] for c in turn.calls] == ["failed"] * 3, turn.calls
        assert "invalid_arguments" in turn.calls[0]["result"] and "no such tool" in turn.calls[1]["result"]
        # the next pass sends the cut call as an empty object, or the engine refuses the whole history
        sent = next(m for m in engine.bodies[1]["messages"] if m.get("tool_calls"))["tool_calls"]
        assert [c["function"]["arguments"] for c in sent][:2] == ["{}", json.dumps({"layer": "parks"})], sent
    with_approved(["gis"], check)


def test_an_approved_tool_never_gets_a_value_nobody_gave():
    def check():
        engine = gis_engine([{"content": "", "tool_calls": [call("gis_count", layer="layer_name")]},
                             {"content": "", "tool_calls": [call(policy.CLARIFY, missing="data")]}], math_tools=False)
        engine.listed[-1]["definition"]["function"]["parameters"] = {
            "type": "object", "properties": {"layer": {"type": "string"}, "unit": {"enum": ["km", "mi"]}}}
        turn, content, outcome = run(engine, "How many features does the layer have?")
        assert engine.calls == [] and outcome == "clarification_required", (engine.calls, outcome)
        assert content == "To answer, I need you to tell me `layer`.", content
        assert "needs_user_input" in turn.calls[0]["result"], turn.calls[0]
        turn = agent.Turn(engine, {"messages": [{"role": "user", "content": "Count the parks layer, 3 times"}]})
        turn.listed, turn.definitions = {"gis_count"}, {"gis_count": engine.listed[-1]["definition"]}
        assert turn._ungiven("gis_count", {"layer": "Parks", "unit": "km", "times": 3, "query": "green areas near"}) == []
        assert turn._ungiven("gis_count", {"layer": "roads", "times": 7}) == ["layer='roads'", "times=7"]
        assert turn._ungiven("gis_count", {"layer": "layer"}) == ["layer='layer'"]
        assert turn._ungiven("gis_count", {"layer": "<parks>"}) == ["layer='<parks>'"]
        assert agent._describe(dict(turn.calls[0], name="gis_count") if turn.calls else {
            "name": "gis_count", "arguments": "{}", "state": "failed",
            "reply": {"success": False, "error": {"code": "needs_user_input"}}})["status"] == "needs_user_input"
    with_approved(["gis"], check)


def test_the_review_sees_what_an_approved_server_returned():
    def check():
        engine = gis_engine([
            {"content": "", "tool_calls": [call("gis_count", layer="parks")]},
            {"content": "", "tool_calls": [call("sympy_expression", operation="evaluate", expression="pi**4/15")]},
            {"content": "Done."}], math_tools=True)
        engine.results["pi**4/15"] = MEMORY
        engine.intent, engine.review = policy.COMPUTATIONAL, "consistent"
        run(engine, "Square the number of features of the parks layer.")
        review = [b for b in engine.side if b["messages"][0]["content"].startswith("You check")]
        assert review and "42 features" in review[0]["messages"][1]["content"], engine.side
        engine = Engine([{"content": "", "tool_calls": [call("sympy_expression", operation="evaluate", expression="pi**4/15")]},
                         {"content": "Done."}], {"pi**4/15": MEMORY}, review="consistent")
        run(engine, "Evaluate pi^4/15.")
        review = [b for b in engine.side if b["messages"][0]["content"].startswith("You check")]
        assert "RETURNED BY THE TOOLS" not in review[0]["messages"][1]["content"]
    with_approved(["gis"], check)


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


def test_the_engine_reads_every_address_the_server_hands_it():
    for url, host in (("http://127.0.0.1:8080", "127.0.0.1"), ("http://localhost:8080", "localhost"),
                      ("http://[::1]:8080", "::1"), ("http://[fe80::1%25lo0]:8080", "fe80::1%lo0")):
        engine = agent.Engine(url, "")
        assert (engine.host, engine.port) == (host, 8080), (url, engine.host, engine.port)


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


class Sized(Engine):
    """A model with a context of `n_ctx` tokens, a token every four characters: a prompt that does not fit is refused
    as the engine refuses it, with what it counted."""

    def __init__(self, rounds, n_ctx, counts=True, **kwargs):
        super().__init__(rounds, {}, tools=False, **kwargs)
        self.n_ctx, self.counts, self.counted = n_ctx, counts, 0

    def context(self, model=None):
        return self.n_ctx

    def tokens(self, body):
        return sum(len(agent._text(m.get("content"))) for m in body["messages"]) // 4

    def count(self, body):
        self.counted += 1
        return self.tokens(body) if self.counts else None

    def complete(self, body):
        tokens = self.tokens(body)
        if tokens >= self.n_ctx:
            return {"error": {"code": 400, "type": "exceed_context_size_error", "n_prompt_tokens": tokens,
                              "n_ctx": self.n_ctx, "message": f"request ({tokens} tokens) exceeds the available context "
                                                              f"size ({self.n_ctx} tokens)"}}
        reply = super().complete(body)
        reply["usage"] = {"prompt_tokens": tokens, "completion_tokens": 5}
        return reply


def _long(engine, turns=10, size=400, question="And the last one?"):
    history = []
    for n in range(turns):
        history += [{"role": "user", "content": f"Question {n}: " + "x" * size},
                    {"role": "assistant", "content": f"Answer {n}: " + "y" * size}]
    original = agent.Engine
    agent.Engine = lambda *args: engine
    try:
        return agent.run({"request": {"messages": [{"role": "system", "content": "Be brief."}] + history +
                                      [{"role": "user", "content": question}], "max_tokens": 200},
                          "base_url": "http://x:1"})
    finally:
        agent.Engine = original


def test_a_conversation_that_no_longer_fits_leaves_out_its_oldest_exchanges_and_says_so():
    engine = Sized([{"content": "The last one."}], 1000)
    reply = _long(engine)
    (sent,) = engine.bodies
    messages = sent["messages"]
    # it fits with room for the answer, goes on from a user message, and the model is told what is missing
    assert engine.tokens(sent) + 200 <= 1000, engine.tokens(sent)
    assert messages[0]["role"] == "system" and messages[0]["content"].startswith("Be brief.\n\nThe ")
    assert messages[1]["role"] == "user" and messages[-1]["content"].startswith("And the last one?")
    left = reply["tosh"]["context"]["left_out"]
    assert left > 0 and left % 2 == 0 and len(messages) == 2 + 20 - left, (left, len(messages))
    assert "Question 9" in json.dumps(messages) and "Question 0" not in json.dumps(messages)
    content = reply["choices"][0]["message"]["content"]
    assert content.startswith("The last one.") and f"The {left} oldest messages" in content, content
    # what the whole conversation takes, which no longer fits: a client that summarizes knows it has to
    assert reply["tosh"]["context"]["n_ctx"] == 1000 and reply["tosh"]["context"]["used"] > 1000, reply["tosh"]


def test_a_conversation_that_fits_is_sent_whole_and_not_counted():
    engine = Sized([{"content": "Fine."}], 100000)
    reply = _long(engine, turns=3)
    assert len(engine.bodies[0]["messages"]) == 1 + 6 + 1 and engine.counted == 0
    assert reply["tosh"]["context"]["left_out"] == 0 and "oldest messages" not in reply["choices"][0]["message"]["content"]


def test_without_a_count_the_engines_own_refusal_says_how_much_to_leave_out():
    engine = Sized([{"content": "The last one."}], 1000, counts=False)
    reply = _long(engine)
    assert reply["choices"][0]["message"]["content"].startswith("The last one."), reply
    assert reply["tosh"]["context"]["left_out"] > 0 and engine.tokens(engine.bodies[0]) < 1000


def test_a_message_that_alone_does_not_fit_is_a_clear_error():
    engine = Sized([{"content": "never"}], 1000)
    reply = _long(engine, turns=2, question="Read this: " + "z" * 6000)
    assert (reply["status"], reply["error"]["type"], reply["error"]["n_ctx"]) == (400, "exceed_context_size_error", 1000)


def test_an_answer_cut_by_a_full_context_is_told_from_one_cut_at_its_limit():
    class Cut(Sized):
        def complete(self, body):
            reply = super().complete(body)
            reply["choices"][0]["finish_reason"] = "length"
            reply["usage"]["completion_tokens"] = self.n_ctx - reply["usage"]["prompt_tokens"] - 1
            return reply
    reply = _long(Cut([{"content": "Once upon"}], 1000), turns=1)
    assert reply["tosh"]["outcome"] == "cut_at_context", reply["tosh"]
    assert reply["choices"][0]["message"]["content"].endswith(
        "cut when the conversation filled the model's context, so it is incomplete._")
    assert reply["choices"][0]["finish_reason"] == "length"


def test_an_operation_a_service_only_accepted_is_never_said_to_be_done():
    for text, accepted in (('{"status": "accepted", "job_id": "j-17"}', True),
                           ('{"result": {"jobStatus": "Submitted", "jobId": "a1"}}', True),
                           ('{"status": "QUEUED"}', True),
                           ('{"status": "completed", "edits": 3}', False),
                           ('{"success": true, "message": "the job was accepted by the team"}', False),
                           ("accepted", False)):
        assert agent.accepted_only(text) is accepted, text

    class Service(Engine):
        def call_checked(self, name, params, timeout=120):
            self.calls.append((name, params))
            return '{"status": "accepted", "job_id": "j-17"}', True
    engine = Service([{"content": "", "tool_calls": [call("ops_update", id="7")]}, {"content": "Update j-17 is pending."}],
                     {}, tools=False)
    engine.listed = [{"tool": "ops_update", "definition": {"type": "function", "function": {
        "name": "ops_update", "parameters": {"type": "object", "properties": {"id": {"type": "string"}}}}}}]
    original_approved = agent.APPROVED
    agent.APPROVED = ("ops_",)
    try:
        reply = _run_with(engine, "Update record 7.")
    finally:
        agent.APPROVED = original_approved
    tool_message = [m for m in engine.bodies[-1]["messages"] if m["role"] == "tool"][0]["content"]
    assert tool_message.startswith(agent.ACCEPTED_NOTE), tool_message
    assert reply["tosh"]["calls"][0]["status"] == "accepted", reply["tosh"]["calls"]


if __name__ == "__main__":
    main()
