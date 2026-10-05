# ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
# Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The engine's math agent. llama-server started with --mcp-agent hands every
/v1/chat/completions request that brings no tools to the "run" tool of this server, which
answers it with the math tools and the rules in policy.py, talking to the engine over HTTP:
raw chat completions to generate, /tools to compute.

Without the math tools on the engine, the request is answered by the model as it is.
"""

import http.client
import json
import os
import socket
import threading
import time
import urllib.parse
import uuid

from . import policy

TRUST_KEY = os.environ.get("TOSH_TRUST_KEY", "")
MAX_ROUNDS = 10
# what a client may set on a request that the agent passes on to each round
_GENERATION = ("temperature", "top_p", "top_k", "min_p", "seed", "repeat_penalty", "presence_penalty",
               "frequency_penalty", "max_tokens", "max_completion_tokens", "stop", "model", "chat_template_kwargs",
               "repeat_last_n", "typ_p", "typical_p", "dynatemp_range", "dynatemp_exponent", "xtc_probability",
               "xtc_threshold", "dry_multiplier", "dry_base", "dry_allowed_length", "dry_penalty_last_n",
               "samplers", "backend_sampling", "thinking_budget_tokens", "reasoning_control", "cache_prompt", "id_slot")

DEFINITION = {
    "name": "run",
    "description": "Answers an OpenAI chat completions request with the math tools. Called by the engine.",
    "inputSchema": {"type": "object", "properties": {"request": {"type": "object"}, "base_url": {"type": "string"},
                                                    "api_key": {"type": "string"}},
                    "required": ["request", "base_url"], "additionalProperties": False},
    "annotations": {"readOnlyHint": True},
}


class Invalid(Exception):
    pass


class Cancelled(Exception):
    pass


class Engine:
    def __init__(self, base_url, api_key):
        parsed = urllib.parse.urlparse(base_url)
        self.host, self.port = parsed.hostname or "127.0.0.1", parsed.port or 80
        self.api_key = api_key or ""
        self.cancelled = threading.Event()
        self._lock = threading.Lock()
        self._connection = None

    def cancel(self):
        """Stops the request in flight: the engine sees the connection close and stops generating."""
        with self._lock:
            self.cancelled.set()
            connection = self._connection
        if connection is not None and connection.sock is not None:
            try:
                connection.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def _send(self, method, path, body=None, timeout=3600):
        if self.cancelled.is_set():
            raise Cancelled()
        connection = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        # rounds of the agent are raw inference, never the agent again
        headers = {"Content-Type": "application/json", "X-Tosh-Agent": "off"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        # connected before it is shared, so a cancellation always finds a socket to close
        connection.connect()
        with self._lock:
            if self.cancelled.is_set():
                connection.close()
                raise Cancelled()
            self._connection = connection
        try:
            connection.request(method, path, None if body is None else json.dumps(body), headers)
            data = connection.getresponse().read()
        except (OSError, http.client.HTTPException):
            if self.cancelled.is_set():
                raise Cancelled()
            raise
        finally:
            with self._lock:
                self._connection = None
            connection.close()
        if self.cancelled.is_set():
            raise Cancelled()
        try:
            return json.loads(data or b"null")
        except ValueError:
            return {"error": {"message": "the engine answered something that is not JSON"}}

    def complete(self, body):
        return self._send("POST", "/v1/chat/completions", dict(body, stream=False))

    def tools(self):
        listed = self._send("GET", "/tools", timeout=60)
        return [t for t in listed if isinstance(t, dict) and policy.is_math(t.get("tool"))] if isinstance(listed, list) else []

    def call(self, name, params):
        reply = self._send("POST", "/tools", {"tool": name, "params": params}, timeout=120)
        if isinstance(reply, dict):
            text = reply.get("plain_text_response", reply.get("error"))
            if isinstance(text, str):
                return text
        return json.dumps(reply)


def _text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def _parse(request):
    """(system, history, question) from a chat request. Tool messages are refused: the tools run
    here, and a result sent by a client could otherwise pass for one Tosh computed."""
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise Invalid("messages must be a non-empty array")
    system, turns = [], []
    for message in messages:
        if not isinstance(message, dict):
            raise Invalid("each message must be an object")
        role = message.get("role")
        if role in ("tool", "function") or message.get("tool_calls") or message.get("function_call"):
            raise Invalid("tool calls and tool results are not accepted: the tools run inside Tosh")
        text = _text(message.get("content"))
        if role in ("system", "developer"):
            system.append(text)
        elif role in ("user", "assistant"):
            turns.append({"role": role, "content": text})
        else:
            raise Invalid(f"unknown role {str(role)[:20]!r}")
    if not turns or turns[-1]["role"] != "user" or not turns[-1]["content"].strip():
        raise Invalid("the last message must come from the user")
    return "\n\n".join(system), turns[:-1], turns[-1]["content"]


class Turn:
    def __init__(self, engine, request, emit=None):
        self.engine = engine
        self.request = request
        self.emit = emit or (lambda event: None)
        self.system, self.history, self.question = _parse(request)
        self.lang = policy.language([t["content"] for t in self.history if t["role"] == "user"] + [self.question])
        self.sources = [t["content"] for t in self.history if t["role"] == "user"] + [self.question]
        self.messages = []          # this turn's rounds, as the model reads them
        self.calls = []             # every call of the turn and how it ended
        self.guard = policy.Guard()
        self.passes = 0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self.intent = policy.NO_MATH

    def _body(self, tools, choice, note):
        body = {k: self.request[k] for k in _GENERATION if k in self.request}
        body.setdefault("chat_template_kwargs", {"enable_thinking": False})
        if body["chat_template_kwargs"].get("enable_thinking") is False:
            body.setdefault("thinking_budget_tokens", 0)
        messages = ([{"role": "system", "content": self.system}] if self.system else []) + self.history
        messages = messages + [{"role": "user", "content": self.question}] + self.messages
        if note:
            messages = messages + [{"role": "user", "content": note}]
        if body["chat_template_kwargs"].get("enable_thinking") is False:
            # Qwen3 templates that ignore enable_thinking still honour the switch in the text, as the app's chat sends it
            last = max(i for i, m in enumerate(messages) if m["role"] == "user")
            messages[last] = dict(messages[last], content=messages[last]["content"] + "\n/no_think")
        body["messages"] = messages
        if tools:
            body["tools"] = tools
            body["tool_choice"] = choice
        return body

    def _generate(self, body):
        self.passes += 1
        self.emit({"type": "pass", "n": self.passes, "state": "start"})
        reply = self.engine.complete(body)
        if not isinstance(reply, dict) or not reply.get("choices"):
            error = reply.get("error") if isinstance(reply, dict) and isinstance(reply.get("error"), dict) else {}
            # a request the engine refuses (no model in router mode, a context too small...) is the client's to fix
            if error.get("type") == "invalid_request_error" or error.get("code") == 400:
                raise Invalid(str(error.get("message") or "the engine refused the request"))
            raise RuntimeError(error.get("message") or "the engine gave no answer")
        usage = reply.get("usage") or {}
        self.usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        self.usage["completion_tokens"] += usage.get("completion_tokens", 0)
        self.emit({"type": "pass", "n": self.passes, "state": "end", "prompt_tokens": usage.get("prompt_tokens", 0),
                   "completion_tokens": usage.get("completion_tokens", 0)})
        return reply["choices"][0].get("message") or {}

    def _rounds(self):
        """The client's own limit on model passes ("tosh": {"max_rounds": n}), as the app's agent turns setting."""
        options = self.request.get("tosh")
        value = options.get("max_rounds") if isinstance(options, dict) else None
        return max(1, min(100, value)) if isinstance(value, int) and not isinstance(value, bool) else MAX_ROUNDS

    def _routed(self, body):
        """A side question to the model goes to the same model as the turn, which a router needs."""
        if self.request.get("model"):
            body["model"] = self.request["model"]
        return body

    def _intent(self):
        decided = policy.classify(self.question, [t["content"] for t in self.history if t["role"] == "user"])
        if decided:
            return decided, False
        reply = self.engine.complete(self._routed({
            "messages": [{"role": "system", "content": policy.INTENT_INSTRUCTIONS},
                         {"role": "user", "content": self.question[:4000] + " /no_think"}],
            "max_tokens": 6, "temperature": 0, "grammar": policy.INTENT_GRAMMAR, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False}}))
        try:
            word = reply["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, AttributeError):
            word = ""
        if word in policy.INTENTS:
            return word, True
        return (policy.COMPUTATIONAL if policy.evidence(self.question)["structure"] else policy.CONCEPTUAL), True

    def _source(self):
        """The user's words and what the tools returned this turn: refused and failed calls stay out."""
        context = [t["content"] for t in self.history if t["role"] == "user"]
        context += [c["result"] for c in self.calls if policy.succeeded(c)]
        return {"request": self.question, "context": "\n".join(context)[-6000:]}

    def _review(self, reply, operation):
        lines = reply.get("interpreted_input") or []
        answer = self.engine.complete(self._routed({
            "messages": [{"role": "system", "content": policy.REVIEW_INSTRUCTIONS},
                         {"role": "user", "content": f"REQUEST:\n{self.question}\n\nCALL: {operation or ''}\n"
                                                     + "\n".join(lines) + " /no_think"}],
            "max_tokens": 4, "temperature": 0, "grammar": policy.REVIEW_GRAMMAR, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False}}))
        try:
            return answer["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, AttributeError):
            return ""

    def _run_call(self, call):
        name = call.get("function", {}).get("name", "")
        raw = call.get("function", {}).get("arguments") or "{}"
        entry = {"id": call.get("id") or uuid.uuid4().hex, "name": name, "arguments": raw, "state": "failed",
                 "result": "", "reply": None}
        try:
            arguments = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except (ValueError, TypeError):
            arguments = None
        if not isinstance(arguments, dict):
            entry["result"] = json.dumps({"success": False, "error": {"code": "invalid_arguments",
                                                                       "message": "the arguments are not a JSON object"}})
        elif not policy.is_math(name):
            entry["result"] = json.dumps({"success": False, "error": {"code": "invalid_arguments",
                                                                       "message": f"no such tool: {name[:60]}"}})
        else:
            # the model's own "_" fields never reach a tool; the trusted ones are added here
            plain = {k: v for k, v in arguments.items() if not str(k).startswith("_")}
            params = dict(plain, _source=self._source(), _trust=TRUST_KEY)
            text = self.engine.call(name, params)
            reply = policy.reply_of(text)
            if policy.error_code(reply) == "needs_review" and self._review(reply, plain.get("operation")) == "consistent":
                text = self.engine.call(name, dict(params, _reviewed="consistent"))
                reply = policy.reply_of(text)
            entry.update(result=text, reply=reply, state="completed" if reply and reply.get("success") else "failed")
            if reply is not None and "timed_out" in reply and reply.get("success") is not True:
                entry["state"] = "completed"
        self.calls.append(entry)
        self.emit({"type": "tool_call", "pass": self.passes,
                   "call": dict(_describe(entry), id=entry["id"], state=entry["state"], result=entry["result"][:20000])})
        return entry

    def run(self):
        listed = self.engine.tools()
        if not listed:
            # no math tools on this engine: the model answers as it is
            message = self._generate(self._body(None, None, None))
            return message.get("content") or "", "answered"
        tools = [t["definition"] for t in listed if isinstance(t.get("definition"), dict)]
        self.intent, asked = self._intent()
        self.emit({"type": "intent", "intent": self.intent, "asked_model": asked})
        gate = policy.requires_tools(self.intent)
        required = gate
        note, finalizing, regrounded = None, False, False
        for round_number in range(self._rounds()):
            first = round_number == 0
            if finalizing:
                body = self._body(None, None, note)
            elif (first and gate) or self.guard.next == "math_only":
                body = self._body(tools + [policy.CLARIFY_TOOL], "required", note)
            else:
                standing = note or (policy.STANDING_NOTE if any(policy.is_math(c["name"]) for c in self.calls) else None)
                body = self._body(tools, "auto", standing)
            note = None
            message = self._generate(body)
            content = message.get("content") or ""
            calls = message.get("tool_calls") or []
            names = [c.get("function", {}).get("name", "") for c in calls]
            arguments = [c.get("function", {}).get("arguments", "") for c in calls]
            closing = None
            if not finalizing:
                closing = self.guard.closing(names, arguments, self.lang)
                if closing is None and first and gate and policy.CLARIFY in names:
                    closing = policy.unresolved(policy._missing(arguments[names.index(policy.CLARIFY)]), self.lang)
            if calls and closing is None and not finalizing:
                entries = [self._run_call(c) for c in calls]
                for entry in entries:
                    self.guard.record(entry["name"], entry["reply"])
                maths = [e for e in entries if policy.is_math(e["name"])]
                kept = content if all(policy.succeeded(e) for e in maths) else ""
                self.messages.append({"role": "assistant", "content": kept, "tool_calls": [
                    {"id": e["id"], "type": "function", "function": {"name": e["name"], "arguments": e["arguments"]}}
                    for e in entries]})
                self.messages += [{"role": "tool", "tool_call_id": e["id"], "content": e["result"][:20000]} for e in entries]
                if self.guard.next == "stop":
                    if policy.ledger(self.calls):
                        finalizing, note = True, policy.final_note(policy.ledger(self.calls))
                        continue
                    return policy.unresolved(lang=self.lang), "unresolved"
                continue
            kind, value = policy.step(self.sources, self.calls, content, closing, finalizing, regrounded, required, self.lang)
            if kind == "keep":
                final = closing if closing is not None else content
                if closing is not None:
                    return final, "clarification_required" if policy.CLARIFY in names else "unresolved"
                return final, "answered"
            if kind == "replace":
                return value, "validated_results_only" if policy.is_safe_answer(value) else "unresolved"
            if kind == "again":
                regrounded, note = True, value
            else:
                finalizing, note = True, value
        results = policy.ledger(self.calls)
        return policy.safe_answer(results, [c for c in self.calls if policy.is_math(c["name"])], self.lang), "turn_limit"


def _describe(call):
    reply = call.get("reply") or {}
    try:
        arguments = json.loads(call["arguments"]) if isinstance(call["arguments"], str) else call["arguments"]
    except ValueError:
        arguments = call["arguments"]
    entry = {"tool": call["name"], "arguments": arguments,
             "status": "ok" if policy.succeeded(call) else policy.error_code(reply) or "execution_error"}
    if reply.get("interpreted_input"):
        entry["interpreted_input"] = reply["interpreted_input"]
    if reply.get("result_kind"):
        entry["result_kind"] = reply["result_kind"]
    return entry


def run(arguments, emit=None, register=None):
    """The completion the engine returns to the client, or an error with its HTTP status. `emit` gets the
    progress events; `register` gets the engine handle, so that a cancellation can stop the turn."""
    request = arguments.get("request") if isinstance(arguments, dict) else None
    if not isinstance(request, dict):
        return {"status": 400, "error": {"message": "request must be a JSON object", "type": "invalid_request_error"}}
    if request.get("tools") or request.get("functions"):
        return {"status": 400, "error": {"message": "requests with tools are answered by the raw endpoint",
                                         "type": "invalid_request_error"}}
    started = time.monotonic()
    engine = Engine(str(arguments.get("base_url") or ""), arguments.get("api_key"))
    if register:
        register(engine)
    try:
        turn = Turn(engine, request, emit)
        content, outcome = turn.run()
    except Invalid as error:
        return {"status": 400, "error": {"message": str(error), "type": "invalid_request_error"}}
    except Cancelled:
        return {"status": 499, "error": {"message": "the request was cancelled", "type": "cancelled"}}
    except (RuntimeError, OSError, ValueError) as error:
        return {"status": 502, "error": {"message": str(error)[:300], "type": "engine_error"}}
    results = policy.ledger(turn.calls)
    return {
        "id": "chatcmpl-tosh-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": str(request.get("model") or "tosh-agent"),
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": dict(turn.usage, total_tokens=turn.usage["prompt_tokens"] + turn.usage["completion_tokens"]),
        "tosh": {
            "version": 1,
            "intent": turn.intent,
            "outcome": outcome,
            "calls": [_describe(c) for c in turn.calls],
            "validated_results": [{"tool": r["tool"], "operation": r["operation"],
                                   "result_kind": "exact" if r["exact"] else "approximate",
                                   "result": {k: v for k, v in (policy.reply_of(r["reply"]) or {}).items()
                                              if k not in ("success", "operation", "warnings", "interpreted_input", "result_kind")},
                                   "interpreted_input": r["input"]} for r in results],
            "passes": turn.passes,
            "seconds": round(time.monotonic() - started, 3),
        },
    }
