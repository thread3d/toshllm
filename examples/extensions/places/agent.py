# ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
# Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The agent of the places example: answers questions about places with its tools, streams its passes, hands
other turns back to the chat and offers a map. Contract in helpers/tosh-sympy/README.md ("Another agent")."""

import http.client
import json
import re
import socket
import sys
import threading
import time
import urllib.parse
import uuid

MAX_ROUNDS = 6
PREFIX = "places_"
INSTRUCTION = {
    "en": "Coordinates, distances and elevations of places come only from the places tools; never state one "
          "they did not return. If a place is unknown to them, say so.",
    "es": "Las coordenadas, distancias y altitudes de los lugares salen solo de las herramientas de lugares; no "
          "des ninguna que no hayan devuelto. Si no conocen un lugar, dilo.",
}
PLACE_WORDS = re.compile(r"\b(where|distance|how far|near(est|by)?|map|capital|mountain|landmark|coordinates|"
                         r"d[oó]nde|distancia|qu[eé] tan lejos|cerca|mapa|monta[ñn]a|monumento|coordenadas)\b", re.I)
MAP_WORDS = re.compile(r"\b(map|show me where|where (is|are)|mapa|mu[eé]strame d[oó]nde|d[oó]nde (est[aá]n?|queda))\b",
                       re.I)
# accents alone say nothing: place names carry them in any language
SPANISH = re.compile(r"[¿¡]|\b(el|los|las|qu[eé]|cu[aá]l|cu[aá]nto|d[oó]nde|entre|desde|est[aá]n?|lejos|cerca)\b",
                     re.I)
# request fields that belong to this agent or to the client, never to a model pass
NOT_FOR_PASSES = {"messages", "stream", "stream_options", "tosh", "tools", "tool_choice", "functions", "function_call"}


class Cancelled(Exception):
    pass


class Engine:
    """The engine over HTTP. X-Tosh-Agent: off keeps every pass away from this agent; the agent key lets /tools
    list and run the servers only the agent may use."""

    def __init__(self, base_url, api_key, agent_key):
        parsed = urllib.parse.urlparse(base_url)
        self.host, self.port = parsed.hostname or "127.0.0.1", parsed.port or 80
        self.headers = {"Content-Type": "application/json", "X-Tosh-Agent": "off"}
        if api_key:
            self.headers["Authorization"] = "Bearer " + api_key
        if agent_key:
            self.headers["X-Tosh-Agent-Key"] = agent_key
        self.cancelled = threading.Event()
        self.lock = threading.Lock()
        self.connection = None

    def cancel(self):
        with self.lock:
            self.cancelled.set()
            connection = self.connection
        if connection is not None and connection.sock is not None:
            try:
                connection.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def send(self, method, path, body=None, on_line=None, timeout=3600):
        if self.cancelled.is_set():
            raise Cancelled()
        connection = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        with self.lock:
            self.connection = connection
        try:
            connection.request(method, path, None if body is None else json.dumps(body), self.headers)
            response = connection.getresponse()
            if on_line is None or response.status != 200:
                data = response.read()
                return json.loads(data or b"null")
            for raw in response:
                on_line(raw.decode("utf-8", "replace").strip())
            return None
        except (OSError, http.client.HTTPException, ValueError):
            if self.cancelled.is_set():
                raise Cancelled()
            raise
        finally:
            with self.lock:
                self.connection = None
            connection.close()

    def tools(self):
        listed = self.send("GET", "/tools", timeout=60)
        return [t for t in listed or [] if isinstance(t, dict) and str(t.get("tool", "")).startswith(PREFIX)]

    def call(self, name, params):
        """A tool's reply as (ok, text)."""
        reply = self.send("POST", "/tools", {"tool": name, "params": params}, timeout=120)
        if isinstance(reply, dict) and isinstance(reply.get("plain_text_response"), str):
            return True, reply["plain_text_response"]
        error = reply.get("error") if isinstance(reply, dict) else None
        return False, error if isinstance(error, str) else json.dumps(reply)

    def stream(self, body, on_delta):
        """One model pass as a stream; on_delta gets ("content" or "reasoning_content", text) as it arrives."""
        message, calls, usage, failure = {"role": "assistant", "content": ""}, {}, {}, []

        def on_line(line):
            if not line.startswith("data: ") or line == "data: [DONE]":
                return
            try:
                chunk = json.loads(line[6:])
            except ValueError:
                return
            if "error" in chunk:
                failure.append(chunk)
            usage.update(chunk.get("usage") or {})
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                for field in ("content", "reasoning_content"):
                    if isinstance(delta.get(field), str) and delta[field]:
                        if field == "content":
                            message["content"] += delta[field]
                        on_delta(field, delta[field])
                # tool calls arrive in pieces, keyed by their index
                for piece in delta.get("tool_calls") or []:
                    call = calls.setdefault(piece.get("index", 0),
                                            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    call["id"] = piece.get("id") or call["id"]
                    function = piece.get("function") or {}
                    call["function"]["name"] += function.get("name") or ""
                    call["function"]["arguments"] += function.get("arguments") or ""

        reply = self.send("POST", "/v1/chat/completions",
                          dict(body, stream=True, stream_options={"include_usage": True}), on_line=on_line)
        if reply is not None or failure:
            error = (reply or failure[0]).get("error") if isinstance(reply or failure[0], dict) else None
            raise RuntimeError((error or {}).get("message") if isinstance(error, dict) else "the engine gave no answer")
        if calls:
            message["tool_calls"] = [calls[i] for i in sorted(calls)]
        return message, usage


def language(text):
    return "es" if SPANISH.search(text or "") else "en"


def text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def completion(request, content, tosh, usage=None):
    usage = usage or {"prompt_tokens": 0, "completion_tokens": 0}
    return {
        "id": "chatcmpl-places-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": str(request.get("model") or "places-agent"),
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": dict(usage, total_tokens=usage["prompt_tokens"] + usage["completion_tokens"]),
        "tosh": dict({"version": 1}, **tosh),
    }


def summary(name, ok, text, lang):
    """One readable line about a call's result, shown on its card in the app."""
    if not ok:
        return ("Falló: " if lang == "es" else "Failed: ") + text[:120]
    try:
        value = json.loads(text)
    except ValueError:
        return None
    if "distance" in value and "to" in value:
        return f"{value['from']['name']} → {value['to']['name']}: {value['distance']} {value['unit']}"
    found = value.get("places")
    if isinstance(found, list):
        names = ", ".join(p.get("name", "") for p in found[:4]) + ("…" if len(found) > 4 else "")
        return (f"{len(found)} lugares: " if lang == "es" else f"{len(found)} places: ") + names
    return None


def places_in(text):
    """Names of the places a tool result mentions, for the map."""
    try:
        value = json.loads(text)
    except ValueError:
        return []
    found = [value.get("from"), value.get("to")] + list(value.get("places") or [])
    return [p["name"] for p in found if isinstance(p, dict) and "latitude" in p]


class Turn:
    def __init__(self, engine, request, emit):
        self.engine, self.request, self.emit = engine, request, emit
        self.started = time.monotonic()
        self.messages = [m for m in request.get("messages") or [] if isinstance(m, dict)]
        users = [m for m in self.messages if m.get("role") == "user"]
        self.question = text_of(users[-1].get("content")) if users else ""
        self.lang = language(self.question)
        self.passes, self.calls, self.mapped = 0, [], []
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        options = request.get("tosh") if isinstance(request.get("tosh"), dict) else {}
        self.options = options
        rounds = options.get("max_rounds")
        self.rounds = max(1, min(100, rounds)) if isinstance(rounds, int) and not isinstance(rounds, bool) else MAX_ROUNDS

    def about_places(self, tools):
        if not tools:
            return False
        if PLACE_WORDS.search(self.question):
            return True
        ok, text = self.engine.call(PREFIX + "find", {"text": self.question})
        return ok and bool(places_in(text))

    def body(self, tools):
        body = {k: v for k, v in self.request.items() if k not in NOT_FOR_PASSES}
        messages = list(self.messages)
        if tools:
            # one system message, first: chat templates expect it there
            note = INSTRUCTION[self.lang]
            if messages and messages[0].get("role") == "system":
                messages[0] = dict(messages[0], content=text_of(messages[0].get("content")) + "\n\n" + note)
            else:
                messages.insert(0, {"role": "system", "content": note})
            body["tools"] = tools
            body["tool_choice"] = "auto"
        body["messages"] = messages
        return body

    def generate(self, body):
        self.passes += 1
        self.emit({"type": "pass", "n": self.passes, "state": "start", "live": True})
        shown = False

        def on_delta(field, text):
            nonlocal shown
            shown = shown or field == "content"
            self.emit({"type": "delta", field: text})

        message, usage = self.engine.stream(body, on_delta)
        for key in self.usage:
            self.usage[key] += usage.get(key, 0)
        self.emit({"type": "pass", "n": self.passes, "state": "end", "prompt_tokens": usage.get("prompt_tokens", 0),
                   "completion_tokens": usage.get("completion_tokens", 0)})
        # the text of a pass that ends in tool calls was not the answer: the client drops it
        if shown and message.get("tool_calls"):
            self.emit({"type": "retract", "n": self.passes})
        return message

    def run_call(self, call):
        name = call["function"]["name"]
        try:
            arguments = json.loads(call["function"]["arguments"] or "{}")
        except ValueError:
            arguments = None
        if not name.startswith(PREFIX) or not isinstance(arguments, dict):
            ok, text = False, "not a places tool call with JSON arguments"
        else:
            ok, text = self.engine.call(name, arguments)
        if ok:
            self.mapped += [n for n in places_in(text) if n not in self.mapped]
        record = {"id": call["id"] or "call-" + uuid.uuid4().hex[:8], "tool": name, "arguments": arguments or {},
                  "status": "ok" if ok else "execution_error", "state": "completed" if ok else "failed",
                  "result": text[:20000], "summary": summary(name, ok, text, self.lang)}
        self.calls.append(record)
        self.emit({"type": "tool_call", "pass": self.passes, "call": record})
        return {"role": "tool", "tool_call_id": record["id"], "content": text}

    def run(self):
        tools = [t["definition"] for t in self.engine.tools() if isinstance(t.get("definition"), dict)]
        places = self.about_places(tools)
        if not places and self.options.get("client_tools") is True:
            # nothing ran: the app answers this turn with its own tools, as it would without this extension
            return completion(self.request, "", {"outcome": "declined", "passes": 0, "seconds": self.seconds()})
        self.emit({"type": "intent", "intent": "places" if places else "no_places"})
        body = self.body(tools if places else [])
        content, outcome = "", "turn_limit"
        for _ in range(self.rounds):
            message = self.generate(body)
            if not message.get("tool_calls"):
                content, outcome = message.get("content") or "", "answered"
                break
            body["messages"].append({"role": "assistant", "content": message.get("content") or "",
                                     "tool_calls": message["tool_calls"]})
            body["messages"] += [self.run_call(call) for call in message["tool_calls"]]
        tosh = {"intent": "places" if places else "no_places", "outcome": outcome, "passes": self.passes,
                "seconds": self.seconds(),
                "calls": [{k: c[k] for k in ("tool", "arguments", "status")} for c in self.calls]}
        if self.mapped:
            # no geometry here: the client sends this back to get the map, so the answer is not delayed by it
            tosh["map_request"] = {"show": bool(MAP_WORDS.search(self.question)), "names": self.mapped[:200],
                                   "lang": self.lang}
        return completion(self.request, content, tosh, self.usage)

    def seconds(self):
        return round(time.monotonic() - self.started, 3)


def map_answer(engine, request, asked):
    """The layers of a map_request, with no model pass."""
    names = [n for n in asked.get("names") or [] if isinstance(n, str)][:200]
    lang = "es" if asked.get("lang") == "es" else "en"
    ok, text = engine.call(PREFIX + "find", {"names": names, "limit": 100}) if names else (False, "")
    found = json.loads(text).get("places", []) if ok else []
    features = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [p["longitude"], p["latitude"]]},
                 "properties": {k: v for k, v in p.items() if k not in ("latitude", "longitude")}} for p in found]
    layer = {"type": "FeatureCollection", "features": features,
             "title": "Lugares" if lang == "es" else "Places",
             "subtitle": "Los de la respuesta" if lang == "es" else "The ones in the answer",
             "fields": [{"name": n, "alias": a[lang == "es"]} for n, a in (
                 ("name", ("Name", "Nombre")), ("category", ("Category", "Categoría")),
                 ("country", ("Country", "País")), ("elevation_m", ("Elevation (m)", "Altitud (m)")))],
             "complete": len(found) == len(names)}
    if found:
        latitude = sum(p["latitude"] for p in found) / len(found)
        longitude = sum(p["longitude"] for p in found) / len(found)
        layer["link"] = {"title": "OpenStreetMap",
                         "url": f"https://www.openstreetmap.org/#map=4/{latitude:.4f}/{longitude:.4f}"}
    return completion(request, "", {"outcome": "answered", "passes": 0, "map": {"layers": [layer] if found else []}})


def run(arguments, emit, register):
    request = arguments.get("request") if isinstance(arguments.get("request"), dict) else {}
    engine = Engine(str(arguments.get("base_url") or ""), arguments.get("api_key"), arguments.get("agent_key"))
    register(engine)
    try:
        asked = (request.get("tosh") or {}).get("map_request") if isinstance(request.get("tosh"), dict) else None
        if isinstance(asked, dict):
            return map_answer(engine, request, asked)
        return Turn(engine, request, emit).run()
    except Cancelled:
        return {"status": 499, "error": {"message": "the request was cancelled", "type": "cancelled"}}


# MCP over stdio

DEFINITION = {
    "name": "run",
    "description": "Answers an OpenAI chat completions request with the places tools. Called by the engine.",
    "inputSchema": {"type": "object", "properties": {"request": {"type": "object"}, "base_url": {"type": "string"},
                                                    "api_key": {"type": "string"}, "agent_key": {"type": "string"}},
                    "required": ["request", "base_url"]},
}
OUT = threading.Lock()
TURNS = {}


def write(message):
    with OUT:
        sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def handle(message):
    method, message_id = message.get("method"), message.get("id")
    params = message.get("params") if isinstance(message.get("params"), dict) else {}
    if method == "notifications/cancelled":
        engine = TURNS.get(params.get("requestId"))
        if engine is not None:
            engine.cancel()
        return
    if message_id is None:
        return
    if method == "initialize":
        write({"jsonrpc": "2.0", "id": message_id, "result": {
            "protocolVersion": params.get("protocolVersion", "2025-06-18"), "capabilities": {"tools": {}},
            "serverInfo": {"name": "places-agent", "version": "1.0.0"}}})
    elif method == "ping":
        write({"jsonrpc": "2.0", "id": message_id, "result": {}})
    elif method == "tools/list":
        write({"jsonrpc": "2.0", "id": message_id, "result": {"tools": [DEFINITION]}})
    elif method == "tools/call":
        token = (params.get("_meta") or {}).get("progressToken")
        count = iter(range(1, 1 << 30))

        def emit(event):
            if token is not None:
                write({"jsonrpc": "2.0", "method": "notifications/progress",
                       "params": {"progressToken": token, "progress": next(count), "message": json.dumps(event)}})

        # its own thread, so that a cancellation sent meanwhile is read
        def work():
            try:
                reply = run(params.get("arguments") or {}, emit, lambda engine: TURNS.__setitem__(message_id, engine))
            except Exception as error:  # the engine always gets an answer
                reply = {"status": 502, "error": {"message": f"{type(error).__name__}: {error}", "type": "agent_error"}}
            TURNS.pop(message_id, None)
            write({"jsonrpc": "2.0", "id": message_id,
                   "result": {"content": [{"type": "text", "text": json.dumps(reply, ensure_ascii=False)}]}})

        threading.Thread(target=work, daemon=True).start()
    else:
        write({"jsonrpc": "2.0", "id": message_id, "error": {"code": -32601, "message": "unknown method"}})


def main():
    for line in sys.stdin:
        try:
            message = json.loads(line)
        except ValueError:
            continue
        if isinstance(message, dict):
            handle(message)


if __name__ == "__main__":
    main()
