# ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
# Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""A stdio MCP server for the engine's tests: each tool misbehaves in one way, with no effect outside
this process, and every message it receives is appended, with its pid, to the log named by its first
argument.

    echo   answers its message, after any answer held by defer
    hang   never answers
    defer  holds its answer until the next call is answered
    gate   answers once the file at `path` exists
    slow   stops reading for `seconds`, then answers
    die    exits mid-call
    eof    closes its output and keeps running

A file at the log's path plus ".hold" holds every initialize until it is removed, as a server slow to start.
"""

import json
import os
import sys
import time

LOG = sys.argv[1]
PID = os.getpid()
TOOLS = [{"name": name, "description": name,
          "inputSchema": {"type": "object", "properties": {"message": {"type": "string"}, "path": {"type": "string"},
                                                            "seconds": {"type": "number"}}}}
         for name in ("echo", "hang", "defer", "gate", "slow", "die", "eof", "run")]


def log(**entry):
    # one write per line, so a reader never sees two lines interleaved
    fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.write(fd, (json.dumps(dict(entry, pid=PID)) + "\n").encode())
    finally:
        os.close(fd)


def send(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def answer(rid, text):
    return {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": text}]}}


log(started=True, timeout_ms=os.environ.get("TOSH_MCP_TIMEOUT_MS"))
held = []
for line in sys.stdin:
    message = json.loads(line)
    log(got=message)
    rid = message.get("id")
    if rid is None:
        continue
    method = message.get("method")
    if method == "initialize":
        while os.path.exists(LOG + ".hold"):
            time.sleep(0.01)
        send({"jsonrpc": "2.0", "id": rid, "result": {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                                                      "serverInfo": {"name": "probe", "version": "1"}}})
        continue
    if method == "ping":
        send({"jsonrpc": "2.0", "id": rid, "result": {}})
        continue
    if method == "tools/list":
        send({"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}})
        continue
    name = message["params"]["name"]
    arguments = message["params"].get("arguments") or {}
    text = arguments.get("message", "")
    if name == "run":
        # as the agent: a pass whose text is only a draft and ends in a call, then a live pass that answers
        token = (message["params"].get("_meta") or {}).get("progressToken")
        events = [{"type": "pass", "n": 1, "state": "start", "live": True}, {"type": "draft", "content": "Let me check."},
                  {"type": "retract", "n": 1}, {"type": "pass", "n": 2, "state": "start", "live": True},
                  {"type": "delta", "content": "The answer"}]
        for n, event in enumerate(events, 1):
            send({"jsonrpc": "2.0", "method": "notifications/progress",
                  "params": {"progressToken": token, "progress": n, "message": json.dumps(event)}})
        reply = {"choices": [{"index": 0, "message": {"role": "assistant", "content": "The answer is 42."},
                              "finish_reason": "stop"}]}
        send(answer(rid, json.dumps(reply)))
        continue
    if name == "hang":
        continue
    if name == "defer":
        held.append(answer(rid, "late answer to %s" % text))
        continue
    if name == "die":
        os._exit(3)
    if name == "eof":
        sys.stdout.flush()
        os.close(1)
        continue
    if name == "gate":
        end = time.monotonic() + 60
        while not os.path.exists(arguments["path"]) and time.monotonic() < end:
            time.sleep(0.01)
    if name == "slow":
        time.sleep(arguments.get("seconds", 1))
        log(woke=rid)
    for late in held:
        send(late)
    held.clear()
    send(answer(rid, text))
