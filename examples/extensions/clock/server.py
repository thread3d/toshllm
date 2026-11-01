# ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
# Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""A minimal MCP server over stdio, standard library only: one tool that tells the time in a zone.
It is the smallest complete extension, meant to be copied."""

import datetime
import json
import os
import sys
import zoneinfo

TOOLS = [{
    "name": "now",
    "description": "Current date and time in an IANA time zone, such as Europe/Madrid or America/Caracas.",
    "inputSchema": {
        "type": "object",
        "properties": {"zone": {"type": "string", "description": "IANA time zone name"}},
    },
}]


def now(arguments):
    zone = arguments.get("zone") or os.environ.get("TOSH_CLOCK_DEFAULT_ZONE") or "UTC"
    try:
        moment = datetime.datetime.now(zoneinfo.ZoneInfo(zone))
    except (zoneinfo.ZoneInfoNotFoundError, ValueError):
        return {"content": [{"type": "text", "text": "unknown time zone: " + zone}], "isError": True}
    return {"content": [{"type": "text", "text": moment.isoformat(timespec="seconds") + " " + zone}]}


def answer(message):
    method = message.get("method")
    if method == "initialize":
        version = (message.get("params") or {}).get("protocolVersion", "2025-06-18")
        return {"protocolVersion": version, "capabilities": {"tools": {}},
                "serverInfo": {"name": "clock", "version": "1.0.0"}}
    if method == "tools/list":
        return {"tools": TOOLS}
    if method == "tools/call":
        params = message.get("params") or {}
        if params.get("name") == "now":
            return now(params.get("arguments") or {})
    if method == "ping":
        return {}
    raise LookupError(method)


def main():
    for line in sys.stdin:
        try:
            message = json.loads(line)
        except ValueError:
            continue
        if "id" not in message:
            continue
        try:
            reply = {"jsonrpc": "2.0", "id": message["id"], "result": answer(message)}
        except LookupError:
            reply = {"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32601, "message": "unknown method"}}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
