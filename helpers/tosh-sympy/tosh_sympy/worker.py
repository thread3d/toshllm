# ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
# Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The process that holds the math libraries of one tool set. One JSON request per line on
stdin, one reply per line on stdout.

The supervisor in server.py owns its lifetime: it kills this process on a timeout or when it
grows past the memory limit.
"""

import ctypes
import importlib
import json
import os
import signal
import sys

HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_BLOCKED_EVENTS = (
    "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork",
    "os.forkpty", "pty.spawn", "socket.", "ctypes.", "urllib.", "http.", "ftplib.", "smtplib.",
    "webbrowser.", "shutil.", "os.remove", "os.rename", "os.mkdir", "os.rmdir", "os.chmod",
    "os.chown", "os.link", "os.symlink", "os.truncate", "os.kill", "os.putenv", "os.unsetenv",
    "tempfile.",
)


def _audit(event, args):
    if event.startswith(_BLOCKED_EVENTS):
        raise PermissionError(f"blocked: {event}")
    if event == "open" and len(args) > 1 and args[1] is not None and set(str(args[1])) & set("wax+"):
        raise PermissionError("blocked: writing files")


def _sandbox():
    """Denies the network, file writes, new processes and reads of user data. macOS only."""
    if sys.platform != "darwin" or '"' in HOME or "\\" in HOME:
        return False
    profile = (
        "(version 1)(allow default)"
        "(deny network*)(deny file-write*)(deny process-exec*)(deny process-fork)"
        '(deny file-read* (subpath "/Users") (subpath "/Volumes"))'
        f'(allow file-read* (subpath "{HOME}"))'
    )
    try:
        library = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        error = ctypes.c_char_p()
        return library.sandbox_init(profile.encode(), 0, ctypes.byref(error)) == 0
    except (OSError, AttributeError):
        return False


def _checked(ops, toolset, operation, plain, source, reviewed):
    """Runs the call unless it does not say what the request said. The reply always carries what was read."""
    from tosh_sympy import anchor
    interpreted = anchor.interpret(operation, plain)
    if isinstance(source, dict) and isinstance(source.get("request"), str):
        status, reasons = anchor.check(source["request"], str(source.get("context") or ""), operation, plain)
        if status == anchor.INCONSISTENT or (status == anchor.UNCERTAIN and reviewed != anchor.CONSISTENT):
            mismatch = status == anchor.INCONSISTENT
            reply = ops.failure(
                operation, "transcription_mismatch" if mismatch else "needs_review",
                ("Not computed: the call does not say what the request says. " if mismatch else
                 "Not computed: the call could not be matched to the request. ")
                + "; ".join(reasons) + ". Call again with only what the request states, written as it states it, "
                "or ask the user. Do not give a result that was not computed.")
            reply["interpreted_input"] = interpreted
            reply["reasons"] = reasons
            return reply
    remarks = []
    if isinstance(source, dict) and isinstance(source.get("request"), str) and status == anchor.CONSISTENT:
        remarks = reasons
    reply = ops.run(operation, plain)
    if isinstance(reply, dict):
        reply["interpreted_input"] = interpreted
        if remarks:
            reply["warnings"] = list(reply.get("warnings") or []) + remarks
        if reply.get("success"):
            approximate = toolset != "sympy" or operation == "nsolve" or ("exact" in reply and reply["exact"] is None)
            reply["result_kind"] = "approximate" if approximate else "exact"
    return reply


def main():
    sys.path.insert(0, HOME)
    toolset = sys.argv[1] if len(sys.argv) > 1 else "sympy"
    ops = importlib.import_module(f"tosh_{toolset}.ops")
    schema = importlib.import_module(f"tosh_{toolset}.schema")

    sandboxed = _sandbox()
    sys.addaudithook(_audit)

    # the kernel ends this process if a call outlives its supervisor
    backstop = int(float(os.environ.get("TOSH_SYMPY_BACKSTOP_SECONDS", "60")))

    reply_stream = sys.stdout
    print(json.dumps({"ready": True, "sandboxed": sandboxed}), file=reply_stream, flush=True)
    for line in sys.stdin:
        signal.alarm(backstop)
        try:
            request = json.loads(line)
            name, arguments = request.get("tool"), request.get("arguments")
            source = reviewed = None
            if isinstance(arguments, dict):
                # set by the app, never by the model: the user's own words and whether a review passed
                source, reviewed = arguments.get("_source"), arguments.get("_reviewed")
                arguments = {k: v for k, v in arguments.items() if not str(k).startswith("_")}
            try:
                operation = schema.check(name, arguments)
            except ValueError as error:
                operation = arguments.get("operation") if isinstance(arguments, dict) else None
                reply = ops.failure(operation if isinstance(operation, str) else None,
                                    "invalid_arguments", str(error))
            else:
                plain = {k: v for k, v in arguments.items() if k != "operation"}
                if "after_timeout" in request:
                    reply = ops.after_timeout(
                        operation, plain, request["after_timeout"],
                        lambda partial: print(json.dumps({"progress": partial}), file=reply_stream, flush=True))
                else:
                    reply = _checked(ops, toolset, operation, plain, source, reviewed)
        except MemoryError:
            os._exit(86)
        except Exception as error:
            reply = ops.failure(None, "internal_error", f"{type(error).__name__}: {str(error)[:300]}")
        signal.alarm(0)
        print(json.dumps(reply), file=reply_stream, flush=True)


if __name__ == "__main__":
    main()
