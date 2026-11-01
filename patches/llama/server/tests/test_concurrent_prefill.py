# ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
# Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Concurrent requests on the patched llama-server with a real model: a prompt that arrives while another request
generates advances a share per step, a single request keeps whole batches, a new request takes the prefix it shares
with a busy slot, and a checkpoint that already holds the prefix is kept. Skipped without a model.

    TOSH_TEST_MODEL=~/models/Qwen3.5-4B-Q4_K_M.gguf TOSH_FA_AMD=1 \\
    LLAMA_SERVER_BIN_PATH=vendor/llama.cpp/build-static/bin/llama-server \\
        python3 -I patches/llama/server/tests/test_concurrent_prefill.py [-v]
"""

import http.client
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
BINARY = os.environ.get("LLAMA_SERVER_BIN_PATH") or os.path.join(ROOT, "vendor", "llama.cpp", "build-static", "bin",
                                                                  "llama-server")
MODEL = os.path.expanduser(os.environ.get("TOSH_TEST_MODEL", ""))
HOST = "127.0.0.1"
KEEP = os.environ.get("TOSH_TEST_KEEP") == "1"

WORDS = ("river delta trade route glacier printing coral railway signal volcanic island bread harbour orchard "
         "lighthouse copper weather balloon monsoon bridge salt desert telescope paper tide").split()
RULES = " ".join(f"Rule {i}: answer with the exact figure, name the layer and the field, and say when a value is missing."
                 for i in range(70))


def free_port():
    with socket.socket() as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def filler(n_words, seed):
    return " ".join(WORDS[(i * 7 + seed) % len(WORDS)] for i in range(n_words))


class Engine:
    def __init__(self, slots, cache_ram=0, env=None):
        self.dir = tempfile.mkdtemp(prefix="tosh-prefill-test-")
        self.port = free_port()
        self.slots, self.cache_ram, self.env = slots, cache_ram, env or {}

    def start(self, seconds=180):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("LLAMA_", "TOSH_PREFILL", "TOSH_PREFIX"))}
        env.update(self.env)
        self.output = open(os.path.join(self.dir, "server.log"), "wb")
        self.process = subprocess.Popen(
            [BINARY, "--host", HOST, "--port", str(self.port), "--offline", "--no-ui", "--jinja", "-m", MODEL,
             "-ngl", "99", "-fa", "auto", "-c", str(self.slots * 8192), "-np", str(self.slots),
             "--cache-ram", str(self.cache_ram), "-lv", "4"],
            stdout=self.output, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        end = time.monotonic() + seconds
        while time.monotonic() < end and self.process.poll() is None:
            try:
                if self.request("GET", "/health", timeout=2)[0] == 200:
                    return
            except OSError:
                pass
            time.sleep(0.3)
        raise RuntimeError("the engine did not start; its output is in " + self.output.name)

    def stop(self):
        self.process.terminate()
        try:
            self.process.wait(30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(10)
        self.output.close()
        if not KEEP:
            shutil.rmtree(self.dir, ignore_errors=True)

    def log(self):
        with open(self.output.name, errors="replace") as f:
            return f.read()

    def request(self, method, path, body=None, timeout=300):
        connection = http.client.HTTPConnection(HOST, self.port, timeout=timeout)
        try:
            connection.request(method, path, None if body is None else json.dumps(body),
                               {"Content-Type": "application/json"})
            response = connection.getresponse()
            return response.status, json.loads(response.read() or b"null")
        finally:
            connection.close()

    def stream(self, path, body, record):
        """Streams a request, noting when each piece of text arrives; the last chunk's timings go in the record."""
        connection = http.client.HTTPConnection(HOST, self.port, timeout=600)
        body = dict(body, stream=True)
        record.update(arrivals=[], text="", t_send=time.monotonic())
        connection.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        for raw in connection.getresponse():
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if not isinstance(chunk, dict):
                continue
            piece = chunk.get("content") or "".join((c.get("delta") or {}).get("content") or ""
                                                    for c in chunk.get("choices") or [])
            if piece:
                record["arrivals"].append(time.monotonic())
                record["text"] += piece
            if chunk.get("timings"):
                record["timings"] = chunk["timings"]
        record["t_end"] = time.monotonic()
        connection.close()


def background(function, *args):
    thread = threading.Thread(target=function, args=args, daemon=True)
    thread.start()
    return thread


def wait_tokens(record, n, seconds=120):
    end = time.monotonic() + seconds
    while len(record.get("arrivals", [])) < n and time.monotonic() < end:
        time.sleep(0.02)
    return len(record.get("arrivals", [])) >= n


@unittest.skipUnless(MODEL and os.path.exists(MODEL), "set TOSH_TEST_MODEL to a GGUF model")
class ConcurrentPrefillTest(unittest.TestCase):
    def engine(self, slots, cache_ram=0, env=None):
        engine = Engine(slots, cache_ram, env)
        engine.start()
        self.addCleanup(engine.stop)
        return engine

    def generation_gap(self, env):
        """The longest wait between two tokens of a request that generates while another one's long prompt fills,
        against how long that prompt took."""
        engine = self.engine(2, env=env)
        gen, long_prompt = {}, {}
        thread = background(engine.stream, "/completion", {"prompt": "Write a long story about a lighthouse keeper.",
                                                           "n_predict": 700, "ignore_eos": True, "temperature": 0,
                                                           "cache_prompt": False}, gen)
        self.assertTrue(wait_tokens(gen, 20))
        engine.stream("/completion", {"prompt": filler(3600, 1), "n_predict": 4, "temperature": 0,
                                      "cache_prompt": False}, long_prompt)
        thread.join()
        start, first = long_prompt["t_send"], long_prompt["arrivals"][0]
        during = [t for t in gen["arrivals"] if start <= t <= first]
        self.assertGreater(len(during), 2, "the first request finished before the prompt filled")
        gap = max(b - a for a, b in zip(during, during[1:]))
        return gap, long_prompt["timings"]["prompt_ms"] / 1000, long_prompt["timings"]["prompt_n"]

    def test_generation_keeps_going_while_a_prompt_fills(self):
        gap, prompt_s, n = self.generation_gap({})
        self.assertGreater(n, 2048)
        self.assertLess(gap, prompt_s / 3, f"a {n}-token prompt of {prompt_s:.2f} s held the other request {gap:.2f} s")

    def test_switch_restores_whole_batches(self):
        gap, prompt_s, n = self.generation_gap({"TOSH_PREFILL_SHARE": "0"})
        self.assertGreater(gap, prompt_s / 3, f"a {n}-token prompt of {prompt_s:.2f} s held the other request only {gap:.2f} s")

    def test_one_request_keeps_whole_batches(self):
        engine = self.engine(4)
        record = {}
        engine.stream("/completion", {"prompt": filler(3600, 2), "n_predict": 4, "temperature": 0,
                                      "cache_prompt": False}, record)
        task = re.findall(r"task (\d+) \| new prompt", engine.log())[-1]
        batches = len(re.findall(rf"task {task} \| cached n_tokens = \d+, memory_seq_rm", engine.log()))
        self.assertGreater(record["timings"]["prompt_n"], 2048)
        self.assertLessEqual(batches, 5, f"one request on four slots took {batches} batches")

    def chat(self, engine, question, n, record):
        engine.stream("/v1/chat/completions", {"messages": [{"role": "system", "content": RULES},
                                                            {"role": "user", "content": question}],
                                               "max_tokens": n, "temperature": 0,
                                               "chat_template_kwargs": {"enable_thinking": False}}, record)

    def busy_prefix(self, env):
        engine = self.engine(2, cache_ram=1024, env=env)
        first, second = {}, {}
        thread = background(self.chat, engine, "Write a long story about the depots of the north.", 500, first)
        self.assertTrue(wait_tokens(first, 10))
        self.chat(engine, "How many depots are in the south?", 8, second)
        busy = len(first["arrivals"]) < 500 and first.get("t_end") is None
        thread.join()
        self.assertTrue(busy, "the first request had finished")
        self.assertTrue(second["text"].strip())
        return second["timings"]["cache_n"], first["timings"]["prompt_n"]

    def test_a_new_request_takes_the_prefix_of_a_busy_slot(self):
        cached, shared = self.busy_prefix({})
        self.assertGreater(cached, shared * 0.8, f"reused {cached} of a {shared}-token prompt")

    def test_prefix_copy_switch(self):
        cached, shared = self.busy_prefix({"TOSH_PREFIX_COPY": "0"})
        self.assertLess(cached, shared * 0.2)

    def test_a_checkpoint_holding_the_prefix_is_kept(self):
        engine = self.engine(1, cache_ram=1024)
        messages = [{"role": "system", "content": RULES}]
        for turn in range(3):
            messages.append({"role": "user", "content": f"Turn {turn}: how many depots are in province {turn}?"})
            status, body = engine.request("POST", "/v1/chat/completions", {
                "messages": messages, "max_tokens": 16, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False}})
            self.assertEqual(status, 200)
            messages.append({"role": "assistant", "content": body["choices"][0]["message"]["content"]})
        log = engine.log()
        if "created context checkpoint" not in log:
            self.skipTest("the model needs no checkpoints")
        self.assertTrue("kept context checkpoint" in log, "a checkpoint holding the prefix was created again")


if __name__ == "__main__":
    unittest.main()
