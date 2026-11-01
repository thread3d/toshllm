#!/bin/zsh
# Runs the engine's own tests (patches/llama/*/tests) against a built llama-server. They need no
# model, network or credentials, only python3. Extra args go to unittest (e.g. -v, -k queued).
# In GitHub Actions each suite's counts go to the job summary.
#
#   ./scripts/test-engine.sh
#   LLAMA_SERVER_BIN_PATH=/path/to/llama-server ./scripts/test-engine.sh -v
set -u
cd "$(dirname "$0")/.."

export LLAMA_SERVER_BIN_PATH="${LLAMA_SERVER_BIN_PATH:-$PWD/vendor/llama.cpp/build-static/bin/llama-server}"
SUMMARY="${GITHUB_STEP_SUMMARY:-/dev/null}"

# whatever kept the suites from running is an error of this environment, never a pass
not_run() {
    if [ -n "${GITHUB_ACTIONS:-}" ]; then echo "::error::$1"; else echo "$1" >&2; fi
    print -r -- "**Engine tests did not run:** $1" >> "$SUMMARY"
    exit 2
}

[ -x "$LLAMA_SERVER_BIN_PATH" ] || not_run "no llama-server at $LLAMA_SERVER_BIN_PATH; build it with ./scripts/build-engines.sh"
"$LLAMA_SERVER_BIN_PATH" --version >/dev/null 2>&1 ||
    not_run "$LLAMA_SERVER_BIN_PATH does not run on this $(uname -m) computer; an x86_64 engine on Apple silicon needs Rosetta"
suites=(patches/llama/*/tests/test_*.py(N))
[ ${#suites} -gt 0 ] || not_run "no test suites under patches/llama/*/tests"

{
    print -r -- "### Engine tests"
    print -r -- ""
    print -r -- "\`llama-server\` $(lipo -archs "$LLAMA_SERVER_BIN_PATH" 2>/dev/null), sha256 $(shasum -a 256 "$LLAMA_SERVER_BIN_PATH" | cut -c1-12)"
    print -r -- ""
    print -r -- "| Suite | Ran | Passed | Failed | Skipped |"
    print -r -- "|---|---|---|---|---|"
} >> "$SUMMARY"

failed=0
for suite in $suites; do
    echo "== $suite"
    python3 -I - "$suite" "$@" <<'PY' || failed=1
import importlib.util
import os
import sys
import unittest

path = sys.argv[1]
spec = importlib.util.spec_from_file_location("engine_suite", path)
module = importlib.util.module_from_spec(spec)
# unittest finds setUpModule through sys.modules
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class Result(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.started, self.broken = set(), set()

    def startTest(self, test):
        super().startTest(test)
        self.started.add(test.id())

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.broken.add(test.id())

    def addError(self, test, err):
        super().addError(test, err)
        self.broken.add(test.id())

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            self.broken.add(test.id())

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self.broken.add(test.id())


class Runner(unittest.TextTestRunner):
    resultclass = Result


result = unittest.main(module=module, argv=[path] + sys.argv[2:], testRunner=Runner, exit=False).result
failed = len(result.broken & result.started)
skipped = len(result.skipped)
passed = result.testsRun - failed - skipped - len(result.expectedFailures)
# an error in setUpModule or setUpClass belongs to no test, and the tests after it never ran
outside = sorted(result.broken - result.started)
row = "| `%s` | %d | %d | %d | %d |" % (path, result.testsRun, passed, failed, skipped)
if outside:
    row += "\n\n**%s:** error before its tests: %s" % (path, ", ".join(outside))
if result.testsRun == 0:
    row += "\n\n**%s:** no test ran" % path
with open(os.environ.get("GITHUB_STEP_SUMMARY") or os.devnull, "a") as summary:
    summary.write(row + "\n")
print("%s: ran %d, passed %d, failed %d, skipped %d" % (path, result.testsRun, passed, failed, skipped))
sys.exit(0 if result.wasSuccessful() and result.testsRun > 0 else 1)
PY
done
exit $failed
