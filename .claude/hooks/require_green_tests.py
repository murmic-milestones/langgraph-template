"""Stop hook: the AI is not done until pytest is green.

The suite is fake-LLM based (~1s, no API key, no network), which is what
makes an every-stop test gate affordable — keep it that way. Exit code 2
blocks the stop and feeds the failure back to the model; the
``stop_hook_active`` guard lets it stop on the second attempt instead of
looping forever.

Written in Python (not shell) so it behaves identically on
Windows/macOS/Linux — keep hook scripts that way.
"""

from __future__ import annotations

import json
import subprocess
import sys


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        payload = {}

    if payload.get("stop_hook_active"):
        return 0  # already blocked once — don't loop

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--tb=line"],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (subprocess.TimeoutExpired, OSError):
        return 0  # never wedge the session on hook infrastructure

    if proc.returncode != 0:
        tail = "\n".join((proc.stdout or "").strip().splitlines()[-15:])
        # ASCII only: hook output must survive Windows console encodings.
        print(
            "pytest is failing - fix the tests before finishing:\n" + tail,
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
