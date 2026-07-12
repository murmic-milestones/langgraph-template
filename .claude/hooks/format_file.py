"""PostToolUse hook: keep AI-edited Python files ruff-clean.

Runs ruff (fix + format) on the file just written/edited, so generated
code always matches the project style — no drift, no review nits.
Never blocks (always exits 0): style is corrected, not policed here;
`ruff check` in CI/the Stop hook still catches unfixable issues.

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
        return 0

    file_path = (payload.get("tool_input") or {}).get("file_path", "")
    if not file_path.endswith(".py"):
        return 0

    for args in (("check", "--fix", "-q"), ("format", "-q")):
        subprocess.run(
            [sys.executable, "-m", "ruff", *args, file_path],
            capture_output=True,
            timeout=60,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
