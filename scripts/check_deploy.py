#!/usr/bin/env python3
"""Streamlit Cloud deploy verification helper.

Diagnostic only. Compares local HEAD with origin/main, pings the demo
URL for liveness, and prints the SHA the user should look for in the
demo's sidebar footer (``Build: <sha>``). The sidebar caption is
populated by ``streamlit_app/demo_v0.py::_read_build_sha`` so the
visual check completes the verification in ~30 seconds.

This tool intentionally does **not** scrape the deployed app:
Streamlit pages are SPAs that render content via WebSocket, so plain
HTTP requests cannot read the sidebar SHA. Selenium would work but is
heavy (chromedriver dependency, 5–10 s per check). The git-side
comparison plus a one-line "expected SHA" prompt is enough to surface
the two failure modes that have actually happened (push not yet
processed by Streamlit Cloud, and Cloud's webhook missing the push so
the container stays on the previous build).

Usage::

    python scripts/check_deploy.py

Exit codes:
- ``0`` when local and origin SHAs match (push state is good).
- ``1`` when they differ (push not yet on origin/main, run ``git push``).
"""
from __future__ import annotations

import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEMO_URL = "https://csliu-toolbox.streamlit.app/"
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, cwd=_REPO_ROOT, text=True).strip()


def _local_head_sha() -> tuple[str, str]:
    full = _run(["git", "rev-parse", "HEAD"])
    subject = _run(["git", "log", "-1", "--pretty=%s", "HEAD"])
    return full, subject


def _origin_head_sha() -> str:
    raw = _run(["git", "ls-remote", "origin", "main"])
    return raw.split()[0] if raw else ""


def _demo_status() -> str:
    """One-line status from a HEAD request to the demo URL."""
    req = urllib.request.Request(DEMO_URL, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return f"HTTP {resp.status} ({resp.reason}) — alive"
    except urllib.error.HTTPError as exc:
        return f"HTTP {exc.code} ({exc.reason})"
    except urllib.error.URLError as exc:
        return f"URL error: {exc.reason}"
    except Exception as exc:  # pragma: no cover — defensive
        return f"unexpected error: {type(exc).__name__}: {exc}"


def main() -> int:
    try:
        local_full, local_subject = _local_head_sha()
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: cannot read local HEAD ({exc})", file=sys.stderr)
        return 2

    try:
        origin_full = _origin_head_sha()
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: cannot read origin/main ({exc})", file=sys.stderr)
        return 2

    local_short = local_full[:7]
    origin_short = origin_full[:7] if origin_full else "(empty)"
    in_sync = bool(origin_full) and local_full == origin_full

    print(f"Local HEAD       : {local_short}  ({local_subject})")
    print(
        f"Origin HEAD      : {origin_short}  "
        f"{'✓ in sync with local' if in_sync else '✗ DIFFERS from local'}"
    )
    print(f"Demo URL         : {DEMO_URL}")
    print(f"  status         : {_demo_status()}")
    print(f"Expected build   : {local_short}")
    print()

    if not in_sync:
        print(
            "→ Local and origin/main differ. The push is incomplete; "
            "run `git push origin main` and re-run this check."
        )
        return 1

    print(
        "→ Push state OK. Open the demo URL in a browser, scroll the "
        "sidebar to the bottom, and look at the\n"
        f"  `Build: {local_short}` caption.\n"
        "  - If the SHA matches, Streamlit Cloud has redeployed.\n"
        "  - If a different SHA is shown, Cloud is still serving the\n"
        "    previous container. Reboot via\n"
        "    https://share.streamlit.io/ → csliu-toolbox → Manage app\n"
        "    → Reboot, wait ~2 min, refresh demo, and check again.\n"
        "  - If `Build: unknown` is shown, the .git/ directory is\n"
        "    missing on the Cloud container; see README §Deployment\n"
        "    workflow for the fallback procedure."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
