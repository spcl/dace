#!/usr/bin/env python3
"""
Complete pip -> uv migration for spcl/dace CI and Release scripts.

Run from the repo root on a CLEAN checkout of PR #2323.

CI Patches:
- Converts pip install / pip uninstall / python -m pip -> uv pip ...
- Removes `pip install --upgrade pip` lines.
- Converts `python -m venv` -> `uv venv`.
- Inserts astral-sh/setup-uv@v4 right after actions/checkout@v7.
- For jobs that do NOT create a virtualenv, injects a job-level
  `env: UV_SYSTEM_PYTHON: '1'` so uv installs into the setup-python
  interpreter. Jobs that create+activate a venv are left alone.

Release Patches (.github/workflows/release.sh):
- Removes `pip install --upgrade twine build` (uv has these built-in).
- Replaces `python -m build --sdist` with `uv build --sdist --no-sources`.
- Replaces `twine upload dist/*` with `uv publish dist/*`.

Does NOT generate or require uv.lock (uv pip install ignores it).
Idempotent: safe to run multiple times.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKFLOWS = sorted(ROOT.glob(".github/workflows/*.yml")) + sorted(ROOT.glob(".github/workflows/*.yaml"))
RELEASE_SH = ROOT / ".github/workflows/release.sh"

CHECKOUT_RE = re.compile(r"^(?P<indent>\s*)- uses: actions/checkout@v\d+\s*$", re.MULTILINE)
RUNS_ON_RE = re.compile(r"^(?P<indent>\s+)runs-on:.*$", re.MULTILINE)


def has_venv(text: str) -> bool:
    return bool(re.search(r"(python\s+-m\s+venv|uv\s+venv)", text))


def convert_commands(text: str) -> str:
    # 1. Drop `... pip install --upgrade pip` lines entirely (any prefix).
    text = re.sub(r"^[ \t]*.*\bpip install --upgrade pip[ \t]*\n", "", text, flags=re.MULTILINE)

    # 2. python -m pip install/uninstall -> uv pip install/uninstall
    text = re.sub(r"python\s+-m\s+pip\s+install\s+", "uv pip install ", text)
    text = re.sub(r"python\s+-m\s+pip\s+uninstall\s+", "uv pip uninstall ", text)

    # 3. pip uninstall -> uv pip uninstall (skip already-prefixed)
    text = re.sub(r"(?<!uv )(?<!-)(?<!uv\t)pip uninstall\s+", "uv pip uninstall ", text)

    # 4. pip install -> uv pip install (skip already-prefixed)
    text = re.sub(r"(?<!uv )(?<!-)(?<!uv\t)pip install\s+", "uv pip install ", text)

    # 5. python -m venv -> uv venv
    text = re.sub(r"python\s+-m\s+venv\s+", "uv venv ", text)

    # 6. Comment touch-up
    text = text.replace("so we can use pip", "so we can use uv")
    return text


def inject_setup_uv(text: str) -> str:
    if "astral-sh/setup-uv" in text:
        return text
    m = CHECKOUT_RE.search(text)
    if not m:
        return text
    indent = m.group("indent")
    start = m.end()

    # Walk past the checkout block (its `with:` and children are more indented)
    for ln in text[start:].splitlines(keepends=True):
        if ln.startswith(indent + " ") or ln.startswith(indent + "\t") or ln.strip() == "":
            start += len(ln)
        else:
            break

    block = (f"{indent}- name: Install uv\n"
             f"{indent}  uses: astral-sh/setup-uv@v4\n"
             f"{indent}  with:\n"
             f"{indent}    enable-cache: true\n"
             f"{indent}    pyproject-file: pyproject.toml\n"
             f"{indent}    cache-dependency-glob: pyproject.toml\n")
    return text[:start] + block + text[start:]


def inject_uv_system_python(text: str) -> str:
    if "UV_SYSTEM_PYTHON" in text:
        return text
    if has_venv(text):
        # A venv is created and activated -> uv uses $VIRTUAL_ENV; do NOT force --system.
        return text
    m = RUNS_ON_RE.search(text)
    if not m:
        return text
    indent = m.group("indent")
    insert = f"\n{indent}env:\n{indent}  UV_SYSTEM_PYTHON: '1'"
    return text[:m.end()] + insert + text[m.end():]


def patch_workflow(path: Path) -> bool:
    orig = path.read_text(encoding="utf-8")
    text = convert_commands(orig)
    text = inject_setup_uv(text)
    text = inject_uv_system_python(text)
    if text != orig:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def patch_release_script(path: Path) -> bool:
    if not path.exists():
        return False

    text = path.read_text(encoding="utf-8")
    original = text

    # 1. Remove "pip install --upgrade twine build" and its comment
    text = re.sub(r"^# Install dependencies\s*\npip install --upgrade twine build\s*\n", "", text, flags=re.MULTILINE)

    # 2. Replace "python -m build --sdist" with "uv build --sdist --no-sources"
    # --no-sources ensures it builds exactly as it would for PyPI users
    text = text.replace("python -m build --sdist", "uv build --sdist --no-sources")

    # 3. Replace "twine upload dist/*" with "uv publish dist/*"
    text = text.replace("twine upload dist/*", "uv publish dist/*")

    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> int:
    print("Patching CI workflows...")
    changed_wf = [p for p in WORKFLOWS if patch_workflow(p)]
    for p in changed_wf:
        print(f"  - {p.relative_to(ROOT)}")

    print("\nPatching release script...")
    if patch_release_script(RELEASE_SH):
        print(f"  - {RELEASE_SH.relative_to(ROOT)}")
        changed_wf.append(RELEASE_SH)

    if not changed_wf:
        print("  (nothing changed)")

    print("\nNotes:")
    print("- uv.lock is NOT generated; uv pip install ignores it.")
    print("- venv jobs (heterogeneous-ci, gpu-ci) use the activated venv automatically.")
    print("- non-venv jobs (general-ci, ml-ci, linting) got UV_SYSTEM_PYTHON='1'.")
    print("- release.sh now uses `uv build` and `uv publish` (no pip install twine/build needed).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
