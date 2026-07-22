#!/usr/bin/env python3
"""Run the focused regression sweep for this branch, and print a single
summary at the end.

Scope is intentionally narrow — only the surfaces this branch touched:

  * ``tests/hlfir/`` — the HLFIR frontend, including the verbatim
    ``ported/`` subdirectory ported from f2dace.
  * ``tests/library/count_node_test.py`` and
    ``tests/library/merge_node_test.py`` — the two standard library
    nodes this branch introduced.

Anything else (transformations, GPU, codegen, …) is out of scope here
because this branch doesn't change it; run those with their own
``pytest`` invocations if needed.

By default each pytest subprocess is parallelised with pytest-xdist
(``-n 4``).  Override with ``--workers N`` — ``--workers 1`` forces a
serial run.  The HLFIR group is forced serial regardless because
many HLFIR tests reuse the SDFG name ``main`` and would race on the
shared ``.dacecache/main/build`` directory under xdist.

Usage:
    python tests/run_relevant_tests.py                       # default 4 workers
    python tests/run_relevant_tests.py --workers 1           # serial
    python tests/run_relevant_tests.py -- -k merge           # forward args to pytest
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Each entry: (label, [paths/args]).  Cache-dir races between parallel
# workers are handled by ``tests/hlfir/conftest.py``, which gives each
# pytest-xdist worker its own ``.dacecache_gw<N>`` directory.
GROUPS: list[tuple[str, list[str]]] = [
    ("hlfir.frontend", ["tests/hlfir/"]),
    ("library.standard", ["tests/library/count_node_test.py", "tests/library/merge_node_test.py"]),
]

# pytest's short-summary tail line: "= 220 passed, 2 skipped, 137 xfailed in 250.81s ="
SUMMARY_RE = re.compile(r"(?P<passed>\d+)\s+passed"
                        r"|(?P<failed>\d+)\s+failed"
                        r"|(?P<errors>\d+)\s+error"
                        r"|(?P<skipped>\d+)\s+skipped"
                        r"|(?P<xfailed>\d+)\s+xfailed"
                        r"|(?P<xpassed>\d+)\s+xpassed")


def parse_summary(stdout: str) -> dict[str, int]:
    """Extract pytest's summary counts.  Walks the last 30 lines (where
    pytest prints its tail) so we don't mis-pick from per-test progress
    lines or warning headers."""
    counts = dict(passed=0, failed=0, errors=0, skipped=0, xfailed=0, xpassed=0)
    # Strip ANSI escapes so the regex doesn't choke on colour codes.
    plain = re.sub(r"\x1b\[[0-9;]*m", "", stdout)
    for line in plain.splitlines()[-30:]:
        # The summary line ends with " in <duration>s" — restrict to it.
        if " in " not in line or "s " not in line:
            continue
        if "passed" not in line and "failed" not in line and "error" not in line:
            continue
        for m in SUMMARY_RE.finditer(line):
            for k, v in m.groupdict().items():
                if v is not None:
                    counts[k] = int(v)
        break
    return counts


def run_group(label: str, args: list[str], extra: list[str], workers: int) -> tuple[dict[str, int], float, int]:
    """Run one pytest group; returns (summary counts, elapsed seconds, exit code)."""
    parallel: list[str] = []
    if workers > 1:
        parallel = ["-n", str(workers)]
    cmd = [sys.executable, "-m", "pytest", "--tb=line", "-p", "no:cacheprovider", *parallel, *args, *extra]
    print(f"\n=== {label}: {' '.join(args)} (workers={workers}) ===", flush=True)
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    elapsed = time.monotonic() - t0
    sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    counts = parse_summary(proc.stdout)
    return counts, elapsed, proc.returncode


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4, help="pytest-xdist worker count (default: 4). 1 = serial.")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER, help="Forwarded to pytest after a literal '--'.")
    ns = parser.parse_args(argv)
    extra = [a for a in ns.pytest_args if a != "--"]

    totals = dict(passed=0, failed=0, errors=0, skipped=0, xfailed=0, xpassed=0)
    rows: list[tuple[str, dict[str, int], float, int]] = []
    for label, args in GROUPS:
        counts, elapsed, rc = run_group(label, args, extra, ns.workers)
        rows.append((label, counts, elapsed, rc))
        for k in totals:
            totals[k] += counts[k]

    # Pretty-print a single summary table.
    width = max(len(r[0]) for r in rows) + 2
    print("\n" + "=" * 78)
    print("Summary (per group):")
    print(f"{'group'.ljust(width)}  pass  fail  err  skip  xfail  xpass   time(s)  exit")
    print("-" * 78)
    for label, c, t, rc in rows:
        print(f"{label.ljust(width)}  {c['passed']:4d}  {c['failed']:4d}  "
              f"{c['errors']:3d}  {c['skipped']:4d}  {c['xfailed']:5d}  "
              f"{c['xpassed']:5d}   {t:7.1f}  {rc:4d}")
    print("-" * 78)
    print(f"{'TOTAL'.ljust(width)}  {totals['passed']:4d}  {totals['failed']:4d}  "
          f"{totals['errors']:3d}  {totals['skipped']:4d}  {totals['xfailed']:5d}  "
          f"{totals['xpassed']:5d}")
    print("=" * 78)

    # Exit non-zero if any group failed or had unexpected XPASS.
    bad = totals['failed'] + totals['errors'] + totals['xpassed']
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
