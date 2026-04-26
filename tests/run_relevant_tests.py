#!/usr/bin/env python3
"""Run every test relevant to the HLFIR frontend + supporting library /
transformation work in this branch, and print a single summary at the end.

Each test group is invoked as its own ``pytest`` subprocess so a crash in
one group doesn't poison the others, and so we can attribute pass / fail
/ xfail counts back to a meaningful bucket.

Usage:
    python tests/run_relevant_tests.py           # full run, all groups
    python tests/run_relevant_tests.py -k merge  # forward extra args to pytest
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Each entry: (label, [paths/args]).  Ignores cover collection-time errors
# we know are unrelated (mpi4py / scipy.sparse missing on this machine).
GROUPS: list[tuple[str, list[str]]] = [
    ("hlfir.frontend", ["tests/hlfir/"]),
    ("library.standard", ["tests/library/count_node_test.py", "tests/library/merge_node_test.py"]),
    ("transformations.bufred", ["tests/transformations/buffered_reduce_to_inplace_test.py"]),
    ("transformations.fusion", [
        "tests/transformations/", "--ignore=tests/transformations/gpu_grid_stride_tiling_test.py",
        "--ignore=tests/transformations/otf_map_fusion_test.py"
    ]),
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


def run_group(label: str, args: list[str], extra: list[str]) -> tuple[dict[str, int], float, int]:
    cmd = [sys.executable, "-m", "pytest", "--tb=line", "-p", "no:cacheprovider", *args, *extra]
    print(f"\n=== {label}: {' '.join(args)} ===", flush=True)
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    elapsed = time.monotonic() - t0
    sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    counts = parse_summary(proc.stdout)
    return counts, elapsed, proc.returncode


def main(extra: list[str]) -> int:
    totals = dict(passed=0, failed=0, errors=0, skipped=0, xfailed=0, xpassed=0)
    rows: list[tuple[str, dict[str, int], float, int]] = []
    for label, args in GROUPS:
        counts, elapsed, rc = run_group(label, args, extra)
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
