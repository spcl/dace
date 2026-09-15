# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Run the TSVC canonicalize->vectorize corpus and report the delta AGAINST A COMMITTED BASELINE.

A raw count is not a result. The corpus has pre-existing failures, so "302 passed" and
"20 failed / 282 passed" are both consistent with a healthy tree and neither answers the only
question worth asking -- did MY change break a test that used to pass. Reporting a count as if it
were a clean run is what cost an hour of false-alarm re-baselining, so this tool never prints one:
it prints ``fixed`` / ``newly broken`` / ``unchanged`` BY TEST ID and exits non-zero only for
``newly broken``.

Usage::

    python tests/corpus_baseline.py              # compare against tests/corpus_baseline.json
    python tests/corpus_baseline.py --regenerate # re-measure and rewrite the baseline

REGENERATING: the baseline is only as trustworthy as the run that produced it, so re-measure with
ONE agent on an otherwise idle machine -- a concurrent build or test sweep changes pass/fail through
resource exhaustion, not just timings. Clean the dace caches first, keep ``-n`` at the baselined
value (a different fan-out is a different measurement), and leave ``--maxfail`` high enough that the
run reaches the end: a truncated failure list silently becomes a baseline asserting the untested
tail is green. This script refuses to write or compare a truncated run for exactly that reason.
Re-fetch immediately before measuring and record the SHA actually tested -- the corpus result moves
whenever a canonicalize or vectorize fix lands, so a baseline attributed to the wrong commit is
worse than none.
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

#: Repository root, so the tool works from any working directory.
ROOT = pathlib.Path(__file__).resolve().parent.parent

#: The committed baseline artifact.
BASELINE_PATH = ROOT / 'tests' / 'corpus_baseline.json'

#: The corpus under baseline: every TSVC kernel, canonicalized then vectorized, checked against
#: its numpy reference.
TARGET = 'tests/passes/vectorization/tsvc_canonicalize_vectorize_corpus_test.py'

#: Fan-out, plugin and seed the baseline is measured at. A delta run MUST reuse these -- pass/fail
#: is fan-out sensitive under memory pressure, and test order reaches codegen through the build
#: folder, so ``-p no:randomly`` and a pinned hash seed are part of the measurement, not decoration.
CONFIG = {
    'n': '4',
    'p': 'no:randomly',
    'maxfail': '30',
    'PYTHONHASHSEED': '0',
}

#: Env every run pins, so an inherited shell cannot change the verdict. MPI is steered off UCX
#: because dace's frontend calls ``MPI_Init`` lazily and a stalled bring-up hangs collection.
PINNED_ENV = {
    'PYTHONHASHSEED': CONFIG['PYTHONHASHSEED'],
    'OMPI_MCA_pml': 'ob1',
    'OMPI_MCA_btl': 'self,vader',
    'UCX_VFS_ENABLE': 'n',
    'MPI4PY_RC_INITIALIZE': '0',
}


def run_corpus(report_path: str) -> int:
    """Run the corpus at :data:`CONFIG` writing a JUnit report; return pytest's exit code."""
    command = [
        sys.executable, '-m', 'pytest', TARGET, '-q', '-p', CONFIG['p'], '-n', CONFIG['n'],
        f"--maxfail={CONFIG['maxfail']}", f'--junitxml={report_path}'
    ]
    print(f"$ {' '.join(command)}", flush=True)
    return subprocess.run(command, cwd=ROOT, env=dict(os.environ, **PINNED_ENV)).returncode


def read_report(report_path: str) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    """``(collected ids, {failing id: signature})`` from a JUnit report.

    The node id is rebuilt as ``file::name`` so it is the same string pytest takes back on the
    command line. The signature is the exception type plus the first line of its message -- enough
    to notice that a test still fails but for a NEW reason, which a bare id would hide.
    """
    collected: List[str] = []
    failing: Dict[str, Dict[str, str]] = {}
    for case in ET.parse(report_path).getroot().iter('testcase'):
        path, name = case.get('file'), case.get('name')
        if not path or not name:
            continue
        node = f'{path}::{name}'
        collected.append(node)
        for outcome in case:
            if outcome.tag not in ('failure', 'error'):
                continue
            message = (outcome.get('message') or '').strip()
            failing[node] = {
                'kind': outcome.tag,
                'type': outcome.get('type') or '',
                'first_line': message.splitlines()[0] if message else '',
            }
    return collected, failing


def assert_not_truncated(collected: List[str], failing: Dict[str, Dict[str, str]]) -> None:
    """Refuse to use a run that ``--maxfail`` cut short -- its untested tail is not evidence."""
    if len(failing) >= int(CONFIG['maxfail']):
        sys.exit(f"REFUSING: {len(failing)} failures reached --maxfail={CONFIG['maxfail']}, so the run stopped "
                 'early and the rest of the corpus never ran. Raise --maxfail and re-measure; a '
                 'truncated list would record the untested tail as green.')
    if not collected:
        sys.exit('REFUSING: the run collected no tests at all.')


def head_sha() -> str:
    """Full SHA of HEAD, the commit a regenerated baseline is attributed to."""
    return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def regenerate() -> int:
    """Measure the corpus and rewrite :data:`BASELINE_PATH`."""
    with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as handle:
        report = handle.name
    try:
        run_corpus(report)
        collected, failing = read_report(report)
    finally:
        os.unlink(report)
    assert_not_truncated(collected, failing)

    baseline = {
        'target': TARGET,
        'commit': head_sha(),
        'config': CONFIG,
        'collected': len(collected),
        'failing': dict(sorted(failing.items())),
    }
    BASELINE_PATH.write_text(json.dumps(baseline, indent=2) + '\n')
    print(f'\nwrote {BASELINE_PATH.relative_to(ROOT)}: {len(failing)} failing of {len(collected)} collected '
          f"at {baseline['commit'][:9]}")
    return 0


def report_delta() -> int:
    """Compare a fresh run to the baseline; non-zero ONLY when something newly broke."""
    if not BASELINE_PATH.exists():
        sys.exit(f'no baseline at {BASELINE_PATH}; run with --regenerate')
    baseline = json.loads(BASELINE_PATH.read_text())
    if baseline['config'] != CONFIG:
        sys.exit(f"REFUSING: baseline was measured at {baseline['config']}, this tool runs {CONFIG}. "
                 'A different fan-out or seed is a different measurement -- regenerate instead.')

    with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as handle:
        report = handle.name
    try:
        run_corpus(report)
        collected, failing = read_report(report)
    finally:
        os.unlink(report)
    assert_not_truncated(collected, failing)

    was_failing, live = baseline['failing'], set(collected)
    # A baselined failure that is no longer COLLECTED was not fixed -- it was renamed or removed.
    fixed = sorted(i for i in was_failing if i in live and i not in failing)
    disappeared = sorted(i for i in was_failing if i not in live)
    newly_broken = sorted(i for i in failing if i not in was_failing)
    unchanged = sorted(i for i in failing if i in was_failing)

    print(f"\ncorpus delta vs baseline @ {baseline['commit'][:9]} ({baseline['target']})")
    print(f'  unchanged failures: {len(unchanged)}')
    for label, ids in (('fixed', fixed), ('newly broken', newly_broken), ('no longer collected', disappeared)):
        print(f'  {label}: {len(ids)}')
        for node in ids:
            detail = failing[node]['first_line'] if node in failing else ''
            print(f'    {node}{f"  -- {detail}" if detail else ""}')
    if len(collected) != baseline['collected']:
        print(f"  NOTE: corpus size moved {baseline['collected']} -> {len(collected)}")

    if newly_broken:
        print(f'\nFAIL: {len(newly_broken)} test(s) newly broken against the baseline')
        return 1
    print('\nOK: no test newly broken against the baseline')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--regenerate', action='store_true', help='re-measure and rewrite the committed baseline')
    return regenerate() if parser.parse_args().regenerate else report_delta()


if __name__ == '__main__':
    sys.exit(main())
