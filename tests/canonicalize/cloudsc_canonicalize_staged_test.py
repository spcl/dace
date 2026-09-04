# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Drive the FULL canonicalization pipeline over CloudSC, phase by phase, checking and timing each one.

The existing ``cloudsc_canonicalize_test.py`` walks the same stages but only validates structurally and
stops at ``parallelize``. This one goes the whole way and, at every phase boundary:

* **validates** the SDFG (``validate_all``-equivalent: a full ``sdfg.validate()`` after every phase, which
  is what ``validate_all`` amounts to at phase granularity);
* **numerically verifies** it against the un-canonicalized reference on identical physical inputs, at the
  IEEE tolerance the CloudSC harness already established as bit-for-bit
  (``generate_data_for_cloudsc.compare_outputs``, ``1e-15``, under ``IEEE_CPU_ARGS``);
* **times** every individual stage, so an expensive stage other than ``LoopToMap`` is visible;
* **caches** the SDFG once a phase is both valid and numerically correct, so a re-run resumes from the
  last good phase instead of repeating hours of work.

Determinism: verification runs on a **deep copy** that ``make_sequential`` rewrites to sequential
schedules -- CloudSC's OpenMP maps reorder FP reductions run-to-run, which is noise, not signal. The
copy is what gets run; the **pipeline SDFG itself is never made sequential**, because that would bake
Sequential schedules into the cached artifact and destroy the parallelism the pipeline exists to
produce. Run the process single-core to keep even the sequential build quiet::

    taskset -c 0 env OMP_NUM_THREADS=1 PYTHONPATH=/path/to/dace \\
        python tests/canonicalize/cloudsc_canonicalize_staged_test.py

This is a slow integration harness: building CloudSC (``simplify=False``) takes minutes and each verified
phase compiles the whole kernel again. It is marked ``integration`` so the unit gate does not run it --
NOT skipped: on a box with a compiler it is expected to run and pass.
"""
import argparse
import contextlib
import copy
import json
import os
import time
from typing import Dict, List, Optional, Tuple

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MPI4PY_RC_INITIALIZE', '0')
os.environ.setdefault('OMPI_MCA_pml', 'ob1')
os.environ.setdefault('OMPI_MCA_btl', 'self,vader')
os.environ.setdefault('UCX_VFS_ENABLE', 'n')

import pytest

import dace
from dace.transformation.passes.canonicalize.pipeline import _build_stages
from tests.corpus.cloudsc.generate_data_for_cloudsc import (IEEE_CPU_ARGS, build_cloudsc_sdfg, compare_outputs,
                                                            generate_cloudsc_inputs, make_sequential)

#: Machine-precision tolerance. Canonicalization is value-preserving and the IEEE build forbids
#: reassociation and FP contraction, so the reference is reproduced bit-for-bit; this is the harness's
#: own established criterion, not a tolerance invented to hide a discrepancy.
RTOL = ATOL = 1e-15


@contextlib.contextmanager
def ieee_build():
    """Compile with deterministic IEEE flags, restoring the prior setting afterwards."""
    saved = dace.Config.get('compiler', 'cpu', 'args')
    try:
        dace.Config.set('compiler', 'cpu', 'args', value=IEEE_CPU_ARGS)
        yield
    finally:
        dace.Config.set('compiler', 'cpu', 'args', value=saved)


@contextlib.contextmanager
def quiet():
    """Swallow a pass's stdout chatter.

    Several passes print one line per application (``Applied 291 TrivialTaskletElimination.``), which on
    a program the size of CloudSC buries the timing table under thousands of lines. Only stdout is
    redirected: warnings and tracebacks go to stderr and still surface.
    """
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            yield


def phase_order() -> List[str]:
    """The pipeline's phase labels, in order, de-duplicated."""
    order: List[str] = []
    for label, _ in _build_stages():
        if not order or order[-1] != label:
            if label not in order:
                order.append(label)
    return order


def build_reference() -> Tuple[dace.SDFG, Dict, Dict]:
    """The un-canonicalized CloudSC reference, its pristine inputs, and its outputs.

    :returns: ``(reference_sdfg, pristine_inputs, reference_outputs)``. Driving an SDFG mutates the
        buffers in place, so the pristine copy is taken BEFORE the reference runs; every candidate is
        later driven from that same copy.
    """
    reference = build_cloudsc_sdfg(simplify=False)
    make_sequential(reference)
    reference_outputs = generate_cloudsc_inputs(reference, seed=0)
    pristine = copy.deepcopy(reference_outputs)
    with ieee_build():
        reference(**reference_outputs)
    return reference, pristine, reference_outputs


def verify(candidate: dace.SDFG, pristine: Dict, reference_outputs: Dict) -> Tuple[bool, str]:
    """Run a COPY of ``candidate`` on the reference's inputs and compare every shared output array.

    The copy is what ``make_sequential`` mutates, so the caller's SDFG keeps its real schedules.
    """
    probe = copy.deepcopy(candidate)
    make_sequential(probe)
    candidate_outputs = copy.deepcopy(pristine)
    with ieee_build():
        probe(**candidate_outputs)
    report = compare_outputs(reference_outputs, candidate_outputs, rtol=RTOL, atol=ATOL)
    bad = {name: (abs_err, rel_err) for name, (abs_err, rel_err, ok) in report.items() if not ok}
    if bad:
        worst = sorted(bad.items(), key=lambda kv: -kv[1][1])[:4]
        return False, 'mismatched: ' + ', '.join(f'{n} (abs={a:.3e} rel={r:.3e})' for n, (a, r) in worst)
    return True, f'{len(report)} arrays bit-exact'


def run_staged(cache_dir: str,
               verify_numerics: bool = True,
               stop_after: Optional[str] = None,
               resume: bool = True) -> List[Dict]:
    """Apply the pipeline phase by phase, validating / verifying / timing / caching each.

    :param cache_dir: Directory holding ``phase-<NN>-<label>.sdfgz`` snapshots and ``timings.json``.
    :param verify_numerics: Compile+run+compare after each phase. Off = timing-only sweep.
    :param stop_after: Stop once this phase label completes.
    :param resume: Load the newest cached snapshot and skip the phases it already covers.
    :returns: One record per phase.
    """
    os.makedirs(cache_dir, exist_ok=True)
    stages = _build_stages()
    order = phase_order()

    reference = pristine = reference_outputs = None
    if verify_numerics:
        t0 = time.perf_counter()
        reference, pristine, reference_outputs = build_reference()
        print(f'[reference] built + ran in {time.perf_counter() - t0:.1f}s', flush=True)

    start_index = 0
    sdfg = None
    if resume:
        for index in range(len(order) - 1, -1, -1):
            snapshot = os.path.join(cache_dir, f'phase-{index:02d}-{order[index]}.sdfgz')
            if os.path.exists(snapshot):
                sdfg = dace.SDFG.from_file(snapshot)
                start_index = index + 1
                print(f'[resume] phase {index} ({order[index]}) from cache; starting at {start_index}', flush=True)
                break
    if sdfg is None:
        sdfg = build_cloudsc_sdfg(simplify=False)
        sdfg.validate()

    records: List[Dict] = []
    for index, label in enumerate(order):
        if index < start_index:
            continue
        stage_times: List[Tuple[str, float]] = []
        phase_start = time.perf_counter()
        error = None
        for stage_label, unit in stages:
            if stage_label != label:
                continue
            name = type(unit).__name__
            t0 = time.perf_counter()
            try:
                with quiet():
                    unit.apply_pass(sdfg, {})
            except Exception as exc:  # noqa: BLE001 - record and stop; the report IS the deliverable
                error = f'{type(exc).__name__}: {exc}'
            dt = time.perf_counter() - t0
            stage_times.append((name, dt))
            if error:
                break
        phase_time = time.perf_counter() - phase_start

        record = {
            'index': index,
            'phase': label,
            'phase_seconds': phase_time,
            'stages': [{
                'pass': n,
                'seconds': s
            } for n, s in stage_times],
            'slowest_stage': max(stage_times, key=lambda kv: kv[1])[0] if stage_times else None,
            'apply_error': error,
            'valid': None,
            'numerically_correct': None,
            'detail': None,
            'cached': False,
        }

        if error is None:
            t0 = time.perf_counter()
            try:
                sdfg.validate()
                record['valid'] = True
            except Exception as exc:  # noqa: BLE001
                record['valid'] = False
                record['detail'] = f'{type(exc).__name__}: {exc}'
            record['validate_seconds'] = time.perf_counter() - t0

        if record['valid'] and verify_numerics:
            t0 = time.perf_counter()
            try:
                ok, detail = verify(sdfg, pristine, reference_outputs)
                record['numerically_correct'] = ok
                record['detail'] = detail
            except Exception as exc:  # noqa: BLE001
                record['numerically_correct'] = False
                record['detail'] = f'{type(exc).__name__}: {exc}'
            record['verify_seconds'] = time.perf_counter() - t0

        good = record['apply_error'] is None and record['valid'] and (record['numerically_correct'] is not False)
        if good:
            snapshot = os.path.join(cache_dir, f'phase-{index:02d}-{label}.sdfgz')
            sdfg.save(snapshot, compress=True)
            record['cached'] = True

        records.append(record)
        flag = 'ok' if good else 'FAIL'
        print(f'[{index:2d}/{len(order) - 1}] {label:28s} {phase_time:8.1f}s  {flag}  {record["detail"] or ""}',
              flush=True)
        with open(os.path.join(cache_dir, 'timings.json'), 'w') as handle:
            json.dump(records, handle, indent=1)

        if not good or (stop_after and label == stop_after):
            break

    return records


def report(records: List[Dict]) -> str:
    lines = ['', f'{"phase":30s} {"total(s)":>9s} {"verify(s)":>9s}  slowest stage', '-' * 92]
    for r in records:
        lines.append(f'{r["phase"]:30s} {r["phase_seconds"]:9.2f} {r.get("verify_seconds", 0.0):9.2f}  '
                     f'{r["slowest_stage"] or ""}')
    lines += ['', 'TOP 15 STAGES BY TIME', '-' * 92]
    flat = [(s['seconds'], s['pass'], r['phase']) for r in records for s in r['stages']]
    for seconds, name, phase in sorted(flat, reverse=True)[:15]:
        lines.append(f'{seconds:9.2f}s  {name:44s} ({phase})')
    total = sum(r['phase_seconds'] for r in records)
    lines.append('')
    lines.append(f'TOTAL pipeline time: {total:.1f}s across {len(records)} phases')
    return '\n'.join(lines)


@pytest.mark.integration
def test_cloudsc_canonicalize_staged_is_valid_and_numerically_faithful(tmp_path):
    """Every canonicalization phase keeps CloudSC valid AND bit-exact against the reference.

    Marked ``integration``: it needs a working C++ compiler and takes hours. It is not skipped -- on a
    box with a toolchain it runs and must pass.
    """
    records = run_staged(str(tmp_path / 'cache'), verify_numerics=True, resume=False)
    print(report(records))

    broken = [r for r in records if r['apply_error'] or not r['valid'] or r['numerically_correct'] is False]
    assert not broken, 'phases failed: ' + '; '.join(f'{r["phase"]} ({r["apply_error"] or r["detail"]})'
                                                     for r in broken)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', default=os.path.join(os.path.dirname(__file__), '.cloudsc_canon_cache'))
    parser.add_argument('--no-verify', action='store_true', help='timing-only sweep (skip compile+run)')
    parser.add_argument('--no-resume', action='store_true', help='ignore cached snapshots')
    parser.add_argument('--stop-after', default=None, help='stop once this phase completes')
    args = parser.parse_args()

    results = run_staged(args.cache_dir,
                         verify_numerics=not args.no_verify,
                         stop_after=args.stop_after,
                         resume=not args.no_resume)
    print(report(results))
