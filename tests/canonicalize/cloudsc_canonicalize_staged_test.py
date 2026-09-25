# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Drive the FULL canonicalization pipeline over CloudSC, phase by phase, checking and timing each one.

The existing ``cloudsc_canonicalize_test.py`` walks the same stages but only validates structurally and
stops at ``parallelize``. This one goes the whole way and, at every phase boundary:

* **validates** the SDFG (``validate_all``-equivalent: a full ``sdfg.validate()`` after every phase, which
  is what ``validate_all`` amounts to at phase granularity);
* **numerically verifies** it against the un-canonicalized reference on identical physical inputs, run
  MULTICORE -- the way the kernel is actually executed -- at the parallel tolerance the CloudSC
  harness already established (``1e-10``, under ``IEEE_CPU_ARGS``);
* **times** every individual stage, so an expensive stage other than ``LoopToMap`` is visible;
* **caches** the SDFG once a phase is both valid and numerically correct, so a re-run resumes from the
  last good phase instead of repeating hours of work.

Why multicore: canonicalization's job is to EXPOSE parallelism, so the mistake it can make is a Map
over a loop that carries a dependence. Such a phase is bit-exact the moment the copy is rewritten to
sequential schedules, and wrong as soon as the same graph runs on more than one thread -- a
sequential check calls it correct. The parallel tolerance bounds the reassociation that running
OpenMP reductions in a different order costs, and nothing else.

A mismatch is then re-run sequentially, which names the cause instead of starting a second hunt:
bit-exact with one thread means the phase parallelized a dependence, still wrong means it changed
the values. Every run is on a **deep copy**; the **pipeline SDFG itself is never made sequential**,
because that would bake Sequential schedules into the cached artifact and destroy the parallelism
the pipeline exists to produce::

    env OMP_NUM_THREADS=4 PYTHONPATH=/path/to/dace \\
        python tests/canonicalize/cloudsc_canonicalize_staged_test.py

Pin a thread COUNT if the run should be reproducible, but never ``1``. Every claim above is about
what more than one thread does: on one thread the multicore leg and its sequential re-run are the
same run and the diagnosis collapses to "correct".

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

os.environ.setdefault('MPI4PY_RC_INITIALIZE', '0')
os.environ.setdefault('OMPI_MCA_pml', 'ob1')
os.environ.setdefault('OMPI_MCA_btl', 'self,vader')
os.environ.setdefault('UCX_VFS_ENABLE', 'n')

import pytest

import dace
from dace.transformation.passes.canonicalize.pipeline import _build_stages
from tests.corpus.cloudsc.generate_data_for_cloudsc import (IEEE_CPU_ARGS, build_cloudsc_sdfg, compare_outputs,
                                                            generate_cloudsc_inputs, make_sequential)

#: Tolerance for the MULTICORE check, taken from the ``parallel`` arm of ``cloudsc_canonicalize_test``
#: and from ``cloudsc_target_pipelines_test``. The candidate runs its Maps as OpenMP regions, so its
#: reductions and WCR accumulations fold in a different ORDER than the sequential reference; this
#: bounds reassociation and nothing else, with the IEEE build strict on both sides.
RTOL = ATOL = 1e-10

#: CloudSC's species PARAMETER constants, the same set the sibling canonicalize / target-pipeline
#: tests bake in. ``canonicalize`` specializes them BEFORE it builds its stages, so a walk that
#: leaves them symbolic walks a different graph: the species and LU loops are constant-trip only
#: once these are in, and a phase that is wrong on the constant-trip form need not misbehave on the
#: symbolic one. ``klev`` / ``klon`` / ``kidia`` / ``kfdia`` stay symbolic.
SPECIES_CONSTANTS = {'nclv': 5, 'ncldql': 1, 'ncldqi': 2, 'ncldqr': 3, 'ncldqs': 4, 'ncldqv': 5}

#: Tolerance for the sequential re-check. Same schedules as the reference and the same fold order, so
#: canonicalization -- being value-preserving -- reproduces it bit-for-bit.
SEQUENTIAL_RTOL = SEQUENTIAL_ATOL = 1e-15


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


def drive(candidate: dace.SDFG, pristine: Dict, reference_outputs: Dict, rtol: float, atol: float,
          sequential: bool) -> Dict[str, Tuple[float, float]]:
    """Run a COPY of ``candidate`` on the reference's inputs; return the arrays that disagree.

    The copy is what ``make_sequential`` may mutate, so the caller's SDFG keeps its real schedules.

    :param sequential: rewrite the copy to sequential schedules before running it.
    :returns: ``{array: (max_abs, max_rel)}`` for the arrays outside tolerance, empty when all agree.
    """
    probe = copy.deepcopy(candidate)
    if sequential:
        make_sequential(probe)
    candidate_outputs = copy.deepcopy(pristine)
    with ieee_build():
        probe(**candidate_outputs)
    report = compare_outputs(reference_outputs, candidate_outputs, rtol=rtol, atol=atol)
    return {name: (abs_err, rel_err) for name, (abs_err, rel_err, ok) in report.items() if not ok}


def verify(candidate: dace.SDFG, pristine: Dict, reference_outputs: Dict) -> Tuple[bool, str]:
    """Check the phase the way the kernel is actually RUN: Maps as OpenMP regions, multicore.

    A sequential check cannot see the failure that matters most here. Canonicalization's whole job
    is to expose parallelism, and a Map it created over a loop that carries a dependence is
    bit-exact when the copy is rewritten to sequential schedules and wrong the moment the same
    graph runs on more than one thread. Verifying sequentially declares such a phase correct.

    On a mismatch the phase is re-run SEQUENTIALLY, which separates the two causes without a second
    hunt: still wrong with one thread = the phase changed the values; correct with one thread =
    the phase parallelized something it may not.
    """
    bad = drive(candidate, pristine, reference_outputs, RTOL, ATOL, sequential=False)
    if not bad:
        return True, 'multicore: every shared array within tolerance'
    worst = sorted(bad.items(), key=lambda kv: -kv[1][1])[:4]
    detail = 'multicore mismatch: ' + ', '.join(f'{n} (abs={a:.3e} rel={r:.3e})' for n, (a, r) in worst)
    seq_bad = drive(candidate, pristine, reference_outputs, SEQUENTIAL_RTOL, SEQUENTIAL_ATOL, sequential=True)
    detail += ('; sequential is bit-exact -> the phase parallelized a dependence'
               if not seq_bad else f'; sequential also wrong ({len(seq_bad)} arrays) -> the phase changed values')
    return False, detail


def run_staged(cache_dir: str,
               verify_numerics: bool = True,
               stop_after: Optional[str] = None,
               resume: bool = True,
               specialize_constants: Optional[Dict[str, int]] = None) -> List[Dict]:
    """Apply the pipeline phase by phase, validating / verifying / timing / caching each.

    :param cache_dir: Directory holding ``phase-<NN>-<label>.sdfgz`` snapshots and ``timings.json``.
    :param verify_numerics: Compile+run+compare after each phase. Off = timing-only sweep.
    :param stop_after: Stop once this phase label completes.
    :param resume: Load the newest cached snapshot and skip the phases it already covers.
    :param specialize_constants: ``{symbol: value}`` baked into the CANDIDATE before the first
        phase, exactly where ``canonicalize`` bakes them in. The reference stays symbolic -- it is
        driven with those same values as arguments, so the two agree unless a phase is wrong.
        Walking without the map the real pipeline uses walks a DIFFERENT graph: CloudSC's species
        loops are constant-trip only once ``nclv`` and friends are baked in, so the phases see
        different loops and a divergence need not reproduce.
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
        if specialize_constants:
            # Only on a fresh build: a resumed snapshot already carries them.
            from dace.sdfg.utils import specialize_symbols
            specialize_symbols(sdfg, specialize_constants)
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
    """Every canonicalization phase keeps CloudSC valid AND numerically correct when run multicore.

    Marked ``integration``: it needs a working C++ compiler and takes hours. It is not skipped -- on a
    box with a toolchain it runs and must pass.
    """
    records = run_staged(str(tmp_path / 'cache'),
                         verify_numerics=True,
                         resume=False,
                         specialize_constants=SPECIES_CONSTANTS)
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
