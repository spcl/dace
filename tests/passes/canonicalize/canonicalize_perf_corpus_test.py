# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end PERFORMANCE / speedup harness over the combined polybench + npbench
corpus: time ``auto_optimize`` (baseline) and ``canonicalize`` on the same machine
and report the canon-vs-auto-opt speedup.

Design (resumable, one result file per kernel)
----------------------------------------------
Each kernel writes a single human-readable JSON file under a results directory
(``CANON_PERF_DIR``, default ``perf_results/`` next to this test). The file holds,
for both dataset presets (``S`` small, ``paper`` larger), the dataset shape, the
per-pipeline correctness flag, ALL timed repetitions (ms) plus min/median/mean/std,
and the speedup of each pipeline vs the baseline.

The pytest harness measures ONE kernel per test (``test_speedup[suite-kernel]``):

* If the kernel's result file already exists it is **skipped** -- so an interrupted
  sweep resumes cheaply and only missing kernels are (re)measured.
* To re-run a kernel, delete its file (or set ``CANON_PERF_FORCE=1`` to remeasure
  all). To re-run a subset, delete just those files.
* After measuring, it asserts the candidate pipelines are not grossly slower than
  the baseline (the ratio is the only assertion; absolute times are never asserted).

CSV export (option)
-------------------
A flat per-(suite, kernel, preset, pipeline) summary CSV can be produced from the
result files with ``--csv PATH`` (script) or ``CANON_PERF_CSV=PATH`` (env, exported
at the end of a sweep). The JSON files remain the precise source (all repetitions).

Usage::

    # measure (writes/refreshes perf_results/<suite>_<kernel>.json), skips existing:
    python -m tests.passes.canonicalize.canonicalize_perf_corpus_test
    # re-measure everything and also export a CSV summary:
    python -m tests.passes.canonicalize.canonicalize_perf_corpus_test --force --csv perf.csv
    # only kernels matching a substring:
    python -m tests.passes.canonicalize.canonicalize_perf_corpus_test --only gemm
    # as a CI gate (per-kernel speedup test, ``perf`` marker):
    pytest -m perf tests/passes/canonicalize/canonicalize_perf_corpus_test.py
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import csv
import datetime
import json
import math
import signal
import socket

import numpy as np
import pytest

import dace
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from tests.corpus import corpus_suite as CS

_CPU = dict(target='cpu',
            peel_limit=4,
            break_anti_dependence=True,
            interchange_carry_with_map=True,
            scatter_to_guarded_maps=True)


def _canon(s):
    return finalize_for_target(canonicalize(s, validate=True, **_CPU), 'cpu')


def _autoopt(s):
    return auto_optimize(s, dace.DeviceType.CPU)


#: Pipelines to time. The first is the BASELINE that speedups are reported against.
_BASELINE = 'auto-opt'
_PIPELINES = {'auto-opt': _autoopt, 'canon': _canon}

#: A candidate pipeline is a "regression" only past this multiple of the baseline
#: (best-of-N). Generous so timing noise never flakes CI; catches gross regressions.
REGRESSION_FACTOR = 6.0
#: Below this baseline runtime (ms) the kernel is too fast to time reliably; gate it
#: very leniently rather than flake on per-call overhead.
_MIN_TIMEABLE_MS = 0.5
_REPS = int(os.environ.get('CANON_PERF_REPS', '7'))
_WARMUP = int(os.environ.get('CANON_PERF_WARMUP', '2'))
_PER_KERNEL_TIMEOUT = int(os.environ.get('CANON_PERF_TIMEOUT', '600'))

#: Where per-kernel result files live (one ``<suite>_<kernel>.json`` each).
_RESULTS_DIR = os.environ.get('CANON_PERF_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                             'perf_results'))
#: Re-measure even when a result file already exists.
_FORCE = os.environ.get('CANON_PERF_FORCE', '') not in ('', '0', 'false', 'False')
#: Also save each pipeline's SDFG BEFORE library-node expansion to ``<dir>/sdfg/``
#: (opt-in; ``--save-sdfg`` from the script sets this too).
_SAVE_SDFG = os.environ.get('CANON_PERF_SAVE_SDFG', '') not in ('', '0', 'false', 'False')

#: Structural-only builders -- the SDFG each pipeline produces BEFORE library-node
#: expansion (Reduce/MatMul/Einsum nodes still present). canon stops at the
#: ``canonicalize`` output (``finalize_for_target`` would expand it); auto-opt uses
#: ``auto_optimize(..., expand=False)``.
_PRE_EXPANSION = {
    'auto-opt': lambda s: auto_optimize(s, dace.DeviceType.CPU, expand=False),
    'canon': lambda s: canonicalize(s, validate=True, **_CPU),
}

_HOST = socket.gethostname()
_CSV_FIELDS = [
    'suite', 'kernel', 'preset', 'pipeline', 'correct', 'min_ms', 'median_ms', 'mean_ms', 'std_ms',
    'speedup_vs_baseline', 'reps', 'omp', 'host', 'timestamp', 'error'
]


class _Timeout(Exception):
    pass


def _now():
    return datetime.datetime.now().isoformat(timespec='seconds')


def _result_path(suite, name):
    return os.path.join(_RESULTS_DIR, f'{suite}_{name}.json')


def _shape_of(ctx):
    """The dataset shape (symbol -> value) for the kernel/preset, for the record."""
    if ctx['suite'] == 'poly':
        return {str(k): int(v) for k, v in ctx['psize'].items()}
    return {k: (v if isinstance(v, float) else int(v)) for k, v in ctx['params'].items()}


def _aggregate(times_ms):
    a = np.asarray(times_ms, dtype=float)
    return dict(
        min_ms=round(float(a.min()), 6),
        median_ms=round(float(np.median(a)), 6),
        mean_ms=round(float(a.mean()), 6),
        std_ms=round(float(a.std(ddof=1)) if a.size > 1 else 0.0, 6),
    )


def _geomean(xs):
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float('nan')


def _save_pre_expansion(ctx, suite, name, preset):
    """Save each pipeline's pre-expansion SDFG (opt-in) to ``<results>/sdfg/`` for
    inspection. Best-effort: a pipeline that fails to build is reported by the
    timing path, so its snapshot is simply skipped here."""
    out_dir = os.path.join(_RESULTS_DIR, 'sdfg')
    os.makedirs(out_dir, exist_ok=True)
    for label, fn in _PRE_EXPANSION.items():
        try:
            s = CS.build(ctx, fn, f'{label.replace("-", "")}_pre')
            s.save(os.path.join(out_dir, f'{suite}_{name}_{preset}_{label.replace("-", "_")}.sdfg'))
        except Exception:
            continue


def _time_all_reps(ctx, sdfg):
    """All timed repetitions for one compiled SDFG, in milliseconds."""
    cs, kw = CS.compiled_call(ctx, sdfg)
    with dace.profile(repetitions=_REPS, warmup=_WARMUP, print_results=False) as prof:
        cs(**kw)
    _report, times = prof.times[-1]
    return np.asarray(times, dtype=float)  # dace.profile reports per-call wall time in ms


def _measure_kernel(suite, name):
    """Measure every preset x pipeline for one kernel; return the result record.

    Each pipeline is correctness-gated against the suite reference; a pipeline that
    errors or miscompiles is recorded (``correct=false`` / ``error``) and excluded
    from speedups instead of failing the whole kernel. Raises ``pytest.skip`` if
    nothing at all was measurable.
    """
    result = dict(suite=suite,
                  kernel=name,
                  host=_HOST,
                  timestamp=_now(),
                  omp_num_threads=os.environ.get('OMP_NUM_THREADS', ''),
                  reps=_REPS,
                  warmup=_WARMUP,
                  baseline=_BASELINE,
                  presets={})
    measured_any = False
    for preset in CS.PRESETS:
        pres = dict(shape=None, pipelines={}, speedup_vs_baseline={})
        try:
            ctx = CS.make(suite, name, preset)
        except Exception as e:  # input/reference build failure -> record and move on
            pres['error'] = f'{type(e).__name__}: {str(e)[:120]}'
            result['presets'][preset] = pres
            continue
        pres['shape'] = _shape_of(ctx)
        if _SAVE_SDFG:
            _save_pre_expansion(ctx, suite, name, preset)
        for label, fn in _PIPELINES.items():
            entry = {}
            try:
                s = CS.build(ctx, fn, label.replace('-', ''))
                entry['correct'] = bool(CS.run_matches(ctx, s))
                if entry['correct']:
                    times = _time_all_reps(ctx, s)
                    entry.update(_aggregate(times))
                    entry['times_ms'] = [round(float(x), 6) for x in times]
                    measured_any = True
            except Exception as e:
                entry['error'] = f'{type(e).__name__}: {str(e)[:120]}'
            pres['pipelines'][label] = entry
        # Speedups vs baseline (best-of-N min): >1 means the candidate is faster.
        base = pres['pipelines'].get(_BASELINE, {})
        b_min = base.get('min_ms') if base.get('correct') else None
        for label, entry in pres['pipelines'].items():
            if label == _BASELINE:
                continue
            if b_min and entry.get('correct') and entry.get('min_ms'):
                pres['speedup_vs_baseline'][label] = round(b_min / entry['min_ms'], 4)
        result['presets'][preset] = pres
    if not measured_any:
        pytest.skip(f"{suite}:{name} not measurable (every pipeline errored or miscompiled)")
    return result


def _write_result(path, result):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    os.replace(tmp, path)  # atomic: never leave a half-written file for skip-existing


def _regressions(result):
    """List of human-readable regression strings (candidate grossly slower than baseline)."""
    out = []
    for preset, pres in result['presets'].items():
        base = pres['pipelines'].get(result['baseline'], {})
        if not (base.get('correct') and base.get('min_ms')):
            continue
        b = base['min_ms']
        factor = REGRESSION_FACTOR if b >= _MIN_TIMEABLE_MS else 50.0
        for label, entry in pres['pipelines'].items():
            if label == result['baseline'] or not entry.get('correct') or not entry.get('min_ms'):
                continue
            if entry['min_ms'] > factor * b:
                out.append(f"{label}[{preset}] {entry['min_ms']:.3f}ms > {factor:g}x "
                           f"{result['baseline']} {b:.3f}ms")
    return out


# ---------------------------------------------------------------------------
# pytest harness: one (resumable) speedup test per kernel.
# ---------------------------------------------------------------------------
@pytest.mark.perf
@pytest.mark.parametrize("suite,name", CS.kernels())
def test_speedup(suite, name):
    path = _result_path(suite, name)
    if os.path.exists(path) and not _FORCE:
        pytest.skip(f"already measured: {os.path.relpath(path)} "
                    f"(delete it or set CANON_PERF_FORCE=1 to re-run)")

    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(_Timeout()))
    signal.alarm(_PER_KERNEL_TIMEOUT)
    try:
        result = _measure_kernel(suite, name)  # may pytest.skip (build error / not measurable)
    except _Timeout:
        pytest.skip(f"{suite}:{name} exceeded {_PER_KERNEL_TIMEOUT}s (not measured; will retry)")
    finally:
        signal.alarm(0)

    _write_result(path, result)  # written only on success -> skip-existing is meaningful
    problems = _regressions(result)
    assert not problems, f"perf regression on {suite}:{name}: " + "; ".join(problems)


# ---------------------------------------------------------------------------
# CSV export: flat per-(suite, kernel, preset, pipeline) summary from result files.
# ---------------------------------------------------------------------------
def export_csv(csv_path, results_dir=None):
    """Aggregate every result JSON in ``results_dir`` into one summary CSV. Returns row count."""
    results_dir = results_dir or _RESULTS_DIR
    rows = []
    for fn in sorted(os.listdir(results_dir)) if os.path.isdir(results_dir) else []:
        if not fn.endswith('.json'):
            continue
        with open(os.path.join(results_dir, fn)) as f:
            r = json.load(f)
        baseline = r.get('baseline', _BASELINE)
        for preset, pres in r.get('presets', {}).items():
            for label, entry in pres.get('pipelines', {}).items():
                speedup = 1.0 if label == baseline else pres.get('speedup_vs_baseline', {}).get(label)
                rows.append(
                    dict(suite=r.get('suite'),
                         kernel=r.get('kernel'),
                         preset=preset,
                         pipeline=label,
                         correct=entry.get('correct'),
                         min_ms=entry.get('min_ms'),
                         median_ms=entry.get('median_ms'),
                         mean_ms=entry.get('mean_ms'),
                         std_ms=entry.get('std_ms'),
                         speedup_vs_baseline=speedup,
                         reps=r.get('reps'),
                         omp=r.get('omp_num_threads'),
                         host=r.get('host'),
                         timestamp=r.get('timestamp'),
                         error=entry.get('error', '')))
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or '.', exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


# ---------------------------------------------------------------------------
# Markdown table: the headline "speedup numbers in a table" deliverable.
# ---------------------------------------------------------------------------
def _md_correct(entry):
    """Correctness glyph for a pipeline entry: numerically verified / wrong / not run."""
    if entry.get('correct') is True:
        return '✓'
    if entry.get('correct') is False:
        return '✗'
    return '·'


def _md_time(entry):
    if entry.get('min_ms') is not None:
        return f"{entry['min_ms']:.3f}"
    if entry.get('correct') is False:
        return 'WRONG'
    if entry.get('error'):
        return 'ERR'
    return '—'


def _md_speed(speedups, label):
    v = speedups.get(label)
    return f"{v:.2f}×" if v else '—'


def export_markdown(md_path, results_dir=None):
    """Render every result JSON in ``results_dir`` into one GitHub-flavored Markdown
    file: per preset a kernel x pipeline table (best-of-N min ms, correctness glyph,
    speedup vs baseline) plus a geomean + correctness summary line. Returns the path."""
    results_dir = results_dir or _RESULTS_DIR
    records = {}
    for fn in sorted(os.listdir(results_dir)) if os.path.isdir(results_dir) else []:
        if not fn.endswith('.json'):
            continue
        with open(os.path.join(results_dir, fn)) as f:
            r = json.load(f)
        records[(r.get('suite'), r.get('kernel'))] = r

    out = [
        f"# Canonicalize speedup vs `{_BASELINE}`",
        "",
        f"host `{_HOST}` · OMP `{os.environ.get('OMP_NUM_THREADS', '')}` · "
        f"best-of `{_REPS}` (warmup {_WARMUP}) · kernels measured `{len(records)}` · generated `{_now()}`",
        "",
        "Speedup = `auto-opt_min / pipeline_min` (best-of-N); **>1× means faster than auto-opt**. "
        "Only numerically-correct pipelines are timed — ✓ verified, ✗ miscompile, · not measured.",
    ]
    for preset in CS.PRESETS:
        out += [
            "",
            f"## preset `{preset}`",
            "",
            "| suite | kernel | shape | auto-opt ms | canon ms | ✓ | canon× |",
            "|:--|:--|:--|--:|--:|:-:|--:|",
        ]
        cspeed = []
        cstat = dict(ok=0, wrong=0, na=0)
        for suite, name in CS.kernels():
            r = records.get((suite, name))
            pres = r.get('presets', {}).get(preset) if r else None
            if not pres:
                continue
            pp = pres.get('pipelines', {})
            sp = pres.get('speedup_vs_baseline', {})
            shape = pres.get('shape') or {}
            shape_s = ' '.join(f"{k}={v}" for k, v in shape.items()) if isinstance(shape, dict) else str(shape)
            ao, cn = pp.get('auto-opt', {}), pp.get('canon', {})
            out.append(f"| {suite} | {name} | {shape_s} | {_md_time(ao)} | {_md_time(cn)} | "
                       f"{_md_correct(cn)} | {_md_speed(sp, 'canon')} |")
            cstat['ok' if cn.get('correct') is True else 'wrong' if cn.get('correct') is False else 'na'] += 1
            if sp.get('canon'):
                cspeed.append(sp['canon'])
        out += [
            "",
            f"**geomean speedup** — canon `{_geomean(cspeed):.3f}×` (n={len(cspeed)}) · "
            f"**correctness** — canon {cstat['ok']}✓ {cstat['wrong']}✗ {cstat['na']}·",
        ]
    out.append("")
    os.makedirs(os.path.dirname(os.path.abspath(md_path)) or '.', exist_ok=True)
    with open(md_path, 'w') as f:
        f.write('\n'.join(out))
    return md_path


# ---------------------------------------------------------------------------
# Script entry point: run the (resumable) sweep, print a table, optional CSV.
# ---------------------------------------------------------------------------
def _print_tables():
    for preset in CS.PRESETS:
        print(
            f"\n### preset={preset}  (best-of-{_REPS}, warmup={_WARMUP}, "
            f"OMP={os.environ.get('OMP_NUM_THREADS')})",
            flush=True)
        print(f"{'suite':5} {'kernel':18} {'auto-opt':>11} | {'canon':>11} {'speedup':>8}", flush=True)
        cspeed = []
        for suite, name in CS.kernels():
            path = _result_path(suite, name)
            if not os.path.exists(path):
                continue
            with open(path) as f:
                r = json.load(f)
            pres = r.get('presets', {}).get(preset, {})
            pp = pres.get('pipelines', {})
            sp = pres.get('speedup_vs_baseline', {})

            def _t(label):
                e = pp.get(label, {})
                return f"{e['min_ms']:.3f}ms" if e.get('min_ms') else ('WRONG' if e.get('correct') is False else 'ERR')

            def _s(label):
                v = sp.get(label)
                return f"{v:.2f}x" if v else '  -  '

            print(f"{suite:5} {name:18} {_t('auto-opt'):>11} | {_t('canon'):>11} {_s('canon'):>8}", flush=True)
            if sp.get('canon'):
                cspeed.append(sp['canon'])

        print(f"# geomean speedup vs {_BASELINE}: canon={_geomean(cspeed):.3f}x (n={len(cspeed)})", flush=True)


def _run_sweep(only=None, force=False):
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    kernels = [(s, n) for s, n in CS.kernels() if not only or only in n]
    for i, (suite, name) in enumerate(kernels, 1):
        path = _result_path(suite, name)
        if os.path.exists(path) and not force:
            print(f"[{i}/{len(kernels)}] skip {suite}:{name} (exists)", flush=True)
            continue
        print(f"[{i}/{len(kernels)}] measure {suite}:{name} ...", flush=True)
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(_Timeout()))
        signal.alarm(_PER_KERNEL_TIMEOUT)
        try:
            result = _measure_kernel(suite, name)
        except (_Timeout, BaseException) as e:  # incl. pytest.skip -> just don't write a file
            print(f"      not measured: {type(e).__name__}: {str(e)[:80]}", flush=True)
            continue
        finally:
            signal.alarm(0)
        _write_result(path, result)
        probs = _regressions(result)
        if probs:
            print(f"      REGRESSION: {'; '.join(probs)}", flush=True)


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--csv', metavar='PATH', help='export a flat summary CSV from the result dir')
    ap.add_argument('--markdown',
                    metavar='PATH',
                    help='write a Markdown speedup table (default <dir>/speedup_table.md)')
    ap.add_argument('--force', action='store_true', help='re-measure even if a result file exists')
    ap.add_argument('--only', metavar='SUBSTR', help='only kernels whose name contains SUBSTR')
    ap.add_argument('--dir', metavar='PATH', help=f'results directory (default {_RESULTS_DIR})')
    ap.add_argument('--no-run', action='store_true', help='skip measuring; only (re)export CSV / tables')
    ap.add_argument('--save-sdfg',
                    action='store_true',
                    help='also save each pipeline SDFG before libnode expansion to <dir>/sdfg/')
    args = ap.parse_args()
    if args.dir:
        _RESULTS_DIR = args.dir
    if args.save_sdfg:
        _SAVE_SDFG = True
    if not args.no_run:
        _run_sweep(only=args.only, force=args.force)
    _print_tables()
    md_out = args.markdown or os.path.join(_RESULTS_DIR, 'speedup_table.md')
    export_markdown(md_out)
    print(f"\nwrote speedup table to {md_out}", flush=True)
    csv_out = args.csv or os.environ.get('CANON_PERF_CSV')
    if csv_out:
        n = export_csv(csv_out)
        print(f"wrote {n} rows to {csv_out}", flush=True)
