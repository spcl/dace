# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Measure how MAXIMALLY PARALLEL the ``canonicalize`` pipeline is over the four
corpora -- polybench, npbench, tsvc, tsvc_2_5 -- with the correct CPU parameters
(``validate_all=False``, the ``_CPU`` knob set the numerical gate uses).

For every kernel it builds the simplified baseline SDFG and, on deep copies,
applies (a) ``LoopToMap`` repeated and (b) full ``canonicalize``. It then counts,
for each, the number of sequential ``LoopRegion`` / parallel ``MapEntry`` /
``Reduce`` / ``Scan`` constructs.

The headline parallelism metric is the number of **residual sequential loops**
after canonicalize (fewer == more parallel), where a loop that is only the
sequential fallback of a runtime guard ``if cond: <Map> else: <seq loop>``
(``ParallelizeUnderConstraint`` / ``ScatterToGuardedMaps``) is NOT counted as
sequential -- the kernel WAS parallelized, under a predicate.

Two orthogonal facets, selectable:

  * PARALLELISM (default): canonicalize-only, no compile -- fast.
  * CORRECTNESS (``--check``): also ``finalize_for_target('cpu')`` + compile +
    run, and assert the output matches the corpus reference. Canonicalization is
    value-preserving, so every kernel must pass.

Run::

    python -m tests.corpus.measure_parallelization                 # all 4, parallelism
    python -m tests.corpus.measure_parallelization --check         # + correctness
    python -m tests.corpus.measure_parallelization tsvc --peel 8   # one corpus, peel study
    python -m tests.corpus.measure_parallelization --peel 0        # peeling disabled
"""
import os

# Pin a deterministic, single-threaded, no-MPI-init run before DaCe/OpenMP load,
# so the value-preserving assertions never flake on thread races or MPI probes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import argparse
import copy
import csv
import functools
import inspect
import time
from typing import Callable, Dict, List, Tuple

import dace
# Import canonicalize FIRST -- it is the clean entry that fully loads the
# passes.vectorization + interstate packages in the right order. Importing
# ``dace.transformation.interstate`` before canonicalize can trip a circular
# import through the vectorization pipeline's top-level interstate import.
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize.assume_symbols_nonnegative import is_assumption_guard_block
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.parallelize import parallelize
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA
from dace.transformation.passes.vectorization.vectorize_multi_dim import VectorizeCPUMultiDim
from dace.libraries.standard.nodes import Reduce
from dace.libraries.standard.nodes.scan import Scan
from dace.sdfg import nodes as nd
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation.interstate import LoopToMap

from tests.corpus.npbench import npbench as _NB
from tests.corpus.polybench import polybench as _PB
from tests.corpus.tsvc import tsvc as _TS
from tests.corpus.tsvc.tsvc_numpy import REFERENCES as _TS_REF
from tests.corpus.tsvc_2_5 import tsvc_2_5 as _T25
from tests.corpus.tsvc_2_5 import tsvc_2_5_numpy as _T25_REF

# hpcagent_bench is out-of-tree and absent in CI. At module scope it broke COLLECTION for every
# consumer, which aborts the whole run; only the hpcagent sweep needs it, so it imports locally.

#: The correct CPU canonicalize parameters (the numerical gate's ``_CPU`` set).
#: ``peel_limit`` is overridable for the peel study; the rest are the CPU defaults.

# Every corpus compares through ``polybench.outputs_match``, whose tolerance is DTYPE-AWARE:
# integers and bools exactly, fp32 with an fp32-appropriate tolerance, fp64 tightly. A single
# global tolerance is wrong across a corpus that mixes precisions, and skipping integer arrays --
# which the tsvc / tsvc_2_5 checkers used to do -- drops OUTPUTS, not just gather indices: the
# argmax and early-exit kernels exist to test index capture, and their index went unchecked.


def cpu_params(peel_limit: int = 4, reconstruct_wavefront_nest: bool = False) -> Dict:
    return dict(target='cpu',
                peel_limit=peel_limit,
                break_anti_dependence=True,
                interchange_carry_with_map=True,
                scatter_to_guarded_maps=True,
                reconstruct_wavefront_nest=reconstruct_wavefront_nest)


# --------------------------------------------------------------------------- #
# Structural counters.                                                         #
# --------------------------------------------------------------------------- #
#: Column names of :func:`count`, in order -- the single source of truth for the
#: report width, so a new counter needs no second literal kept in sync.
COUNTERS = ('loops', 'maps', 'reduce', 'scan', 'libnode', 'states', 'guards')


def count(sdfg) -> List[int]:
    """``[loops, maps, reduces, scans, libnodes, states, guards]`` structural counts.

    ``states`` tracks the standing invariant that an SDFG carries as few states as
    possible, so a fusion/coalesce regression is visible and not just inferred from
    the map count. Counted over nested SDFGs, matching the node counters' scope.

    ``guards`` counts the assumption-guard states canonicalization appends last. They
    are infrastructure, not payload, and every kernel gains one, so a raw ``states``
    delta reads as fusion regression when it is only the guard: ``states - guards`` is
    the number to compare against the baseline.
    """
    # Count loops at the SAME scope as the node counters below: ``all_control_flow_regions``
    # defaults to ``recursive=False``, which crosses nested REGIONS but not nested SDFGs, while
    # ``all_nodes_recursive`` does cross them. Left asymmetric, a sequential loop that ended up
    # inside a nested SDFG vanished from the metric while its Maps still counted -- polybench
    # ``deriche`` reported 0 residual sequential loops when it actually has 4.
    loops = sum(1 for sd in sdfg.all_sdfgs_recursive() for cfr in sd.all_control_flow_regions()
                if isinstance(cfr, LoopRegion))
    maps = sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nd.MapEntry))
    reduces = sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Reduce))
    scans = sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Scan))
    # Work lifted into a LibraryNode (a MatMul, a BLAS call) is neither a loop nor a Map, so
    # without this column a fully-lifted kernel reads as "0 maps" -- i.e. as a parallelization
    # failure -- when the work simply moved somewhere the other counters do not look. polybench
    # ``trisolv`` is the case: its inner product is a MatMul libnode. Reduce/Scan are
    # LibraryNodes too and already have their own columns, so they are excluded here.
    libnodes = sum(1 for n, _ in sdfg.all_nodes_recursive()
                   if isinstance(n, nd.LibraryNode) and not isinstance(n, (Reduce, Scan)))
    all_states = [st for sd in sdfg.all_sdfgs_recursive() for st in sd.all_states()]
    guards = sum(1 for st in all_states if is_assumption_guard_block(st))
    return [loops, maps, reduces, scans, libnodes, len(all_states), guards]


def guarded_fallback_loops(sdfg) -> int:
    """Number of LoopRegions that are the sequential FALLBACK of a
    conditional-parallelization guard ``if cond: <Map> else: <seq loop>``.

    Such a loop means the kernel WAS parallelized under a runtime predicate, so
    it must count as a parallelized map, not a residual sequential loop. A
    fallback is a LoopRegion in a ``ConditionalBlock`` branch that holds no Map,
    where a sibling branch DOES hold a Map (the parallel form).
    """
    n = 0
    for cfr in sdfg.all_control_flow_regions():
        if not isinstance(cfr, ConditionalBlock):
            continue
        branch_has_map = [
            any(isinstance(x, nd.MapEntry) for x, _ in br.all_nodes_recursive()) for _, br in cfr.branches
        ]
        if not any(branch_has_map):
            continue
        for has_map, (_, br) in zip(branch_has_map, cfr.branches):
            if has_map:
                continue
            n += sum(1 for sub in br.all_control_flow_regions(recursive=True) if isinstance(sub, LoopRegion))
    return n


# --------------------------------------------------------------------------- #
# Per-corpus adapters: name -> (baseline_sdfg, check(finalized_sdfg) -> bool).  #
# The baseline is the simplified SDFG; ``check`` compiles + runs a finalized    #
# copy and compares to the corpus reference (value-preserving == True).         #
# --------------------------------------------------------------------------- #
def _poly_names() -> List[str]:
    return [k.name for k in _PB.collect()]


def _poly_case(name):
    k = _PB.collect(name)[0]
    arrays, psize = _PB.make_inputs(k, size_index=0, cap=None)  # preset S
    ref = _PB.reference(k, arrays, psize)
    return _PB.fresh_sdfg(k), lambda fin: bool(_PB.outputs_match(ref, _PB.run(fin, arrays, psize)))


def _np_names() -> List[str]:
    return [c["name"] for c in _NB.collect()]


def _np_case(name):
    c = _NB.collect(name)[0]
    arrays, params = _NB.make_inputs(c, cap=32)  # preset S
    ref = _NB.reference_outputs(c, arrays, params)
    return _NB.fresh_sdfg(c), lambda fin: bool(_NB.outputs_match(ref, _NB.run_outputs(c, fin, arrays, params)))


def _tsvc_names() -> List[str]:
    return [k.name for k in _TS.collect()]


def tsvc_reference(name):
    """``(arrays, call_kwargs, ref)`` for one tsvc kernel: the inputs, and what the numpy oracle
    makes of them. Shared with the non-vacuity test, which asserts ``ref != arrays``."""
    k = _TS.collect(name=name)[0]
    arrays, ck = _TS.make_inputs(k, seed=1234)
    ref = {n: a.copy() for n, a in arrays.items()}
    _TS_REF[k.name](**ref, **ck)  # numpy oracle writes outputs into ref in place
    return arrays, ck, ref


def _tsvc_case(name):
    k = _TS.collect(name=name)[0]
    base = _TS.to_sdfg(k, tag='measurepar', simplify=True)
    arrays, ck, ref = tsvc_reference(name)

    def check(fin):
        work = {n: a.copy() for n, a in arrays.items()}
        fin.compile()(**work, **ck)
        return bool(_PB.outputs_match(ref, work))

    return base, check


def _tsvc25_names() -> List[str]:
    return [p.name for p in _T25.collect()]


def _tsvc25_oracle(program):
    base = program.name.rsplit("tsvc_2_5_", 1)[-1]
    return vars(_T25_REF)["ref_" + (base[4:] if base.startswith("ext_") else base)]


def tsvc25_reference(program):
    """``(arrays, scalars, ref)`` for one tsvc_2_5 kernel. Shared with the non-vacuity test."""
    arrays, scalars = _T25.make_inputs(program)
    oracle = _tsvc25_oracle(program)
    pool = {
        **{
            n: a.copy()
            for n, a in arrays.items()
        },
        **scalars,
        **{
            s.lower(): v
            for s, v in _T25.SIZES.items()
        }, "n": _T25.SIZES["LEN_1D"]
    }
    oracle(**{p: pool[p] for p in inspect.signature(oracle).parameters})
    return arrays, scalars, {n: pool[n] for n in arrays}


def _tsvc25_case(name):
    program = [p for p in _T25.collect() if p.name == name][0]
    arrays, scalars, ref = tsvc25_reference(program)
    base = program.to_sdfg(simplify=True)

    def check(fin):
        free = {str(s) for s in fin.free_symbols}
        for s in free:
            if s not in fin.symbols:
                fin.add_symbol(s, dace.int64)
        symbols = {s: _T25.SIZES[s] for s in _T25.SIZES if s in free}
        got = {n: a.copy() for n, a in arrays.items()}
        fin.compile()(**got, **scalars, **symbols)
        return bool(_PB.outputs_match(ref, got))

    return base, check


#: One shared DaceFramework('dace_cpu'), pinned to fp64/complex128 BEFORE any kernel's first
#: import: a kernel's ``<module>_dace.py`` does ``from ...dace_framework import dc_float`` at
#: import time -- a plain attribute copy, not a live reference -- so ``set_datatype`` must run
#: before this corpus's first kernel import, once, for the whole process.
@functools.lru_cache(maxsize=1, typed=True)
def _hpcagent_framework():
    from hpcagent_bench.frameworks import DaceFramework

    dfw = DaceFramework('dace_cpu')
    dfw.set_datatype(None)  # fp64 / complex128, matching the corpus's numpy references
    return dfw


def _hpcagent_names() -> List[str]:
    """Path-keys of the hpcagent_bench kernels this sweep measures: every ``foundation`` /
    ``hpc`` / ``ml`` track kernel EXCEPT the ones the ``np`` / ``poly`` corpora above already
    run (``subtrack == 'polybench'`` or tagged ``'npbench'`` -- the identical kernel, ported
    into hpcagent_bench) and any kernel with no DaCe implementation at all. Skips are counted
    and named below, never silent.
    """
    from hpcagent_bench import paths as hpcagent_paths
    from hpcagent_bench.spec import KERNELS, BenchSpec

    covered: List[str] = []
    no_dace: List[str] = []
    names: List[str] = []
    for key in KERNELS.select_keys('all'):
        spec = BenchSpec.load(key)
        if spec.subtrack == 'polybench' or 'npbench' in spec.tags:
            covered.append(key)
            continue
        kdir = hpcagent_paths.BENCHMARKS / spec.relative_path
        if not (kdir / f'{spec.module_name}_dace.py').exists():
            no_dace.append(key)
            continue
        names.append(key)
    print(
        f"[hpcagent] {len(names)} kernels selected; skipped {len(covered)} "
        f"(np/poly duplicates already covered above), {len(no_dace)} (no DaCe impl): "
        f"{sorted(no_dace)}",
        flush=True)
    return sorted(names)


def _hpcagent_case(name):
    from hpcagent_bench.frameworks import Benchmark, TimedCompiledSDFG

    dfw = _hpcagent_framework()
    bench = Benchmark(name)
    program = dfw.implementations(bench)[0][0]
    base = program.to_sdfg(simplify=True)

    def check(fin):
        bdata = bench.get_data(preset='S', datatype=None)
        reference = dfw.reference_outputs(bench, bdata)
        if reference is None:  # no numpy reference available -- nothing to check against
            return True
        variant = TimedCompiledSDFG(fin.compile(), fin, 'canon')
        return dfw.verify(variant, reference, bench, bdata)

    return base, check


CORPORA: Dict[str, Tuple[Callable, Callable]] = {
    'poly': (_poly_names, _poly_case),
    'np': (_np_names, _np_case),
    'tsvc': (_tsvc_names, _tsvc_case),
    'tsvc25': (_tsvc25_names, _tsvc25_case),
    'hpcagent': (_hpcagent_names, _hpcagent_case),
}

#: Pipeline configurations under measurement. Each maps an SDFG in place.
#:
#: * ``canon``          -- the production canonicalize recipe.
#: * ``canon+vec``      -- canonicalize, then the multi-dimensional CPU vectorizer.
#: * ``parallelize+vec``-- the lighter ``parallelize`` recipe, then the vectorizer.
#:
#: The vectorizer runs at a fixed width with the scalar ISA so the measurement
#: is machine-independent: what is being compared is how much of each corpus
#: each recipe leaves parallel, not the throughput of a particular target.
CONFIGS = ('canon', 'canon+vec', 'parallelize+vec')


def _vectorize(sdfg):
    """Apply the CPU multi-dim vectorizer in place."""
    VectorizeCPUMultiDim(VectorizeConfig(widths=(8, ), target_isa=ISA.SCALAR)).apply_pass(sdfg, {})
    return sdfg


def apply_config(sdfg, config: str, params: Dict):
    """Run one pipeline configuration over ``sdfg`` in place.

    :param sdfg: The SDFG to transform.
    :param config: One of :data:`CONFIGS`.
    :param params: Canonicalize knob set from :func:`cpu_params`.
    :returns: The transformed SDFG.
    """
    if config == 'canon':
        canonicalize(sdfg, validate=True, validate_all=False, **params)
    elif config == 'canon+vec':
        canonicalize(sdfg, validate=True, validate_all=False, **params)
        _vectorize(sdfg)
    elif config == 'parallelize+vec':
        parallelize(sdfg, validate=True, validate_all=False, peel_limit=params.get('peel_limit', 4))
        _vectorize(sdfg)
    else:
        raise ValueError(f'unknown config {config!r}')
    return sdfg


def sweep(corpus: str,
          peel_limit: int = 4,
          check: bool = False,
          verbose: bool = True,
          config: str = 'canon',
          shard: Tuple[int, int] = (0, 1)) -> Dict:
    """Measure one corpus. :returns: a result dict with per-kernel rows.

    ``shard=(index, count)`` keeps only every ``count``-th kernel starting at ``index``,
    so a batch job can fan the corpus out over ranks without any rank-to-kernel table.
    Round-robin rather than contiguous blocks: kernel cost varies by an order of
    magnitude and neighbours in the (sorted) name list tend to be similar sizes.
    """
    names_fn, case_fn = CORPORA[corpus]
    index, total = shard
    names = names_fn()[index::total]
    params = cpu_params(peel_limit)
    rows: Dict[str, Dict] = {}
    t0 = time.perf_counter()
    for i, name in enumerate(names, 1):
        row = dict(base=None, l2m=None, canon=None, guarded=0, correct=None, error=None)
        try:
            base, checker = case_fn(name)
            row['base'] = count(base)
            l2m = copy.deepcopy(base)
            l2m.apply_transformations_repeated(LoopToMap, validate=False, validate_all=False)
            row['l2m'] = count(l2m)
            canon = copy.deepcopy(base)
            apply_config(canon, config, params)
            row['canon'] = count(canon)
            row['guarded'] = guarded_fallback_loops(canon)
            if check:
                fin = finalize_for_target(copy.deepcopy(canon), 'cpu')
                fin.name = f"{fin.name}_mp_{i}"
                row['correct'] = bool(checker(fin))
        except Exception as e:  # a build/codegen raise is recorded, not fatal to the sweep
            # Some validation errors carry a stale node/state id whose own ``__str__`` re-raises
            # (e.g. ``NodeNotFoundError`` after the offending node was removed); stringifying the
            # message must not itself abort the sweep, so fall back to the type name alone.
            try:
                detail = str(e)[:160]
            except Exception:
                detail = '<message unavailable>'
            row['error'] = f"{type(e).__name__}: {detail}"
        rows[name] = row
        if verbose:
            flag = 'OK ' if row['correct'] else ('.. ' if row['correct'] is None and not row['error'] else
                                                 ('ERR' if row['error'] else 'BAD'))
            print(
                f"[{corpus} {config} p{peel_limit} {i:3d}/{len(names)}] {flag} {name:28s} "
                f"base={row['base']} l2m={row['l2m']} canon={row['canon']} g={row['guarded']} "
                f"{row['error'] or ''}",
                flush=True)
    return dict(corpus=corpus,
                config=config,
                peel_limit=peel_limit,
                seconds=round(time.perf_counter() - t0, 1),
                rows=rows)


#: Per-kernel CSV header: the three measured phases crossed with COUNTERS, plus the verdicts.
CSV_FIELDS = (('config', 'corpus', 'kernel') + tuple(f'{phase}_{c}' for phase in ('base', 'l2m', 'canon')
                                                     for c in COUNTERS) + ('guarded', 'correct', 'error'))


def write_csv(res: Dict, path: str) -> None:
    """Append ``res``'s per-kernel rows to ``path`` (writing the header if new)."""
    fresh = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', newline='') as fh:
        writer = csv.DictWriter(fh, CSV_FIELDS)
        if fresh:
            writer.writeheader()
        for name, row in res['rows'].items():
            flat = dict(config=res['config'], corpus=res['corpus'], kernel=name)
            for phase in ('base', 'l2m', 'canon'):
                counts = row.get(phase) or [''] * len(COUNTERS)
                flat.update({f'{phase}_{c}': v for c, v in zip(COUNTERS, counts)})
            flat.update(guarded=row['guarded'], correct=row['correct'], error=row['error'] or '')
            writer.writerow(flat)


def summarize_csv(paths: List[str]) -> int:
    """Print per-(config, corpus) totals and every recorded error from sharded CSVs.

    :returns: the number of kernels that errored OR answered wrong -- the batch job's exit
              status, so a corpus that stops canonicalizing (or silently miscompiles) fails
              the job instead of scrolling past.
    """
    rows: List[Dict[str, str]] = []
    for path in paths:
        with open(path, newline='') as fh:
            rows.extend(csv.DictReader(fh))

    groups: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row['config'], row['corpus']), []).append(row)

    def is_wrong(row: Dict[str, str]) -> bool:
        # Only ``--check`` runs fill this in; an unchecked run leaves it empty, which is
        # "not measured", not "correct".
        return row['correct'] == 'False'

    print(f"\n{'config':16s} {'corpus':10s} {'n':>4s}  " + '  '.join(f'{c:>7s}' for c in COUNTERS) + '   err  wrong')
    for (config, corpus), grp in sorted(groups.items()):
        totals = [sum(int(r[f'canon_{c}']) for r in grp if r[f'canon_{c}']) for c in COUNTERS]
        errs = sum(1 for r in grp if r['error'])
        wrong = sum(1 for r in grp if is_wrong(r))
        print(f"{config:16s} {corpus:10s} {len(grp):4d}  " + '  '.join(f'{t:7d}'
                                                                       for t in totals) + f"   {errs:3d}  {wrong:5d}")

    errored = [r for r in rows if r['error']]
    if errored:
        print(f"\n=== {len(errored)} ERRORS ===")
        for r in sorted(errored, key=lambda r: (r['corpus'], r['kernel'])):
            print(f"  {r['corpus']:10s} {r['kernel']:28s} {r['error']}")

    # A kernel that canonicalizes cleanly and answers wrong is the worse failure of the two:
    # nothing in the structural counts reveals it. Report it last so it is what you see.
    wrong = [r for r in rows if is_wrong(r)]
    if wrong:
        print(f"\n=== {len(wrong)} MISCOMPILES (canonicalize is value-preserving; these are bugs) ===")
        for r in sorted(wrong, key=lambda r: (r['corpus'], r['kernel'])):
            print(f"  {r['corpus']:10s} {r['kernel']}")
    return len(errored) + len(wrong)


def _agg(rows, key) -> List[int]:
    tot = None
    for r in rows.values():
        c = r.get(key)
        if c:
            if tot is None:
                tot = list(c)
            else:
                for j, v in enumerate(c):
                    tot[j] += v
    return tot if tot is not None else [0] * len(COUNTERS)


def summarize(res: Dict) -> None:
    rows = res['rows']
    ok = [n for n, r in rows.items() if r['correct'] is True]
    bad = [n for n, r in rows.items() if r['correct'] is False and not r['error']]
    err = [n for n, r in rows.items() if r['error']]
    b, l, c = _agg(rows, 'base'), _agg(rows, 'l2m'), _agg(rows, 'canon')
    guarded = sum(r.get('guarded') or 0 for r in rows.values())
    eff = c[0] - guarded
    print(f"\n===== {res['corpus']} [{res.get('config', 'canon')}] peel_limit={res['peel_limit']} "
          f"({len(rows)} kernels, {res['seconds']}s) =====")
    if ok or bad or err:
        # Errored kernels count against the denominator: a kernel that failed to build was NOT
        # shown to be correct, and "CORRECT: 10/10, ERROR: 20" reads as full coverage.
        print(f"  CORRECT: {len(ok)}/{len(ok) + len(bad) + len(err)}   WRONG: {len(bad)}   ERROR: {len(err)}")
        if bad:
            print(f"    WRONG: {', '.join(sorted(bad))}")
        for n in sorted(err):
            print(f"    ERROR {n}: {rows[n]['error']}")
    header = ''.join(f'{name:>8s}' for name in COUNTERS)
    print(f"  {'strategy':14s}{header}")
    for label, totals in (('baseline', b), ('LoopToMap', l), (res.get('config', 'canon'), c)):
        print(f"  {label:14s}" + ''.join(f'{v:8d}' for v in totals))
    print(f"  residual sequential loops: baseline={b[0]}  L2M={l[0]}  canon={c[0]}")
    print(f"  guarded (if cond: map else: seq) fallbacks counted as parallel: {guarded}")
    print(f"  EFFECTIVE residual sequential (canon - guarded): {eff}  "
          f"(parallelized {b[0] - eff}/{b[0]} = {100 * (b[0] - eff) / max(1, b[0]):.1f}%)")

    def _eff(r):
        return (r['canon'][0] - (r.get('guarded') or 0)) if r['canon'] else None

    worse = [n for n, r in rows.items() if r['l2m'] and r['canon'] and _eff(r) > r['l2m'][0]]
    if worse:
        print(f"  * canon MORE sequential than L2M: {', '.join(sorted(worse))}")
    seqleft = sorted(n for n, r in rows.items() if r['canon'] and _eff(r) > 0)
    if seqleft:
        print(f"  loops still sequential after canon ({len(seqleft)}): {', '.join(seqleft)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('corpus', nargs='?', choices=list(CORPORA) + ['all'], default='all')
    ap.add_argument('--peel', type=int, default=4, help='peel_limit (default 4; 0 disables peeling)')
    ap.add_argument('--check', action='store_true', help='also compile+run and assert value-preserving')
    ap.add_argument('--config',
                    default='canon',
                    choices=list(CONFIGS) + ['all'],
                    help='pipeline configuration to measure (default canon)')
    ap.add_argument('--shard', default='0/1', help='"i/n": measure only every n-th kernel starting at i')
    ap.add_argument('--csv', default=None, help='append per-kernel rows to this CSV')
    ap.add_argument('--summarize', nargs='+', default=None, help='report on existing CSVs instead of measuring')
    args = ap.parse_args()
    if args.summarize:
        raise SystemExit(1 if summarize_csv(args.summarize) else 0)
    index, total = (int(p) for p in args.shard.split('/'))
    targets = list(CORPORA) if args.corpus == 'all' else [args.corpus]
    configs = list(CONFIGS) if args.config == 'all' else [args.config]
    for config in configs:
        for corpus in targets:
            res = sweep(corpus, peel_limit=args.peel, check=args.check, config=config, shard=(index, total))
            if args.csv:
                write_csv(res, args.csv)
            summarize(res)


if __name__ == '__main__':
    main()
