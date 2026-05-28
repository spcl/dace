# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Measure loop-to-map parallelization rate of ``LoopToMap`` alone vs the full
``canonicalize`` pipeline over the TSVC corpus.

Run::

    python -m tests.corpus.measure_parallelization

For every kernel: build the simplified baseline SDFG, then count LoopRegions /
MapEntries / Reduce libnodes after (a) ``LoopToMap`` repeated and (b) full
``canonicalize``. Prints an aggregate table, a per-kernel table, and an
"inspection" section that dumps the actual map ranges + remaining loops for a
configurable set of kernels (default: those where canon and L2M produce
different map counts or where canon leaves more loops sequential).
"""
import copy
import time
import traceback
from typing import List, Tuple

from dace.libraries.standard.nodes import Reduce
from dace.sdfg import nodes as nd
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.canonicalize.pipeline import canonicalize

from tests.corpus import tsvc

#: Match the corpus-test settings.
_PEEL_LIMIT = 4
_BREAK_ANTI_DEP = True

#: Kernels to dump in detail at the end (in addition to auto-detected anomalies).
_FORCE_INSPECT = {'s1115_d_single', 's152_d_single', 's172_d_single'}


def _count(sdfg) -> Tuple[int, int, int]:
    """:returns: ``(loops, maps, reduces)``."""
    loops = sum(1 for cfr in sdfg.all_control_flow_regions() if isinstance(cfr, LoopRegion))
    maps = sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nd.MapEntry))
    reduces = sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Reduce))
    return loops, maps, reduces


def _structure(sdfg) -> Tuple[List[str], List[str], List[str]]:
    """Per-kernel structural summary: list of loop / map / reduce descriptions."""
    loops = [
        f"loop {cfr.label} var={cfr.loop_variable!s} cond={cfr.loop_condition.as_string!s}"
        for cfr in sdfg.all_control_flow_regions() if isinstance(cfr, LoopRegion)
    ]
    maps = [f"map {n.map.label} dims={len(n.map.range)} range={n.map.range}"
            for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nd.MapEntry)]
    reduces = [f"reduce axes={n.axes} wcr={n.wcr}"
               for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Reduce)]
    return loops, maps, reduces


def _build(kernel):
    """Build the kernel's simplified baseline SDFG and apply each strategy on a deep copy."""
    base = kernel.program.to_sdfg(simplify=True)
    l2m = copy.deepcopy(base)
    l2m.apply_transformations_repeated(LoopToMap, validate=False, validate_all=False)
    canon = copy.deepcopy(base)
    canonicalize(canon, peel_limit=_PEEL_LIMIT, break_anti_dependence=_BREAK_ANTI_DEP)
    return base, l2m, canon


def _pct(num: int, den: int) -> str:
    return f"{(100.0 * num / den):5.1f}%" if den else "  n/a"


def _status(b, l, c) -> str:
    """One-line status marker for a kernel: parallel-construct delta + sequential-regression flag."""
    l2m_par = l[1] + l[2]
    canon_par = c[1] + c[2]
    seq_delta = c[0] - l[0]
    parts = []
    if canon_par > l2m_par:
        parts.append(f"+{canon_par - l2m_par}")
    elif canon_par < l2m_par:
        parts.append(f"-{l2m_par - canon_par}")
    else:
        parts.append("=")
    if (c[2] - b[2]) > 0:
        parts.append(f"R+{c[2] - b[2]}")
    if seq_delta > 0:
        parts.append(f"!seq+{seq_delta}")
    return " ".join(parts)


def _print_inspection(name, base, l2m, canon) -> None:
    print(f"Inspection: {name}")
    for label, s in (("baseline", base), ("LoopToMap", l2m), ("canon", canon)):
        loops, maps, reduces = _structure(s)
        print(f"  {label}: loops={len(loops)}  maps={len(maps)}  reduces={len(reduces)}")
        for x in loops:
            print(f"    {x}")
        for x in maps:
            print(f"    {x}")
        for x in reduces:
            print(f"    {x}")
    print()


def main() -> None:
    kernels = tsvc.collect()
    print(f"TSVC parallelization measurement ({len(kernels)} kernels)")
    print(f"  peel_limit={_PEEL_LIMIT}  break_anti_dependence={_BREAK_ANTI_DEP}")
    print()

    results = {}   # name -> (base_counts, l2m_counts, canon_counts)
    inspect = {}   # name -> (base_sdfg, l2m_sdfg, canon_sdfg) for kernels we want to dump
    errors = {}
    t0 = time.perf_counter()
    for i, k in enumerate(kernels, 1):
        try:
            base, l2m, canon = _build(k)
            results[k.name] = (_count(base), _count(l2m), _count(canon))
            l2m_par = results[k.name][1][1] + results[k.name][1][2]
            canon_par = results[k.name][2][1] + results[k.name][2][2]
            seq_delta = results[k.name][2][0] - results[k.name][1][0]
            if k.name in _FORCE_INSPECT or canon_par < l2m_par or seq_delta > 0:
                inspect[k.name] = (base, l2m, canon)
        except Exception:
            errors[k.name] = traceback.format_exc(limit=2)
        if i % 20 == 0 or i == len(kernels):
            print(f"  [{i:3d}/{len(kernels)}] {time.perf_counter()-t0:6.1f}s elapsed", flush=True)
    print()

    # Aggregate
    agg_b = [sum(r[0][k] for r in results.values()) for k in range(3)]
    agg_l = [sum(r[1][k] for r in results.values()) for k in range(3)]
    agg_c = [sum(r[2][k] for r in results.values()) for k in range(3)]
    print("Aggregate (sum across kernels):")
    print(f"  {'strategy':16s} {'loops':>7s} {'maps':>7s} {'reduces':>8s}")
    print(f"  {'baseline':16s} {agg_b[0]:7d} {agg_b[1]:7d} {agg_b[2]:8d}")
    print(f"  {'LoopToMap-only':16s} {agg_l[0]:7d} {agg_l[1]:7d} {agg_l[2]:8d}")
    print(f"  {'canonicalize':16s} {agg_c[0]:7d} {agg_c[1]:7d} {agg_c[2]:8d}")
    print()
    print("Conversion vs baseline:")
    print(f"  loops -> maps                    L2M={agg_l[1]-agg_b[1]:4d}   canon={agg_c[1]-agg_b[1]:4d}")
    print(f"  loops -> reduce                  L2M={0:4d}   canon={agg_c[2]-agg_b[2]:4d}")
    print(f"  loops eliminated (any cause)     L2M={agg_b[0]-agg_l[0]:4d}   canon={agg_b[0]-agg_c[0]:4d}")
    print(f"  baseline loops parallelized      L2M={_pct(agg_l[1]-agg_b[1], agg_b[0])}  "
          f"canon={_pct((agg_c[1]-agg_b[1])+(agg_c[2]-agg_b[2]), agg_b[0])}")
    print(f"  final iteration is parallel      L2M={_pct(agg_l[1]+agg_l[2], agg_l[0]+agg_l[1]+agg_l[2])}  "
          f"canon={_pct(agg_c[1]+agg_c[2], agg_c[0]+agg_c[1]+agg_c[2])}")
    print()

    # Per-kernel table (all 151)
    print("Per-kernel (status: '=' same parallel-construct count; '+N' canon has N more parallel "
          "constructs; '-N' canon has N fewer; 'R+N' canon emits N reduces; '!seq+N' canon left N "
          "more loops sequential than L2M -- the regression marker):")
    print(f"  {'kernel':22s}  {'base':>10s}  {'l2m':>10s}  {'canon':>10s}  status")
    for name in sorted(results.keys()):
        b, l, c = results[name]
        print(f"  {name:22s}  {str(b):>10s}  {str(l):>10s}  {str(c):>10s}  {_status(b, l, c)}")
    print()

    if inspect:
        print("=" * 78)
        print(f"Detailed inspection of {len(inspect)} kernels (auto-anomalies + forced):")
        print("=" * 78)
        for name in sorted(inspect.keys()):
            base, l2m, canon = inspect[name]
            _print_inspection(name, base, l2m, canon)

    if errors:
        print(f"Errors ({len(errors)} kernels failed measurement):")
        for name, tb in errors.items():
            print(f"  {name}: {tb.strip().splitlines()[-1]}")


if __name__ == '__main__':
    main()
