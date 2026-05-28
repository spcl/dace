# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Measure loop-to-map parallelization rate of ``LoopToMap`` alone vs the full
``canonicalize`` pipeline over the TSVC corpus.

Run::

    python -m tests.corpus.measure_parallelization

Prints aggregate counts of ``LoopRegion`` / ``MapEntry`` / ``Reduce`` nodes
before and after each strategy, the per-strategy conversion deltas, and a
per-kernel table of kernels where the two strategies disagree in map count.
"""
import copy
import time
import traceback
from typing import Tuple

from dace.libraries.standard.nodes import Reduce
from dace.sdfg import nodes as nd
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.canonicalize.pipeline import canonicalize

from tests.corpus import tsvc

#: Match the corpus-test settings.
_PEEL_LIMIT = 4
_BREAK_ANTI_DEP = True


def _count(sdfg) -> Tuple[int, int, int]:
    """:returns: ``(loops, maps, reduces)`` -- LoopRegions, MapEntries, Reduce libnodes."""
    loops = sum(1 for cfr in sdfg.all_control_flow_regions() if isinstance(cfr, LoopRegion))
    maps = sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nd.MapEntry))
    reduces = sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Reduce))
    return loops, maps, reduces


def _measure(kernel) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]:
    """Build the kernel's SDFG (simplified), then measure ``(L, M, R)`` for:

    * baseline (simplified only),
    * after ``LoopToMap`` repeated,
    * after ``canonicalize(peel_limit=4, break_anti_dependence=True)``.
    """
    base = kernel.program.to_sdfg(simplify=True)
    base_counts = _count(base)

    l2m = copy.deepcopy(base)
    l2m.apply_transformations_repeated(LoopToMap, validate=False, validate_all=False)
    l2m_counts = _count(l2m)

    canon = copy.deepcopy(base)
    canonicalize(canon, peel_limit=_PEEL_LIMIT, break_anti_dependence=_BREAK_ANTI_DEP)
    canon_counts = _count(canon)

    return base_counts, l2m_counts, canon_counts


def _pct(num: int, den: int) -> str:
    return f"{(100.0 * num / den):5.1f}%" if den else "  n/a"


def main() -> None:
    kernels = tsvc.collect()
    print(f"TSVC parallelization measurement ({len(kernels)} kernels)")
    print(f"  peel_limit={_PEEL_LIMIT}  break_anti_dependence={_BREAK_ANTI_DEP}")
    print()

    results = {}  # name -> (base, l2m, canon)
    errors = {}   # name -> str
    t0 = time.perf_counter()
    for i, k in enumerate(kernels, 1):
        try:
            results[k.name] = _measure(k)
        except Exception:
            errors[k.name] = traceback.format_exc(limit=2)
        if i % 10 == 0 or i == len(kernels):
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

    init_iter = agg_b[0] + agg_b[1]  # initial loops+maps (iteration constructs)
    l2m_eliminated = agg_b[0] - agg_l[0]
    canon_eliminated = agg_b[0] - agg_c[0]
    l2m_to_map = agg_l[1] - agg_b[1]
    canon_to_map = agg_c[1] - agg_b[1]
    canon_to_reduce = agg_c[2] - agg_b[2]

    print("Conversion (vs. baseline):")
    print(f"  {'loops -> maps':32s} L2M={l2m_to_map:4d}   canon={canon_to_map:4d}")
    print(f"  {'loops -> reduce':32s} L2M={0:4d}   canon={canon_to_reduce:4d}")
    print(f"  {'loops eliminated (any cause)':32s} L2M={l2m_eliminated:4d}   canon={canon_eliminated:4d}")
    print(f"  {'baseline loops parallelized':32s} L2M={_pct(l2m_to_map, agg_b[0])}  "
          f"canon={_pct(canon_to_map, agg_b[0])}")
    print(f"  {'baseline loops eliminated':32s} L2M={_pct(l2m_eliminated, agg_b[0])}  "
          f"canon={_pct(canon_eliminated, agg_b[0])}")
    print(f"  {'final iteration is parallel':32s} L2M={_pct(agg_l[1], agg_l[0]+agg_l[1])}  "
          f"canon={_pct(agg_c[1], agg_c[0]+agg_c[1])}")
    print()

    # Per-kernel: where canon parallelizes more than L2M (most interesting deltas)
    diffs = []
    for name, (b, l, c) in results.items():
        extra_maps = (c[1] - b[1]) - (l[1] - b[1])
        if extra_maps != 0 or (c[0] != l[0]):
            diffs.append((extra_maps, name, b, l, c))
    diffs.sort(key=lambda r: (-r[0], r[1]))

    if diffs:
        print(f"Kernels where canonicalize differs from L2M ({len(diffs)} of {len(results)}):")
        print(f"  {'kernel':22s}  base(L,M,R)   l2m(L,M,R)    canon(L,M,R)  extra_maps")
        for extra, name, b, l, c in diffs[:40]:
            print(f"  {name:22s}  {str(b):12s}  {str(l):12s}  {str(c):14s} {extra:+d}")
        if len(diffs) > 40:
            print(f"  ... and {len(diffs) - 40} more")
        print()

    if errors:
        print(f"Errors ({len(errors)} kernels failed measurement):")
        for name, tb in errors.items():
            print(f"  {name}: {tb.strip().splitlines()[-1]}")


if __name__ == '__main__':
    main()
