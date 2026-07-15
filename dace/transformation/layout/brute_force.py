# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Brute-force GLOBAL layout sweep -- the layout optimizer (no cost model).

The SC26 layout picker is deliberately brute force: enumerate global layout candidates for a
kernel (dimension permutations, block factors, zip groupings), apply each, compile, VERIFY against
a numpy oracle, measure time, and keep the fastest correct one. The capability -- the transforms +
algebra + ``LayoutChange`` node -- is the deliverable; the picker is a sweep over global layouts.

This module is the reusable engine:

  * :func:`sweep` runs a set of named candidate SDFG builders through compile -> run -> verify ->
    (optional) time, and returns the results ranked (correct first, then by time).
  * :func:`permutation_candidates` / :func:`block_candidates` enumerate the common candidate
    families (per-array dimension permutations; per-dimension block factors) as ``apply`` closures
    over the layout passes.
  * :func:`time_cpu` is a median-of-reps wall-clock timer (best-effort; timing on a shared/loaded
    host is noisy -- correctness is the invariant, speed is advisory).

The caller supplies a ``run`` closure that binds fresh inputs, executes a compiled SDFG, and returns
the outputs to compare, plus the reference outputs; the engine owns the compile/verify/time/rank
loop. Timing never runs for an incorrect candidate.
"""
import itertools
import time as _time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy

import dace


@dataclass
class SweepResult:
    """One candidate's outcome: whether it verified, its median time (if timed), and any error."""
    name: str
    correct: bool
    time: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def time_cpu(fn: Callable[[], Any], reps: int = 5, warmup: int = 1) -> float:
    """Median wall-clock time (seconds) of ``fn`` over ``reps`` runs after ``warmup`` warmups."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(reps):
        t0 = _time.perf_counter()
        fn()
        samples.append(_time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2]


def sweep(candidates: Dict[str, Callable[[], dace.SDFG]],
          run: Callable[[dace.SDFG], Dict[str, numpy.ndarray]],
          reference: Dict[str, numpy.ndarray],
          compare: Callable[[numpy.ndarray, numpy.ndarray], bool] = numpy.allclose,
          reps: int = 5,
          warmup: int = 1,
          do_time: bool = True) -> List[SweepResult]:
    """Compile, run, verify and (optionally) time each candidate; return results ranked.

    :param candidates: ``{name: make_sdfg}`` -- each ``make_sdfg()`` returns a FRESH SDFG with that
                       candidate's global layout already applied.
    :param run: ``run(sdfg) -> {output_name: array}`` -- binds fresh inputs, executes ``sdfg`` and
                returns the outputs to check (the caller owns argument marshalling).
    :param reference: the numpy-oracle outputs, ``{output_name: array}``.
    :param compare: elementwise correctness predicate (default ``numpy.allclose``).
    :param do_time: time each CORRECT candidate (never an incorrect one).
    :returns: results, correct-first then ascending time.
    """
    results: List[SweepResult] = []
    for name, make in candidates.items():
        try:
            sdfg = make()
            out = run(sdfg)
            correct = all(name_ in out and compare(out[name_], ref) for name_, ref in reference.items())
            t = time_cpu(lambda: run(sdfg), reps, warmup) if (do_time and correct) else None
            results.append(SweepResult(name, correct, t))
        except Exception as ex:  # a candidate that fails to build/compile/run is simply not viable
            results.append(SweepResult(name, False, None, f"{type(ex).__name__}: {ex}"))
    results.sort(key=lambda r: (not r.correct, r.time if r.time is not None else float('inf')))
    return results


def best(results: List[SweepResult]) -> Optional[SweepResult]:
    """The fastest correct candidate (results are already ranked), or ``None`` if none verified."""
    return next((r for r in results if r.correct), None)


# --------------------------------------------------------------------------- #
#  Candidate-family enumerators (apply closures over the layout passes)
# --------------------------------------------------------------------------- #
def permutation_candidates(array: str, ndim: int):
    """Yield ``(name, apply)`` for every dimension permutation of ``array`` (identity included).

    ``apply(sdfg)`` runs ``PermuteDimensions`` in-place for that permutation.
    """
    from dace.transformation.layout.permute_dimensions import PermuteDimensions

    for perm in itertools.permutations(range(ndim)):

        def apply(sdfg, perm=perm):
            PermuteDimensions(permute_map={array: list(perm)}, add_permute_maps=True).apply_pass(sdfg, {})

        yield f"permute_{array}_{''.join(map(str, perm))}", apply


def block_candidates(array: str, ndim: int, factors: Tuple[int, ...] = (8, 16, 32)):
    """Yield ``(name, apply)`` for blocking ONE dimension of ``array`` by each factor (plus the
    unblocked identity). ``apply(sdfg)`` runs ``SplitDimensions`` then
    ``normalize_schedule_for_layout``."""
    from dace.transformation.layout.split_dimensions import SplitDimensions
    from dace.transformation.layout.normalize_schedule import normalize_schedule_for_layout

    yield f"noblock_{array}", (lambda sdfg: None)
    for dim in range(ndim):
        for factor in factors:

            def apply(sdfg, dim=dim, factor=factor):
                masks = [i == dim for i in range(ndim)]
                facs = [factor if i == dim else 1 for i in range(ndim)]
                SplitDimensions(split_map={array: (masks, facs)}).apply_pass(sdfg, {})
                normalize_schedule_for_layout(sdfg)

            yield f"block_{array}_d{dim}_{factor}", apply
