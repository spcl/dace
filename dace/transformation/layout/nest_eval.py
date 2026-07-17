# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Extract one kernel, evaluate its layout candidates, return the ranked timings and the best
(GLOBAL_LAYOUT_DESIGN.md, task B3's per-nest core).

This is the reusable wrapper the per-nest phases share: :func:`evaluate_nest` externalizes a nest
(task A1), runs the UNMODIFIED copy once for the reference outputs, then drives layout candidates
through :func:`brute_force.sweep` -- compile, verify against the reference, time with the
compute-region timer (median of ~10 reps + the spread trust signal; the relayout copies a wrap-mode
candidate inserts are excluded by construction). One engine end to end: candidates are ``{tag:
apply}`` closures exactly as ``brute_force`` already defines them, results are ``SweepResult``s,
``best`` is ``brute_force.best``.

Two laws from the design live here, not in callers:

  * **Candidate identity is ``{nest}__{layout_tag}``** -- every candidate SDFG is renamed to it, so
    each gets a disjoint build folder (the same-name build-cache hazard).
  * **The identity candidate is enumerated FIRST and wins ties** -- enumeration order is
    load-bearing; ``sweep``'s stable ranking then breaks equal times toward identity.

Candidates default to the wrap-mode permutation family (interface-preserving: the caller's buffers
stay logical, the relayout happens inside), so the same argument set drives every candidate; pass
custom ``{tag: apply}`` closures for anything richer (task B1 supplies the pruned generator).
"""
import copy
import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy

from dace import SDFG, SDFGState
from dace.sdfg import nodes
from dace.transformation.layout.brute_force import SweepResult, best, sweep
from dace.transformation.layout.externalize import (externalize_nest, nest_arguments, written_array_names)
from dace.transformation.layout.timing import compute_region_stats_timer

#: The identity candidate's tag -- FIRST in every candidate dict (the tie-break law).
IDENTITY_TAG = "identity"


def default_permutation_candidates(ext: SDFG) -> Dict[str, Callable[[SDFG], None]]:
    """The wrap-mode permutation family: identity FIRST, then every non-identity dimension
    permutation of every >=2-D non-transient array. ``apply(sdfg)`` permutes the array's storage and
    inserts the boundary relayout maps, so the caller-facing interface stays logical."""
    from dace.transformation.layout.permute_dimensions import PermuteDimensions

    candidates: Dict[str, Callable[[SDFG], None]] = {IDENTITY_TAG: lambda sdfg: None}
    for aname in sorted(ext.arrays):
        desc = ext.arrays[aname]
        if desc.transient or len(desc.shape) < 2:
            continue
        ndim = len(desc.shape)
        for perm in itertools.permutations(range(ndim)):
            if list(perm) == list(range(ndim)):
                continue

            def apply(sdfg, aname=aname, perm=perm):
                PermuteDimensions(permute_map={aname: list(perm)}, add_permute_maps=True).apply_pass(sdfg, {})

            candidates[f"permute_{aname}_{''.join(map(str, perm))}"] = apply
    return candidates


@dataclass
class NestEvaluation:
    """One nest's evaluated candidate sweep: the externalized baseline, the reference outputs it
    produced, the deterministic argument set, and the ranked results."""
    ext: SDFG
    reference: Dict[str, numpy.ndarray]
    arguments: Dict[str, numpy.ndarray]
    symbols: Dict[str, int]
    results: List[SweepResult] = field(default_factory=list)

    def best(self) -> Optional[SweepResult]:
        """The fastest correct candidate, or ``None`` if none verified."""
        return best(self.results)


def call_symbols(ext: SDFG, symbols: Dict[str, int]) -> Dict[str, int]:
    """The subset of ``symbols`` the externalized nest actually declares (an SDFG call refuses
    unknown keyword arguments, and a shared config may carry sizes for other nests)."""
    return {name: value for name, value in symbols.items() if name in ext.symbols}


def evaluate_nest(state: SDFGState,
                  map_entry: Optional[nodes.MapEntry] = None,
                  *,
                  symbols: Dict[str, int],
                  provided: Optional[Dict[str, numpy.ndarray]] = None,
                  candidates: Optional[Dict[str, Callable[[SDFG], None]]] = None,
                  device: str = "cpu",
                  reps: int = 10,
                  warmup: int = 2,
                  timer: Optional[Callable] = compute_region_stats_timer,
                  seed: int = 0,
                  name: Optional[str] = None) -> NestEvaluation:
    """Externalize the nest under ``map_entry`` and rank its layout candidates by measured time.

    Pipeline (one nest): externalize -> deterministic arguments -> reference run on the pristine
    copy -> per candidate: deepcopy + rename to ``{nest}__{tag}`` + ``apply`` -> ``sweep``
    (compile, verify vs the reference, time). The default timer is the compute-region stats timer,
    so each ``SweepResult`` carries the median plus ``spread``/``contended`` metadata and wrap-mode
    relayout copies never pollute the ranking.

    :param state: the state holding the nest (or the state of an already-split program).
    :param map_entry: the nest's top-level map entry (``None`` = the state's single nest).
    :param symbols: concrete sizes for free symbols -- the v1 concrete-shapes contract.
    :param provided: caller-supplied input arrays by name; everything else is deterministic random.
    :param candidates: ``{tag: apply}`` closures; default = wrap-mode permutation family.
    :param device: ``"cpu"`` or ``"gpu"`` (forwarded to ``sweep``).
    :param timer: ``sweep`` timer; default records median + spread of the compute region.
    :param seed: seed for the deterministic argument fill.
    :param name: base name for the externalized nest (default from the SDFG + map label).
    :return: the :class:`NestEvaluation` (ranked results; ``.best()`` for the winner).
    """
    ext = externalize_nest(state, map_entry, name=name)
    written = written_array_names(ext)
    if not written:
        raise ValueError(f"evaluate_nest: nest '{ext.name}' writes no non-transient array; "
                         f"nothing to verify a candidate against")
    args = nest_arguments(ext, symbols, provided, seed)
    syms = call_symbols(ext, symbols)

    reference_args = {k: v.copy() for k, v in args.items()}
    reference_sdfg = copy.deepcopy(ext)
    reference_sdfg.name = f"{ext.name}__reference"
    reference_sdfg(**reference_args, **syms)
    reference = {out: reference_args[out] for out in sorted(written)}

    if candidates is None:
        candidates = default_permutation_candidates(ext)
    if next(iter(candidates), None) != IDENTITY_TAG:
        raise ValueError(f"evaluate_nest: the '{IDENTITY_TAG}' candidate must be enumerated first "
                         f"(the tie-break law); got order {list(candidates)[:3]}...")

    def make_for(tag: str, apply: Callable[[SDFG], None]) -> Callable[[], SDFG]:

        def make() -> SDFG:
            cand = copy.deepcopy(ext)
            cand.name = f"{ext.name}__{tag}"
            apply(cand)
            return cand

        return make

    def run(sdfg: SDFG) -> Dict[str, numpy.ndarray]:
        run_args = {k: v.copy() for k, v in args.items()}
        sdfg(**run_args, **syms)
        return {out: run_args[out] for out in sorted(written)}

    results = sweep({tag: make_for(tag, apply) for tag, apply in candidates.items()},
                    run,
                    reference,
                    reps=reps,
                    warmup=warmup,
                    device=device,
                    timer=timer)
    return NestEvaluation(ext=ext, reference=reference, arguments=args, symbols=syms, results=results)
