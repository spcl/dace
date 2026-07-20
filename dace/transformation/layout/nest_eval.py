# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Extract one kernel, evaluate its layout candidates, and return the ranked timings and the best."""
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

#: identity candidate's tag; must be enumerated first (tie-break law)
IDENTITY_TAG = "identity"

#: v1 bound on full permutation enumeration (d!); refused loudly above. Shared with the modelled path
#: (``assignment_costs.permutation_layouts``) so the measured and modelled candidate spaces cannot drift.
MAX_PERMUTE_NDIM = 3


def default_permutation_candidates(ext: SDFG) -> Dict[str, Callable[[SDFG], None]]:
    """Wrap-mode permutation family: identity first, then every non-identity dimension permutation of each >=2-D non-transient array."""
    from dace.transformation.layout.permute_dimensions import PermuteDimensions

    candidates: Dict[str, Callable[[SDFG], None]] = {IDENTITY_TAG: lambda sdfg: None}
    for aname in sorted(ext.arrays):
        desc = ext.arrays[aname]
        if desc.transient or len(desc.shape) < 2:
            continue
        ndim = len(desc.shape)
        if ndim > MAX_PERMUTE_NDIM:
            # each candidate here is deepcopy'd, compiled, run and timed -- d! of them wedges a campaign
            raise NotImplementedError(f"default_permutation_candidates: array {aname!r} has rank {ndim} > "
                                      f"{MAX_PERMUTE_NDIM} -- the full-enumeration candidate space explodes; "
                                      f"pass an explicit candidate dict, or prune by stride (task B1) first")
        for perm in itertools.permutations(range(ndim)):
            if list(perm) == list(range(ndim)):
                continue

            def apply(sdfg, aname=aname, perm=perm):
                PermuteDimensions(permute_map={aname: list(perm)}, add_permute_maps=True).apply_pass(sdfg, {})

            candidates[f"permute_{aname}_{''.join(map(str, perm))}"] = apply
    return candidates


@dataclass
class NestEvaluation:
    """One nest's evaluated candidate sweep: externalized baseline, reference outputs, arguments, and ranked results."""
    ext: SDFG
    reference: Dict[str, numpy.ndarray]
    arguments: Dict[str, numpy.ndarray]
    symbols: Dict[str, int]
    results: List[SweepResult] = field(default_factory=list)

    def best(self) -> Optional[SweepResult]:
        """Fastest correct candidate, or None if none verified."""
        return best(self.results)


def call_symbols(ext: SDFG, symbols: Dict[str, int]) -> Dict[str, int]:
    """Subset of `symbols` the externalized nest actually declares (SDFG calls refuse unknown kwargs)."""
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
    """Externalize the nest under `map_entry` and rank its layout candidates by measured time against a reference run."""
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
