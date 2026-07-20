# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Cost providers for the global layout assignment: fill an ``AssignmentCosts`` table from either the LogGP cost model or measured per-nest timings. Candidate layouts are dimension permutations (v1-lite)."""
import itertools
import warnings
from typing import Dict, List, Optional

import numpy

import dace
from dace import SDFG
from dace.libraries.layout.algebra import Permute
from dace.transformation.layout.apply_assignment import (IDENTITY_LAYOUT, Layout, reads_before_write, state_touches,
                                                         writes_cover_array)
from dace.transformation.layout.cost_model.loggp import LogGP, gap_from_bandwidth, nest_memory_time
from dace.transformation.layout.cost_model.logp_analysis import count_loop_nest, exposed_concurrency
from dace.transformation.layout.cost_model.relayout import streaming_relayout_time
from dace.transformation.layout.externalize import externalize_nest, nest_entries
from dace.transformation.layout.global_assign import AssignmentCosts
from dace.transformation.layout.line_graph import KernelState
from dace.transformation.layout.nest_eval import evaluate_nest
from dace.transformation.layout.permute_dimensions import PermuteDimensions

#: Illustrative CPU parameters; shared so both providers price relayouts identically.
EXAMPLE_CPU = LogGP(L=95e-9,
                    o=0.0,
                    g=4e-9,
                    G=gap_from_bandwidth(100e9),
                    line_bytes=64,
                    bw_saturated=100e9,
                    bw_core=40e9)

#: Illustrative NVIDIA A100-class parameters (HBM2e ~1.55 TB/s, 128B L2 line, 32B coalescing sector,
#: ~500 ns global-memory latency, ~500 warp-slots of device MLP). Illustrative like EXAMPLE_CPU --
#: replace with a measured microbenchmark fit (fit_message_size + validate) for a real device.
#: ``sector_bytes=32`` is load-bearing: it is the coalescing granularity, so an uncoalesced (strided/
#: transposed) access is counted as one sector PER element. Under the bandwidth regime that small
#: sector makes a strided GPU read CHEAPER (relatively) than on CPU's 64B line; the real coalescing
#: penalty lives in the LATENCY term (messages * L / concurrency) and only engages under FINITE device
#: MLP -- see cost_model_gpu_relayout_test. Feed a finite ``concurrency`` (n_cores * core_mlp), never
#: the parallel-schedule ``inf`` default, when pricing GPU nests.
EXAMPLE_GPU = LogGP(L=500e-9,
                    o=0.0,
                    g=1e-9,
                    G=gap_from_bandwidth(1.555e12),
                    line_bytes=128,
                    sector_bytes=32,
                    bw_saturated=1.555e12,
                    bw_core=25e9)

#: v1 bound on full permutation enumeration (d!); refused loudly above.
MAX_PERMUTE_NDIM = 3


def permutation_tag(perm) -> str:
    return "perm" + "".join(map(str, perm))


def permutation_layouts(ndim: int) -> List[Layout]:
    """Identity FIRST, then every non-identity dimension permutation as a one-op layout."""
    if ndim > MAX_PERMUTE_NDIM:
        raise NotImplementedError(f"permutation_layouts: rank {ndim} > {MAX_PERMUTE_NDIM} -- the "
                                  f"full-enumeration candidate space explodes; stride-driven "
                                  f"pruning (task B1) is required first")
    layouts = [IDENTITY_LAYOUT]
    for perm in itertools.permutations(range(ndim)):
        if list(perm) != list(range(ndim)):
            layouts.append(Layout(permutation_tag(perm), (Permute(tuple(perm)), )))
    return layouts


def assignment_arrays(sdfg: SDFG, kernels: List[KernelState]) -> List[str]:
    """Arrays the assignment optimizes: non-transient, rank >= 2, touched by at least one kernel."""
    touched = set()
    for kernel in kernels:
        touched.update(n.data for n in kernel.state.data_nodes())
    return sorted(name for name in touched if not sdfg.arrays[name].transient and len(sdfg.arrays[name].shape) >= 2)


def liveness_facts(sdfg: SDFG, kernels: List[KernelState], arrays: List[str]):
    """Liveness facts the objective prices, using the same predicates ``apply_assignment`` decides conversions with."""
    entry_needed: Dict[str, bool] = {}
    last_write: Dict[str, Optional[int]] = {}
    for array in arrays:
        touching = [k for k in kernels if state_touches(k.state, array)]
        if touching:
            first = touching[0].state
            entry_needed[array] = reads_before_write(first, array) or not writes_cover_array(first, array)
        lw = max((k.index for k in kernels if any(node.data == array and k.state.in_degree(node) > 0
                                                  for node in k.state.data_nodes())),
                 default=None)
        if lw is not None:
            last_write[array] = lw
    return entry_needed, last_write


def relayout_edge_costs(sdfg: SDFG, arrays: Dict[str, List[Layout]], symbols: Dict[str, int], p: LogGP) -> Dict:
    """Streaming relayout bound per (array, from, to) pair; keyed per pair since measured costs can differ."""
    edges = {}
    for array, layouts in arrays.items():
        seconds = float(
            dace.symbolic.evaluate(streaming_relayout_time(sdfg.arrays[array], p), {
                dace.symbol(s): v
                for s, v in symbols.items()
            }))
        for a, b in itertools.permutations([l.tag for l in layouts], 2):
            edges[(array, a, b)] = seconds
    return edges


def model_costs(sdfg: SDFG,
                kernels: List[KernelState],
                symbols: Dict[str, int],
                p: LogGP = EXAMPLE_CPU) -> AssignmentCosts:
    """Model provider: externalizes each nest, permutes in place, prices with count_loop_nest/nest_memory_time."""
    arrays = assignment_arrays(sdfg, kernels)
    layouts = {a: permutation_layouts(len(sdfg.arrays[a].shape)) for a in arrays}
    subs = {dace.symbol(s): v for s, v in symbols.items()}
    node_cost = {}
    for kernel in kernels:
        touched = {n.data for n in kernel.state.data_nodes()}
        for array in arrays:
            for layout in layouts[array]:
                key = (array, kernel.index, layout.tag)
                if array not in touched:
                    node_cost[key] = 0.0
                    continue
                ext = externalize_nest(kernel.state,
                                       kernel.map_entry,
                                       name=f"score_k{kernel.index}_{array}_{layout.tag}")
                if not layout.is_identity:
                    perm = list(layout.ops[0].perm)
                    PermuteDimensions(permute_map={array: perm}, add_permute_maps=False).apply_pass(ext, {})
                state = next(iter(ext.states()))
                entry = nest_entries(state)[0]
                counts = count_loop_nest(state, entry, line_bytes=p.line_bytes, sector_bytes=p.sector_bytes)
                own = counts.arrays[array]
                messages = float(dace.symbolic.evaluate(own.messages_per_iter * counts.total_iters, subs))
                moved = float(dace.symbolic.evaluate(own.bytes_moved_per_iter * counts.total_iters, subs))
                concurrency = exposed_concurrency(state, entry, p)
                node_cost[key] = float(nest_memory_time(p, moved, messages, concurrency))
    entry_needed, last_write = liveness_facts(sdfg, kernels, arrays)
    return AssignmentCosts(layouts=layouts,
                           node_cost=node_cost,
                           relayout_cost=relayout_edge_costs(sdfg, layouts, symbols, p),
                           entry_conversion_needed=entry_needed,
                           last_write_kernel=last_write)


def eval_costs(sdfg: SDFG,
               kernels: List[KernelState],
               symbols: Dict[str, int],
               provided: Optional[Dict[str, numpy.ndarray]] = None,
               p: LogGP = EXAMPLE_CPU,
               reps: int = 10,
               warmup: int = 2,
               seed: int = 0) -> AssignmentCosts:
    """Eval provider: runs evaluate_nest per kernel, tables measured medians; a failed candidate is a hard error."""
    arrays = assignment_arrays(sdfg, kernels)
    layouts = {a: permutation_layouts(len(sdfg.arrays[a].shape)) for a in arrays}
    node_cost = {}
    untrusted = set()
    for kernel in kernels:
        evaluation = evaluate_nest(kernel.state,
                                   kernel.map_entry,
                                   symbols=symbols,
                                   provided=provided,
                                   reps=reps,
                                   warmup=warmup,
                                   seed=seed,
                                   name=f"evalcost_k{kernel.index}")
        by_name = {r.name: r for r in evaluation.results}
        bad = [r.name for r in evaluation.results if not r.correct or r.time is None]
        if bad:
            raise RuntimeError(f"eval_costs: kernel {kernel.index} candidates failed to "
                               f"verify/time: {bad} -- refusing to fill the table with guesses")
        contended = sorted(r.name for r in evaluation.results if r.metadata.get("contended", False))
        if contended:
            warnings.warn(f"eval_costs: kernel {kernel.index} candidates {contended} measured "
                          f"CONTENDED (spread above threshold); medians kept but marked untrusted "
                          f"in the table -- decisions consuming them are flagged in the conflict "
                          f"report")
        identity_seconds = by_name["identity"].time * 1e-3
        touched = {n.data for n in kernel.state.data_nodes()}
        for array in arrays:
            for layout in layouts[array]:
                key = (array, kernel.index, layout.tag)
                if array not in touched:
                    node_cost[key] = 0.0
                    continue
                if layout.is_identity:
                    candidate_name = "identity"
                else:
                    digits = "".join(map(str, layout.ops[0].perm))
                    candidate_name = f"permute_{array}_{digits}"
                node_cost[key] = by_name[candidate_name].time * 1e-3
                if by_name[candidate_name].metadata.get("contended", False):
                    untrusted.add(key)
    entry_needed, last_write = liveness_facts(sdfg, kernels, arrays)
    return AssignmentCosts(layouts=layouts,
                           node_cost=node_cost,
                           relayout_cost=relayout_edge_costs(sdfg, layouts, symbols, p),
                           entry_conversion_needed=entry_needed,
                           last_write_kernel=last_write,
                           untrusted=untrusted)
