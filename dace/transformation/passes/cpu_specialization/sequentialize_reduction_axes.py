# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CPU specialization of a parallel map that also REDUCES: the reduced axes become an inner sequential loop.

Canonicalization leaves ``for j: for i: a[i] += b[i - j] * c[j]`` (tsvc s176) as ONE parallel map over
``(j, i)`` whose write to ``a[i]`` resolves conflicts over ``j``. The CPU code generator shares out only
the first parameter of a map, so OpenMP split ``j`` across the threads, every thread wrote every ``a[i]``,
and each of the 3.5e9 updates was an ``omp atomic`` -- 7.1 s against numba's 1.0 s on 16 cores.

The parameters that index the reduced container are the ones whose iterations write DIFFERENT elements,
so they are the ones to share out. This pass moves them to the front of the map and splits the rest off
into an inner ``Sequential`` map, which is the same iteration space in an order the code generator proves
conflict-free: the write leaves the outer map indexed by its own parameter, so no atomic is emitted.
"""
from typing import Any, Dict, List, Optional, Set

from dace import SDFG, dtypes, properties
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow import MapDimShuffle, MapExpansion


def reduction_output_params(state: SDFGState, entry: nodes.MapEntry) -> Optional[Set[str]]:
    """The parameters of ``entry`` that index its write-conflict-resolved outputs.

    ``None`` when there is nothing to split: no such output, outputs indexed by DIFFERENT parameters
    (no one order serves them all), or every parameter indexes them (no axis is reduced) or none does
    (a reduction to one element, which OpenMP's ``reduction`` clause already serves).
    """
    params = set(entry.map.params)
    indexing: Optional[Set[str]] = None
    for edge in state.in_edges(state.exit_node(entry)):
        if edge.data.is_empty() or edge.data.wcr is None:
            continue
        used = {str(symbol) for symbol in edge.data.subset.free_symbols} & params
        if indexing is not None and used != indexing:
            return None
        indexing = used
    if not indexing or indexing == params:
        return None
    return indexing


@properties.make_properties
class SequentializeReductionAxes(ppl.Pass):
    """Split a parallel CPU map into an outer parallel map over the axes its reduced outputs are indexed by,
    and an inner sequential map over the axes it reduces."""

    CATEGORY: str = 'Device Specialization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Scopes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Split every eligible ``CPU_Multicore`` map in ``sdfg``.

        :param sdfg: the SDFG to specialize, in place.
        :param _pipeline_results: unused.
        :returns: how many maps were split, or ``None`` if none were.
        """
        candidates: List[tuple] = []
        for node, state in sdfg.all_nodes_recursive():
            if (not isinstance(node, nodes.MapEntry) or node.map.schedule != dtypes.ScheduleType.CPU_Multicore
                    or node.map.collapse > 1):
                continue
            indexing = reduction_output_params(state, node)
            if indexing is not None:
                candidates.append((node, state, indexing))
        for entry, state, indexing in candidates:
            order = ([p
                      for p in entry.map.params if p in indexing] + [p for p in entry.map.params if p not in indexing])
            if order != entry.map.params:
                MapDimShuffle.apply_to(state.sdfg, map_entry=entry, options={'parameters': order}, save=False)
            MapExpansion.apply_to(state.sdfg,
                                  map_entry=entry,
                                  options={
                                      'inner_schedule': dtypes.ScheduleType.Sequential,
                                      'expansion_limit': 1
                                  },
                                  save=False)
        return len(candidates) or None
