# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CPU specialization of loop-invariant code motion: split a map nest where the outer map alone feeds the team.

The canonical form keeps ``map i { map j { t } }`` whole when ``t`` is invariant in ``j`` only, since a GPU runs
the nest as one multi-dimensional kernel. On the CPU only the outermost parallel map opens an OpenMP team, so a
tasklet invariant in ``j`` can be computed once per ``i`` instead of once per ``(i, j)`` -- provided ``map i``
has enough iterations to keep every thread busy on its own (:func:`few_trips` decides it, the same cost model
the small-axis sequentialization uses).
"""

from dace import properties
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState, SymbolResolver
from dace.transformation import transformation as xf
from dace.transformation.passes.cpu_specialization.decide_cpu_parallel_axes import CPU_PARALLEL_SCHEDULES, few_trips
from dace.transformation.passes.loop_invariant_code_motion import LoopInvariantCodeMotion


@properties.make_properties
@xf.explicit_cf_compatible
class SplitInvariantMapNests(LoopInvariantCodeMotion):
    """Loop-invariant code motion that may hoist a tasklet between a CPU parallel map and the map nested in it."""

    CATEGORY: str = "CPU Specialization"

    def may_split_nest(self, state: SDFGState, inner: nodes.MapEntry) -> bool:
        outer = state.entry_node(inner)
        if not isinstance(outer, nodes.MapEntry) or outer.map.schedule not in CPU_PARALLEL_SCHEDULES:
            return False
        return not few_trips(outer.map.range.num_elements(), lambda: SymbolResolver().facts_at(state, outer))
