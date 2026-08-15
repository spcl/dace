# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Mark every sequential map for unrolling.

``cpu.py`` already emits ``#pragma unroll`` for any :class:`~dace.sdfg.nodes.Map`
whose ``unroll`` property is set (``dace/codegen/targets/cpu.py:2027``), but
nothing in the pipeline sets that flag. This pass closes that gap: it walks the
whole SDFG tree, including maps nested inside GPU kernels (which reach the same
CPU-codegen emission path once GPU transformation lowers their body), and sets
``unroll = True`` on every map whose schedule is exactly ``Sequential``.

Maps with an OpenMP or GPU schedule are left untouched -- ``cpu.py:1985``
raises if an OpenMP map is marked ``unroll``, so this pass must never set the
flag on ``CPU_Multicore``/``CPU_Persistent``/GPU-scheduled maps.
"""
from typing import Any, Optional

from dace import properties
from dace.dtypes import ScheduleType
from dace.sdfg import nodes
from dace.sdfg.sdfg import SDFG
from dace.transformation import pass_pipeline as ppl
from dace.transformation.transformation import explicit_cf_compatible


@properties.make_properties
@explicit_cf_compatible
class UnrollSequentialMaps(ppl.Pass):
    """Set ``Map.unroll = True`` on every ``ScheduleType.Sequential`` map.

    Idempotent: a map already marked ``unroll`` is simply confirmed again, so
    re-running the pass never changes the final SDFG.
    """

    CATEGORY: str = 'Simplification'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Scopes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.Scopes)

    def apply_pass(self, sdfg: SDFG, _: dict[str, Any]) -> Optional[int]:
        """Set ``unroll = True`` on every sequential map in ``sdfg`` and its nested SDFGs.

        :param sdfg: The root SDFG to mutate in place.
        :param _: Pipeline results (unused).
        :return: The number of sequential maps found (and left marked as unrolled), or ``None``
                 if the SDFG tree has none.
        """
        touched = 0
        stack = [sdfg]
        while stack:
            graph = stack.pop()
            for state in graph.all_states():
                for node in state.nodes():
                    if isinstance(node, nodes.NestedSDFG):
                        stack.append(node.sdfg)
                    elif isinstance(node, nodes.MapEntry) and node.map.schedule == ScheduleType.Sequential:
                        node.map.unroll = True
                        touched += 1
        return touched or None
