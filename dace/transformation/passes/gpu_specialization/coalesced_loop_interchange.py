# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU column form: ``for k { map i; if c: map i }`` becomes ``map i { for k { ..; if c: .. } }``, one kernel whose
threads run the loop instead of one launch per map per trip. Taken only when ``i`` is the contiguous axis, so the
threads coalesce: the mirror of the CPU gate, which moves the loop inward when the LOOP axis is the contiguous one.
"""
from typing import Any, Dict, Optional

from dace import SDFG, properties
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.interstate.move_loop_into_map import MoveLoopIntoMap
from dace.transformation.passes.canonicalize.move_loop_into_map_gated import interchange_coalesces


@properties.make_properties
@transformation.explicit_cf_compatible
class CoalescedLoopInterchange(ppl.Pass):
    """Interchange every loop whose maps coalesce once outermost; ``MoveLoopIntoMap(cfg_body)`` judges legality."""

    CATEGORY: str = 'Device Specialization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _: Dict[str, Any]) -> Optional[int]:
        """Interchange until no loop qualifies.

        :param sdfg: The device-neutral SDFG to specialize, in place.
        :returns: The number of loops interchanged, or ``None`` if none.
        """
        xform = MoveLoopIntoMap()
        xform.cfg_body = True
        applied = 0
        changed = True
        while changed:
            changed = False
            # An interchange nests the loop in a new SDFG, so the walk restarts after each one.
            for loop in [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]:
                if not interchange_coalesces(loop, loop.sdfg):
                    continue
                xform.setup_match(loop.sdfg, -1, -1, {MoveLoopIntoMap.loop: loop}, 0, override=True)
                if not xform.can_be_applied(loop.parent_graph, 0, loop.sdfg):
                    continue
                xform.apply(loop.parent_graph, loop.sdfg)
                applied += 1
                changed = True
                break
        return applied or None
