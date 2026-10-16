# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CPU specialization of a parallel map over a ``SMALL`` axis: the ``SMALL`` axes become an inner sequential map.

A symbol declared ``SMALL`` (``nblks``) is a handful, much smaller than every other size. The CPU code generator
shares out the first parameter of a map, so a parallel map over ``(b, i)`` with ``b`` in ``range(nblks)`` gives
``nblks`` threads a block each and leaves the others idle. This pass keeps the axes that are not ``SMALL``, in their
order, in the parallel map and runs the ``SMALL`` ones in an inner ``Sequential`` map: the same iteration space, with
a large axis shared out. A parallel map over ``SMALL`` axes only that holds nothing but a map over larger ones is
collapsed with it first; any other one has no other axis to share out and is left alone.
"""

from typing import Any

from dace import SDFG, dtypes, properties
from dace.sdfg import nodes
from dace.sdfg.state import SymbolResolver
from dace.transformation import pass_pipeline as ppl
from dace.transformation.helpers import isolate_small_map_axes


@properties.make_properties
class SequentializeSmallAxes(ppl.Pass):
    """Move the ``SMALL`` axes of every parallel CPU map into an inner sequential map."""

    CATEGORY: str = "Device Specialization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Scopes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        """Split every ``CPU_Multicore`` map in ``sdfg`` with a ``SMALL`` axis and a larger one.

        :param sdfg: the SDFG to specialize, in place.
        :param _pipeline_results: unused.
        :returns: how many maps were split, or ``None`` if none were; nothing is unless a ``SMALL`` symbol is declared.
        """
        if not any(nested.symbol_repo.small for nested in sdfg.all_sdfgs_recursive()):
            return None
        candidates = [
            (node, state)
            for node, state in sdfg.all_nodes_recursive()
            if isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.CPU_Multicore
        ]
        resolver = SymbolResolver()
        split = sum(isolate_small_map_axes(state, entry, resolver) is not None for entry, state in candidates)
        return split or None
