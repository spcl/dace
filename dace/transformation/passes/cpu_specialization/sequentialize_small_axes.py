# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CPU specialization of a parallel map whose first axis is ``SMALL``: the leading ``SMALL`` axes are shared out
together with the first larger one.

A symbol declared ``SMALL`` (``nblks``) or a literal extent below ``SMALL_CONSTANT_EXTENT`` is a handful, much
smaller than every other size. The CPU code generator shares out the first parameter of a map, so a parallel map over
``(b, k, i)`` with ``b`` in ``range(nblks)`` gives ``nblks`` threads a block each and leaves the others idle. This
pass collapses the leading ``SMALL`` axes with the first larger one (``collapse(2)`` over ``b, k``), keeping the order
of every axis: the stride order, contiguous axis innermost, survives, in one parallel region
(:func:`~dace.transformation.helpers.share_out_past_leading_small_axes`).
"""

from typing import Any

from dace import SDFG, dtypes, properties
from dace.sdfg import nodes
from dace.sdfg.state import SymbolResolver
from dace.transformation import pass_pipeline as ppl
from dace.transformation.helpers import share_out_past_leading_small_axes


@properties.make_properties
class SequentializeSmallAxes(ppl.Pass):
    """Share out the leading ``SMALL`` axes of every parallel CPU map together with the first larger one."""

    CATEGORY: str = "Device Specialization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Scopes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        """Split every ``CPU_Multicore`` map in ``sdfg`` whose first axis is ``SMALL`` and that has a larger one.

        :param sdfg: the SDFG to specialize, in place.
        :param _pipeline_results: unused.
        :returns: how many maps were split, or ``None`` if none were.
        """
        candidates = [
            (node, state)
            for node, state in sdfg.all_nodes_recursive()
            if isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.CPU_Multicore
        ]
        resolver = SymbolResolver()
        split = sum(
            share_out_past_leading_small_axes(state, entry, resolver) is not None for entry, state in candidates
        )
        return split or None
