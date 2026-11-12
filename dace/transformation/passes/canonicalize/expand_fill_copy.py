# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Fill and copy library nodes as the maps they stand for, so fusion joins them with their producers and consumers.

The one exception to canonicalization keeping library calls: a fill or a copy is an element-wise map, and as a
library node it keeps the maps around it apart (a ``c *= beta`` map, its copy into a scratch array and the scratch
array's reader stay three loops). Each node expands to the map its ``Auto`` selection takes for a large transfer
outside a parallel scope. Three kinds stay library calls: a node inside a parallel map, where that selection takes
the single call; a copy across storages (the map form cannot cross the host/device boundary); and a copy that
changes rank, whose map walks one flattened index that no neighbouring map shares.
"""

from typing import Any

from dace import SDFG
from dace.libraries.standard.helper import collapse_to_elements, is_in_parallel_scope
from dace.libraries.standard.nodes.copy.node import CopyLibraryNode
from dace.libraries.standard.nodes.fill.node import FillLibraryNode
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.cpu_specialization.hoist_parallel_region import expand_in_place

#: The map expansion of each node.
MAP_EXPANSION = {FillLibraryNode: "pure", CopyLibraryNode: "MappedTasklet"}


def expands_to_map(node: Any, state: SDFGState) -> bool:
    """Whether ``node`` is a fill, or a copy within one storage and of one rank, outside any parallel map."""
    if type(node) not in MAP_EXPANSION or is_in_parallel_scope(node, state):
        return False
    if isinstance(node, FillLibraryNode):
        return True
    if node.src_storage(state) != node.dst_storage(state):
        return False
    _, inp, in_subset, _, out, out_subset = node.validate(state.sdfg, state)
    return len(collapse_to_elements(in_subset, inp.strides)[0]) == len(collapse_to_elements(out_subset, out.strides)[0])


@transformation.explicit_cf_compatible
class ExpandFillCopy(ppl.Pass):
    """Expand every fill and copy library node :func:`expands_to_map` admits to its map and inline it."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.Nodes

    def apply_pass(self, sdfg: SDFG, _: dict[str, Any]) -> int | None:
        """:returns: How many nodes were expanded, or ``None`` if none was."""
        found = [(n, s) for n, s in sdfg.all_nodes_recursive() if expands_to_map(n, s)]
        for node, state in found:
            expand_in_place(node, state, MAP_EXPANSION[type(node)])
        return len(found) or None
