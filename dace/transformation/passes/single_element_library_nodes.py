# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A library node whose every operand is one element is a tasklet: its class says which (``scalar_code``)."""

from typing import Any

from dace import SDFG, SDFGState, properties
from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES
from dace.sdfg import nodes
from dace.sdfg.scope import is_devicelevel_gpu
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation


def single_element(state: SDFGState, node: nodes.LibraryNode) -> bool:
    """Every data edge of ``node`` moves exactly one element, statically."""
    edges = [e for e in state.all_edges(node) if not e.data.is_empty()]
    return bool(edges) and all(
        not e.data.dynamic and e.data.subset is not None and e.data.subset.num_elements_exact() == 1 for e in edges
    )


def as_tasklet(state: SDFGState, node: nodes.LibraryNode) -> nodes.Tasklet | None:
    """Replace a single-element ``node`` by the tasklet its class gives (an input connector also named as an output
    is ``<name>_in``), rewiring its edges; ``None``, and nothing changed, when it has none."""
    if not single_element(state, node):
        return None
    sdfg = state.sdfg
    names = {e.data.data for e in state.all_edges(node) if not e.data.is_empty()}
    if not is_devicelevel_gpu(sdfg, state, node) and any(
        sdfg.arrays[n].storage in GPU_RESIDENT_STORAGES for n in names
    ):
        return None
    code = node.scalar_code(state)
    if code is None:
        return None
    reads = {e.dst_conn for e in state.in_edges(node) if e.dst_conn is not None}
    writes = {e.src_conn for e in state.out_edges(node) if e.src_conn is not None}
    renamed = {c: f"{c}_in" if c in writes else c for c in reads}
    tasklet = state.add_tasklet(
        node.label,
        {renamed[c]: node.in_connectors[c] for c in sorted(reads)},
        {c: node.out_connectors[c] for c in sorted(writes)},
        code,
        debuginfo=node.debuginfo,
    )
    for e in state.in_edges(node):
        state.add_edge(e.src, e.src_conn, tasklet, renamed.get(e.dst_conn), e.data)
    for e in state.out_edges(node):
        state.add_edge(tasklet, e.src_conn, e.dst, e.dst_conn, e.data)
    state.remove_node(node)
    return tasklet


@properties.make_properties
@transformation.explicit_cf_compatible
class SingleElementLibraryNodes(ppl.Pass):
    """Every library node whose operands are all one element becomes the tasklet it computes (:func:`as_tasklet`)."""

    CATEGORY: str = "Simplification"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & (ppl.Modifies.Nodes | ppl.Modifies.Memlets))

    def apply_pass(self, sdfg: SDFG, _: dict[str, Any]) -> int | None:
        replaced = 0
        for node, state in list(sdfg.all_nodes_recursive()):
            if isinstance(node, nodes.LibraryNode) and as_tasklet(state, node) is not None:
                replaced += 1
        return replaced or None
