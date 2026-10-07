# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Turn every top-level loop nest of the root SDFG into an ``ExternalCall`` library node.

``OutlineTopLevelNests`` first wraps each top-level map nest and loop region in a ``no_inline`` nested SDFG;
each of those becomes an ``ExternalCall`` whose default ``DaceReference`` expansion is the nest itself, so the
program is unchanged until a caller points a node at a compiled library and selects ``ExternCall``. A nest
whose symbol mapping is not the identity stays a nested SDFG: an expansion re-binds symbols by name.
"""

import copy
from typing import Any, Dict, List, Optional

from dace import SDFG
from dace.libraries.standard.nodes import external_call
from dace.properties import make_properties
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import passes


def identity_mapping(nsdfg: nodes.NestedSDFG) -> bool:
    return all(str(value) == name for name, value in nsdfg.symbol_mapping.items())


def reference_sdfg(nsdfg: nodes.NestedSDFG) -> SDFG:
    """A detached copy of the nest with each boundary container renamed to its connector; a container both read
    and written keeps its body on ``_out_`` and gets an ``_in_`` twin descriptor."""
    ref = copy.deepcopy(nsdfg.sdfg)
    ref.parent = None
    ref.parent_sdfg = None
    ref.parent_nsdfg_node = None
    ref.reset_cfg_list()
    inputs, outputs = list(nsdfg.in_connectors), list(nsdfg.out_connectors)
    for name in inputs:
        if name not in outputs:
            ref.replace(name, external_call.in_conn(name))
    for name in outputs:
        ref.replace(name, external_call.out_conn(name))
    for name in sorted(n for n in inputs if n in outputs):
        ref.add_datadesc(external_call.in_conn(name), copy.deepcopy(ref.arrays[external_call.out_conn(name)]))
    return ref


def replace_with_external_call(state: SDFGState, nsdfg: nodes.NestedSDFG, name: str) -> external_call.ExternalCall:
    """An ``ExternalCall`` in place of ``nsdfg``, calling ``name`` in DaCe's nested-SDFG argument order."""
    node = external_call.ExternalCall(
        name,
        inputs=[external_call.in_conn(i) for i in nsdfg.in_connectors],
        outputs=[external_call.out_conn(o) for o in nsdfg.out_connectors],
        standalone_sdfg=reference_sdfg(nsdfg),
    )
    state.add_node(node)
    for edge in state.in_edges(nsdfg):
        conn = None if edge.dst_conn is None else external_call.in_conn(edge.dst_conn)
        state.add_edge(edge.src, edge.src_conn, node, conn, copy.deepcopy(edge.data))
    for edge in state.out_edges(nsdfg):
        conn = None if edge.src_conn is None else external_call.out_conn(edge.src_conn)
        state.add_edge(node, conn, edge.dst, edge.dst_conn, copy.deepcopy(edge.data))
    symbols = [str(s) for s in nsdfg.sdfg.free_symbols]
    state.remove_node(nsdfg)
    node.symbol = name
    node.abi_order = external_call.nested_sdfg_order(node, state, symbols)
    node.signature = external_call.derive_signature(node, state)
    return node


@make_properties
class OutlineToExternalCalls(ppl.Pass):
    """Replace each top-level loop nest of the root SDFG with an ``ExternalCall``. See the module docstring."""

    CATEGORY: str = "Optimization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.States | ppl.Modifies.Memlets | ppl.Modifies.Symbols

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _: Dict[str, Any]) -> Optional[List[external_call.ExternalCall]]:
        """Outline and replace; returns the new nodes, or ``None`` if there were none."""
        passes.outline_top_level_nests(sdfg)
        created: List[external_call.ExternalCall] = []
        for state in [block for block in sdfg.nodes() if isinstance(block, SDFGState)]:
            nests = [n for n in state.nodes() if isinstance(n, nodes.NestedSDFG) and n.no_inline]
            for nsdfg in nests:
                if identity_mapping(nsdfg):
                    created.append(replace_with_external_call(state, nsdfg, nsdfg.unique_name or nsdfg.label))
        return created or None


def outline_to_external_calls(sdfg: SDFG) -> List[external_call.ExternalCall]:
    """Replace the root SDFG's top-level loop nests with ``ExternalCall`` nodes in place; returns them."""
    return OutlineToExternalCalls().apply_pass(sdfg, {}) or []
