# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Converting copies around nested SDFG arguments whose dtype differs from the container passed in."""

import copy
import warnings
from typing import Any

from dace import data, dtypes, nodes, properties
from dace.libraries.standard.nodes.copy import CopyLibraryNode
from dace.sdfg import SDFG
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation


def retyped(desc: data.Data, dtype: dtypes.typeclass) -> data.Data:
    """A copy of ``desc`` holding ``dtype`` elements."""
    result = copy.deepcopy(desc)
    result.dtype = dtype
    return result


def dtype_only_mismatch(inner: data.Data, outer: data.Data, symbol_mapping: dict[str, Any] | None) -> bool:
    """True if ``inner`` (a nested SDFG connector) and ``outer`` differ only in their plain scalar dtype.

    Pointers, vectors, structs and opaque types are not converted by value, so they never count.
    """
    plain = (data.Scalar, data.Array)
    return (
        type(inner) in plain
        and type(outer) in plain
        and type(inner.dtype) is dtypes.typeclass
        and type(outer.dtype) is dtypes.typeclass
        and inner.dtype != outer.dtype
        and inner.is_equivalent(retyped(outer, inner.dtype), symbol_mapping=symbol_mapping)
    )


@properties.make_properties
@transformation.explicit_cf_compatible
class CastNestedSDFGArguments(ppl.Pass):
    """Passes every nested SDFG argument whose dtype differs from the container through a converting copy.

    The nested SDFG would otherwise reinterpret the container's bytes as its own type. An input is cast into a
    transient of the nested dtype before the call, an output is cast back after it, both by a
    :class:`CopyLibraryNode`; the transient is the container retyped, so the memlets keep their subsets. Each
    rewrite warns.
    """

    CATEGORY: str = "Cleanup"

    __slots__ = ()

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.Descriptors | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        targets = [(n, s) for n, s in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG)]
        count = sum(self.cast_arguments(state, node) for node, state in targets)
        return count or None

    def cast_arguments(self, state: SDFGState, nsdfg: nodes.NestedSDFG) -> int:
        """Rewrite the mismatched arguments of ``nsdfg``; an in-out argument shares one transient.

        :returns: the number of rewritten edges.
        """
        sdfg = state.sdfg
        transients: dict[str, str] = {}
        edges = [(e, True) for e in state.in_edges(nsdfg)] + [(e, False) for e in state.out_edges(nsdfg)]
        count = 0
        for edge, is_input in edges:
            conn = edge.dst_conn if is_input else edge.src_conn
            if edge.data.is_empty() or edge.data.wcr is not None:
                continue
            outer_name = edge.data.data
            outer, inner = sdfg.arrays[outer_name], nsdfg.sdfg.arrays[conn]
            if not dtype_only_mismatch(inner, outer, nsdfg.symbol_mapping):
                continue
            if conn not in transients:
                desc = retyped(outer, inner.dtype)
                desc.transient = True
                desc.lifetime = dtypes.AllocationLifetime.Scope
                # Inside a scope the call is per iteration (per GPU thread), so is its argument
                if state.entry_node(nsdfg) is not None:
                    desc.storage = dtypes.StorageType.Register
                transients[conn] = sdfg.add_datadesc(f"__cast_{conn}", desc, find_new_name=True)
            name = transients[conn]
            warnings.warn(
                f"Nested SDFG '{nsdfg.label}' declares {conn!r} as {inner.dtype} but is passed '{outer_name}' of "
                f"{outer.dtype}; inserted a converting copy through '{name}'."
            )
            self.insert_copy(state, edge, is_input, name)
            count += 1
        return count

    @staticmethod
    def insert_copy(state: SDFGState, edge, is_input: bool, name: str) -> None:
        """Route ``edge`` through ``name`` and a :class:`CopyLibraryNode` on the container's side."""
        src, dst = (edge.data.data, name) if is_input else (name, edge.data.data)
        cpy = CopyLibraryNode(f"copy_{src}_to_{dst}")
        cpy.instrument = state.instrument
        access = state.add_access(name)
        outer_memlet = copy.deepcopy(edge.data)
        cast_memlet = copy.deepcopy(edge.data)
        cast_memlet.data = name
        state.remove_edge(edge)
        if is_input:
            state.add_edge(edge.src, edge.src_conn, cpy, CopyLibraryNode.INPUT_CONNECTOR_NAME, outer_memlet)
            state.add_edge(cpy, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, access, None, copy.deepcopy(cast_memlet))
            state.add_edge(access, None, edge.dst, edge.dst_conn, cast_memlet)
        else:
            state.add_edge(edge.src, edge.src_conn, access, None, cast_memlet)
            state.add_edge(access, None, cpy, CopyLibraryNode.INPUT_CONNECTOR_NAME, copy.deepcopy(cast_memlet))
            state.add_edge(cpy, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, edge.dst, edge.dst_conn, outer_memlet)
