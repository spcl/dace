# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Promotes transients that only ever store literals to SDFG constants."""

import ast
import copy
from typing import Any

import numpy as np

from dace import SDFG, Memlet, SDFGState, dtypes, properties, subsets
from dace import data as dt
from dace.frontend.python import astutils
from dace.sdfg import nodes as nd
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import MultiConnectorEdge
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation

Write = tuple[SDFGState, MultiConnectorEdge]


@properties.make_properties
@transformation.explicit_cf_compatible
class PromoteConstantTransients(ppl.Pass):
    """Turns a host transient into an SDFG constant (emitted as ``constexpr``) when every write stores a literal
    to a constant subset and no two writes overlap. A write is a data-free tasklet or a map filling one literal.
    Reading an element before its only write is undefined, so the literal is a valid value for every read."""

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, top_sdfg: SDFG, pipeline_results: dict[str, Any]) -> dict[int, set[str]] | None:
        """:return: ``{cfg_id: promoted names}``, or ``None`` if nothing was promoted."""
        result: dict[int, set[str]] = {}
        for sdfg in top_sdfg.all_sdfgs_recursive():
            writes: dict[str, list[Write]] = {}
            skip = symbolic_reads(sdfg)
            for state in sdfg.states():
                for node in state.data_nodes():
                    writes.setdefault(node.data, []).extend(
                        (state, e) for e in state.in_edges(node) if not e.data.is_empty()
                    )
                    # A reference to the data writes it through another name.
                    if any(e.dst_conn == "set" for e in state.out_edges(node)):
                        skip.add(node.data)
            for name, desc in list(sdfg.arrays.items()):
                if name in skip or name in sdfg.constants_prop or not is_candidate(desc):
                    continue
                value = constant_value(desc, writes.get(name, []))
                if value is None:
                    continue
                sdfg.add_constant(name, value, copy.deepcopy(desc))
                for state, edge in writes[name]:
                    remove_write(state, edge)
                result.setdefault(sdfg.cfg_id, set()).add(name)
        return result or None


def is_candidate(desc: dt.Data) -> bool:
    """A scope-lifetime host transient scalar or array of constant shape."""
    return (
        type(desc) in (dt.Scalar, dt.Array)
        and desc.transient
        and desc.lifetime == dtypes.AllocationLifetime.Scope
        and dtypes.can_access(dtypes.ScheduleType.CPU_Multicore, desc.storage)
        and all(to_int(d) is not None for d in desc.shape)
    )


def to_int(dim) -> int | None:
    try:
        return int(dim)
    except (TypeError, ValueError):
        return None


def symbolic_reads(sdfg: SDFG) -> set[str]:
    """Data names read by interstate edges or control-flow conditions, which access nodes do not show."""
    names = set(sdfg.arrays.keys())
    refs = set()
    for edge in sdfg.all_interstate_edges():
        refs |= (edge.data.free_symbols | edge.data.read_symbols()) & names
    for cfr in sdfg.all_control_flow_regions():
        refs |= cfr.used_symbols(all_symbols=True, with_contents=False) & names
    return refs


def literal_written(state: SDFGState, edge: MultiConnectorEdge) -> Any | None:
    """The literal a write edge stores: from a data-free tasklet, or through a map exit holding only one."""
    if edge.data.wcr is not None:
        return None
    src, conn = edge.src, edge.src_conn
    if isinstance(src, nd.MapExit):
        if state.out_degree(src) != 1 or not conn or not conn.startswith("OUT_"):
            return None
        inner = [e for e in state.in_edges(src) if e.dst_conn == "IN_" + conn[4:]]
        body = state.scope_subgraph(state.entry_node(src), include_entry=False, include_exit=False).nodes()
        if len(inner) != 1 or inner[0].data.wcr is not None or list(body) != [inner[0].src]:
            return None
        src, conn = inner[0].src, inner[0].src_conn
    if not isinstance(src, nd.Tasklet) or src.language != dtypes.Language.Python or state.out_degree(src) != 1:
        return None
    if any(not e.data.is_empty() for e in state.in_edges(src)):
        return None
    stmts = src.code.code
    if len(stmts) != 1 or not isinstance(stmts[0], ast.Assign) or len(stmts[0].targets) != 1:
        return None
    if astutils.rname(stmts[0].targets[0]) != conn:
        return None
    try:
        value = astutils.evalnode(stmts[0].value, {})
    except SyntaxError:
        return None
    return value if isinstance(value, (bool, int, float, complex)) else None


def constant_value(desc: dt.Data, writes: list[Write]) -> Any | None:
    """The initializer if every write stores a literal to a disjoint constant subset, else ``None``."""
    # Only a plain scalar type has a numpy value (not e.g. an opaque ``MPI_Request``).
    if not writes or type(desc.dtype) is not dtypes.typeclass:
        return None
    shape = tuple(int(d) for d in desc.shape)
    array = np.zeros(shape, dtype=desc.dtype.as_numpy_dtype())
    touched = np.zeros(shape, dtype=bool)
    for state, edge in writes:
        value = literal_written(state, edge)
        subset = edge.data.get_dst_subset(edge, state)
        if value is None or not isinstance(subset, subsets.Range) or len(subset) != len(shape):
            return None
        index = []
        # Memlets index with the descriptor offset applied (Fortran arrays carry -1 per dimension).
        for dim, (begin, end, step) in enumerate(subset.offset_new(desc.offset, False)):
            begin, end, step = to_int(begin), to_int(end), to_int(step)
            if begin is None or end is None or step is None or begin < 0 or end >= shape[dim]:
                return None
            index.append(slice(begin, end + 1, step))
        if touched[tuple(index)].any():
            return None
        touched[tuple(index)] = True
        array[tuple(index)] = value
    return desc.dtype.type(array.flat[0]) if isinstance(desc, dt.Scalar) else array


def remove_write(state: SDFGState, edge: MultiConnectorEdge):
    """Removes a promoted write and its producer. The access node keeps the producer's ordering anchors."""
    node = edge.dst
    producer = edge.src
    if isinstance(producer, nd.MapExit):
        entry = state.entry_node(producer)
        body = [entry, producer] + list(state.scope_subgraph(entry, include_entry=False, include_exit=False).nodes())
        anchors = [e.src for e in state.in_edges(entry)]
    else:
        body = [producer]
        anchors = [e.src for e in state.in_edges(producer)]
    sdutil.remove_edge_and_dangling_path(state, edge)
    state.remove_nodes_from([n for n in body if n in state.nodes()])
    if node not in state.nodes():
        return
    if state.degree(node) == 0:
        state.remove_node(node)
    elif state.in_degree(node) == 0:
        for anchor in anchors:
            if anchor in state.nodes():
                state.add_nedge(anchor, node, Memlet())
