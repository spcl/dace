# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Fold ``X = X + c * T`` onto the contraction library node that writes ``T``: the node writes ``X`` with ``beta=1``.

Contract: the node (``Einsum``, ``MatMul``, ``Gemm``, ``BatchedMatMul`` or ``Gemv``, optionally through one view)
writes the whole of a transient ``T`` with ``beta == 0``; ``T``'s only use, in its only state, is an elementwise map
over all of ``T`` computing ``X[idx] = X[idx] + c * T[idx]`` (either operand order) with identical indices; ``X``
has ``T``'s shape, strides, dtype and storage and is no operand of the node. ``T`` and the map are removed, and
``alpha`` is scaled by ``c``.
"""

from dataclasses import dataclass
from typing import Any

import sympy

from dace import SDFG, data, symbolic
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.canonicalize.loop_to_einsum import _tasklet_expr
from dace.transformation.transformation import explicit_cf_compatible


@dataclass(slots=True)
class AddMap:
    """The elementwise ``X = X + c * T`` map: its outermost scope, the ``X`` access nodes and ``c``."""

    entry: nodes.MapEntry
    read: nodes.AccessNode
    write: nodes.AccessNode
    scale: Any


def foldable_types() -> tuple[type, ...]:
    from dace.libraries.blas import nodes as blas  # Avoid import loop

    return (blas.Einsum, blas.MatMul, blas.Gemm, blas.BatchedMatMul, blas.Gemv)


def scale_of(tasklet: nodes.Tasklet, x_conn: str, t_conn: str) -> Any | None:
    """``c`` when ``tasklet`` computes ``x + c * t`` for a numeric ``c``, else ``None``."""
    expr = _tasklet_expr(tasklet)
    if expr is None:
        return None
    x, t = symbolic.pystr_to_symbolic(x_conn), symbolic.pystr_to_symbolic(t_conn)
    rest = sympy.expand(expr - x)
    scale = sympy.simplify(rest / t)
    if not scale.is_Number or sympy.simplify(rest - scale * t) != 0:
        return None
    return int(scale) if scale.is_Integer else float(scale)


def map_chain_tasklet(state: SDFGState, entry: nodes.MapEntry) -> tuple[list[str], list, nodes.Tasklet] | None:
    """The parameters and ranges of a chain of maps from ``entry``, each holding only the next, down to one tasklet."""
    children = state.scope_children()
    params, ranges = [], []
    while True:
        params += entry.map.params
        ranges += list(entry.map.range)
        level = [n for n in children[entry] if n is not state.exit_node(entry)]
        if len(level) != 1:
            return None
        if isinstance(level[0], nodes.Tasklet):
            return params, ranges, level[0]
        if not isinstance(level[0], nodes.MapEntry):
            return None
        entry = level[0]


def scaled_add(state: SDFGState, tasklet: nodes.Tasklet, temp: str, params: list[str]) -> tuple[Any, Any, Any] | None:
    """The ``x`` read edge, the write edge and ``c`` of a tasklet computing ``x[p] + c * temp[p]`` into ``x[p]``, all
    at the map point ``p``."""
    ins = [e for e in state.in_edges(tasklet) if not e.data.is_empty()]
    outs = [e for e in state.out_edges(tasklet) if not e.data.is_empty()]
    if len(ins) != 2 or len(outs) != 1 or outs[0].data.wcr is not None:
        return None
    t_in = [e for e in ins if e.data.data == temp]
    x_in = [e for e in ins if e.data.data != temp]
    write = outs[0].data
    if len(t_in) != 1 or len(x_in) != 1 or x_in[0].data.data != write.data:
        return None
    for memlet in (write, t_in[0].data, x_in[0].data):
        if [str(b) for b, _, _ in memlet.subset] != params or any(str(b) != str(e) for b, e, _ in memlet.subset):
            return None
    scale = scale_of(tasklet, x_in[0].dst_conn, t_in[0].dst_conn)
    if scale is None:
        return None
    return x_in[0], outs[0], scale


def add_map(state: SDFGState, temp: nodes.AccessNode) -> AddMap | None:
    """The elementwise map ``temp`` feeds, as a chain of maps down to one tasklet over all of ``temp``."""
    outs = state.out_edges(temp)
    if len(outs) != 1 or not isinstance(outs[0].dst, nodes.MapEntry):
        return None
    shape = temp.desc(state.sdfg).shape
    outer = outs[0].dst
    chain = map_chain_tasklet(state, outer)
    if chain is None:
        return None
    params, ranges, tasklet = chain
    if len(params) != len(shape) or any(
        not symbolic.equal_valued(0, b) or not symbolic.equal_valued(1, s) or symbolic.inequal_symbols(e, n - 1)
        for (b, e, s), n in zip(ranges, shape)
    ):
        return None
    added = scaled_add(state, tasklet, temp.data, params)
    if added is None:
        return None
    x_in, out, scale = added
    write = out.data
    read = state.memlet_path(x_in)[0].src
    written = state.memlet_path(out)[-1].dst
    if not isinstance(read, nodes.AccessNode) or not isinstance(written, nodes.AccessNode):
        return None
    if read.data != write.data or written.data != write.data or state.in_degree(read) != 0:
        return None
    if any(e.data.is_empty() for e in state.in_edges(outer)) or any(
        e.data.is_empty() for e in state.out_edges(state.exit_node(outer))
    ):
        return None
    return AddMap(outer, read, written, scale)


def operand_names(state: SDFGState, node: nodes.LibraryNode) -> dict[str, None]:
    """Every container ``node`` reads, a view's viewed array included."""
    names: dict[str, None] = {}
    for edge in state.in_edges(node):
        names[edge.data.data] = None
        if isinstance(edge.src, nodes.AccessNode):
            names[edge.src.data] = None
            names.update((e.data.data, None) for e in state.in_edges(edge.src) if e.data.data is not None)
    return names


def same_layout(a: data.Data, b: data.Data) -> bool:
    return (
        a.dtype == b.dtype
        and a.storage == b.storage
        and len(a.shape) == len(b.shape)
        and not any(symbolic.inequal_symbols(x, y) for x, y in zip(a.shape, b.shape))
        and not any(symbolic.inequal_symbols(x, y) for x, y in zip(a.strides, b.strides))
    )


def whole_temp_output(
    sdfg: SDFG, state: SDFGState, node: nodes.LibraryNode, counts: dict[str, int]
) -> tuple[Any, Any, nodes.AccessNode] | None:
    """The edge into ``T``, the view hop (or ``None``) and ``T``, when ``node``'s one output writes the whole of a
    transient ``T`` used in this state alone, directly or through one view."""
    outs = [e for e in state.out_edges(node) if not e.data.is_empty()]
    if len(outs) != 1 or len(state.out_edges(node)) != 1 or not isinstance(outs[0].dst, nodes.AccessNode):
        return None
    view_edge, temp = None, outs[0].dst
    if isinstance(temp.desc(sdfg), data.View):
        hops = state.out_edges(temp)
        if len(hops) != 1 or state.in_degree(temp) != 1 or not isinstance(hops[0].dst, nodes.AccessNode):
            return None
        view_edge, temp = hops[0], hops[0].dst
    desc = temp.desc(sdfg)
    if not desc.transient or isinstance(desc, data.View) or state.in_degree(temp) != 1 or counts.get(temp.data) != 1:
        return None
    if sum(1 for n in state.data_nodes() if n.data == temp.data) != 1:
        return None
    full = Memlet.from_array(temp.data, desc).subset
    written = view_edge.data if view_edge is not None else outs[0].data
    written_subset = written.subset if written.data == temp.data else written.other_subset
    if written_subset is None or written_subset != full:
        return None
    return (view_edge if view_edge is not None else outs[0]), view_edge, temp


def fold(sdfg: SDFG, state: SDFGState, node: nodes.LibraryNode, counts: dict[str, int]) -> bool:
    """Fold the add map after ``node`` onto it; ``False`` (graph untouched) when the contract does not hold."""
    from dace.libraries.blas import nodes as blas  # Avoid import loop

    if not symbolic.equal_valued(0, node.beta) or isinstance(node, blas.Gemm) and (node.alpha_input or node.beta_input):
        return False
    output = whole_temp_output(sdfg, state, node, counts)
    if output is None:
        return False
    edge, view_edge, temp = output
    desc = temp.desc(sdfg)
    found = add_map(state, temp)
    if found is None:
        return False
    target = found.write.data
    if not same_layout(desc, sdfg.arrays[target]) or target in operand_names(state, node):
        return False
    # Gemm (``cin``) reads the prior output through a connector; the others, Gemv included (its BLAS call takes
    # ``_y`` as one in/out pointer), accumulate in place.
    prior = "_c" if isinstance(node, blas.Gemm) and node.cin else None

    state.remove_nodes_from(state.scope_subgraph(found.entry, include_entry=True, include_exit=True).nodes())
    state.remove_edge(edge)
    memlet = Memlet(data=target, subset=Memlet.from_array(target, sdfg.arrays[target]).subset)
    if view_edge is not None:
        memlet.other_subset = view_edge.data.subset if view_edge.data.data != temp.data else view_edge.data.other_subset
    state.add_edge(edge.src, edge.src_conn, found.write, None, memlet)
    state.remove_node(temp)
    sdfg.remove_data(temp.data, validate=False)
    if found.scale != 1:
        node.alpha = node.alpha * found.scale
    node.beta = 1
    if prior is not None:
        node.add_in_connector(prior, force=True)
        state.add_edge(found.read, None, node, prior, Memlet.from_array(target, sdfg.arrays[target]))
    elif state.degree(found.read) == 0:
        state.remove_node(found.read)
    return True


@explicit_cf_compatible
class BetaFold(ppl.Pass):
    """Fold ``X = X + c * T`` onto the contraction node that writes ``T`` (module docstring)."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Memlets | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.Nodes)

    def apply_pass(self, sdfg: SDFG, _: dict[str, Any]) -> int | None:
        kinds = foldable_types()
        count = 0
        for sd in sdfg.all_sdfgs_recursive():
            counts: dict[str, int] | None = None
            for state in sd.states():
                for node in [n for n in state.nodes() if isinstance(n, kinds)]:
                    if counts is None:
                        counts = {}
                        for st in sd.states():
                            for name in dict.fromkeys(n.data for n in st.data_nodes()):
                                counts[name] = counts.get(name, 0) + 1
                    if fold(sd, state, node, counts):
                        count += 1
        return count or None


__all__ = ["BetaFold"]
