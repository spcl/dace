# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared plumbing for the symmetric rank-k update nodes ``Syrk`` and ``Syr2k``.

The two nodes are close siblings -- same ``uplo`` / ``trans`` / ``alpha`` / ``beta``
properties, same triangular output semantics, same runtime-coefficient wiring -- and
differ only in operand count (``Syrk`` reads ``A``; ``Syr2k`` reads ``A`` and ``B``)
and in the vendor routine they call. Everything that does not depend on that
difference lives here, so the two node modules stay free of copy-paste.
"""
from copy import deepcopy as dc
from typing import Callable, Dict, List, Optional, Tuple

import dace.library
import dace.sdfg.nodes
from dace import SDFG, SDFGState, data as dt, dtypes, memlet as mm
from dace.symbolic import symstr

# Connector names of the runtime coefficient inputs.
COEFF_CONNECTORS = ("_alpha", "_beta")


def operand_info(node, state: SDFGState, sdfg: SDFG, connectors: Tuple[str,
                                                                       ...]) -> Dict[str, Tuple[dt.Data, List, List]]:
    """Resolve ``(descriptor, shape, strides)`` per input connector in ``connectors``
    plus the ``_c`` output, from the connector memlets. The shape is the memlet
    subset's size (what the node actually operates on); the strides are the
    descriptor's (what the BLAS leading dimension must follow)."""
    info: Dict[str, Tuple[dt.Data, List, List]] = {}
    for conn in connectors:
        edge = next((e for e in state.in_edges(node) if e.dst_conn == conn), None)
        if edge is None:
            raise ValueError(f"{node.name}: expected a '{conn}' input")
        desc = sdfg.arrays[edge.data.data]
        info[conn] = (desc, edge.data.subset.size(), desc.strides)
    out = next((e for e in state.out_edges(node) if e.src_conn == "_c"), None)
    if out is None:
        raise ValueError(f"{node.name}: expected a '_c' output")
    cdesc = sdfg.arrays[out.data.data]
    info["_c"] = (cdesc, out.data.subset.size(), cdesc.strides)
    return info


def scalar_conn_descs(node, state: SDFGState, sdfg: SDFG) -> Dict[str, dt.Data]:
    """Descriptors of the runtime coefficient connectors (``_alpha`` / ``_beta``)
    that are actually wired, keyed by connector name."""
    return {e.dst_conn: sdfg.arrays[e.data.data] for e in state.in_edges(node) if e.dst_conn in COEFF_CONNECTORS}


def render_scalar(value, dtype: dtypes.typeclass) -> str:
    """Render a scalar constant at ``dtype`` for a (Python-language) tasklet."""
    return f"dace.{dtype.to_string()}({value})"


def scalar_ctype(value, dtype: dtypes.typeclass) -> str:
    return f"{dtype.ctype}({value})"


def coeff_decl(var: str, prop, dtype: dtypes.typeclass, scalar: Optional[str]) -> str:
    """C declaration of an effective coefficient: the symbolic property value, times
    the runtime scalar connector ``scalar`` when one is wired (the two compose,
    mirroring the pure path). A single-element connector reaches the tasklet by
    value, so it is used directly (no dereference)."""
    if scalar is not None:
        return f"{dtype.ctype} {var} = {scalar_ctype(prop, dtype)} * {scalar};"
    return f"{dtype.ctype} {var} = {scalar_ctype(prop, dtype)};"


def triangle_range(uplo: str, row: str, n) -> str:
    """The column range of row ``row`` within the ``uplo`` triangle of an ``n x n``
    matrix -- ``0:row+1`` (lower, diagonal included) or ``row:n`` (upper)."""
    return f"0:{row} + 1" if uplo == "L" else f"{row}:{symstr(n)}"


def blas_inplace(node, state: SDFGState, sdfg: SDFG, operands: Tuple[str, ...], code_fn: Callable):
    """Build the vendor-BLAS node. ``code_fn(ptrs, pa, pb)`` renders the BLAS call,
    where ``ptrs`` maps each operand connector (and ``_c``) to its pointer name and
    ``pa`` / ``pb`` are the runtime ``_alpha`` / ``_beta`` scalar names (``None`` when
    that coefficient is a compile-time property).

    ``xSYRK`` / ``xSYR2K`` read and write ``C`` through one pointer, but a CPP tasklet
    with the same name as both an in- and an out-connector redeclares that pointer.
    When ``C`` is read (``beta != 0``) or a runtime scalar is wired, the tasklet is
    wrapped in a nested SDFG whose incoming arrays reach it through prefixed
    connectors (``__cin``, ``__alpha_in``, ``__beta_in``) that never collide with the
    nested array names; the plain ``beta == 0`` no-scalar case needs no wrapper, so a
    bare tasklet suffices. Mirrors ``symm``'s equivalent helper.
    """
    reads_c = "_c" in node.in_connectors
    scalars = scalar_conn_descs(node, state, sdfg)
    if not reads_c and not scalars:
        ptrs = {conn: conn for conn in operands}
        ptrs["_c"] = "_c"
        return dace.sdfg.nodes.Tasklet(node.name,
                                       set(operands), {"_c"},
                                       code_fn(ptrs, None, None),
                                       language=dtypes.Language.CPP)

    info = operand_info(node, state, sdfg, operands)
    nsdfg = SDFG(node.label + "_inplace")
    for conn in operands + ("_c", ):
        d = dc(info[conn][0])
        d.transient = False
        nsdfg.add_datadesc(conn, d)
    for conn, desc in scalars.items():
        d = dc(desc)
        d.transient = False
        nsdfg.add_datadesc(conn, d)

    inner = {"_alpha": "__alpha_in", "_beta": "__beta_in"}
    pa = inner["_alpha"] if "_alpha" in scalars else None
    pb = inner["_beta"] if "_beta" in scalars else None
    # Inner tasklet connector per operand: ``_a`` -> ``__a``, which never collides with
    # the nested array names (they keep the outer ``_a`` / ``_c`` spelling).
    tconn = {conn: "__" + conn.lstrip("_") for conn in operands}
    in_conns = set(tconn.values()) | ({"__cin"} if reads_c else set()) | {inner[c] for c in scalars}

    ptrs = dict(tconn)
    ptrs["_c"] = "__c"
    st = nsdfg.add_state(node.label + "_state")
    t = st.add_tasklet(node.name, in_conns, {"__c"}, code_fn(ptrs, pa, pb), language=dtypes.Language.CPP)
    for conn in operands:
        st.add_edge(st.add_read(conn), None, t, tconn[conn], mm.Memlet.from_array(conn, nsdfg.arrays[conn]))
    if reads_c:
        st.add_edge(st.add_read("_c"), None, t, "__cin", mm.Memlet.from_array("_c", nsdfg.arrays["_c"]))
    for conn in scalars:
        st.add_edge(st.add_read(conn), None, t, inner[conn], mm.Memlet.from_array(conn, nsdfg.arrays[conn]))
    st.add_edge(t, "__c", st.add_write("_c"), None, mm.Memlet.from_array("_c", nsdfg.arrays["_c"]))
    return nsdfg


def add_coeff_arrays(nsdfg: SDFG, scalars: Dict[str, dt.Data], dtype: dtypes.typeclass) -> None:
    """Add a ``[1]`` input array per wired runtime coefficient connector.

    Feeding the coefficient as a tasklet input -- rather than binding a symbol from
    ``_alpha[0]`` on an interstate edge -- keeps it a proper array read (a ``[1]``
    connector reaches the nested SDFG as a scalar reference, which ``[0]`` cannot
    subscript)."""
    for conn, desc in scalars.items():
        nsdfg.add_array(conn, [1], dtype, storage=desc.storage)


def add_triangular_tasklet(state: SDFGState,
                           uplo: str,
                           n,
                           label: str,
                           inputs: Dict[str, mm.Memlet],
                           code: str,
                           outputs: Dict[str, mm.Memlet],
                           extra_map: Optional[Tuple[str, str]] = None) -> None:
    """Emit ``code`` over the ``uplo`` triangle of an ``n x n`` matrix as nested maps
    ``__i`` (row) then ``__j`` (that row's triangular column range), optionally with a
    third, innermost map ``extra_map = (param, range)`` (the contraction axis).

    The triangle is walked as NESTED maps rather than one 2-D map so the
    data-dependent inner bound (``0:__i + 1``) never lands on a collapsed parallel
    map. Every memlet is deep-copied per edge: DaCe rejects two edges sharing one
    subset instance.
    """
    row_me, row_mx = state.add_map(label + "_row", {"__i": f"0:{symstr(n)}"})
    col_me, col_mx = state.add_map(label + "_col", {"__j": triangle_range(uplo, "__i", n)})
    entries, exits = [row_me, col_me], [col_mx, row_mx]
    if extra_map is not None:
        red_me, red_mx = state.add_map(label + "_" + extra_map[0].lstrip("_"), {extra_map[0]: extra_map[1]})
        entries.append(red_me)
        exits.insert(0, red_mx)

    tasklet = state.add_tasklet(label, set(inputs), set(outputs), code)
    for conn, memlet in inputs.items():
        state.add_memlet_path(state.add_read(memlet.data), *entries, tasklet, dst_conn=conn, memlet=dc(memlet))
    if not inputs:
        # A tasklet with no inputs (``beta == 0`` zero-fill) still needs the scope
        # chained together, so connect the entries and the tasklet with empty memlets.
        prev = entries[0]
        for entry in entries[1:]:
            state.add_nedge(prev, entry, mm.Memlet())
            prev = entry
        state.add_nedge(prev, tasklet, mm.Memlet())
    for conn, memlet in outputs.items():
        state.add_memlet_path(tasklet, *exits, state.add_write(memlet.data), src_conn=conn, memlet=dc(memlet))


def beta_scale_state(nsdfg: SDFG, node, dtype: dtypes.typeclass, n, rt_beta: bool, label: str) -> Optional[SDFGState]:
    """First state of a pure expansion: apply ``beta`` to the ``uplo`` triangle of
    ``_c`` -- scale it, zero it, or (compile-time ``beta == 1``) leave it alone, in
    which case ``None`` is returned and the caller chains its contraction directly.

    Only the ``uplo`` triangle is touched: BLAS leaves the opposite triangle of ``C``
    untouched, and so must the reference lowering. A runtime beta always takes the
    scale path -- its value is unknown at build time.
    """
    beta = node.beta
    if not rt_beta and beta == 1:
        return None  # C keeps its prior value; the WCR contraction accumulates onto it.

    st = nsdfg.add_state(label)
    if rt_beta:
        factor = "__beta" if beta == 1 else f"{render_scalar(beta, dtype)} * __beta"
        code = f"__o = {factor} * __c"
        inputs = {"__c": mm.Memlet("_c[__i, __j]"), "__beta": mm.Memlet("_beta[0]")}
    elif beta == 0:
        code = "__o = 0"
        inputs = {}
    else:
        code = f"__o = {render_scalar(beta, dtype)} * __c"
        inputs = {"__c": mm.Memlet("_c[__i, __j]")}
    add_triangular_tasklet(st, node.uplo, n, label, inputs, code, {"__o": mm.Memlet("_c[__i, __j]")})
    return st
