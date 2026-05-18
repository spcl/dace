# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Subset and memlet rewriting helpers for the vectorization pipeline.

Three rough sub-families:

- Pattern replace (``repl_subset``, ``repl_subset_to_use_*_offset``):
  symbolic substitution on a single subset.
- Memlet rewrite (``replace_memlet_expression``,
  ``expand_memlet_expression``, ``offset_memlets``,
  ``replace_all_access_subsets``): walk edges and replace the payload
  in-place.
- Post-collapse (``squeeze_memlets_of_packed_arrays``,
  ``use_previous_subsets``, ``try_clean_other_subset_going_out_from_map_entry``):
  fix up memlets after upstream passes changed the descriptor shape or
  surrounding map.
"""
import copy
from typing import Dict, Iterable, List, Optional, Set, Union

import dace
from dace import SDFGState, typeclass
from dace.memlet import Memlet
from dace.sdfg.graph import Edge
from dace.transformation.passes.vectorization.utils.lane_access import classify_lane_access


def repl_subset(subset: dace.subsets.Range, repl_dict: Dict[str, str]) -> dace.subsets.Range:
    """Apply ``repl_dict`` to a copy of ``subset`` (non-in-place ``.replace``).

    :param subset: Subset to copy and rewrite.
    :param repl_dict: Symbol-name to replacement-expression mapping.
    :returns: A new subset with the replacements applied.
    """
    new_subset = copy.deepcopy(subset)
    new_subset.replace(repl_dict)
    return new_subset


def _assert_no_new_free_symbols(sdfg: dace.SDFG, prev_sdfg_free_syms: Set, free_syms: Set, helper_name: str) -> None:
    """Raise if a subset rewrite introduced new free symbols into the SDFG.

    :param sdfg: The SDFG being rewritten.
    :param prev_sdfg_free_syms: SDFG free symbols before the rewrite.
    :param free_syms: Free symbols of the rewritten subset.
    :param helper_name: Caller name, used in the error message.
    :raises Exception: if the rewrite produced a free symbol absent before.
    """
    newly_free = sdfg.free_symbols - prev_sdfg_free_syms
    for free_sym in free_syms:
        if str(free_sym) in newly_free:
            raise Exception(f"`{helper_name}` has introduced new free symbols (this will cause problems as the new "
                            f"symbols should not be free). This will result an invalid SDFG, either call with "
                            f"`add_missing_symbols=True` or fix this issue")


def repl_subset_to_use_laneid_offset(sdfg: dace.SDFG, subset: dace.subsets.Range, symbol_offset: str,
                                     vector_map_param: str) -> dace.subsets.Range:
    """Rewrite a subset's free symbols to their per-lane variants.

    Each free symbol ``s`` becomes ``s_laneid_<symbol_offset>`` (added to the
    SDFG if absent), except the vector map param, which becomes
    ``(s + symbol_offset)``.

    :param sdfg: The SDFG containing the subset.
    :param subset: The subset whose symbols should be offset.
    :param symbol_offset: Integer-valued string suffix / offset.
    :param vector_map_param: The vector map parameter name.
    :returns: A new subset with the offset symbols applied.
    """
    # Offset needs to be positive integer
    assert symbol_offset.isdigit()
    prev_sdfg_free_syms = sdfg.free_symbols

    free_syms = subset.free_symbols

    repl_dict = {
        str(free_sym):
        str(free_sym) + "_laneid_" + str(symbol_offset) if str(free_sym) != vector_map_param else "(" + str(free_sym) +
        " + " + str(symbol_offset) + ")"
        for free_sym in free_syms
    }

    for free_sym in free_syms:
        if str(free_sym) in sdfg.symbols:
            stype = sdfg.symbols[str(free_sym)]
        else:
            stype = dace.int64
        if str(free_sym) != vector_map_param:
            offset_symbol_name = str(free_sym) + "_laneid_" + str(symbol_offset)
            if offset_symbol_name not in sdfg.symbols:
                sdfg.add_symbol(offset_symbol_name, stype)

    new_subset = repl_subset(subset=subset, repl_dict=repl_dict)
    _assert_no_new_free_symbols(sdfg, prev_sdfg_free_syms, free_syms, "repl_subset_to_use_laneid_offset")
    return new_subset


def repl_subset_to_use_with_int_offset(sdfg: dace.SDFG, subset: dace.subsets.Range, symbols_to_offset: Set[str],
                                       int_offset: int) -> dace.subsets.Range:
    """Add an integer offset to selected free symbols in a subset.

    Each symbol in ``symbols_to_offset`` becomes ``(s + int_offset)``. No new
    symbol is added.

    :param sdfg: The SDFG containing the subset.
    :param subset: The subset to rewrite.
    :param symbols_to_offset: Names of the symbols to offset.
    :param int_offset: Integer offset to add.
    :returns: A new subset with the offset applied.
    """
    prev_sdfg_free_syms = sdfg.free_symbols

    free_syms = subset.free_symbols
    new_range_list = []
    repl_dict = {str(free_sym): "(" + str(free_sym) + " + " + str(int_offset) + ")" for free_sym in symbols_to_offset}
    for (b, e, s) in subset:
        if hasattr(b, "subs"):
            nb = b.subs(repl_dict)
        else:
            nb = b
        if hasattr(e, "subs"):
            ne = e.subs(repl_dict)
        else:
            ne = e
        ns = 1
        new_range_list.append((nb, ne, ns))

    new_subset = dace.subsets.Range(new_range_list)
    _assert_no_new_free_symbols(sdfg, prev_sdfg_free_syms, free_syms, "repl_subset_to_use_with_int_offset")
    return new_subset


def replace_memlet_expression(state: SDFGState,
                              edges: Iterable[Edge[Memlet]],
                              old_subset_expr: dace.subsets.Range,
                              new_subset_expr: dace.subsets.Range,
                              repl_scalars_with_arrays: bool,
                              edges_to_skip: Set[Edge[Memlet]],
                              vector_numeric_type: typeclass,
                              dataname: Union[str, None] = None) -> Set[str]:
    """Replace memlet subsets matching ``old_subset_expr`` with a new subset.

    Optionally converts Scalar / size-1 Array nodes on matching edges to
    arrays shaped to ``new_subset_expr`` using ``vector_numeric_type``.

    :param state: The SDFG state containing the edges.
    :param edges: Edges whose memlets are checked and potentially replaced.
    :param old_subset_expr: The subset pattern to match.
    :param new_subset_expr: The replacement subset.
    :param repl_scalars_with_arrays: If True, convert Scalar / size-1 Array
        nodes on matching edges to arrays shaped to ``new_subset_expr``.
    :param edges_to_skip: Edges that must not be modified (a match here is a
        bug).
    :param vector_numeric_type: Dtype used when converting scalars to arrays.
    :param dataname: If not None, also require the memlet data to match.
    :raises Exception: if an edge in ``edges_to_skip`` matches the pattern.
    """
    arr_dim = [((e + 1 - b) // s) for (b, e, s) in new_subset_expr]

    for edge in edges:
        src_node: dace.nodes.Node = edge.src
        dst_node: dace.nodes.Node = edge.dst

        if edge.data is not None and edge.data.subset == old_subset_expr:
            if edge in edges_to_skip:
                raise Exception("AA")
            if edge.data.data != dataname and dataname is not None:
                continue
            if repl_scalars_with_arrays:
                for data_node in [src_node, dst_node]:
                    if isinstance(data_node, dace.nodes.AccessNode):
                        arr = state.sdfg.arrays[data_node.data]
                        if isinstance(arr, dace.data.Scalar) or (isinstance(arr, dace.data.Array)
                                                                 and arr.shape == (1, )):
                            state.sdfg.remove_data(data_node.data, validate=False)
                            state.sdfg.add_array(name=data_node.data,
                                                 shape=tuple(arr_dim),
                                                 dtype=vector_numeric_type,
                                                 storage=arr.storage,
                                                 location=arr.location,
                                                 transient=arr.transient,
                                                 lifetime=arr.lifetime,
                                                 find_new_name=False)
            edge.data = dace.memlet.Memlet(data=edge.data.data, subset=copy.deepcopy(new_subset_expr))


def expand_memlet_expression(state: SDFGState,
                             edges: Iterable[Edge[Memlet]],
                             edges_to_skip: Set[Edge[Memlet]],
                             vector_width: int,
                             vector_map_param: Optional[str] = None) -> Set[Edge[Memlet]]:
    """Widen single-element memlet subsets to ``vector_width`` on the lane dim.

    The W lanes step the dimension that carries the vectorized map
    parameter, *not* necessarily the array's storage-contiguous dim. For
    a contiguous body access (``A[..., i]`` with ``i`` in the stride-1
    dim) the two coincide. But a *transposed* read — e.g. a branch
    condition ``zqx[z1, i, j]`` where ``i`` (the lane var) sits in a
    non-contiguous dim while the stride-1 dim holds the outer-loop
    constant ``j`` — must widen the ``i`` dim (the memory gather stride
    is then implicit in the array layout). Widening the storage-stride-1
    dim there would read ``W`` wrong elements along the constant ``j``
    and corrupt the per-lane condition.

    :param state: The SDFG state containing the edges.
    :param edges: The memlet edges to inspect and possibly modify.
    :param edges_to_skip: Edges that must not be expanded.
    :param vector_width: Number of elements to widen the lane dim to.
    :param vector_map_param: The vectorized map parameter. When given,
        the dim whose ``begin`` references it is widened (index step 1);
        the array's per-dim stride supplies the memory gather stride
        downstream. When ``None`` (or the param is absent from every
        dim's ``begin``), fall back to widening the storage-stride-1 dim.
    :returns: The set of edges whose memlets were modified.
    :raises Exception: if a subset is neither all length-1 nor has exactly
        one ``vector_width``-length dim.
    """
    param_sym = dace.symbolic.symbol(vector_map_param) if vector_map_param is not None else None
    modified_edges = set()
    for edge in edges:
        if edge.data is not None:
            if not all(((e + 1 - b) // s) == 1 for b, e, s in edge.data.subset):
                # Edge found where not all memlet subsets are length 1.
                # That's still acceptable iff exactly one dim has length
                # equal to ``vector_width`` (the dim we'll expand); any
                # other mix raises below.
                vlens = {((e + 1 - b) // s) == vector_width for b, e, s in edge.data.subset}
                if len(vlens) > 1:
                    raise Exception(
                        f"Memlet subsets for edge {edge}: {[(b, e, s) for b, e, s in edge.data.subset]},"
                        f"is not all length one or max 1 vector width subset: {[((e + 1 - b) // s) == 1 for b, e, s in edge.data.subset]}"
                    )
                else:
                    # Do not do anything
                    continue

            subset = edge.data.subset
            # The dim whose ``begin`` references the vectorized map
            # parameter is the lane dim — that, not the storage
            # stride-1 dim, must be widened. ``classify_lane_access``
            # is the single source of truth: ``lane_dim`` is set for a
            # single-param dim (CONTIGUOUS / STRIDED / TRANSPOSED) and
            # ``None`` for the param-free (CONSTANT) and multi-param
            # (DIAGONAL) cases, which both take the legacy path.
            ld = None
            if param_sym is not None:
                ld = classify_lane_access(subset, state.sdfg.arrays[edge.data.data].strides,
                                          vector_map_param).lane_dim

            new_subset_list = []
            if ld is not None:
                for d, (b, e, s) in enumerate(subset):
                    assert b == e and s == 1
                    if d == ld:
                        new_subset_list.append((b, b + vector_width - 1, s))
                    else:
                        new_subset_list.append((b, e, s))
            else:
                # No (or ambiguous multi-dim) lane param in the subset —
                # keep the legacy storage-stride-1 widening (unchanged
                # behaviour for contiguous accesses and the param-free
                # boundary case).
                for (b, e, s), stride in zip(subset, state.sdfg.arrays[edge.data.data].strides):
                    assert b == e and s == 1
                    if stride == 1:
                        new_subset_list.append((b, b + vector_width - 1, s))
                    else:
                        new_subset_list.append((b, e, s))
            new_subset_expr = dace.subsets.Range(new_subset_list)

            if new_subset_expr != edge.data.subset:
                edge.data = dace.memlet.Memlet(data=edge.data.data, subset=copy.deepcopy(new_subset_expr))
                modified_edges.add(edge)
    return modified_edges


def offset_memlets(sdfg: dace.SDFG, dataname: str, offsets: List[dace.symbolic.SymExpr]):
    """Subtract ``offsets`` from every memlet subset of ``dataname``.

    Length-1 dimensions are collapsed out of the resulting subset.

    :param sdfg: The SDFG to walk.
    :param dataname: Data name whose memlets are offset.
    :param offsets: Per-dimension offsets to subtract.
    """
    from dace.transformation.passes.vectorization.utils.iteration import walk_memlets_of
    for _state, edge in walk_memlets_of(sdfg, dataname):
        subset = edge.data.subset.offset_new(dace.subsets.Range(offsets), negative=True)
        # If subset is not one dimensional we need to collapse 0 accesses
        collapsed_subset_list = [(b, e, s) for (b, e, s) in subset if (e + 1 - b) // s != 1]
        edge.data.subset = dace.subsets.Range(collapsed_subset_list)


def replace_all_access_subsets(state: dace.SDFGState, name: str, new_subset_expr: str):
    """Replace every memlet subset for ``name`` in ``state`` with a new subset.

    :param state: The SDFG state to modify.
    :param name: Array name whose accesses are replaced.
    :param new_subset_expr: The new subset expression (e.g. ``"0:4"``).
    """
    for edge in state.edges():
        if edge.data is not None and edge.data.data == name:
            nm = dace.memlet.Memlet(expr=f"{name}[{new_subset_expr}]")
            edge.data = nm


def squeeze_memlets_of_packed_arrays(state: dace.SDFGState, map_entry: dace.nodes.MapEntry,
                                     array_accesses_to_be_packed: Set[str]):
    """Collapse memlet subsets of packed arrays to single-element accesses.

    :param state: The SDFG state to modify.
    :param map_entry: Map whose body edges are inspected.
    :param array_accesses_to_be_packed: Data names to squeeze.
    """
    all_nodes = state.all_nodes_between(map_entry, state.exit_node(map_entry))
    all_edges = state.all_edges(*all_nodes)
    for edge in all_edges:
        if edge.data.data in array_accesses_to_be_packed:
            new_range_list = [(b, b, 1) for (b, e, s) in edge.data.subset]
            edge.data = dace.memlet.Memlet(data=edge.data.data, subset=dace.subsets.Range(new_range_list))


def use_previous_subsets(state: dace.SDFGState, map_entry: dace.nodes.MapEntry, vector_width: int,
                         vectorizable_arrays: Set[str]):
    """Rewrite out-edge memlet subsets of a single-param inner map to refer to the parent map.

    For each outgoing memlet, the matching incoming memlet's subset is cloned,
    the outer map symbol is substituted by the inner param, and an exact
    ``vector_width``-length dim has its end bound shrunk by ``vector_width - 1``.

    :param state: The SDFG state containing the map entry.
    :param map_entry: The map entry to rewrite; must have exactly one param.
    :param vector_width: Width of the structured vector access.
    :param vectorizable_arrays: Data names eligible for the rewrite.
    """

    # Inner map has exactly one parameter, e.g., `i`.
    assert len(map_entry.map.params) == 1
    inner_param = map_entry.map.params[0]

    # Extract parent-map lower bound symbol as string, e.g. `[tile_i : ...]`.
    outer_param = str(map_entry.map.range[0][0])

    for out_edge in state.out_edges(map_entry):
        if out_edge.src_conn is None:
            continue

        # Find the corresponding incoming edge with IN_<idx> for OUT_<idx>.
        in_edges = set(state.in_edges_by_connector(map_entry, out_edge.src_conn.replace("OUT_", "IN_")))
        if not in_edges:
            continue

        # Safe: at most one incoming edge per OUT connector.
        assert len(in_edges) == 1
        in_edge = next(iter(in_edges))

        if in_edge.data.data not in vectorizable_arrays:
            continue

        # Copy original subset.
        orig_subset = copy.deepcopy(in_edge.data.subset)

        new_ranges = []
        volume = 1

        for (begin, end, stride) in orig_subset:
            # Rewrite begin expression
            if hasattr(begin, "subs"):
                begin_str = str(begin.subs({outer_param: inner_param}))
                new_begin = dace.symbolic.SymExpr(begin_str).simplify()
            else:
                new_begin = begin

            # Rewrite end expression
            if hasattr(end, "subs"):
                # Subset extent length
                extent = (end + 1 - begin) // stride
                # If exact vector access, shrink by vector_width - 1
                tail_adjust = vector_width - 1 if extent == vector_width else 0
                end_str = f"{end.subs({outer_param: inner_param})} - {tail_adjust}"
                new_end = dace.symbolic.SymExpr(end_str).simplify()
                volume *= extent
            else:
                new_end = end
            new_ranges.append((new_begin, new_end, stride))

        # Assign new memlet with updated subset and volume
        out_edge.data = dace.memlet.Memlet(
            data=out_edge.data.data,
            subset=dace.subsets.Range(new_ranges),
            volume=volume,
        )


def try_clean_other_subset_going_out_from_map_entry(state: SDFGState, map_entry: dace.nodes.MapEntry):
    """Replace map-entry out-edges carrying an ``other_subset`` with assign tasklets.

    Each such edge to an AccessNode is split into an in-memlet and an
    out-memlet via an inserted ``_out = _in`` tasklet.

    :param state: The SDFG state to modify.
    :param map_entry: Map entry whose out-edges are inspected.
    """
    id = 0
    for oe in state.out_edges(map_entry):
        if oe.data.other_subset is not None and isinstance(oe.dst, dace.nodes.AccessNode):
            assert oe.data.data is not None and oe.data.data != oe.dst.data
            # Add assignment tasklet
            t = state.add_tasklet(f"other_subset_assign_{id}", {"_in"}, {"_out"}, "_out = _in")
            state.remove_edge(oe)
            state.add_edge(oe.src, oe.src_conn, t, "_in", dace.memlet.Memlet(data=oe.data.data, subset=oe.data.subset))
            state.add_edge(t, "_out", oe.dst, oe.dst_conn,
                           dace.memlet.Memlet(data=oe.dst.data, subset=oe.data.other_subset))
            id += 1
