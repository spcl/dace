# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Subset and memlet rewriting helpers.

This module groups the helpers that mutate ``dace.subsets.Range``
expressions and the memlets that carry them. Three rough sub-families:

- **Pattern replace** (``repl_subset``, ``repl_subset_to_use_*_offset``)
  — symbolic substitution on a single subset.
- **Memlet rewrite** (``replace_memlet_expression``,
  ``expand_memlet_expression``, ``offset_memlets``,
  ``replace_all_access_subsets``) — walk a set of edges and replace the
  memlet payload in-place.
- **Post-collapse** (``squeeze_memlets_of_packed_arrays``,
  ``use_previous_subsets``, ``try_clean_other_subset_going_out_from_map_entry``)
  — fix up memlets after upstream passes have already changed the
  data-descriptor shape or the surrounding map.

Per the locked policy (mechanical-only relocation + defensive checks
stay), every helper is moved verbatim. Tier-1 / Tier-2 audit fixes that
target this family (``replace_memlet_expression`` matching by edge
identity, ``expand_memlet_expression`` raise-instead-of-assert, etc.)
are tracked separately and not applied during this move.
"""
import copy
from typing import Dict, Iterable, List, Set, Union

import dace
from dace import SDFGState, typeclass
from dace.memlet import Memlet
from dace.sdfg.graph import Edge


def repl_subset(subset: dace.subsets.Range, repl_dict: Dict[str, str]) -> dace.subsets.Range:
    """ Convenience wrapper to make the .replace not in-place """
    new_subset = copy.deepcopy(subset)
    new_subset.replace(repl_dict)
    return new_subset


def _assert_no_new_free_symbols(sdfg: dace.SDFG, prev_sdfg_free_syms: Set, free_syms: Set, helper_name: str) -> None:
    """Raise if a subset rewrite introduced new free symbols into the SDFG.

    Both ``repl_subset_to_use_laneid_offset`` and
    ``repl_subset_to_use_with_int_offset`` check that the rewrite has
    not produced free symbols that did not exist in the SDFG before; an
    invalid SDFG with un-bound symbols is the silent-corruption failure
    mode this guard catches. The two callers pass their own
    ``helper_name`` for the error message so the trace points at the
    real callsite.
    """
    newly_free = sdfg.free_symbols - prev_sdfg_free_syms
    for free_sym in free_syms:
        if str(free_sym) in newly_free:
            raise Exception(f"`{helper_name}` has introduced new free symbols (this will cause problems as the new "
                            f"symbols should not be free). This will result an invalid SDFG, either call with "
                            f"`add_missing_symbols=True` or fix this issue")


def repl_subset_to_use_laneid_offset(sdfg: dace.SDFG, subset: dace.subsets.Range, symbol_offset: str,
                                     vector_map_param: str) -> dace.subsets.Range:
    """
    Apply a symbolic offset to all free symbols in a subset.

    This function replaces each free symbol in the subset with a new symbol
    that has the offset appended to its name (e.g., 'i' becomes 'i_{offset}' where offset is an integer).
    New symbols are automatically added to the SDFG if they don't exist.

    If symbol is vector map param always add + 1 instead of laneid

    Args:
        sdfg: The SDFG containing the subset
        subset: The subset whose symbols should be offset
        symbol_offset: String to append to each symbol name (should be an integer)
        add_missing_symbols: If True, adds symbol mappings and assignments for
                           free symbols in the parent SDFG

    Returns:
        A new subset with offset symbols applied

    Example:
        If subset contains symbol 'i' and symbol_offset is '_v':
        - 'i' becomes 'i_v'
        - Symbol 'i_v' is added to SDFG if not present
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
    """
    Apply a int offset to all free symbols appearing on `symbols_to_offset` in a subset.

    that has the offset appended to its name (e.g., 'i' becomes 'i + {int_offset}' where offset is an integer).
    No new symbol is added
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
    """
    Replace memlet subsets matching a pattern with a new subset expression.

    Optionally converts scalar/size-1 arrays to arrays that match the new_subset_expr's sizes
    using the `vector_numeric_type` as dtype to accommodate the new subset dimensions.

    Args:
        state: The SDFG state containing the edges
        edges: Edges whose memlets should be checked and potentially replaced
        old_subset: The subset pattern to match
        new_subset: The replacement subset
        convert_scalars_to_arrays: If True, converts Scalar/size-1 Array nodes
                                  to proper Arrays with shape matching new_subset
        edges_to_skip: Set of edges that should not be modified (validation)
        vector_dtype: Data type to use when converting scalars to arrays
        dataname: if not None checks for memlet data too

    Raises:
        Exception: If an edge marked to skip is encountered during replacement
        because it indicates a bug in the auto-vectorization logic

    Side Effects:
        - Modifies memlet subsets on matching edges
        - May remove and re-add array data descriptors with new shapes
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


def expand_memlet_expression(state: SDFGState, edges: Iterable[Edge[Memlet]], edges_to_skip: Set[Edge[Memlet]],
                             vector_width: int) -> Set[Edge[Memlet]]:
    """
    Expand single-element memlet subsets along stride-1 dimensions to a given vector length.
    Pre-condition: all subset dimensions need to be 1

    For each memlet edge, this function modifies subsets that represent a single element
    and extend them to cover `vector_width` elements when the corresponding array stride is 1.
    Trying to modify an edge listed in `edges_to_skip` raises an error as it indicates a
    bug in the auto-vectorization logic.

    Args:
        state (SDFGState): The SDFG state containing the edges.
        edges (Iterable[Edge[Memlet]]): The memlet edges to inspect and possibly modify.
        edges_to_skip (Set[Edge[Memlet]]): Edges that should not be expanded.
        vector_width (int): The number of elements to expand contiguous subsets to.

    Returns:
        Set[Edge[Memlet]]: The set of edges whose memlets were modified.
    """
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
            new_subset_list = []
            for (b, e, s), stride in zip(edge.data.subset, state.sdfg.arrays[edge.data.data].strides):
                if stride == 1:
                    assert b == e
                    assert s == 1
                    new_subset_list.append((b, b + vector_width - 1, s))
                else:
                    assert b == e
                    assert s == 1
                    new_subset_list.append((b, e, s))
            new_subset_expr = dace.subsets.Range(new_subset_list)

            if new_subset_expr != edge.data.subset:
                edge.data = dace.memlet.Memlet(data=edge.data.data, subset=copy.deepcopy(new_subset_expr))
                modified_edges.add(edge)
    return modified_edges


def offset_memlets(sdfg: dace.SDFG, dataname: str, offsets: List[dace.symbolic.SymExpr]):
    from dace.transformation.passes.vectorization.utils.iteration import walk_memlets_of
    for _state, edge in walk_memlets_of(sdfg, dataname):
        subset = edge.data.subset.offset_new(dace.subsets.Range(offsets), negative=True)
        # If subset is not one dimensional we need to collapse 0 accesses
        collapsed_subset_list = [(b, e, s) for (b, e, s) in subset if (e + 1 - b) // s != 1]
        edge.data.subset = dace.subsets.Range(collapsed_subset_list)


def replace_all_access_subsets(state: dace.SDFGState, name: str, new_subset_expr: str):
    """
    Replaces all memlet subsets for a given array in a state with a new subset expression.

    Args:
        state: The SDFG state to modify.
        name: Array name whose accesses are replaced.
        new_subset_expr: The new subset expression (e.g., "0:4").
    """
    for edge in state.edges():
        if edge.data is not None and edge.data.data == name:
            nm = dace.memlet.Memlet(expr=f"{name}[{new_subset_expr}]")
            edge.data = nm


def squeeze_memlets_of_packed_arrays(state: dace.SDFGState, map_entry: dace.nodes.MapEntry,
                                     array_accesses_to_be_packed: Set[str]):
    all_nodes = state.all_nodes_between(map_entry, state.exit_node(map_entry))
    all_edges = state.all_edges(*all_nodes)
    for edge in all_edges:
        if edge.data.data in array_accesses_to_be_packed:
            new_range_list = [(b, b, 1) for (b, e, s) in edge.data.subset]
            edge.data = dace.memlet.Memlet(data=edge.data.data, subset=dace.subsets.Range(new_range_list))


def use_previous_subsets(state: dace.SDFGState, map_entry: dace.nodes.MapEntry, vector_width: int,
                         vectorizable_arrays: Set[str]):
    """
    Rewrite memlet subsets on edges leaving a single-parameter inner map so that
    structured vector accesses correctly refer to the surrounding parent map.

    The function performs:
        1. Extract the inner map parameter (e.g., `i`).
        2. Extract the parent's map lower bound symbol (e.g., `tile_i`).
        3. For each outgoing memlet:
             - Identify its corresponding incoming memlet (IN_x -> OUT_x).
             - Clone its subset.
             - Substitute outer -> inner symbols for begin/end expressions.
             - Adjust end bound when the subset length matches `vector_width` (=structured access).
             - Compute the memlet volume.
             - Assign a new Memlet with the updated subset and volume.

    Parameters
    ----------
    state : dace.SDFGState
        The current SDFG state containing the map entry.
    map_entry : dace.nodes.MapEntry
        The map entry whose outgoing memlet subsets will be rewritten.
        Must have exactly one map parameter.
    vector_width : int
        Width of the structured vector access. If a subset dimension has length
        equal to `vector_width`, we shrink its end bound by `vector_width - 1`.
        As in this case we have an exact subset, otherwise we pass a complete dimension or something in that fay that we cant change.

    Notes
    -----
    We cast symbolic expressions to string and re-sympify them to force SymPy
    to reattach the same symbol objects used by DaCe.
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
