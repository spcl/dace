# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Privatize a WCR into an array element (``dot[0]``, after the reduction-to-WCR-map stage) to a transient scalar
seeded from the element before the map and written back after it, so CPU codegen emits an OpenMP
``reduction(op:scalar)`` clause instead of an atomic per iteration. The seed read and the writeback are
unconditional, so the rewrite holds when the map runs no iteration.
"""

from dace import data, subsets
from dace import memlet as mm
from dace.optionals import required
from dace.sdfg import SDFGState, nodes
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.narrowing import as_map_entry


def order_after_writeback(state: SDFGState, arr_node: nodes.AccessNode, after_map) -> None:
    """Order every access to ``arr_node``'s container that followed the map behind the in-state writeback into
    ``arr_node``: the map's WCR wrote the slot before it ended, the writeback copy only runs once the map has. A write
    happens where its producer runs, so the producers are ordered; an access inside a scope through its outermost
    entry."""
    behind, scope = set(state.bfs_nodes(arr_node)), state.scope_dict()
    for n in after_map:
        if not isinstance(n, nodes.AccessNode) or n.data != arr_node.data or n in behind:
            continue
        for target in [e.src for e in state.in_edges(n) if not e.data.is_empty()] or [n]:
            while scope[target] is not None:
                target = scope[target]
            if target not in behind and arr_node not in state.bfs_nodes(target):
                state.add_nedge(arr_node, target, mm.Memlet())
                behind |= set(state.bfs_nodes(target))


def reduction_slot(
    state: SDFGState, map_exit: nodes.MapExit, wcr_edge: MultiConnectorEdge[mm.Memlet]
) -> tuple[MultiConnectorEdge[mm.Memlet], list[MultiConnectorEdge[mm.Memlet]]] | None:
    """The map exit's out-edge to the array and the in-scope edges of the path, when ``wcr_edge`` reduces into one
    array element, the same for every iteration, of a map in the state's top scope."""
    if wcr_edge.data is None or wcr_edge.data.wcr is None:
        return None
    in_conn = wcr_edge.dst_conn
    if not in_conn or not in_conn.startswith("IN_"):
        return None
    out_edges = [e for e in state.out_edges(map_exit) if e.src_conn == "OUT_" + in_conn[3:]]
    if len(out_edges) != 1:
        return None
    oedge = out_edges[0]
    arr_node = oedge.dst
    if not isinstance(arr_node, nodes.AccessNode):
        return None
    desc = state.sdfg.arrays.get(arr_node.data)
    if desc is None or isinstance(desc, data.Scalar):
        return None
    # The in-edge carries the slot each iteration writes; the out-edge the union over iterations.
    write_subset = wcr_edge.data.subset
    if write_subset is None or write_subset.num_elements() != 1:
        return None
    map_entry = state.entry_node(map_exit)
    map_param_set = dict.fromkeys(as_map_entry(map_entry).map.params)
    if any(s in map_param_set for s in (str(x) for x in write_subset.free_symbols)):
        return None
    # Every edge of the scope's path writes that slot: a map nest renames them all to the scalar.
    inner = [e for e in state.memlet_path(wcr_edge) if e is not oedge]
    if any(e.data.data != arr_node.data or e.data.subset != write_subset for e in inner):
        return None
    # A map nested in another scope may reduce into a slot that scope's parameters pick, and the seed would bypass
    # that scope's entry.
    if state.entry_node(map_entry) is not None:
        return None
    return oedge, inner


def in_state_seed(state: SDFGState, arr_node: nodes.AccessNode, after_map: dict) -> nodes.AccessNode | None:
    """The access node of a plain write of ``arr_node``'s container ahead of the map in ``state`` (the kernel's own
    ``acc[c] = init``), which seeds the scalar instead of the buffer's prior contents."""
    for n in state.nodes():
        if not isinstance(n, nodes.AccessNode) or n is arr_node or n.data != arr_node.data:
            continue
        # A write after the map (``acc[c] = acc[c] * 2``) seeds nothing: reading the seed from it closes a cycle.
        if n in after_map:
            continue
        if not any(e.data is not None and not e.data.is_empty() for e in state.in_edges(n)):
            continue
        # A WCR write is another reduction target, not an init.
        if any(e.data is not None and e.data.wcr is not None for e in state.in_edges(n)):
            continue
        return n
    return None


def privatize_reduction_accumulator(
    state: SDFGState, map_exit: nodes.MapExit, wcr_edge: MultiConnectorEdge[mm.Memlet]
) -> bool:
    """Rewrite one map's WCR write into an array element as a WCR into a transient scalar, seeded before the map and
    written back after it.

    :param state: The state containing ``map_exit``.
    :param map_exit: The MapExit whose WCR out-write should be privatized.
    :param wcr_edge: An in-edge of ``map_exit`` carrying a WCR into a single array element.
    :returns: ``True`` if the rewrite landed, ``False`` (the SDFG unchanged) if the edge is not eligible.
    """
    slot = reduction_slot(state, map_exit, wcr_edge)
    if slot is None:
        return False
    oedge, inner = slot
    arr_node, write_subset, map_entry = oedge.dst, wcr_edge.data.subset, state.entry_node(map_exit)
    sdfg = state.sdfg
    after_map = dict.fromkeys(state.bfs_nodes(map_exit))
    in_state_init_an = in_state_seed(state, arr_node, after_map)

    # Another writer of ``arr`` in this state (a sibling map reducing into the same slot) would get its own seed and
    # writeback, and only the last writeback would survive. A writer after the map runs after the writeback.
    if state.in_degree(arr_node) != 1 or any(
        n.data == arr_node.data
        and n is not arr_node
        and n is not in_state_init_an
        and n not in after_map
        and any(not e.data.is_empty() for e in state.in_edges(n))
        for n in state.data_nodes()
    ):
        return False

    # An access to ``arr`` that follows the map in this state (lu's division of the reduced ``A[i, j]``) reads the
    # reduced value: the writeback stays in this state, ahead of it
    writeback_here = in_state_init_an is None and any(
        isinstance(n, nodes.AccessNode) and n.data == arr_node.data and (n is not arr_node or state.out_degree(n))
        for n in after_map
    )

    scalar_name, _ = sdfg.add_scalar(
        f"_priv_{arr_node.data}", dtype=sdfg.arrays[arr_node.data].dtype, transient=True, find_new_name=True
    )
    for edge in inner:
        edge.data.data = scalar_name
        edge.data.subset = subsets.Range([(0, 0, 1)])

    def slot_memlet() -> mm.Memlet:
        return mm.Memlet(data=arr_node.data, subset=subsets.Range.from_string(str(write_subset)))

    if in_state_init_an is not None:
        seed_an = state.add_access(scalar_name)
        state.add_edge(in_state_init_an, None, seed_an, None, slot_memlet())
        # The map's WCR accumulates onto the seed, so the seed must be written before the map starts.
        state.add_nedge(seed_an, required(map_entry), mm.Memlet())
    else:
        init_state = state.parent_graph.add_state_before(state, label=f"priv_init_{scalar_name}")
        init_state.add_edge(
            init_state.add_read(arr_node.data), None, init_state.add_write(scalar_name), None, slot_memlet()
        )
    new_scalar_an = state.add_write(scalar_name)
    state.add_edge(
        map_exit, oedge.src_conn, new_scalar_an, None, mm.Memlet(data=scalar_name, subset=subsets.Range([(0, 0, 1)]))
    )
    state.remove_edge(oedge)
    if in_state_init_an is not None or writeback_here:
        state.add_edge(new_scalar_an, None, arr_node, None, slot_memlet())
        order_after_writeback(state, arr_node, after_map)
        return True
    if state.degree(arr_node) == 0:
        state.remove_node(arr_node)
    wb_state = state.parent_graph.add_state_after(state, label=f"priv_wb_{scalar_name}")
    wb_state.add_edge(wb_state.add_read(scalar_name), None, wb_state.add_write(arr_node.data), None, slot_memlet())
    return True
