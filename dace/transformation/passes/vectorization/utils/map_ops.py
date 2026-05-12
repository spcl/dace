# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Map mutation helpers used by the vectorization pipeline.

``remove_map`` is the only entry here today. The plan originally
suggested promoting it to ``dace.sdfg.utils`` because the body has no
vectorization-specific logic, but the actual call set is narrow (only
``RemoveVectorMaps`` invokes it) and keeping it inside the
vectorization package matches the reuse-threshold directive
("don't shuffle copies across directories without a measured win").
If a third party later wants it, the move out is one line.
"""
import copy

import dace
from dace.memlet import Memlet
import dace.sdfg.tasklet_utils as tutil
from dace.symbolic import DaceSympyPrinter


def remove_map(map_entry: dace.nodes.MapEntry, state: dace.SDFGState):
    assert map_entry in state.nodes()
    map_exit = state.exit_node(map_entry)

    # Replace symbol dictionary
    repldict = {str(p): str(r[0]) for p, r in zip(map_entry.map.params, map_entry.map.range)}

    # Redirect map entry's out edges
    write_only_map = True
    for edge in state.out_edges(map_entry):
        if edge.data.is_empty() or edge.data.data is None:
            parent_map_entry = state.entry_node(map_entry)
            if parent_map_entry is not None:
                state.add_edge(parent_map_entry, None, edge.dst, edge.dst_conn, edge.data)
        else:
            # Add an edge directly from the previous source connector to the destination
            path = state.memlet_path(edge)
            index = path.index(edge)
            state.add_edge(path[index - 1].src, path[index - 1].src_conn, edge.dst, edge.dst_conn, edge.data)
            write_only_map = False

    # Redirect map exit's in edges.
    for edge in state.in_edges(map_exit):
        path = state.memlet_path(edge)
        index = path.index(edge)

        # Add an edge directly from the source to the next destination connector
        if len(path) > index + 1:
            state.add_edge(edge.src, edge.src_conn, path[index + 1].dst, path[index + 1].dst_conn, edge.data)

            if write_only_map:
                outer_exit = path[index + 1].dst
                outer_entry = state.entry_node(outer_exit)
                if outer_entry is not None:
                    if any({e.src == map_entry for e in state.in_edges(edge.src)}):
                        state.add_edge(outer_entry, None, edge.src, None, Memlet(None))
                    else:
                        for src in {e.src for e in state.in_edges(edge.src)}:
                            state.add_edge(outer_entry, None, src, None, Memlet(None))

            else:
                outer_exit = path[index + 1].dst
                outer_entry = state.entry_node(outer_exit)

    state.remove_node(map_entry)
    state.remove_node(map_exit)

    # Replace symbols
    all_nodes = state.all_nodes_between(outer_entry, outer_exit)
    all_edges = state.all_edges(*all_nodes)
    for n in all_nodes:
        if isinstance(n, dace.nodes.Tasklet):
            code_before = copy.deepcopy(n.code.as_string)
            tutil.tasklet_replace_code(n, repldict, py_only=False, use_sym_expr=False)
            #print("Repldict:", repldict, "\nCode Before:", code_before, "\nCode After:", n.code.as_string)
        if isinstance(n, dace.nodes.NestedSDFG):
            for k, v in repldict.items():
                if k in n.symbol_mapping:
                    sym_expr = dace.symbolic.SymExpr(n.symbol_mapping[k])
                    if k in {str(s) for s in sym_expr.free_symbols}:
                        printer = DaceSympyPrinter(arrays=state.sdfg.arrays)
                        n.symbol_mapping[v] = printer.doprint(sym_expr.subs(k, v))
                    else:
                        n.symbol_mapping[v] = n.symbol_mapping[k]
                    del n.symbol_mapping[k]
            n.sdfg.replace_dict(repldict)
            for k, v in repldict.items():
                assert k not in n.sdfg.symbols
                assert k not in n.sdfg.free_symbols
            # SDFG repldict does not change edge subsets
            for _is in n.sdfg.all_states():
                for _se in _is.edges():
                    if _se.data.data is not None:
                        _se.data.subset.replace(repldict)
    for e in all_edges:
        if e.data.data is None:
            continue
        e.data.subset.replace(repldict)
