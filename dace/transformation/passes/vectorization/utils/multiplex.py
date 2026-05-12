# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Halve-index multiplex pattern detection.

When a memlet begin expression is ``int_floor(i, k)`` (or the
deprecated alias ``floor_int(i, k)``), it indicates an access pattern
where ``vector_length / k`` distinct lanes share each load. The
vectorizer rewrites such accesses by allocating a packed transient
sized ``vector_length`` and emitting a ``multiplex_elements`` tasklet
that fans the loaded values out to the right lanes.

The two helpers here live in their own module because the surrounding
plumbing (``multiplex_elements`` C++ runtime in
``dace/runtime/include/dace/vector_intrinsics/multiplex.h`` +
``multiplexed_<array>`` transient naming) is its own concern,
independent of subset rewriting or NSDFG shape reconciliation.
"""
import sympy

import dace
from dace import SDFGState


def detect_halve_index(state: SDFGState, new_inner_map: dace.nodes.MapEntry, vector_length):
    all_nodes = state.all_nodes_between(new_inner_map, state.exit_node(new_inner_map))
    map_param = new_inner_map.map.params[-1]
    all_edges = state.out_edges(new_inner_map)
    modified_nodes = set()
    modified_edges = set()
    for edge in all_edges:
        if edge.data.subset is not None:
            detected_param = None
            detected_divisor = None
            for b, e, s in edge.data.subset:
                param, divisor = detect_halve_index_impl(b)
                if param is not None and divisor is not None:
                    if detected_param is not None:
                        raise NotImplementedError(f"Multiple halve-indexed dimensions on memlet {edge.data}; "
                                                  f"only one supported (state {state.label}, edge {edge})")
                    detected_param = param
                    detected_divisor = divisor
            if detected_param is not None:
                # Multiply end expression with
                desc = state.sdfg.arrays[edge.data.data]
                arr_name, arr = state.sdfg.add_array(name=f"multiplexed_{edge.data.data}",
                                                     shape=(vector_length, ),
                                                     dtype=desc.dtype,
                                                     transient=True,
                                                     storage=dace.dtypes.StorageType.Register,
                                                     find_new_name=True)
                if vector_length % detected_divisor != 0:
                    raise NotImplementedError(f"vector_length={vector_length} not divisible by halve-index divisor "
                                              f"{detected_divisor} on memlet {edge.data}")
                t = state.add_tasklet(
                    "pack_tasklet", {"_in"}, {"_out"},
                    f"multiplex_elements(_in, _out, {vector_length // detected_divisor}, {detected_divisor});",
                    language=dace.dtypes.Language.CPP,
                    code_global=f'#include "dace/vector_intrinsics/multiplex.h"')
                modified_nodes.add(t)
                state.remove_edge(edge)
                new_range_list = list()
                # Detection means we should have b -> b+vector_length step size 1 on the param dim
                for (b, e, s) in edge.data.subset:
                    nb = b
                    if not hasattr(nb, "subs"):
                        raise NotImplementedError(f"detect_halve_index expected symbolic begin, got {type(nb)}: {nb}")
                    ne = nb.subs(detected_param, f"({detected_param}+{vector_length})")
                    ns = 1
                    new_range_list.append((nb, ne, ns))
                e1 = state.add_edge(edge.src, edge.src_conn, t, "_in",
                                    dace.memlet.Memlet(data=edge.data.data, subset=dace.subsets.Range(new_range_list)))
                access = state.add_access(arr_name)
                modified_nodes.add(access)
                modified_edges.add(e1)
                modified_edges.add(edge)
                e2 = state.add_edge(t, "_out", access, None,
                                    dace.memlet.Memlet.from_array(dataname=arr_name, datadesc=arr))
                e3 = state.add_edge(access, None, edge.dst, edge.dst_conn,
                                    dace.memlet.Memlet.from_array(dataname=arr_name, datadesc=arr))
                modified_edges.add(e2)
                modified_edges.add(e3)
    return modified_nodes, modified_edges


def detect_halve_index_impl(expr):
    """
    Detect patterns like int_floor(i, k) or floor_int(i, k)
    where k is ANY positive integer.

    Returns:
        (symbol, divisor) or (None, None)
    """
    # Only custom functions
    if isinstance(expr, sympy.Function) and expr.func.__name__ in ("int_floor", "floor_int"):
        if len(expr.args) != 2:
            return None, None

        i, den = expr.args

        # Divisor must be a positive integer
        if isinstance(i, sympy.Symbol) and isinstance(den, (int, sympy.Integer)) and den > 0:
            return i, int(den)

    return None, None
