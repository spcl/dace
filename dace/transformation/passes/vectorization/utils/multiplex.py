# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Halve-index multiplex pattern detection for the vectorization pipeline.

A memlet begin expression of ``int_floor(i, k)`` (or alias ``floor_int(i, k)``)
means ``vector_length / k`` lanes share each load; such accesses are rewritten
through a packed transient and a ``multiplex_elements`` tasklet.
"""
import sympy

import dace
from dace import SDFGState


def detect_halve_index(state: SDFGState, new_inner_map: dace.nodes.MapEntry, vector_length):
    """Rewrite halve-indexed accesses on a map's out-edges into a multiplex tasklet.

    A begin expression ``int_floor(i, D)`` (``i`` the vectorized param) reads
    ``ceil(W / D)`` distinct contiguous source elements across the ``W`` lanes.
    It lowers to a contiguous strided load of those distinct elements plus a
    per-lane replication (``multiplex_elements``: ``out[l] = in[(i % D + l) / D]``).
    ``D`` may be any positive integer or a loop-invariant symbol (the tile base
    ``i`` is a multiple of ``W`` but not necessarily of ``D``, so the phase
    ``i % D`` shifts the per-lane source index; it is ``0`` when ``D`` divides
    ``W``).

    :param state: State containing the inner vector map.
    :param new_inner_map: Inner MapEntry whose out-edges are scanned.
    :param vector_length: Number of vector lanes.
    :returns: ``(modified_nodes, modified_edges)`` — the nodes and edges added
        or removed during the rewrite.
    :raises NotImplementedError: if a memlet has more than one halve-indexed
        dimension, or if a begin expression is not symbolic.
    """
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
                # ``out[l] = in[(i % D + l) / D]`` covers any divisor (constant or
                # loop-invariant symbol); the phase ``i % D`` corrects for a tile
                # base that is not a multiple of ``D`` (e.g. D=3, W=8).
                divisor_str = dace.symbolic.symstr(detected_divisor)
                param_str = dace.symbolic.symstr(detected_param)
                t = state.add_tasklet("pack_tasklet", {"_in"}, {"_out"},
                                      f"multiplex_elements(_in, _out, {vector_length}, {divisor_str}, "
                                      f"({param_str}) % ({divisor_str}));",
                                      language=dace.dtypes.Language.CPP,
                                      code_global=f'#include "dace/vector_intrinsics/multiplex.h"')
                modified_nodes.add(t)
                state.remove_edge(edge)
                new_range_list = list()
                # Input region = the ceil(W/D) distinct source elements:
                # [int_floor(i, D) : int_floor(i + W - 1, D)] inclusive.
                for (b, e, s) in edge.data.subset:
                    nb = b
                    if not isinstance(nb, sympy.Basic):
                        raise NotImplementedError(f"detect_halve_index expected symbolic begin, got {type(nb)}: {nb}")
                    ne = nb.subs(detected_param, f"({detected_param}+{vector_length - 1})")
                    new_range_list.append((nb, ne, 1))
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
    """Detect an ``int_floor(i, D)`` / ``floor_int(i, D)`` replication pattern.

    The numerator must be the bare vectorized index ``i``; the divisor ``D``
    may be a positive integer constant (2, 3, 4, ...) or a loop-invariant
    symbol (any symbolic expression not containing ``i``).

    :param expr: Symbolic begin expression to inspect.
    :returns: ``(symbol, divisor)`` if ``expr`` is ``int_floor(i, D)`` with a
        positive-int or loop-invariant-symbolic divisor ``D``, else
        ``(None, None)``. ``divisor`` is a Python ``int`` for a constant or a
        sympy expression for a symbolic divisor.
    """
    # Only custom functions
    if isinstance(expr, sympy.Function) and expr.func.__name__ in ("int_floor", "floor_int"):
        if len(expr.args) != 2:
            return None, None

        i, den = expr.args
        if not isinstance(i, sympy.Symbol):
            return None, None

        # Positive integer constant divisor.
        if isinstance(den, (int, sympy.Integer)) and den > 0:
            return i, int(den)

        # Loop-invariant symbolic divisor: any symbolic expression that does
        # not depend on the vectorized index ``i`` (so it is constant across
        # the lanes). A data-dependent or i-dependent divisor is rejected.
        if isinstance(den, sympy.Basic) and den.free_symbols and i not in den.free_symbols:
            return i, den

    return None, None
