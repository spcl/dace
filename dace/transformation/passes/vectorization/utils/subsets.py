# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Subset and memlet rewriting helpers for the vectorization pipeline.

- Endpoint inspection (``an_side_subset``, ``infer_edge_endpoints``): read an AccessNode's subset
  from an incident edge without picking the wrong side.
- Pattern replace (``repl_subset``, ``repl_subset_to_use_laneid_offset``): symbolic substitution on
  a single subset.
- Memlet rewrite (``replace_all_access_subsets``): walk edges, replace the payload in-place.
"""
import copy
from typing import Dict, Optional, Set

import dace
from dace.sdfg.graph import Edge
from dace.sdfg.nodes import AccessNode
from dace.subsets import Range
from dace.transformation.passes.vectorization.utils.name_schemes import LaneIdScheme


def infer_edge_endpoints(edge: Edge, sdfg: dace.SDFG, state: 'dace.SDFGState'):
    """``(src_data_name, src_subset, dst_data_name, dst_subset)`` for a memlet edge, both endpoints
    inferred.

    Per user direction 2026-06-10 (design 3.7 + 3.8.3): classifiers and validators should ALWAYS
    know both real array names + subsets, not just the one matching ``memlet.data``. Centralises the
    "which side is which" reasoning so call sites don't reimplement the lookup.

    ``data_name`` = ``None`` when the endpoint isn't an :class:`AccessNode` (lib node connector /
    tasklet); its ``subset`` is then also ``None`` (connector descriptor defines that side's shape,
    not the memlet).

    :param edge: A memlet-carrying edge.
    :param sdfg: SDFG owning the endpoint descriptors.
    :param state: The state holding ``edge`` (resolves which end each subset belongs to).
    :returns: ``(src_data_name, src_subset, dst_data_name, dst_subset)``; ``subset`` fields are fresh
        :class:`Range` copies (safe to mutate) or ``None`` for non-AN endpoints.
    """
    from dace.sdfg.nodes import AccessNode as _AccessNode
    mem = edge.data
    if mem is None:
        raise ValueError(f"infer_edge_endpoints: edge {edge} has no memlet")
    src_an = edge.src if isinstance(edge.src, _AccessNode) else None
    dst_an = edge.dst if isinstance(edge.dst, _AccessNode) else None
    src_data = src_an.data if src_an is not None else None
    dst_data = dst_an.data if dst_an is not None else None
    src_subset: Optional[Range] = None
    dst_subset: Optional[Range] = None
    if src_an is not None:
        src_subset = an_side_subset(edge, src_an, sdfg, state)
    if dst_an is not None:
        dst_subset = an_side_subset(edge, dst_an, sdfg, state)
    return src_data, src_subset, dst_data, dst_subset


def an_side_subset(edge: Edge, an: AccessNode, sdfg: dace.SDFG, state: 'dace.SDFGState') -> Range:
    """Return the subset belonging to ``an`` on the AN-incident ``edge``.

    An AN-incident edge carries one endpoint's region in ``edge.data.subset`` and the other's in
    ``edge.data.other_subset``. **Which end ``subset`` names is carried by the memlet's own
    ``_is_data_src`` flag, not by ``edge.data.data``**: on a self-copy ``A -> A`` both endpoints match
    ``data``, so a name test hands the SAME subset to both ends and the read region silently doubles
    as the write region. ``get_src_subset`` / ``get_dst_subset`` are the only correct readers of the
    pair, and are what the CPU generator lowers a copy edge with.

    Absent side = implicit full-shape copy, so the AN's full descriptor range is the answer.

    :param edge: An edge with ``an`` as one of its endpoints.
    :param an: The :class:`AccessNode` whose subset is wanted.
    :param sdfg: The SDFG owning ``an`` (descriptor lookup for the full-shape fallback).
    :param state: The state holding ``edge`` (resolves which end ``an`` is).
    :returns: A fresh :class:`Range` carrying ``an``'s subset on ``edge``.
    :raises ValueError: When ``edge`` has no memlet or its endpoints don't include ``an``.
    """
    mem = edge.data
    if mem is None:
        raise ValueError(f"an_side_subset: edge {edge} has no memlet")
    if edge.src is not an and edge.dst is not an:
        raise ValueError(f"an_side_subset: AN {an.data!r} is not an endpoint of edge {edge}")
    side = mem.get_src_subset(edge, state) if edge.src is an else mem.get_dst_subset(edge, state)
    if side is not None:
        return copy.deepcopy(side)
    desc = an.desc(sdfg)
    return Range([(0, s - 1, 1) for s in desc.shape])


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

    offset_lane = int(symbol_offset)
    repl_dict = {
        str(free_sym): (LaneIdScheme.make_dim(str(free_sym), 0, offset_lane)
                        if str(free_sym) != vector_map_param else f"({str(free_sym)} + {symbol_offset})")
        for free_sym in free_syms
    }

    for free_sym in free_syms:
        if str(free_sym) in sdfg.symbols:
            stype = sdfg.symbols[str(free_sym)]
        else:
            stype = dace.int64
        if str(free_sym) != vector_map_param:
            offset_symbol_name = LaneIdScheme.make_dim(str(free_sym), 0, offset_lane)
            if offset_symbol_name not in sdfg.symbols:
                sdfg.add_symbol(offset_symbol_name, stype)

    new_subset = repl_subset(subset=subset, repl_dict=repl_dict)
    _assert_no_new_free_symbols(sdfg, prev_sdfg_free_syms, free_syms, "repl_subset_to_use_laneid_offset")
    return new_subset


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
