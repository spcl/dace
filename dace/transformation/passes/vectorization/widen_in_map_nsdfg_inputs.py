# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tile-expand the inner memlets of an in-map body NSDFG.

Runs after :class:`ExpandNestedSDFGInputs` has normalized the outer
boundary (every NSDFG in/out edge now reads/writes the full source
array; the inner descriptor mirrors the source; inner memlets carry the
per-iter offset). What remains is to expand the K tile-var-bearing dims
from per-iter points (``arr[jc, jk, ...]``) to per-tile ranges
(``arr[jc:jc+W, jk:jk+W, ...]``) so the downstream classify / promote
pipeline sees uniformly tile-shaped reads / writes.

Position-aware tile-var detection
---------------------------------

A dim is tile-expanded only when its lower-bound expression contains a
tile iter-var as a **direct top-level symbol** (under arithmetic ops
only). When the tile-var only appears nested inside a ``Subscript``
(``arr[edge_blk[jb, jc, 0]]``) the dim is a data-dependent gather and
the gather machinery is the right consumer; this rewrite leaves it
alone.
"""
from typing import Dict, Optional, Set

import sympy

from dace import SDFG, symbolic
from dace.sdfg import SDFGState, nodes
from dace.subsets import Range


def _direct_symbols(expr) -> Set[str]:
    """Return the set of symbol *names* that appear at the top level of
    ``expr`` (i.e. under arithmetic / boolean operators only, not nested
    inside any :class:`~dace.symbolic.Subscript` or function call).

    A :class:`Subscript` represents an indirect / gather access -- its
    inner symbols are gather-index inputs, not stride contributors --
    so recursion stops at that boundary.
    """
    if expr is None:
        return set()
    if isinstance(expr, sympy.Symbol):
        return {str(expr)}
    # Subscript and any function call are stop-boundaries: their symbols
    # are nested (gather-index inputs / function arguments), not direct.
    if isinstance(expr, symbolic.Subscript):
        return set()
    if isinstance(expr, sympy.Function):
        return set()
    args = getattr(expr, 'args', None)
    if not args:
        return set()
    result: Set[str] = set()
    for arg in args:
        result |= _direct_symbols(arg)
    return result


def _tilevar_direct_in(expr, tile_widths: Dict[str, int]) -> Optional[str]:
    """Return a tile iter-var name that appears as a direct top-level
    symbol in ``expr``, or ``None`` if none does (including the case
    where a tile-var only appears nested inside a Subscript)."""
    try:
        sym = symbolic.pystr_to_symbolic(str(expr))
    except Exception:
        return None
    direct = _direct_symbols(sym)
    for name in tile_widths:
        if name in direct:
            return name
    return None


def _is_point(lo, hi) -> bool:
    """Whether ``[lo : hi]`` is a single-element range."""
    try:
        return symbolic.simplify(hi - lo) == 0
    except Exception:
        return False


def widen_in_map_nsdfg_inputs(sdfg: SDFG, state: SDFGState, nsdfg_node: nodes.NestedSDFG,
                              tile_widths: Dict[str, int]) -> bool:
    """Tile-expand inner memlets of the body NSDFG ``nsdfg_node`` so each
    tile-var-bearing point dim grows from ``[expr : expr]`` to
    ``[expr : expr + W - 1]``. Operates entirely inside the NSDFG body --
    outer subsets / connector descriptors / interstate-edge bindings are
    left untouched (:class:`ExpandNestedSDFGInputs` is the upstream pass
    that normalizes those).

    :param sdfg: The parent SDFG owning ``state`` (unused; kept for
        signature parity with prior versions).
    :param state: State containing ``nsdfg_node`` (unused; kept for
        signature parity).
    :param nsdfg_node: The body NestedSDFG node whose interior is
        rewritten.
    :param tile_widths: Mapping ``iter_var_name -> tile_width`` from
        :class:`TileDimSpec`. The iter-var names match the inner-body
        symbol names (identity-binding through ``symbol_mapping``).

    :returns: ``True`` if any memlet subset was modified.
    """
    if state.entry_node(nsdfg_node) is None:
        # Top-level NSDFG -- no parent map, nothing to tile-expand.
        return False

    inner = nsdfg_node.sdfg
    changed = False
    for istate in inner.all_states():
        for edge in istate.edges():
            mm = edge.data
            if mm is None:
                continue
            for attr in ("subset", "other_subset"):
                sub = getattr(mm, attr, None)
                if sub is None or not isinstance(sub, Range):
                    continue
                new_ranges = []
                modified = False
                for (lo, hi, stp) in sub.ranges:
                    tvar = _tilevar_direct_in(lo, tile_widths)
                    if tvar is not None and _is_point(lo, hi):
                        new_ranges.append((lo, lo + tile_widths[tvar] - 1, stp))
                        modified = True
                    else:
                        new_ranges.append((lo, hi, stp))
                if modified:
                    sub.ranges = new_ranges
                    changed = True
    return changed
