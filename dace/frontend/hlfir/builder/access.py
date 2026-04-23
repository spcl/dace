"""Memlet subset construction + access-node caching + indirect-index lifting.

The key helper is ``acc`` — a per-state cached access-node factory that keeps
the "live sink" for a given data name, so reads and writes across multiple
tasklets in the same state thread through one connected graph.

``build_memlet_index`` turns an ``AccessInfo`` (from the bridge) into a
DaCe-style subset, offsetting Fortran 1-based indices to 0-based and
resolving indirect-index expressions (``edge_idx[jc,1]``) against the
symbols minted by ``collect_indirect``.
"""
from __future__ import annotations

import re

_INDIRECT_RE = re.compile(r'^(\w+)\[([^\]]*)\]$')


def acc(builder, state, name: str):
    """Single access node for ``name`` in ``state``, reused across reads /
    writes.  Without this, every tasklet in the same state would fabricate
    its own disconnected access node, so a later read could not see the
    value produced by an earlier write in the same state.
    """
    cache = getattr(state, '_hlfir_access', None)
    if cache is None:
        cache = {}
        state._hlfir_access = cache
    node = cache.get(name)
    if node is None:
        node = state.add_access(name)
        cache[name] = node
    return node


def get_access(accesses: list, array_name: str, is_read: bool):
    """Return the matching ``AccessInfo`` (exact read/write match preferred)."""
    for ac in accesses:
        if ac.array_name == array_name:
            if is_read and ac.is_read:
                return ac
            if not is_read and ac.is_write:
                return ac
    for ac in accesses:
        if ac.array_name == array_name:
            return ac
    return None


def indirect_host(expr: str) -> str:
    """Given ``edge_idx[jc,1]`` return ``edge_idx``; empty for non-indirect."""
    m = _INDIRECT_RE.match(expr)
    return m.group(1) if m else ""


def collect_indirect(builder, assigns: list) -> dict:
    """Walk every access in ``assigns`` and mint a fresh SDFG symbol for
    each distinct indirect index expression.  Returns a map from the
    Fortran-style expression (``edge_idx[jc,1]``) to the symbol name."""
    out = {}
    for a in assigns:
        for ac in a.accesses:
            for expr in getattr(ac, 'index_exprs', None) or []:
                if '[' in expr and expr not in out:
                    out[expr] = f"_idx_{builder.nid()}"
    return out


def indirect_to_dace(builder, expr: str, iter_map: dict) -> str:
    """Convert ``arr[i,j]`` (Fortran 1-based) into DaCe's 0-based subscript
    form using the array's lower bounds and the current loop iter_map."""
    m = _INDIRECT_RE.match(expr)
    if not m:
        return expr
    arr, inner = m.group(1), m.group(2)
    info = builder.arrays.get(arr)
    lbs = info.lower_bounds if info else []
    parts = []
    for dim, raw in enumerate(p.strip() for p in inner.split(',')):
        lb = lbs[dim] if dim < len(lbs) else "1"
        parts.append(offset_index_token(raw, lb, iter_map))
    return f"{arr}[{', '.join(parts)}]"


def offset_index_token(tok: str, lb: str, iter_map: dict) -> str:
    """Apply lower-bound offset to a single index token (``jc`` or ``3``)."""
    try:
        lb_int = int(lb)
    except (TypeError, ValueError):
        lb_int = 1

    if tok.lstrip('-').isdigit():
        return str(int(tok) - lb_int)
    uid = iter_map.get(tok, tok)
    if lb_int == 0:
        return uid
    return f"{uid} - {lb_int}" if lb_int >= 0 else f"{uid} + {-lb_int}"


def build_memlet_index(builder, array_name: str, access, iter_map: dict, indirect_syms: dict = None) -> str:
    """Build a memlet subset, offsetting Fortran→DaCe indices and resolving
    indirect index expressions against their minted symbols."""
    indirect_syms = indirect_syms or {}
    arr = builder.arrays.get(array_name)
    lbs = arr.lower_bounds if arr else []
    if access is None:
        return ""
    exprs = list(access.index_exprs) if access.index_exprs else []
    ivars = list(access.index_vars)

    parts = []
    for dim, v in enumerate(ivars):
        lb = lbs[dim] if dim < len(lbs) else "1"
        expr = exprs[dim] if dim < len(exprs) else v

        # Indirect: use the minted symbol (holds the Fortran 1-based index).
        if '[' in expr and expr in indirect_syms:
            parts.append(offset_index_token(indirect_syms[expr], lb, iter_map))
            continue

        # Constant literal: subtract the lower bound directly.
        if expr.lstrip('-').isdigit():
            parts.append(offset_index_token(expr, lb, iter_map))
            continue

        uid = iter_map.get(v, v)

        if lb == "0":
            parts.append(uid)
        elif lb == "1":
            parts.append(f"{uid} - 1")
        else:
            try:
                lb_int = int(lb)
                if lb_int > 0:
                    parts.append(f"{uid} - {lb_int}")
                else:
                    parts.append(f"{uid} + {-lb_int}")
            except ValueError:
                parts.append(f"{uid} - {lb}")

    return ", ".join(parts)
