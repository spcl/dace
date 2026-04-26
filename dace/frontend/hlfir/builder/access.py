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

# Process-level monotonic counter used to mint stable, grep-able names for
# inline indirection loads (``<arr>_at<gid>``).  Kept process-level rather
# than per-SDFG so multi-file runs don't re-issue the same name in two
# unrelated kernels — each lift gets a unique tag, easy to find in
# transcripts and SDFG dumps.
_INDIRECTION_GID_COUNTER = 0


def _next_indirection_gid() -> int:
    global _INDIRECTION_GID_COUNTER
    gid = _INDIRECTION_GID_COUNTER
    _INDIRECTION_GID_COUNTER += 1
    return gid


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
    each distinct *inline* indirect index expression.  Returns a map from
    the Fortran-style expression (``edge_idx[jc,1]``) to the symbol name.

    Naming: ``<arr>_at<gid>`` — the prefix carries the source array's
    Fortran name so the SDFG dump shows which load the symbol holds; the
    process-level monotonic ``gid`` disambiguates same-expression-different-
    call-site without us having to normalise the inner expression.
    """
    out = {}
    for a in assigns:
        for ac in a.accesses:
            for expr in getattr(ac, 'index_exprs', None) or []:
                if '[' in expr and expr not in out:
                    arr = indirect_host(expr) or "idx"
                    out[expr] = f"{arr}_at{_next_indirection_gid()}"
    return out


def array_read_to_dace_expr(builder, assign_node, iter_map: dict) -> str:
    """Render a scalar-target assign whose RHS is a single array read
    (``ci0 = icidx(je, jb, 1)``) as a DaCe-style indexed expression with
    Fortran→0-based offsets and ``iter_map`` remap applied.  Used to lift
    the assign onto an interstate-edge assignment so the loaded value
    becomes a live SDFG symbol the consuming tasklet's memlet can index
    by.  Falls back to ``assign_node.expr`` if there's no array read."""
    reads = [ac for ac in assign_node.accesses if ac.is_read and ac.array_name in builder.arrays]
    if not reads:
        return assign_node.expr
    ac = reads[0]
    arr = ac.array_name
    info = builder.arrays.get(arr)
    lbs = info.lower_bounds if info else []
    parts = []
    for dim, raw in enumerate(ac.index_exprs):
        lb = lbs[dim] if dim < len(lbs) else "1"
        parts.append(offset_index_token(raw.strip(), lb, iter_map))
    return f"{arr}[{', '.join(parts)}]"


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

        # Closed-form arithmetic expression from the bridge (e.g. the
        # assumed-shape rebase ``(1 - 3)`` on an aliased declare, or
        # ``arr(k-1)`` -> ``(k - 1)``).  The bridge prints the raw
        # Fortran iter name; rewrite it through ``iter_map`` so the
        # subset matches the LoopRegion's uniquified loop_var, then
        # apply the outer array's ``lb`` offset uniformly.
        if any(op in expr for op in "+-*/") or expr.startswith("("):
            parts.append(_apply_lb(_remap_iters(expr, iter_map), lb))
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


def _remap_iters(expr: str, iter_map: dict) -> str:
    """Replace each whole-word Fortran iter name in ``expr`` with its
    uniquified counterpart from ``iter_map``.  Used for closed-form
    arithmetic subscripts where the bridge printed the raw block-arg
    name; the plain-identifier path does this lookup elsewhere."""
    if not iter_map:
        return expr
    return re.sub(r"\b([A-Za-z_]\w*)\b", lambda m: iter_map.get(m.group(1), m.group(1)), expr)


def _apply_lb(expr: str, lb: str) -> str:
    """Render ``expr - lb`` keeping both int and symbolic ``lb`` forms
    readable.  ``lb`` can be a Python int (as string), a Fortran symbol,
    or ``"?"`` (use as-is)."""
    try:
        lb_int = int(lb)
    except (TypeError, ValueError):
        return f"({expr}) - {lb}"
    if lb_int == 0:
        return expr
    if lb_int > 0:
        return f"({expr}) - {lb_int}"
    return f"({expr}) + {-lb_int}"
