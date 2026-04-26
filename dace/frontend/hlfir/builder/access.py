"""Memlet subset construction + access-node caching + indirect-index lifting.

The key helper is ``acc`` — a per-state cached access-node factory that keeps
the "live sink" for a given data name, so reads and writes across multiple
tasklets in the same state thread through one connected graph.

``build_memlet_index`` turns an ``AccessInfo`` (from the bridge) into a
DaCe-style subset, offsetting Fortran 1-based indices to 0-based and
resolving indirect-index expressions (``edge_idx[jc,1]``) against the
symbols minted by ``collect_indirect``.

Three small primitives anchor the Fortran→DaCe subscript conversion and
are reused everywhere that emits a subset string:

  * ``rename_iters(expr, iter_map)`` — whole-word substitution of Fortran
    iter names with their uniquified DaCe counterparts.
  * ``_remap_token(token, iter_map)`` — single-subscript rewrite that
    handles literal / arithmetic / bare-identifier tokens uniformly.
  * ``_format_offset_subset(arr, parts)`` — wrap a per-dim list in the
    uniform ``arr[(p0) - offset_arr_d0, …]`` form.
"""
from __future__ import annotations

import re

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


def rename_iters(expr, iter_map):
    """Whole-word substitution of Fortran iter names with their
    uniquified DaCe counterparts.  Word boundaries protect against
    partial matches inside identifiers (``i`` shouldn't rewrite
    inside ``input1``).  Pass-through for ``None`` / non-string."""
    if not iter_map or not isinstance(expr, str):
        return expr
    return re.sub(r"\b([A-Za-z_]\w*)\b", lambda m: iter_map.get(m.group(1), m.group(1)), expr)


def _remap_token(token, iter_map):
    """Rewrite a single subscript token through ``iter_map``.  Three
    forms collapse to one helper: integer literals pass through;
    arithmetic / parenthesised expressions go through whole-word
    substitution; bare identifiers do a direct dict lookup."""
    token = token.strip()
    if token.lstrip('-').isdigit():
        return token
    if any(op in token for op in "+-*/") or token.startswith("("):
        return rename_iters(token, iter_map)
    return iter_map.get(token, token)


def _format_offset_subset(arr, parts):
    """Wrap a per-dim expression list in the uniform offset-symbol
    form: ``arr[(p0) - offset_arr_d0, (p1) - offset_arr_d1, …]``."""
    items = ", ".join(f"({p}) - {_offset_token(arr, d)}" for d, p in enumerate(parts))
    return f"{arr}[{items}]"


def find_array_subscripts(expr, names):
    """Generator yielding ``(start, end, arr_name, parts)`` for each
    top-level ``<arr>[…]`` substring in ``expr`` whose ``<arr>`` is in
    ``names``.  Walks brackets balanced (handles nested subscripts
    like ``a[idx[i],j]``) and splits the inner range on top-level
    commas only.  Replaces the brittle ``^(\\w+)\\[([^\\]]*)\\]$``
    regex used by indirect_to_dace / indirect_host."""
    n = len(expr)
    i = 0
    while i < n:
        m = re.match(r'([A-Za-z_]\w*)\[', expr[i:])
        if not m:
            i += 1
            continue
        arr = m.group(1)
        if arr not in names:
            i += 1
            continue
        start = i
        inner_start = i + len(arr) + 1
        depth = 1
        j = inner_start
        while j < n and depth > 0:
            ch = expr[j]
            if ch in '([{':
                depth += 1
            elif ch in ')]}':
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth != 0:
            return  # unbalanced; bail
        inner = expr[inner_start:j]
        # Split top-level commas only.
        parts, d, sp = [], 0, 0
        for k, ch in enumerate(inner):
            if ch in '([{':
                d += 1
            elif ch in ')]}':
                d -= 1
            elif ch == ',' and d == 0:
                parts.append(inner[sp:k].strip())
                sp = k + 1
        parts.append(inner[sp:].strip())
        yield (start, j + 1, arr, parts)
        i = j + 1


def indirect_host(expr):
    """Given ``edge_idx[jc,1]`` return ``edge_idx``; empty for non-indirect.
    Robust to nested brackets via the bracket-balanced walker."""
    if not isinstance(expr, str) or '[' not in expr:
        return ""
    m = re.match(r'^([A-Za-z_]\w*)\[', expr)
    return m.group(1) if m and expr.endswith(']') else ""


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


def _offset_token(arr: str, dim: int) -> str:
    """The per-axis offset-symbol name every memlet subtracts."""
    return f"offset_{arr}_d{dim}"


def array_read_to_dace_expr(builder, assign_node, iter_map: dict) -> str:
    """Render a scalar-target assign whose RHS is a single array read
    (``ci0 = icidx(je, jb, 1)``) as a DaCe-style indexed expression
    with the uniform offset-symbol form
    (``arr[(idx) - offset_arr_d<i>, …]``).  Used to lift the assign
    onto an interstate-edge so the loaded value becomes a live SDFG
    symbol the consuming tasklet's memlet can index by.  Falls back
    to ``assign_node.expr`` if there's no array read."""
    reads = [ac for ac in assign_node.accesses if ac.is_read and ac.array_name in builder.arrays]
    if not reads:
        return assign_node.expr
    ac = reads[0]
    parts = [_remap_token(raw, iter_map) for raw in ac.index_exprs]
    return _format_offset_subset(ac.array_name, parts)


def indirect_to_dace(builder, expr: str, iter_map: dict) -> str:
    """Convert ``arr[i,j]`` (Fortran 1-based) into the uniform offset-
    symbol DaCe subscript form.  Robust to nested brackets via the
    bracket-balanced walker."""
    if not isinstance(expr, str) or '[' not in expr:
        return expr
    matches = list(find_array_subscripts(expr, builder.arrays))
    # Single full-string match — the typical inline-indirection shape.
    if len(matches) == 1:
        start, end, arr, parts = matches[0]
        if start == 0 and end == len(expr):
            return _format_offset_subset(arr, [_remap_token(p, iter_map) for p in parts])
    return expr


def build_memlet_index(builder, array_name: str, access, iter_map: dict, indirect_syms: dict = None) -> str:
    """Build a memlet subset using the uniform offset-symbol form.

    For every dim of ``array_name``, the resulting subset token is
    ``(<fortran-1-based-expr>) - offset_<arr>_d<i>`` — the offset symbol
    was declared in ``add_descriptors`` and gets folded by
    ``sdfg.specialize`` at the end of ``build()`` (default value ``1``
    collapses the form to ``expr - 1``).  Indirect-index symbols are
    used in place of the index_expr but otherwise follow the same
    form.

    Constants stay outside the subtraction: ``A(3)`` produces
    ``3 - offset_A_d0`` (sympy folds to ``2`` once specialise runs).

    Arrays not in ``builder.arrays`` (struct members the bridge
    synthesises ad hoc, etc.) fall back to a literal ``- 1`` so the
    memlet still validates — this matches the pre-symbolic behaviour
    for those cases and avoids a missing-symbol crash at specialise
    time.
    """
    indirect_syms = indirect_syms or {}
    arr = builder.arrays.get(array_name)
    if access is None:
        return ""
    exprs = list(access.index_exprs) if access.index_exprs else []
    ivars = list(access.index_vars)
    rank = max(len(ivars), len(exprs))

    has_offset_sym = arr is not None
    parts = []
    for dim in range(rank):
        v = ivars[dim] if dim < len(ivars) else ""
        expr = exprs[dim] if dim < len(exprs) else v
        offset_sym = f"offset_{array_name}_d{dim}" if has_offset_sym else "1"

        # Indirect: substitute the minted symbol that holds the
        # Fortran 1-based runtime value, then offset uniformly.
        if '[' in expr and expr in indirect_syms:
            tok = indirect_syms[expr]
            parts.append(f"({tok}) - {offset_sym}")
            continue

        # Closed-form arithmetic: remap the iter names through the
        # current LoopRegion's uniquified iter_map, then offset.
        if any(op in expr for op in "+-*/") or expr.startswith("("):
            parts.append(f"({rename_iters(expr, iter_map)}) - {offset_sym}")
            continue

        # Constant literal: keep as-is, offset symbolically (sympy
        # folds it to the right value after specialise).
        if expr.lstrip('-').isdigit():
            parts.append(f"{expr} - {offset_sym}")
            continue

        # Bare iter name: remap through iter_map, then offset.
        uid = iter_map.get(v, v)
        parts.append(f"{uid} - {offset_sym}")

    return ", ".join(parts)
