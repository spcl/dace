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


def resolve_section_alias(builder, array_name: str, access):
    """If ``array_name`` is a ``section_alias`` dummy (trivial section
    slice — full-range triplets + scalar drops only), return
    ``(source_name, spliced_access)`` where ``spliced_access`` carries
    the source's full index list (the dummy's ``index_exprs`` spliced
    into the dim-map placeholders).  Otherwise return
    ``(array_name, access)`` unchanged.

    The bridge records ``view_dim_map`` as one entry per source-array
    dim: ``"_d<N>"`` for surviving (triplet) dims with N = 0-based
    dummy-dim index, or a 1-based scalar expression for dropped dims.
    Splicing yields a Fortran-1-based index list that
    ``build_memlet_index`` then offsets uniformly.
    """
    v = builder.arrays.get(array_name)
    if v is None or getattr(v, 'role', '') != 'section_alias':
        return array_name, access
    src = v.view_source
    if access is None:
        return src, access
    from types import SimpleNamespace
    dummy_exprs = list(getattr(access, 'index_exprs', None) or [])
    dummy_vars = list(getattr(access, 'index_vars', None) or [])
    new_exprs, new_vars = [], []
    for slot in v.view_dim_map:
        if slot.startswith('_d'):
            try:
                d_idx = int(slot[2:])
            except ValueError:
                d_idx = len(new_exprs)
            new_exprs.append(dummy_exprs[d_idx] if d_idx < len(dummy_exprs) else '')
            new_vars.append(dummy_vars[d_idx] if d_idx < len(dummy_vars) else '')
        else:
            new_exprs.append(slot)
            new_vars.append('')
    spliced = SimpleNamespace(
        array_name=src,
        is_read=getattr(access, 'is_read', False),
        is_write=getattr(access, 'is_write', False),
        index_exprs=new_exprs,
        index_vars=new_vars,
    )
    return src, spliced


def acc(builder, state, name: str):
    """Single access node for ``name`` in ``state``, reused across reads /
    writes.  Without this, every tasklet in the same state would fabricate
    its own disconnected access node, so a later read could not see the
    value produced by an earlier write in the same state.

    View-alias entries (Fortran storage-association reshape — see
    ``extract_vars::view_source`` / ``view_subset``) get an additional
    source→view linking memlet auto-installed the first time they're
    accessed in a state.  The link tells DaCe codegen which slab of
    the source array the view points at; subsequent reads / writes
    of the view in the same state pass through to the source.
    """
    # Trivial section-slice dummies (``role == 'section_alias'``) have
    # no SDFG descriptor — every access routes through the source array
    # with indices spliced via ``view_dim_map``.  Redirect the access-
    # node lookup to the source.
    v_alias = builder.arrays.get(name)
    if v_alias is not None and getattr(v_alias, 'role', '') == 'section_alias':
        return acc(builder, state, v_alias.view_source)
    cache = getattr(state, '_hlfir_access', None)
    if cache is None:
        cache = {}
        state._hlfir_access = cache
    node = cache.get(name)
    if node is None:
        node = state.add_access(name)
        cache[name] = node
        v = builder.arrays.get(name)
        if v is not None and getattr(v, 'role', '') == 'view_alias' \
                and v.view_source and v.view_source in state.parent.arrays:
            from dace import Memlet
            src = v.view_source
            src_subset = ", ".join(v.view_subset)
            view_dims = [str(d) for d in state.parent.arrays[name].shape]
            view_subset = ", ".join(f"0:{d}" for d in view_dims)
            src_node = cache.get(src) or state.add_access(src)
            cache.setdefault(src, src_node)
            state.add_edge(src_node, None, node, None, Memlet(data=src, subset=src_subset, other_subset=view_subset))
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
    each distinct *inline* indirect index expression -- recursively, so
    nested forms like ``idx1[idx2[idx3[i]]]`` mint one symbol per
    indirection level.  Returns a map from the Fortran-style expression
    (``edge_idx[jc,1]``) to the symbol name.

    Iteration order in the returned dict matters: ``dict`` preserves
    insertion order, and we insert innermost-first so the caller can
    materialise interstate-edge assignments in the same order without
    forward references.

    Naming: ``<arr>_at<gid>`` -- the prefix carries the source array's
    Fortran name so the SDFG dump shows which load the symbol holds; the
    process-level monotonic ``gid`` disambiguates same-expression-different-
    call-site without us having to normalise the inner expression.
    """
    out: dict[str, str] = {}

    def _intern_recursive(expr: str):
        """Visit ``expr`` and intern any ``<arr>[...]`` substring with
        ``<arr>`` in ``builder.arrays``, innermost-first.  An expression
        like ``idx1[idx2[i]]`` produces two entries: first ``idx2[i]``
        (the inner load), then ``idx1[idx2[i]]`` (the outer load).
        """
        if not isinstance(expr, str) or '[' not in expr:
            return
        for start, end, arr, parts in find_array_subscripts(expr, builder.arrays):
            # Recurse into each part FIRST so inner indirections are
            # interned ahead of the enclosing one.
            for part in parts:
                _intern_recursive(part)
            sub = expr[start:end]
            if sub not in out:
                out[sub] = f"{arr}_at{_next_indirection_gid()}"

    for a in assigns:
        for ac in a.accesses:
            for expr in getattr(ac, 'index_exprs', None) or []:
                _intern_recursive(expr)
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


def _rewrite_inner_indirects(part: str, indirect_syms: dict) -> str:
    """In-place substitute every ``<arr>[...]`` substring of ``part`` with
    its minted symbol from ``indirect_syms``.  Recurses through the part
    walker so a part like ``idx2[idx3[i]]`` collapses to whichever
    minted symbol covers it (innermost first, then the outer).
    Returns ``part`` unchanged if no nested indirect appears.
    """
    if not isinstance(part, str) or '[' not in part:
        return part
    # Walk inside-out: keep replacing the first match whose substring is
    # in ``indirect_syms`` until no more replacements are possible.  We
    # don't reuse ``find_array_subscripts`` since we need indexes into
    # ``part`` (not into a parent expression) for slicing.
    arr_names = set(indirect_syms.keys())
    # Sort longest-first so a ``a[b[i]]`` form picks the outer first only
    # after the inner ``b[i]`` has been substituted.  But since we scan
    # innermost-first via the bracket walker each pass, longest doesn't
    # actually matter -- still, keep deterministic ordering.
    changed = True
    out = part
    while changed:
        changed = False
        # Iterate through every <arr>[...] substring in out, replace the
        # first whose exact substring is interned.
        for st in range(len(out)):
            if out[st] not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_':
                continue
            # Try to match an array name starting at st.
            j = st
            while j < len(out) and (out[j].isalnum() or out[j] == '_'):
                j += 1
            if j >= len(out) or out[j] != '[':
                continue
            # Walk balanced brackets to find end.
            depth = 1
            k = j + 1
            while k < len(out) and depth > 0:
                if out[k] == '[':
                    depth += 1
                elif out[k] == ']':
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            if depth != 0:
                break
            sub = out[st:k + 1]
            if sub in indirect_syms:
                out = out[:st] + indirect_syms[sub] + out[k + 1:]
                changed = True
                break  # restart the scan from the beginning
    return out


def indirect_to_dace(builder, expr: str, iter_map: dict, indirect_syms: dict | None = None) -> str:
    """Convert ``arr[i,j]`` (Fortran 1-based) into the uniform offset-
    symbol DaCe subscript form.  Robust to nested brackets via the
    bracket-balanced walker.

    When ``indirect_syms`` is supplied, every nested ``<inner>[...]``
    substring inside an index part is first replaced by its minted
    symbol -- so ``idx1[idx2[i]]`` becomes ``idx1[(idx2_at2) - offset_idx1_d0]``
    rather than dragging the raw inner ``idx2[i]`` text into the SDFG
    symbolic expression (where DaCe's sympy parser misreads ``[]`` and
    falls back to a function-call shape that the C++ codegen can't
    compile).
    """
    if not isinstance(expr, str) or '[' not in expr:
        return expr
    matches = list(find_array_subscripts(expr, builder.arrays))
    # Single full-string match -- the typical inline-indirection shape.
    if len(matches) == 1:
        start, end, arr, parts = matches[0]
        if start == 0 and end == len(expr):
            if indirect_syms:
                parts = [_rewrite_inner_indirects(p, indirect_syms) for p in parts]
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
        #
        # ``v`` is the bridge-side ``index_vars[dim]`` value —
        # produced by ``resolveIndex(idx)`` which returns:
        #   * the loop-iter name for a tracked block-arg iter
        #     (``i``, ``je``, …); ``iter_map`` folds the SSA rename.
        #   * ``traceToDecl(idx)`` as a fallback for everything
        #     else, including ``fir.load %dgt(%c)`` where ``%dgt``
        #     designates a flattened struct-member array — that
        #     trace returns the WHOLE array's name (``ind_indices``
        #     for ``fir.load %ind_indices_decl(%c1)``).
        #
        # The whole-array name is NOT a valid memlet index.  The
        # authoritative form lives on ``index_exprs[dim]`` (``expr``):
        # the bridge has already lifted the load to either a
        # ``__sym_<arr>_<n>`` symbol (constant-indexed read of a
        # read-only array) or to ``<arr>[idx]`` form (which the
        # caller has further folded via the indirect machinery).
        #
        # Defaulting the iter_map fallback to ``expr`` instead of
        # ``v`` keeps tracked iters working (iter_map.get(v, …)
        # finds them) while letting non-iter v values fall through
        # to the richer ``expr`` rendering.  See the matching
        # ``internPosSymbol`` mutability gate in
        # ``bridge/ast/assigns.cpp`` — the two together close the
        # ``arr(struct_member(const))`` indirect-index path
        # exercised by ``long_tasklet_test``.
        uid = iter_map.get(v, expr)
        parts.append(f"{uid} - {offset_sym}")

    return ", ".join(parts)
