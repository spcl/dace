"""Tasklet emission — both the full per-occurrence-connector path and the
flat ``emit_scalar_assign`` fast path.

``emit_tasklet`` is the heart of the frontend: takes an ``AssignNode``
from the bridge and produces a tasklet whose RHS expression has each
array-read occurrence wired to its own input connector with the right
memlet subset.  That per-occurrence wiring is what lets
``e_bln(jc,1)*z(...) + e_bln(jc,2)*z(...)`` produce three distinct
memlets instead of collapsing to one.

``emit_scalar_assign`` is the flush path for plain scalar assigns queued
on ``ctx.pending`` (``i = i + 1``, ``c = 0.5``) — simpler, no memlet
index math needed.

``assign_reads_array`` is the predicate used by ``emit_assign`` to
decide which path an incoming assign should take.
"""
from __future__ import annotations

import re

from dace import Memlet

from dace.frontend.hlfir.builder.access import (acc, build_memlet_index, get_access, indirect_host, rename_iters,
                                                resolve_section_alias)


def _ensure_view_writeback_link(builder, state, write_node, target: str):
    """When the Phase I self-update split creates a fresh access node
    for the write side of a view alias, the view's required source
    linking memlet (added by ``acc()`` on the read side) is missing
    on this new node — DaCe's ``get_view_edge`` returns None and
    validation fails.  Mirror the link in the writeback direction
    (view -> source) so the write node is also a recognised view edge.

    Two extra structural rules to keep ``get_view_edge`` happy:

    * Use a FRESH source access node for the writeback (not the
      cached one), otherwise ``src -> view_read -> tasklet ->
      view_write -> src`` forms a cycle on the source.
    * Drop ``target`` from the per-state cache after adding the
      writeback.  Subsequent reads in the same state then mint a new
      read-side view (with its own ``src -> view`` linking) instead
      of pulling through this write-side node — leaving the write
      node with the clean ``in=1 (tasklet) / out=1 (writeback)``
      shape ``get_view_edge`` recognises, and the read-side with the
      clean ``in=1 (linking) / out=N (tasklets)`` shape.
    """
    v = builder.arrays.get(target)
    if v is None or getattr(v, 'role', '') != 'view_alias':
        return
    if not v.view_source or v.view_source not in state.parent.arrays:
        return
    src = v.view_source
    src_subset = ", ".join(v.view_subset)
    view_dims = [str(d) for d in state.parent.arrays[target].shape]
    view_subset = ", ".join(f"0:{d}" for d in view_dims)
    src_node = state.add_access(src)
    cache = getattr(state, '_hlfir_access', None)
    if cache is not None:
        cache[src] = src_node
        cache.pop(target, None)
    state.add_edge(write_node, None, src_node, None, Memlet(data=src, subset=src_subset, other_subset=view_subset))


def assign_reads_array(assign_node, arrays: dict) -> bool:
    """True iff any ``accesses`` entry on ``assign_node`` is a read against
    an array descriptor.  Used to promote a nominally-scalar assign
    (``s = d(i) + 1``) onto the per-occurrence-connector tasklet path so
    the array read gets a real memlet instead of a bare identifier in
    the code string.
    """
    for ac in assign_node.accesses:
        if ac.is_read and ac.array_name in arrays:
            return True
    return False


def emit_tasklet(builder, state, assign_node, idx: int, iter_map: dict, indirect_syms: dict = None):
    """One Tasklet per array assignment.

    Expressions like ``e_bln(jc,1)*z_kin(...) + e_bln(jc,2)*z_kin(...)``
    access the same array at several positions.  Each *occurrence* in
    the RHS becomes its own tasklet input connector so every access
    carries the correct memlet; otherwise the generated code would
    collapse all three terms onto a single connector and silently
    compute a wrong result.
    """
    indirect_syms = indirect_syms or {}
    accesses = assign_node.accesses

    tokens = set(re.findall(r'[a-zA-Z_]\w*', assign_node.expr))
    r_arr = tokens & set(builder.arrays)
    r_scl = tokens & set(builder.scalars)
    target = assign_node.target

    # Index arrays (e.g. edge_idx) show up in the RHS token scan but we
    # move their values onto the interstate edge as symbols.
    indirect_arrays = {indirect_host(expr) for expr in indirect_syms}
    r_arr -= indirect_arrays

    # One AccessInfo per textual occurrence, in the order buildExpr
    # produced.  We mint ONE connector + ONE memlet per occurrence —
    # even when two occurrences read the same array element.  Sharing
    # connectors (dedup) used to misalign textual-occurrence-to-access
    # mapping when the bridge's accesses list and the textual expr
    # disagree on count (e.g., the MIN/MAX cmp+select pattern), so
    # the simpler 1:1 mapping is the contract now.
    reads_by_name = {}
    for ac in accesses:
        if ac.is_read and ac.array_name in r_arr:
            reads_by_name.setdefault(ac.array_name, []).append(ac)

    # Rewrite the RHS, replacing the Nth occurrence of each array name
    # (and consuming its balanced ``[...]`` subscript) with
    # ``_in_<name>_<N>``.  Scalars get a single bare-name connector
    # ``_in_<name>`` since they don't carry a subscript.
    occ = {nm: 0 for nm in r_arr}
    sorted_tokens = sorted(r_arr | r_scl, key=len, reverse=True)

    def rewrite(code: str) -> str:
        for nm in sorted_tokens:
            if nm in r_scl:
                code = re.sub(rf'\b{re.escape(nm)}\b', f'_in_{nm}', code)
                continue
            # Array references in the bridge-emitted RHS come with their
            # subscript: ``zsolqa[(i)-1, (j)-1, (k)-1]``.  Replace the
            # whole ``name[...]`` group with the connector — the
            # connector's memlet already targets that one element, so
            # leaving the subscript on it would surface as DaCe's
            # "Subscript ... contains an invalid number of dimensions"
            # validator error.  Walk balanced brackets manually since
            # ``re`` can't.
            new_chunks = []
            cursor = 0
            pat = re.compile(rf'\b{re.escape(nm)}\b')
            for m in pat.finditer(code):
                start = m.start()
                end = m.end()
                # If the very next char is '[', consume the balanced [...].
                if end < len(code) and code[end] == '[':
                    depth = 1
                    j = end + 1
                    while j < len(code) and depth > 0:
                        ch = code[j]
                        if ch in '([{':
                            depth += 1
                        elif ch in ')]}':
                            depth -= 1
                            if depth == 0:
                                break
                        j += 1
                    if depth == 0:
                        end = j + 1
                new_chunks.append(code[cursor:start])
                n = occ[nm]
                occ[nm] += 1
                new_chunks.append(f"_in_{nm}_{n}")
                cursor = end
            new_chunks.append(code[cursor:])
            code = ''.join(new_chunks)
        return code

    in_c = {f"_in_{sc}" for sc in r_scl}
    for nm, acs in reads_by_name.items():
        for i in range(len(acs)):
            in_c.add(f"_in_{nm}_{i}")
    out_c = {f"_out_{target}"}

    # Apply iter_map rename to bare symbol references in the RHS BEFORE
    # the array/scalar connector rewrite.  An assign like
    # ``d(i) = i*2.0`` inside a ``do i = 50, 54`` loop produces RHS
    # ``i * 2.0`` in the AST; the LoopRegion's iter is ``i_0`` (after
    # uniquification), but the SDFG-level symbol ``i`` may also exist
    # (a separate dummy with the same Fortran name, or just the
    # extract_vars symbol slot).  Without this rename, the tasklet
    # binds ``i`` to whatever the SDFG-level ``i`` symbol holds —
    # typically zero — instead of the per-iteration value ``i_0``.
    expr = rename_iters(assign_node.expr, iter_map)
    code = f"_out_{target} = {rewrite(expr)}"
    t = state.add_tasklet(f"t_{idx}", in_c, out_c, code)

    for nm in sorted(reads_by_name):
        r = acc(builder, state, nm)
        # One edge per textual occurrence (1:1 with connector names).
        # Section-alias dummies route through the source array with
        # indices spliced via ``view_dim_map``; the dummy itself has
        # no SDFG descriptor.
        for i, ac in enumerate(reads_by_name[nm]):
            eff_nm, eff_ac = resolve_section_alias(builder, nm, ac)
            ix = build_memlet_index(builder, eff_nm, eff_ac, iter_map, indirect_syms)
            state.add_edge(r, None, t, f"_in_{nm}_{i}", Memlet(f"{eff_nm}[{ix}]"))

    for sc in sorted(r_scl):
        r = acc(builder, state, sc)
        state.add_edge(r, None, t, f"_in_{sc}", Memlet(data=sc, subset="0"))

    # ----------------------------------------------------------------
    # Pick the write-side access node for the tasklet's output edge.
    # ----------------------------------------------------------------
    #
    # An SDFGState is a DAG of AccessNodes and Tasklets.  For each data
    # name we keep ONE "live sink" in ``state._hlfir_access[name]`` —
    # the access node that subsequent reads from that name should pull
    # from (because it holds the latest write).  Two rules govern
    # whether a new write reuses that sink or allocates a fresh one:
    #
    # 1. A write that is paired with a read of the SAME name in the
    #    SAME tasklet must target a NEW access node, not the one the
    #    read came from.  Otherwise the tasklet would have both an
    #    incoming and outgoing edge on the same node — a cycle — and
    #    DaCe's state validator would reject it.  Fortran patterns
    #    that trigger this: ``i = i + 1``, ``d(1) = d(1) * 2.0``,
    #    ``temp = min(d(1), temp)``.
    #
    # 2. A write whose cached sink has ALREADY been read by a later
    #    tasklet in this state must also get a new access node.
    #    Sharing it would let the DAG scheduler reorder the new write
    #    before the earlier read, changing observable data.
    #
    # Otherwise — pure write-only update over the latest sink — reuse
    # the cached access node.  Multiple in-edges to one node are legal;
    # sharing keeps the data-flow graph connected.
    # For section-alias targets, the write retargets to the source
    # array — the dummy has no SDFG descriptor.  Read-side reads also
    # route through the source name, so cache / self-update bookkeeping
    # must use the source name.
    v_target = builder.arrays.get(target)
    eff_target = target
    if v_target is not None and getattr(v_target, 'role', '') == 'section_alias':
        eff_target = v_target.view_source
    cache = getattr(state, '_hlfir_access', None)
    is_self_update = (target in r_scl) or (target in reads_by_name) \
                  or (eff_target in reads_by_name)
    cached_has_readers = False
    if cache is not None and eff_target in cache:
        cached_has_readers = state.out_degree(cache[eff_target]) > 0
    if is_self_update or cached_has_readers:
        w = state.add_access(eff_target)
        if cache is not None:
            cache[eff_target] = w
        _ensure_view_writeback_link(builder, state, w, eff_target)
    else:
        w = acc(builder, state, eff_target)

    if target in builder.scalars:
        # Scalar target: no buildable index, subset is always element 0.
        state.add_edge(t, f"_out_{target}", w, None, Memlet(data=target, subset="0"))
    else:
        ac = get_access(accesses, target, is_read=False)
        eff_nm, eff_ac = resolve_section_alias(builder, target, ac)
        ix = build_memlet_index(builder, eff_nm, eff_ac, iter_map, indirect_syms)
        state.add_edge(t, f"_out_{target}", w, None, Memlet(f"{eff_nm}[{ix}]"))


def emit_scalar_assign(builder, state, target: str, value: str):
    """Tasklet for ``target = value`` on a scalar target.

    Inputs are derived from the identifier tokens that appear in
    ``value`` — every one that names an SDFG scalar gets its own
    input connector so the tasklet can read ``i`` for ``i = i + 1``
    and similar self-updates.
    """
    value = str(value)
    tokens = set(re.findall(r'[a-zA-Z_]\w*', value))
    # ``nm != target`` was wrong — ``i = i + 1`` genuinely needs a read
    # edge on the target itself.
    reads = [nm for nm in sorted(tokens, key=len, reverse=True) if nm in builder.scalars]

    code = value
    for nm in reads:
        code = re.sub(rf'\b{re.escape(nm)}\b', f'_in_{nm}', code)

    in_c = {f"_in_{nm}" for nm in reads}
    out_c = {'_out'}
    t = state.add_tasklet(f"set_{target}", in_c, out_c, f"_out = {code}")

    for nm in reads:
        r = acc(builder, state, nm)
        state.add_edge(r, None, t, f"_in_{nm}", Memlet(data=nm, subset='0'))

    # Self-update (``i = i + 1``): the read and write need DIFFERENT
    # access nodes so the state remains a DAG — ``Access(i_read) →
    # Tasklet → Access(i_write)`` instead of a cycle on one node.
    # Same rule applies (Phase I) when an EARLIER tasklet in the same
    # state already read ``target`` through the cached access node —
    # reusing it for our write would put both an in-edge and an
    # out-edge on the same node, creating the same cycle.  Velocity-
    # tendencies surfaces this with the two-line pattern
    # ``max_vcfl_dyn = MAX(p_diag%max_vcfl_dyn, ...)``
    # ``p_diag%max_vcfl_dyn = max_vcfl_dyn``: the second assign's
    # writeback target was the first assign's RHS read.
    cache = getattr(state, '_hlfir_access', None)
    cached_has_readers = (cache is not None and target in cache and state.out_degree(cache[target]) > 0)
    if (target in reads) or cached_has_readers:
        a = state.add_access(target)
        if cache is not None:
            cache[target] = a
        _ensure_view_writeback_link(builder, state, a, target)
    else:
        a = acc(builder, state, target)
    state.add_edge(t, '_out', a, None, Memlet(data=target, subset='0'))
