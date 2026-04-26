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

from dace.frontend.hlfir.builder.access import acc, build_memlet_index, get_access, indirect_host, rename_iters


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

    # One AccessInfo per occurrence, in the order buildExpr produced.
    reads_by_name = {}
    for ac in accesses:
        if ac.is_read and ac.array_name in r_arr:
            reads_by_name.setdefault(ac.array_name, []).append(ac)

    # Rewrite the RHS, replacing the Nth occurrence of each array name
    # with `_in_<name>_<N>`.  Longest-first guards against partial
    # matches between related names.
    occ = {nm: 0 for nm in r_arr}
    sorted_tokens = sorted(r_arr | r_scl, key=len, reverse=True)

    def rewrite(code: str) -> str:
        for nm in sorted_tokens:
            if nm in r_scl:
                code = re.sub(rf'\b{re.escape(nm)}\b', f'_in_{nm}', code)
                continue

            def sub(_m, _nm=nm):
                n = occ[_nm]
                occ[_nm] += 1
                return f"_in_{_nm}_{n}"

            code = re.sub(rf'\b{re.escape(nm)}\b', sub, code)
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
        for i, ac in enumerate(reads_by_name[nm]):
            ix = build_memlet_index(builder, nm, ac, iter_map, indirect_syms)
            state.add_edge(r, None, t, f"_in_{nm}_{i}", Memlet(f"{nm}[{ix}]"))

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
    cache = getattr(state, '_hlfir_access', None)
    is_self_update = (target in r_scl) or (target in reads_by_name)
    cached_has_readers = False
    if cache is not None and target in cache:
        cached_has_readers = state.out_degree(cache[target]) > 0
    if is_self_update or cached_has_readers:
        w = state.add_access(target)
        if cache is not None:
            cache[target] = w
    else:
        w = acc(builder, state, target)

    if target in builder.scalars:
        # Scalar target: no buildable index, subset is always element 0.
        state.add_edge(t, f"_out_{target}", w, None, Memlet(data=target, subset="0"))
    else:
        ac = get_access(accesses, target, is_read=False)
        ix = build_memlet_index(builder, target, ac, iter_map, indirect_syms)
        state.add_edge(t, f"_out_{target}", w, None, Memlet(f"{target}[{ix}]"))


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
    # Plain ``i = 0`` still reuses the cached node.
    if target in reads:
        a = state.add_access(target)
        cache = getattr(state, '_hlfir_access', None)
        if cache is not None:
            cache[target] = a
    else:
        a = acc(builder, state, target)
    state.add_edge(t, '_out', a, None, Memlet(data=target, subset='0'))
