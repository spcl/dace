# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lowering of :class:`~dace.sdfg.memlet_access_policies.loop_cursor.LoopCursor` policies to ordinary SDFG constructs
(called by :class:`~dace.transformation.passes.memlet_access_policy_lowering.LowerMemletAccessPolicies` through
:meth:`LoopCursor.lower <dace.sdfg.memlet_access_policies.loop_cursor.LoopCursor.lower>`).

:func:`lower_loop_cursors` materializes one loop-carried integer *cursor symbol* per cursor class (assigned in the
loop's init statement, advanced in its update statement), a flat :class:`~dace.data.Reference` per array set once at
SDFG entry, and rewrites each memlet to ``flat[cursor + immediate]`` (or, for non-contiguous reads, to a
per-iteration *window* reference). A cursor of a loop in an enclosing SDFG is passed to the nested SDFGs on the way
as a symbol, and the flat reference is set at the entry of the memlet's own SDFG. Code generation then emits the loop
as ``for (i = ..., cur = ...; ...; i = i + 1, cur = cur + step)`` and every access as ``flat[cur + imm]``, with no
policy-specific code paths. See :mod:`~dace.transformation.passes.memlet_access_policies.assign_loop_cursors` for
the offset decomposition (cursor classes, immediates, anchors) this lowering relies on.
"""

import re
import warnings

import sympy as sp

from dace import SDFG, SDFGState, dtypes, symbolic
from dace import data as dt
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.memlet_access_policies import CopyOnAccess, LoopCursor
from dace.sdfg.state import LoopRegion
from dace.subsets import Range
from dace.transformation import gpu_helpers
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.memlet_access_policies.assign_loop_cursors import (
    OffsetDecomposition,
    decompose_on_path,
)
from dace.transformation.passes.memlet_access_policies.helpers import (
    GlobalPath,
    enclosing_loops,
    flat_addressable_array,
    flat_length,
    inner_symbols,
    path_root,
    symbol_named,
    symbol_names,
)

_INT32_MAX_BYTES = 2**31
_INIT_STATE_LABEL = "__dace_memlet_access_policy_init"


def _extent_bytes_int32(desc: dt.Data) -> bool:
    """True iff the array's total byte extent is provably below 2**31 (so element offsets fit an int32)."""
    try:
        total = sp.sympify(desc.total_size) * desc.dtype.bytes
        return bool(total.is_number and int(total) < _INT32_MAX_BYTES)
    except (TypeError, ValueError):
        return False


def _invariant_part(expr: sp.Basic, inner: set[str]) -> sp.Basic:
    """The terms of ``expr`` that do not depend on symbols defined inside the loop body."""
    return sp.Add(*[t for t in sp.Add.make_args(sp.expand(expr)) if not (symbol_names(t) & inner)])


def _choose_anchor(candidates: list[sp.Basic]) -> sp.Basic:
    """Pick the offset a cursor class is anchored at, among its members' nest-invariant offsets: the member that
    is smallest in every monomial coefficient if there is one (e.g. the first element of a window; every access
    then carries a non-negative immediate), otherwise the coefficient-wise (lower) median, which centres a
    stencil neighbourhood so that the immediates are the +-stride differences. Deterministic."""
    coeffs: list[dict[sp.Basic, sp.Basic]] = []
    for cand in candidates:
        d: dict[sp.Basic, sp.Basic] = {}
        for term in sp.Add.make_args(sp.expand(cand)):
            coeff, monomial = term.as_coeff_Mul()
            d[monomial] = d.get(monomial, sp.Integer(0)) + coeff
        coeffs.append(d)
    monomials = set().union(*coeffs)
    columns = {m: sorted((d.get(m, sp.Integer(0)) for d in coeffs), key=float) for m in monomials}
    lowest = sp.Add(*[col[0] * m for m, col in columns.items()])
    if any(sp.expand(cand - lowest) == 0 for cand in candidates):
        return lowest
    n = len(candidates)
    return sp.Add(*[col[(n - 1) // 2] * m for m, col in columns.items()])


def _append_statement(loop: LoopRegion, which: str, statement: str) -> None:
    """Append ``statement`` to the loop's init or update code block (code blocks may hold several statements)."""
    block: CodeBlock | None = getattr(loop, which)
    code = statement if block is None else f"{block.as_string}\n{statement}"
    setattr(loop, which, CodeBlock(code))


def _pystr(expr) -> str:
    return symbolic.symstr(expr, cpp_mode=False)


class _CursorTable:
    """Cursor symbols of one SDFG during lowering: creation (symbol registration + loop statements), sharing per
    cursor class and chaining across loop levels."""

    def __init__(self, sdfg: SDFG, chain_outer_loops: bool):
        self.sdfg = sdfg
        self.chain = chain_outer_loops
        self.classes: dict[tuple, tuple[symbolic.symbol, sp.Basic]] = {}  #: class key -> (cursor symbol, anchor)
        self.used: set[str] = set(sdfg.symbols.keys())
        self.created = 0

    @staticmethod
    def class_key(
        loop: LoopRegion, array: str, dtype: dtypes.typeclass, share_key: str | None, class_base: sp.Basic
    ) -> tuple:
        """Cursor-class identity: same loop, array, cursor type, override key, per-iteration step and tracked base
        without the loop-variable term. Members of a class differ only by a loop-invariant immediate."""
        v = symbol_named(class_base, loop.loop_variable)
        delta = sp.expand(sp.diff(class_base, v)) if v is not None else sp.Integer(0)
        step = sp.expand(delta * loop_analysis.get_loop_stride(loop))
        base_wo_v = sp.expand(class_base - (v * delta if v is not None else 0))
        return (loop.label, array, dtype, share_key, str(step), str(base_wo_v))

    def cursor_for(
        self,
        loop: LoopRegion,
        array: str,
        dtype: dtypes.typeclass,
        class_base: sp.Basic,
        anchor: sp.Basic,
        share_key: str | None,
    ) -> tuple[symbolic.symbol, sp.Basic]:
        """Return (creating if needed) the cursor symbol of ``loop`` that tracks ``class_base`` -- an expression
        affine in the loop variable and free of symbols defined inside the loop body -- together with the
        nest-invariant ``anchor`` the cursor additionally holds (the anchor requested here if the cursor is
        created now, the existing cursor's anchor otherwise; callers add the difference to their immediates).
        The symbol is the typed DaCe symbol registered in the SDFG; callers use it directly in expressions.

        Chaining: the cursor's entry value ``class_base(v_0) + anchor`` is itself an address that may be affine in
        an enclosing loop's variable. Instead of recomputing it at every entry of ``loop`` (a multiply-add per
        outer iteration), an outer cursor tracking that expression is created (or shared) on the enclosing loop
        and the inner cursor is initialized from it (``inner = outer``, or ``outer + const``). Applied recursively
        outward, so a loop nest walks its arrays with one add per loop level and no multiplies, without assuming
        anything about inner trip counts.
        """
        key = self.class_key(loop, array, dtype, share_key, class_base)
        if key in self.classes:
            return self.classes[key]

        var = loop.loop_variable
        v = symbol_named(class_base, var)
        delta = sp.expand(sp.diff(class_base, v)) if v is not None else sp.Integer(0)
        step = sp.expand(delta * loop_analysis.get_loop_stride(loop))
        init_value = symbolic.pystr_to_symbolic(loop_analysis.get_init_assignment(loop))
        init = sp.expand((class_base.subs(v, init_value) if v is not None else class_base) + anchor)
        # Split the entry value into the part that varies with an enclosing loop (chained to that loop's cursor)
        # and the nest-invariant remainder (a constant or a symbolic expression, added once per entry).
        outer_loops = enclosing_loops(loop)
        outer_vars = {outer.loop_variable for outer in outer_loops if outer.loop_variable}
        rest = sp.Add(*[t for t in sp.Add.make_args(init) if symbol_names(t) & outer_vars])
        const = sp.expand(init - rest)
        init_expr = init
        if self.chain and rest != 0:
            for outer in outer_loops:
                vo = symbol_named(rest, outer.loop_variable) if outer.loop_variable else None
                if vo is None:
                    continue
                d_out = sp.expand(sp.diff(rest, vo))
                if (
                    d_out.has(sp.Derivative)
                    or d_out.has(sp.floor)
                    or d_out.has(sp.ceiling)
                    or outer.loop_variable in symbol_names(d_out)
                    or (symbol_names(rest) & inner_symbols(outer))
                    or loop_analysis.get_init_assignment(outer) is None
                    or loop_analysis.get_loop_stride(outer) is None
                ):
                    break
                # The outer cursor absorbs the invariant remainder, so the inner cursor starts exactly at it.
                # A per-memlet ``share_key`` override applies to the memlet's own cursor only, so outer cursors
                # are shared by every nest that needs them.
                outer_cursor, outer_anchor = self.cursor_for(outer, array, dtype, rest, const, None)
                init_expr = sp.expand(outer_cursor + const - outer_anchor)
                break

        name = self._name(array, loop)
        self.sdfg.add_symbol(name, dtype)
        cursor = symbolic.symbol(name, dtype)
        _append_statement(loop, "init_statement", f"{name} = {_pystr(init_expr)}")
        _append_statement(loop, "update_statement", f"{name} = {_pystr(cursor + step)}")
        self.classes[key] = (cursor, anchor)
        self.created += 1
        return cursor, anchor

    def _name(self, array: str, loop: LoopRegion) -> str:
        """Deterministic identifier ``__dace_cur_<array>_<loop>`` (suffixed on collision, e.g. when one loop
        holds several cursor classes of the same array)."""
        base = re.sub(r"\W", "_", f"__dace_cur_{array}_{loop.label}")
        name, n = base, 1
        while name in self.used:
            name = f"{base}_{n}"
            n += 1
        self.used.add(name)
        return name


class _References:
    """Flat base references (one per array per SDFG, set once at SDFG entry) and per-memlet window references."""

    def __init__(self, sdfg: SDFG):
        self.sdfg = sdfg
        self.init_state: SDFGState | None = None
        self.flat: dict[str, str] = {}
        self.nodes: dict[tuple[SDFGState, str, bool], nodes.AccessNode] = {}
        self.windows = 0

    def flat_reference(self, array: str) -> str:
        """The flat reference of ``array`` (created and set at SDFG entry on first use)."""
        if array in self.flat:
            return self.flat[array]
        desc = self.sdfg.arrays[array]
        name = f"__dace_flat_{array}"
        if name not in self.sdfg.arrays:
            self.sdfg.add_reference(name, [desc.total_size], desc.dtype, storage=desc.storage)
            if self.init_state is None:
                start = self.sdfg.start_block
                if start.label == _INIT_STATE_LABEL and isinstance(start, SDFGState):
                    self.init_state = start
                else:
                    self.init_state = self.sdfg.add_state_before(start, _INIT_STATE_LABEL, is_start_block=True)
            # ``from_array`` covers the whole array starting at index ``-offset``, i.e. the set points the flat
            # reference at the array's physical element 0 (cursors are physical offsets, see ``base_offset``).
            self.init_state.add_edge(
                self.init_state.add_read(array),
                None,
                self.init_state.add_write(name),
                "set",
                Memlet.from_array(array, desc),
            )
        self.flat[array] = name
        return name

    def node(self, state: SDFGState, reference: str, is_read: bool) -> nodes.AccessNode:
        """One shared read (or write) access node of ``reference`` per state."""
        key = (state, reference, is_read)
        if key not in self.nodes:
            self.nodes[key] = state.add_read(reference) if is_read else state.add_write(reference)
        return self.nodes[key]

    def window(self, state: SDFGState, array: str, subset: Range, index: sp.Basic) -> nodes.AccessNode:
        """A window reference with the memlet's shape and the array's strides, set (in ``state``) to the flat
        reference at element ``index``; returns its access node."""
        desc = self.sdfg.arrays[array]
        flat = self.flat_reference(array)
        sizes = subset.size()
        total = sum((sp.sympify(s) - 1) * sp.sympify(st) for s, st in zip(sizes, desc.strides)) + 1
        name = f"__dace_win_{array}_{self.windows}"
        self.windows += 1
        self.sdfg.add_reference(name, sizes, desc.dtype, storage=desc.storage, strides=desc.strides, total_size=total)
        win = state.add_access(name)
        set_memlet = Memlet(data=flat, subset=Range([(index, index, 1)]))
        state.add_edge(self.node(state, flat, True), None, win, "set", set_memlet)
        return win


def _reroute(
    state: SDFGState, edge: MultiConnectorEdge[Memlet], new_root: nodes.AccessNode, is_read: bool, new_memlet: Memlet
) -> None:
    """Replace the memlet path of ``edge`` by one from/to ``new_root`` carrying ``new_memlet`` at the leaf
    (outer memlets are re-propagated through the scopes). The new path is added before the old one is removed,
    so scope nodes and the leaf connector are never orphaned in between."""
    path = state.memlet_path(edge)
    if is_read:
        old_root = path[0].src
        node_seq = [new_root] + [e.dst for e in path]
        src_conn, dst_conn = None, path[-1].dst_conn
    else:
        old_root = path[-1].dst
        node_seq = [e.src for e in path] + [new_root]
        src_conn, dst_conn = path[0].src_conn, None
    state.add_memlet_path(*node_seq, memlet=new_memlet, src_conn=src_conn, dst_conn=dst_conn, propagate=True)
    state.remove_memlet_path(edge, remove_orphans=True)
    if old_root in state.nodes() and state.degree(old_root) == 0:
        state.remove_node(old_root)


def lower_loop_cursors(
    sdfg: SDFG,
    entries: list[tuple[SDFGState, MultiConnectorEdge[Memlet]]],
    assume_int32: bool = False,
    chain_outer_loops: bool = True,
    **_,
) -> dict[str, int]:
    """Lower the :class:`~dace.sdfg.memlet_access_policies.loop_cursor.LoopCursor` policies of an SDFG and its nested
    SDFGs (see module docstring). A cursor is created in the SDFG of its loop and passed down, as a symbol, to the
    nested SDFGs along the memlet path of each memlet it addresses; flat references are created in the SDFG of the
    memlet.

    :param sdfg: The root SDFG.
    :param entries: ``(state, edge)`` pairs whose memlets carry ``LoopCursor`` policies, in any SDFG of the tree.
    :param assume_int32: Treat ``auto`` cursors as int32 even when the array extent is not provably < 2**31.
    :param chain_outer_loops: Initialize inner cursors from outer cursors (see :meth:`_CursorTable.cursor_for`).
    :return: ``{'cursors': n, 'memlets': m, 'dropped': d}`` (``memlets`` includes already-lowered ones).
    """
    tables: dict[SDFG, _CursorTable] = {}
    refs: dict[SDFG, _References] = {}
    lowered = dropped = 0
    kernel_dims: dict[nodes.MapEntry, gpu_helpers.KernelDimensions | None] = {}

    # Collect per loop, skipping memlets that were already lowered and dropping stale policies.
    per_loop: dict[LoopRegion, list[tuple[SDFGState, MultiConnectorEdge[Memlet], GlobalPath, int]]] = {}
    for state, edge in entries:
        policy: LoopCursor = edge.data.access_policy
        if policy.is_lowered:
            if edge.data.data in (policy.reference, policy.window):
                lowered += 1
            else:
                warnings.warn(f'Memlet "{edge.data}" carries a lowered policy of another memlet; dropping it.')
                edge.data.access_policy = CopyOnAccess()
                dropped += 1
            continue
        root, is_read = path_root(state, edge)
        path = GlobalPath(state, edge, is_read, kernel_dims) if root is not None else None
        found = path.find_loop(policy.loop, policy.variable) if path is not None else None
        if found is None:
            warnings.warn(
                f'Memlet access policy of "{edge.data}" refers to loop "{policy.loop}" (variable '
                f"{policy.variable}) which no longer encloses it; dropping the policy."
            )
            edge.data.access_policy = CopyOnAccess()
            dropped += 1
            continue
        level, loop = found
        per_loop.setdefault(loop, []).append((state, edge, path, level))

    # Outer loops first, so a chained inner cursor can share the class cursor an outer memlet created.
    inner_cache: dict[LoopRegion, set[str]] = {}
    ordered = sorted(per_loop.items(), key=lambda item: len(enclosing_loops(item[0])))
    for loop, loop_entries in ordered:
        if loop_analysis.get_init_assignment(loop) is None:
            warnings.warn(
                f'Cannot lower memlet access policies of loop "{loop.label}": no recognizable init '
                "assignment; policies dropped."
            )
            for _, edge, _, _ in loop_entries:
                edge.data.access_policy = CopyOnAccess()
            dropped += len(loop_entries)
            continue
        loop_sdfg = loop.sdfg
        if loop_sdfg not in tables:
            tables[loop_sdfg] = _CursorTable(loop_sdfg, chain_outer_loops)
        # Re-derive every policy (dropping stale ones), then group the memlets into cursor classes.
        members: dict[tuple, list[tuple[SDFGState, MultiConnectorEdge[Memlet], GlobalPath, int, OffsetDecomposition]]]
        members = {}
        dtypes_of: dict[tuple, dtypes.typeclass] = {}
        for state, edge, path, level in loop_entries:
            memlet = edge.data
            policy: LoopCursor = memlet.access_policy
            desc = state.sdfg.arrays.get(memlet.data)
            dec = None
            if (
                flat_addressable_array(desc)
                and isinstance(memlet.subset, Range)
                and (path.is_read or flat_length(desc, memlet.subset) is not None)
            ):
                dec = decompose_on_path(path, edge, level, loop, inner_cache)
            if dec is None or sp.expand(dec.step - policy.step) != 0:
                warnings.warn(
                    f'Memlet access policy of "{memlet}" is stale or not lowerable (recorded step '
                    f"{policy.step}, derived {None if dec is None else dec.step}); dropping it."
                )
                memlet.access_policy = CopyOnAccess()
                dropped += 1
                continue
            dtype = _cursor_dtype(policy, desc, assume_int32)
            array = path.levels[level].array
            key = _CursorTable.class_key(loop, array, dtype, policy.share_key, dec.class_base)
            members.setdefault(key, []).append((state, edge, path, level, dec))
            dtypes_of[key] = dtype
        for key, items in members.items():
            _, edge0, _, _, dec0 = items[0]
            array = key[1]
            # Anchor the class at the nest-invariant offset of one member (see _choose_anchor), so that a window's
            # base offset is added once, on loop entry, and each access carries only its small difference.
            anchor = _choose_anchor(
                [_invariant_part(dec.immediate, inner_cache[loop] | path.atom_names) for _, _, path, _, dec in items]
            )
            cursor, cursor_anchor = tables[loop_sdfg].cursor_for(
                loop, array, dtypes_of[key], dec0.class_base, anchor, edge0.data.access_policy.share_key
            )
            for state, edge, path, level, dec in items:
                if state.sdfg not in refs:
                    refs[state.sdfg] = _References(state.sdfg)
                _rewrite(
                    state,
                    edge,
                    path.to_leaf(cursor, level),
                    path.to_leaf(sp.expand(dec.immediate - cursor_anchor), level),
                    refs[state.sdfg],
                )
                lowered += 1
    return {"cursors": sum(t.created for t in tables.values()), "memlets": lowered, "dropped": dropped}


def _cursor_dtype(policy: LoopCursor, desc: dt.Data, assume_int32: bool) -> dtypes.typeclass:
    if policy.cursor_type is not None:
        return policy.cursor_type
    return dtypes.int32 if (assume_int32 or _extent_bytes_int32(desc)) else dtypes.int64


def _rewrite(
    state: SDFGState,
    edge: MultiConnectorEdge[Memlet],
    cursor: sp.Basic,
    immediate: sp.Basic,
    refs: _References,
) -> None:
    """Rewrite one loop-cursor memlet to address through the cursor: ``flat[cursor + immediate]`` for contiguous
    memlets, a window reference for non-contiguous reads.

    :param cursor: The cursor symbol, in the symbols of the memlet's SDFG.
    :param immediate: The memlet's offset relative to the cursor, in the symbols of the memlet's SDFG.
    """
    old = edge.data
    policy: LoopCursor = old.access_policy
    array = old.data
    desc = state.sdfg.arrays[array]
    root, is_read = path_root(state, edge)
    index = cursor + immediate
    flat = refs.flat_reference(array)
    length = flat_length(desc, old.subset)
    if length is not None:
        subset = Range([(index, index + length - 1, 1)])
        new_root = refs.node(state, flat, is_read)
        target, window = flat, None
    else:
        new_root = refs.window(state, array, old.subset, index)
        subset = Range([(0, sp.sympify(s) - 1, 1) for s in old.subset.size()])
        target, window = new_root.data, new_root.data
    new_memlet = Memlet(
        data=target,
        subset=subset,
        other_subset=old.other_subset,
        volume=old.volume,
        dynamic=old.dynamic,
        wcr=old.wcr,
        wcr_nonatomic=old.wcr_nonatomic,
        allow_oob=old.allow_oob,
        debuginfo=old.debuginfo,
    )
    policy.cursor, policy.reference, policy.window, policy.immediate = str(cursor), flat, window, immediate
    new_memlet.access_policy = policy
    _reroute(state, edge, new_root, is_read, new_memlet)
