# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Widen a top-level NSDFG's in/out subsets to the full outer arrays so
:class:`~dace.transformation.interstate.multistate_inline.InlineMultistateSDFG` can inline it.

``InlineMultistateSDFG.can_be_applied`` refuses NSDFGs whose in/out edges don't cover the full
outer array: its ``apply()`` lacks the dimension-offsetting to fix inner memlets after inline
(literal ``TODO: Modify memlets by offsetting`` in its body). Correctness gate -- bypassing it
renames inner ``IN_a`` → outer ``a`` without adjusting the baked-in per-iteration offset, so
``IN_a[i]`` lands on ``a[i]`` not ``a[ii + i]``.

This transformation does that offset adjustment up-front. Per in/out edge:
  * offset = lower bound of the narrowed subset (``ii`` for ``a[ii:ii+6, 0:M]``);
  * replace inner descriptor to mirror OUTER shape/strides;
  * widen outer-side subset to the full range;
  * add the offset to every inner memlet referencing the connector's array.
After it runs, every edge passes the full-array check and ``InlineMultistateSDFG.apply()`` is correct.

Refuses when: the widening would change inner-descriptor rank (axis-collapse, ``a[0:1, 0:M]`` →
1-D ``[M]``); the outer array is absent from the parent SDFG (orphan descriptor).
"""
import copy
from typing import List, Set, Dict, Tuple

from dace import SDFG, dtypes, subsets, symbolic, data
from dace.codegen.common import CodeBlock
from dace.sdfg import SDFGState, nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation import transformation
from dace.subsets import Range
from dace.memlet import Memlet
import sympy


def _resolve_outer_symbol_type(sym_name: str, sdfg: SDFG, default=None):
    """Walk up the SDFG nesting + the parent CFR tree to locate the
    type ``sym_name`` was declared with. Falls back to ``default`` (or
    ``dace.int64`` if ``default`` is ``None``)."""
    if default is None:
        default = dtypes.int64
    cur = sdfg
    while cur is not None:
        if sym_name in cur.symbols:
            return cur.symbols[sym_name]
        cur = getattr(cur, 'parent_sdfg', None)
    # Try LoopRegion loop-variables in the original SDFG (the loop var
    # carries the type via its enclosing SDFG's symbol table).
    for cfg in sdfg.all_control_flow_regions():
        if isinstance(cfg, LoopRegion) and cfg.loop_variable == sym_name:
            owner = getattr(cfg, 'sdfg', None)
            if owner is not None and sym_name in owner.symbols:
                return owner.symbols[sym_name]
    return default


def _full_subset(sdfg: SDFG, arr_name: str) -> subsets.Range:
    return subsets.Range.from_array(sdfg.arrays[arr_name])


def _collect_read_subsets(state: SDFGState, nsdfg_node: nodes.NestedSDFG) -> Dict[str, Tuple[str, Range]]:
    """Collect the original read subsets on every NSDFG input edge, keyed
    by the inner connector name. Used for the Map-scope case to capture
    the per-iteration tile offset."""
    read_subsets = {}
    for edge in state.in_edges(nsdfg_node):
        if edge.data is None or edge.data.data is None:
            continue
        # ``other_subset`` (e.g. K>=2 broadcast read into collapsed inner) folded to None at
        # widen time by _replace_desc_and_uncollapse_dims; tolerate here.
        conn = edge.dst_conn
        if conn is None:
            continue
        read_subsets[conn] = (edge.data.data, edge.data.subset)
    return read_subsets


def _collect_write_subsets(state: SDFGState, nsdfg_node: nodes.NestedSDFG) -> Dict[str, Tuple[str, Range]]:
    """Collect the original write subsets on every NSDFG output edge, keyed
    by the inner connector name. Used for the Map-scope case to capture
    the per-iteration tile offset."""
    write_subsets = {}
    for edge in state.out_edges(nsdfg_node):
        if edge.data is None or edge.data.data is None:
            continue
        # WCR edges excluded from offset/uncollapse (reduction handled separately; subset
        # conceptually ``[0]``). ``apply``'s 3-tuple upgrade loop skips them too, so a 2-tuple
        # entry here would crash the downstream 3-tuple unpack.
        if edge.data.wcr is not None:
            continue
        # other_subset tolerated -- folded at widen time (see _collect_read_subsets).
        conn = edge.src_conn
        if conn is None:
            continue
        write_subsets[conn] = (edge.data.data, edge.data.subset)
    return write_subsets


def _rewrite_memlets_with_offset(inner_sdfg: SDFG, inner_name: str, offset_dims: List[sympy.Basic],
                                 collapsed_dims: List[bool]) -> None:
    """Rewrite every memlet referencing ``inner_name``: add ``offset_dims``, uncollapse
    ``collapsed_dims``. Runs BEFORE ``replace_dict({inner_name: outer_name})`` so it matches
    only THIS inner_name's memlets -- else two connectors binding the same outer array at
    different offsets (``A[1,i,j]`` AND ``A[0,i,j]``) clobber cross-iteration.
    """
    for state in inner_sdfg.all_states():
        for edge in state.edges():
            memlet = edge.data
            if memlet is None or memlet.data != inner_name:
                continue
            new_range_list = []
            memlet_access_idx = 0
            inner_subset = memlet.subset.ranges
            # ``offset_dims`` / ``collapsed_dims`` span the FULL outer rank. Inner memlet aligns
            # two ways:
            #  * Full-rank inner (``len(inner_subset) == len(offset_dims)``): 1:1 dim map. Add
            #    boundary begin to EACH dim's begin, including length-1 collapsed dims (inner
            #    begin 0 → just offset). Modern NestInnermostMapBodyIntoNSDFG produces this: a
            #    3-point j-stencil reads ``A[0,0]/A[0,1]/A[0,2]`` as 2-D memlets even with dim0
            #    ``i:i`` length-1, so the intra-window ``+1``/``+2`` lives in the inner begin and
            #    MUST be carried through.
            #  * Rank-reduced inner (collapsed dims dropped): one entry per non-collapsed outer
            #    dim; collapsed dim contributes only the offset, ``memlet_access_idx`` walks the
            #    surviving inner dims.
            # Indexing rank-reduced against a full-rank inner = classic stencil miscompile:
            # ``A[0,1]`` consumes collapsed dim0's inner begin (``0``) for dim1, dropping ``+1``.
            inner_is_full_rank = (len(inner_subset) == len(offset_dims))
            for d, (offset, collapsed) in enumerate(zip(offset_dims, collapsed_dims)):
                if inner_is_full_rank:
                    (lo, hi, stp) = inner_subset[d]
                    # Nest rebases each access RELATIVE to boundary begin
                    # (``lo = original_begin - offset``): ``0`` collapsed, intra-window
                    # ``0``/``1``/``2`` stencil, or ``begin - bbox_begin`` for a multi-access
                    # array (non-affine ``Min(...)`` bbox begin). So absolute outer begin =
                    # ``lo + offset`` in every relative case. ONE exception: in-place RMW keeps
                    # its access ABSOLUTE (``A[i,j]`` verbatim, inner begin == boundary begin);
                    # re-adding there double-counts (``i + i = 2*i``). Detect via ``lo == offset``
                    # -- a free-symbol overlap test mis-fires on relative ``i - Min(i, i+H)``
                    # (begin shares ``i`` yet must still rebase), dropping the slide. ``lo==offset``
                    # found via sympy Add cancellation (``i - i`` → 0, relative ``(i-Min)-Min`` stays
                    # ``i-2*Min``). No ``sympy.simplify`` -- too slow on Min/int_floor in a codegen-hot pass.
                    if (sympy.sympify(lo) - sympy.sympify(offset)) == 0:
                        new_range_list.append((lo, hi, stp))
                    else:
                        new_range_list.append((lo + offset, hi + offset, stp))
                elif collapsed is True:
                    new_range_list.append((offset, offset, 1))
                else:
                    (lo, hi, stp) = inner_subset[memlet_access_idx]
                    new_range_list.append((lo + offset, hi + offset, stp))
                    memlet_access_idx += 1
            # WCR (reduction) memlet only relocates -- accumulation preserved. Offset the data
            # subset like any memlet and carry the ``wcr`` lambda through (dropping it would
            # miscompile gramschmidt / correlation).
            if memlet.other_subset is not None:
                src = edge.src
                dst = edge.dst
                # ``memlet.subset`` (== inner_name) just offset into ``new_range_list``.
                # ``other_subset`` addresses the OTHER endpoint; it needs the SAME offset ONLY in
                # a genuine self-copy ``a -> a`` (both endpoints ``inner_name`` access nodes). In
                # every other shape -- array<->temp copy, View<->array reshape (View has its own
                # indexing), or boundary read/write THROUGH a MapEntry/MapExit while the inner map
                # is un-lowered -- the other endpoint is a different array independent of the tile
                # offset, so preserve it verbatim and offset only the named-array subset.
                src_is_inner = isinstance(src, nodes.AccessNode) and src.data == inner_name
                dst_is_inner = isinstance(dst, nodes.AccessNode) and dst.data == inner_name
                if src_is_inner and dst_is_inner:
                    # Both sides ARE the offset array: ``other_subset`` would need the offset too.
                    # No current lowering produces this; refuse rather than drop it (miscompile).
                    raise NotImplementedError("Cannot offset a self-copy of the boundary array %r on both sides" %
                                              inner_name)
                new_memlet = Memlet(data=memlet.data,
                                    subset=subsets.Range(new_range_list),
                                    other_subset=copy.deepcopy(memlet.other_subset))
                new_memlet.wcr = memlet.wcr
                # Preserve dynamic flag: a masked/conditional write (``A[mask] = v``) may not
                # write every element; dropping it → codegen writes unconditionally.
                new_memlet.dynamic = memlet.dynamic
                edge.data = new_memlet
            else:
                new_memlet = Memlet(data=memlet.data, subset=subsets.Range(new_range_list))
                new_memlet.wcr = memlet.wcr
                new_memlet.dynamic = memlet.dynamic  # preserve conditional-write flag (see above)
                edge.data = new_memlet


def _replace_desc_and_uncollapse_dims(nsdfg_node: nodes.NestedSDFG,
                                      state: SDFGState,
                                      inner_name: str,
                                      outer_name: str,
                                      desc: data.Array,
                                      collapsed_dims: List[bool],
                                      offset_dims: List[sympy.Basic],
                                      direction: str,
                                      apply_offset: bool = True) -> None:
    # Replace inner_name occurrences + data descriptor with outer_name.
    assert isinstance(inner_name, str) and isinstance(outer_name, str)

    # Remove old array, add new, so occurrences can be safely replaced.
    inner_sdfg: SDFG = nsdfg_node.sdfg
    inner_sdfg.remove_data(inner_name, validate=False)
    copy_desc = copy.deepcopy(desc)
    copy_desc.transient = False
    if outer_name not in inner_sdfg.arrays:
        inner_sdfg.add_datadesc(outer_name, copy_desc)

    # Rewrite inner memlets BEFORE the rename so offset_dims apply ONLY to THIS inner_name's
    # memlets. After renaming, a previous iteration's memlets (same outer_name, different
    # inner_name/offset) would be clobbered. E.g. ``B = A[1,i,j] + A[0,i,j]`` → connectors
    # ``__tmp_a`` (offset [1,0,0]) + ``__tmp_b`` (offset [0,0,0]); after ``__tmp_a`` → A,
    # renaming ``__tmp_b`` → A would match BOTH A-memlets and erase the [1] offset.
    # Offset once per array (``apply_offset``): a second pass (array read AND written, shared
    # outer name) still renames/widens its own connector, but re-offsetting double-counts.
    if apply_offset:
        _rewrite_memlets_with_offset(inner_sdfg, inner_name, offset_dims, collapsed_dims)

    # ``expr.replace(SubscriptClass, fn)``: SymPy splats the matched node's args positionally
    # (not the node), so the callback takes ``*args`` = Subscript arity: ``args[0]`` container,
    # ``args[1:]`` indices.
    #
    # These callbacks + the interstate-edge/loop-head rewrites below run BEFORE ``replace_dict``
    # and match ``inner_name`` (still-distinct connector), NOT post-rename ``outer_name`` --
    # interstate analogue of the memlet ordering fix above. Two connectors binding the same
    # outer array at different offsets (spmv ``indptr[i]`` + ``indptr[i+1]``): matching
    # ``outer_name`` after rename let a later connector re-collapse an earlier rewrite
    # (``row_start = indptr[i]`` → ``indptr[i+1]``), conflating both row bounds (size-0 buffer).
    def _uncollapse_subscript(*args):
        base = args[0]
        # Compare by NAME: the base is a ``dace.symbolic.symbol`` (dtype-carrying Symbol
        # subclass) that does NOT compare equal to a plain ``sympy.Symbol(inner_name)``. Old
        # equality silently failed → gather index ``edge_idx[0,0,0]`` rebuilt verbatim not
        # uncollapsed to ``edge_idx[jb,jc,0]`` (every lane gathered the same element). Dataflow-
        # memlet path unaffected: rewrites subset ranges directly, never comparing symbols.
        if str(base) == inner_name:
            new_indices = []
            for idx, offset, collapsed in zip(args[1:], offset_dims, collapsed_dims):
                if collapsed:
                    new_indices.append(offset)
                else:
                    new_indices.append(idx)
            return symbolic.Subscript(base, *new_indices)
        # Not our target: rebuild the original Subscript verbatim.
        return symbolic.Subscript(*args)

    def _uncollapse_scalar(node):
        # ``node`` is the actual ``inner_name`` dace symbol; subscript with per-axis offsets
        # directly. A plain ``sympy.Symbol`` + ``node.subs`` wouldn't match (see
        # :func:`_uncollapse_subscript`), leaving the reference uncollapsed.
        return symbolic.Subscript(node, *offset_dims)

    # Per user direction 2026-06-10: "on memlets subset 0,0,1 but on codeblocks treat scalar as
    # symbol." A true ``dace.data.Scalar`` has no dim to subscript -- bare symbol is the correct
    # C++ form. Only wrap ``[offset_dims]`` when the source is an Array.
    outer_is_scalar = isinstance(desc, data.Scalar)

    # Uncollapse dims (interstate edges).
    for edge in inner_sdfg.edges():
        assignments = edge.data.assignments
        new_assignments = dict()
        for var, str_expr in assignments.items():
            symexpr = symbolic.pystr_to_symbolic(str_expr)
            if inner_name in symbolic.arrays(symexpr):
                new_assignments[var] = symbolic.symstr(symexpr.replace(symbolic.Subscript, _uncollapse_subscript))
            elif inner_name in {str(s) for s in symexpr.free_symbols}:  # Could be scalar and fully collapsed
                if outer_is_scalar:
                    # Scalar source: keep bare symbol; codegen handles it like any free symbol.
                    new_assignments[var] = str_expr
                else:
                    matching_syms = {s for s in symexpr.free_symbols if str(s) == inner_name}
                    assert len(matching_syms) == 1, \
                        f"Expected exactly one matching symbol for {inner_name} in {symexpr}, found {matching_syms}"
                    sym = matching_syms.pop()
                    new_assignments[var] = symbolic.symstr(symexpr.subs(sym, _uncollapse_scalar(sym)))
            else:
                new_assignments[var] = str_expr
        edge.data.assignments = new_assignments

    # Uncollapse dims in loop-head CodeBlocks: ``loop_condition``, ``update_statement``
    # (``i = i + 1``, not a pure expr), ``init_statement`` (``i = 0``). Each may be ``None``;
    # assignment-style statements aren't parseable by :func:`pystr_to_symbolic` -- rewrite when
    # parseable, else leave (those don't reference connector arrays).
    def _rewrite_connector_refs(sym):
        """Rewrite ``inner_name`` refs in a codeblock/branch-condition. Subscripted
        (``__tmp[x]``) uncollapses via :func:`_uncollapse_subscript`; bare scalar (``if __tmp``,
        single-element mask as condition) must become ``__tmp[offset_dims]`` once widened, else
        the bare name tests the WHOLE array (always truthy) -- the collapsed-mask bug
        (azimint_naive unmasked mean-reduction). A true ``Scalar`` stays a bare symbol."""
        if inner_name in symbolic.arrays(sym):
            return sym.replace(symbolic.Subscript, _uncollapse_subscript)
        if (not outer_is_scalar) and inner_name in {str(s) for s in sym.free_symbols}:
            for s in [x for x in sym.free_symbols if str(x) == inner_name]:
                sym = sym.subs(s, _uncollapse_scalar(s))
        return sym

    def _rewrite_codeblock(cb):
        if cb is None:
            return cb
        try:
            sym = symbolic.pystr_to_symbolic(cb.as_string)
        except Exception:
            return cb
        return CodeBlock(symbolic.symstr(_rewrite_connector_refs(sym)))

    for cfg in inner_sdfg.all_control_flow_regions():
        if isinstance(cfg, LoopRegion):
            cfg.loop_condition = _rewrite_codeblock(cfg.loop_condition)
            cfg.update_statement = _rewrite_codeblock(cfg.update_statement)
            cfg.init_statement = _rewrite_codeblock(cfg.init_statement)
    # Uncollapse dims in conditional-branch heads.
    for cfg in inner_sdfg.all_control_flow_blocks():
        if isinstance(cfg, ConditionalBlock):
            for i, (cond, body) in enumerate(cfg.branches):
                cfg.branches[i] = (CodeBlock(
                    symbolic.symstr(_rewrite_connector_refs(symbolic.pystr_to_symbolic(cond.as_string)))), body)

    # Uncollapse dims in dataflow memlet SUBSET exprs -- 4th connector-reference site (after
    # interstate assignments, loop heads, branch conditions). A gather/scatter INDEX array is
    # referenced ONLY inside ANOTHER memlet's subset: scatter ``dst[__tmp]`` (``memlet.data ==
    # dst``) carries collapsed scalar ``__tmp`` (= ``idx[i]``) as index. ``_rewrite_memlets_with_offset``
    # matches ``memlet.data == inner_name`` → never touches it; left alone, ``replace_dict``
    # renames bare ``__tmp`` → ``idx`` and DROPS ``[i]`` → loop-invariant ``dst[idx]`` (scatter
    # misread as constant).
    #
    # Rewrite straight to OUTER subscripted by ``offset_dims``: ``__tmp`` → ``idx[i]``,
    # ``__tmp[x]`` → ``idx[x]``. Emit OUTER name HERE not ``inner_name[offset]`` for
    # ``replace_dict``: its sympy ``subs`` does NOT descend into a Subscript base built from a
    # dace symbol (see ``_uncollapse_subscript``), so a base left ``inner_name`` survives
    # unrenamed. A true ``Scalar`` outer needs no subscript (bare rename is correct) → only Arrays.
    if not outer_is_scalar:
        outer_sym = symbolic.symbol(outer_name)

        def _index_to_outer(*args):
            # ``inner_name[x...]`` -> ``outer_name[x...]`` (keep the existing index).
            if str(args[0]) == inner_name:
                return symbolic.Subscript(outer_sym, *args[1:])
            return symbolic.Subscript(*args)

        def _rw_index_expr(expr):
            symexpr = symbolic.pystr_to_symbolic(str(expr))
            if inner_name not in symbolic.arrays(symexpr) and \
                    inner_name not in {str(s) for s in symexpr.free_symbols}:
                return expr
            # Already-subscripted uses first, then any surviving bare-scalar use.
            symexpr = symexpr.replace(symbolic.Subscript, _index_to_outer)
            for s in [x for x in symexpr.free_symbols if str(x) == inner_name]:
                symexpr = symexpr.subs(s, symbolic.Subscript(outer_sym, *offset_dims))
            return symexpr

        def _uncollapse_subset_refs(sub) -> None:
            if isinstance(sub, subsets.Range):
                sub.ranges = [(_rw_index_expr(b), _rw_index_expr(e), _rw_index_expr(s)) for (b, e, s) in sub.ranges]
            elif isinstance(sub, subsets.Indices):
                sub.indices = [_rw_index_expr(i) for i in sub.indices]

        for st in inner_sdfg.all_states():
            for edge in st.edges():
                memlet = edge.data
                # Own-array memlets already handled by ``_rewrite_memlets_with_offset``; here
                # only OTHER memlets carrying ``inner_name`` as a subset index.
                if memlet is None or memlet.data == inner_name:
                    continue
                _uncollapse_subset_refs(memlet.subset)
                _uncollapse_subset_refs(memlet.other_subset)

    inner_sdfg.replace_dict({inner_name: outer_name})

    # Replace connectors. If ``outer_name`` already has an edge on this side (a previous
    # iteration merged another connector binding the same outer array, e.g. ``A[1,i,j]`` →
    # ``__tmp_a`` AND ``A[0,i,j]`` → ``__tmp_b`` both bind outer ``A``), keep ONE merged edge and
    # REMOVE the duplicate. Its subset is the full-array ``_full_subset`` so the post-widening
    # contract holds regardless of which connector contributed which slice.
    assert direction in ('in', 'out')
    if direction == 'in':
        existing_target = any(e.dst_conn == outer_name and e.dst_conn != inner_name for e in state.in_edges(nsdfg_node))
        for iedge in list(state.in_edges(nsdfg_node)):
            if iedge.dst_conn == inner_name:
                if existing_target:
                    # Drop this edge -- existing ``outer_name`` edge already covers the full
                    # subset. Remove the whole memlet PATH: the source is typically a MapEntry
                    # pass-through (``ME.OUT_x -> NSDFG``), so dropping only the NSDFG-incident
                    # edge dangles ``ME``'s ``IN_x``/``OUT_x`` + feeding edge (invalid SDFG).
                    # symm (``A[i,k]`` + ``A[k,i]`` through two connectors) hits this.
                    state.remove_memlet_path(iedge, remove_orphans=True)
                else:
                    iedge.dst_conn = outer_name
                    iedge.data.subset = _full_subset(state.sdfg, outer_name)
                    # Inner descriptor now mirrors the full outer array; inner memlets carry the
                    # offset (above). Any ``other_subset`` described the OLD collapsed inner shape
                    # (K>=2 broadcast ``a[i//2] -> inner_a[0]``, inner ``(1,)``) → stale after
                    # widening. Clear it for a clean full-array passthrough (design 2.4).
                    iedge.data.other_subset = None
        # ``remove_memlet_path`` already dropped the connector on the dedup branch; guard the
        # idempotent explicit removal.
        if inner_name in nsdfg_node.in_connectors:
            nsdfg_node.remove_in_connector(inner_name)
        if outer_name not in nsdfg_node.in_connectors:
            nsdfg_node.add_in_connector(outer_name, force=True)
    else:
        existing_target = any(e.src_conn == outer_name and e.src_conn != inner_name
                              for e in state.out_edges(nsdfg_node))
        for oedge in list(state.out_edges(nsdfg_node)):
            if oedge.src_conn == inner_name:
                if existing_target:
                    # Drop the whole path (mirrors in-edge dedup): the sink is typically a
                    # MapExit pass-through, so removing only the NSDFG-incident edge dangles its
                    # ``IN_x``/``OUT_x``.
                    state.remove_memlet_path(oedge, remove_orphans=True)
                else:
                    oedge.src_conn = outer_name
                    oedge.data.subset = _full_subset(state.sdfg, outer_name)
                    oedge.data.other_subset = None  # stale after widening (see in-edge note)
        if inner_name in nsdfg_node.out_connectors:
            nsdfg_node.remove_out_connector(inner_name)
        if outer_name not in nsdfg_node.out_connectors:
            nsdfg_node.add_out_connector(outer_name, force=True)

    # (Inner memlets, interstate assignments, loop/branch heads all rewritten above with
    # per-inner_name offsets BEFORE replace_dict, to avoid clobbering an earlier connector.)


class ExpandNestedSDFGInputs(transformation.SingleStateTransformation):
    """Pre-processor for :class:`InlineMultistateSDFG`: widen narrowed NSDFG in/out memlets to
    full-array subsets, reshape inner descriptors, offset inner memlets.

    Handles top-level NSDFGs and NSDFGs inside a Map scope. In-map case: caller must first widen
    the parent map's IN/OUT memlets to full arrays (via ``propagate_full_array_subsets_through_map``)
    so the MapEntry→NSDFG connector carries the full extent; the per-iteration tile offset is then
    captured from the original narrowed subset and threaded onto every inner memlet.
    """

    nested_sdfg = transformation.PatternNode(nodes.NestedSDFG)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.nested_sdfg)]

    @staticmethod
    def annotates_memlets():
        return True

    def can_be_applied(self, state: SDFGState, expr_index, sdfg: SDFG, permissive=False) -> bool:
        nsdfg_node = self.nested_sdfg
        if nsdfg_node.no_inline:
            return False
        # Refuse when every in/out edge already reads the full outer array -- nothing to widen,
        # and re-applying would spin ``apply_transformations_repeated`` forever.
        for edge in (*state.in_edges(nsdfg_node), *state.out_edges(nsdfg_node)):
            if edge.data is None or edge.data.data is None:
                continue
            # A WCR boundary edge is NOT widened by ``apply`` (subset = reduction target slot,
            # kept as-is; see the ``wcr is not None`` skip in write-subset collection). Flagging
            # it here (subset is a per-iter slice, never full) would re-match forever → reduce-at-
            # output hang.
            if edge.data.wcr is not None:
                continue
            outer_arr = sdfg.arrays.get(edge.data.data)
            if outer_arr is None:
                continue
            if edge.data.subset != _full_subset(sdfg, edge.data.data):
                return True
        return False

    def apply(self, state: SDFGState, sdfg: SDFG) -> None:
        nsdfg_node = self.nested_sdfg
        inner = nsdfg_node.sdfg

        # Offset symbols to propagate outer→NSDFG (into ``symbol_mapping`` + inner ``symbols``
        # so inner memlet refs validate).
        introduced_symbols: Set[str] = set()

        # Inner arrays already widened. A connector used for BOTH an in- and out-edge (same outer
        # array read AND written in-place, e.g. ``A[i,j,k+1] = A[i,j,k] + A[i,j,k-1]``) shares the
        # inner array; without dedup we'd offset its memlets TWICE, corrupting numerics.
        processed_inner_arrays: Set[str] = set()

        read_subsets = _collect_read_subsets(state, nsdfg_node)
        write_subsets = _collect_write_subsets(state, nsdfg_node)
        inner_sdfg = nsdfg_node.sdfg

        # Collect read subsets, per inner name (an array may appear on multiple edges).
        for iedge in state.in_edges(nsdfg_node):
            if iedge.data is None or iedge.data.data is None:
                continue
            # other_subset tolerated -- folded to None at widen time.
            in_conn = iedge.dst_conn
            if in_conn is None:
                continue

            # Widen if not already full: find collapsed dims (outer [N,N,N], read [1,0:N,0:N] →
            # inner [N,N]).
            inner_desc = inner_sdfg.arrays[in_conn]
            collapsed_dims = []
            for (b, e, s) in iedge.data.subset.ranges:
                if (e + 1 - b) // s == 1:
                    collapsed_dims.append(True)
                else:
                    collapsed_dims.append(False)

            read_subsets[in_conn] = (iedge.data.data, iedge.data.subset, collapsed_dims)

        for oedge in state.out_edges(nsdfg_node):
            if oedge.data is None or oedge.data.data is None:
                continue
            # other_subset tolerated -- folded to None at widen time.
            if oedge.data.wcr is not None:
                continue  # WCR edges have special meaning -> also wcr means subset should be [0]
            out_conn = oedge.src_conn
            if out_conn is None:
                continue

            inner_desc = inner_sdfg.arrays[out_conn]
            collapsed_dims = []
            for (b, e, s) in oedge.data.subset.ranges:
                if (e + 1 - b) // s == 1:
                    collapsed_dims.append(True)
                else:
                    collapsed_dims.append(False)

            write_subsets[out_conn] = (oedge.data.data, oedge.data.subset, collapsed_dims)

        # ``_rewrite_memlets_with_offset`` offsets EVERY inner memlet referencing ``conn``'s
        # array, not just the boundary edge. When the same inner array is read AND written
        # (in-place kernel like s173 ``a[i+H] = a[i] + b[i]``, connector shares the outer name),
        # it appears on an in- AND out-edge; offsetting both passes re-bases each memlet twice
        # (``i - Min`` → ``i`` → ``i + Min``), collapsing the slide. So offset only on the FIRST
        # pass (``apply_offset``); BOTH passes still rename/widen their own connector (direction-
        # gated; skipping a pass leaves a dangling connector → hang).
        for (conn, (outer_arr_name, outer_subset, collapsed_dims)) in read_subsets.items():
            apply_offset = conn not in processed_inner_arrays
            processed_inner_arrays.add(conn)
            _replace_desc_and_uncollapse_dims(nsdfg_node,
                                              state,
                                              conn,
                                              outer_arr_name,
                                              sdfg.arrays[outer_arr_name],
                                              collapsed_dims, [lo for (lo, _hi, _stp) in outer_subset.ranges],
                                              direction='in',
                                              apply_offset=apply_offset)

        for (conn, (outer_arr_name, outer_subset, collapsed_dims)) in write_subsets.items():
            apply_offset = conn not in processed_inner_arrays
            processed_inner_arrays.add(conn)
            _replace_desc_and_uncollapse_dims(nsdfg_node,
                                              state,
                                              conn,
                                              outer_arr_name,
                                              sdfg.arrays[outer_arr_name],
                                              collapsed_dims, [lo for (lo, _hi, _stp) in outer_subset.ranges],
                                              direction='out',
                                              apply_offset=apply_offset)

        # Thread any gather/scatter INDEX array referenced inside an inner memlet SUBSET but
        # absent from ``inner_sdfg.arrays``. ``A[B[i]]`` where ``B`` was never a boundary
        # connector (nesting threaded DATA ``A`` but not index ``B``, which appears only in
        # ``A``'s subset) leaves ``B`` dangling → can't codegen through inline, no per-lane index
        # tile. Add ``B`` as a full-array read boundary (like ``A``): non-transient inner
        # descriptor, in-connector, access-node edge routed through the enclosing Map (if any).
        def _subset_arrays(sub) -> Set[str]:
            if isinstance(sub, subsets.Range):
                exprs = [x for r in sub.ranges for x in r]
            elif isinstance(sub, subsets.Indices):
                exprs = list(sub.indices)
            else:
                return set()
            names: Set[str] = set()
            for ex in exprs:
                names |= symbolic.arrays(symbolic.pystr_to_symbolic(str(ex)))
            return names

        referenced: Set[str] = set()
        for st in inner_sdfg.all_states():
            for edge in st.edges():
                if edge.data is None:
                    continue
                referenced |= _subset_arrays(edge.data.subset)
                referenced |= _subset_arrays(edge.data.other_subset)
        entry = state.entry_node(nsdfg_node)
        for arr_name in sorted(referenced):
            # Only a genuinely-dangling OUTER array (a symbol / already-threaded
            # connector array is skipped).
            if arr_name in inner_sdfg.arrays or arr_name not in sdfg.arrays:
                continue
            index_desc = copy.deepcopy(sdfg.arrays[arr_name])
            index_desc.transient = False
            inner_sdfg.add_datadesc(arr_name, index_desc)
            if arr_name not in nsdfg_node.in_connectors:
                nsdfg_node.add_in_connector(arr_name, force=True)
            src = state.add_access(arr_name)
            # Gather/scatter reads the WHOLE index array (data-dependent) → thread the FULL
            # subset, not a per-iter slice. FRESH Memlet per edge (DaCe forbids shared subset
            # objects); route through the enclosing Map with the full extent on both edges.
            if entry is not None:
                in_conn, out_conn = 'IN_' + arr_name, 'OUT_' + arr_name
                entry.add_in_connector(in_conn)
                entry.add_out_connector(out_conn)
                state.add_edge(src, None, entry, in_conn, Memlet(data=arr_name, subset=_full_subset(sdfg, arr_name)))
                state.add_edge(entry, out_conn, nsdfg_node, arr_name,
                               Memlet(data=arr_name, subset=_full_subset(sdfg, arr_name)))
            else:
                state.add_edge(src, None, nsdfg_node, arr_name,
                               Memlet(data=arr_name, subset=_full_subset(sdfg, arr_name)))

        # A connector's inner array MUST be non-transient (boundary interface, not storage).
        # When a connector name collides with a same-named inner transient (accumulator ``tmp``
        # that is BOTH an inout connector and an internal transient, from ``if mask[j]: tmp +=
        # data[j]`` after branch lowering), the rename may leave it transient → validation
        # rejects it. Clear the flag on every connector's inner array.
        for conn in (*nsdfg_node.in_connectors, *nsdfg_node.out_connectors):
            inner_desc = inner_sdfg.arrays.get(conn)
            if inner_desc is not None and inner_desc.transient:
                inner_desc.transient = False

        defined_syms = set(sdfg.arrays.keys()) | set(inner_sdfg.symbols.keys()) | set(nsdfg_node.symbol_mapping.keys())
        for conn, (outer_arr_name, outer_subset, collapsed_dims) in read_subsets.items():
            free_syms = outer_subset.free_symbols - defined_syms
            introduced_symbols.update(str(s) for s in free_syms)
        for conn, (outer_arr_name, outer_subset, collapsed_dims) in write_subsets.items():
            free_syms = outer_subset.free_symbols - defined_syms
            introduced_symbols.update(str(s) for s in free_syms)
        # Per user direction 2026-06-10: all symbols in any array's shape/strides/offsets must be
        # added to the inner NSDFG (if absent) AND bound in symbol_mapping (identity default). Use
        # ``Data.free_symbols`` (aggregates shape+strides+offset) not per-field walking.
        for inner_arr_name, inner_desc in inner_sdfg.arrays.items():
            for sym in inner_desc.free_symbols:
                sym_name = str(sym)
                if sym_name in nsdfg_node.in_connectors or sym_name in nsdfg_node.out_connectors:
                    continue
                if sym_name not in nsdfg_node.symbol_mapping:
                    introduced_symbols.add(sym_name)

        # Propagate offset symbols not already in symbol_mapping. Identity binding is the default
        # (outer ``ii`` → inner ``ii``); validation surfaces a truly-undefined symbol on the next
        # pass (a real caller bug). Resolve each symbol's type by walking the outer SDFG ancestry
        # + LoopRegion loop variables -- silently defaulting to ``int64`` would mismatch a caller
        # that declared another integer width.
        for sym_name in introduced_symbols:
            if sym_name in nsdfg_node.symbol_mapping:
                continue
            if sym_name in inner.arrays:
                continue  # an array name -- not a symbol, skip
            if sym_name in inner.symbols:
                # Symbol exists in inner but not mapped from outer -- bind it.
                nsdfg_node.symbol_mapping[sym_name] = symbolic.pystr_to_symbolic(sym_name)
                continue
            # New to both inner and the mapping: resolve type from outer scope and copy through.
            outer_type = _resolve_outer_symbol_type(sym_name, sdfg)
            inner.add_symbol(sym_name, outer_type)
            nsdfg_node.symbol_mapping[sym_name] = symbolic.pystr_to_symbolic(sym_name)
