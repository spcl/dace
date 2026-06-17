# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Widen a top-level NSDFG's in/out subsets to the full outer arrays so
:class:`~dace.transformation.interstate.multistate_inline.InlineMultistateSDFG`
can subsequently inline it.

Background
----------

``InlineMultistateSDFG.can_be_applied`` refuses NSDFGs whose in/out edges
do not read/write the full outer array, because the corresponding
``apply()`` does not perform the dimension-offsetting required to correct
inner memlets after the inline (there is a literal ``TODO: Modify
memlets by offsetting`` in its body). This refusal is a correctness
gate: bypassing it would inline the body and rename the inner connector
``IN_a`` to the outer array ``a`` without adjusting the per-iteration
offset baked into inner memlets, so ``IN_a[i]`` would land on ``a[i]``
instead of ``a[ii + i]``.

This transformation performs the missing offset adjustment up-front:

1. For each in/out edge of the NSDFG node:

   * Compute the per-axis offset (the lower bound of the original
     narrowed subset, e.g. ``ii`` for ``a[ii:ii+6, 0:M]``).
   * Replace the inner array descriptor with one that mirrors the OUTER
     descriptor's shape / strides.
   * Widen the outer-side subset to the full outer-array range.
   * Walk every inner memlet that references this connector's array
     and add the offset to its subset coordinates.

2. After this transformation runs, every NSDFG edge satisfies the
   strict full-array check and ``InlineMultistateSDFG.apply()`` produces
   a correct inlined SDFG.

Refusal criteria
----------------

* The NSDFG is inside a Map scope (``state.entry_node(nsdfg) is not
  None``). Per-iteration narrowing is intentional inside a Map, so
  widening it would lose the parallelism contract.
* The narrowed-to-full expansion would change the rank of the inner
  descriptor (axis-collapse case, e.g. ``a[0:1, 0:M]`` whose inner
  descriptor was simplified to a 1-D ``[M]`` array). The transformation
  refuses rather than guess how to re-promote the missing axis.
* The outer array doesn't exist in the parent SDFG (orphan descriptor).
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
        # ``other_subset`` (if present, e.g. a K>=2 broadcast read into a
        # collapsed inner) is folded to None at widen time by
        # _replace_desc_and_uncollapse_dims; tolerate it here.
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
        # other_subset tolerated -- folded at widen time (see _collect_read_subsets).
        conn = edge.src_conn
        if conn is None:
            continue
        write_subsets[conn] = (edge.data.data, edge.data.subset)
    return write_subsets


def _rewrite_memlets_with_offset(inner_sdfg: SDFG, inner_name: str, offset_dims: List[sympy.Basic],
                                 collapsed_dims: List[bool]) -> None:
    """Rewrite every memlet referencing ``inner_name`` to add ``offset_dims``
    and uncollapse the ``collapsed_dims``. Runs BEFORE the inner SDFG's
    ``replace_dict({inner_name: outer_name})`` so the rewrite only matches
    memlets that originated from THIS inner_name -- avoiding the
    cross-iteration clobber where two inner connectors bind the same outer
    array with different constant-dim offsets (e.g. ``A[1,i,j]`` AND
    ``A[0,i,j]``).
    """
    for state in inner_sdfg.all_states():
        for edge in state.edges():
            memlet = edge.data
            if memlet is None or memlet.data != inner_name:
                continue
            new_range_list = []
            memlet_access_idx = 0
            inner_subset = memlet.subset.ranges
            # ``offset_dims`` / ``collapsed_dims`` span the FULL outer-array rank.
            # The inner memlet aligns to them in one of two ways:
            #
            #  * **Full-rank inner** (``len(inner_subset) == len(offset_dims)``):
            #    the inner descriptor mirrors every outer dim, so each inner dim
            #    maps 1:1 to an outer dim. Add the boundary begin to EACH dim's
            #    own begin -- including the length-1 (collapsed) dims, whose
            #    inner begin is 0 so the result is just the offset. This is what
            #    modern ``NestInnermostMapBodyIntoNSDFG`` produces: a 3-point
            #    j-stencil reads ``A[0,0] / A[0,1] / A[0,2]`` as 2-D memlets even
            #    though the boundary dim0 ``i:i`` is length-1, so the per-access
            #    intra-window offset (the ``+1`` / ``+2``) lives in the inner
            #    begin and MUST be carried through (not skipped).
            #  * **Rank-reduced inner** (inner dropped the collapsed dims): the
            #    inner subset has one entry per NON-collapsed outer dim, so a
            #    collapsed dim contributes only the offset and the running
            #    ``memlet_access_idx`` walks the surviving inner dims.
            #
            # Indexing the rank-reduced way against a full-rank inner is the
            # classic stencil miscompile: ``A[0,1]`` would consume the collapsed
            # dim0's inner begin (``0``) for dim1 and silently drop the ``+1``.
            inner_is_full_rank = (len(inner_subset) == len(offset_dims))
            for d, (offset, collapsed) in enumerate(zip(offset_dims, collapsed_dims)):
                if inner_is_full_rank:
                    (lo, hi, stp) = inner_subset[d]
                    # Add the window base to a full-rank inner dim ONLY when the
                    # inner begin is expressed RELATIVE to the window (an
                    # intra-window stencil offset ``0`` / ``1`` / ``2``, or the
                    # NSDFG-boundary connector binding ``[0:1]``). When the inner
                    # begin ALREADY references the offset's iteration symbol(s) it
                    # is in absolute outer coordinates -- an in-place RMW body
                    # keeps ``A[i, j]`` verbatim (the inner SDFG receives ``i``,
                    # ``j`` as symbols) -- and re-adding the base double-counts
                    # (``i + i = 2*i``), reading/writing only every other element.
                    # Detect the absolute case via free-symbol overlap with the
                    # offset and leave such dims untouched.
                    off_syms = sympy.sympify(offset).free_symbols
                    lo_syms = sympy.sympify(lo).free_symbols
                    if off_syms and (off_syms & lo_syms):
                        new_range_list.append((lo, hi, stp))
                    else:
                        new_range_list.append((lo + offset, hi + offset, stp))
                elif collapsed is True:
                    new_range_list.append((offset, offset, 1))
                else:
                    (lo, hi, stp) = inner_subset[memlet_access_idx]
                    new_range_list.append((lo + offset, hi + offset, stp))
                    memlet_access_idx += 1
            assert memlet.wcr is None
            if memlet.other_subset is not None:
                src = edge.src
                dst = edge.dst
                if (isinstance(src, nodes.AccessNode) and isinstance(dst, nodes.AccessNode)
                        and (memlet.other_subset == subsets.Range([(0, 0, 1)]) and memlet.data == src.data)):
                    new_memlet = Memlet(data=memlet.data,
                                        subset=subsets.Range(new_range_list),
                                        other_subset=subsets.Range([(0, 0, 1)]))
                    edge.data = new_memlet
                else:
                    raise NotImplementedError("Unsupported other subset case for memlet with data == outer array")
            else:
                new_memlet = Memlet(data=memlet.data, subset=subsets.Range(new_range_list))
                edge.data = new_memlet


def _replace_desc_and_uncollapse_dims(nsdfg_node: nodes.NestedSDFG, state: SDFGState, inner_name: str, outer_name: str,
                                      desc: data.Array, collapsed_dims: List[bool], offset_dims: List[sympy.Basic],
                                      direction: str) -> None:
    # Replace all occurencess of conn with outer name
    # Replace data descriptor
    assert isinstance(inner_name, str) and isinstance(outer_name, str)

    # Remoce old array and add new one, such that we can safely replace occurences
    inner_sdfg: SDFG = nsdfg_node.sdfg
    inner_sdfg.remove_data(inner_name, validate=False)
    copy_desc = copy.deepcopy(desc)
    copy_desc.transient = False
    if outer_name not in inner_sdfg.arrays:
        inner_sdfg.add_datadesc(outer_name, copy_desc)

    # Rewrite inner memlets BEFORE the rename so the offset_dims apply ONLY to
    # memlets that belonged to THIS inner_name. If we rewrote AFTER renaming,
    # memlets from a previous iteration of this helper (binding the SAME
    # outer_name from a different inner_name with a different offset_dims)
    # would get clobbered. Concretely: for kernel ``B = A[1,i,j] + A[0,i,j]``
    # we get two inner connectors ``__tmp_a`` (offset [1,0,0]) and ``__tmp_b``
    # (offset [0,0,0]). Iteration 1 renames ``__tmp_a`` -> ``A`` + offsets by
    # [1,0,0]. Iteration 2 would then rename ``__tmp_b`` -> ``A`` and match
    # BOTH already-A memlets, re-applying offset [0,0,0] and erasing the [1]
    # offset from iteration 1.
    _rewrite_memlets_with_offset(inner_sdfg, inner_name, offset_dims, collapsed_dims)

    # Transform Subscript nodes during ``expr.replace(SubscriptClass, fn)``.
    # SymPy invokes ``fn(*matched.args)`` (the matched expression's args
    # are splatted positionally, NOT passed as the matched node itself),
    # so the callback signature has to accept ``*args`` matching the
    # Subscript's arity: ``args[0]`` is the subscripted container and
    # ``args[1:]`` are the indices.
    #
    # These uncollapse callbacks + the interstate-edge / loop-head rewrites
    # below run BEFORE the ``replace_dict`` rename and match ``inner_name``
    # (the still-distinct connector array), NOT the post-rename ``outer_name``.
    # This is the interstate-edge analogue of the memlet ordering fix above:
    # when two inner connectors bind the SAME outer array at different offsets
    # (spmv reads ``indptr[i]`` and ``indptr[i+1]`` -> two scalar connectors
    # both widening to outer ``indptr``), matching ``outer_name`` AFTER the
    # rename let a later connector re-collapse an assignment an earlier one had
    # already rewritten (``row_start = indptr[i]`` -> ``indptr[i+1]``),
    # conflating both row bounds to ``indptr[i+1]`` (size-0 reduction buffer).
    def _uncollapse_subscript(*args):
        base = args[0]
        # Compare by NAME, not ``base == sympy.Symbol(inner_name)``: the
        # subscript base is a ``dace.symbolic.symbol`` (DaCe's dtype-carrying
        # ``sympy.Symbol`` subclass), which does NOT compare equal to a freshly
        # built plain ``sympy.Symbol(inner_name)``. The old equality silently
        # failed, so an interstate-edge gather index like
        # ``edge_idx_index = edge_idx[0, 0, 0]`` was rebuilt verbatim instead of
        # uncollapsed to ``edge_idx[jb, jc, 0]`` -- every lane then gathered the
        # same element. (The dataflow-memlet path is unaffected: it rewrites
        # subset ranges directly, never comparing symbols.)
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
        # ``node`` is the actual ``inner_name`` symbol pulled from the
        # expression (a ``dace.symbolic.symbol``); subscript it with the
        # per-axis offsets directly. Building a plain ``sympy.Symbol`` and
        # calling ``node.subs`` would not match the dace symbol (see
        # :func:`_uncollapse_subscript`), leaving the reference uncollapsed.
        return symbolic.Subscript(node, *offset_dims)

    # Per user direction 2026-06-10: "On memlets we use subset 0,0,1 but on codeblocks we
    # treat scalar as if it is a symbol." A true ``dace.data.Scalar`` source has no
    # array dimension to subscript -- the bare symbol reference IS the correct C++
    # form. Only wrap with ``[offset_dims]`` when the source is an Array (i.e. has a
    # dimension to address).
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
                    # Scalar source: keep the bare symbol reference; codegen handles it
                    # the same way it would handle any other free symbol.
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

    # Uncollapse dims in code blocks (loop heads). LoopRegion's loop-head
    # CodeBlocks are ``loop_condition``, ``update_statement`` (per-iter
    # ``i = i + 1`` style update -- not a pure expression), and
    # ``init_statement`` (``i = 0`` style). Each may be ``None`` when
    # omitted, and assignment-style statements aren't parseable by
    # :func:`pystr_to_symbolic`; rewrite when parseable, leave alone
    # otherwise (those statements don't reference connector arrays).
    def _rewrite_codeblock(cb):
        if cb is None:
            return cb
        try:
            sym = symbolic.pystr_to_symbolic(cb.as_string)
        except Exception:
            return cb
        return CodeBlock(symbolic.symstr(sym.replace(symbolic.Subscript, _uncollapse_subscript)))

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
                    symbolic.symstr(
                        symbolic.pystr_to_symbolic(cond.as_string).replace(symbolic.Subscript,
                                                                           _uncollapse_subscript))), body)

    inner_sdfg.replace_dict({inner_name: outer_name})

    # Replace connectors. When the rename target ``outer_name`` already has
    # an edge on this side (because a previous iteration of this helper
    # already merged ANOTHER inner connector binding the same outer array,
    # e.g. ``A[1,i,j]`` -> ``__tmp_a`` AND ``A[0,i,j]`` -> ``__tmp_b`` BOTH
    # bind to outer ``A``), keep ONE merged edge and REMOVE the duplicate.
    # The merged edge's subset is the full-array slice ``_full_subset`` so
    # the post-widening contract holds regardless of which inner connector
    # contributed which slice.
    assert direction in ('in', 'out')
    if direction == 'in':
        existing_target = any(e.dst_conn == outer_name and e.dst_conn != inner_name for e in state.in_edges(nsdfg_node))
        for iedge in list(state.in_edges(nsdfg_node)):
            if iedge.dst_conn == inner_name:
                if existing_target:
                    # Drop this edge -- the existing ``outer_name`` edge
                    # already covers the full-array subset.
                    state.remove_edge(iedge)
                else:
                    iedge.dst_conn = outer_name
                    iedge.data.subset = _full_subset(state.sdfg, outer_name)
                    # The inner descriptor now mirrors the full outer array and
                    # the inner memlets carry the per-iteration offset (rewritten
                    # above). Any ``other_subset`` on the boundary edge described
                    # the OLD (collapsed / length-1) inner shape -- e.g. a K>=2
                    # broadcast read ``a[i//2] -> inner_a[0]`` with inner ``(1,)``
                    # -- and is stale after widening. Clear it so the edge is a
                    # clean full-array passthrough (design 2.4).
                    iedge.data.other_subset = None
        nsdfg_node.remove_in_connector(inner_name)
        if outer_name not in nsdfg_node.in_connectors:
            nsdfg_node.add_in_connector(outer_name, force=True)
    else:
        existing_target = any(e.src_conn == outer_name and e.src_conn != inner_name
                              for e in state.out_edges(nsdfg_node))
        for oedge in list(state.out_edges(nsdfg_node)):
            if oedge.src_conn == inner_name:
                if existing_target:
                    state.remove_edge(oedge)
                else:
                    oedge.src_conn = outer_name
                    oedge.data.subset = _full_subset(state.sdfg, outer_name)
                    oedge.data.other_subset = None  # stale after widening (see in-edge note)
        nsdfg_node.remove_out_connector(inner_name)
        if outer_name not in nsdfg_node.out_connectors:
            nsdfg_node.add_out_connector(outer_name, force=True)

    # (Inner memlets, interstate-edge assignments and loop/branch heads were all
    # rewritten above with the per-inner_name offsets, BEFORE replace_dict, to
    # avoid clobbering an earlier connector's rewrite when two connectors bind
    # the same outer array.)


class ExpandNestedSDFGInputs(transformation.SingleStateTransformation):
    """Pre-processor for :class:`InlineMultistateSDFG`: widen narrowed
    NSDFG in/out memlets to full-array subsets, reshape inner descriptors
    to match, and offset inner memlets accordingly.

    Handles both top-level NSDFGs and NSDFGs nested inside a Map scope.
    For the in-map case the caller is expected to have first widened the
    parent map's IN/OUT memlets to full arrays (via
    ``propagate_full_array_subsets_through_map``) so the
    MapEntry-to-NSDFG connector already carries the full extent; the
    per-iteration tile offset is then captured from the original
    narrowed subset and threaded onto every inner memlet referencing
    that connector.
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
        # Refuse when every in/out edge already reads the full outer
        # array -- nothing to widen, and an unconditional re-apply would
        # spin the orchestrator's ``apply_transformations_repeated``
        # forever.
        for edge in (*state.in_edges(nsdfg_node), *state.out_edges(nsdfg_node)):
            if edge.data is None or edge.data.data is None:
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

        # Symbols introduced by the offsets that need to be propagated
        # from the outer scope into the NSDFG (added to
        # ``symbol_mapping`` + the inner ``symbols`` table so the inner
        # memlet references validate).
        introduced_symbols: Set[str] = set()

        # Track which inner array names we have already widened. When a
        # connector name is used for BOTH an in-edge and an out-edge
        # (the same outer array is read AND written by the NSDFG;
        # ``MapToForLoop`` does this when the kernel mutates the array
        # in-place, e.g. ``A[i,j,k+1] = A[i,j,k] + A[i,j,k-1]``), the
        # inner array is shared. Without de-duplication we would offset
        # every inner memlet referencing that array TWICE, corrupting
        # numerics.
        processed_inner_arrays: Set[str] = set()

        read_subsets = _collect_read_subsets(state, nsdfg_node)
        write_subsets = _collect_write_subsets(state, nsdfg_node)
        inner_sdfg = nsdfg_node.sdfg

        # Collect read subsets
        # The array might appear in multiple edges, so we need to keep track per inner name
        for iedge in state.in_edges(nsdfg_node):
            if iedge.data is None or iedge.data.data is None:
                continue
            # other_subset tolerated -- folded to None at widen time.
            in_conn = iedge.dst_conn
            if in_conn is None:
                continue

            # If read subset if already full, skip, else we need to widen it.
            # First we need to check if inner descriptor collapses any dimensions, if yes we need to widen it
            inner_desc = inner_sdfg.arrays[in_conn]
            # Assume outer shape is: [N, N, N]
            # Read: [1, 0:N, 0:N] -> inner shape is [N, N]
            # We need to expand inner shapes to be all length 0.
            # Do this by finding collapsed dimensions:
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

        for (conn, (outer_arr_name, outer_subset, collapsed_dims)) in read_subsets.items():
            _replace_desc_and_uncollapse_dims(nsdfg_node,
                                              state,
                                              conn,
                                              outer_arr_name,
                                              sdfg.arrays[outer_arr_name],
                                              collapsed_dims, [lo for (lo, _hi, _stp) in outer_subset.ranges],
                                              direction='in')

        for (conn, (outer_arr_name, outer_subset, collapsed_dims)) in write_subsets.items():
            _replace_desc_and_uncollapse_dims(nsdfg_node,
                                              state,
                                              conn,
                                              outer_arr_name,
                                              sdfg.arrays[outer_arr_name],
                                              collapsed_dims, [lo for (lo, _hi, _stp) in outer_subset.ranges],
                                              direction='out')

        defined_syms = set(sdfg.arrays.keys()) | set(inner_sdfg.symbols.keys()) | set(nsdfg_node.symbol_mapping.keys())
        for conn, (outer_arr_name, outer_subset, collapsed_dims) in read_subsets.items():
            free_syms = outer_subset.free_symbols - defined_syms
            introduced_symbols.update(str(s) for s in free_syms)
        for conn, (outer_arr_name, outer_subset, collapsed_dims) in write_subsets.items():
            free_syms = outer_subset.free_symbols - defined_syms
            introduced_symbols.update(str(s) for s in free_syms)
        # Per user direction 2026-06-10: all symbols needed for any array's shape /
        # strides / offsets must be added to the inner NSDFG (if not present) AND
        # bound in symbol_mapping (identity if no entry). Use ``Data.free_symbols``
        # (which aggregates shape + strides + offset/start_offset free symbols)
        # instead of walking each field by hand.
        for inner_arr_name, inner_desc in inner_sdfg.arrays.items():
            for sym in inner_desc.free_symbols:
                sym_name = str(sym)
                if sym_name in nsdfg_node.in_connectors or sym_name in nsdfg_node.out_connectors:
                    continue
                if sym_name not in nsdfg_node.symbol_mapping:
                    introduced_symbols.add(sym_name)

        # Propagate any offset symbols not already passed in via
        # symbol_mapping. Identity binding is the right default --
        # outer ``ii`` becomes inner ``ii``. The validation step will
        # complain on the next pass if the symbol isn't defined at the
        # outer scope where the NSDFG sits; that's a real bug in the
        # caller and we want it surfaced.
        #
        # Symbol-type preservation: when a symbol isn't already in the
        # inner ``symbols`` table we resolve its type by walking the
        # outer SDFG ancestry (parent_sdfg chain) + LoopRegion loop
        # variables. Defaulting to ``int64`` silently would mismatch a
        # caller that declared the symbol with another integer width.
        for sym_name in introduced_symbols:
            if sym_name in nsdfg_node.symbol_mapping:
                continue
            if sym_name in inner.arrays:
                continue  # an array name -- not a symbol, skip
            if sym_name in inner.symbols:
                # Symbol exists in inner but not mapped from outer -- bind it.
                nsdfg_node.symbol_mapping[sym_name] = symbolic.pystr_to_symbolic(sym_name)
                continue
            # Symbol is new to both inner and the mapping. Resolve its
            # type from the outer scope and copy it through.
            outer_type = _resolve_outer_symbol_type(sym_name, sdfg)
            inner.add_symbol(sym_name, outer_type)
            nsdfg_node.symbol_mapping[sym_name] = symbolic.pystr_to_symbolic(sym_name)
