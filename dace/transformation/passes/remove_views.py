# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Eliminates View access nodes from an SDFG by composing the view's address
mapping into the viewed array.

A View node V between a viewed array A and a subgraph a
map ``phi : index_space(V) -> index_space(A)`` encoded by the
view edge memlet. We can rewrite memlets m
that references V in terms of A by composing ``m' = phi.m``.  After
rewriting, V has no remaining uses and can be removed.

Two strategies are used depending on the view kind:

Four strategies are tried in order until one succeeds:
 
1. **Slice / squeeze / unsqueeze** --
   :func:`dace.sdfg.utils.map_view_to_array` matches view strides to
   array strides and returns a per-dimension mapping.  Memlet subsets are
   rewritten by affine composition: ``new = rb + rs * v``.
 
1b. **Mapping from view edge subset** -- when ``map_view_to_array`` fails,
   the mapping is derived directly from the view edge subset: dimensions with
   size > 1 are data dimensions, size-1 dimensions are squeezed.  Handles
   strided slices like ``A[0:M:2, col]``.
 
2. **Dense reshape** -- both descriptors are gap-free
   (``total_size == prod(shape)``) with the same element count.  Each
   memlet range ``(b, e, s)`` is linearized using ``desc.strides``, then
   delinearized via the mixed-radix formula
   ``(flat // stride_d) % shape_d``.  Python tasklet code is rewritten
   analogously via AST transformation. (This type of tasklet should not exist,
   but we can't prevent them from being written as they also appear in unit
   tests)
 
3. **Pure linearization** -- same as strategy 2 but without the
   dense guard.  Relies on per-memlet feasibility only: if the flat
   offset range can be delinearized, the view is removed.  Catches
   padded or non-standard layouts.
"""

import ast
import copy
import functools
import operator
from typing import Any, Dict, List, Optional, Set, Tuple
 
from dace import SDFG, SDFGState, config, data as dt, dtypes, properties, subsets, symbolic
from dace.sdfg import nodes as nd, utils as sdutil, graph as gr
from dace.transformation import pass_pipeline as ppl, transformation
 
_PASS = 'RemoveViews'
_DEBUGPRINT = config.Config.get('debugprint') in (True, '1', 'true', 'yes')
 
def _fmt_desc(desc, name):
    """One-line summary of a data descriptor."""
    kind = type(desc).__name__
    return f'{name}: {kind}{list(desc.shape)} strides={list(desc.strides)}'
 
 
# ---------------------------------------------------------------------------
# Helpers -- slice / squeeze / unsqueeze path
# ---------------------------------------------------------------------------
 
def _classify_view(
    state: SDFGState,
    view_node: nd.AccessNode,
    sdfg: SDFG,
) -> Optional[Tuple[nd.AccessNode, gr.MultiConnectorEdge, subsets.Range, bool]]:
    """
    Classifies a View access node.
 
    :return: ``(viewed_node, view_edge, viewed_subset, is_viewed_src)``
             or None if the view is not removable.  ``is_viewed_src`` is
             True when the viewed array feeds into the view.
    """
    desc = sdfg.arrays[view_node.data]
    # Not view
    if not isinstance(desc, dt.View):
        return None

    # Can't handle structs or container arrays
    if isinstance(desc, (dt.StructureView, dt.ContainerView)):
        return None

    # Check view is valid
    view_edge = sdutil.get_view_edge(state, view_node)
    if view_edge is None:
        return None

    # Distinguish Array -> View or View -> Array
    if view_edge.dst is view_node:
        mpath = state.memlet_path(view_edge)
        viewed_node = mpath[0].src
        if not isinstance(viewed_node, nd.AccessNode):
            return None
        viewed_subset = view_edge.data.get_src_subset(view_edge, state)
        is_viewed_src = True
    else:
        mpath = state.memlet_path(view_edge)
        viewed_node = mpath[-1].dst
        if not isinstance(viewed_node, nd.AccessNode):
            return None
        viewed_subset = view_edge.data.get_dst_subset(view_edge, state)
        is_viewed_src = False
 
    viewed_desc = sdfg.arrays[viewed_node.data]
    assert viewed_desc.dtype == desc.dtype, "View and array must have the same dtype"
 
    if viewed_subset is None:
        viewed_subset = subsets.Range.from_array(viewed_desc)
 
    return viewed_node, view_edge, viewed_subset, is_viewed_src
 
def _derive_mapping_from_subset(
    viewed_subset: subsets.Range,
    view_ndim: int,
) -> Optional[Dict[int, int]]:
    """
    Derive a view-dim -> array-dim mapping directly from the view edge
    subset. Dimensions with size == 1 are squeezed.

    Returns the mapping dict, or None on mismatch.
    """
    sizes = viewed_subset.size()
    data_dims = [d for d, s in enumerate(sizes) if s != 1]
    if len(data_dims) != view_ndim:
        return None
    return {vd: ad for vd, ad in enumerate(data_dims)}
 
 
def _compute_rewritten_subset(
    mapping: Dict[int, int],
    view_subset: subsets.Range,
    edge_subset: subsets.Range,
) -> subsets.Range:
    """
    Composes ``edge_subset`` (in view space) with ``view_subset`` (in array
    space) using the dimension ``mapping`` from :func:`map_view_to_array`.
 
    For each mapped dimension, applies the affine composition:
 
        new_start = rb + rs * vb
        new_end   = rb + rs * ve
        new_step  = rs * vs
 
    where ``(rb, re, rs)`` is the array-side range from ``view_subset``
    and ``(vb, ve, vs)`` is the view-side range from ``edge_subset``.
    """
    new_ranges: List[Tuple] = list(view_subset.ndrange())
    for vdim, adim in mapping.items():
        rb, re, rs = new_ranges[adim]
        vb, ve, vs = edge_subset.ranges[vdim]
        new_ranges[adim] = (rb + rs * vb, rb + rs * ve, rs * vs)
 
    return subsets.Range(new_ranges)


def _int_shape(desc: dt.Data) -> Optional[List[int]]:
    """Return the shape as a list of Python ints, or None if symbolic."""
    try:
        return [int(s) for s in desc.shape]
    except (TypeError, ValueError):
        return None


def _int_strides(desc: dt.Data) -> Optional[List[int]]:
    """Return the strides as a list of Python ints, or None if symbolic."""
    try:
        return [int(s) for s in desc.strides]
    except (TypeError, ValueError):
        return None


def _is_dense_reshape(vdesc: dt.Data, adesc: dt.Data) -> bool:
    """
    True if ``vdesc`` is a dense reshape of ``adesc``: same dtype, same
    element count, and neither descriptor has gaps (``total_size ==
    prod(shape)``).  All comparisons are symbolic.
    """
    if vdesc.dtype != adesc.dtype:
        return False
    vprod = functools.reduce(operator.mul, vdesc.shape, 1)
    aprod = functools.reduce(operator.mul, adesc.shape, 1)
    if symbolic.equal(vprod, aprod) is not True:
        return False
    if symbolic.equal(vdesc.total_size, vprod) is not True:
        return False
    if symbolic.equal(adesc.total_size, aprod) is not True:
        return False
    return True
 
 
def _delinearize_flat(flat, astrides: List[int], array_shape: List[int]):
    """
    Convert a flat offset to multi-dimensional indices via the
    mixed-radix decomposition ``(flat // stride_d) % shape_d``
    per dimension.
    """
    if len(array_shape) == 1:
        return [flat]
    return [(flat // astr) % shp
            for astr, shp in zip(astrides, array_shape)]
 
 
# ---- Subset rewriting for reshapes ----------------------------------------
 
def _reshape_subset(
    edge_subset: subsets.Range,
    vstrides: List[int],
    view_shape: List[int],
    astrides: List[int],
    array_shape: List[int],
) -> Optional[subsets.Range]:
    """
    Rewrite ``edge_subset`` from view coordinates to array coordinates
    by linearizing each range ``(b, e, s)`` with ``vstrides``, then
    delinearizing with ``astrides``.
 
    :return: New Range in array coordinates, or None on failure.
    """
    sizes = edge_subset.size()
 
    # --- Case 1: single element (all dims (e+1-b)//s == 1 for all dims) ----
    if all(s == 1 for s in sizes):
        starts = edge_subset.min_element()
        flat = sum(s * vs for s, vs in zip(starts, vstrides))
        indices = _delinearize_flat(flat, astrides, array_shape)
        return subsets.Range([(idx, idx, 1) for idx in indices])
 
    # --- Case 2: full view range -------------------------------------------
    try:
        is_full = all(
            int(r[0]) == 0 and int(r[1]) == d - 1 and int(r[2]) == 1
            for r, d in zip(edge_subset.ranges, view_shape)
        )
    except (TypeError, ValueError):
        is_full = False
    if is_full:
        return subsets.Range([(0, d - 1, 1) for d in array_shape])
 
    # --- Case 3: general -- linearize, check contiguity, delinearize -------
    # Requires the subset to be contiguous in the view's flat space, otherwise the
    # delinearization would need to handle multiple distinct step sizes.
    starts = [r[0] for r in edge_subset.ranges]
    ends = [r[1] for r in edge_subset.ranges]
    flat_start = sum(s * vs for s, vs in zip(starts, vstrides))
    flat_end = sum(e * vs for e, vs in zip(ends, vstrides))
 
    total_elements = functools.reduce(operator.mul, sizes, 1)
    try:
        if int(flat_end - flat_start + 1) != int(total_elements):
            return None  # non-contiguous in flat space, we can't handle this
    except (TypeError, ValueError):
        return None
 
    start_indices = _delinearize_flat(flat_start, astrides, array_shape)
    end_indices = _delinearize_flat(flat_end, astrides, array_shape)
    return subsets.Range([(s, e, 1) for s, e in zip(start_indices, end_indices)])
 
 
# ---- AST rewriting for Python tasklets ------------------------------------
 
def _try_constant_fold(node: ast.expr) -> ast.expr:
    """Try to evaluate an AST expression to a Python int constant."""
    try:
        code = compile(ast.Expression(body=ast.fix_missing_locations(node)),
                       '<fold>', 'eval')
        result = eval(code)
        if isinstance(result, int):
            return ast.Constant(value=result)
    except Exception:
        pass
    return node
 
 
class _ReshapeIndexRewriter(ast.NodeTransformer):
    """
    Rewrites ``connector[i0, i1, ...]`` subscript expressions in a
    Python tasklet AST from view coordinates to array coordinates via
    linearization with ``vstrides`` and delinearization with ``astrides``.

    Example (view shape [2, 10], vstrides [10, 1] -> array shape [20], astrides [1]):
        out[1, 3]  ->  out[13]           (constant: 1*10 + 3 = 13)
        out[i, j]  ->  out[10*i + j]     (symbolic)

    Example (view shape [20], vstrides [1] -> array shape [4, 5], astrides [5, 1]):
        inp[13]    ->  inp[2, 3]          (constant: 13//5=2, 13%5=3)
        inp[k]     ->  inp[k // 5, k % 5] (symbolic)
    """

    def __init__(self, connector: str,
                 vstrides: List[int], astrides: List[int],
                 array_shape: List[int]):
        self.connector = connector
        self.vstrides = vstrides
        self.astrides = astrides
        self.array_shape = array_shape
        self.changed = False

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)

        # Only rewrite subscripts on our connector, e.g. ``out[...]``
        if not (isinstance(node.value, ast.Name)
                and node.value.id == self.connector):
            return node

        # Unpack index expressions
        if isinstance(node.slice, ast.Tuple):
            indices = list(node.slice.elts)
        else:
            indices = [node.slice]

        if len(indices) != len(self.vstrides):
            return node

        # Step 1: linearize view indices to a flat offset expression
        #   flat = i0 * vstride[0] + i1 * vstride[1] + ...
        flat_expr = self._linearize(indices)
        flat_expr = _try_constant_fold(flat_expr)

        # Step 2: delinearize flat offset to array indices
        #   idx_d = (flat // astride[d]) % shape[d]
        new_indices = self._delinearize(flat_expr)
        new_indices = [_try_constant_fold(idx) for idx in new_indices]

        new_slice = (new_indices[0] if len(new_indices) == 1
                     else ast.Tuple(elts=new_indices, ctx=ast.Load()))
        self.changed = True
        return ast.fix_missing_locations(
            ast.Subscript(value=node.value, slice=new_slice, ctx=node.ctx))

    def _linearize(self, indices: List[ast.expr]) -> ast.expr:
        """
        Build an AST expression for the flat offset:
            flat = idx[0] * vstride[0] + idx[1] * vstride[1] + ...

        Skips zero-stride dimensions and avoids multiplying by 1.
        """
        result = None
        for idx, vs in zip(indices, self.vstrides):
            if vs == 0:
                continue
            # Build term: idx * vstride  (or just idx when vstride == 1)
            term = (idx if vs == 1
                    else ast.BinOp(left=idx, op=ast.Mult(),
                                   right=ast.Constant(value=vs)))
            # Accumulate: result + term
            result = (term if result is None
                      else ast.BinOp(left=result, op=ast.Add(), right=term))
        return result if result is not None else ast.Constant(value=0)

    def _delinearize(self, flat: ast.expr) -> List[ast.expr]:
        """
        Build AST expressions for the mixed-radix decomposition:
            idx_d = (flat // astride[d]) % shape[d]

        For a 1D target array, returns [flat] directly.
        Skips ``// 1`` when astride == 1 for the last dimension.
        """
        if len(self.array_shape) == 1:
            return [flat]
        out: List[ast.expr] = []
        for astr, ashp in zip(self.astrides, self.array_shape):
            # flat // stride  (skip division when stride == 1)
            expr = (flat if astr == 1
                    else ast.BinOp(left=flat, op=ast.FloorDiv(),
                                   right=ast.Constant(value=astr)))
            # (flat // stride) % shape  -- extract this dimension's digit
            expr = ast.BinOp(left=expr, op=ast.Mod(),
                             right=ast.Constant(value=ashp))
            out.append(expr)
        return out
 
@properties.make_properties
@transformation.explicit_cf_compatible
class RemoveViews(ppl.Pass):
    def modifies(self) -> ppl.Modifies:
        return (ppl.Modifies.Descriptors
                | ppl.Modifies.AccessNodes
                | ppl.Modifies.Memlets
                | ppl.Modifies.Tasklets)
 
    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.AccessNodes)
 
    def depends_on(self):
        return set()
 
    def apply_pass(
        self,
        sdfg: SDFG,
        pipeline_results: Dict[str, Any],
    ) -> Optional[Set[str]]:
        removed: Set[str] = set()
 
        iteration = 0
        changed = True
        while changed:
            changed = False
            iteration += 1
            if _DEBUGPRINT:
                print(f'[{_PASS}] --- fixpoint iteration {iteration} ---')
            for state in sdfg.states():
                changed |= self._process_state(sdfg, state, removed)
 
        for name in list(removed):
            if name in sdfg.arrays:
                still_used = any(
                    isinstance(n, nd.AccessNode) and n.data == name
                    for st in sdfg.states()
                    for n in st.nodes()
                )
                if not still_used:
                    sdfg.remove_data(name, validate=False)
                    if _DEBUGPRINT:
                        print(f'[{_PASS}] garbage-collected descriptor'
                              f' "{name}"')
 
        if _DEBUGPRINT:
            if removed:
                print(f'[{_PASS}] === done: removed {len(removed)} views:'
                      f' {removed} ===')
            else:
                print(f'[{_PASS}] === done: nothing to remove ===')
 
        return removed or None
 
    def report(self, pass_retval: Set[str]) -> str:
        return f'Removed {len(pass_retval)} views: {pass_retval}.'
 
    # ---- Per-state logic ---------------------------------------------------
 
    def _process_state(self, sdfg, state, removed):
        changed = False
 
        view_nodes = [
            n for n in state.data_nodes()
            if isinstance(sdfg.arrays.get(n.data), dt.View)
        ]
 
        if _DEBUGPRINT and view_nodes:
            print(f'[{_PASS}] state "{state.label}": found {len(view_nodes)}'
                  f' view node(s): {[n.data for n in view_nodes]}')
 
        for vnode in view_nodes:
            if vnode not in state.nodes():
                continue
 
            info = _classify_view(state, vnode, sdfg)
            if info is None:
                if _DEBUGPRINT:
                    print(f'[{_PASS}]   "{vnode.data}": cannot classify'
                          f' (StructureView / no view edge / dtype mismatch)'
                          f' -- skipping')
                continue
            viewed_node, view_edge, viewed_subset, is_viewed_src = info
 
            if _DEBUGPRINT:
                vdesc = sdfg.arrays[vnode.data]
                adesc = sdfg.arrays[viewed_node.data]
                direction = ('read' if is_viewed_src
                             else 'write')
                print(f'[{_PASS}]   "{vnode.data}" -> "{viewed_node.data}"'
                      f' ({direction})')
                print(f'[{_PASS}]     view:  {_fmt_desc(vdesc, vnode.data)}')
                print(f'[{_PASS}]     array: {_fmt_desc(adesc, viewed_node.data)}')
                print(f'[{_PASS}]     view edge subset: {viewed_subset}')
 
 
            vdesc = sdfg.arrays[vnode.data]
            adesc = sdfg.arrays[viewed_node.data]
 
            # --- Strategy 1: slice / squeeze / unsqueeze --------------------
            mapping_result = sdutil.map_view_to_array(
                vdesc, adesc, viewed_subset)
            if mapping_result is not None:
                mapping, _unsqueezed, _squeezed = mapping_result
                if _DEBUGPRINT:
                    print(f'[{_PASS}]     strategy 1 (map_view_to_array):'
                          f' mapping={mapping},'
                          f' squeezed={_squeezed},'
                          f' unsqueezed={_unsqueezed}')
                self._rewrite_memlets(
                    state, vnode, viewed_node, view_edge, viewed_subset,
                    mapping, is_viewed_src)
                self._reconnect_edges(
                    state, vnode, viewed_node, view_edge, is_viewed_src)
                state.remove_node(vnode)
                removed.add(vnode.data)
                changed = True
                if _DEBUGPRINT:
                    print(f'[{_PASS}]     REMOVED "{vnode.data}"'
                          f' via strategy 1')
                continue
 
            if _DEBUGPRINT:
                print(f'[{_PASS}]     strategy 1: map_view_to_array'
                      f' returned None')
 
            # --- Strategy 1b: derive mapping from view edge subset ----------
            mapping_1b = _derive_mapping_from_subset(
                viewed_subset, len(vdesc.shape))
            if mapping_1b is not None:
                if _DEBUGPRINT:
                    print(f'[{_PASS}]     strategy 1b'
                          f' (derive_mapping_from_subset):'
                          f' mapping={mapping_1b}')
                self._rewrite_memlets(
                    state, vnode, viewed_node, view_edge, viewed_subset,
                    mapping_1b, is_viewed_src)
                self._reconnect_edges(
                    state, vnode, viewed_node, view_edge, is_viewed_src)
                state.remove_node(vnode)
                removed.add(vnode.data)
                changed = True
                if _DEBUGPRINT:
                    print(f'[{_PASS}]     REMOVED "{vnode.data}"'
                          f' via strategy 1b')
                continue
 
            if _DEBUGPRINT:
                print(f'[{_PASS}]     strategy 1b:'
                      f' derive_mapping_from_subset returned None')
 
            # --- Strategy 2: dense reshape ----------------------------------
            if self._try_linearize_removal(state, vnode, viewed_node,
                                           view_edge, viewed_subset,
                                           is_viewed_src,
                                           require_dense=True):
                removed.add(vnode.data)
                changed = True
                continue
 
            # --- Strategy 3: pure linearization (last resort) ---------------
            if self._try_linearize_removal(state, vnode, viewed_node,
                                           view_edge, viewed_subset,
                                           is_viewed_src,
                                           require_dense=False):
                removed.add(vnode.data)
                changed = True
                continue
 
            if _DEBUGPRINT:
                print(f'[{_PASS}]     all strategies failed for'
                      f' "{vnode.data}" -- keeping view')
 
        return changed
 
    # ---- Strategy 1 helpers ------------------------------------------------
 
    def _rewrite_memlets(self, state, view_node, viewed_node, view_edge,
                         viewed_subset, mapping, is_viewed_src):
        sdfg = state.parent
        full_view_range = subsets.Range.from_array(sdfg.arrays[view_node.data])
        non_view_edges = (list(state.out_edges(view_node)) if is_viewed_src
                          else list(state.in_edges(view_node)))
        for edge in non_view_edges:
            for tree_edge in state.memlet_tree(edge):
                m = tree_edge.data
                if m.data == view_node.data:
                    old = f'{m.data}[{m.subset}]'
                    m.data = viewed_node.data
                    if m.subset is not None and m.subset == full_view_range:
                        # Full view range: copy the view edge subset
                        # directly to preserve the original end / step.
                        m.subset = copy.deepcopy(viewed_subset)
                    elif m.subset is not None:
                        m.subset = _compute_rewritten_subset(
                            mapping, viewed_subset, m.subset)
                    else:
                        m.subset = copy.deepcopy(viewed_subset)
                    if _DEBUGPRINT:
                        print(f'[{_PASS}]       memlet: {old}'
                              f' -> {m.data}[{m.subset}]')
                elif m.other_subset is not None:
                    old_other = f'other_subset={m.other_subset}'
                    m.other_subset = _compute_rewritten_subset(
                        mapping, viewed_subset, m.other_subset)
                    if _DEBUGPRINT:
                        print(f'[{_PASS}]       memlet {m.data}:'
                              f' {old_other}'
                              f' -> other_subset={m.other_subset}')
 
    def _reconnect_edges(self, state, view_node, viewed_node, view_edge,
                         is_viewed_src):
        if is_viewed_src:
            for e in list(state.out_edges(view_node)):
                if e is view_edge:
                    continue
                if _DEBUGPRINT:
                    print(f'[{_PASS}]       reconnect:'
                          f' {view_node.data}:{e.src_conn}'
                          f' -> {e.dst}:{e.dst_conn}'
                          f'  =>  {viewed_node.data}:{view_edge.src_conn}'
                          f' -> {e.dst}:{e.dst_conn}')
                state.remove_edge(e)
                state.add_edge(viewed_node, view_edge.src_conn,
                               e.dst, e.dst_conn, e.data)
        else:
            for e in list(state.in_edges(view_node)):
                if e is view_edge:
                    continue
                if _DEBUGPRINT:
                    print(f'[{_PASS}]       reconnect:'
                          f' {e.src}:{e.src_conn}'
                          f' -> {view_node.data}:{e.dst_conn}'
                          f'  =>  {e.src}:{e.src_conn}'
                          f' -> {viewed_node.data}:{view_edge.dst_conn}')
                state.remove_edge(e)
                state.add_edge(e.src, e.src_conn,
                               viewed_node, view_edge.dst_conn, e.data)
        if view_edge in state.edges():
            state.remove_edge(view_edge)
 
    # ---- Strategy 2 & 3: linearization via strides --------------------------
 
    def _try_linearize_removal(self, state, vnode, viewed_node,
                               view_edge, viewed_subset, is_viewed_src,
                               require_dense=True):
        """
        Attempt to remove a view by linearizing memlet ranges with the
        view's strides and delinearizing with the array's strides.
        Rewrites Python tasklet code analogously.
 
        When ``require_dense`` is True (strategy 2), both descriptors
        must be gap-free (``total_size == prod(shape)``).  When False
        (strategy 3), the pass skips that global check and relies on
        per-memlet feasibility only.
 
        Linearization via ``vstrides`` assumes the view's strides encode
        physical memory offsets.  This is only true when the view edge
        covers the **full** array; partial views (slices) are handled by
        strategy 1 instead.
 
        :return: True on success.
        """
        sdfg = state.parent
        vdesc = sdfg.arrays[vnode.data]
        adesc = sdfg.arrays[viewed_node.data]
        strat_name = 'strategy 2 (dense reshape)' if require_dense \
                     else 'strategy 3 (pure linearization)'
 
        if require_dense and not _is_dense_reshape(vdesc, adesc):
            if _DEBUGPRINT:
                print(f'[{_PASS}]     {strat_name}: not a dense reshape'
                      f' -- skipping')
            return False
 
        # Linearization via vstrides only works when the view maps to
        # the entire array.  If the view edge is a partial subset
        # (e.g. a strided slice), vstrides are virtual, not physical.
        full_range = subsets.Range.from_array(adesc)
        if viewed_subset != full_range:
            if _DEBUGPRINT:
                print(f'[{_PASS}]     {strat_name}: view edge subset'
                      f' {viewed_subset} != full range {full_range}'
                      f' -- skipping')
            return False
 
        view_shape = _int_shape(vdesc)
        array_shape = _int_shape(adesc)
        vstrides = _int_strides(vdesc)
        astrides = _int_strides(adesc)
        if None in (view_shape, array_shape, vstrides, astrides):
            if _DEBUGPRINT:
                print(f'[{_PASS}]     {strat_name}: symbolic'
                      f' shapes/strides -- skipping')
            return False
        if vdesc.dtype != adesc.dtype:
            if _DEBUGPRINT:
                print(f'[{_PASS}]     {strat_name}: dtype mismatch'
                      f' -- skipping')
            return False
 
        if _DEBUGPRINT:
            print(f'[{_PASS}]     {strat_name}: vstrides={vstrides},'
                  f' astrides={astrides}')
 
        non_view_edges = (list(state.out_edges(vnode)) if is_viewed_src
                          else list(state.in_edges(vnode)))
 
        # -- feasibility: every memlet subset must be reshapable,
        #    every leaf tasklet must be Python. --------------------------
        for edge in non_view_edges:
            for te in state.memlet_tree(edge):
                leaf = te.dst if is_viewed_src else te.src
                if isinstance(leaf, nd.Tasklet):
                    if leaf.language != dtypes.Language.Python:
                        if _DEBUGPRINT:
                            print(f'[{_PASS}]     {strat_name}: non-Python'
                                  f' tasklet "{leaf.label}" -- aborting')
                        return False
                m = te.data
                if m.data == vnode.data and m.subset is not None:
                    if _reshape_subset(m.subset, vstrides, view_shape,
                                       astrides, array_shape) is None:
                        if _DEBUGPRINT:
                            print(f'[{_PASS}]     {strat_name}: cannot'
                                  f' reshape subset {m.data}[{m.subset}]'
                                  f' -- aborting')
                        return False
                if m.data != vnode.data and m.other_subset is not None:
                    if _reshape_subset(m.other_subset, vstrides, view_shape,
                                       astrides, array_shape) is None:
                        if _DEBUGPRINT:
                            print(f'[{_PASS}]     {strat_name}: cannot'
                                  f' reshape other_subset {m.other_subset}'
                                  f' -- aborting')
                        return False
 
        # -- apply: rewrite memlets -----------------------------------------
        for edge in non_view_edges:
            for te in state.memlet_tree(edge):
                m = te.data
                if m.data == vnode.data:
                    old = f'{m.data}[{m.subset}]'
                    m.data = viewed_node.data
                    if m.subset is not None:
                        m.subset = _reshape_subset(
                            m.subset, vstrides, view_shape,
                            astrides, array_shape)
                    else:
                        m.subset = subsets.Range.from_array(adesc)
                    if _DEBUGPRINT:
                        print(f'[{_PASS}]       memlet: {old}'
                              f' -> {m.data}[{m.subset}]')
                elif m.data != vnode.data and m.other_subset is not None:
                    old_other = f'{m.other_subset}'
                    m.other_subset = _reshape_subset(
                        m.other_subset, vstrides, view_shape,
                        astrides, array_shape)
                    if _DEBUGPRINT:
                        print(f'[{_PASS}]       memlet {m.data}:'
                              f' other_subset={old_other}'
                              f' -> {m.other_subset}')
 
        # -- apply: rewrite tasklet Python code -----------------------------
        for edge in non_view_edges:
            for te in state.memlet_tree(edge):
                leaf = te.dst if is_viewed_src else te.src
                conn = te.dst_conn if is_viewed_src else te.src_conn
                if (isinstance(leaf, nd.Tasklet) and conn
                        and leaf.language == dtypes.Language.Python
                        and leaf.code.code):
                    old_code = leaf.code.as_string
                    rw = _ReshapeIndexRewriter(conn, vstrides,
                                               astrides, array_shape)
                    leaf.code.code = [rw.visit(copy.deepcopy(stmt))
                                      for stmt in leaf.code.code]
                    if _DEBUGPRINT and rw.changed:
                        print(f'[{_PASS}]       tasklet "{leaf.label}"'
                              f' connector "{conn}": {old_code!r}'
                              f' -> {leaf.code.as_string!r}')
 
        # -- reconnect and remove -------------------------------------------
        self._reconnect_edges(state, vnode, viewed_node,
                              view_edge, is_viewed_src)
        state.remove_node(vnode)
        if _DEBUGPRINT:
            print(f'[{_PASS}]     REMOVED "{vnode.data}"'
                  f' via {strat_name}')
        return True
 