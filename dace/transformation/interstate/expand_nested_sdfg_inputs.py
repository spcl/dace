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

A connector of lower rank than its outer array (axis-collapse, ``a[0:1, 0:M]`` as 1-D ``[M]``) is widened
back to the outer rank. Refuses when the outer array is absent from the parent SDFG (orphan descriptor).
"""
import ast
import copy
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from dace import SDFG, dtypes, subsets, symbolic, data
from dace.codegen.common import CodeBlock
from dace.frontend.python import astutils
from dace.properties import Property, make_properties
from dace.sdfg import SDFGState, nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation import transformation
from dace.transformation.passes.analysis import scopes
from dace.subsets import Range
from dace.memlet import Memlet
import sympy
from dace.sdfg.narrowing import as_expr, as_range


class _RenameLoadName(ast.NodeTransformer):
    """Rename every ``Load`` use of ``old`` to ``new`` in a tasklet's Python code."""

    def __init__(self, old: str, new: str):
        self._old = old
        self._new = new

    def visit_Name(self, node: ast.Name):
        if node.id == self._old and isinstance(node.ctx, ast.Load):
            return ast.copy_location(ast.Name(id=self._new, ctx=ast.Load()), node)
        return node


def _rewrite_scalar_reads_in_tasklets(inner_sdfg: SDFG, inner_name: str, outer_name: str,
                                      offset_dims: List[sympy.Basic]) -> None:
    """Convert a folded scalar read of a widened connector into a dataflow read.

    A scalar NSDFG input can be consumed SYMBOLICALLY -- its value bound into an interstate
    assignment (``C_slice = tmp``) that ``ConstantPropagation`` then inlines into a tasklet's code
    (``__out = tmp * ...``). Once :func:`_replace_desc_and_uncollapse_dims` widens ``tmp`` to the full
    array ``outer_name``, that bare name is an array in a scalar expression -- invalid C (pointer
    arithmetic on ``const T*``). A tasklet cannot index an array in its code either; the value must
    arrive through a CONNECTOR. So for every tasklet that reads ``inner_name`` as a bare (non-
    connector) name, add an input connector fed by the single element ``outer_name[offset_dims]`` and
    rewrite the code reference to that connector. Runs BEFORE ``replace_dict`` so it matches the still-
    distinct ``inner_name``; the memlet already names the widened ``outer_name`` descriptor."""
    for state in inner_sdfg.states():
        for tnode in [n for n in state.nodes() if isinstance(n, nodes.Tasklet)]:
            if tnode.language != dtypes.Language.Python:
                # The code below is parsed as PYTHON. A C++ tasklet is not, and the scatter guard's
                # C++ tasklet (e.g. the tile vectorizer's remainder guard) raises SyntaxError here rather than
                # being skipped. Nothing to rewrite either way: a non-Python tasklet reads its
                # symbols directly, not through this connector-folding path.
                continue
            if inner_name in tnode.in_connectors:
                continue  # already a dataflow read, not a symbolic one
            loads = {
                n.id
                for n in ast.walk(ast.parse(tnode.code.as_string))
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
            }
            if inner_name not in loads:
                continue

            cin = "__" + inner_name + "_elem"
            while cin in tnode.in_connectors or cin in tnode.out_connectors:
                cin += "_"
            tnode.add_in_connector(cin)
            read = state.add_read(outer_name)
            memlet = Memlet(data=outer_name, subset=Range([(o, o, 1) for o in offset_dims]))
            scope = state.entry_node(tnode)
            if scope is None:
                state.add_edge(read, None, tnode, cin, memlet)
            else:
                state.add_memlet_path(read, scope, tnode, dst_conn=cin, memlet=memlet)

            tree = ast.parse(tnode.code.as_string)
            _RenameLoadName(inner_name, cin).visit(tree)
            ast.fix_missing_locations(tree)
            tnode.code = CodeBlock(astutils.unparse(tree))


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


def keeps_absolute_index(lo: Union[int, sympy.Basic], offset: Union[int, sympy.Basic],
                         inner_shape: Tuple[Union[int, sympy.Basic], ...], dim: int) -> bool:
    """True if inner begin ``lo`` of axis ``dim`` is an absolute in-place access (``lo == offset``).
    A constant ``lo`` inside the inner extent is relative even when it equals the window start
    (``A[4, 1:3]`` written at inner ``[0, 1]``)."""
    lo_expr = sympy.sympify(lo)
    if as_expr(lo_expr) - as_expr(sympy.sympify(offset)) != 0:
        return False
    in_extent = (lo_expr.is_Integer and as_expr(lo_expr) >= 0 and dim < len(inner_shape) and bool(
        (sympy.sympify(inner_shape[dim]) - lo_expr).is_positive))
    return not in_extent


def widened_range(lo: sympy.Basic, hi: sympy.Basic, stp: sympy.Basic, offset: sympy.Basic,
                  step: sympy.Basic) -> Tuple[sympy.Basic, sympy.Basic, sympy.Basic]:
    """An inner range of a window starting at ``offset`` with outer step ``step``, in outer coordinates: inner
    position ``p`` is outer ``offset + p * step``."""
    # a single element keeps its own step: there is nothing to stride over
    return (offset + as_expr(lo) * as_expr(step), offset + as_expr(hi) * as_expr(step),
            stp if lo == hi else as_expr(stp) * as_expr(step))


def outer_indices(indices: Sequence[sympy.Basic], offset_dims: List[sympy.Basic], collapsed_dims: List[bool],
                  step_dims: List[sympy.Basic], inner_shape: Tuple) -> List[sympy.Basic]:
    """Map the inner indices of a symbolic subscript to the outer array, as ``_rewrite_memlets_with_offset`` maps
    memlet begins: a full-rank subscript maps each axis, a rank-reduced one reads collapsed axes at their offset."""
    if len(indices) == len(offset_dims):
        return [
            index if keeps_absolute_index(index, offset, inner_shape, dim) else offset + as_expr(index) * as_expr(step)
            for dim, (index, offset, step) in enumerate(zip(indices, offset_dims, step_dims))
        ]
    surviving = iter(indices)
    return [
        offset if collapsed else offset + as_expr(next(surviving)) * as_expr(step)
        for offset, collapsed, step in zip(offset_dims, collapsed_dims, step_dims)
    ]


def uncollapsed_indices(indices: Sequence[sympy.Basic], offset_dims: List[sympy.Basic],
                        collapsed_dims: List[bool]) -> List[sympy.Basic]:
    """Reinsert the collapsed axes of an already offset subscript at their offsets."""
    surviving = iter(indices)
    return [offset if collapsed else next(surviving) for offset, collapsed in zip(offset_dims, collapsed_dims)]


def window_steps(outer_subset: subsets.Range, collapsed_dims: List[bool], inner_desc: data.Data,
                 outer_desc: data.Data) -> List[sympy.Basic]:
    """Per outer dim, the step an inner index is scaled by when the window is widened: the window's step where
    the inner array is a COMPACT view of it (inner stride = outer stride * step, so ``x[k]`` is element ``k``
    of ``a[0:N:2]``), ``1`` where the inner array keeps the outer stride and its indices already carry the
    step (``x[2*i]`` over the same window)."""
    full_rank = len(inner_desc.shape) == len(outer_subset.ranges)
    steps, inner_dim = [], 0
    for d, ((_lo, _hi, stp), collapsed) in enumerate(zip(outer_subset.ranges, collapsed_dims)):
        if not full_rank and collapsed:
            steps.append(1)
            continue
        inner_stride = inner_desc.strides[d if full_rank else inner_dim]
        inner_dim += 1
        compact = stp != 1 and symbolic.simplify(inner_stride - outer_desc.strides[d] * stp) == 0
        steps.append(stp if compact else 1)
    return steps


def widen_far_side_of_copy(state: SDFGState, edge: MultiConnectorEdge, inner_name: str, inner_shape: Tuple,
                           outer_ranges: Callable[[list], Tuple[list, bool]]) -> None:
    """A copy whose memlet names the OTHER array addresses ``inner_name`` through ``other_subset``, or,
    with none, through the whole of it. Once ``inner_name`` becomes a window of the outer array that
    side has to move with it: ``tmp -> x`` with ``x`` widened to ``s[j]`` otherwise writes ``s[0]``."""
    endpoints = (edge.src, edge.dst)
    if not any(isinstance(n, nodes.AccessNode) and n.data == inner_name for n in endpoints):
        return
    memlet = edge.data
    far = memlet.other_subset
    if far is None:
        far = subsets.Range([(0, s - 1, 1) for s in inner_shape])
    memlet.other_subset = subsets.Range(outer_ranges(far.ranges)[0])


def remap_reduce_axes(node: nodes.Node, collapsed_dims: List[bool]) -> None:
    """A ``Reduce`` whose rank-reduced input memlet is uncollapsed keeps reducing the same data: its
    ``axes`` (indices into the input subset) move onto the dims that survived the collapse. Left as
    they were, ``axes=[0]`` over ``x[0:M]`` widened to ``a[j, 0:M]`` reduces the length-1 dim, a copy."""
    from dace.libraries.standard.nodes.reduce import Reduce
    if not isinstance(node, Reduce) or node.axes is None:
        return
    surviving = [d for d, collapsed in enumerate(collapsed_dims) if not collapsed]
    node.axes = [surviving[axis] for axis in node.axes]


def _rewrite_memlets_with_offset(inner_sdfg: SDFG,
                                 inner_name: str,
                                 offset_dims: List[sympy.Basic],
                                 collapsed_dims: List[bool],
                                 inner_shape: Tuple,
                                 step_dims: Optional[List[sympy.Basic]] = None) -> None:
    """Rewrite every memlet referencing ``inner_name``: add ``offset_dims``, uncollapse
    ``collapsed_dims``, and scale a strided window's inner index by its outer ``step_dims``. Runs BEFORE
    ``replace_dict({inner_name: outer_name})`` so it matches only THIS inner_name's memlets -- else two
    connectors binding the same outer array at different offsets (``A[1,i,j]`` AND ``A[0,i,j]``) clobber
    cross-iteration.
    """
    step_dims = step_dims or [1] * len(offset_dims)

    def outer_ranges(inner_subset: list) -> Tuple[list, bool]:
        # ``offset_dims`` / ``collapsed_dims`` span the FULL outer rank; an inner subset aligns two ways.
        #  * Full-rank (``len(inner_subset) == len(offset_dims)``): 1:1 dim map, the boundary begin is added to
        #    EACH dim, length-1 collapsed ones included -- a 3-point stencil reads ``A[0,0]/A[0,1]/A[0,2]`` with
        #    dim0 collapsed, and the ``+1``/``+2`` lives in the inner begin.
        #  * Rank-reduced (collapsed dims dropped): a collapsed dim contributes only its offset and
        #    ``memlet_access_idx`` walks the surviving inner dims.
        new_range_list = []
        memlet_access_idx = 0
        inner_is_full_rank = len(inner_subset) == len(offset_dims)
        for d, (offset, collapsed, step) in enumerate(zip(offset_dims, collapsed_dims, step_dims)):
            if inner_is_full_rank:
                (lo, hi, stp) = inner_subset[d]
                # Nest rebases each access relative to the boundary begin, so the outer begin is
                # ``lo + offset``; in-place RMW keeps its access ABSOLUTE (``lo == offset``), and re-adding
                # there double-counts (``i + i = 2*i``). No ``sympy.simplify``: too slow on Min/int_floor.
                if keeps_absolute_index(lo, offset, inner_shape, d):
                    new_range_list.append((lo, hi, stp))
                else:
                    new_range_list.append(widened_range(lo, hi, stp, offset, step))
            elif collapsed is True:
                new_range_list.append((offset, offset, 1))
            else:
                (lo, hi, stp) = inner_subset[memlet_access_idx]
                new_range_list.append(widened_range(lo, hi, stp, offset, step))
                memlet_access_idx += 1
        return new_range_list, inner_is_full_rank

    for state in inner_sdfg.states():
        for edge in state.edges():
            memlet = edge.data
            if memlet is None or memlet.is_empty():
                continue
            if memlet.data != inner_name:
                widen_far_side_of_copy(state, edge, inner_name, inner_shape, outer_ranges)
                continue
            new_range_list, inner_is_full_rank = outer_ranges(as_range(memlet.subset).ranges)
            if not inner_is_full_rank:
                remap_reduce_axes(edge.dst, collapsed_dims)
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


def reaches_further(state: SDFGState, edge: MultiConnectorEdge, other: MultiConnectorEdge) -> bool:
    """True if ``edge``'s memlet path leaves more scopes than ``other``'s, so it keeps the connector.

    A path ending at an access node inside the enclosing map publishes nothing past it: keeping that
    one strands the node beyond the MapExit without a producer, and its readers (CloudSC's ``pfsqrf``
    device-to-host copy) run before the kernel.
    """
    return len(state.memlet_path(edge)) > len(state.memlet_path(other))


def fold_duplicate_boundary_edge(state: SDFGState, edge: MultiConnectorEdge, is_input: bool) -> None:
    """Remove ``edge``'s whole memlet path (a partial removal dangles the scope's ``IN_x``/``OUT_x``).

    The access node at the far end of the path can outlive it, still inside its scope and still
    ordering whatever reads or writes it. Its last hop is restated as an empty memlet, which keeps
    it in its scope and after (for a write) or before (for a read) the nested SDFG.
    """
    path = state.memlet_path(edge)
    end, hop = (path[0].src, path[0].dst) if is_input else (path[-1].dst, path[-1].src)
    state.remove_memlet_path(edge, remove_orphans=True)
    nodes_left = state.nodes()
    if end not in nodes_left or hop not in nodes_left:
        return
    # A fresh Memlet per edge -- never the object the old edge carried.
    if is_input:
        state.add_nedge(end, hop, Memlet())
    else:
        state.add_nedge(hop, end, Memlet())


def _replace_desc_and_uncollapse_dims(nsdfg_node: nodes.NestedSDFG,
                                      state: SDFGState,
                                      inner_name: str,
                                      outer_name: str,
                                      desc: data.Array,
                                      collapsed_dims: List[bool],
                                      offset_dims: List[sympy.Basic],
                                      direction: str,
                                      apply_offset: bool = True,
                                      step_dims: Optional[List[sympy.Basic]] = None) -> None:
    # Replace inner_name occurrences + data descriptor with outer_name.
    assert isinstance(inner_name, str) and isinstance(outer_name, str)

    # Remove old array, add new, so occurrences can be safely replaced.
    inner_sdfg: SDFG = nsdfg_node.sdfg
    # An output connector can already lack its descriptor; remove_data tolerates that, so the shape lookup does too.
    inner_shape = inner_sdfg.arrays[inner_name].shape if inner_name in inner_sdfg.arrays else ()
    inner_sdfg.remove_data(inner_name, validate=False)
    copy_desc = copy.deepcopy(desc)
    # A View is a view only next to the data it views: the ``views`` edge stays in the outer state,
    # which resolves it before the memlet reaches this connector. Carrying the View class inward
    # leaves an access node with nothing to view, which validation rejects as an ambiguous edge.
    if isinstance(copy_desc, data.StructureView):
        copy_desc = copy_desc.as_structure()
    elif isinstance(copy_desc, data.View):
        copy_desc = copy_desc.as_array()
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
    step_dims = step_dims or [1] * len(offset_dims)
    if apply_offset:
        _rewrite_memlets_with_offset(inner_sdfg, inner_name, offset_dims, collapsed_dims, inner_shape, step_dims)

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
            # Iterate the FULL outer rank (offset_dims/collapsed_dims), pulling an original
            # index only for non-collapsed dims. The original subscript ``args[1:]`` spans the
            # INNER rank (one index per surviving dim), which is shorter than the outer rank
            # whenever a dim was collapsed to size 1. A ``zip(args[1:], ...)`` truncated to that
            # shorter length, so a collapsed mask ``I[0]`` on an (N, N) map widened to ``I[__i0]``
            # (a whole row) instead of ``I[__i0, __i1]`` -- dropping every trailing map dim.
            if not apply_offset:
                return symbolic.Subscript(base, *uncollapsed_indices(args[1:], offset_dims, collapsed_dims))
            return symbolic.Subscript(base, *outer_indices(args[1:], offset_dims, collapsed_dims, step_dims,
                                                           inner_shape))
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
    for edge in inner_sdfg.all_interstate_edges():
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

    for edge in inner_sdfg.all_interstate_edges():
        if not edge.data.is_unconditional():
            edge.data.condition = _rewrite_codeblock(edge.data.condition)
    for cfg in inner_sdfg.all_control_flow_regions():
        if isinstance(cfg, LoopRegion):
            cfg.loop_condition = _rewrite_codeblock(cfg.loop_condition)
            cfg.update_statement = _rewrite_codeblock(cfg.update_statement)
            cfg.init_statement = _rewrite_codeblock(cfg.init_statement)
    # Uncollapse dims in conditional-branch heads.
    for cfg in inner_sdfg.all_control_flow_blocks():
        if isinstance(cfg, ConditionalBlock):
            for i, (cond, body) in enumerate(cfg.branches):
                if cond is None:  # else branch -- nothing to rewrite
                    continue
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
            # ``inner_name[x...]`` -> ``outer_name[offset + x * step...]``.
            if str(args[0]) == inner_name and not apply_offset:
                return symbolic.Subscript(outer_sym, *args[1:])
            if str(args[0]) == inner_name:
                return symbolic.Subscript(outer_sym,
                                          *outer_indices(args[1:], offset_dims, collapsed_dims, step_dims, inner_shape))
            return symbolic.Subscript(*args)

        def _rw_index_expr(expr):
            # A strip-mined bound is a SymExpr, whose ``str`` prints "main (approx)" -- text that
            # sympy then reads as a call and rejects with "'One' object is not callable". Rewrite
            # the two halves and keep the pair.
            if isinstance(expr, symbolic.SymExpr):
                return symbolic.SymExpr(_rw_index_expr(expr.expr), _rw_index_expr(expr.approx))
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

        for st in inner_sdfg.states():
            for edge in st.edges():
                memlet = edge.data
                # Own-array memlets already handled by ``_rewrite_memlets_with_offset``; here
                # only OTHER memlets carrying ``inner_name`` as a subset index.
                if memlet is None or memlet.data == inner_name:
                    continue
                _uncollapse_subset_refs(memlet.subset)
                _uncollapse_subset_refs(memlet.other_subset)

    # A widened scalar consumed symbolically (its value inlined into a tasklet's code as a bare
    # name) has no valid array form as a bare name -- convert those reads to a dataflow connector
    # fed by ``outer_name[offset_dims]``. Only for an Array source read into the NSDFG (``in``);
    # a Scalar source stays a bare symbol, and outputs already flow through a memlet.
    if not outer_is_scalar and direction == 'in':
        _rewrite_scalar_reads_in_tasklets(inner_sdfg, inner_name, outer_name, offset_dims)

    inner_sdfg.replace_dict({inner_name: outer_name})

    # Replace connectors. If ``outer_name`` already has an edge on this side (a previous
    # iteration merged another connector binding the same outer array, e.g. ``A[1,i,j]`` →
    # ``__tmp_a`` AND ``A[0,i,j]`` → ``__tmp_b`` both bind outer ``A``), keep ONE merged edge and
    # fold the other away. Its subset is the full-array ``_full_subset`` so the post-widening
    # contract holds regardless of which connector contributed which slice. The two edges are
    # interchangeable as transfers, NOT as dependences: see :func:`reaches_further`.
    assert direction in ('in', 'out')
    is_input = direction == 'in'
    boundary = state.in_edges(nsdfg_node) if is_input else state.out_edges(nsdfg_node)
    existing = [
        e for e in boundary if (e.dst_conn if is_input else e.src_conn) == outer_name and outer_name != inner_name
    ]
    for edge in [e for e in boundary if (e.dst_conn if is_input else e.src_conn) == inner_name]:
        if existing and not all(reaches_further(state, edge, other) for other in existing):
            fold_duplicate_boundary_edge(state, edge, is_input)
            continue
        for other in existing:
            fold_duplicate_boundary_edge(state, other, is_input)
        existing = []
        if is_input:
            edge.dst_conn = outer_name
        else:
            edge.src_conn = outer_name
        edge.data.subset = _full_subset(state.sdfg, outer_name)
        # Inner descriptor now mirrors the full outer array; inner memlets carry the offset
        # (above). Any ``other_subset`` described the OLD collapsed inner shape (K>=2 broadcast
        # ``a[i//2] -> inner_a[0]``, inner ``(1,)``) → stale after widening. Clear it for a clean
        # full-array passthrough (design 2.4).
        edge.data.other_subset = None
    # A fold already dropped its connector; guard the idempotent explicit removal.
    if is_input:
        if inner_name in nsdfg_node.in_connectors:
            nsdfg_node.remove_in_connector(inner_name)
        if outer_name not in nsdfg_node.in_connectors:
            nsdfg_node.add_in_connector(outer_name, force=True)
    else:
        if inner_name in nsdfg_node.out_connectors:
            nsdfg_node.remove_out_connector(inner_name)
        if outer_name not in nsdfg_node.out_connectors:
            nsdfg_node.add_out_connector(outer_name, force=True)

    # (Inner memlets, interstate assignments, loop/branch heads all rewritten above with
    # per-inner_name offsets BEFORE replace_dict, to avoid clobbering an earlier connector.)


@make_properties
class ExpandNestedSDFGInputs(transformation.SingleStateTransformation):
    """Pre-processor for :class:`InlineMultistateSDFG`: widen narrowed NSDFG in/out memlets to
    full-array subsets, reshape inner descriptors, offset inner memlets.

    Handles top-level NSDFGs and NSDFGs inside a Map scope. In-map case: caller must first widen
    the parent map's IN/OUT memlets to full arrays (via ``propagate_full_array_subsets_through_map``)
    so the MapEntry→NSDFG connector carries the full extent; the per-iteration tile offset is then
    captured from the original narrowed subset and threaded onto every inner memlet.
    """

    nested_sdfg = transformation.PatternNode(nodes.NestedSDFG)

    top_level_only = Property(dtype=bool,
                              default=False,
                              desc='If True, only apply to top-level NSDFGs, i.e. NSDFGs that are '
                              'not nested inside a Map scope (their enclosing scope entry is None).')

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
        # Restrict to top-level NSDFGs (no enclosing Map) when requested.
        if self.top_level_only and state.entry_node(nsdfg_node) is not None:
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
            # A whole-array edge can still bind a connector of lower rank (``mass[0:N, 0]`` of an
            # ``(N, 1)`` array as an ``(N,)`` connector), which the inner memlets index by that rank.
            inner_arr = nsdfg_node.sdfg.arrays.get(edge.dst_conn if edge.dst is nsdfg_node else edge.src_conn)
            if inner_arr is not None and len(inner_arr.shape) != len(outer_arr.shape):
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
            for (b, e, s) in as_range(iedge.data.subset).ranges:
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
            for (b, e, s) in as_range(oedge.data.subset).ranges:
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
        # Before any rename: an array both read and written loses its inner name on the first pass.
        steps = {
            (direction, conn):
            window_steps(outer_subset, collapsed_dims, nsdfg_node.sdfg.arrays[conn], sdfg.arrays[outer_arr_name])
            for direction, subsets_by_conn in (('in', read_subsets), ('out', write_subsets))
            for conn, (outer_arr_name, outer_subset, collapsed_dims) in subsets_by_conn.items()
        }
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
                                              apply_offset=apply_offset,
                                              step_dims=steps['in', conn])

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
                                              apply_offset=apply_offset,
                                              step_dims=steps['out', conn])

        # Thread any gather/scatter INDEX array referenced inside an inner memlet SUBSET but
        # absent from ``inner_sdfg.arrays``. ``A[B[i]]`` where ``B`` was never a boundary
        # connector (nesting threaded DATA ``A`` but not index ``B``, which appears only in
        # ``A``'s subset) leaves ``B`` dangling → can't codegen through inline, no per-lane index
        # tile. Add ``B`` as a full-array read boundary (like ``A``): non-transient inner
        # descriptor, in-connector, access-node edge routed through the enclosing Map (if any).
        def _subset_arrays(sub) -> Set[str]:
            if isinstance(sub, subsets.Range):
                exprs = [x for r in sub.ranges for x in r]
            else:
                return set()
            names: Set[str] = set()
            for ex in exprs:
                # Same SymExpr trap as ``_rw_index_expr``: read the two halves, not the printed pair.
                for part in ((ex.expr, ex.approx) if isinstance(ex, symbolic.SymExpr) else (ex, )):
                    names |= symbolic.arrays(symbolic.pystr_to_symbolic(str(part)))
            return names

        referenced: Set[str] = set()
        for st in inner_sdfg.states():
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
        #
        # A symbol the inner SDFG DEFINES itself is excluded: a scope-lifetime transient may be
        # sized by an inner loop iterator (a triangular reduction buffer ``_red_buf[M - it - 1]``
        # inside the ``it`` loop). Such a name has no binding in the caller's scope, so putting it
        # in ``symbol_mapping`` makes codegen pass an undeclared variable at the call site.
        inner_defined = set(inner_sdfg.symbols.keys()) - {str(s) for s in inner_sdfg.free_symbols}
        for inner_arr_name, inner_desc in inner_sdfg.arrays.items():
            for sym in inner_desc.free_symbols:
                sym_name = str(sym)
                if sym_name in nsdfg_node.in_connectors or sym_name in nsdfg_node.out_connectors:
                    continue
                if sym_name in inner_sdfg.constants_prop or sym_name in inner_defined:
                    continue
                if sym_name not in nsdfg_node.symbol_mapping:
                    introduced_symbols.add(sym_name)

        # Propagate offset symbols not already in symbol_mapping. Identity binding is the default
        # (outer ``ii`` → inner ``ii``). Each type is resolved at the NSDFG node's scope in the outer
        # SDFG, the only table that sees an enclosing loop iterator or map parameter; a name nothing
        # declares raises instead of defaulting to ``int64`` against another integer width.
        resolver = scopes.ScopedSymbolResolver()
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
            outer_type = resolver.resolve_dtype(sym_name, sdfg, state=state, node=nsdfg_node)
            inner.add_symbol(sym_name, outer_type)
            nsdfg_node.symbol_mapping[sym_name] = symbolic.pystr_to_symbolic(sym_name)
