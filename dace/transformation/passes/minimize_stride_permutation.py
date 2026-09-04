# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Permute perfectly-nested map dimensions to minimize array access strides.

This pass rewrites perfect map nests so that the innermost map parameter
indexes the contiguous (smallest-stride) array axis with a unit coefficient.
The transformation is purely a reordering of the loop nest: it never changes
the iteration space, the body, or the memlets' data, so the resulting SDFG is
numerically identical to the input.

The deterministic ordering is a pure function of the body's access pattern,
which makes the pass idempotent (a second run finds the nest already
canonical and performs no further interchange):

1. For every array access reachable inside the innermost map scope, the
   linear coefficient of each map parameter is extracted per array axis
   using ``dace.symbolic``. An axis indexed by a parameter with unit
   coefficient is a candidate "stride home" for that parameter.
2. Each parameter receives a score equal to the smallest array stride it
   linearly indexes with a unit coefficient (``+inf`` if it never does). The
   parameter whose smallest indexed stride is smallest belongs innermost.
3. Ties are broken first by the accumulated absolute constant offset of the
   accesses the parameter participates in (ascending), then by the
   parameter's original textual position in the nest (ascending), so the
   target order is total and stable.

The desired permutation is realized exclusively through a sequence of legal
adjacent :class:`~dace.transformation.dataflow.map_interchange.MapInterchange`
applications (an adjacent-transposition bubble sort). An interchange that is
not legal at its turn is skipped rather than raising, so the pass degrades
gracefully on nests it cannot fully reorder.
"""
import functools
from typing import Dict, List, Optional, Tuple

import sympy

from dace import SDFG, properties, symbolic
from dace import data as dt
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState
from dace.symbolic import pystr_to_symbolic
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.dataflow.map_interchange import MapInterchange

#: Sentinel score for a parameter that never indexes any axis with a unit
#: coefficient (it has no contiguous "home" and is sorted outermost).
_NO_HOME_SCORE = sympy.oo


def _to_float(value: object) -> float:
    """Best-effort conversion of a (possibly symbolic) score to a sortable float.

    Non-numeric symbolic values cannot be reliably ordered, so they map to
    ``+inf`` (treated as the worst/largest stride and sorted outermost).

    :param value: A number or ``sympy`` expression.
    :returns: A finite float, or ``+inf`` for non-numeric symbolic input.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('inf')


def score_indexed_strides(edges, sdfg, var_names) -> Dict[str, Tuple[object, object, object]]:
    """Score how each name in ``var_names`` indexes arrays across ``edges``.

    For every array memlet on an edge, each axis the variable indexes is
    examined. A variable "homes" on an axis when it appears there with unit
    absolute coefficient; that axis's absolute stride is then a contiguity
    score. Strides of other listed variables are zeroed out so the per-variable
    constant offset reflects only that variable's contribution.

    Shared by :class:`MinimizeStridePermutation` (map-nest reordering) and the
    loop<->map interchange gate, which both rank variables by indexed stride.

    :param edges: Iterable of graph edges to scan (each carrying a ``Memlet``).
    :param sdfg: The SDFG owning the arrays (for descriptor strides).
    :param var_names: The variable names to score.
    :returns: ``{name: (min_home_stride, total_home_stride, abs_offset_sum)}``.
              ``min_home_stride`` is :data:`_NO_HOME_SCORE` when the variable
              never homes on any axis; ``total_home_stride`` sums home strides
              over all accesses (repeated contiguous accesses weigh more);
              ``abs_offset_sum`` accumulates the absolute constant offsets.
    """
    var_set = set(var_names)
    min_stride: Dict[str, object] = {v: _NO_HOME_SCORE for v in var_set}
    total_stride: Dict[str, object] = {v: sympy.S.Zero for v in var_set}
    offset_sum: Dict[str, object] = {v: sympy.S.Zero for v in var_set}
    for edge in edges:
        memlet = edge.data
        if memlet is None or memlet.data is None:
            continue
        desc = sdfg.arrays.get(memlet.data)
        if not isinstance(desc, dt.Array):
            continue
        subset = memlet.subset
        if subset is None or len(subset) != len(desc.strides):
            continue
        for rng, stride in zip(subset.ndrange(), desc.strides):
            index_expr = pystr_to_symbolic(rng[0])
            # The expression may carry the map parameter as a plain sympy
            # ``Symbol`` while ``pystr_to_symbolic`` returns a DaCe ``symbol``.
            # Coefficient extraction is object-sensitive, so resolve each name to
            # the actual symbol instance present in the expression.
            symbols_by_name = {str(s): s for s in index_expr.free_symbols if str(s) in var_set}
            if not symbols_by_name:
                continue
            stride_val = sympy.Abs(pystr_to_symbolic(str(stride)))
            for vname, vsym in symbols_by_name.items():
                coeff = index_expr.coeff(vsym, 1)
                if coeff == 0:
                    continue
                others = {s: 0 for name, s in symbols_by_name.items() if name != vname}
                constant = index_expr.subs(others).subs({vsym: 0})
                offset_sum[vname] = offset_sum[vname] + sympy.Abs(constant)
                if sympy.Abs(coeff) == 1:
                    total_stride[vname] = total_stride[vname] + stride_val
                    prev = min_stride[vname]
                    min_stride[vname] = stride_val if prev is _NO_HOME_SCORE else sympy.Min(prev, stride_val)
    return {v: (min_stride[v], total_stride[v], offset_sum[v]) for v in var_set}


class UndecidableStride(Exception):
    """Two stride scores whose relative magnitude does not follow from the shape contract."""


def stride_difference_sign(a, b) -> int:
    """``-1``/``0``/``1`` as ``a`` is less than / equal to / greater than ``b``.

    An array's strides are PRODUCTS OF ITS SHAPE EXTENTS, and an extent is at least one wherever the
    array exists at all, so a symbolic stride is not an undecidable one: ``1`` versus ``LEN_2D`` and
    ``M`` versus ``N*M`` both follow from that. The symbols are rebuilt ``positive`` here rather than
    read off the incoming expression because assumptions do not survive the string round trip a
    stride takes to reach this pass.

    A pair that genuinely does not follow (``N*M`` versus ``M + K``) raises
    :class:`UndecidableStride`, which leaves the nest untouched -- the behaviour this pass had for
    every symbolic shape before.
    """
    diff = symbolic.simplify(pystr_to_symbolic(str(a)) - pystr_to_symbolic(str(b)))
    if diff == 0:
        return 0
    if diff.is_number:
        return -1 if diff < 0 else 1
    rebuilt = diff.subs({sym: sympy.Symbol(sym.name, positive=True, integer=True) for sym in diff.free_symbols})
    # Non-STRICT: two scores that may coincide (``1`` vs ``LEN_2D`` at ``LEN_2D == 1``) impose no
    # order, and reporting them as tied keeps the stable sort's current-order behaviour.
    if rebuilt.is_nonpositive:
        return -1
    if rebuilt.is_nonnegative:
        return 1
    raise UndecidableStride(f'{a} vs {b}')


@properties.make_properties
@transformation.explicit_cf_compatible
class MinimizeStridePermutation(ppl.Pass):
    """Reorder perfect map nests so the innermost parameter has minimal stride.

    See the module docstring for the full design and the deterministic
    ordering key. The pass only emits adjacent ``MapInterchange`` applications
    and therefore preserves the SDFG's numerical result.
    """

    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Scopes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & (ppl.Modifies.Scopes | ppl.Modifies.Memlets | ppl.Modifies.Nodes))

    def apply_pass(self, sdfg: SDFG, _: Dict[str, object]) -> Optional[Dict[int, int]]:
        """Apply the pass to ``sdfg``.

        :param sdfg: The SDFG to canonicalize.
        :returns: A mapping ``{state id: number of interchanges applied}`` for
                 states that changed, or ``None`` if nothing was modified.
        """
        result: Dict[int, int] = {}
        for state in sdfg.all_states():
            applied = self._process_state(state, state.sdfg)
            if applied:
                result[state.block_id] = applied
        return result or None

    def _process_state(self, state: SDFGState, sdfg: SDFG) -> int:
        """Find and reorder every top-level perfect map nest in one state.

        :param state: The state to scan.
        :param sdfg: The SDFG owning ``state`` (for data descriptors).
        :returns: The number of adjacent interchanges applied in this state.
        """
        scope_children = state.scope_children()
        applied = 0
        for node in scope_children[None]:
            if not isinstance(node, nodes.MapEntry):
                continue
            if state.entry_node(node) is not None:
                continue
            nest = self._collect_perfect_nest(state, node, scope_children)
            if len(nest) < 2:
                continue
            applied += self._reorder_nest(state, sdfg, nest)
        return applied

    def _collect_perfect_nest(self, state: SDFGState, outer: nodes.MapEntry,
                              scope_children: Dict[Optional[nodes.Node], List[nodes.Node]]) -> List[nodes.MapEntry]:
        """Collect the chain of perfectly-nested single-parameter map entries.

        A level is part of the perfect nest only if its scope contains exactly
        one child node and that node is the next map entry, and the map carries
        a single parameter (multi-parameter maps are left untouched as they
        cannot be split by ``MapInterchange``).

        :param state: The containing state.
        :param outer: The outermost candidate map entry.
        :param scope_children: Precomputed scope-children mapping for ``state``.
        :returns: The list of map entries from outermost to innermost.
        """
        nest: List[nodes.MapEntry] = []
        current: Optional[nodes.MapEntry] = outer
        while current is not None:
            if len(current.map.params) != 1:
                break
            nest.append(current)
            # ``scope_children`` always lists the scope's own MapExit; the nest
            # is perfect only if the sole remaining child is the next map.
            exit_node = state.exit_node(current)
            body = [c for c in scope_children.get(current, []) if c is not exit_node]
            inner_maps = [c for c in body if isinstance(c, nodes.MapEntry)]
            if len(body) == 1 and len(inner_maps) == 1:
                current = inner_maps[0]
            else:
                current = None
        return nest

    def _reorder_nest(self, state: SDFGState, sdfg: SDFG, nest: List[nodes.MapEntry]) -> int:
        """Compute the canonical order for ``nest`` and realize it.

        :param state: The containing state.
        :param sdfg: The SDFG owning ``state``.
        :param nest: Map entries from outermost to innermost.
        :returns: The number of adjacent interchanges applied.
        """
        params = [me.map.params[0] for me in nest]
        scores = self._score_parameters(state, sdfg, nest, params)

        # Build a totally-ordered numeric key per parameter. Smaller indexed
        # stride => deeper (more inner), so the outer..inner target order
        # sorts by *descending* stride. A parameter without a unit-coefficient
        # "home" gets ``+inf`` and is forced outermost. Ties break on ascending
        # accumulated absolute offset. Parameters with equal (or incomparable,
        # e.g. symbolic) scores keep their current relative order via the
        # stable sort below -- no positional tiebreak, otherwise reordering
        # would change the key and the pass would not be idempotent. When all
        # strides are symbolic the scores tie and the order is left unchanged,
        # which is the intended behavior (magnitudes are undecidable).
        def compare(i: int, j: int) -> int:
            # Descending stride (the smallest-strided parameter belongs innermost), then ascending
            # accumulated offset. Decided symbolically: see :func:`stride_difference_sign`.
            stride_sign = stride_difference_sign(scores[i][0], scores[j][0])
            if stride_sign != 0:
                return -stride_sign
            return stride_difference_sign(scores[i][1], scores[j][1])

        try:
            order = sorted(range(len(params)), key=functools.cmp_to_key(compare))
        except UndecidableStride:
            return 0
        # ``order[k]`` is the original index that should end up at depth ``k``.
        if order == list(range(len(params))):
            return 0

        # Bubble-sort the permutation using only adjacent transpositions, each
        # realized by a single MapInterchange (outer = depth k, inner = k+1).
        permutation = list(order)
        applied = 0
        n = len(permutation)
        for end in range(n - 1, 0, -1):
            for k in range(end):
                if permutation[k] > permutation[k + 1]:
                    if self._swap_adjacent(sdfg, nest, k):
                        permutation[k], permutation[k + 1] = permutation[k + 1], permutation[k]
                        applied += 1
        return applied

    def _swap_adjacent(self, sdfg: SDFG, nest: List[nodes.MapEntry], depth: int) -> bool:
        """Interchange the maps at ``depth`` and ``depth + 1`` if legal.

        On success ``nest`` is updated in place to reflect the new ordering.

        :param sdfg: The SDFG owning the nest.
        :param nest: Map entries (mutated in place on success).
        :param depth: Zero-based depth of the outer of the adjacent pair.
        :returns: True if the interchange was applied, False if it was skipped.
        """
        outer_entry = nest[depth]
        inner_entry = nest[depth + 1]
        # ``transform_bounds``: a triangular / trapezoidal nest is exactly the shape whose traversal
        # order is worth fixing and the one a plain swap refuses, because the inner range mentions
        # the outer parameter. TSVC ``s1232`` (``for j: for i in range(j*VLEN, N): aa[i, j] = ...``)
        # keeps a stride-``N`` innermost loop without it -- the pass scores the nest correctly and
        # then cannot act on the score.
        options = {'transform_bounds': True}
        if not MapInterchange.can_be_applied_to(
                sdfg, options=options, outer_map_entry=outer_entry, inner_map_entry=inner_entry):
            return False
        MapInterchange.apply_to(sdfg,
                                options=options,
                                outer_map_entry=outer_entry,
                                inner_map_entry=inner_entry,
                                verify=False,
                                save=False)
        nest[depth], nest[depth + 1] = inner_entry, outer_entry
        return True

    def _score_parameters(self, state: SDFGState, sdfg: SDFG, nest: List[nodes.MapEntry],
                          params: List[str]) -> List[Tuple[object, object]]:
        """Compute the ``(stride_score, offset_tiebreak)`` key for each param.

        :param state: The containing state.
        :param sdfg: The SDFG owning ``state``.
        :param nest: Map entries from outermost to innermost.
        :param params: The single parameter of each level (parallel to
                       ``nest``).
        :returns: A list of ``(min_indexed_stride, accumulated_abs_offset)``
                 tuples parallel to ``params``.
        """
        # Include the innermost entry and exit so that the per-iteration
        # indexed memlets (which live on the edges between the inner map
        # entry/exit and the body) are part of the scanned subgraph.
        innermost_entry = nest[-1]
        body = state.scope_subgraph(innermost_entry, include_entry=True, include_exit=True)
        scores = score_indexed_strides(body.edges(), sdfg, params)
        return [(scores[p][0], scores[p][2]) for p in params]
