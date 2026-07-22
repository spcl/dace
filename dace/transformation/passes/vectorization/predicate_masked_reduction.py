# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Predicate a masked WCR reduction so the tile path can fold it.

The DaCe frontend emits ``for j in dc.map: if mask[j]: acc op= f`` as a
:class:`~dace.sdfg.state.ConditionalBlock` whose single if-true branch body
holds the reduction WCR edges (``... -[wcr: lambda x,y: x op y]-> acc``, ``acc``
a scalar). That in-branch WCR survives every downstream reduction lift:
``LiftMapReductionToReduce`` needs an unconditional per-element fold, and
``WCRToAugAssign`` refuses to read back the accumulator inside the branch (its
nested-SDFG guard sees the constant-index write under an enclosing parallel map
as a cross-iteration race). So the WCR reaches the tile-entry precondition
``no_wcr_inside_nested_sdfgs`` and trips it.

This pass hoists each such reduction out of the conditional by predicating its
ADDEND with the branch condition and leaving the WCR unconditional::

    if c: acc op= addend      ->      acc op= ITE(c, addend, identity(op))

``x op identity == x``, so where ``c`` is false the reduction contributes the
op's neutral element -- a bit-exact no-op (``x + 0.0 == x``). The
ConditionalBlock is then dissolved (its now-unconditional body spliced up), and
the ordinary reduction machinery (``NormalizeWCR`` -> ``TileReduce`` / boundary
OpenMP reduction) folds the addend-select like any masked-if the ``merge`` /
``fp_factor`` branch modes already vectorize.

Runs in vectorize-prep BEFORE ``NormalizeWCR``: it needs the mask still visible
as a ConditionalBlock in the CFG (afterwards the reduction is smeared across the
nested-SDFG boundary and there is no single edge to gate).
"""
import copy
from typing import List, Optional, Set, Tuple

import numpy

from dace import SDFG, dtypes, symbolic
from dace.frontend.operations import detect_reduction_type
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, SDFGState
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.helpers import move_branch_cfg_up_discard_conditions
from dace.transformation.passes.vectorization.utils.tile_access import expr_is_data_dependent

#: Reduction types this pass predicates, and their infix operator.
_REDTYPE_OP = {
    dtypes.ReductionType.Sum: '+',
    dtypes.ReductionType.Product: '*',
    dtypes.ReductionType.Max: 'max',
    dtypes.ReductionType.Min: 'min',
}


def _identity_literal(op: str, dtype: dtypes.typeclass) -> str:
    """The op's identity as a Python literal for an ITE tasklet body, dtype-aware.

    Integer accumulators use exact ``iinfo`` bounds for ``min`` / ``max`` (a float
    ``inf`` would be wrong once truncated to an integer)."""
    is_int = numpy.issubdtype(dtype.type, numpy.integer)
    if op == '+':
        return '0' if is_int else '0.0'
    if op == '*':
        return '1' if is_int else '1.0'
    if op == 'min':
        return repr(int(numpy.iinfo(dtype.type).max)) if is_int else "float('inf')"
    if op == 'max':
        return repr(int(numpy.iinfo(dtype.type).min)) if is_int else "float('-inf')"
    raise ValueError(f'no identity for op {op!r}')


def _body_has_data_dependent_read(sd: SDFG, state: SDFGState) -> bool:
    """True if any read memlet subset in ``state`` is a data-dependent (gather)
    index -- ``a[idx[j]]`` and friends. Such a load would fault on the mask-false
    lanes once the branch is dissolved to unconditional, so predication must
    refuse. Structured affine subsets (functions of iteration/scope symbols only)
    are in-bounds by construction and return False."""
    for edge in state.edges():
        m = edge.data
        if m is None or m.data is None or m.subset is None:
            continue
        for rng in m.subset.ranges:
            for bound in rng:  # (start, end, step)
                if bound is not None and expr_is_data_dependent(bound, sd):
                    return True
    return False


@transformation.explicit_cf_compatible
class PredicateMaskedReduction(ppl.Pass):
    """Rewrite ``if c: acc op= addend`` into ``acc op= ITE(c, addend, identity)``."""

    CATEGORY: str = 'Vectorization Preparation'

    def modifies(self) -> ppl.Modifies:
        return (ppl.Modifies.CFG | ppl.Modifies.Nodes | ppl.Modifies.Memlets | ppl.Modifies.Descriptors)

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> Set:
        return set()

    def apply_pass(self, sdfg: SDFG, _) -> Optional[int]:
        applied = 0
        for sd in sdfg.all_sdfgs_recursive():
            for cfg in list(sd.all_control_flow_regions(recursive=True)):
                for block in list(cfg.nodes()):
                    if isinstance(block, ConditionalBlock):
                        match = self._match(sd, block)
                        if match is not None:
                            self._apply_one(sd, block, *match)
                            applied += 1
        if applied:
            sdfg.reset_cfg_list()
        return applied or None

    def _match(self, sd: SDFG, cb: ConditionalBlock):
        """Return ``(cond_text, body_state, [(edge, op)], cond_materialize)`` if ``cb``
        is a predicable masked reduction, else None (leaving the graph untouched).

        ``cond_materialize`` is ``None`` when the condition is already a materialized
        scalar mask (recipe 1); otherwise it is the ``(cleaned, connectors, subsets)``
        staging info for a compound predicate ``(a[i] > K)`` that :meth:`_apply_one`
        first lowers into a bool mask scalar (recipe 2)."""
        # Shape: one if-true branch, optionally an EMPTY else.
        true_branches = [(c, b) for c, b in cb.branches if c is not None]
        else_branches = [(c, b) for c, b in cb.branches if c is None]
        if len(true_branches) != 1 or len(else_branches) > 1:
            return None
        cond, body = true_branches[0]
        if else_branches is not None:
            for _, eb in else_branches:
                if any(len(st.nodes()) > 0 for st in eb.all_states()):
                    return None
        # The dissolver needs a single-state body (single start == single sink).
        states = list(body.all_states())
        if len(states) != 1 or len(body.nodes()) != 1:
            return None
        state = states[0]

        # Reduction WCR edges: an associative op into a scalar target we can predicate.
        reductions: List[Tuple] = []
        for edge in state.edges():
            m = edge.data
            if m is None or m.data is None or m.wcr is None:
                continue
            op = _REDTYPE_OP.get(detect_reduction_type(m.wcr))
            if op is None:
                return None
            if not isinstance(edge.dst, nodes.AccessNode):
                return None
            if m.subset is None or m.subset.num_elements() != 1:
                return None
            acc_desc = sd.arrays.get(edge.dst.data)
            if acc_desc is None or acc_desc.total_size != 1:
                return None
            reductions.append((edge, op))
        if not reductions:
            return None
        # The source of each reduction WCR edge is a legitimate addend buffer -- a plain
        # write into it is part of the reduction, not an escaping side effect. After
        # canonicalize's ``NormalizeWCRSource`` the addend arrives through a transient
        # ``_wcr_priv`` intermediate (``a -> _Add_ -> _wcr_priv -[wcr]-> acc``), not a
        # direct WCR edge; recognise that shape too.
        red_src_nodes = {e.src for e, _ in reductions}
        for edge in state.edges():
            m = edge.data
            if m is None or m.data is None or m.wcr is not None:
                continue
            if isinstance(edge.dst, nodes.AccessNode):
                desc = sd.arrays.get(edge.dst.data)
                # Allow only a plain write into a transient buffer that feeds a reduction
                # WCR; a plain write into any other access node is a real side effect ->
                # not a pure masked reduction -> refuse.
                if not (edge.dst in red_src_nodes and desc is not None and desc.transient):
                    return None

        cond_text = cond.as_string.strip()
        cond_desc = sd.arrays.get(cond_text)
        cond_materialize = None
        if cond_desc is not None and cond_desc.total_size == 1:
            # Recipe 1: the condition already names a materialized per-lane scalar mask.
            # It must not be produced inside the branch (else unconditional evaluation
            # changes semantics).
            if any(isinstance(n, nodes.AccessNode) and n.data == cond_text and state.in_degree(n) > 0
                   for n in state.nodes()):
                return None
        else:
            # Recipe 2: a compound predicate ``(a[i] > K)`` reading arrays. Materialize
            # it into a bool mask scalar, then predicate as in recipe 1. Refuse if the
            # predicate is not materialisable (loop-invariant / gather-indexed).
            cond_materialize = self._materializable_predicate(sd, cond_text)
            if cond_materialize is None:
                return None
        # Fault gate: dissolving the ConditionalBlock makes the ENTIRE body run on
        # every lane, including the mask-false lanes. That is only sound when
        # evaluating the addend cannot fault -- a data-dependent (gather) load
        # ``a[idx[j]]`` whose index is valid only under the mask would fault at the
        # eager load (the discarded value doesn't un-fault the access). Refuse when
        # any read subset in the body is data-dependent; the masked reduction then
        # keeps its branch (safe, just not predicated). Structured affine loads
        # (``a[j]``, ``a[j+1]``) are always in-bounds regardless of the mask and pass.
        if _body_has_data_dependent_read(sd, state):
            return None
        return cond_text, state, reductions, cond_materialize

    def _materializable_predicate(self, sd: SDFG, cond_text: str):
        """Return ``(cleaned, connectors, subsets)`` to stage a compound predicate
        ``(a[i] > K)`` into a bool mask scalar, or ``None`` if it is not a per-element
        array predicate we can eagerly evaluate on every lane.

        ``cleaned`` is the predicate with each ``arr[...]`` read replaced by a bare
        connector; ``connectors`` maps array name -> in-connector; ``subsets`` maps
        array name -> the ``[i]`` subset string. A predicate reading a GATHER index
        (``w[idx[i]]``) is refused -- eager evaluation on the would-be-masked lanes
        could fault, and its nested subscript is not a plain memlet (that indirect
        masked reduction is a separate feature)."""
        arrays = set(sd.arrays.keys())
        try:
            names = set(symbolic.arrays(cond_text)) | set(symbolic.free_symbols_and_functions(cond_text))
        except Exception:
            return None
        arr_reads = sorted(a for a in names if a in arrays)
        if not arr_reads:
            return None  # loop-invariant predicate -- not the per-element masked-reduction target
        connectors = {a: f'_pmr_in_{i}' for i, a in enumerate(arr_reads)}
        try:
            cleaned, subsets = symbolic.replace_array_accesses_with_connectors(cond_text, connectors, arrays)
        except Exception:
            return None
        for sub in subsets.values():
            if '[' in sub.strip('[]'):  # nested subscript = gather -> refuse (fault + not a plain memlet)
                return None
        return cleaned, connectors, subsets

    def _apply_one(self, sd: SDFG, cb: ConditionalBlock, cond_text: str, state: SDFGState, reductions: List[Tuple],
                   cond_materialize=None):
        """Mutate the branch body in place (predicate each addend), then dissolve ``cb``."""
        if cond_materialize is not None:
            # Recipe 2: lower the compound predicate into a bool mask scalar the ITE reads.
            # The mask runs unconditionally on every lane (affine reads are in-bounds), then
            # the ITE selects the identity on the false lanes -- bit-exact with the branch.
            cleaned, connectors, subsets = cond_materialize
            mask_name, _ = sd.add_scalar('_pmr_mask', dtypes.bool_, transient=True, find_new_name=True)
            mask_node = state.add_access(mask_name)
            mask_t = state.add_tasklet('pmr_mask', set(connectors.values()), {'_m'}, f'_m = ({cleaned})')
            for arr, conn in connectors.items():
                sub = subsets.get(arr, '[0]')
                state.add_edge(state.add_access(arr), None, mask_t, conn, Memlet(expr=f'{arr}{sub}'))
            state.add_edge(mask_t, '_m', mask_node, None, Memlet(expr=f'{mask_name}[0]'))
            cond_text = mask_name
        for edge, op in reductions:
            acc = edge.dst.data
            acc_desc = sd.arrays[acc]
            identity = _identity_literal(op, acc_desc.dtype)

            # Scratch scalar carrying the raw (un-predicated) addend.
            addend_name, _ = sd.add_scalar('_pmr_addend', acc_desc.dtype, transient=True, find_new_name=True)
            addend_node = state.add_access(addend_name)
            # Redirect the addend producer into the scratch (plain, no WCR).
            state.add_edge(edge.src, edge.src_conn, addend_node, None, Memlet(expr=f'{addend_name}[0]'))

            # Identity else-arm through a TYPED seeded scalar rather than a bare literal:
            # ``ITE(_c, _t, 0)`` would carry the wrong C++ type for an integer accumulator
            # (``int`` vs ``int64_t``) and the ITE template would not match. A scalar of the
            # accumulator dtype makes all three ITE arms the same type.
            ident_name, _ = sd.add_scalar('_pmr_ident', acc_desc.dtype, transient=True, find_new_name=True)
            ident_node = state.add_access(ident_name)
            seed = state.add_tasklet('pmr_ident', set(), {'_s'}, f'_s = {identity}')
            state.add_edge(seed, '_s', ident_node, None, Memlet(expr=f'{ident_name}[0]'))

            # ITE select: identity where the mask is false -> op no-op.
            ite = state.add_tasklet('pmr_ite', {'_c', '_t', '_e'}, {'_o'}, '_o = ITE(_c, _t, _e)')
            state.add_edge(state.add_access(cond_text), None, ite, '_c', Memlet(expr=f'{cond_text}[0]'))
            state.add_edge(addend_node, None, ite, '_t', Memlet(expr=f'{addend_name}[0]'))
            state.add_edge(ident_node, None, ite, '_e', Memlet(expr=f'{ident_name}[0]'))
            # Preserve the original WCR on the write into the accumulator.
            out = copy.deepcopy(edge.data)
            state.add_edge(ite, '_o', edge.dst, None, out)
            state.remove_edge(edge)

        move_branch_cfg_up_discard_conditions(cb, [b for c, b in cb.branches if c is not None][0])
