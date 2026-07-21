# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""Normalize every map range and loop counter to canonical ``0:trip:1`` form.

``OffsetLoopsAndMaps`` only shifts by a fixed expression and never changes the
step, so it cannot produce the canonical zero-based / unit-stride form. This
pass does: for a map parameter ``p`` with range ``b:e:s`` it substitutes
``p -> b + s*p`` in the map's own scope (memlets + tasklets, param-local) and
sets the range to ``0:(e-b)//s:1``. ``LoopRegion`` counters are normalized the
same way. The substitution is value-preserving, so the SDFG result is
unchanged. It reuses ``OffsetLoopsAndMaps``' tasklet token-replacement helpers.
"""
import copy
from typing import Dict, Optional, Set

import dace
import sympy

from dace import properties
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.offset_loop_and_maps import OffsetLoopsAndMaps
from dace.transformation.passes.analysis import loop_analysis


@properties.make_properties
@transformation.explicit_cf_compatible
class NormalizeLoopsAndMaps(OffsetLoopsAndMaps):
    """Rewrite map ranges and loop counters to ``0:trip:1`` (zero-based, unit-stride).

    Maps are normalized in their own scope via ``_normalize_map`` and
    ``LoopRegion`` counters via ``_normalize_loop``; both rewrites are
    value-preserving.
    """

    CATEGORY: str = 'Canonicalization'

    def __init__(self):
        # Identity offset/begin: this pass overrides ``apply_pass`` entirely
        # and does not use the base shifting behavior.
        super().__init__(offset_expr="0", begin_expr=None)

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Scopes | ppl.Modifies.Tasklets | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> Set:
        return set()

    def _create_new_memlet(self, edge_data: dace.memlet.Memlet, repldict: Dict[str,
                                                                               str]) -> Optional[dace.memlet.Memlet]:
        """Subset-substitute a memlet via proper dace symbols.

        Overrides the base, which sympifies string values and so mis-parses a
        symbol literally named ``S`` as ``sympy.S`` (and raises on
        ``other_subset``). The pipeline's preparation ``_cleanup`` removes
        ``other_subset`` copies before this pass runs; the ``other_subset``
        branch below only guards standalone callers that skip that cleanup.

        :param edge_data: The memlet to rewrite.
        :param repldict: Symbol-name to replacement-expression mapping.
        :returns: A new memlet with the substituted subsets, or ``None`` if the
                  input carries no subset.
        """
        if edge_data is None or edge_data.subset is None:
            return None
        sd = {dace.symbolic.pystr_to_symbolic(k): dace.symbolic.pystr_to_symbolic(v) for k, v in repldict.items()}

        def _r(sub):
            if sub is None:
                return None
            return dace.subsets.Range([
                (b.subs(sd) if isinstance(b, sympy.Basic) else b, e.subs(sd) if isinstance(e, sympy.Basic) else e,
                 s.subs(sd) if isinstance(s, sympy.Basic) else s) for b, e, s in sub.ndrange()
            ])

        m = copy.deepcopy(edge_data)
        m.subset = _r(m.subset)
        if m.other_subset is not None:
            m.other_subset = _r(m.other_subset)
        return m

    def _normalize_map(self, state: dace.SDFGState, me: nodes.MapEntry) -> bool:
        """Normalize one map's non-canonical parameters in place.

        :param state: The state owning the map.
        :param me: The map entry.
        :returns: ``True`` if any parameter was rewritten.
        """
        new_ranges = list(me.map.range.ranges)
        repldict: Dict[str, str] = {}
        subsdict: Dict = {}
        changed = False
        for i, (p, (b, e, s)) in enumerate(zip(me.map.params, me.map.range.ranges)):
            if b == 0 and s == 1:
                continue
            psym = dace.symbolic.pystr_to_symbolic(p)
            # p_original = b + s * p_new ; p_new in 0 : floor((e-b)/s) : 1
            subsdict[psym] = b + s * psym
            repldict[str(p)] = f"({b} + ({s}) * {p})"
            new_ranges[i] = (0, dace.symbolic.int_floor(e - b, s), 1)
            changed = True
        if not changed:
            return False

        me.map.range = dace.subsets.Range(new_ranges)

        def _subs(x):
            return x.subs(subsdict) if isinstance(x, sympy.Basic) else x

        # Param-local: substitute only within this map's scope.
        scope = state.scope_subgraph(me, include_entry=True, include_exit=True)
        for edge in scope.edges():
            md = edge.data
            if md is None or md.data is None or md.subset is None:
                continue
            md.subset = dace.subsets.Range([(_subs(rb), _subs(re), _subs(rs)) for rb, re, rs in md.subset.ndrange()])
        self._repl_tasklets_on_node_list(state, list(scope.nodes()), repldict)
        for n in scope.nodes():
            if isinstance(n, nodes.NestedSDFG):
                n.symbol_mapping = {
                    k: dace.symbolic.pystr_to_symbolic(str(v)).subs(subsdict)
                    for k, v in n.symbol_mapping.items()
                }
        return True

    def _normalize_loop(self, loop: LoopRegion) -> bool:
        """Normalize one ``LoopRegion`` to a ``0 : n : 1`` counter in place.

        ``loop_analysis.get_loop_end`` returns the inclusive last iteration
        value for ``< <= > >=``; with ``start``/``step`` the exact trip count
        is ``floor((end-start)/step)+1``. The body is rewritten
        ``var -> start + step*var`` (value-preserving) and the header reset to
        ``var=0 ; var < n ; var = var+1``.

        :param loop: The loop region.
        :returns: ``True`` if the loop was rewritten.
        """
        var = loop.loop_variable
        start = loop_analysis.get_init_assignment(loop)
        end = loop_analysis.get_loop_end(loop)
        step = loop_analysis.get_loop_stride(loop)
        if not var or start is None or end is None or step is None:
            return False  # Conservative: only the affine canonical forms.
        if start == 0 and step == 1:
            return False

        n = dace.symbolic.int_floor(end - start, step) + 1
        repldict = {str(var): f"(({start}) + ({step}) * {var})"}
        # Rewrite the body (memlets/tasklets/interstate/nested); the loop's own
        # header is not a node within itself, so reset it explicitly after.
        self._repl_recursive(loop, repldict)
        loop.init_statement = CodeBlock(f"{var} = 0")
        loop.loop_condition = CodeBlock(f"{var} < ({n})")
        loop.update_statement = CodeBlock(f"{var} = {var} + 1")
        return True

    def apply_pass(self, sdfg: dace.SDFG, _: Dict) -> Optional[int]:
        """Normalize every map and loop in ``sdfg`` (recursively).

        :param sdfg: The SDFG to normalize in place.
        :returns: The number of maps/loops rewritten, or ``None`` if unchanged.
        """
        count = 0
        for node, parent in sdfg.all_nodes_recursive():
            if isinstance(node, nodes.MapEntry) and isinstance(parent, dace.SDFGState):
                if self._normalize_map(parent, node):
                    count += 1
        for cfg in sdfg.all_control_flow_regions(recursive=True):
            if isinstance(cfg, LoopRegion) and self._normalize_loop(cfg):
                count += 1
        return count or None


@transformation.explicit_cf_compatible
class NormalizeLoopBounds(NormalizeLoopsAndMaps):
    """Rebase every ``LoopRegion`` counter to start at 0, KEEPING its stride:
    ``for i in b:e:s`` -> ``for j in 0:(e-b):s`` with body ``i -> b + j`` (an
    offset-only shift, no ``s*j`` step-collapse). Maps are left untouched.

    Two loops of the same extent but different offset (a slice map ``0:N-2``
    beside an in-row scan ``1:N-1``) become the same range once both are rebased
    to 0, so the same-range ``LoopFusion`` can join them. Keeping the stride is
    deliberate: the sibling ``NormalizeLoopsAndMaps`` (``i -> b + s*j``) was
    dropped from the pipeline because the ``a[b+s*j]`` form blocks ``LoopToMap``
    (it no longer sees a unique ``j`` index); the offset-only ``a[b+j]`` here
    keeps a constant shift ``LoopToMap`` already handles. Idempotent: a loop
    already based at 0 is skipped, so the pass may be re-run freely.
    """

    def _rebase_loop(self, loop: LoopRegion) -> bool:
        """Rebase one ``LoopRegion`` to a 0-based counter, keeping its stride.

        Substitutes ``var -> start + var`` in the body (value-preserving) and
        resets the header to the canonical ``var = 0 ; var <= (end - start) ;
        var = var + step`` (``end`` is the inclusive last value, so the shifted
        inclusive last is ``end - start``). The canonical ``var <= bound`` form is
        required: a substituted ``start + var < E`` leaves the iterator offset on
        the LHS, which ``loop_analysis`` cannot read back (breaking codegen).
        ``True`` if rewritten.
        """
        var = loop.loop_variable
        start = loop_analysis.get_init_assignment(loop)
        end = loop_analysis.get_loop_end(loop)  # inclusive last iteration value
        step = loop_analysis.get_loop_stride(loop)
        if not var or start is None or end is None or step is None:
            return False  # only the affine forms loop_analysis can resolve
        if start == 0:
            return False  # already 0-based (any stride) -> idempotent no-op
        repldict = {str(var): f"(({start}) + {var})"}
        self._repl_recursive(loop, repldict)
        new_end = dace.symbolic.simplify(end - start)
        loop.init_statement = CodeBlock(f"{var} = 0")
        loop.loop_condition = CodeBlock(f"{var} <= ({new_end})")
        loop.update_statement = CodeBlock(f"{var} = {var} + ({step})")
        return True

    def apply_pass(self, sdfg: dace.SDFG, _: Dict) -> Optional[int]:
        """Rebase every loop counter to 0 (keeping stride); leave maps untouched."""
        count = 0
        for cfg in sdfg.all_control_flow_regions(recursive=True):
            if isinstance(cfg, LoopRegion) and self._rebase_loop(cfg):
                count += 1
        return count or None


@transformation.explicit_cf_compatible
class NormalizeStridedMaps(NormalizeLoopsAndMaps):
    """Normalize only maps that carry a NON-UNIT step to ``0:trip:1``, folding
    the step into the index (``a[i]`` under ``0:N:2`` -> ``a[2*k]`` under
    ``0:int_floor(N-1,2)+1:1``). Unit-step maps and every ``LoopRegion`` counter
    are left untouched.

    The multi-dim tiler requires unit-step maps -- ``MarkTileDims`` and
    ``LiftMapReductionToReduce`` both bail on ``step != 1``. Rather than teach
    the whole tile lowering to carry a per-lane step (and hit the non-uniform
    step/index composition pitfall -- ``a[i]`` needs ``*step`` but a reduction
    buffer ``buf[i//step]`` must not), run this at the vectorizer entry: AFTER
    ``LoopToMap`` (so, unlike an early loop-arm rewrite, it can never block a
    lift by making a scatter's write index non-provably-injective) and BEFORE
    the reduction lifts. The strided reduction map a frontend
    ``for i in range(0, N, 2)`` lifts to then becomes dense, and its ``a[2*k]``
    read is a plain STRIDED index the tiler's existing ``dim_strides`` machinery
    vectorizes with no step-aware change anywhere in the tile lowering. The
    ``p -> b + s*p`` substitution is value-preserving; the ``int_floor(e-b, s)``
    trip is nonnegative under the canon "symbols nonnegative" contract.
    """

    def apply_pass(self, sdfg: dace.SDFG, _: Dict) -> Optional[int]:
        """Normalize every map with a non-unit step; leave unit-step maps and
        all loops alone."""
        count = 0
        for node, parent in sdfg.all_nodes_recursive():
            if not (isinstance(node, nodes.MapEntry) and isinstance(parent, dace.SDFGState)):
                continue
            if any(str(s) != "1" for (_, _, s) in node.map.range.ranges) and self._normalize_map(parent, node):
                count += 1
        return count or None
