# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""Normalize every map range to canonical ``0:trip:1`` form.

``OffsetLoopsAndMaps`` only shifts by a fixed expression and never changes the
step, so it cannot produce the canonical zero-based / unit-stride form. This
pass does: for a map parameter ``p`` with range ``b:e:s`` it substitutes
``p -> b + s*p`` in the map's own scope (memlets + tasklets, param-local) and
sets the range to ``0:(e-b)//s:1``. The substitution is value-preserving, so
the SDFG result is unchanged. It reuses ``OffsetLoopsAndMaps``' tasklet
token-replacement helpers.
"""
import copy
from typing import Dict, Optional

import dace
from dace import properties
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.offset_loop_and_maps import OffsetLoopsAndMaps
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.vectorization.insert_assign_tasklets_at_map_boundary import (
    InsertAssignTaskletsAtMapBoundary)


@properties.make_properties
@transformation.explicit_cf_compatible
class NormalizeLoopsAndMaps(OffsetLoopsAndMaps):
    """Rewrite map ranges to ``0:trip:1`` (zero-based, unit-stride).

    Loop (``LoopRegion``) normalization is a TODO: loops are turned into maps
    by the pipeline's ``loop_to_map`` stage, so normalizing maps suffices for
    the canonical form here.
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

    def depends_on(self):
        return set()

    def _create_new_memlet(self, edge_data, repldict):
        """Subset-substitute a memlet via proper dace symbols.

        Overrides the base (which sympifies string values and so mis-parses a
        symbol literally named ``S`` as ``sympy.S``, and raises on
        ``other_subset``). ``other_subset`` AN->AN copies are removed first by
        ``InsertAssignTaskletsAtMapBoundary``; handle it defensively anyway.
        """
        if edge_data is None or edge_data.subset is None:
            return None
        sd = {dace.symbolic.pystr_to_symbolic(k): dace.symbolic.pystr_to_symbolic(v) for k, v in repldict.items()}

        def _r(sub):
            if sub is None:
                return None
            return dace.subsets.Range([
                (b.subs(sd) if hasattr(b, 'subs') else b, e.subs(sd) if hasattr(e, 'subs') else e,
                 s.subs(sd) if hasattr(s, 'subs') else s) for b, e, s in sub.ndrange()
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
        :return: ``True`` if any parameter was rewritten.
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
            return x.subs(subsdict) if hasattr(x, 'subs') else x

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
        :return: ``True`` if the loop was rewritten.
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
        :return: The number of maps/loops rewritten, or ``None`` if unchanged.
        """
        # Split ``AccessNode -[other_subset]-> AccessNode`` copies into assign
        # tasklets so the reused memlet-replacement never hits ``other_subset``.
        InsertAssignTaskletsAtMapBoundary().apply_pass(sdfg, {})

        count = 0
        for node, parent in sdfg.all_nodes_recursive():
            if isinstance(node, nodes.MapEntry) and isinstance(parent, dace.SDFGState):
                if self._normalize_map(parent, node):
                    count += 1
        for cfg in sdfg.all_control_flow_regions(recursive=True):
            if isinstance(cfg, LoopRegion) and self._normalize_loop(cfg):
                count += 1
        return count or None
