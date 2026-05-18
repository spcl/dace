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
from typing import Dict, Optional

import sympy

import dace
from dace import properties
from dace.sdfg import nodes
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.offset_loop_and_maps import OffsetLoopsAndMaps


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

    def _normalize_map(self, state: dace.SDFGState, me: nodes.MapEntry) -> bool:
        """Normalize one map's non-canonical parameters in place.

        :param state: The state owning the map.
        :param me: The map entry.
        :return: ``True`` if any parameter was rewritten.
        """
        new_ranges = list(me.map.range.ranges)
        repldict: Dict[str, str] = {}
        subsdict: Dict[sympy.Symbol, sympy.Expr] = {}
        changed = False
        for i, (p, (b, e, s)) in enumerate(zip(me.map.params, me.map.range.ranges)):
            if b == 0 and s == 1:
                continue
            psym = dace.symbolic.pystr_to_symbolic(p)
            # p_original = b + s * p_new ; p_new in 0 : floor((e-b)/s) : 1
            subsdict[psym] = b + s * psym
            repldict[str(p)] = f"({b} + ({s}) * {p})"
            new_ranges[i] = (sympy.Integer(0), sympy.floor((e - b) / s), sympy.Integer(1))
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

    def apply_pass(self, sdfg: dace.SDFG, _: Dict) -> Optional[int]:
        """Normalize every map in ``sdfg`` (recursively).

        :param sdfg: The SDFG to normalize in place.
        :return: The number of maps rewritten, or ``None`` if unchanged.
        """
        count = 0
        for node, parent in sdfg.all_nodes_recursive():
            if isinstance(node, nodes.MapEntry) and isinstance(parent, dace.SDFGState):
                if self._normalize_map(parent, node):
                    count += 1
        return count or None
