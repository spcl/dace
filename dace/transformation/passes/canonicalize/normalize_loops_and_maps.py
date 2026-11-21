# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""Normalize every map range and loop counter to canonical ``0:trip:1`` form."""

from typing import Any

import sympy

import dace
from dace import properties, symbolic
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.subsets import Subset
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.offset_loop_and_maps import (
    OffsetLoopsAndMaps,
    repl_recursive,
    repl_tasklets_on_node_list,
    tasklets_assign,
)


@properties.make_properties
@transformation.explicit_cf_compatible
class NormalizeLoopsAndMaps(OffsetLoopsAndMaps):
    """Rewrite map ranges and loop counters to ``0:trip:1`` (zero-based, unit-stride)."""

    CATEGORY: str = "Canonicalization"

    def __init__(self) -> None:
        # Identity offset/begin: this pass overrides ``apply_pass`` entirely
        # and does not use the base shifting behavior.
        super().__init__(offset_expr="0", begin_expr=None)

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Scopes | ppl.Modifies.Tasklets | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> list[type[ppl.Pass] | ppl.Pass]:
        return []

    def _normalize_map(self, state: dace.SDFGState, me: nodes.MapEntry) -> bool:
        """Normalize one map's non-canonical parameters in place."""
        new_ranges = list(me.map.range.ranges)
        repldict: dict[str, str] = {}
        subsdict: dict = {}
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
        scope = state.scope_subgraph(me, include_entry=True, include_exit=True)
        if not changed or tasklets_assign(scope.nodes(), repldict):
            return False

        me.map.range = dace.subsets.Range(new_ranges)

        def _subs(x: symbolic.SymbolicType) -> symbolic.SymbolicType:
            return x.subs(subsdict) if isinstance(x, (sympy.Basic, symbolic.SymExpr)) else x

        def subs_range(sub: Subset) -> dace.subsets.Range:
            return dace.subsets.Range([(_subs(rb), _subs(re), _subs(rs)) for rb, re, rs in sub.ndrange()])

        # Param-local: substitute only within this map's scope.
        for edge in scope.edges():
            md = edge.data
            if md is None or md.data is None or md.subset is None:
                continue
            md.subset = subs_range(md.subset)
            # The other side of a copy memlet is indexed by the same parameter, so it moves too.
            if md.other_subset is not None:
                md.other_subset = subs_range(md.other_subset)
        repl_tasklets_on_node_list(list(scope.nodes()), repldict)
        for n in scope.nodes():
            # An inner map's RANGE reads the rewritten parameter as often as a memlet does: a tiled nest spells the
            # inner sweep as ``outer : min(outer + C - 1, hi)``, and after the outer param stops being an index and
            # starts being a tile ordinal, that bound means something else entirely.
            if isinstance(n, nodes.MapEntry) and n is not me:
                n.map.range = dace.subsets.Range(
                    [(_subs(rb), _subs(re), _subs(rs)) for rb, re, rs in n.map.range.ranges]
                )
            if isinstance(n, nodes.NestedSDFG):
                new_mapping = {}
                for k, v in n.symbol_mapping.items():
                    vsym = dace.symbolic.pystr_to_symbolic(str(v))
                    # AGENT EXPERIMENT B: identity binding of a rewritten param -> push the
                    # ``b + s*p`` substitution INTO the nested SDFG so the strided index stays
                    # VISIBLE in its memlets, instead of hiding it in ``symbol_mapping``.
                    if vsym in subsdict and str(k) == str(vsym):
                        n.sdfg.replace_dict({str(k): repldict[str(vsym)]})
                        new_mapping[k] = vsym
                        # The bounds and step now read inside, so the node maps their symbols too.
                        for name in map(str, subsdict[vsym].free_symbols):
                            if name not in n.symbol_mapping and name not in new_mapping:
                                new_mapping[name] = dace.symbolic.pystr_to_symbolic(name)
                                if name not in n.sdfg.symbols:
                                    n.sdfg.add_symbol(name, state.symbols_defined_at(n)[name])
                        continue
                    new_mapping[k] = vsym.subs(subsdict)
                n.symbol_mapping = new_mapping
        return True

    def _normalize_loop(self, loop: LoopRegion) -> bool:
        """Normalize one ``LoopRegion`` to a ``0 : n : 1`` counter in place."""
        var = loop.loop_variable
        start = loop_analysis.get_init_assignment(loop)
        end = loop_analysis.get_loop_end(loop)
        step = loop_analysis.get_loop_stride(loop)
        if not var or start is None or end is None or step is None:
            return False  # Conservative: only the affine canonical forms.
        if start == 0 and step == 1:
            return False
        # The body would read ``start + step * var`` with the CURRENT start and step
        if loop_analysis.body_writes(loop, start) or loop_analysis.body_writes(loop, step):
            return False

        n = dace.symbolic.int_floor(end - start, step) + 1
        repldict = {str(var): f"(({start}) + ({step}) * {var})"}
        if tasklets_assign((node for st in loop.states() for node in st.nodes()), repldict):
            return False
        # Rewrite the body (memlets/tasklets/interstate/nested)
        repl_recursive(loop, repldict)
        loop.init_statement = CodeBlock(f"{var} = 0")
        loop.loop_condition = CodeBlock(f"{var} < ({n})")
        loop.update_statement = CodeBlock(f"{var} = {var} + 1")
        return True

    def apply_pass(self, sdfg: dace.SDFG, _: dict[str, Any]) -> int | None:
        """Normalize every map and loop in ``sdfg`` (recursively)."""
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
class NormalizeStridedMaps(NormalizeLoopsAndMaps):
    """Normalize only maps that carry a NON-UNIT step to ``0:trip:1``, folding the step into the index (``a[i]`` under
    ``0:N:2`` -> ``a[2*k]`` under ``0:int_floor(N-1,2)+1:1``)."""

    def apply_pass(self, sdfg: dace.SDFG, _: dict[str, Any]) -> int | None:
        """Normalize every map with a non-unit step; leave unit-step maps and
        all loops alone."""
        count = 0
        for node, parent in sdfg.all_nodes_recursive():
            if not (isinstance(node, nodes.MapEntry) and isinstance(parent, dace.SDFGState)):
                continue
            if any(str(s) != "1" for (_, _, s) in node.map.range.ranges) and self._normalize_map(parent, node):
                count += 1
        return count or None
