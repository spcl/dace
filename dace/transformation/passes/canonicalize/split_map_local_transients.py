# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Give each map scope its own instance of a map-local transient that several scopes share by name.

A ``Scope``-lifetime transient used in more than one state is allocated once for the whole SDFG
(:meth:`~dace.codegen.targets.framecode.DaCeCodeGenerator.determine_allocation_lifetime`). When every use
is inside a map body, that one buffer is shared by the threads of a parallel map: a race. Loop unswitching
clones a loop into both branches of a guard and leaves exactly that (velocity_tendencies' ``lvn_only``).

Contract: a transient (not a view, not ``Persistent``/``Global``, not named in control flow) whose every
access node is inside a map scope, in two or more ``(state, outermost map)`` scopes, each of which writes it
before reading it, gets a fresh descriptor per scope after the first. Values never flow between the scopes,
so renaming keeps the semantics; each instance is then allocated inside its own map.
"""

import copy
from typing import Any

from dace import SDFG, data, dtypes, properties
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl

#: Lifetimes whose buffer outlives one scope by design.
_KEEP_LIFETIMES = (dtypes.AllocationLifetime.Persistent, dtypes.AllocationLifetime.Global)


def _control_flow_names(sdfg: SDFG) -> set[str]:
    """Free symbols of the interstate edges and loop/branch code of ``sdfg`` (not nested SDFGs)."""
    names: set[str] = set()
    for edge in sdfg.all_interstate_edges():
        names |= edge.data.free_symbols
    for region in sdfg.all_control_flow_regions():
        if isinstance(region, LoopRegion):
            for code in (region.loop_condition, region.init_statement, region.update_statement):
                if code is not None:
                    names |= code.get_free_symbols()
        elif isinstance(region, ConditionalBlock):
            for condition, _ in region.branches:
                if condition is not None:
                    names |= condition.get_free_symbols()
    return names


def _written_before_read(state: SDFGState, group: list[nodes.AccessNode]) -> bool:
    """Every read access node of ``group`` is the written node itself or reachable from a written one."""
    writes = [n for n in group if any(not e.data.is_empty() for e in state.in_edges(n))]
    if not writes:
        return False
    reached: set[nodes.Node] = set()
    for w in writes:
        if w not in reached:
            reached.update(state.bfs_nodes(w))
    return all(n in reached for n in group if any(not e.data.is_empty() for e in state.out_edges(n)))


@properties.make_properties
class SplitMapLocalTransients(ppl.Pass):
    """Rename the per-scope instances of map-local transients shared by name across scopes."""

    CATEGORY: str = "Optimization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.AccessNodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> list[type[ppl.Pass] | ppl.Pass]:
        return []

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        count = 0
        for g in sdfg.all_sdfgs_recursive():
            count += self._apply_sdfg(g)
        return count or None

    @staticmethod
    def _apply_sdfg(sdfg: SDFG) -> int:
        groups: dict[str, dict[tuple[SDFGState, nodes.MapEntry], list[nodes.AccessNode]]] = {}
        unscoped: set[str] = set()
        for state in sdfg.states():
            sdict = None
            for node in state.data_nodes():
                desc = sdfg.arrays[node.data]
                if not desc.transient or isinstance(desc, data.View) or desc.lifetime in _KEEP_LIFETIMES:
                    continue
                if node.data in unscoped:
                    continue
                sdict = sdict or state.scope_dict()
                top = sdict[node]
                if top is None:
                    unscoped.add(node.data)
                    continue
                while sdict[top] is not None:
                    top = sdict[top]
                if not isinstance(top, nodes.MapEntry):
                    unscoped.add(node.data)
                    continue
                groups.setdefault(node.data, {}).setdefault((state, top), []).append(node)
        candidates = {n: s for n, s in groups.items() if n not in unscoped and len(s) > 1}
        if not candidates:
            return 0
        named = _control_flow_names(sdfg)
        count = 0
        for name, scopes in candidates.items():
            if name in named or not all(_written_before_read(st, grp) for (st, _), grp in scopes.items()):
                continue
            desc = sdfg.arrays[name]
            for (state, top), grp in list(scopes.items())[1:]:
                fresh = sdfg.add_datadesc(name, copy.deepcopy(desc), find_new_name=True)
                for node in grp:
                    node.data = fresh
                for edge in state.scope_subgraph(top, include_entry=True, include_exit=True).edges():
                    if edge.data.data == name:
                        edge.data.data = fresh
                count += 1
        return count

    def report(self, pass_retval: int) -> str:
        return f"Split {pass_retval} map-local transient instance(s) shared by name across scopes."
