# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Replicate a conditional per independent output so it can be fissioned.

The dace frontend lowers ``for i: if c: A[i]=..; B[i]=..`` to a map whose
body is one ``NestedSDFG`` holding a ``ConditionalBlock``. ``MapFission``
refuses such a node (the branch cannot be split in place). This pass clones
the NestedSDFG once per independent output connector group -- the shared
condition is deep-copied into every clone -- and prunes each clone to its
group. The map then has several independent NestedSDFG children, which
ordinary ``MapFission`` splits into one map per output. It is a no-op when a
NestedSDFG has a single output group (nothing to fission).
"""
import copy
from typing import Any, Dict, Optional, Set

from dace import SDFG
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock
from dace.transformation import pass_pipeline as ppl, transformation


def _has_conditional(sdfg: SDFG) -> bool:
    """Whether ``sdfg`` contains a ``ConditionalBlock`` (recursively)."""
    return any(isinstance(cfg, ConditionalBlock) for cfg in sdfg.all_control_flow_regions(recursive=True))


def _output_dependency(sdfg: SDFG, out_name: str, input_names: Set[str]) -> Set[str]:
    """Inner array names that feed ``out_name``, excluding pure shared inputs.

    Backward-reachable AccessNode data from any writer of ``out_name`` across
    every state of ``sdfg``; NestedSDFG-input arrays are treated as shared
    leaves and not followed.

    :param sdfg: The NestedSDFG's inner SDFG.
    :param out_name: The output connector / array name to slice for.
    :param input_names: NestedSDFG input connector names (shared reads).
    :returns: The set of non-input array names ``out_name`` depends on.
    """
    deps: Set[str] = set()
    for state in sdfg.all_states():
        writers = [n for n in state.nodes() if isinstance(n, nodes.AccessNode) and n.data == out_name]
        seen = set()
        stack = list(writers)
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            if isinstance(node, nodes.AccessNode):
                if node.data in input_names:
                    continue
                deps.add(node.data)
            for e in state.in_edges(node):
                stack.append(e.src)
    return deps


@transformation.explicit_cf_compatible
class ConditionalComponentFission(ppl.Pass):
    """Replicate a NestedSDFG's conditional per independent output group."""
    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Split every qualifying NestedSDFG-with-conditional in ``sdfg``.

        :param sdfg: The SDFG to transform in place.
        :returns: Number of NestedSDFGs replicated, or ``None`` if none.
        """
        from dace.transformation.passes.simplify import SimplifyPass

        count = 0
        for nsdfg in list(sdfg.all_sdfgs_recursive()):
            for state in list(nsdfg.states()):
                for node in [n for n in state.nodes() if isinstance(n, nodes.NestedSDFG)]:
                    groups = self._independent_output_groups(state, node)
                    if groups is None or len(groups) < 2:
                        continue
                    self._split(nsdfg, state, node, groups, SimplifyPass)
                    count += 1
        return count or None

    @staticmethod
    def _independent_output_groups(state, node: nodes.NestedSDFG):
        """Partition ``node``'s output connectors into independent groups.

        :param state: The state owning ``node``.
        :param node: The NestedSDFG node.
        :returns: A list of connector-name sets, or ``None`` if the node is
            not a clean conditional-bearing candidate.
        """
        if not _has_conditional(node.sdfg):
            return None
        out_conns = [c for c in node.out_connectors]
        if len(out_conns) < 2:
            return None
        # No WCR on the boundary (it would not be replicable per group).
        for e in state.out_edges(node):
            if e.data is None or e.data.wcr is not None:
                return None
        in_names = set(node.in_connectors)
        dep = {oc: _output_dependency(node.sdfg, oc, in_names) for oc in out_conns}
        # Union-find: connectors sharing a non-input array are dependent.
        parent = {oc: oc for oc in out_conns}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, a in enumerate(out_conns):
            for b in out_conns[i + 1:]:
                if dep[a] & dep[b]:
                    parent[find(a)] = find(b)
        groups: Dict[str, Set[str]] = {}
        for oc in out_conns:
            groups.setdefault(find(oc), set()).add(oc)
        return list(groups.values())

    @staticmethod
    def _split(parent_sdfg: SDFG, state, node: nodes.NestedSDFG, groups, simplify_cls):
        """Clone ``node`` once per group, prune each, rewire, drop original."""
        in_edges = list(state.in_edges(node))
        out_edges = list(state.out_edges(node))
        for grp in groups:
            clone_sdfg = copy.deepcopy(node.sdfg)
            clone = state.add_nested_sdfg(clone_sdfg,
                                          inputs=set(node.in_connectors),
                                          outputs=set(grp),
                                          symbol_mapping=dict(node.symbol_mapping))
            for e in in_edges:
                state.add_edge(e.src, e.src_conn, clone, e.dst_conn, copy.deepcopy(e.data))
            for e in out_edges:
                if e.src_conn in grp:
                    state.add_edge(clone, e.src_conn, e.dst, e.dst_conn, copy.deepcopy(e.data))
            # The dropped outputs' arrays are now unreferenced from the
            # boundary; DCE removes their producing slice, the shared
            # condition is kept because the group's writes still need it.
            for arr in [c for c in clone_sdfg.arrays if c not in grp and c not in node.in_connectors]:
                desc = clone_sdfg.arrays[arr]
                if not desc.transient:
                    desc.transient = True
            simplify_cls().apply_pass(clone_sdfg, {})
        for e in in_edges + out_edges:
            state.remove_edge(e)
        state.remove_node(node)
