# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

import copy
import re
from dataclasses import dataclass
from dace.transformation import pass_pipeline as ppl, transformation
from dace import SDFG, properties, SDFGState
from typing import Dict, Set, Optional, Tuple
from dace.sdfg.nodes import AccessNode, Tasklet, Node
from dace.data import View


@dataclass(unsafe_hash=True)
@properties.make_properties
@transformation.explicit_cf_compatible
class ViewConflictSanitization(ppl.Pass):
    """
    Checks if any views with the same name have conflicting access patterns.
    Renames the views to avoid conflicts.
    """

    CATEGORY: str = "Sanitization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Memlets | ppl.Modifies.AccessNodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If anything was modified, reapply
        return modified != ppl.Modifies.Nothing

    def apply_pass(self, sdfg: SDFG, _) -> Optional[Set[str]]:
        # Can only occur within a single state
        for node, parent in sdfg.all_nodes_recursive():
            if isinstance(parent, SDFGState):
                self._apply_to_state(parent.sdfg, parent)

    # Applies to a single state
    def _apply_to_state(self, sdfg: SDFG, state: SDFGState) -> None:
        # Collect all views with their incoming indices and the tasklets they are connected to
        views = self._get_views(sdfg, state)

        # Find pairs that have overlapping tasklets and different indices
        for v1, (i1, t1) in views.items():
            for v2, (i2, t2) in views.items():
                if v1 == v2:
                    continue
                if v1.data != v2.data:
                    continue
                if t1.intersection(t2) == set():
                    continue
                if i1 == i2:
                    continue

                # Rename conflicting views
                self._rename_view(sdfg, state, v2)

    # Returns a dictionary of all views in the state, with their name as key and a tuple: their indices, tasklets they are connected to
    def _get_views(
        self, sdfg: SDFG, state: SDFGState
    ) -> Dict[AccessNode, Tuple[Set[str], Set[Tasklet]]]:
        views = {}
        for node in state.nodes():
            if not isinstance(node, AccessNode):
                continue
            if not isinstance(node.desc(sdfg), View):
                continue
            tasklets = self._get_connected_tasklets(sdfg, state, node)

            # If "views" connector is in the in_connectors, we need to use the in_edge
            if "views" in node.in_connectors:
                assert state.in_degree(node) == 1
                in_edge = state.in_edges(node)[0]
                indices = in_edge.data.subset
                assert indices is not None
                assert node not in views
                views[node] = (indices, tasklets)
            # Otherwise, we need to use the out_edge
            else:
                assert "views" in node.out_connectors
                assert state.out_degree(node) == 1
                out_edge = state.out_edges(node)[0]
                indices = out_edge.data.subset
                assert indices is not None
                assert node not in views
                views[node] = (indices, tasklets)
        return views

    # Returns a list of all tasklets in the same subgraph as the provided node
    def _get_connected_tasklets(
        self, sdfg: SDFG, state: SDFGState, node: Node
    ) -> Set[Tasklet]:
        tasklets = set()
        preds = set([node])
        succs = set([node])
        changed = True
        while changed:
            changed = False
            new_preds = preds.copy()
            new_succs = succs.copy()
            for pas in preds:
                new_preds.update(state.predecessors(pas))
            for pas in succs:
                new_succs.update(state.successors(pas))
            if new_preds != preds or new_succs != succs:
                changed = True
                preds = new_preds
                succs = new_succs

        for n in preds.union(succs):
            if isinstance(n, Tasklet):
                tasklets.add(n)
        return tasklets

    # Given a view, change its name to a new one
    def _rename_view(self, sdfg: SDFG, state: SDFGState, view: AccessNode) -> None:
        array_desc = copy.deepcopy(view.desc(sdfg))
        old_name = view.data
        new_name = sdfg.add_datadesc_view(old_name, array_desc, find_new_name=True)
        view.data = new_name

        # XXX: Strong assumption: The view is only directly used in the incoming or outgoing edges
        repl_pattern = r"\b" + re.escape(old_name) + r"\b"
        for edge in state.in_edges(view) + state.out_edges(view):
            assert edge.data.data is not None
            edge.data.data = re.sub(repl_pattern, new_name, edge.data.data)

        # If the views connector is in the in_connectors, any view descendants also need to be renamed
        if "views" in view.in_connectors:
            for succ in state.successors(view):
                if not isinstance(succ, AccessNode):
                    continue
                if not isinstance(succ.desc(sdfg), View):
                    continue
                self._rename_view(sdfg, state, succ)

        # If the views connector is in the out_connectors, any view predecessors also need to be renamed
        else:
            assert "views" in view.out_connectors
            for pred in state.predecessors(view):
                if not isinstance(pred, AccessNode):
                    continue
                if not isinstance(pred.desc(sdfg), View):
                    continue
                self._rename_view(sdfg, state, pred)
