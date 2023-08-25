# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set

from dace import SDFG, SDFGState, data, properties
from dace.sdfg import nodes
from dace.sdfg.analysis import cfg
from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow import (RedundantArray, RedundantReadSlice, RedundantSecondArray, RedundantWriteSlice,
                                          SqueezeViewRemove, UnsqueezeViewRemove, RemoveSliceView)
from dace.transformation.passes import analysis as ap
from dace.transformation.transformation import SingleStateTransformation


@properties.make_properties
class ArrayElimination(ppl.Pass):
    """
    Merges and removes arrays and their corresponding accesses. This includes redundant array copies, unnecessary views,
    and duplicate access nodes.
    """

    CATEGORY: str = 'Simplification'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.AccessNodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.AccessNodes

    def depends_on(self):
        return {ap.StateReachability, ap.FindAccessStates}

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Set[str]]:
        """
        Removes redundant arrays and access nodes.
        
        :param sdfg: The SDFG to modify.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is populated with prior Pass
                                 results as ``{Pass subclass name: returned object from pass}``. If not run in a
                                 pipeline, an empty dictionary is expected.
        :return: A set of removed data descriptor names, or None if nothing changed.
        """
        result: Set[str] = set()
        reachable: Dict[SDFGState, Set[SDFGState]] = pipeline_results[ap.StateReachability.__name__][sdfg.sdfg_id]
        # Get access nodes and modify set as pass continues
        access_sets: Dict[str, Set[SDFGState]] = pipeline_results[ap.FindAccessStates.__name__][sdfg.sdfg_id]

        # Traverse SDFG backwards
        try:
            state_order = list(cfg.stateorder_topological_sort(sdfg))
        except KeyError:
            return None
        for state in reversed(state_order):
            # Find all data descriptors that will no longer be used after this state
            removable_data: Set[str] = set(
                s for s in access_sets if state in access_sets[s] and not (access_sets[s] & reachable[state]) - {state})

            # Find duplicate access nodes as an ordered list
            access_nodes: Dict[str, List[nodes.AccessNode]] = defaultdict(list)
            for node in state.nodes():
                if isinstance(node, nodes.AccessNode):
                    access_nodes[node.data].append(node)

            # Merge source and sink access nodes
            removed_nodes = self.merge_access_nodes(state, access_nodes, lambda n: state.in_degree(n) == 0)
            removed_nodes |= self.merge_access_nodes(state, access_nodes, lambda n: state.out_degree(n) == 0)

            # Update access nodes with merged nodes
            access_nodes = {k: [n for n in v if n not in removed_nodes] for k, v in access_nodes.items()}

            # Remove redundant views
            removed_nodes |= self.remove_redundant_views(sdfg, state, access_nodes)

            # Remove redundant copies and views
            removed_nodes |= self.remove_redundant_copies(sdfg, state, removable_data, access_nodes)

            # Update access set if all nodes were removed
            for aname, anodes in access_nodes.items():
                if len(set(anodes) - removed_nodes) == 0:
                    access_sets[aname].remove(state)

            if removed_nodes:
                result.update({n.data for n in removed_nodes})

        # If node is completely removed from graph, erase data descriptor
        for aname, desc in list(sdfg.arrays.items()):
            if not desc.transient or isinstance(desc, data.Scalar):
                continue
            if aname not in access_sets or not access_sets[aname]:
                sdfg.remove_data(aname, validate=False)
                result.add(aname)

        return result or None

    def report(self, pass_retval: Set[str]) -> str:
        return f'Eliminated {len(pass_retval)} arrays: {pass_retval}.'

    def merge_access_nodes(self, state: SDFGState, access_nodes: Dict[str, List[nodes.AccessNode]],
                           condition: Callable[[nodes.AccessNode], bool]):
        """
        Merges access nodes that follow the same conditions together to the first access node.
        """
        removed_nodes: Set[nodes.AccessNode] = set()
        for nodeset in access_nodes.values():
            if len(nodeset) > 1:
                # Merge all other access nodes to the first one
                first_node = nodeset[0]
                if not condition(first_node):
                    continue
                for node in nodeset[1:]:
                    if not condition(node):
                        continue

                    # Reconnect edges to first node
                    for edge in state.all_edges(node):
                        if edge.dst is node:
                            state.add_edge(edge.src, edge.src_conn, first_node, edge.dst_conn, edge.data)
                        else:
                            state.add_edge(first_node, edge.src_conn, edge.dst, edge.dst_conn, edge.data)
                    # Remove merged node and associated edges
                    state.remove_node(node)
                    removed_nodes.add(node)
        return removed_nodes

    def remove_redundant_views(self, sdfg: SDFG, state: SDFGState, access_nodes: Dict[str, List[nodes.AccessNode]]):
        """
        Removes access nodes that contain views, which can be represented normally by memlets. For example, slices.
        """
        removed_nodes: Set[nodes.AccessNode] = set()
        xforms = [RemoveSliceView()]
        state_id = sdfg.node_id(state)

        for nodeset in access_nodes.values():
            for anode in list(nodeset):
                for xform in xforms:
                    # Quick path to setup match
                    candidate = {type(xform).view: anode}
                    xform.setup_match(sdfg, sdfg.sdfg_id, state_id, candidate, 0, override=True)

                    # Try to apply
                    if xform.can_be_applied(state, 0, sdfg):
                        xform.apply(state, sdfg)
                        removed_nodes.add(anode)
                        nodeset.remove(anode)
        return removed_nodes

    def remove_redundant_copies(self, sdfg: SDFG, state: SDFGState, removable_data: Set[str],
                                access_nodes: Dict[str, List[nodes.AccessNode]]):
        """
        Removes access nodes that represent redundant copies and/or views.
        """
        removed_nodes: Set[nodes.AccessNode] = set()
        state_id = sdfg.node_id(state)

        # Transformations that remove the first access node
        xforms_first: List[SingleStateTransformation] = [RedundantWriteSlice(), UnsqueezeViewRemove(), RedundantArray()]
        # Transformations that remove the second access node
        xforms_second: List[SingleStateTransformation] = [
            RedundantReadSlice(), SqueezeViewRemove(),
            RedundantSecondArray()
        ]

        # Try the different redundant copy/view transformations on the node
        removed = {1}
        while removed:
            removed = set()
            for aname in removable_data:
                if aname not in access_nodes:  # May be in inter-state edges
                    continue
                for anode in access_nodes[aname]:
                    if anode in removed_nodes:
                        continue
                    if anode not in state.nodes():
                        removed_nodes.add(anode)
                        continue

                    if state.out_degree(anode) == 1:
                        succ = state.successors(anode)[0]
                        if isinstance(succ, nodes.AccessNode):
                            for xform in xforms_first:
                                # Quick path to setup match
                                candidate = {type(xform).in_array: anode, type(xform).out_array: succ}
                                xform.setup_match(sdfg, sdfg.sdfg_id, state_id, candidate, 0, override=True)

                                # Try to apply
                                if xform.can_be_applied(state, 0, sdfg):
                                    ret = xform.apply(state, sdfg)
                                    if ret is not None:  # A view was created
                                        continue
                                    removed_nodes.add(anode)
                                    removed.add(anode)
                                    break

                    if anode in removed_nodes:  # Node was removed, skip second check
                        continue

                    if state.in_degree(anode) == 1:
                        pred = state.predecessors(anode)[0]
                        if isinstance(pred, nodes.AccessNode):
                            for xform in xforms_second:
                                # Quick path to setup match
                                candidate = {type(xform).in_array: pred, type(xform).out_array: anode}
                                xform.setup_match(sdfg, sdfg.sdfg_id, state_id, candidate, 0, override=True)

                                # Try to apply
                                if xform.can_be_applied(state, 0, sdfg):
                                    ret = xform.apply(state, sdfg)
                                    if ret is not None:  # A view was created
                                        continue
                                    removed_nodes.add(anode)
                                    removed.add(anode)
                                    break

        return removed_nodes
