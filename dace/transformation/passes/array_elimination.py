# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set

from dace import SDFG, SDFGState, data, properties
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.analysis import cfg
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.validation import InvalidSDFGNodeError
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.dataflow import (RedundantArray, RedundantReadSlice, RedundantSecondArray, RedundantWriteSlice,
                                          SqueezeViewRemove, UnsqueezeViewRemove, RemoveSliceView)
from dace.transformation.passes import analysis as ap
from dace.transformation.transformation import SingleStateTransformation


@properties.make_properties
@transformation.explicit_cf_compatible
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
        reachable: Dict[SDFGState, Set[SDFGState]] = pipeline_results[ap.StateReachability.__name__][sdfg.cfg_id]
        # Get access nodes and modify set as pass continues
        access_sets: Dict[str, Set[SDFGState]] = pipeline_results[ap.FindAccessStates.__name__][sdfg.cfg_id]

        # Traverse SDFG backwards
        try:
            state_order = list(cfg.blockorder_topological_sort(sdfg, recursive=True, ignore_nonstate_blocks=True))
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
                desc = sdfg.arrays[aname]
                if not isinstance(desc, data.View) and (isinstance(desc, data.Structure) and len(desc.members) > 0):
                    continue
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
        for data_container in access_nodes.keys():
            nodeset = access_nodes[data_container]
            if len(nodeset) > 1:
                # Merge all other access nodes to the first one that fits the condition, if one exists.
                first_node = None
                first_node_idx = 0
                for i, node in enumerate(nodeset[:-1]):
                    if condition(node):
                        first_node = node
                        first_node_idx = i
                        break
                if first_node is None:
                    continue

                for node in nodeset[first_node_idx + 1:]:
                    if not condition(node):
                        continue

                    # Reconnect edges to first node.
                    # If we are handling views, we do not want to add more than one edge going into a 'views' connector,
                    # so we only merge nodes if the memlets match exactly (which they should). But in that case without
                    # copying the edge.
                    edges: List[MultiConnectorEdge[Memlet]] = state.all_edges(node)
                    other_edges: List[MultiConnectorEdge[Memlet]] = []
                    for edge in edges:
                        if edge.dst is node:
                            if edge.dst_conn == 'views':
                                other_edges = list(state.in_edges_by_connector(first_node, 'views'))
                                if len(other_edges) != 1:
                                    raise InvalidSDFGNodeError('Multiple edges connected to views connector',
                                                               state.sdfg, state.block_id, state.node_id(first_node))
                                other_view_edge = other_edges[0]
                                if other_view_edge.data != edge.data:
                                    # The memlets do not match, skip the node.
                                    continue
                            else:
                                state.add_edge(edge.src, edge.src_conn, first_node, edge.dst_conn, edge.data)
                        else:
                            if edge.src_conn == 'views':
                                other_edges = list(state.out_edges_by_connector(first_node, 'views'))
                                if len(other_edges) != 1:
                                    raise InvalidSDFGNodeError('Multiple edges connected to views connector',
                                                               state.sdfg, state.block_id, state.node_id(first_node))
                                other_view_edge = other_edges[0]
                                if other_view_edge.data != edge.data:
                                    # The memlets do not match, skip the node.
                                    continue
                            else:
                                state.add_edge(first_node, edge.src_conn, edge.dst, edge.dst_conn, edge.data)
                    # Remove merged node and associated edges
                    state.remove_node(node)
                    removed_nodes.add(node)
                access_nodes[data_container] = [n for n in nodeset if n not in removed_nodes]
        return removed_nodes

    def remove_redundant_views(self, sdfg: SDFG, state: SDFGState, access_nodes: Dict[str, List[nodes.AccessNode]]):
        """
        Removes access nodes that contain views, which can be represented normally by memlets. For example, slices.
        """
        removed_nodes: Set[nodes.AccessNode] = set()
        xforms = [RemoveSliceView()]
        state_id = state.block_id

        for nodeset in access_nodes.values():
            for anode in list(nodeset):
                for xform in xforms:
                    # Quick path to setup match
                    candidate = {type(xform).view: anode}
                    xform.setup_match(sdfg, state.parent_graph.cfg_id, state_id, candidate, 0, override=True)

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
        state_id = state.block_id

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
                                xform.setup_match(sdfg, state.parent_graph.cfg_id, state_id, candidate, 0,
                                                  override=True)

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
                                xform.setup_match(sdfg, state.parent_graph.cfg_id, state_id, candidate, 0,
                                                  override=True)

                                # Try to apply
                                if xform.can_be_applied(state, 0, sdfg):
                                    ret = xform.apply(state, sdfg)
                                    if ret is not None:  # A view was created
                                        continue
                                    removed_nodes.add(anode)
                                    removed.add(anode)
                                    break

        return removed_nodes
# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set

from dace import SDFG, InterstateEdge, SDFGState, data, properties
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.analysis import cfg
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.sdfg.validation import InvalidSDFGNodeError
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.dataflow import (RedundantArray, RedundantReadSlice, RedundantSecondArray, RedundantWriteSlice,
                                          SqueezeViewRemove, UnsqueezeViewRemove, RemoveSliceView)
from dace.transformation.passes import analysis as ap
from dace.transformation.transformation import SingleStateTransformation
from dace.symbolic import pystr_to_symbolic, free_symbols_and_functions


@properties.make_properties
@transformation.explicit_cf_compatible
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
        reachable: Dict[SDFGState, Set[SDFGState]] = pipeline_results[ap.StateReachability.__name__][sdfg.cfg_id]
        # Get access nodes and modify set as pass continues
        access_sets: Dict[str, Set[SDFGState]] = pipeline_results[ap.FindAccessStates.__name__][sdfg.cfg_id]
        views_used_in_interstate_edges_and_cfgs = self._get_views_used_in_interstate_edges_and_cfgs(sdfg)
        # Traverse SDFG backwards
        try:
            state_order = list(cfg.blockorder_topological_sort(sdfg, recursive=True, ignore_nonstate_blocks=True))
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

            # Remove redundant views
            removed_nodes |= self.remove_redundant_views(sdfg, state, access_nodes, views_used_in_interstate_edges_and_cfgs)

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
                desc = sdfg.arrays[aname]
                if not isinstance(desc, data.View) and (isinstance(desc, data.Structure) and len(desc.members) > 0):
                    continue
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
        for data_container in access_nodes.keys():
            nodeset = access_nodes[data_container]
            if len(nodeset) > 1:
                # Merge all other access nodes to the first one that fits the condition, if one exists.
                first_node = None
                first_node_idx = 0
                for i, node in enumerate(nodeset[:-1]):
                    if condition(node):
                        first_node = node
                        first_node_idx = i
                        break
                if first_node is None:
                    continue

                for node in nodeset[first_node_idx + 1:]:
                    if not condition(node):
                        continue

                    # Reconnect edges to first node.
                    # If we are handling views, we do not want to add more than one edge going into a 'views' connector,
                    # so we only merge nodes if the memlets match exactly (which they should). But in that case without
                    # copying the edge.
                    edges: List[MultiConnectorEdge[Memlet]] = state.all_edges(node)
                    other_edges: List[MultiConnectorEdge[Memlet]] = []
                    for edge in edges:
                        if edge.dst is node:
                            if edge.dst_conn == 'views':
                                other_edges = list(state.in_edges_by_connector(first_node, 'views'))
                                if len(other_edges) != 1:
                                    raise InvalidSDFGNodeError('Multiple edges connected to views connector',
                                                               state.sdfg, state.block_id, state.node_id(first_node))
                                other_view_edge = other_edges[0]
                                if other_view_edge.data != edge.data:
                                    # The memlets do not match, skip the node.
                                    continue
                            else:
                                state.add_edge(edge.src, edge.src_conn, first_node, edge.dst_conn, edge.data)
                        else:
                            if edge.src_conn == 'views':
                                other_edges = list(state.out_edges_by_connector(first_node, 'views'))
                                if len(other_edges) != 1:
                                    raise InvalidSDFGNodeError('Multiple edges connected to views connector',
                                                               state.sdfg, state.block_id, state.node_id(first_node))
                                other_view_edge = other_edges[0]
                                if other_view_edge.data != edge.data:
                                    # The memlets do not match, skip the node.
                                    continue
                            else:
                                state.add_edge(first_node, edge.src_conn, edge.dst, edge.dst_conn, edge.data)
                    # Remove merged node and associated edges
                    state.remove_node(node)
                    removed_nodes.add(node)
                access_nodes[data_container] = [n for n in nodeset if n not in removed_nodes]
        return removed_nodes

    def remove_redundant_views(self, sdfg: SDFG, state: SDFGState, access_nodes: Dict[str, List[nodes.AccessNode]],
                               views_used_in_interstate_edges_and_cfgs: Set[str]):
        """
        Removes access nodes that contain views, which can be represented normally by memlets. For example, slices.
        """
        removed_nodes: Set[nodes.AccessNode] = set()
        xforms = [RemoveSliceView()]
        state_id = state.block_id

        for nodeset in access_nodes.values():
            for anode in list(nodeset):
                for xform in xforms:
                    # Quick path to setup match
                    candidate = {type(xform).view: anode}
                    xform.setup_match(sdfg, state.parent_graph.cfg_id, state_id, candidate, 0, override=True)

                    # Try to apply
                    if xform.can_be_applied(state, 0, sdfg) and anode.data not in views_used_in_interstate_edges_and_cfgs:
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
        state_id = state.block_id

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
                                xform.setup_match(sdfg, state.parent_graph.cfg_id, state_id, candidate, 0,
                                                  override=True)

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
                                xform.setup_match(sdfg, state.parent_graph.cfg_id, state_id, candidate, 0,
                                                  override=True)

                                # Try to apply
                                if xform.can_be_applied(state, 0, sdfg):
                                    ret = xform.apply(state, sdfg)
                                    if ret is not None:  # A view was created
                                        continue
                                    removed_nodes.add(anode)
                                    removed.add(anode)
                                    break

        return removed_nodes

    def _get_views_used_in_interstate_edges_and_cfgs(self, sdfg: SDFG):
        used_names = set()
        view_names = [k for k, v in sdfg.arrays.items() if isinstance(v, data.View)]
        for cfg in sdfg.nodes():
            if isinstance(cfg, LoopRegion):
                        assert cfg.loop_variable not in view_names
                        for free_name in cfg.free_symbols:
                            if free_name in view_names:
                                used_names.add(free_name)

            elif isinstance(cfg, ConditionalBlock):
                        for branch in cfg.branches:
                            cb = branch[0]
                            cfg = branch[1]
                            symbols_to_check = (
                                set.union(
                                    cb.get_free_symbols(), cfg.free_symbols
                                ) if cb is not None else cfg.free_symbols
                            )
                            for free_name in symbols_to_check:
                                if free_name in view_names:
                                    used_names.add(free_name)
                                    
            elif not isinstance(cfg, SDFGState): # Needs to be CFG
                for node in cfg.nodes():
                    if isinstance(node, LoopRegion):
                        assert node.loop_variable not in view_names
                        for free_name in cfg.free_symbols:
                            if free_name in view_names:
                                used_names.add(free_name)
                    elif isinstance(node, ConditionalBlock):
                        for branch in node.branches:
                            cb = branch[0]
                            cfg = branch[1]
                            symbols_to_check = (
                                set.union(
                                    cb.get_free_symbols(), cfg.free_symbols
                                ) if cb is not None else cfg.free_symbols
                            )
                            for free_name in symbols_to_check:
                                if free_name in view_names:
                                    used_names.add(free_name)
                    else:
                        assert isinstance(node, SDFGState)

        for edge, _ in sdfg.all_edges_recursive():
            interstate_edge: InterstateEdge = edge.data
            if not isinstance(interstate_edge, InterstateEdge):
                continue
            
            # Get parents if possible
            dst_parent = None if not hasattr(edge.dst, 'parent') else edge.dst.parent
            src_parent = None if not hasattr(edge.src, 'parent') else edge.src.parent
            
            # Skip edges in nested SDFGs
            if not (dst_parent is sdfg and src_parent is sdfg):
                continue
            
            # array reads are treated as functions
            value_syms = set().union(*(free_symbols_and_functions(pystr_to_symbolic(v)) for v in edge.data.assignments.values()))

            for free_name in set.union(interstate_edge.condition.get_free_symbols(),
                                       interstate_edge.assignments.keys(),
                                       value_syms):
                if free_name in view_names:
                    used_names.add(free_name)

        return used_names
