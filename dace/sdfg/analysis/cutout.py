# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
"""
Functionality that allows users to "cut out" parts of an SDFG in a smart way (i.e., memory preserving) for localized
testing or optimization.
"""
import networkx as nx
from networkx.algorithms.flow import edmondskarp
import sympy as sp
from collections import deque
import copy
from typing import Deque, Dict, List, Set, Tuple, Union, Optional, Any
from dace import data, DataInstrumentationType
from dace.sdfg import nodes as nd, SDFG, SDFGState, utils as sdutil, InterstateEdge
from dace.memlet import Memlet
from dace.sdfg.graph import Edge, MultiConnectorEdge
from dace.sdfg.state import StateSubgraphView, SubgraphView
from dace.transformation.transformation import (MultiStateTransformation,
                                                PatternTransformation,
                                                SubgraphTransformation,
                                                SingleStateTransformation)
from dace.transformation.interstate.loop_detection import DetectLoop
from dace.transformation.passes.analysis import StateReachability


class SDFGCutout(SDFG):

    # The base SDFG the cutout was created from.
    _base_sdfg: Optional[SDFG] = None
    # The input / output configurations of the cutout.
    input_config: Set[str] = set()
    output_config: Set[str] = set()

    def __init__(self, name: str, constants_prop: Optional[Dict[str, Tuple[data.Data, Any]]] = None):
        super(SDFGCutout, self).__init__(name + '_cutout', constants_prop)
        self._base_sdfg = None
        self.input_config = set()
        self.output_config = set()
        self._in_translation = dict()
        self._out_translation = dict()

    def _instrument_base_sdfg(self) -> None:
        # Instrument symbol dumping in the base SDFG.
        base_sdfg_start_state: SDFGState = self._out_translation[self.start_state]
        base_sdfg_start_state.symbol_instrument = DataInstrumentationType.Save

        # Instrument all data containers in the base SDFG that belong to the cutout's input configuration to be dumped.
        for state in self.states():
            for dn in state.data_nodes():
                if dn.data in self.input_config:
                    base_sdfg_dn: nd.AccessNode = self._out_translation[dn]
                    base_sdfg_dn.instrument = DataInstrumentationType.Save

    def _dry_run_base_sdfg(self, *args, **kwargs) -> None:
        self._instrument_base_sdfg()
        self._base_sdfg(*args, **kwargs)

    def find_inputs(self, *args, **kwargs) -> Dict[str, Union[data.ArrayLike, data.Number]]:
        self._dry_run_base_sdfg(*args, **kwargs)

        drep = self._base_sdfg.get_instrumented_data()
        if drep:
            vals: Dict[str, Union[data.ArrayLike, data.Number]] = dict()
            for ip in self.input_config.union(set(self.symbols)):
                val = drep.get_first_version(ip)
                vals[ip] = val
            return vals
        else:
            raise RuntimeError('No data report found for the base SDFG.')

    def translate_transformation_into(self, transformation: Union[PatternTransformation, SubgraphTransformation]):
        if isinstance(transformation, SingleStateTransformation):
            old_state = self._base_sdfg.node(transformation.state_id)
            transformation.state_id = self.node_id(self.start_state)
            transformation._sdfg = self
            transformation.sdfg_id = 0
            for k in transformation.subgraph.keys():
                old_node = old_state.node(transformation.subgraph[k])
                try:
                    transformation.subgraph[k] = self.start_state.node_id(self._in_translation[old_node])
                except KeyError:
                    # Ignore.
                    pass
        elif isinstance(transformation, MultiStateTransformation):
            new_sdfg_id = self._in_translation[transformation.sdfg_id]
            new_sdfg = self.sdfg_list[new_sdfg_id]
            transformation._sdfg = new_sdfg
            transformation.sdfg_id = new_sdfg_id
            for k in transformation.subgraph.keys():
                old_state = self._base_sdfg.node(transformation.subgraph[k])
                try:
                    transformation.subgraph[k] = self.node_id(self._in_translation[old_state])
                except KeyError:
                    # Ignore.
                    pass
        else:
            old_state = self._base_sdfg.node(transformation.state_id)
            transformation.state_id = self.node_id(self.start_state)
            new_subgraph = set()
            for k in transformation.subgraph:
                old_node = old_state.node(k)
                try:
                    new_subgraph.add(self.start_state.node_id(self._in_translation[old_node]))
                except KeyError:
                    # Ignore.
                    pass
            transformation.subgraph = new_subgraph

    def to_json(self, hash=False):
        cutout_json = super().to_json(hash)
        cutout_json['type'] = SDFG.__name__
        return cutout_json

    @classmethod
    def from_json(cls, json_obj, context_info=None):
        return super(SDFGCutout, cls).from_json(json_obj, context_info)

    @classmethod
    def from_transformation(
        cls, sdfg: SDFG, transformation: Union[PatternTransformation, SubgraphTransformation],
        make_side_effects_global = True, use_alibi_nodes: bool = True, reduce_input_config = True,
        symbols_map: Optional[Dict[str, Any]] = None
    ) -> Union['SDFGCutout', SDFG]:
        """
        Create a cutout from a transformation's set of affected graph elements.

        :param sdfg: The SDFG to create the cutout from.
        :param transformation: The transformation to create the cutout from.
        :param make_side_effects_global: Whether to make side effect data containers global, i.e. non-transient.
        :param use_alibi_nodes: Whether to use alibi nodes for the cutout across scope borders.
        :param reduce_input_config: Whether to reduce the input configuration where possible in singlestate cutouts.
        :param symbols_map: A mapping of symbols to values to use for the cutout. Optional, only used when reducing the
                            input configuration.
        :return: The cutout.
        """
        affected_nodes = _transformation_determine_affected_nodes(sdfg, transformation)

        if len(affected_nodes) == 0:
            cut_sdfg = copy.deepcopy(sdfg)
            transformation._sdfg = cut_sdfg
            return cut_sdfg

        target_sdfg = sdfg
        if transformation.sdfg_id >= 0 and target_sdfg.sdfg_list is not None:
            target_sdfg = target_sdfg.sdfg_list[transformation.sdfg_id]

        if (all(isinstance(n, nd.Node) for n in affected_nodes) or
            isinstance(transformation, (SubgraphTransformation, SingleStateTransformation))):
            state = target_sdfg.parent
            if transformation.state_id >= 0:
                state = target_sdfg.node(transformation.state_id)
            cutout = cls.singlestate_cutout(state, *affected_nodes, make_side_effects_global=make_side_effects_global,
                                            use_alibi_nodes=use_alibi_nodes, reduce_input_config=reduce_input_config,
                                            symbols_map=symbols_map)
            cutout.translate_transformation_into(transformation)
            return cutout
        elif isinstance(transformation, MultiStateTransformation):
            cutout = cls.multistate_cutout(*affected_nodes, make_side_effects_global=make_side_effects_global)
            # If the cutout is an SDFG, there's no need to translate the transformation.
            if isinstance(cutout, SDFGCutout):
                cutout.translate_transformation_into(transformation)
            return cutout
        raise Exception('Unsupported transformation type: {}'.format(type(transformation)))
                    
    @classmethod
    def singlestate_cutout(cls,
                           state: SDFGState,
                           *nodes: nd.Node,
                           make_copy: bool = True,
                           make_side_effects_global: bool = True,
                           use_alibi_nodes: bool = True,
                           reduce_input_config: bool = False,
                           symbols_map: Optional[Dict[str, Any]] = None) -> 'SDFGCutout':
        """
        Cut out a subgraph of a state from an SDFG to run separately for localized testing or optimization.
        The subgraph defined by the list of nodes will be extended to include access nodes of data containers necessary
        to run the graph separately. In addition, all transient data containers that may contain data when the cutout is
        executed are made global, as well as any transient data containers which are written to inside the cutout but
        may be read after the cutout.
        
        :param state: The SDFG state in which the subgraph resides.
        :param nodes: The nodes in the subgraph to cut out.
        :param make_copy: If True, deep-copies every SDFG element in the copy. Otherwise, original references are kept.
        :param make_side_effects_global: If True, all transient data containers which are read inside the cutout but may
                                         be written to _before_ the cutout, or any data containers which are written to
                                         inside the cutout but may be read _after_ the cutout, are made global.
        :param use_alibi_nodes: If True, do not extend the cutout with access nodes that span outside of a scope, but
                                introduce alibi nodes instead that represent only the accesses subset.
        :param reduce_input_config: Whether to reduce the input configuration where possible in singlestate cutouts.
        :param symbols_map: A mapping of symbols to values to use for the cutout. Optional, only used when reducing the
                            input configuration.
        :return: The created SDFGCutout.
        """
        if reduce_input_config:
            nodes = _reduce_in_configuration(state, nodes, use_alibi_nodes, symbols_map)
        create_element = copy.deepcopy if make_copy else (lambda x: x)
        sdfg = state.parent
        subgraph: StateSubgraphView = StateSubgraphView(state, nodes)
        subgraph = _extend_subgraph_with_access_nodes(state, subgraph, use_alibi_nodes)

        # Make a new SDFG with the included constants, used symbols, and data containers.
        cutout = SDFGCutout(sdfg.name + '_cutout', sdfg.constants_prop)
        cutout._base_sdfg = sdfg
        defined_syms = subgraph.defined_symbols()
        freesyms = subgraph.free_symbols
        for sym in freesyms:
            cutout.add_symbol(sym, defined_syms[sym])

        sg_edges: List[MultiConnectorEdge[Memlet]] = subgraph.edges()
        for edge in sg_edges:
            if edge.data is None or edge.data.data is None:
                continue

            memlet = edge.data
            if memlet.data in cutout.arrays:
                continue
            new_desc = sdfg.arrays[memlet.data].clone()
            cutout.add_datadesc(memlet.data, new_desc)

        # Add a single state with the extended subgraph
        new_state = cutout.add_state(state.label, is_start_state=True)
        in_translation = dict()
        out_translation = dict()
        for e in sg_edges:
            if e.src not in in_translation:
                new_el = create_element(e.src)
                in_translation[e.src] = new_el
                out_translation[new_el] = e.src
            if e.dst not in in_translation:
                new_el = create_element(e.dst)
                in_translation[e.dst] = new_el
                out_translation[new_el] = e.dst
            new_memlet = create_element(e.data)
            in_translation[e.data] = new_memlet
            out_translation[new_memlet] = e.data
            new_state.add_edge(
                in_translation[e.src], e.src_conn, in_translation[e.dst], e.dst_conn, new_memlet
            )

        # Insert remaining isolated nodes
        for n in subgraph.nodes():
            if n not in in_translation:
                new_el = create_element(n)
                in_translation[n] = new_el
                out_translation[new_el] = n
                new_state.add_node(new_el)

        # Remove remaining dangling connectors from scope nodes and add new data containers corresponding to accesses
        # for dangling connectors on other nodes.
        translation_add_pairs: Set[Tuple[nd.AccessNode, nd.AccessNode]] = set()
        for orig_node, new_node in in_translation.items():
            if isinstance(new_node, nd.Node):
                if isinstance(orig_node, (nd.EntryNode, nd.ExitNode)):
                    used_connectors = set(e.dst_conn for e in new_state.in_edges(new_node))
                    for conn in (new_node.in_connectors.keys() - used_connectors):
                        new_node.remove_in_connector(conn)
                    used_connectors = set(e.src_conn for e in new_state.out_edges(new_node))
                    for conn in (new_node.out_connectors.keys() - used_connectors):
                        new_node.remove_out_connector(conn)
                else:
                    used_connectors = set(e.dst_conn for e in new_state.in_edges(new_node))
                    for conn in (new_node.in_connectors.keys() - used_connectors):
                        prune = True
                        for e in state.in_edges(orig_node):
                            if e.dst_conn and e.dst_conn == conn:
                                _, n_access = _create_alibi_access_node_for_edge(
                                    cutout, new_state, sdfg, e, None, None, new_node, conn
                                )
                                e_path = state.memlet_path(e)
                                translation_add_pairs.add((e_path[0].src, n_access))
                                prune = False
                                break
                        if prune:
                            new_node.remove_in_connector(conn)
                    used_connectors = set(e.src_conn for e in new_state.out_edges(new_node))
                    for conn in (new_node.out_connectors.keys() - used_connectors):
                        prune = True
                        for e in state.out_edges(orig_node):
                            if e.src_conn and e.src_conn == conn:
                                _, n_access = _create_alibi_access_node_for_edge(
                                    cutout, new_state, sdfg, e, new_node, conn, None, None
                                )
                                e_path = state.memlet_path(e)
                                translation_add_pairs.add((e_path[-1].dst, n_access))
                                prune = False
                                break
                        if prune:
                            new_node.remove_out_connector(conn)
        for (outer, inner) in translation_add_pairs:
            in_translation[outer] = inner
            out_translation[inner] = outer

        in_translation[state] = new_state
        out_translation[new_state] = state
        in_translation[sdfg.sdfg_id] = cutout.sdfg_id
        out_translation[cutout.sdfg_id] = sdfg.sdfg_id

        # Determine what counts as inputs / outputs to the cutout and make those data containers global / non-transient.
        if make_side_effects_global:
            in_reach, out_reach = _determine_cutout_reachability(cutout, sdfg, in_translation, out_translation)
            cutout.input_config = _cutout_determine_input_config(cutout, in_reach, in_translation, out_translation)
            cutout.output_config = _cutout_determine_output_configuration(
                cutout, out_reach, in_translation, out_translation
            )
            for d_name in cutout.input_config.union(cutout.output_config):
                cutout.arrays[d_name].transient = False

        cutout._in_translation = in_translation
        cutout._out_translation = out_translation

        # Translate in nested SDFG nodes and their SDFGs (their list id, specifically).
        cutout.reset_sdfg_list()
        outers = set(in_translation.keys())
        for outer in outers:
            if isinstance(outer, nd.NestedSDFG):
                inner: nd.NestedSDFG = in_translation[outer]
                cutout._in_translation[outer.sdfg.sdfg_id] = inner.sdfg.sdfg_id
        _recursively_set_nsdfg_parents(cutout)

        return cutout

    @classmethod
    def multistate_cutout(cls,
                          *states: SDFGState,
                          make_side_effects_global: bool = True) -> Union['SDFGCutout', SDFG]:
        """
        Cut out a multi-state subgraph from an SDFG to run separately for localized testing or optimization.

        The subgraph defined by the list of states will be extended to include any additional states necessary to make
        the resulting cutout valid and executable, i.e, to ensure that there is a distinct start state. This is achieved
        by gradually adding more states from the cutout's predecessor frontier until a distinct, single entry state is
        obtained.

        :see: _stateset_predecessor_frontier

        :param states: The subgraph states to cut out.
        :param make_side_effects_global: If True, all transient data containers which are read inside the cutout but may
                                        be written to _before_ the cutout, or any data containers which are written to
                                        inside the cutout but may be read _after_ the cutout, are made global.
        :return: The created SDFGCutout or the original SDFG where no smaller cutout could be obtained.
        """
        create_element = copy.deepcopy

        # Check that all states are inside the same SDFG.
        sdfg = list(states)[0].parent
        if any(i.parent != sdfg for i in states):
            raise Exception('Not all cutout states reside in the same SDFG')

        cutout_states: Set[SDFGState] = set(states)

        # Determine the start state and ensure there IS a unique start state. If there is no unique start state, keep
        # adding states from the predecessor frontier in the state machine until a unique start state can be determined.
        start_state: Optional[SDFGState] = None
        for state in cutout_states:
            if state == sdfg.start_state:
                start_state = state
                break

        if start_state is None:
            bfs_queue: Deque[Tuple[Set[SDFGState], Set[Edge[InterstateEdge]]]] = deque()
            bfs_queue.append(_stateset_predecessor_frontier(cutout_states))

            while len(bfs_queue) > 0:
                frontier, frontier_edges = bfs_queue.popleft()
                if len(frontier_edges) == 0:
                    # No explicit start state, but also no frontier to select from.
                    return copy.deepcopy(sdfg)
                elif len(frontier_edges) == 1:
                    # If there is only one predecessor frontier edge, its destination must be the start state.
                    start_state = list(frontier_edges)[0].dst
                else:
                    if len(frontier) == 0:
                        # No explicit start state, but also no frontier to select from.
                        return copy.deepcopy(sdfg)
                    if len(frontier) == 1:
                        # For many frontier edges but only one frontier state, the frontier state is the new start state
                        # and is included in the cutout.
                        start_state = list(frontier)[0]
                        cutout_states.add(start_state)
                    else:
                        for s in frontier:
                            cutout_states.add(s)
                        bfs_queue.append(_stateset_predecessor_frontier(cutout_states))

        subgraph: SubgraphView = SubgraphView(sdfg, cutout_states)

        # Make a new SDFG with the included constants, used symbols, and data containers.
        cutout = SDFGCutout(sdfg.name + '_cutout', sdfg.constants_prop)
        cutout._base_sdfg = sdfg
        defined_symbols: Dict[str, data.Data] = dict()
        free_symbols: Set[str] = set()
        for state in cutout_states:
            free_symbols |= state.free_symbols
            state_defined_symbols = state.defined_symbols()
            for sym in state_defined_symbols:
                defined_symbols[sym] = state_defined_symbols[sym]
        for edge in subgraph.edges():
            is_edge: InterstateEdge = edge.data
            available_symbols = sdfg.symbols.keys()
            free_symbols |= (is_edge.free_symbols & available_symbols)
            for rmem in is_edge.get_read_memlets(sdfg.arrays):
                if rmem.data in cutout.arrays:
                    continue
                new_desc = sdfg.arrays[rmem.data].clone()
                cutout.add_datadesc(rmem.data, new_desc)
        for sym in free_symbols:
            if not sym in cutout.symbols:
                cutout.add_symbol(sym, defined_symbols[sym])

        for state in cutout_states:
            for dnode in state.data_nodes():
                if dnode.data in cutout.arrays:
                    continue
                new_desc = sdfg.arrays[dnode.data].clone()
                cutout.add_datadesc(dnode.data, new_desc)

        # Add all states and state transitions required to the new cutout SDFG by traversing the state machine edges.
        sg_edges: List[Edge[InterstateEdge]] = subgraph.edges()
        in_translation = dict()
        out_translation = dict()
        for is_edge in sg_edges:
            if is_edge.src not in in_translation:
                new_el: SDFGState = create_element(is_edge.src)
                in_translation[is_edge.src] = new_el
                out_translation[new_el] = is_edge.src
                cutout.add_node(new_el, is_start_state=(is_edge.src == start_state))
                new_el.parent = cutout
            if is_edge.dst not in in_translation:
                new_el: SDFGState = create_element(is_edge.dst)
                in_translation[is_edge.dst] = new_el
                out_translation[new_el] = is_edge.dst
                cutout.add_node(new_el, is_start_state=(is_edge.dst == start_state))
                new_el.parent = cutout
            new_isedge: InterstateEdge = create_element(is_edge.data)
            in_translation[is_edge.data] = new_isedge
            out_translation[new_isedge] = is_edge.data
            cutout.add_edge(in_translation[is_edge.src], in_translation[is_edge.dst], new_isedge)

        # Add remaining necessary states.
        for state in subgraph.nodes():
            if state not in in_translation:
                new_el = create_element(state)
                in_translation[state] = new_el
                out_translation[new_el] = state
                cutout.add_node(new_el, is_start_state=(state == start_state))
                new_el.parent = cutout

        in_translation[sdfg.sdfg_id] = cutout.sdfg_id
        out_translation[cutout.sdfg_id] = sdfg.sdfg_id

        # Check interstate edges for missing data descriptors.
        for e in cutout.edges():
            for s in e.data.free_symbols:
                if s in sdfg.arrays and s not in cutout.arrays:
                    desc = sdfg.arrays[s]
                    cutout.add_datadesc(s, desc)

        # Determine what counts as inputs / outputs to the cutout and make those data containers global / non-transient.
        if make_side_effects_global:
            in_reach, out_reach = _determine_cutout_reachability(cutout, sdfg, in_translation, out_translation)
            cutout.input_config = _cutout_determine_input_config(cutout, in_reach, in_translation, out_translation)
            cutout.output_config = _cutout_determine_output_configuration(
                cutout, out_reach, in_translation, out_translation
            )
            for d_name in cutout.input_config.union(cutout.output_config):
                cutout.arrays[d_name].transient = False

        cutout._in_translation = in_translation
        cutout._out_translation = out_translation

        cutout.reset_sdfg_list()
        _recursively_set_nsdfg_parents(cutout)

        return cutout


def _transformation_determine_affected_nodes(
        sdfg: SDFG, transformation: Union[PatternTransformation, SubgraphTransformation], strict: bool = False
) -> Set[Union[nd.Node, SDFGState]]:
    """
    For a given SDFG and transformation, determine the set of nodes that are affected by the transformation.

    :param sdfg: The SDFG the transformation applies to.
    :param p: The transformation.
    :param strict: If True, include only nodes directly affected by the transformation. If False (default), ensure that
                   if scope nodes are affected, their entire scope and corresponding entry/exit nodes are part of the
                   returned set.
    :return: A set of nodes affected by the transformation.

    .. warning::
        The set of affected nodes is not guaranteed to be complete. While the set of affected nodes should by design
        always equate to the set of nodes a transformation's subgraph or pattern matches to, there is no mechanism
        preventing a transformation from affecting nodes that are not part of the pattern or subgraph they match to.
    """
    target_sdfg = sdfg
    affected_nodes = set()

    if isinstance(transformation, PatternTransformation):
        if transformation.sdfg_id >= 0 and target_sdfg.sdfg_list:
            target_sdfg = target_sdfg.sdfg_list[transformation.sdfg_id]

        for k, _ in transformation._get_pattern_nodes().items():
            try:
                affected_nodes.add(getattr(transformation, k))
            except KeyError:
                # Ignored.
                pass

        # Transformations that modify a loop in any way must also include the loop init node, i.e. the state directly
        # before the loop guard. Also make sure that ALL loop body states are part of the set of affected nodes.
        # TODO: This is hacky and should be replaced with a more general mechanism - this is something that
        #       transformation intents / transactions will need to solve.
        if isinstance(transformation, DetectLoop):
            if transformation.loop_guard is not None and transformation.loop_guard in target_sdfg.nodes():
                for iedge in target_sdfg.in_edges(transformation.loop_guard):
                    affected_nodes.add(iedge.src)
            if transformation.loop_begin is not None and transformation.loop_begin in target_sdfg.nodes():
                to_visit = [transformation.loop_begin]
                while to_visit:
                    state = to_visit.pop(0)
                    for _, dst, _ in target_sdfg.out_edges(state):
                        if dst not in affected_nodes and dst is not transformation.loop_guard:
                            to_visit.append(dst)
                    affected_nodes.add(state)

        if len(affected_nodes) == 0 and transformation.state_id < 0 and target_sdfg.parent_nsdfg_node is not None:
            # This is a transformation that affects a nested SDFG node, grab that NSDFG node.
            affected_nodes.add(target_sdfg.parent_nsdfg_node)
    else:
        if transformation.sdfg_id >= 0 and target_sdfg.sdfg_list:
            target_sdfg = target_sdfg.sdfg_list[transformation.sdfg_id]

        subgraph = transformation.get_subgraph(target_sdfg)
        for n in subgraph.nodes():
            affected_nodes.add(n)

    if strict:
        return affected_nodes

    # If strict is not set and a scope node is affected, expand the returned set to include all nodes in that scope.
    if hasattr(transformation, 'state_id') and transformation.state_id >= 0:
        state = target_sdfg.node(transformation.state_id)
        expanded = set()
        for node in affected_nodes:
            expanded.add(node)
            scope_entry = None
            if isinstance(node, nd.MapEntry):
                scope_entry = node
            elif isinstance(node, nd.MapExit):
                scope_entry = state.entry_node(node)

            if scope_entry is not None:
                scope = state.scope_subgraph(scope_entry, include_entry=True, include_exit=True)
                for n in scope.nodes():
                    expanded.add(n)
        return expanded

    return affected_nodes

def _reduce_in_configuration(state: SDFGState, affected_nodes: Set[nd.Node], use_alibi_nodes: bool = False,
                             symbols_map: Optional[Dict[str, Any]] = None) -> Set[nd.Node]:
    """
    For a given set of nodes that should be cut out in a single state cutout, try to reduce the size of the input
    configuration as much as possible by adding more nodes to find a S-T minimum 2-cut in the state.

    :param state: The state in which to cut out.
    :param affected_nodes: The set of nodes that should be cut out.
    :param use_alibi_nodes: If True, use alibi nodes across scope borders.
    :param symbols_map: A map of symbols to values. An assumption will be made about symbol values if None is provided.
    :return: A new set of node greater than or equal to the initial cutout nodes, which makes up a minimized cutout.
    """
    subgraph: StateSubgraphView = StateSubgraphView(state, affected_nodes)
    subgraph = _extend_subgraph_with_access_nodes(state, subgraph, use_alibi_nodes)
    subgraph_nodes = set(subgraph.nodes())

    # For the given state, determine what should count as the input configuration if we were to cut out the entire
    # state.
    state_reachability_dict = StateReachability().apply_pass(state.parent, None)
    state_reach = state_reachability_dict[state.parent.sdfg_id]
    reaching_cutout: Set[SDFGState] = set()
    for k, v in state_reach.items():
        if state in v:
            reaching_cutout.add(k)
    state_input_configuration = set()
    check_for_write_before = set()
    for dn in state.data_nodes():
        if state.out_degree(dn) > 0:
            # This is read from, add to the system state if it is written anywhere else in the graph.
            # Except if it is also written to at the same time and is scalar or of size 1.
            array = state.parent.arrays[dn.data]
            if state.in_degree(dn) > 0 and (array.total_size == 1 or isinstance(array, data.Scalar)):
                continue
            elif not array.transient:
                # Non-transients are always part of the input config if they are read and not overwritten anyway.
                state_input_configuration.add(dn.data)
            else:
                check_for_write_before.add(dn.data)
    for pre_state in reaching_cutout:
        for dn in pre_state.data_nodes():
            if pre_state.in_degree(dn) > 0:
                # For any writes, check if they are reads from the cutout that need to be checked. If they are, they're
                # part of the system state.
                if dn.data in check_for_write_before:
                    state_input_configuration.add(dn.data)

    # If no explicit symbol map was provided, we have to make an assumption about symbol values to determine a minimum
    # cut.
    # TODO: This is a hack. Ideally, we should be able to determine the minimum cut without having to make assumptions
    # about symbol values. Not sure how to do that yet.
    if symbols_map is None:
        symbols_map = dict()
        consts = state.parent.constants
        for s in state.parent.symbols:
            if s in consts:
                symbols_map[s] = consts[s]
            else:
                symbols_map[s] = 20

    # Use a proxy graph to compute the minium cut.
    proxy_graph = nx.DiGraph()

    # By expanding over the borders of a scope (e.g. over the entry of a map), we know that we universally can only
    # increase the size of the input configuration. Consequently, we can use the outer-most scope entry node as our
    # source node for the minimum cut, if there is such a unique outer entry node.
    source_candidates = set()
    for n in subgraph_nodes:
        source_candidates.add(state.entry_node(n))

    source = None
    scope_children = state.scope_children()
    transitive_scope_children: Dict[SDFGState, Set[SDFGState]] = dict()
    for k, v in scope_children.items():
        queue = deque(v)
        k_children = set(v)
        while queue:
            child = queue.popleft()
            if child in scope_children:
                n_children = set(scope_children[child])
                queue.extend(n_children)
                k_children.update(n_children)
        transitive_scope_children[k] = k_children
    if len(source_candidates) > 1:
        for cand in source_candidates:
            if all(other_cand in transitive_scope_children[cand] for other_cand in source_candidates):
                source = cand
                break
    elif len(source_candidates) == 1:
        source = list(source_candidates)[0]

    # If there is no unique outer entry node, we use a proxy node as the source.
    scope_nodes: Set[nd.Node] = set()
    if source == None:
        source = nd.Node()
        scope_nodes = set(scope_children[None])
    else:
        scope_nodes = set(scope_children[source])
        scope_nodes.add(source)
    expand_with = set()
    for n in scope_nodes:
        if isinstance(n, nd.EntryNode):
            exit = state.exit_node(n)
            expand_with.add(exit)
    scope_nodes.update(expand_with)
    scope_subgraph = StateSubgraphView(state, scope_nodes)

    # Add the source and a proxy sink to the proxy graph.
    proxy_graph.add_node(source)
    sink = nd.Node()
    proxy_graph.add_node(sink)

    # Build up the proxy graph.
    for edge in scope_subgraph.edges():
        proxy_edge_src = edge.src
        proxy_edge_dst = edge.dst

        vol = 0
        memlet: Memlet = edge.data
        if memlet.data:
            vol = memlet.volume
            if isinstance(vol, sp.Expr):
                vol = vol.subs(symbols_map)

        remain_free = False
        if edge.src in subgraph_nodes and edge.dst in subgraph_nodes:
            # Edge completely in subgraph, don't do anything. Unless the destination is an access node which is in the
            # state input configuration, in which case we add an edge from the source to the sink with that volume.
            if isinstance(edge.dst, nd.AccessNode) and memlet.data in state_input_configuration:
                if proxy_graph.has_edge(source, sink):
                    proxy_graph[source][sink]['capacity'] += vol
                else:
                    proxy_graph.add_node(source)
                    proxy_graph.add_node(sink)
                    proxy_graph.add_edge(source, sink, capacity=vol)
            continue
        elif edge.src in subgraph_nodes:
            # Edge starts in subgraph, ends outside.
            # If there's no path back inside, it's source is the proxy sink. Otherwise, it's source is set to the proxy
            # source and the volume is made 0, since the value will already be part of the cutout.
            if any([n in nx.descendants(state.nx, proxy_edge_src) for n in subgraph_nodes]):
                proxy_edge_src = source
                vol = 0
                remain_free = True
            else:
                proxy_edge_src = sink
        elif edge.dst in subgraph_nodes:
            # Edge starts outside, ends in the subgraph. It's destination thus is the proxy sink.
            proxy_edge_dst = sink

        if isinstance(proxy_edge_dst, nd.AccessNode) and memlet.data in state_input_configuration:
            # If the destination is an access node that is part of the state input configuration, we add an edge from
            # the source with that volume.
            if proxy_graph.has_edge(source, proxy_edge_dst):
                proxy_graph[source][proxy_edge_dst]['capacity'] += vol
            else:
                proxy_graph.add_edge(source, proxy_edge_dst, capacity=vol)
            # The actual edge between src and dst is set to have infinite capacity.
            vol = float('inf')
        elif isinstance(proxy_edge_src, nd.AccessNode) and not remain_free:
            # All outgoing edges from access nodes (with data) are set to have infinite capacity.
            vol = float('inf')

        if isinstance(proxy_edge_src, nd.ExitNode):
            proxy_edge_src = state.entry_node(proxy_edge_src)

        if proxy_graph.has_edge(proxy_edge_src, proxy_edge_dst):
            proxy_graph[proxy_edge_src][proxy_edge_dst]['capacity'] += vol
        else:
            proxy_graph.add_node(proxy_edge_src)
            proxy_graph.add_node(proxy_edge_dst)
            proxy_graph.add_edge(proxy_edge_src, proxy_edge_dst, capacity=vol)

    for node in scope_nodes:
        if isinstance(node, nd.AccessNode) and node.data in state_input_configuration:
            if not proxy_graph.has_edge(source, node) and node.data in state.parent.arrays:
                vol = state.parent.arrays[node.data].total_size
                if isinstance(vol, sp.Expr):
                    vol = vol.subs(symbols_map)
                proxy_graph.add_edge(source, node, capacity=vol)

    _, (_, non_reachable) = nx.minimum_cut(proxy_graph,
                                                 source,
                                                 sink,
                                                 flow_func=edmondskarp.edmonds_karp)

    non_reachable -= {sink}
    if len(non_reachable) > 0:
        subscope_expansions = set()
        for n in non_reachable:
            if isinstance(n, nd.EntryNode):
                subscope_expansions.update(transitive_scope_children[n])
            elif isinstance(n, nd.ExitNode):
                subscope_expansions.update(transitive_scope_children[state.entry_node(n)])
        return subgraph_nodes.union(non_reachable.union(subscope_expansions))
    return subgraph_nodes

def _stateset_predecessor_frontier(states: Set[SDFGState]) -> Tuple[Set[SDFGState], Set[Edge[InterstateEdge]]]:
    """
    For a set of states, return their predecessor frontier.
    The predecessor frontier refers to the predecessor states leading into any of the states in the given set.

    For example, if the given set is {C, D}, and the graph is induced by the edges
    {(A, B), (B, C), (C, D), (D, E), (B, D), (A, E), (A, C)}, then the predecessor frontier consists of the states
    {A, B} and edges {(A, C), (B, C), (B, D)}.

    :param states: The set of states to find the predecessor frontier of.
    :return: A tuple of the predecessor frontier states and the predecessor frontier edges.
    """
    pred_frontier = set()
    pred_frontier_edges = set()
    for state in states:
        for iedge in state.parent.in_edges(state):
            if iedge.src not in states:
                if iedge.src not in pred_frontier:
                    pred_frontier.add(iedge.src)
                if iedge not in pred_frontier_edges:
                    pred_frontier_edges.add(iedge)
    return pred_frontier, pred_frontier_edges


def _create_alibi_access_node_for_edge(target_sdfg: SDFG, target_state: SDFGState, original_sdfg: SDFG,
                                       original_edge: MultiConnectorEdge[Memlet], from_node: Union[nd.Node, None],
                                       from_connector: Union[str, None], to_node: Union[nd.Node, None],
                                       to_connector: Union[str, None]) -> Tuple[data.Data, nd.AccessNode]:
    """
    Add an alibi data container and access node to a dangling connector inside of scopes.
    Alibi nodes are never transient because they always represent a 'border' of the cutout and will consequently
    be accessed by other nodes outside of the cutout (i.e. at a minimum the data containers they are extracted from).
    """
    original_edge.data
    access_size = original_edge.data.subset.size_exact()
    container_name = '__cutout_' + str(original_edge.data.data)
    container_name = data.find_new_name(container_name, target_sdfg._arrays.keys())
    original_array = original_sdfg._arrays[original_edge.data.data]
    memlet_str = ''
    if original_edge.data.subset.num_elements_exact() > 1:
        access_size = original_edge.data.subset.size_exact()
        target_sdfg.add_array(container_name, access_size, original_array.dtype)
        memlet_str = container_name + '['
        sep = None
        for dim_len in original_edge.data.subset.bounding_box_size():
            if sep is not None:
                memlet_str += ','
            if dim_len > 1:
                memlet_str += '0:' + str(dim_len - 1)
            else:
                memlet_str += '0'
            sep = ','
        memlet_str += ']'
    else:
        target_sdfg.add_scalar(container_name, original_array.dtype)
        memlet_str = container_name + '[0]'
    alibi_access_node = target_state.add_access(container_name)
    if from_node is None:
        target_state.add_edge(alibi_access_node, None, to_node, to_connector, Memlet(memlet_str))
    else:
        target_state.add_edge(from_node, from_connector, alibi_access_node, None, Memlet(memlet_str))
    return target_sdfg.arrays[container_name], alibi_access_node


def _extend_subgraph_with_access_nodes(state: SDFGState, subgraph: StateSubgraphView,
                                       use_alibi_nodes: bool) -> StateSubgraphView:
    """ Expands a subgraph view to include necessary input/output access nodes, using memlet paths. """
    sdfg = state.parent
    result: List[nd.Node] = copy.copy(subgraph.nodes())
    queue: Deque[nd.Node] = deque(subgraph.nodes())

    # Add all nodes in memlet paths
    while len(queue) > 0:
        node = queue.pop()
        if isinstance(node, nd.AccessNode):
            if isinstance(node.desc(sdfg), data.View):
                vnode = sdutil.get_view_node(state, node)
                result.append(vnode)
                queue.append(vnode)
            continue
        for e in state.in_edges(node):
            # Special case: IN_* connectors are not traversed further
            if isinstance(e.dst, (nd.EntryNode, nd.ExitNode)) and (e.dst_conn is None or e.dst_conn.startswith('IN_')):
                continue

            # We don't want to extend access nodes over scope entry nodes, but rather we want to introduce alibi data
            # containers for the correct subset instead. Handled separately in _create_alibi_access_node_for_edge.
            if use_alibi_nodes:
                if isinstance(e.src, nd.EntryNode) and e.src not in result and state.exit_node(e.src) not in result:
                    continue

            mpath = state.memlet_path(e)
            new_nodes = [mpe.src for mpe in mpath if mpe.src not in result]
            result.extend(new_nodes)
            # Memlet path may end in a code node, continue traversing and expanding graph
            queue.extend(new_nodes)

        for e in state.out_edges(node):
            # Special case: OUT_* connectors are not traversed further
            if isinstance(e.src, (nd.EntryNode, nd.ExitNode)) and (e.src_conn is None or e.src_conn.startswith('OUT_')):
                continue

            # We don't want to extend access nodes over scope exit nodes, but rather we want to introduce alibi data
            # containers for the correct subset instead. Handled separately in _create_alibi_access_node_for_edge.
            if use_alibi_nodes:
                if isinstance(e.dst, nd.ExitNode) and e.dst not in result and state.entry_node(e.dst) not in result:
                    continue

            mpath = state.memlet_path(e)
            new_nodes = [mpe.dst for mpe in mpath if mpe.dst not in result]
            result.extend(new_nodes)
            # Memlet path may end in a code node, continue traversing and expanding graph
            queue.extend(new_nodes)

    # Check for mismatch in scopes
    for node in result:
        enode = None
        if isinstance(node, nd.EntryNode) and state.exit_node(node) not in result:
            enode = state.exit_node(node)
        if isinstance(node, nd.ExitNode) and state.entry_node(node) not in result:
            enode = state.entry_node(node)
        if enode is not None:
            raise ValueError(f'Cutout cannot expand graph implicitly since "{node}" is in the graph and "{enode}" is '
                             'not. Please provide more nodes in the subgraph as necessary.')

    return StateSubgraphView(state, result)


def _determine_cutout_reachability(
        ct: SDFG,
        sdfg: SDFG,
        in_translation: Dict[Any, Any],
        out_translation: Dict[Any, Any],
        state_reach: Dict[SDFGState, Set[SDFGState]] = None) -> Tuple[Set[SDFGState], Set[SDFGState]]:
    """
    For a given cutout and its original SDFG, determine what parts of the SDFG (set of states) can reach the cutout,
    and what set of states can be reached from the cutout.

    :param ct: The cutout SDFG.
    :param sdfg: The original SDFG.
    :param in_translation: The translation dictionary from the original SDFG elements to cutout elements.
    :param out_translation: The translation dictionary from the cutout elements to original SDFG elements.
    :param state_reach: The state reachability dictionary for the original SDFG. If not provided, it will be computed
                        on-the-fly.
    :return: A tuple of two sets of states. The first set contains the states that can reach the cutout, and the second
             set contains the states that can be reached from the cutout.
    """
    if state_reach is None:
        original_sdfg_id = out_translation[ct.sdfg_id]
        state_reachability_dict = StateReachability().apply_pass(sdfg.sdfg_list[original_sdfg_id], None)
        state_reach = state_reachability_dict[original_sdfg_id]
    inverse_cutout_reach: Set[SDFGState] = set()
    cutout_reach: Set[SDFGState] = set()
    cutout_states = set(ct.states())
    for state in cutout_states:
        original_state = out_translation[state]
        for k, v in state_reach.items():
            if (k not in in_translation or in_translation[k] not in cutout_states):
                if original_state is not None and original_state in v:
                    inverse_cutout_reach.add(k)
        for rstate in state_reach[original_state]:
            if (rstate not in in_translation or in_translation[rstate] not in cutout_states):
                cutout_reach.add(rstate)
    return (inverse_cutout_reach, cutout_reach)


def _cutout_determine_input_config(ct: SDFG, inverse_cutout_reach: Set[SDFGState], in_translation: Dict[Any, Any],
                                   out_translation: Dict[Any, Any]) -> Set[str]:
    """
    Determines the input configuration for a given cutout SDFG.
    The input configuration is the set of data descriptors that are read inside the cutout, but may be written to
    before the cutout is executed. Consequently, this contains any data container that may influence the computations
    inside the cutout.

    :param ct: The cutout SDFG.
    :param inverse_cutout_reach: The set of states that can reach the cutout.
    :param in_translation: The translation dictionary from the original SDFG elements to cutout elements.
    :param out_translation: The translation dictionary from the cutout elements to original SDFG elements.
    :return: The set of data descriptor names that are part of the input configuration.
    """
    input_configuration = set()
    check_for_write_before = set()
    cutout_states = set(ct.states())

    noded_descriptors = set()

    for state in cutout_states:
        for dn in state.data_nodes():
            noded_descriptors.add(dn.data)
            if state.out_degree(dn) > 0:
                # This is read from, add to the system state if it is written anywhere else in the graph.
                # Except if it is also written to at the same time and is scalar or of size 1.
                array = ct.arrays[dn.data]
                if state.in_degree(dn) > 0 and (array.total_size == 1 or isinstance(array, data.Scalar)):
                    continue
                elif not array.transient:
                    # Non-transients are always part of the input config if they are read and not overwritten anyway.
                    input_configuration.add(dn.data)
                else:
                    check_for_write_before.add(dn.data)

        original_state: Optional[SDFGState] = None
        try:
            original_state = out_translation[state]
        except KeyError:
            original_state = None

        # If the cutout consists of only one state, we need to check inside the same state of the original SDFG as well.
        if len(cutout_states) == 1 and original_state is not None:
            for dn in original_state.data_nodes():
                if original_state.in_degree(dn) > 0:
                    iedges = original_state.in_edges(dn)
                    if any([i.src not in in_translation for i in iedges]):
                        if dn.data in check_for_write_before:
                            input_configuration.add(dn.data)

    for state in inverse_cutout_reach:
        for dn in state.data_nodes():
            if state.in_degree(dn) > 0:
                # For any writes, check if they are reads from the cutout that need to be checked. If they are, they're
                # part of the system state.
                if dn.data in check_for_write_before:
                    input_configuration.add(dn.data)

    # Anything that doesn't have a correpsonding access node must be used as well.
    for desc in ct.arrays.keys():
        if desc not in noded_descriptors:
            input_configuration.add(desc)

    return input_configuration


def _cutout_determine_output_configuration(ct: SDFG, cutout_reach: Set[SDFGState], in_translation: Dict[Any, Any],
                                           out_translation: Dict[Any, Any]) -> Set[str]:
    """
    Determines the output configuration for a given cutout SDFG.
    The output configuration is the set of data descriptors that are written inside the cutout, but may be read from
    after the cutout is executed. Consequently, this contains any data container that may influence the computations
    of the original SDFG after the cutout.

    :param ct: The cutout SDFG.
    :param cutout_reach: The set of states that can be reached from the cutout.
    :param in_translation: The translation dictionary from the original SDFG elements to cutout elements.
    :param out_translation: The translation dictionary from the cutout elements to original SDFG elements.
    :return: The set of data descriptor names that are part of the output configuration.
    """
    system_state = set()
    check_for_read_after = set()
    cutout_states = set(ct.states())
    border_out_edges: Set[InterstateEdge] = set()

    for state in cutout_states:
        for dn in state.data_nodes():
            array = ct.arrays[dn.data]
            if not array.transient:
                # Non-transients are always part of the system state.
                system_state.add(dn.data)
            elif state.in_degree(dn) > 0:
                # This is written to, add to the system state if it is read anywhere else in the graph.
                check_for_read_after.add(dn.data)

        original_state: SDFGState = out_translation[state]
        for edge in original_state.parent.out_edges(original_state):
            if edge.dst in cutout_reach:
                border_out_edges.add(edge.data)

        # If the cutout consists of only one state, we need to check inside the same state of the original SDFG as well.
        if len(cutout_states) == 1:
            for dn in original_state.data_nodes():
                if original_state.out_degree(dn) > 0:
                    oedges = original_state.out_edges(dn)
                    if any([o.dst not in in_translation for o in oedges]):
                        if dn.data in check_for_read_after:
                            system_state.add(dn.data)

    for state in cutout_reach:
        for dn in state.data_nodes():
            if state.out_degree(dn) > 0:
                # For any reads, check if they are writes from the cutout that need to be checked. If they are, they're
                # part of the system state.
                if dn.data in check_for_read_after:
                    system_state.add(dn.data)

    return system_state


def _recursively_set_nsdfg_parents(target: SDFG):
    for state in target.states():
        for n in state.nodes():
            if isinstance(n, nd.NestedSDFG):
                n.sdfg.parent_sdfg = target
                _recursively_set_nsdfg_parents(n.sdfg)
