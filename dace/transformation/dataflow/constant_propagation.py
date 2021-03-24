from copy import deepcopy as dcpy
from dace import registry, symbolic, subsets
from dace.sdfg import nodes
from dace.memlet import Memlet
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from typing import List, Union
import networkx as nx
from dace.sdfg import state as dace_state
from dace.sdfg import sdfg as dace_sdfg
from dace import memlet
from dace.sdfg import graph as dace_graph
from dace.transformation.helpers import nest_state_subgraph
from dace.sdfg.graph import SubgraphView
from dace.sdfg.replace import replace_properties
from dace import dtypes
from dace import data as dace_data
import itertools
from dace.transformation.dataflow.clean_connectors import find_read_write_states
from typing import Dict, Set, Tuple


def detect_read_data_dependencies(sdfg: dace_sdfg.SDFG):
    # map from data descriptor strings to
    # maps from read state to the set of potential write states that produce value for read
    reads: Dict[str, Dict[dace_state.SDFGState, Set[dace_state.SDFGState]]] = {}

    source_states = sdfg.source_nodes()
    assert len(source_states) == 1
    source_state = source_states[0]

    sink_states = sdfg.sink_nodes()

    for data_desc in sdfg.arrays:
        data_desc: str

        reads[data_desc] = {}

        read_states, write_states = find_read_write_states(sdfg, data_desc)

        for rs in read_states:
            reads[data_desc][rs] = set()
        reads[data_desc][None] = set()

        for write_state in write_states:
            def src_not_writes(src, dst, edge):
                if src == write_state:
                    return True # src is allowed to write only if it is write state
                return src not in write_states

            for e in sdfg.dfs_edges(source=write_state, condition=src_not_writes):
                if e.dst in read_states:
                    reads[data_desc][e.dst].add(write_state)

                # special case: write propagates outside from nested sdfg
                if (e.dst in sink_states) and (e.dst not in write_states) and (not sdfg.arrays[data_desc].transient):
                    reads[data_desc][None].add(write_state)

            # special case: write propagates outside from nested sdfg
            if (write_state in sink_states) and (not sdfg.arrays[data_desc].transient):
                reads[data_desc][None].add(write_state)

        # special case: source state uses value from input to sdfg
        if (source_state in read_states) and (not sdfg.arrays[data_desc].transient):
            reads[data_desc][source_state].add(None)

        def src_really_not_writes(src, dst, edge):
            return src not in write_states

        # special case: other states than source state use value from SDFG input
        for e in sdfg.dfs_edges(source=source_state, condition=src_really_not_writes):
            if (e.dst in read_states) and (not sdfg.arrays[data_desc].transient):
                reads[data_desc][e.dst].add(None) # None means that write happens outside

    return reads


def detect_data_dependencies(sdfg: dace_sdfg.SDFG) -> Tuple[
    Dict[str, Dict[dace_state.SDFGState, Set[dace_state.SDFGState]]],
    Dict[str, Dict[dace_state.SDFGState, Set[dace_state.SDFGState]]]]:
    """
    Returns two maps (read_deps, write_deps): one with read dependencies and the other with write dependencies.

    read_deps: array name -> read_deps_map
    read_deps_map: state where read happens -> states that may write this value
    write_deps: array name -> write_deps_map
    write_deps_map: state where write happens -> states that may read this value

    read_deps_map, write_deps_map can contain None value instead of states both in keys and values.
    They represent when read or write goes from/to of current sdfg.
    """

    read_deps = detect_read_data_dependencies(sdfg)
    write_deps: Dict[str, Dict[dace_state.SDFGState, Set[dace_state.SDFGState]]] = {}

    for data_desc, rd in read_deps.items():
        write_deps[data_desc] = {}

        # initialize sets
        for read, writes in rd.items():
            for write in writes:
                if write not in write_deps[data_desc]:
                    write_deps[data_desc][write] = set()

        for read, writes in rd.items():
            for write in writes:
                write_deps[data_desc][write].add(read)

    return read_deps, write_deps


def nest_if_not_nested(sdfg: dace_sdfg.SDFG, state: dace_state.SDFGState):
    """
    puts state content inside nested sdfg if it is not already nested
    returns nested sdfg
    """
    for node in state.nodes():
        if isinstance(node, nodes.NestedSDFG):
            return node
        if isinstance(node, nodes.AccessNode):
            continue
        if isinstance(node, nodes.MapEntry) or isinstance(node, nodes.MapExit):
            continue
        return nest_state_subgraph(sdfg, state, SubgraphView(state, [node]))


def remove_state_content(state: dace_state.SDFGState):
    state_nodes = state.nodes()
    for node in state_nodes:
        state.remove_node(node)


@registry.autoregister_params(singlestate=True)
class ConstantPropagation(transformation.Transformation):
    """
    Detects "constant writing state" that has only a single tasklet that writes to a single access node.
    Such write is propagated to the subsequent states.
    """

    tasklet = transformation.PatternNode(nodes.Tasklet)
    access_node = transformation.PatternNode(nodes.AccessNode)

    @staticmethod
    def annotates_memlets():
        return False

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(
                ConstantPropagation.tasklet,
                ConstantPropagation.access_node,
            )
        ]

    @staticmethod
    def can_be_applied(state: dace_state.SDFGState, candidate, expr_index, sdfg, strict=False):
        tasklet: nodes.Tasklet = state.nodes()[candidate[ConstantPropagation.tasklet]]
        access_node: nodes.AccessNode = state.nodes()[candidate[ConstantPropagation.access_node]]

        original_edge: dace_graph.MultiConnectorEdge = state.edges_between(tasklet, access_node)[0]

        data_desc = access_node.data

        if len(state.nodes()) != 2:
            # we support only this small pattern in the whole state
            return False

        # check that we actually can propagate constant somewhere
        read_deps, write_deps = detect_data_dependencies(sdfg)

        read_states = write_deps[data_desc][state]

        # remove states to which this transformation will not be applied:
        # output implicit dependency and all states without nested sdfgs
        if None in read_states:
            read_states.remove(None)

        skip_states = set()

        for s in read_states:
            skip_states.add(s)
            for n in s.nodes():
                if isinstance(n, nodes.NestedSDFG):
                    skip_states.remove(s)

        read_states -= skip_states

        # if there are some states left, then this transformatin can be applied
        if read_states:
            return True

        return False

    def apply(self, sdfg: dace_sdfg.SDFG):

        state: dace_state.SDFGState = sdfg.nodes()[self.state_id]
        candidate = self.subgraph
        tasklet: nodes.Tasklet = state.nodes()[candidate[ConstantPropagation.tasklet]]
        access_node: nodes.AccessNode = state.nodes()[candidate[ConstantPropagation.access_node]]

        original_edge: dace_graph.MultiConnectorEdge = state.edges_between(tasklet, access_node)[0]

        data_desc = access_node.data
        read_deps, write_deps = detect_data_dependencies(sdfg)

        read_states = write_deps[data_desc][state]

        is_used_outside = False

        skip_states = set()

        for s in read_states:
            skip_states.add(s)
            for n in s.nodes():
                if isinstance(n, nodes.NestedSDFG):
                    skip_states.remove(s)

        if None in read_states:
            read_states.remove(None)
            is_used_outside = True

        if skip_states:
            read_states -= skip_states
            is_used_outside = True

        for read_state in read_states:

            nsdfg: nodes.NestedSDFG = nest_if_not_nested(sdfg, read_state)

            # find input corresponding to the read and remove it with connector
            connector_name = None

            for e in read_state.in_edges(nsdfg):
                an: nodes.AccessNode = e.src
                if an.data != data_desc:
                    continue

                # remove access node
                connector_name = e.dst_conn
                read_state.remove_node(an)

            if connector_name:
                # remove connector
                nsdfg.remove_in_connector(connector_name)
            else:
                for e in read_state.out_edges(nsdfg):
                    an: nodes.AccessNode = e.dst
                    if an.data != data_desc:
                        continue

                    # this edge has WCR, this is why it is detected as one of the reads for this variable
                    assert e.data.wcr

                    connector_name = e.src_conn

                    # after transformation WCR is not requires since Array is written only
                    e.data.wcr = None

            # make data object inside nested sdfg transient if it is not written
            if connector_name not in nsdfg.out_connectors:
                nsdfg.sdfg.arrays[connector_name].transient = True

            # add tasklet inside nested sdfg that initializes data object
            new_state: dace_state.SDFGState = nsdfg.sdfg.add_state_before(nsdfg.sdfg.start_state, is_start_state=True)

            new_tasklet: nodes.Tasklet = new_state.add_tasklet(
                name=tasklet.name,
                inputs=tasklet.in_connectors,
                outputs=tasklet.out_connectors,
                code=tasklet.code,
                language=tasklet.language,
            )

            new_access_node: nodes.AccessNode = new_state.add_access(connector_name)

            new_state.add_edge(new_tasklet, original_edge.src_conn, new_access_node, None,
                               memlet.Memlet(data=connector_name, subset=original_edge.data.subset))

        if not is_used_outside:
            # we can safely remove initial state
            # we will remove state content becase we don't want to worry about state transition changes here
            remove_state_content(state)