import copy
from typing import Dict, List, Union, Optional

from dace import registry, sdfg, properties, data
from dace.codegen.targets import cpp
from dace.sdfg import nodes, SDFG, SDFGState, graph
from dace.sdfg import utils as sdutil
from dace.transformation import transformation, helpers

from dace.util import find_str_not_in_set


def _invert_map_connector(conn):
    if conn.startswith("IN"):
        return "OUT" + conn[2:]
    elif conn.startswith("OUT"):
        return "IN" + conn[3:]
    else:
        raise ValueError("Could not parse map connector '{}'".format(conn))


def in_edges(
        state: SDFGState, node: nodes.ExitNode,
        exit_edge: graph.MultiConnectorEdge) -> List[graph.MultiConnectorEdge]:
    inverse_connector = _invert_map_connector(exit_edge.src_conn)
    return list(state.in_edges_by_connector(node, inverse_connector))


def _edges_between_ok(
        sdfg: SDFG, init_state: SDFGState,
        compute_state: SDFGState) -> Optional[graph.Edge[sdfg.InterstateEdge]]:
    """
    Conservative check that there is only one basic edge between the init_state
    and compute_state.
    :param init_state: the init state.
    :param compute_state: the compute state.
    :return: None if there are weird interstate edges, or the single interstate
             edge if it's ok.
    """

    out_edges = sdfg.out_edges(init_state)
    in_edges = sdfg.in_edges(init_state)

    # First state must have only one output edge (with dst the second
    # state).
    if len(out_edges) != 1:
        return None
    # The interstate edge must not have a condition.
    if not out_edges[0].data.is_unconditional():
        return None
    # The interstate edge may not have assignments.
    if out_edges[0].data.assignments:
        return None

    # There can be no state that have output edges pointing to both the
    # first and the second state.
    for src, _, _ in in_edges:
        for _, dst, _ in sdfg.out_edges(src):
            if dst == compute_state:
                return None

    return out_edges[0]


@properties.make_properties
class InitStateFusion(transformation.MultiStateTransformation):

    init_state = transformation.PatternNode(SDFGState)
    compute_state = transformation.PatternNode(SDFGState)

    accumulate_transient = properties.Property(
        dtype=bool,
        default=True,
        desc='Make the init state and compute state write to a local variable'
        ' that is later copied out to the global.')

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(InitStateFusion.init_state,
                                   InitStateFusion.compute_state)
        ]

    def can_be_applied(self,
                       graph: Union[SDFG, SDFGState],
                       expr_index: int,
                       sdfg: SDFG,
                       permissive: bool = False) -> bool:

        init_state: SDFGState = self.init_state
        compute_state: SDFGState = self.compute_state

        if not _edges_between_ok(sdfg, init_state, compute_state):
            return False

        # both states should only contain a single map
        init_maps: List[nodes.MapEntry] = [
            n for n in init_state.scope_children()[None]
            if isinstance(n, nodes.MapEntry)
        ]
        if len(init_maps) != 1:
            return False

        compute_maps: List[nodes.MapEntry] = [
            n for n in compute_state.scope_children()[None]
            if isinstance(n, nodes.MapEntry)
        ]
        if len(compute_maps) != 1:
            return False

        init_map = init_state.exit_node(init_maps[0])
        compute_map = compute_state.exit_node(compute_maps[0])

        # for each node that is written in the init state, there may be no self-
        # intersecting writes to that node (intersecting in terms of the map)
        for n in init_state.sink_nodes():
            # this should be connected to the map
            if not init_state.edges_between(init_map, n):
                return False

            # check that none of the edges on the tree are conflicted w.r.t the
            # outer map
            for map_edge in init_state.edges_between(init_map, n):
                for map_in_edge in in_edges(init_state, init_map, map_edge):
                    if not cpp._check_map_conflicts(init_map.map, map_in_edge):
                        return False

            # do the same thing for the edge that writes to n in compute_state
            writes_to_n = [
                w for w in compute_state.out_edges(compute_map)
                if w.data.data == n.data
            ]
            for compute_write in writes_to_n:
                for map_in_edge in in_edges(compute_state, compute_map,
                                            compute_write):
                    if not cpp._check_map_conflicts(compute_map.map,
                                                    map_in_edge):
                        return False
        return True

    def apply(self, _, sdfg: SDFG):
        init_state: SDFGState = self.init_state
        compute_state: SDFGState = self.compute_state

        # both states should only contain a single map
        init_maps: List[nodes.MapEntry] = [
            n for n in init_state.scope_children()[None]
            if isinstance(n, nodes.MapEntry)
        ]

        compute_maps: List[nodes.MapEntry] = [
            n for n in compute_state.scope_children()[None]
            if isinstance(n, nodes.MapEntry)
        ]

        init_map = init_maps[0]
        compute_map = compute_maps[0]

        # nest both map scope subgraphs
        init_nsdfg = helpers.nest_state_subgraph(
            sdfg, init_state,
            init_state.scope_subgraph(init_map,
                                      include_entry=False,
                                      include_exit=False))
        compute_nsdfg = helpers.nest_state_subgraph(
            sdfg, compute_state,
            compute_state.scope_subgraph(compute_map,
                                         include_entry=False,
                                         include_exit=False))

        # move the init state into the compute nsdfg
        assert len(init_nsdfg.sdfg.nodes()) == 1
        assert len(compute_nsdfg.sdfg.nodes()) == 1
        init_nested_state = init_nsdfg.sdfg.node(0)

        compute_nested_state = compute_nsdfg.sdfg.node(0)
        compute_nsdfg.sdfg.add_node(init_nested_state)
        init_nested_state.parent = compute_nsdfg.sdfg

        interstate_edge = _edges_between_ok(sdfg, init_state, compute_state)
        compute_nsdfg.sdfg.add_edge(init_nested_state, compute_nested_state,
                                    interstate_edge.data)

        # find a new name for the state in case it conflicts
        existing_state_names = {
            n.label
            for n, _ in compute_nsdfg.sdfg.all_nodes_recursive()
            if isinstance(n, SDFGState)
        }

        new_name = find_str_not_in_set(existing_state_names,
                                       init_nested_state.label)
        # modify state label 
        init_nested_state._label = new_name

        # remove the init state from the top-level sdfg
        sdfg.remove_node(init_state)
        nsdfg = compute_nsdfg.sdfg

        if self.accumulate_transient:
            # insert a copy out of the transient
            copy_out_state = nsdfg.add_state_after(compute_nested_state)
            for n in init_nested_state.sink_nodes():
                # make the descs that both states are writing to transient
                new_desc = copy.deepcopy(nsdfg.arrays[n.data])

                nsdfg.arrays[n.data].transient = True
                # we need to update the total size and strides now that it's smaller
                shape = nsdfg.arrays[n.data].shape
                nsdfg.arrays[n.data].total_size = data._prod(shape)
                nsdfg.arrays[n.data].strides = tuple(
                    data._prod(shape[i + 1:]) for i, _ in enumerate(shape))

                # add a non-transient that we will copy out to
                new_desc.transient = False
                new_name = nsdfg.add_datadesc(n.data + "_global",
                                              new_desc,
                                              find_new_name=True)

                # copy out the transients
                copy_out_state.add_edge(copy_out_state.add_read(n.data), None,
                                        copy_out_state.add_write(new_name),
                                        None,
                                        nsdfg.make_array_memlet(new_name))

                # rename the nsdfg connector to the new non-transient
                compute_nsdfg.remove_out_connector(n.data)
                compute_nsdfg.add_out_connector(new_name)

                for out_edge in compute_state.out_edges_by_connector(
                        compute_nsdfg, n.data):
                    out_edge.src_conn = new_name
