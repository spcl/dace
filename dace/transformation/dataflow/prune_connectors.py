# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Any, AnyStr, Dict, Set, Tuple, Union
import itertools

from dace import memlet, registry, SDFG, SDFGState
from dace.transformation import transformation as pm, helpers
from dace.sdfg import nodes, utils


def connectors_to_prune(
        nsdfg: nodes.NestedSDFG) -> Tuple[Set[AnyStr], Set[AnyStr]]:
    """ Determines the sets of input and output connectors that can be safely
        pruned from a nested SDFG.
        :param nsdfg: Nested SDFG to prune.
        :return: A two-tuple of the input connectors and output connectors that
                 can be pruned.
    """

    read_set = set()
    write_set = set()

    for state in nsdfg.sdfg.states():
        rs, ws = helpers.read_and_write_sets(state)
        read_set |= rs
        write_set |= ws

    return (nsdfg.in_connectors.keys() - read_set,
            nsdfg.out_connectors.keys() - write_set)


@registry.autoregister_params(singlestate=True, strict=True)
class PruneConnectors(pm.Transformation):
    """ Removes unused connectors from nested SDFGs, as well as their memlets
        in the outer scope, replacing them with empty memlets if necessary.
    """

    nsdfg = pm.PatternNode(nodes.NestedSDFG)

    @staticmethod
    def expressions():
        return [utils.node_path_graph(PruneConnectors.nsdfg)]

    @staticmethod
    def can_be_applied(graph: Union[SDFG, SDFGState],
                       candidate: Dict['PatternNode', int],
                       expr_index: int,
                       sdfg: SDFG,
                       strict: bool = False) -> bool:

        nsdfg = graph.node(candidate[PruneConnectors.nsdfg])

        prune_in, prune_out = connectors_to_prune(nsdfg)

        if len(prune_in) > 0 or len(prune_out) > 0:
            return True

        return False

    def apply(self, sdfg: SDFG) -> Union[Any, None]:

        state = sdfg.node(self.state_id)
        nsdfg = self.nsdfg(sdfg)

        prune_in, prune_out = connectors_to_prune(nsdfg)

        # Detect which nodes are used, so we can delete unused nodes after the
        # connectors have been pruned
        all_data_used = set()
        for s in nsdfg.sdfg.states():
            for n in s.data_nodes():
                all_data_used.add(n.data)

        for conn in itertools.chain(prune_in, prune_out):
            for e in state.edges_by_connector(nsdfg, conn):
                state.remove_memlet_path(e, remove_orphans=True)
                if conn in nsdfg.sdfg.arrays and conn not in all_data_used:
                    # If the data is now unused, we can purge it from the SDFG
                    nsdfg.sdfg.remove_data(conn)
