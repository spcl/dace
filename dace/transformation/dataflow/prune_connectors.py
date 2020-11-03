# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Any, AnyStr, Dict, Set, Tuple, Union
import itertools

from dace import memlet, registry, SDFG, SDFGState
from dace.transformation import transformation as pm, helpers
from dace.sdfg import nodes, utils


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

        read_set, write_set = nsdfg.sdfg.read_and_write_sets()
        prune_in = nsdfg.in_connectors.keys() - read_set
        prune_out = nsdfg.out_connectors.keys() - write_set

        if len(prune_in) > 0 or len(prune_out) > 0:
            return True

        return False

    def apply(self, sdfg: SDFG) -> Union[Any, None]:

        state = sdfg.node(self.state_id)
        nsdfg = self.nsdfg(sdfg)

        read_set, write_set = nsdfg.sdfg.read_and_write_sets()
        prune_in = nsdfg.in_connectors.keys() - read_set
        prune_out = nsdfg.out_connectors.keys() - write_set

        # Detect which nodes are used, so we can delete unused nodes after the
        # connectors have been pruned
        all_data_used = read_set | write_set

        for conn in prune_in:
            for e in state.in_edges_by_connector(nsdfg, conn):
                state.remove_memlet_path(e, remove_orphans=True)
                if conn in nsdfg.sdfg.arrays and conn not in all_data_used:
                    # If the data is now unused, we can purge it from the SDFG
                    nsdfg.sdfg.remove_data(conn)

        for conn in prune_out:
            for e in state.out_edges_by_connector(nsdfg, conn):
                state.remove_memlet_path(e, remove_orphans=True)
                if conn in nsdfg.sdfg.arrays and conn not in all_data_used:
                    # If the data is now unused, we can purge it from the SDFG
                    nsdfg.sdfg.remove_data(conn)
