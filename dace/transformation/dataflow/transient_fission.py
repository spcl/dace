# Copyright 2022-2024 ETH Zurich and the Daisytuner authors.
import copy
import dace

from dace.sdfg import nodes as nds
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import SDFGState
from dace.transformation import transformation
from dace.sdfg import utils as sdutil


class TransientFission(transformation.SingleStateTransformation):

    nested_sdfg = transformation.PatternNode(nds.NestedSDFG)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.nested_sdfg)]

    def can_be_applied(
        self,
        graph: dace.SDFGState,
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        nested_sdfg = self.nested_sdfg
        nsdfg = nested_sdfg.sdfg
        if len(nsdfg.states()) != 1:
            return False

        nstate = nested_sdfg.sdfg.start_state

        # Search for outputs which are used as intermediate storage
        potential_dnodes = []
        for dnode in nstate.data_nodes():
            if not dnode.data in nested_sdfg.out_connectors:
                continue

            if nstate.out_degree(dnode) == 0:
                continue

            if nstate.in_degree(dnode) != 1:
                continue

            if any(
                [
                    edge.data is None or edge.data.num_elements() > 1
                    for edge in nstate.out_edges(dnode)
                ]
            ):
                continue

            # Written to but reused
            potential_dnodes.append(dnode)

        # For now, only consider arrays which are used exactly once
        potential_arrays = list(map(lambda n: n.data, potential_dnodes))
        potential_dnodes = list(
            filter(lambda n: potential_arrays.count(n.data) == 1, potential_dnodes)
        )
        if not potential_dnodes:
            return False

        return True

    def apply(self, graph: SDFGState, sdfg: SDFG):
        nested_sdfg = self.nested_sdfg
        nsdfg = nested_sdfg.sdfg
        nstate = nsdfg.states()[0]

        # Search for outputs which are used as intermediate storage
        potential_dnodes = []
        for dnode in nstate.data_nodes():
            if not dnode.data in nested_sdfg.out_connectors:
                continue

            if nstate.out_degree(dnode) == 0:
                continue

            if nstate.in_degree(dnode) != 1:
                continue

            if any(
                [
                    edge.data is None or edge.data.num_elements() > 1
                    for edge in nstate.out_edges(dnode)
                ]
            ):
                continue

            # Written to but reused
            potential_dnodes.append(dnode)

        # For now, only consider arrays which are used exactly once
        potential_arrays = list(map(lambda n: n.data, potential_dnodes))
        potential_dnodes = list(
            filter(lambda n: potential_arrays.count(n.data) == 1, potential_dnodes)
        )

        for dnode in potential_dnodes:
            desc = nsdfg.arrays[dnode.data]
            tmp, tmp_desc = nsdfg.add_scalar(
                "tmp", dtype=desc.dtype, transient=True, find_new_name=True
            )
            tmp_node = nstate.add_access(tmp)

            copy_memlet = None
            for in_edge in nstate.in_edges(dnode):
                nstate.add_edge(
                    in_edge.src,
                    in_edge.src_conn,
                    tmp_node,
                    None,
                    dace.Memlet.from_array(dataname=tmp, datadesc=tmp_desc),
                )
                nstate.remove_edge(in_edge)
                copy_memlet = copy.deepcopy(in_edge.data)

            copy_tasklet = nstate.add_tasklet(
                "copy", set(("_in",)), outputs=set(("_out",)), code="_out = _in"
            )
            memlet_in = dace.Memlet.from_array(dataname=tmp, datadesc=tmp_desc)
            nstate.add_edge(tmp_node, None, copy_tasklet, "_in", memlet_in)

            nstate.add_edge(copy_tasklet, "_out", dnode, None, copy_memlet)

            for out_edge in nstate.out_edges(dnode):
                nstate.add_edge(
                    tmp_node,
                    None,
                    out_edge.dst,
                    out_edge.dst_conn,
                    dace.Memlet.from_array(dataname=tmp, datadesc=tmp_desc),
                )
                nstate.remove_edge(out_edge)
