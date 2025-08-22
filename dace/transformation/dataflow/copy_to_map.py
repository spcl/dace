# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

from dace import dtypes, symbolic, data, subsets, Memlet, properties
from dace.transformation import transformation as xf
from dace.sdfg import SDFGState, SDFG, nodes, utils as sdutil, memlet_utils as mutils


@properties.make_properties
class CopyToMap(xf.SingleStateTransformation):
    """
    Converts an access node -> access node copy into a map. Useful for generating manual code and
    controlling schedules for N-dimensional strided copies.

    Note that the transformation will turn _all_ suitable edges into Maps. For more information
    see `dace.sdfg.memlet_utils.can_memlet_be_turned_into_a_map()` and
    `dace.sdfg.memlet_utils.memlet_to_map()`.
    """
    a = xf.PatternNode(nodes.AccessNode)
    b = xf.PatternNode(nodes.AccessNode)
    ignore_strides = properties.Property(
        default=False,
        desc='Ignore the stride of the data container; Defaults to `False`.',
    )

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.a, cls.b)]

    def can_be_applied(self, graph: SDFGState, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:
        a_b_edges = graph.edges_between(self.a, self.b)
        if len(a_b_edges) == 0:
            return False
        # NOTE: Technically it would be enough to check only one edge, but this would
        #   impose a restriction on `can_memlet_be_turned_into_a_map()`.
        if not any(
                mutils.can_memlet_be_turned_into_a_map(
                    edge=a_b_edge, state=graph, sdfg=sdfg, ignore_strides=self.ignore_strides)
                for a_b_edge in a_b_edges):
            return False

        return True

    def apply(self, state: SDFGState, sdfg: SDFG):
        a_b_edges = list(state.edges_between(self.a, self.b))
        for a_b_edge in a_b_edges:
            if mutils.can_memlet_be_turned_into_a_map(edge=a_b_edge,
                                                      state=state,
                                                      sdfg=sdfg,
                                                      ignore_strides=self.ignore_strides):
                _ = mutils.memlet_to_map(
                    edge=a_b_edge,
                    state=state,
                    sdfg=sdfg,
                    ignore_strides=self.ignore_strides,
                )
