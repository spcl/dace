# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" polyhedral loop transformation """

from dace import Memlet
from dace.subsets import Range
from dace.sdfg import nodes
from poly_builder import Polytope
from dace.transformation.dataflow.map_fusion import MapFusion


def ranges_to_polytopes(sdfg):
    sdfg.apply_strict_transformations()
    sdfg.apply_transformations_repeated(MapFusion, validate=True)

    def replace_memlets(sdfg):
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, nodes.NestedSDFG):
                    replace_memlets(node.sdfg)
                elif isinstance(node, nodes.CodeNode) or \
                        isinstance(node, nodes.EntryNode) or \
                        isinstance(node, nodes.ExitNode):
                    for e in state.out_edges(node):
                        if isinstance(e.data.subset, Range):
                            label = e.data.data
                            subset = e.data.subset
                            new_range = Polytope(subset)
                            e.data = Memlet(data=label, subset=new_range)
                    for e in state.in_edges(node):
                        if isinstance(e.data.subset, Range):
                            label = e.data.data
                            subset = e.data.subset
                            new_range = Polytope(subset)
                            e.data = Memlet(data=label, subset=new_range)
                    if isinstance(node, nodes.EntryNode):
                        node.range = Polytope(node.range, node.params)

    replace_memlets(sdfg)
    return sdfg