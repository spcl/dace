# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes that implement the map-expansion transformation. """

from dace import Memlet, subsets
from dace.sdfg import nodes
from poly_builder import Polytope
from dace.transformation.dataflow.map_expansion import MapExpansion
from sympy_isl_conversion import get_overapprox_range_list_from_set


def polytopes_to_ranges(sdfg):
    """
    convert all Polytope subsets in the SDFG to Range subsets
    """
    def replace_memlets(sdfg):
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, nodes.NestedSDFG):
                    replace_memlets(node.sdfg)
                elif isinstance(node, nodes.CodeNode) or \
                        isinstance(node, nodes.EntryNode) or \
                        isinstance(node, nodes.ExitNode):
                    for e in state.out_edges(node):
                        if isinstance(e.data.subset, Polytope):
                            label = e.data.data
                            subset = e.data.subset
                            bounding_ranges = get_overapprox_range_list_from_set(subset.isl_set.polyhedral_hull())
                            new_range = subsets.Range(bounding_ranges)
                            # new_range = subset.to_ranges()
                            e.data = Memlet(data=label, subset=new_range)
                    for e in state.in_edges(node):
                        if isinstance(e.data.subset, Polytope):
                            label = e.data.data
                            subset = e.data.subset
                            bounding_ranges = get_overapprox_range_list_from_set(subset.isl_set.polyhedral_hull())
                            new_range = subsets.Range(bounding_ranges)
                            # new_range = subset.to_ranges()
                            e.data = Memlet(data=label, subset=new_range)
                    if isinstance(node, nodes.EntryNode):
                        if node.map.get_param_num() == 1:
                            # new_range = node.range.to_ranges()
                            new_range = subsets.Range(node.range.ranges)
                            node.range = new_range

    replace_memlets(sdfg)
    sdfg.apply_transformations_repeated(MapExpansion, validate=True)
    return sdfg