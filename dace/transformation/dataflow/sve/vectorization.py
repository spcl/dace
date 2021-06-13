# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
    SVE Vectorization: This module is responsible to vectorize SDFGs for the Arm SVE codegen.
"""
from dace import registry, symbolic
from dace.properties import make_properties, Property, ShapeProperty
from dace.sdfg import nodes, SDFG
from dace.sdfg import utils as sdutil
from dace.transformation import transformation


@registry.autoregister_params(singlestate=True)
class SVEVectorization(transformation.Transformation):
    """ Hello world

        Lorem ipsum
"""
    _map_entry = transformation.PatternNode(nodes.MapEntry)
    
    @staticmethod
    def expressions():
        return [sdutil.node_path_graph(SVEVectorization._map_entry)]

    def can_be_applied(self, graph, candidate, expr_index, sdfg, strict=False):
        return True

    def apply(self, sdfg: SDFG):
        pass
