# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" File defining the reduction library node. """

import dace
from dace.transformation import transformation as pm
from dace.sdfg.nodes import Tasklet
from dace.sdfg.state import SDFGState
from dace.sdfg import SDFG


@dace.library.expansion
class ExpandWarpAllReduce(pm.ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node: 'Barrier', state: SDFGState, sdfg: SDFG):
        return None


@dace.library.expansion
class ExpandCUDAWarpAllReduce(pm.ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node: 'Barrier', state: SDFGState, sdfg: SDFG):
        return Tasklet(label='warp_all_reduce_expansion')


@dace.library.node
class WarpAllReduce(dace.sdfg.nodes.LibraryNode):
    implementations = {
        'pure': ExpandWarpAllReduce,
        'CUDA (warp)': ExpandCUDAWarpAllReduce,
    }

    default_implementation = 'pure'
