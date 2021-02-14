# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" File defining the reduction library node. """

import dace
from dace.transformation import transformation as pm
from dace.sdfg.nodes import Tasklet
from dace.sdfg.state import SDFGState
from dace.sdfg import SDFG
from dace import dtypes

@dace.library.expansion
class ExpandBarrierPure(pm.ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node: 'Barrier', state: SDFGState, sdfg: SDFG):
        raise Exception('ExpandBarrierPure not implemented')


@dace.library.expansion
class ExpandBarrierCUDADevice(pm.ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node: 'Barrier', state: SDFGState, sdfg: SDFG):
        raise Exception('ExpandBarrierCUDADevice not implemented')


@dace.library.expansion
class ExpandBarrierCUDABlock(pm.ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node: 'Barrier', state: SDFGState, sdfg: SDFG):
        tasklet = Tasklet('barrier_tasklet', code="__syncthreads();", language=dtypes.Language.CPP)
        state.add_node(tasklet)
        return tasklet


@dace.library.node
class Barrier(dace.sdfg.nodes.LibraryNode):
    implementations = {
        'pure': ExpandBarrierPure,
        'CUDA (device)': ExpandBarrierCUDADevice,
        'CUDA (block)': ExpandBarrierCUDABlock,
    }

    default_implementation = 'CUDA (block)'
