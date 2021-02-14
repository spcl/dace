# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" File defining the reduction library node. """

import dace
from dace.transformation import transformation as pm
from dace.sdfg.nodes import Tasklet
from dace.sdfg.state import SDFGState
from dace.sdfg import SDFG
from dace import dtypes
from dace.properties import Property
from dace.frontend import operations
import textwrap


@dace.library.expansion
class ExpandParallelAllReducePure(pm.ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node: 'ParallelAllReduce', state: SDFGState, sdfg: SDFG):
        raise Exception('ExpandParallelAllReducePure not implemented')


@dace.library.expansion
class ExpandParallelAllReduceCUDAWarp(pm.ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node: 'ParallelAllReduce', state: SDFGState, sdfg: SDFG):
        input_name = 'in'
        output_name = 'out'

        reduction_type = operations.detect_reduction_type(node.wcr)

        if reduction_type == dtypes.ReductionType.Max:
            reduction_op = 'if (new_val > out) { out = new_val; }'
        elif reduction_type == dtypes.ReductionType.Sum:
            reduction_op = 'out += new_val;'
        else:
            raise Exception("Unknown reduction type")

        code = textwrap.dedent("""
                    out = in;
                    # pragma unroll
                    for (int _ari = 1; _ari < 32; _ari = _ari * 2) {{
                        auto new_val = __shfl_xor_sync(0xffffffff, out, _ari);
                        {reduction_op}
                    }}
                """.format(reduction_op=reduction_op))

        warp_all_reduce_tasklet: nodes.Tasklet = state.add_tasklet(
            name='warp_all_reduce',
            inputs={input_name},
            outputs={output_name},
            code=code,
            language=dtypes.Language.CPP)

        for in_edge in state.in_edges(node):
            in_edge.dst_conn = input_name
            #state.add_edge(in_edge.src, in_edge.src_conn, warp_all_reduce_tasklet, input_name, in_edge.data)

        for out_edge in state.out_edges(node):
            out_edge.src_conn = output_name
            #state.add_edge(warp_all_reduce_tasklet, output_name, out_edge.dst, out_edge.dst_conn, out_edge.data)

        return warp_all_reduce_tasklet


@dace.library.node
class ParallelAllReduce(dace.sdfg.nodes.LibraryNode):
    implementations = {
        'pure': ExpandParallelAllReducePure,
        'CUDA (warp)': ExpandParallelAllReduceCUDAWarp,
    }

    default_implementation = 'CUDA (warp)'

    wcr = Property(dtype=str)

    def __init__(self, name: str, wcr: str):
        dace.sdfg.nodes.LibraryNode.__init__(self, name)
        self.wcr = wcr