# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from numbers import Number
from typing import Sequence, Union, Callable, Any
from dace import Config

import dace.serialize
import dace.library
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import graph, nodes

from dace.properties import Property, LambdaProperty, SymbolicProperty
from dace.memlet import Memlet
from dace.transformation.transformation import ExpandTransformation
from dace.frontend.common import op_repository as oprepo
from dace import dtypes, symbolic
from dace.dtypes import NcclGroupCalls
from dace.libraries.nccl import environments, utils as nutil


@dace.library.expansion
class ExpandRecvNCCL(ExpandTransformation):

    environments = [environments.nccl.NCCL]

    @staticmethod
    def expansion(node: 'Recv', state: SDFGState, sdfg: SDFG, **kwargs):

        node.validate(sdfg, state)
        output_edge: graph.MultiConnectorEdge = state.out_edges(node)[0]
        output_dims = output_edge.data.subset.size_exact()
        output_data = sdfg.arrays[output_edge.data.data]

        # Verify that data is on the GPU
        if output_data.storage is not dtypes.StorageType.GPU_Global:
            raise ValueError('Output of NCCL Recv must reside '
                             ' in global GPU memory.')

        peer = node.peer

        nccl_dtype_str = nutil.Nccl_dtypes(output_data.dtype.base_type)
        count_str = "*".join(str(e) for e in output_dims)

        if output_data.dtype.veclen > 1:
            raise (NotImplementedError)

        code = f"""ncclRecv(_outbuffer, {count_str}, {nccl_dtype_str}, {peer}, __state->ncclCommunicators->at(__dace_cuda_device),  __dace_current_stream)"""
        if Config.get('compiler', 'build_type') == 'Debug':
            code = '''DACE_NCCL_CHECK(''' + code + ''');\n'''
        else:
            code = code + ''';\n'''

        if node.group_calls in [NcclGroupCalls.Start, NcclGroupCalls.Both]:
            code = """
            ncclGroupStart();\n""" + code
        if node.group_calls in [NcclGroupCalls.End, NcclGroupCalls.Both]:
            code += """ncclGroupEnd();"""

        tasklet = nodes.Tasklet(node.name,
                                node.in_connectors,
                                node.out_connectors,
                                code,
                                location=node.location,
                                language=dtypes.Language.CPP)
        return tasklet


@dace.library.node
class Recv(nodes.LibraryNode):

    # Global properties
    implementations = {
        "NCCL": ExpandRecvNCCL,
    }
    default_implementation = "NCCL"

    # Object fields
    peer = SymbolicProperty(default=0,
                            allow_none=True,
                            desc="The gpu on which the receive buffer resides")

    group_calls = Property(
        dtype=NcclGroupCalls,
        default=NcclGroupCalls.NoGroupCalls,
        desc='For the aggregation of multiple NCCL collective operations.'
        'Start: First of the aggregated collectives.'
        'End:  Last of the aggregated collectives.'
        'Both: Use group calls for just this collective.'
        'NoGroupCalls: Do not use group calls.')

    def __init__(self,
                 peer: symbolic.SymbolicType = 0,
                 group_calls: NcclGroupCalls = NcclGroupCalls.NoGroupCalls,
                 debuginfo=None,
                 *args,
                 **kwargs):

        super().__init__(name='nccl_Recv',
                         *args,
                         outputs={"_outbuffer"},
                         **kwargs)
        self.peer = peer
        self.group_calls = group_calls
        self.schedule = dtypes.ScheduleType.GPU_Multidevice
        self.debuginfo = debuginfo

    @staticmethod
    def from_json(json_obj, context=None):
        ret = Recv(None)
        dace.serialize.set_properties_from_json(ret, json_obj, context=context)
        return ret

    def __str__(self):
        return f'nccl_Recv (peer={self.peer})'

    def __label__(self, sdfg, state):
        return str(self).replace(' Axes', '\nAxes')

    def validate(self, sdfg: SDFG, state: SDFGState):
        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("NCCL Recv must have one output.")


@oprepo.replaces('dace.nccl.recv')
@oprepo.replaces('dace.nccl.Recv')
def nccl_recv(pv: 'ProgramVisitor',
              sdfg: SDFG,
              state: SDFGState,
              out_array: str,
              peer: symbolic.SymbolicType = 0,
              group_calls: NcclGroupCalls = NcclGroupCalls.NoGroupCalls):

    # Add nodes
    out_node = state.add_write(out_array)
    libnode = Recv(peer=peer, group_calls=group_calls)

    # Connect nodes
    state.add_edge(libnode, '_outbuffer', out_node, None, Memlet(out_array))

    return []
