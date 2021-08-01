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
class ExpandSendNCCL(ExpandTransformation):

    environments = [environments.nccl.NCCL]

    @staticmethod
    def expansion(node: 'Send', state: SDFGState, sdfg: SDFG, **kwargs):

        node.validate(sdfg, state)
        input_edge: graph.MultiConnectorEdge = state.in_edges(node)[0]
        input_dims = input_edge.data.subset.size_exact()
        input_data = sdfg.arrays[input_edge.data.data]

        # Verify that data is on the GPU
        if input_data.storage is not dtypes.StorageType.GPU_Global:
            raise ValueError('Input of NCCL Send must reside '
                             ' in global GPU memory.')

        peer = node.peer

        nccl_dtype_str = nutil.Nccl_dtypes(input_data.dtype.base_type)
        count_str = "*".join(str(e) for e in input_dims)

        if input_data.dtype.veclen > 1:
            raise (NotImplementedError)

        code = f"""ncclSend(_inbuffer, {count_str}, {nccl_dtype_str}, {peer}, __state->ncclCommunicators->at(__dace_cuda_device),  __dace_current_stream)"""
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
class Send(nodes.LibraryNode):

    # Global properties
    implementations = {
        "NCCL": ExpandSendNCCL,
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

        super().__init__(name='nccl_Send',
                         *args,
                         inputs={"_inbuffer"},
                         **kwargs)
        self.peer = peer
        self.group_calls = group_calls
        self.schedule = dtypes.ScheduleType.GPU_Multidevice
        self.debuginfo = debuginfo

    @staticmethod
    def from_json(json_obj, context=None):
        ret = Send(None)
        dace.serialize.set_properties_from_json(ret, json_obj, context=context)
        return ret

    def __str__(self):
        return f'nccl_Send (peer={self.peer})'

    def __label__(self, sdfg, state):
        return str(self).replace(' Axes', '\nAxes')

    def validate(self, sdfg: SDFG, state: SDFGState):
        in_edges = state.in_edges(self)
        if len(in_edges) != 1:
            raise ValueError("NCCL Send must have one input.")


@oprepo.replaces('dace.nccl.send')
@oprepo.replaces('dace.nccl.Send')
def nccl_send(pv: 'ProgramVisitor',
              sdfg: SDFG,
              state: SDFGState,
              in_array: str,
              peer: symbolic.SymbolicType = 0,
              group_calls: NcclGroupCalls = NcclGroupCalls.NoGroupCalls):

    # Add nodes
    in_node = state.add_read(in_array)
    libnode = Send(peer=peer, group_calls=group_calls)

    # Connect nodes
    state.add_edge(in_node, None, libnode, '_inbuffer', Memlet(in_array))

    return []
