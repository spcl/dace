# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from numbers import Number
from typing import Sequence, Union, Callable, Any

import dace.serialize
import dace.library
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import graph, nodes

from dace.properties import Property, ListProperty, LambdaProperty
from dace.frontend.operations import detect_reduction_type
from dace.memlet import Memlet
from dace.transformation.transformation import ExpandTransformation
from dace.frontend.common import op_repository as oprepo
from dace import dtypes, symbolic, Config
from dace.dtypes import NcclGroupCalls
from dace.libraries.nccl import environments, utils as nutil


@dace.library.expansion
class ExpandAllreduceNCCL(ExpandTransformation):

    environments = [environments.nccl.NCCL]

    @staticmethod
    def expansion(node: 'Allreduce', state: SDFGState, sdfg: SDFG, **kwargs):

        node.validate(sdfg, state)
        input_edge: graph.MultiConnectorEdge = state.in_edges(node)[0]
        output_edge: graph.MultiConnectorEdge = state.out_edges(node)[0]
        input_dims = input_edge.data.subset.size_exact()
        output_dims = output_edge.data.subset.size_exact()
        input_data = sdfg.arrays[input_edge.data.data]
        output_data = sdfg.arrays[output_edge.data.data]

        # Verify that data is on the GPU
        if input_data.storage is not dtypes.StorageType.GPU_Global:
            raise ValueError('Input of NCCL Send must reside '
                             ' in global GPU memory.')
        if output_data.storage is not dtypes.StorageType.GPU_Global:
            raise ValueError('Output of NCCL Recv must reside '
                             ' in global GPU memory.')

        redtype = node.reduction_type

        redtype = dtypes.NCCL_SUPPORTED_OPERATIONS[redtype]
        wcrstr = str(redtype)
        wcrstr = wcrstr[wcrstr.find('.') + 1:]  # Skip "NcclReductionType."

        nccl_dtype_str = nutil.Nccl_dtypes(input_data.dtype.base_type)
        count_str = "*".join(str(e) for e in input_dims)

        if input_data.dtype.veclen > 1:
            raise (NotImplementedError)

        code = f"""ncclAllReduce(_inbuffer, _outbuffer, {count_str}, {nccl_dtype_str}, {wcrstr},  __state->ncclCommunicators->at(__dace_cuda_device),  __dace_current_stream)"""

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
class Allreduce(nodes.LibraryNode):

    # Global properties
    implementations = {
        "NCCL": ExpandAllreduceNCCL,
    }
    default_implementation = "NCCL"

    # Object fields
    wcr = LambdaProperty(default='lambda a, b: a + b')

    group_calls = Property(
        dtype=NcclGroupCalls,
        default=NcclGroupCalls.NoGroupCalls,
        desc='For the aggregation of multiple NCCL collective operations.'
        'Start: First of the aggregated collectives.'
        'End:  Last of the aggregated collectives.'
        'Both: Use group calls for just this collective.'
        'NoGroupCalls: Do not use group calls.')

    def __init__(self,
                 wcr: str = "lambda a, b: a + b",
                 group_calls: NcclGroupCalls = NcclGroupCalls.NoGroupCalls,
                 debuginfo=None,
                 *args,
                 **kwargs):

        super().__init__(name='nccl_AllReduce',
                         *args,
                         inputs={"_inbuffer"},
                         outputs={"_outbuffer"},
                         **kwargs)
        self.wcr = wcr
        self.group_calls = group_calls
        self.schedule = dtypes.ScheduleType.GPU_Multidevice
        self.debuginfo = debuginfo

    @staticmethod
    def from_json(json_obj, context=None):
        ret = Allreduce("lambda a, b: a + b", None)
        dace.serialize.set_properties_from_json(ret, json_obj, context=context)
        return ret

    def __str__(self):
        redtype = self.reduction_type

        wcrstr = str(redtype)
        wcrstr = wcrstr[wcrstr.find('.') + 1:]  # Skip "ReductionType."

        return f'nccl_AllReduce ({wcrstr})'

    def __label__(self, sdfg, state):
        return str(self).replace(' Axes', '\nAxes')

    @property
    def reduction_type(self):
        # Autodetect reduction type
        redtype = detect_reduction_type(self.wcr)
        if redtype not in dtypes.NCCL_SUPPORTED_OPERATIONS:
            raise ValueError(
                'NCCL only supports sum, product, min and max operations.')
        return redtype

    def validate(self, sdfg: SDFG, state: SDFGState):
        redtype = self.reduction_type

        in_edges = state.in_edges(self)
        if len(in_edges) != 1:
            raise ValueError("NCCL Allreduce must have one input.")

        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("NCCL Allreduce must have one output.")


@oprepo.replaces('dace.nccl.allreduce')
@oprepo.replaces('dace.nccl.Allreduce')
@oprepo.replaces('dace.nccl.AllReduce')
@oprepo.replaces('dace.nccl.allReduce')
def nccl_allreduce(
    pv: 'ProgramVisitor',
    sdfg: SDFG,
    state: SDFGState,
    redfunction: Callable[[Any, Any], Any],
    in_array: str,
    out_array: Union[str, None] = None,
    group_calls: NcclGroupCalls = NcclGroupCalls.NoGroupCalls,
):

    # If out_array is not specified, the operation will be in-place.
    if out_array is None:
        out_array = in_array

    # Add nodes
    in_node = state.add_read(in_array)
    out_node = state.add_write(out_array)

    libnode = Allreduce(redfunction, group_calls=group_calls)

    # Connect nodes
    state.add_edge(in_node, None, libnode, '_inbuffer', Memlet(in_array))
    state.add_edge(libnode, '_outbuffer', out_node, None, Memlet(out_array))

    return []
