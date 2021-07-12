# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from numbers import Number
from typing import Sequence, Union, Callable, Any

import dace.serialize
import dace.library
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import graph, nodes

from dace.frontend.python.astutils import unparse
from dace.properties import Property, LambdaProperty, SymbolicProperty
from dace.frontend.operations import detect_reduction_type
from dace.memlet import Memlet
from dace.transformation.transformation import ExpandTransformation
from dace.frontend.common import op_repository as oprepo
from dace import dtypes, symbolic
from dace.libraries.nccl import environments, utils as nutil


@dace.library.expansion
class ExpandReduceNCCL(ExpandTransformation):

    environments = [environments.nccl.NCCL]

    @staticmethod
    def expansion(node: 'Reduce', state: SDFGState, sdfg: SDFG, **kwargs):
        
        node.validate(sdfg, state)
        input_edge: graph.MultiConnectorEdge = state.in_edges(node)[0]
        output_edge: graph.MultiConnectorEdge = state.out_edges(node)[0]
        input_dims = input_edge.data.subset.size_exact()
        output_dims = output_edge.data.subset.size_exact()
        input_data = sdfg.arrays[input_edge.data.data]
        output_data = sdfg.arrays[output_edge.data.data]

        # Verify that data is on the GPU
        if input_data.storage not in [
                dtypes.StorageType.GPU_Global, dtypes.StorageType.CPU_Pinned
        ]:
            raise ValueError('Input of NCCL Reduce must either reside '
                             ' in global GPU memory or pinned CPU memory')
        if output_data.storage not in [
                dtypes.StorageType.GPU_Global, dtypes.StorageType.CPU_Pinned
        ]:
            raise ValueError('Output of NCCL Reduce must either reside '
                             ' in global GPU memory or pinned CPU memory')

        redtype = detect_reduction_type(node.wcr)
        if redtype not in dtypes.NCCL_SUPPORTED_REDUCTIONS:
            raise ValueError('NCCL only supports sum, product, min and max reductions.')
        redtype = dtypes.NCCL_SUPPORTED_REDUCTIONS[redtype]
        wcr_str = str(redtype)
        wcr_str = wcr_str[wcr_str.find('.') + 1:]  # Skip "NcclReductionType."

        root = node.root

        nccl_dtype_str = nutil.NCCL_DDT(input_data.dtype.base_type)
        count_str = "*".join(str(e) for e in input_dims)

        if input_data.dtype.veclen > 1:
            raise (NotImplementedError)
        code = f"""
            ncclReduce(_inbuffer, _outbuffer, {count_str}, {nccl_dtype_str}, {wcr_str}, {root}, __state->nccl_handle->ncclCommunicators->at({node.location['gpu']}),  __dace_current_stream_id);
            """
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          location=node.location,
                                          language=dtypes.Language.CPP)
        return tasklet


@dace.library.node
class Reduce(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "NCCL": ExpandReduceNCCL,
    }
    default_implementation = "NCCL"

    # Object fields
    root = SymbolicProperty(default=0, allow_none=True,
                            desc="The gpu on which the receive buffer resides")
    wcr = LambdaProperty(default='lambda a, b: a + b')

    def __init__(self, 
                 wcr="lambda a, b: a + b",
                 debuginfo=None,
                 *args, 
                 **kwargs):

        super().__init__(name='nccl_AllReduce',
                         *args,
                         inputs={"_inbuffer"},
                         outputs={"_outbuffer"},
                         **kwargs)
        self.wcr = wcr
        self.root = root
        self.schedule = dtypes.ScheduleType.GPU_Multidevice
        self.debuginfo = debuginfo
    
    @staticmethod
    def from_json(json_obj, context=None):
        ret = Reduce("lambda a, b: a + b", None)
        dace.serialize.set_properties_from_json(ret, json_obj, context=context)
        return ret
    
    def __str__(self):
        # Autodetect reduction type
        redtype = detect_reduction_type(self.wcr)
        if redtype not in dtypes.NCCL_SUPPORTED_REDUCTIONS:
            raise ValueError('NCCL only supports sum, product, min and max reductions.')
        
        wcrstr = str(redtype)
        wcrstr = wcrstr[wcrstr.find('.') + 1:]  # Skip "ReductionType."
        
        return 'nccl_AllReduce ({op})'.format(op=wcrstr)
    
    def __label__(self, sdfg, state):
        return str(self).replace(' Axes', '\nAxes')
    
    def _ged_redtype(self):
        redtype = detect_reduction_type(self.wcr)
        if redtype not in dtypes.NCCL_SUPPORTED_REDUCTIONS:
            raise ValueError('NCCL only supports sum, product, min and max reductions.')
        return redtype

    def validate(self, sdfg: SDFG, state: SDFGState):
        redtype = detect_reduction_type(self.wcr)
        if redtype not in dtypes.NCCL_SUPPORTED_REDUCTIONS:
            raise ValueError('NCCL only supports sum, product, min and max reductions.')
        
        in_edges = state.in_edges(self)
        if len(in_edges) != 1:
            raise ValueError("NCCL Reduce must have one input.")
        
        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("NCCL Reduce must have one output.")

@oprepo.replaces('dace.nccl.reduce')
@oprepo.replaces('dace.nccl.Reduce')
def nccl_reduce(pv: 'ProgramVisitor',
            sdfg: SDFG,
            state: SDFGState,
            redfunction: Callable[[Any, Any], Any],
            in_array: str,
            out_array: str,):
    
    # Add nodes
    in_node = state.add_read(in_array)
    out_node = state.add_write(out_array)

    libnode = Reduce(redfunction)

    # Connect nodes
    state.add_edge(in_node, None, libnode, '_inbuffer', Memlet(in_array))
    state.add_edge(libnode, '_outbuffer', out_node, None, Memlet(out_array))

    return []