# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from numbers import Number
from typing import Sequence, Union
import dace.library
from dace.properties import Property
import dace.sdfg.nodes
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState, graph
from dace.transformation.transformation import ExpandTransformation
from dace.frontend.common import op_repository as oprepo
from dace import dtypes, symbolic
from .. import environments, utils as nutil


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
        if input_data.storage not in [
                dtypes.StorageType.GPU_Global, dtypes.StorageType.CPU_Pinned
        ]:
            raise ValueError('Input of NCCL allreduce must either reside '
                             ' in global GPU memory or pinned CPU memory')
        if output_data.storage not in [
                dtypes.StorageType.GPU_Global, dtypes.StorageType.CPU_Pinned
        ]:
            raise ValueError('Output of NCCL allreduce must either reside '
                             ' in global GPU memory or pinned CPU memory')

        nccl_dtype_str = nutil.NCCL_DDT(input_data.dtype.base_type)
        count_str = "*".join(str(e) for e in input_dims)

        if input_data.dtype.veclen > 1:
            raise (NotImplementedError)
        code = f"""
            ncclAllReduce(_inbuffer, _outbuffer, {count_str}, {nccl_dtype_str}, {node.operation},  __state->nccl_handle->ncclCommunicators->at({node.location['gpu']}),  __dace_current_stream_id);
            """
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          location=node.location,
                                          language=dtypes.Language.CPP)
        return tasklet


@dace.library.node
class Allreduce(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "NCCL": ExpandAllreduceNCCL,
    }
    default_implementation = "NCCL"

    # Object fields
    operation = Property(
                    allow_none=True,
                    dtype=str,
                    choices = ['ncclSum', 'ncclProd', 'ncclMax', 'ncclMin'],
                    default="ncclSum",
                    desc="Supported reduction operations.")

    def __init__(self, name, operation="ncclSum", *args, **kwargs):
        super().__init__(name,
                         *args,
                         inputs={"_inbuffer"},
                         outputs={"_outbuffer"},
                         **kwargs)
        # Supported NCCL operations                         
        # nccl_operations = ['ncclSum', 'ncclProd', 'ncclMax', 'ncclMin']
        # if operation not in nccl_operations:
        #     raise ValueError(f'Operation {operation} is not supported by nccl, supported operations are: { ", ".join(map(str, nccl_operations))}.')
        self.operation = operation
        self.schedule = dtypes.ScheduleType.GPU_Multidevice

    def validate(self, sdfg: SDFG, state: SDFGState):
        
        in_edges = state.in_edges(self)
        if len(in_edges) != 1:
            raise ValueError("NCCL Allreduce must have one input.")
        
        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("NCCL Allreduce must have one output.")
        
        # in_data = in_edges[0].data
        # in_dims = in_data.subset.size_exact()
        
        # out_data = out_edges[0].data
        # out_dims = out_data.subset.size_exact()

        # if in_dims != out_dims:
        #     raise ValueError('')

    # def validate(self, sdfg, state):
    #     """
    #     :return: A three-tuple (buffer, root) of the three data descriptors in the
    #              parent SDFG.
    #     """

    #     inbuffer, outbuffer = None, None
    #     for e in state.out_edges(self):
    #         if e.src_conn == "_outbuffer":
    #             outbuffer = sdfg.arrays[e.data.data]
    #     for e in state.in_edges(self):
    #         if e.dst_conn == "_inbuffer":
    #             inbuffer = sdfg.arrays[e.data.data]

    #     count_str = "XXX"
    #     for _, src_conn, _, _, data in state.out_edges(self):
    #         if src_conn == '_outbuffer':
    #             dims = [str(e) for e in data.subset.size_exact()]
    #             count_str = "*".join(dims)

    #     return (inbuffer, count_str), outbuffer


@oprepo.replaces('dace.nccl.allreduce')
@oprepo.replaces('dace.nccl.Allreduce')
@oprepo.replaces('dace.nccl.AllReduce')
def _allreduce(pv: 'ProgramVisitor',
            sdfg: SDFG,
            state: SDFGState,
            in_buffer: str,
            out_buffer: str,
            operation: str = 'ncclSum'):
    
    # Add nodes
    in_node = state.add_read(in_buffer)
    out_node = state.add_write(out_buffer)

    libnode = Allreduce('nccl_AllReduce', operation=operation)

    # Connect nodes
    state.add_edge(in_node, None, libnode, '_inbuffer', Memlet(in_buffer))
    state.add_edge(libnode, '_outbuffer', out_node, None, Memlet(out_buffer))

    return []


# @oprepo.replaces('dace.nccl.allReduce')
# @oprepo.replaces('dace.nccl.AllReduce')
# def _allreduce(pv: 'ProgramVisitor',
#             sdfg: SDFG,
#             state: SDFGState,
#             in_buffer: str,
#             out_buffer: str,
#             op: str = 'ncclSum',
#             root: Union[str, symbolic.SymbolicType, Number] = 0):

#     from dace.libraries.nccl.nodes.reduce import Reduce

#     libnode = Reduce('nccl_AllReduce_', op=op)
#     in_desc = sdfg.arrays[in_buffer]
#     out_desc = sdfg.arrays[out_buffer]
#     in_node = state.add_read(in_buffer)
#     out_node = state.add_write(out_buffer)
#     # if isinstance(root, str) and root in sdfg.arrays.keys():
#     #     root_node = state.add_read(root)
#     # else:
#     #     storage = in_desc.storage
#     #     root_name = _define_local_scalar(pv, sdfg, state, dace.int32, storage)
#     #     root_node = state.add_access(root_name)
#     #     root_tasklet = state.add_tasklet('_set_root_', {}, {'__out'},
#     #                                      '__out = {}'.format(root))
#     #     state.add_edge(root_tasklet, '__out', root_node, None,
#     #                    Memlet.simple(root_name, '0'))
#     state.add_edge(in_node, None, libnode, '_inbuffer',
#                    Memlet.from_array(in_buffer, in_desc))
#     # state.add_edge(root_node, None, libnode, '_root',
#     #                Memlet.simple(root_node.data, '0'))
#     state.add_edge(libnode, '_outbuffer', out_node, None,
#                    Memlet.from_array(out_buffer, out_desc))

#     return None