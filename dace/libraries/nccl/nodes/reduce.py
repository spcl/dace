# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from numbers import Number, Integral
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace import dtypes
from dace.transformation.transformation import ExpandTransformation
from .. import environments


@dace.library.expansion
class ExpandReduceNCCL(ExpandTransformation):

    environments = [environments.nccl.NCCL]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (inbuffer,
         count_str), outbuffer = node.validate(parent_sdfg, parent_state)
        nccl_dtype_str = dace.libraries.nccl.utils.NCCL_DDT(
            inbuffer.dtype.base_type)
        # if inbuffer.dtype.veclen > 1:
        #     raise (NotImplementedError)
        # if root.dtype.base_type != dace.dtypes.int32:
        #     raise ValueError("Reduce root must be an integer!")
        
        # Verify that data is accessible from the GPUs
        if inbuffer.storage not in [
                dtypes.StorageType.GPU_Global, dtypes.StorageType.CPU_Pinned
        ]:
            raise ValueError('Inputbuffer of multi-GPU reduction must either'
                            ' reside in global GPU memory or pinned CPU memory')
        if outbuffer.storage not in [
                dtypes.StorageType.GPU_Global, dtypes.StorageType.CPU_Pinned
        ]:
            raise ValueError('Outputbuffer of multi-GPU reduction must either'
                            ' reside in global GPU memory or pinned CPU memory')
        code = (
            f"ncclReduce(sendbuff, recvbuff, {count_str}, {nccl_dtype_str}, {node._op}, {node._root}, __state->nccl_handle->ncclCommunicators->at({node.location['gpu']}), __dace_current_stream_id);")
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.node
class Reduce(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "NCCL": ExpandReduceNCCL,
    }
    default_implementation = "NCCL"

    def __init__(self, name, op="ncclSum", root=None, *args, **kwargs):
        super().__init__(name,
                         *args,
                         inputs={"_inbuffer"},
                         outputs={"_outbuffer"},
                         **kwargs)
        # Supported NCCL operations                         
        nccl_operations = ['ncclSum', 'ncclProd', 'ncclMax', 'ncclMin']
        if op not in nccl_operations:
            raise ValueError(f'Operation {op} is not supported by nccl, supported operations are: { ", ".join(map(str, nccl_operations))}.')
        self._op = op
        self.schedule = dtypes.ScheduleType.GPU_Multidevice
        if root != None:
            self._root = root
        # else:
            

    def validate(self, sdfg, state):
        """
        :return: A three-tuple (buffer, root) of the three data descriptors in the
                 parent SDFG.
        """

        inbuffer, outbuffer = None, None
        for e in state.out_edges(self):
            if e.src_conn == "_outbuffer":
                outbuffer = sdfg.arrays[e.data.data]
        for e in state.in_edges(self):
            if e.dst_conn == "_inbuffer":
                inbuffer = sdfg.arrays[e.data.data]
            # if e.dst_conn == "_root":
            #     root = sdfg.arrays[e.data.data]

        if isinstance(self._root.dtype, Integral):
            raise (ValueError("Reduce root must be an integer!"))

        count_str = "XXX"
        for _, src_conn, _, _, data in state.out_edges(self):
            if src_conn == '_outbuffer':
                dims = [str(e) for e in data.subset.size_exact()]
                count_str = "*".join(dims)

        return (inbuffer, count_str), outbuffer
