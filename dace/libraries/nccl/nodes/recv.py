# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from numbers import Number
from typing import Sequence, Union, Callable, Any, Set
import warnings
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
from dace.libraries.nccl import environments, utils as nutil
from dace.frontend.python.replacements import _define_local_scalar


@dace.library.expansion
class ExpandRecvNCCL(ExpandTransformation):

    environments = [environments.nccl.NCCL]

    @staticmethod
    def expansion(node: 'Recv', state: SDFGState, sdfg: SDFG, **kwargs):

        node.validate(sdfg, state)
        for edge in state.out_edges(node):
            if edge.src_conn == '_outbuffer':
                output_edge = edge
        output_dims = output_edge.data.subset.size_exact()
        output_data = sdfg.arrays[output_edge.data.data]

        # Verify that data is on the GPU
        if output_data.storage is not dtypes.StorageType.GPU_Global:
            raise ValueError('Output of NCCL Recv must reside '
                             ' in global GPU memory.')

        peer = node.peer
        peerstr = str(peer)
        for fs in peer.free_symbols:
            if fs.name in sdfg.arrays:
                sdfg.arrays[fs.name].lifetime = dtypes.AllocationLifetime.SDFG
            if fs.name in sdfg.parent_sdfg.arrays:
                sdfg.parent_sdfg.arrays[
                    fs.name].lifetime = dtypes.AllocationLifetime.SDFG

        nccl_dtype_str = nutil.Nccl_dtypes(output_data.dtype.base_type)
        count_str = "*".join(str(e) for e in output_dims)

        if output_data.dtype.veclen > 1:
            raise (NotImplementedError)

        code = f"""ncclRecv(_outbuffer, {count_str}, {nccl_dtype_str}, {peerstr}, __state->ncclCommunicators->at(__dace_cuda_device),  __dace_current_stream)"""
        if Config.get('compiler', 'build_type') == 'Debug':
            code = '''DACE_NCCL_CHECK(''' + code + ''');\n'''

        else:
            code = code + ''';\n'''

        if Config.get_bool('debugprint'):
            code = (
                f'''printf("{str(node)}: begin;  dev,peer: %d, %d\\n", __dace_cuda_device, {peerstr});\n'''
                + code +
                f'''printf("{str(node)}: end;  dev,peer: %d, %d\\n\\n", __dace_cuda_device, {peerstr});\n'''
            )

        code = nutil.aggregate_calls(sdfg, state, node, code)

        tasklet = nodes.Tasklet(node.name,
                                node.in_connectors,
                                node.out_connectors,
                                code,
                                location=node.location,
                                language=dtypes.Language.CPP,
                                library_expansion_symbols=set(
                                    map(str, peer.free_symbols)))

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

    def __init__(self,
                 peer: symbolic.SymbolicType = 0,
                 debuginfo=None,
                 *args,
                 **kwargs):

        super().__init__(name='nccl_Recv', *args, **kwargs)
        self.peer = peer
        self.schedule = dtypes.ScheduleType.GPU_Multidevice
        self.debuginfo = debuginfo

    @staticmethod
    def from_json(json_obj, context=None):
        ret = Recv(None)
        dace.serialize.set_properties_from_json(ret, json_obj, context=context)
        return ret

    def __str__(self):
        return f'nccl_Recv(peer={self.peer})'

    def validate(self, sdfg: SDFG, state: SDFGState):
        in_edges = state.in_edges(self)
        if len(in_edges) not in [0, 1]:
            raise ValueError("NCCL Recv must have zero or one input.")
        out_edges = state.out_edges(self)
        if len(out_edges) not in [1, 2]:
            raise ValueError("NCCL Recv must have one or two outputs.")

    @property
    def free_symbols(self) -> Set[str]:
        result = super().free_symbols
        result.update(map(str, self.peer.free_symbols))
        return result


@oprepo.replaces('dace.comm.nccl.recv')
@oprepo.replaces('dace.comm.nccl.Recv')
def nccl_recv(pv: 'ProgramVisitor',
              sdfg: SDFG,
              state: SDFGState,
              out_buffer: str,
              peer: symbolic.SymbolicType = 0,
              group_handle: str = None):

    inputs = set()
    outputs = {"_outbuffer"}

    if isinstance(group_handle, str):
        gh_start = False
        if group_handle in sdfg.arrays.keys():
            gh_name = group_handle
            gh_out = state.add_access(gh_name)
            gh_in = state.add_access(gh_name)
            inputs.add("_group_handle")
        else:
            gh_start = True
            gh_name = _define_local_scalar(pv, sdfg, state, dace.int32,
                                           dtypes.StorageType.GPU_Global)
            gh_out = state.add_access(gh_name)
        outputs.add("_group_handle")

    libnode = Recv(inputs=inputs, outputs=outputs, peer=peer)

    if isinstance(group_handle, str):
        gh_memlet = Memlet.simple(gh_name, '0')
        if not gh_start:
            state.add_edge(gh_in, None, libnode, "_group_handle", gh_memlet)
        state.add_edge(libnode, "_group_handle", gh_out, None, gh_memlet)

    out_range = None
    if isinstance(out_buffer, tuple):
        out_name, out_range = out_buffer
        out_node = state.add_write(out_name)
    elif isinstance(out_buffer, str) and out_buffer in sdfg.arrays.keys():
        out_name = out_buffer
        out_node = state.add_write(out_name)
    else:
        raise ValueError(
            "NCCL_Recv out_buffer must be an array, or a an array range tuple.")

    if out_range:
        out_mem = Memlet.simple(out_name, out_range)
    else:
        out_mem = Memlet.simple(out_name, '0')

    state.add_edge(libnode, '_outbuffer', out_node, None, out_mem)

    return []
