# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Sequence, Union, Callable, Any, Set
import warnings
import dace
from dace import Config
import dace.serialize
import dace.library
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import nodes

from dace.properties import Property, LambdaProperty, SymbolicProperty
from dace.memlet import Memlet
from dace.transformation.transformation import ExpandTransformation
from dace.frontend.common import op_repository as oprepo
from dace import dtypes, symbolic
from dace.libraries.nccl import environments, utils as nutil
from dace.frontend.python.replacements import _define_local_scalar


@dace.library.expansion
class ExpandSendNCCL(ExpandTransformation):

    environments = [environments.nccl.NCCL]

    @staticmethod
    def expansion(node: 'Send', state: SDFGState, sdfg: SDFG, **kwargs):

        node.validate(sdfg, state)
        for edge in state.in_edges(node):
            if edge.dst_conn == '_inbuffer':
                input_edge = edge

        input_dims = input_edge.data.subset.size_exact()
        input_data = sdfg.arrays[input_edge.data.data]

        # Verify that data is on the GPU
        if input_data.storage is not dtypes.StorageType.GPU_Global:
            raise ValueError('Input of NCCL Send must reside '
                             ' in global GPU memory.')

        peer = node.peer
        peerstr = str(peer)
        for fs in peer.free_symbols:
            if fs.name in sdfg.arrays or fs.name in sdfg.parent_sdfg.arrays:
                sdfg.arrays[fs.name].lifetime = dtypes.AllocationLifetime.SDFG

        nccl_dtype_str = nutil.Nccl_dtypes(input_data.dtype.base_type)
        count_str = "*".join(str(e) for e in input_dims)

        if input_data.dtype.veclen > 1:
            raise (NotImplementedError)

        code = f"""ncclSend(_inbuffer, {count_str}, {nccl_dtype_str}, {peerstr}, __state->ncclCommunicators->at(__dace_cuda_device),  __dace_current_stream)"""
        if Config.get('compiler', 'build_type') == 'Debug':
            code = '''DACE_NCCL_CHECK(''' + code + ''');\n'''
        else:
            code = code + ''';\n'''

        group_handle_conn = '_group_handle'
        if group_handle_conn in node.in_connectors:
            for edge in state.in_edges(node):
                if edge.dst_conn == group_handle_conn:
                    in_gh_edge = edge
                    in_gh_node = edge.src
            if not state.predecessors(in_gh_node):
                code = """ncclGroupStart();\n""" + code
            else:
                predecessor_node = state.in_edges(in_gh_node)[0].src
                state.add_edge(predecessor_node, None, node, None, Memlet())
                state.remove_edge_and_connectors(state.in_edges(in_gh_node)[0])
            state.remove_edge_and_connectors(in_gh_edge)
            node.remove_in_connector(group_handle_conn)
            state.remove_node(in_gh_node)

        if group_handle_conn in node.out_connectors:
            for edge in state.out_edges(node):
                if edge.src_conn == group_handle_conn:
                    out_gh_edge = edge
                    out_gh_node = edge.dst
            if not state.successors(out_gh_node):
                code += """ncclGroupEnd();"""
                out_gh_data = out_gh_node.data
                state.remove_edge_and_connectors(out_gh_edge)
                state.remove_node(out_gh_node)
                try:
                    sdfg.remove_data(out_gh_data)
                except ValueError as ex:
                    warnings.warn(str(ex))
            node.remove_out_connector(group_handle_conn)

        tasklet = nodes.Tasklet(str(node),
                                node.in_connectors,
                                node.out_connectors,
                                code,
                                location=node.location,
                                language=dtypes.Language.CPP,
                                library_expansion_symbols=set([str(peer)]))

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

    def __init__(self,
                 peer: symbolic.SymbolicType = 0,
                 debuginfo=None,
                 *args,
                 **kwargs):

        super().__init__(name='nccl_Send', *args, **kwargs)
        self.peer = peer
        self.schedule = dtypes.ScheduleType.GPU_Multidevice
        self.debuginfo = debuginfo

    @staticmethod
    def from_json(json_obj, context=None):
        ret = Send(None)
        dace.serialize.set_properties_from_json(ret, json_obj, context=context)
        return ret

    def __str__(self):
        return f'nccl_Send(peer={self.peer})'

    def validate(self, sdfg: SDFG, state: SDFGState):
        in_edges = state.in_edges(self)
        if len(in_edges) not in [1, 2]:
            raise ValueError("NCCL Send must have one or two inputs.")
        out_edges = state.out_edges(self)
        if len(out_edges) not in [0, 1]:
            raise ValueError("NCCL Send must have zero or one output.")

    @property
    def free_symbols(self) -> Set[str]:
        result = super().free_symbols
        result.update(map(str, self.peer.free_symbols))
        return result


@oprepo.replaces('dace.comm.nccl.send')
@oprepo.replaces('dace.comm.nccl.Send')
def nccl_send(pv: 'ProgramVisitor',
              sdfg: SDFG,
              state: SDFGState,
              in_buffer: str,
              peer: symbolic.SymbolicType = 0,
              group_handle: str = None):

    inputs = {"_inbuffer"}
    outputs = set()

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

    libnode = Send(inputs=inputs, outputs=outputs, peer=peer)

    if isinstance(group_handle, str):
        gh_memlet = Memlet.simple(gh_name, '0')
        if not gh_start:
            state.add_edge(gh_in, None, libnode, "_group_handle", gh_memlet)
        state.add_edge(libnode, "_group_handle", gh_out, None, gh_memlet)

    in_range = None
    if isinstance(in_buffer, tuple):
        in_name, in_range = in_buffer
    else:
        in_name = in_buffer

    desc = sdfg.arrays[in_name]
    conn = libnode.in_connectors
    conn = {
        c: (dtypes.pointer(desc.dtype) if c == '_buffer' else t)
        for c, t in conn.items()
    }
    libnode.in_connectors = conn
    in_node = state.add_read(in_name)

    if in_range:
        buf_mem = Memlet.simple(in_name, in_range)
    else:
        buf_mem = Memlet.from_array(in_name, desc)

    state.add_edge(in_node, None, libnode, '_inbuffer', buf_mem)

    return []
