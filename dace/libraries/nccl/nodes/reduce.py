# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Sequence, Union, Callable, Any, Set
import warnings
import dace
from dace import Config

import dace.serialize
import dace.library
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import graph, nodes

from dace.properties import Property, LambdaProperty, SymbolicProperty
from dace.frontend.operations import detect_reduction_type
from dace.memlet import Memlet
from dace.transformation.transformation import ExpandTransformation
from dace.frontend.common import op_repository as oprepo
from dace import dtypes, symbolic
from dace.libraries.nccl import environments, utils as nutil
from dace.frontend.python.replacements import _define_local_scalar


@dace.library.expansion
class ExpandReduceNCCL(ExpandTransformation):

    environments = [environments.nccl.NCCL]

    @staticmethod
    def expansion(node: 'Reduce', state: SDFGState, sdfg: SDFG, **kwargs):

        node.validate(sdfg, state)
        for edge in state.in_edges(node):
            if edge.dst_conn == '_inbuffer':
                input_edge = edge
        for edge in state.out_edges(node):
            if edge.src_conn == '_outbuffer':
                output_edge = edge
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

        root = node.root
        rootstr = str(root)
        for fs in root.free_symbols:
            if fs.name in sdfg.arrays:
                sdfg.arrays[fs.name].lifetime = dtypes.AllocationLifetime.SDFG
            if fs.name in sdfg.parent_sdfg.arrays:
                sdfg.parent_sdfg.arrays[
                    fs.name].lifetime = dtypes.AllocationLifetime.SDFG
        redtype = node.reduction_type
        redtype = nutil.NCCL_SUPPORTED_OPERATIONS[redtype]
        wcr_str = str(redtype)
        wcr_str = wcr_str[wcr_str.find('.') + 1:]  # Skip "NcclReductionType."

        nccl_dtype_str = nutil.Nccl_dtypes(input_data.dtype.base_type)
        count_str = "*".join(str(e) for e in input_dims)

        if input_data.dtype.veclen > 1:
            raise (NotImplementedError)

        code = f"""ncclReduce(_inbuffer, _outbuffer, {count_str}, {nccl_dtype_str}, {wcr_str}, {rootstr}, __state->ncclCommunicators->at(__dace_cuda_device),  __dace_current_stream)"""
        if Config.get('compiler', 'build_type') == 'Debug':
            code = '''DACE_NCCL_CHECK(''' + code + ''');\n'''

        else:
            code = code + ''';\n'''

        if Config.get_bool('debugprint'):
            code = (
                f'''printf("{str(node)}: begin;  dev,peer: %d, %d\\n", __dace_cuda_device, {rootstr});\n'''
                + code +
                f'''printf("{str(node)}: end;  dev,peer: %d, %d\\n\\n", __dace_cuda_device, {rootstr});\n'''
            )

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
                                library_expansion_symbols=set([str(root)]))

        return tasklet


@dace.library.node
class Reduce(nodes.LibraryNode):

    # Global properties
    implementations = {
        "NCCL": ExpandReduceNCCL,
    }
    default_implementation = "NCCL"

    # Object fields
    wcr = LambdaProperty(default='lambda a, b: a + b')
    root = SymbolicProperty(default=0,
                            allow_none=True,
                            desc="The gpu on which the receive buffer resides")

    def __init__(self,
                 wcr="lambda a, b: a + b",
                 root: symbolic.SymbolicType = 0,
                 debuginfo=None,
                 *args,
                 **kwargs):

        super().__init__(name='nccl_Reduce', *args, **kwargs)
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
        redtype = self.reduction_type

        wcrstr = str(redtype)
        wcrstr = wcrstr[wcrstr.find('.') + 1:]  # Skip "ReductionType."

        return f'nccl_Reduce({wcrstr})'

    @property
    def reduction_type(self):
        # Autodetect reduction type
        redtype = detect_reduction_type(self.wcr)
        if redtype not in nutil.NCCL_SUPPORTED_OPERATIONS:
            raise ValueError(
                'NCCL only supports sum, product, min and max operations.')
        return redtype

    def validate(self, sdfg: SDFG, state: SDFGState):
        redtype = self.reduction_type

        in_edges = state.in_edges(self)
        if len(in_edges) not in [1, 2]:
            raise ValueError("NCCL Reduce must have one or two inputs.")

        out_edges = state.out_edges(self)
        if len(out_edges) not in [1, 2]:
            raise ValueError("NCCL Reduce must have one or two outputs.")

    @property
    def free_symbols(self) -> Set[str]:
        result = super().free_symbols
        result.update(map(str, self.root.free_symbols))
        return result


@oprepo.replaces('dace.comm.nccl.reduce')
@oprepo.replaces('dace.comm.nccl.Reduce')
def nccl_reduce(pv: 'ProgramVisitor',
                sdfg: SDFG,
                state: SDFGState,
                redfunction: Callable[[Any, Any], Any],
                in_array: str,
                out_array: Union[str, None] = None,
                root: str = None,
                group_handle: str = None):

    inputs = {"_inbuffer"}
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

    libnode = Reduce(inputs=inputs, outputs=outputs, wcr=redfunction, root=root)

    if isinstance(group_handle, str):
        gh_memlet = Memlet.simple(gh_name, '0')
        if not gh_start:
            state.add_edge(gh_in, None, libnode, "_group_handle", gh_memlet)
        state.add_edge(libnode, "_group_handle", gh_out, None, gh_memlet)

    # If out_array is not specified, the operation will be in-place.
    if out_array is None:
        out_array = in_array

    # Add nodes
    in_node = state.add_read(in_array)
    out_node = state.add_write(out_array)

    # Connect nodes
    state.add_edge(in_node, None, libnode, '_inbuffer', Memlet(in_array))
    state.add_edge(libnode, '_outbuffer', out_node, None, Memlet(out_array))

    return []
