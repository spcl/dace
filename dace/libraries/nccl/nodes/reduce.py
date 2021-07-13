# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from numbers import Number
from typing import Sequence, Union, Callable, Any
from dace import Config

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
from dace.frontend.python.replacements import _define_local_scalar



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
                             ' in global GPU memory or pinned CPU memory.')
        if output_data.storage not in [
                dtypes.StorageType.GPU_Global, dtypes.StorageType.CPU_Pinned
        ]:
            raise ValueError('Output of NCCL Reduce must either reside '
                             ' in global GPU memory or pinned CPU memory.')

        root = node.root

        redtype = node.reduction_type
        redtype = dtypes.NCCL_SUPPORTED_OPERATIONS[redtype]
        wcr_str = str(redtype)
        wcr_str = wcr_str[wcr_str.find('.') + 1:]  # Skip "NcclReductionType."

        nccl_dtype_str = nutil.Nccl_dtypes(input_data.dtype.base_type)
        count_str = "*".join(str(e) for e in input_dims)

        if input_data.dtype.veclen > 1:
            raise (NotImplementedError)

        
        
        code = f"""ncclReduce(_inbuffer, _outbuffer, {count_str}, {nccl_dtype_str}, {wcr_str}, {root}, __state->ncclCommunicators->at(__dace_cuda_device),  __dace_current_stream)"""
        if Config.get('compiler', 'build_type') == 'Debug':
            code = '''\ndace::nccl::CheckNcclError('''+code+''');\n'''
        else:
            code = '''\n''' + code + ''';\n'''

        if node.use_group_calls:
            code = """
            ncclGroupStart();""" + code
            code += """
            ncclGroupEnd();"""

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
    wcr = LambdaProperty(default='lambda a, b: a + b')
    # root = Property(default=0,
    #                 dtype=dace.int32,
    #                         allow_none=True,
    #                         desc="The gpu on which the receive buffer resides")
    root = SymbolicProperty(default=0,
                            allow_none=True,
                            desc="The gpu on which the receive buffer resides")

    use_group_calls = Property(dtype=bool,
                               default=False,
                               desc='True: use NCCL group calls.')

    def __init__(self,
                 wcr="lambda a, b: a + b",
                 root: symbolic.SymbolicType = 0,
                 use_group_calls: bool = False,
                 debuginfo=None,
                 *args,
                 **kwargs):

        super().__init__(name='nccl_Reduce',
                         *args,
                        #  inputs={"_inbuffer", "_root"},
                         inputs={"_inbuffer"},
                         outputs={"_outbuffer"},
                         **kwargs)
        self.wcr = wcr
        self.root = root
        self.use_group_calls = use_group_calls
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

        return 'nccl_Reduce ({op})'.format(op=wcrstr)

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
        if len(in_edges) not in [1,2]:
            raise ValueError("NCCL Reduce must have two inputs.")

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
                out_array: Union[str, None] = None,
                root: str = None,
                use_group_calls: bool = False):

    # If out_array is not specified, the operation will be in-place.
    if out_array is None:
        out_array = in_array

    # if isinstance(root, str) and root in sdfg.arrays.keys():
    #     root_node = state.add_read(root)
    # else:
    # root_name = _define_local_scalar(pv, sdfg, state, dace.int32, dtypes.StorageType.GPU_Global)
    # root_node = state.add_access(root_name)
    # root_tasklet = state.add_tasklet('_set_root_', {}, {'__out'},
    #                                     '__out = {}'.format(root))
    # state.add_edge(root_tasklet, '__out', root_node, None,
    #                 Memlet.simple(root_name, '0'))

    # Add nodes
    in_node = state.add_read(in_array)
    out_node = state.add_write(out_array)
    libnode = Reduce(redfunction, root=root, use_group_calls=use_group_calls)

    # Connect nodes
    state.add_edge(in_node, None, libnode, '_inbuffer', Memlet(in_array))
    state.add_edge(libnode, '_outbuffer', out_node, None, Memlet(out_array))
    # state.add_edge(root_node, None, libnode, '_root', Memlet(root_node.data))
    return []