# # Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
# from numbers import Number
# from typing import Sequence, Union

# import dace
# from dace import dtypes, symbolic
# from dace.frontend.common import op_repository as oprepo
# from dace.memlet import Memlet
# from dace.sdfg import SDFG, SDFGState

# import sympy as sp

# from dace.frontend.python.replacements import _define_local_scalar


# @oprepo.replaces('dace.nccl.Reduce')
# def _reduce(pv: 'ProgramVisitor',
#             sdfg: SDFG,
#             state: SDFGState,
#             in_buffer: str,
#             out_buffer: str,
#             op: str = 'ncclSum',
#             root: Union[str, symbolic.SymbolicType, Number] = 0):

#     from dace.libraries.nccl.nodes.reduce import Reduce

#     libnode = Reduce('nccl_Reduce_', op=op, root = root)
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


# @oprepo.replaces('dace.nccl.AllReduce')
# def _allreduce(pv: 'ProgramVisitor',
#             sdfg: SDFG,
#             state: SDFGState,
#             in_buffer: str,
#             out_buffer: str,
#             op: str = 'ncclSum',
#             root: Union[str, sp.Expr, Number] = 0):

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