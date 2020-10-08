"""
This sample shows how to extend the frontend and backend of DaCe by adding
NVIDIA Tensor Core storage type and code generation support.

Running the sample requires an NVIDIA GPU with Tensor Cores.
"""

import dace
from dace.sdfg import nodes
from dace.codegen.targets.framecode import DaCeCodeGenerator
from dace.codegen.targets.cpp import cpp_array_expr, cpp_offset_expr
from dace.frontend.common.op_repository import replaces
from dace.codegen.targets.target import TargetCodeGenerator
from dace.transformation.interstate import GPUTransformSDFG
import itertools
import numpy as np

############################################################################
# Tensor core code generator

_TC_STORAGE_TYPES = ['TensorCore_A', 'TensorCore_B', 'TensorCore_Accumulator']


class TensorCoreCodegen(TargetCodeGenerator):
    def __init__(self, frame_codegen: DaCeCodeGenerator, sdfg: dace.SDFG):
        self._frame = frame_codegen
        self._dispatcher = frame_codegen.dispatcher

        # Register array allocation/deallocation
        for dtype in _TC_STORAGE_TYPES:
            enum_type = dace.StorageType[dtype]
            self._dispatcher.register_array_dispatcher(enum_type, self)

        # Register copies to/from tensor cores
        gpu_storages = [
            dace.StorageType.GPU_Global, dace.StorageType.CPU_Pinned,
            dace.StorageType.GPU_Shared, dace.StorageType.GPU_Stack,
            dace.StorageType.Register
        ]
        for src_storage, dst_storage in itertools.product(
                _TC_STORAGE_TYPES, gpu_storages):
            src_storage = dace.StorageType[src_storage]
            self._dispatcher.register_copy_dispatcher(src_storage, dst_storage,
                                                      None, self)
            self._dispatcher.register_copy_dispatcher(dst_storage, src_storage,
                                                      None, self)

    def allocate_array(self, sdfg, dfg, state_id, node, function_stream,
                       callsite_stream):
        name = node.data
        nodedesc = node.desc(sdfg)

        # Based on the hardware, the total size must be 16^2
        assert nodedesc.total_size == 16 * 16
        # Majority is detected by the strides of the data
        maj = 'row' if nodedesc.strides[-1] == 1 else 'col'

        # Write a fragment based on the storage type
        if nodedesc.storage == dace.StorageType.TensorCore_Accumulator:
            callsite_stream.write(
                'wmma::fragment<wmma::accumulator, '
                '16, 16, 16, float> {};'.format(name), sdfg, state_id, node)
        else:
            callsite_stream.write(
                'wmma::fragment<wmma::matrix_{mat}, '
                '16, 16, 16, half, wmma::{maj}_major> '
                '{name};'.format(
                    mat=('a' if 'A' in nodedesc.storage.name else 'b'),
                    maj=maj,
                    name=name), sdfg, state_id, node)

    def initialize_array(self, sdfg, dfg, state_id, node, function_stream,
                         callsite_stream):
        pass  # Nothing to initialize (wmma::fragment is a C++ object)

    def deallocate_array(self, sdfg, dfg, state_id, node, function_stream,
                         callsite_stream):
        pass  # Nothing to deallocate (wmma::fragment is a C++ object)

    def copy_memory(self, sdfg, dfg, state_id, src_node, dst_node, edge,
                    function_stream, callsite_stream):
        # Obtain source and destination information, handle access<->tasklet
        # If copying from tensor core fragments to/from tasklets, we only need
        # to emit a reference, as the fragment contains the memory.
        src_desc = (src_node.desc(sdfg)
                    if isinstance(src_node, nodes.AccessNode) else None)
        if not src_desc:
            local_name = dfg.memlet_path(edge)[0].src_conn
            callsite_stream.write(
                'auto& %s = %s;' % (local_name, dst_node.data), sdfg, state_id,
                [src_node, dst_node])
            return

        dst_desc = (dst_node.desc(sdfg)
                    if isinstance(dst_node, nodes.AccessNode) else None)
        if not dst_desc:
            local_name = dfg.memlet_path(edge)[-1].dst_conn
            callsite_stream.write(
                'auto& %s = %s;' % (local_name, src_node.data), sdfg, state_id,
                [src_node, dst_node])
            return

        nontc_desc = (dst_desc
                      if 'TensorCore' in src_desc.storage.name else src_desc)
        nontc_node = (dst_node
                      if 'TensorCore' in src_desc.storage.name else src_node)
        # Majority is detected by the strides of the data
        row_major = True if nontc_desc.strides[-1] == 1 else False
        ###

        # Set non-tensor-core C++ expression based on memlet
        if edge.data.data == nontc_node.data:
            other_expr = cpp_array_expr(sdfg, edge.data)
        elif edge.data.other_subset is not None:
            offset_cppstr = cpp_offset_expr(nontc_desc, edge.data.other_subset)
            other_expr = '%s[%s]' % (nontc_node.data, offset_cppstr)
        else:
            other_expr = '%s[0]' % nontc_node.data
        ###

        # Emit copy code
        if 'TensorCore' in dst_desc.storage.name:
            # GPU memory to Tensor Cores
            callsite_stream.write(
                'wmma::load_matrix_sync({tc}, &{other}, '
                '{stride});'.format(
                    tc=dst_node.data,
                    other=other_expr,
                    stride=src_desc.strides[0 if row_major else 1]), sdfg,
                state_id, [src_node, dst_node])
        else:
            # Tensor Cores to GPU memory
            callsite_stream.write(
                'wmma::store_matrix_sync(&{other}, {tc}, '
                '{stride}, wmma::mem_{maj}_major);'.format(
                    tc=src_node.data,
                    other=other_expr,
                    maj='row' if row_major else 'col',
                    stride=dst_desc.strides[0 if row_major else 1]), sdfg,
                state_id, [src_node, dst_node])


############################################################################
# Tensor core frontend support


def _include_mma(sdfg: dace.SDFG):
    """ Add the Tensor Core includes into global code, if not included. """
    global_code = '''
#ifdef __CUDACC__
#include <mma.h>
using namespace nvcuda;
#endif
'''
    if ('cuda' not in sdfg.global_code
            or 'mma.h' not in sdfg.global_code['cuda'].code):
        sdfg.append_global_code(global_code, 'cuda')


@replaces('frag_fill')
def frag_fill(sdfg: dace.SDFG, state: dace.SDFGState, frag: str, fill: str):
    wnode = state.add_write(frag)
    tasklet = state.add_tasklet('fill',
                                set(), {'out'},
                                '''
      wmma::fill_fragment(out, %s);''' % fill,
                                language=dace.Language.CPP)

    state.add_edge(tasklet, 'out', wnode, None,
                   dace.Memlet.from_array(frag, wnode.desc(sdfg)))

    _include_mma(sdfg)

    # Function has no return value
    return []


@replaces('wmma')
def wmma(sdfg: dace.SDFG, state: dace.SDFGState, a_frag: str, b_frag: str,
         c_frag: str):
    anode = state.add_read(a_frag)
    bnode = state.add_read(b_frag)
    cnode = state.add_read(c_frag)
    tasklet = state.add_tasklet('wmma', {'afrag', 'bfrag'}, {'cfrag'},
                                '''
      wmma::mma_sync(cfrag, afrag, bfrag, cfrag);''',
                                language=dace.Language.CPP)

    state.add_edge(anode, None, tasklet, 'afrag',
                   dace.Memlet.from_array(a_frag, anode.desc(sdfg)))
    state.add_edge(bnode, None, tasklet, 'bfrag',
                   dace.Memlet.from_array(b_frag, bnode.desc(sdfg)))
    state.add_edge(tasklet, 'cfrag', cnode, None,
                   dace.Memlet.from_array(c_frag, cnode.desc(sdfg)))

    _include_mma(sdfg)

    # Function has no return value
    return []


############################################################################
# Extend DaCe with new storage types and code generator


def extend_dace():
    # Register code generator
    TargetCodeGenerator.register(TensorCoreCodegen, name='tensorcore')

    # Register storage types
    for dtype in _TC_STORAGE_TYPES:
        dace.StorageType.register(dtype)


############################################################################
# Sample code that uses tensor cores

N = dace.symbol('N')


@dace.program
def tc_hgemm(A: dace.float16[N, N], B: dace.float16[N, N], C: dace.float32[N,
                                                                           N]):
    for i, j in dace.map[0:N:16, 0:N:16]:
        ctile = dace.ndarray([16, 16],
                             dtype=dace.float32,
                             storage=dace.StorageType.TensorCore_Accumulator)
        frag_fill(ctile, 0.0)
        for k in range(0, N, 16):
            atile = dace.ndarray([16, 16],
                                 dtype=dace.float16,
                                 storage=dace.StorageType.TensorCore_A)
            btile = dace.ndarray([16, 16],
                                 dtype=dace.float16,
                                 storage=dace.StorageType.TensorCore_B)
            atile << A[i:i + 16, k:k + 16]
            btile << B[k:k + 16, j:j + 16]
            wmma(atile, btile, ctile)

        ctile >> C[i:i + 16, j:j + 16]


############################################################################
# Main function

if __name__ == '__main__':
    extend_dace()
    A = np.random.rand(1024, 1024).astype(np.float16)
    B = np.random.rand(1024, 1024).astype(np.float16)
    C = np.random.rand(1024, 1024).astype(np.float32)

    sdfg = tc_hgemm.to_sdfg()
    sdfg.apply_transformations(GPUTransformSDFG)
    sdfg(A=A, B=B, C=C, N=1024)

    diff = np.linalg.norm(A @ B - C) / (1024 * 1024)
    print('Difference:', diff)
    exit(1 if diff > 1e-3 else 0)
