# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
This sample shows how to extend the frontend and backend of DaCe by adding
NVIDIA Tensor Core storage type and code generation support.

Running the sample requires an NVIDIA GPU with Tensor Cores.
"""

# General DaCe imports
import dace
from dace import data as dt
from dace.sdfg import nodes

# Code generator imports and helpers
from dace.codegen.targets.framecode import DaCeCodeGenerator
from dace.codegen.targets.target import TargetCodeGenerator
from dace.codegen.targets.cpp import cpp_array_expr, cpp_offset_expr

# Frontend imports and helpers
from dace.frontend.common.op_repository import replaces
from dace.frontend.python.newast import ProgramVisitor

# Transformations
from dace.transformation.interstate import GPUTransformSDFG

# Type hints
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import StateSubgraphView
from dace.codegen.prettycode import CodeIOStream
from typing import Any, List

# Other imports
import itertools
import numpy as np

############################################################################
# Tensor core code generator

# Define new storage types for Tensor Core locations
_TC_STORAGE_TYPES = ['TensorCore_A', 'TensorCore_B', 'TensorCore_Accumulator']


class TensorCoreCodegen(TargetCodeGenerator):
    """ 
    The code generator target for Tensor Core code. This class contains 
    dispatchers for memlet paths from/to the tensor core storage locations.
    
    To do so, the code generator must register itself with the SDFG code
    generator dispatcher (in `__init__`) for:
        1. Every allocation/deallocation of storage locations.
        2. Every copy between TC types and other (legitimate) storages.

    The allocation dispatcher requires the code generator to implement the
    `{allocate, deallocate, initialize}_array` methods, whereas the copy
    dispatcher requires the `copy_memory` and `define_out_memlet` methods.
    """
    def __init__(self, frame_codegen: DaCeCodeGenerator, sdfg: dace.SDFG):
        self._frame = frame_codegen
        self._dispatcher = frame_codegen.dispatcher

        # Register array allocation/deallocation
        for dtype in _TC_STORAGE_TYPES:
            enum_type = dace.StorageType[dtype]
            self._dispatcher.register_array_dispatcher(enum_type, self)

        # Register copies to/from tensor cores
        gpu_storages = [
            dace.StorageType.GPU_Global, dace.StorageType.CPU_Pinned, dace.StorageType.GPU_Shared,
            dace.StorageType.Register
        ]
        for src_storage, dst_storage in itertools.product(_TC_STORAGE_TYPES, gpu_storages):
            src_storage = dace.StorageType[src_storage]
            self._dispatcher.register_copy_dispatcher(src_storage, dst_storage, None, self)
            self._dispatcher.register_copy_dispatcher(dst_storage, src_storage, None, self)

    def allocate_array(self, sdfg: dace.SDFG, dfg: StateSubgraphView, state_id: int, node: nodes.AccessNode,
                       nodedesc: dt.Array, function_stream: CodeIOStream, declaration_stream: CodeIOStream,
                       allocation_stream: CodeIOStream):
        name = node.data

        # Based on the hardware, the total size must be 16^2
        assert nodedesc.total_size == 16 * 16
        # Majority is detected by the strides of the data
        maj = 'row' if nodedesc.strides[-1] == 1 else 'col'

        # Write a fragment based on the storage type
        if nodedesc.storage == dace.StorageType.TensorCore_Accumulator:
            declaration_stream.write('wmma::fragment<wmma::accumulator, '
                                     '16, 16, 16, float> {};'.format(name), sdfg, state_id, node)
        else:
            declaration_stream.write(
                'wmma::fragment<wmma::matrix_{mat}, '
                '16, 16, 16, half, wmma::{maj}_major> '
                '{name};'.format(mat=('a' if 'A' in nodedesc.storage.name else 'b'), maj=maj, name=name), sdfg,
                state_id, node)

    def deallocate_array(self, sdfg: dace.SDFG, dfg: StateSubgraphView, state_id: int, node: nodes.AccessNode,
                         nodedesc: dt.Array, function_stream: CodeIOStream, callsite_stream: CodeIOStream):
        pass  # Nothing to deallocate (wmma::fragment is a C++ object)

    def copy_memory(self, sdfg: dace.SDFG, dfg: StateSubgraphView, state_id: int, src_node: nodes.Node,
                    dst_node: nodes.Node, edge: MultiConnectorEdge, function_stream: CodeIOStream,
                    callsite_stream: CodeIOStream):
        # Obtain source and destination information, handle access<->tasklet
        # If copying from tensor core fragments to/from tasklets, we only need
        # to emit a reference, as the fragment contains the memory.
        src_desc = (src_node.desc(sdfg) if isinstance(src_node, nodes.AccessNode) else None)
        # Tasklet -> Array
        if not src_desc:
            local_name = dfg.memlet_path(edge)[0].src_conn
            callsite_stream.write('auto& %s = %s;' % (local_name, dst_node.data), sdfg, state_id, [src_node, dst_node])
            return

        dst_desc = (dst_node.desc(sdfg) if isinstance(dst_node, nodes.AccessNode) else None)
        # Array -> Tasklet
        if not dst_desc:
            local_name = dfg.memlet_path(edge)[-1].dst_conn
            callsite_stream.write('auto& %s = %s;' % (local_name, src_node.data), sdfg, state_id, [src_node, dst_node])
            return

        nontc_desc = (dst_desc if 'TensorCore' in src_desc.storage.name else src_desc)
        nontc_node = (dst_node if 'TensorCore' in src_desc.storage.name else src_node)

        # Majority is detected by the strides of the data
        row_major = True if nontc_desc.strides[-1] == 1 else False
        #####################################################################

        # Set non-tensor-core C++ expression based on memlet
        if edge.data.data == nontc_node.data:
            other_expr = cpp_array_expr(sdfg, edge.data)
        elif edge.data.other_subset is not None:
            offset_cppstr = cpp_offset_expr(nontc_desc, edge.data.other_subset)
            other_expr = '%s[%s]' % (nontc_node.data, offset_cppstr)
        else:
            other_expr = '%s[0]' % nontc_node.data
        #####################################################################

        # Emit copy code
        if 'TensorCore' in dst_desc.storage.name:
            # GPU memory to Tensor Cores
            callsite_stream.write(
                'wmma::load_matrix_sync({tc}, &{other}, '
                '{stride});'.format(tc=dst_node.data, other=other_expr, stride=src_desc.strides[0 if row_major else 1]),
                sdfg, state_id, [src_node, dst_node])
        else:
            # Tensor Cores to GPU memory
            callsite_stream.write(
                'wmma::store_matrix_sync(&{other}, {tc}, '
                '{stride}, wmma::mem_{maj}_major);'.format(tc=src_node.data,
                                                           other=other_expr,
                                                           maj='row' if row_major else 'col',
                                                           stride=dst_desc.strides[0 if row_major else 1]), sdfg,
                state_id, [src_node, dst_node])

    def define_out_memlet(self, sdfg: dace.SDFG, dfg: StateSubgraphView, state_id: int, src_node: nodes.Node,
                          dst_node: nodes.Node, edge: MultiConnectorEdge, function_stream: CodeIOStream,
                          callsite_stream: CodeIOStream):
        # Output memlets that are directed at WMMA fragments can use the "auto"
        # keyword for simplicity.
        callsite_stream.write(f'auto& {edge.src_conn} = {edge.data.data};')


############################################################################
# Tensor core frontend support

# Here we introduce two new functions that can be used in the Python frontend:
#   * frag_fill: Fill a TC fragment with a value (usually used to zero memory)
#   * wmma: The actual TC matrix-multiplication-addition


def _include_mma(sdfg: dace.SDFG):
    """ 
    Add the Tensor Core includes into global code, if not already included. 
    """

    global_code = '''
#ifdef __CUDACC__
#include <mma.h>
using namespace nvcuda;
#endif
'''
    # We append the global code only to the CUDA-generated file. Every
    # file generated by each code generators creates an entry in the SDFG
    # global code dictionary. The `None` key refers to global code that will
    # be added to every generated file.
    if ('cuda' not in sdfg.global_code or 'mma.h' not in sdfg.global_code['cuda'].code):
        sdfg.append_global_code(global_code, 'cuda')


@replaces('frag_fill')
def frag_fill(pv: ProgramVisitor, sdfg: dace.SDFG, state: dace.SDFGState, frag: str, fill: Any) -> List[str]:
    # Replacement functions receive the SDFG and the current state as the first
    # two arguments, followed by all the other arguments. Here we treat them as
    # two strings representing the array name to fill and what to fill it with.

    # NOTE: If a slice is used in the `frag` argument, the Python frontend
    # automatically creates a new array for it, and uses the correct string as
    # the argument.
    wnode = state.add_write(frag)
    tasklet = state.add_tasklet('fill',
                                set(), {'out'},
                                '''
      wmma::fill_fragment(out, %s);''' % fill,
                                language=dace.Language.CPP)

    state.add_edge(tasklet, 'out', wnode, None, dace.Memlet.from_array(frag, wnode.desc(sdfg)))

    _include_mma(sdfg)

    # Function has no return value
    return []


@replaces('wmma')
def wmma(pv: ProgramVisitor, sdfg: dace.SDFG, state: dace.SDFGState, a_frag: str, b_frag: str,
         c_frag: str) -> List[str]:
    # Implemented similarly to `frag_fill`, but with inputs and outputs.
    anode = state.add_read(a_frag)
    bnode = state.add_read(b_frag)
    cnode = state.add_write(c_frag)
    tasklet = state.add_tasklet('wmma', {'afrag', 'bfrag'}, {'cfrag'},
                                '''
      wmma::mma_sync(cfrag, afrag, bfrag, cfrag);''',
                                language=dace.Language.CPP)

    state.add_edge(anode, None, tasklet, 'afrag', dace.Memlet.from_array(a_frag, anode.desc(sdfg)))
    state.add_edge(bnode, None, tasklet, 'bfrag', dace.Memlet.from_array(b_frag, bnode.desc(sdfg)))
    state.add_edge(tasklet, 'cfrag', cnode, None, dace.Memlet.from_array(c_frag, cnode.desc(sdfg)))

    _include_mma(sdfg)

    # Function has no return value
    return []


############################################################################
# Extend DaCe with new storage types and code generator


# We call this function before using the new types in the system, and it will
# extend the `dace.StorageType` enumeration with the new Tensor Core types,
# as well as register the new code generator with the SDFG code dispatcher.
def extend_dace():
    # Register storage types
    for dtype in _TC_STORAGE_TYPES:
        dace.StorageType.register(dtype)

    # Register code generator
    TargetCodeGenerator.register(TensorCoreCodegen, name='tensorcore')


############################################################################
# Sample code that uses tensor cores

N = dace.symbol('N')


@dace.program
def hgemm(A: dace.float16[N, N], B: dace.float16[N, N], C: dace.float32[N, N]):
    for i, j in dace.map[0:N:16, 0:N:16]:  # Thread-block map
        for _ in dace.map[0:32]:  # Warp map
            ctile = dace.ndarray([16, 16], dtype=dace.float32, storage=dace.StorageType.TensorCore_Accumulator)
            frag_fill(ctile, 0.0)
            for k in range(0, N, 16):
                atile = dace.ndarray([16, 16], dtype=dace.float16, storage=dace.StorageType.TensorCore_A)
                btile = dace.ndarray([16, 16], dtype=dace.float16, storage=dace.StorageType.TensorCore_B)
                atile[:] = A[i:i + 16, k:k + 16]
                btile[:] = B[k:k + 16, j:j + 16]
                wmma(atile, btile, ctile)

            C[i:i + 16, j:j + 16] = ctile


############################################################################
# Main function

if __name__ == '__main__':
    extend_dace()

    # Prerequisite for sample: CUDA compute capability >= 70
    dace.Config.set('compiler', 'cuda', 'cuda_arch', value='70')

    A = np.random.rand(1024, 1024).astype(np.float16)
    B = np.random.rand(1024, 1024).astype(np.float16)
    C = np.random.rand(1024, 1024).astype(np.float32)

    sdfg: dace.SDFG = hgemm.to_sdfg()

    # Transform the code to run on the GPU, while ensuring that the warp map
    # in the example runs within a single thread-block.
    sdfg.apply_transformations(GPUTransformSDFG, options=dict(sequential_innermaps=False))

    sdfg(A=A, B=B, C=C, N=1024)

    diff = np.linalg.norm(A @ B - C) / (1024 * 1024)
    print('Difference:', diff)
    exit(1 if diff > 1e-3 else 0)
