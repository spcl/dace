# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
This sample shows how to extend the frontend and backend of DaCe by adding
NVIDIA Tensor Core storage type and code generation support.

Running the sample requires an NVIDIA GPU with Tensor Cores.
"""

import copy

# General DaCe imports
import dace
from dace import data as dt
import dace.codegen
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
from dace.sdfg.state import ControlFlowRegion, StateSubgraphView
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.dispatcher import DefinedType
from typing import Any, List

# Other imports
import itertools
import numpy as np
import tqdm

############################################################################
# Tensor core code generator

# Define new storage types for Tensor Core locations
_TC_STORAGE_TYPES = ["TensorCore_A", "TensorCore_B", "TensorCore_Accumulator"]

TC_SIZE_M = 16
TC_SIZE_K = 16
TC_SIZE_N = 16
TC_A = dace.float16[TC_SIZE_M, TC_SIZE_K]
TC_B = dace.float16[TC_SIZE_K, TC_SIZE_N]
TC_accumulator = dace.float32[TC_SIZE_M, TC_SIZE_N]


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
            dace.StorageType.GPU_Global,
            dace.StorageType.CPU_Pinned,
            dace.StorageType.GPU_Shared,
            dace.StorageType.Register,
        ]
        for src_storage, dst_storage in itertools.product(
            _TC_STORAGE_TYPES, gpu_storages
        ):
            src_storage = dace.StorageType[src_storage]
            self._dispatcher.register_copy_dispatcher(
                src_storage, dst_storage, None, self
            )
            self._dispatcher.register_copy_dispatcher(
                dst_storage, src_storage, None, self
            )

    def allocate_array(
        self,
        sdfg: dace.SDFG,
        cfg: ControlFlowRegion,
        dfg: StateSubgraphView,
        state_id: int,
        node: nodes.AccessNode,
        nodedesc: dt.Array,
        function_stream: CodeIOStream,
        declaration_stream: CodeIOStream,
        allocation_stream: CodeIOStream,
    ):
        # Make sure the codegen includes the appropriate header files
        _include_mma(sdfg)

        name = node.data

        # Majority is detected by the strides of the data
        maj = "row" if nodedesc.strides[-1] == 1 else "col"

        # The tensor core fragment size is at least two dimensions
        assert len(nodedesc.shape) >= 2

        size = "[1]"
        if len(nodedesc.shape) > 2:
            total_tc_size = nodedesc.shape[-2] * nodedesc.shape[-1]
            size = f"[{nodedesc.total_size // total_tc_size}]"
            nodedesc.strides = [
                s // total_tc_size for s in nodedesc.strides[:-2]
            ] + list(nodedesc.strides[-2:])

        # Write a fragment based on the storage type
        if nodedesc.storage == dace.StorageType.TensorCore_Accumulator:
            ctype = f"wmma::fragment<wmma::accumulator, {TC_SIZE_M}, {TC_SIZE_K}, {TC_SIZE_N}, {TC_accumulator.dtype}>"
            declaration_stream.write(f"{ctype} {name}{size};", cfg, state_id, node)
        else:
            mat = "a" if "A" in nodedesc.storage.name else "b"
            ctype = f"wmma::fragment<wmma::matrix_{mat}, {TC_SIZE_M}, {TC_SIZE_K}, {TC_SIZE_N}, {TC_A.dtype}, wmma::{maj}_major>"
            declaration_stream.write(f"{ctype} {name}{size};", cfg, state_id, node)

        # Add the ctype to defined_vars so that the codegen can properly pass
        # fragments to functions as an object reference.
        self._dispatcher.defined_vars.add(name, DefinedType.Pointer, ctype + "*")

    def deallocate_array(
        self,
        sdfg: dace.SDFG,
        cfg: ControlFlowRegion,
        dfg: StateSubgraphView,
        state_id: int,
        node: nodes.AccessNode,
        nodedesc: dt.Array,
        function_stream: CodeIOStream,
        callsite_stream: CodeIOStream,
    ):
        pass  # Nothing to deallocate (wmma::fragment is a C++ object)

    def copy_memory(
        self,
        sdfg: dace.SDFG,
        cfg: ControlFlowRegion,
        dfg: StateSubgraphView,
        state_id: int,
        src_node: nodes.Node,
        dst_node: nodes.Node,
        edge: MultiConnectorEdge,
        function_stream: CodeIOStream,
        callsite_stream: CodeIOStream,
    ) -> None:
        # Obtain source and destination information, handle access<->tasklet
        # If copying from tensor core fragments to/from tasklets, we only need
        # to emit a reference, as the fragment contains the memory.
        src_desc = (
            src_node.desc(sdfg) if isinstance(src_node, nodes.AccessNode) else None
        )
        # Tasklet -> Array
        if not src_desc:
            local_name = dfg.memlet_path(edge)[0].src_conn
            ptr = self._make_tensorcore_pointer(sdfg, edge.data.data, edge.data.subset)
            callsite_stream.write(
                "auto& %s = %s;" % (local_name, ptr),
                cfg,
                state_id,
                [src_node, dst_node],
            )
            return

        dst_desc = (
            dst_node.desc(sdfg) if isinstance(dst_node, nodes.AccessNode) else None
        )
        # Array -> Tasklet
        if not dst_desc:
            local_name = dfg.memlet_path(edge)[-1].dst_conn
            ptr = self._make_tensorcore_pointer(sdfg, edge.data.data, edge.data.subset)

            callsite_stream.write(
                "auto& %s = %s;" % (local_name, ptr),
                cfg,
                state_id,
                [src_node, dst_node],
            )
            return

        nontc_desc = dst_desc if "TensorCore" in src_desc.storage.name else src_desc
        nontc_node = dst_node if "TensorCore" in src_desc.storage.name else src_node
        tc_node = src_node if "TensorCore" in src_desc.storage.name else dst_node

        # Majority is detected by the strides of the data
        row_major = True if nontc_desc.strides[-1] == 1 else False
        #####################################################################
        # Set the tensor core C++ expression based on memlet
        if edge.data.data == nontc_node.data:
            ptr = self._make_tensorcore_pointer(
                sdfg, tc_node.data, edge.data.other_subset
            )
        else:
            ptr = self._make_tensorcore_pointer(sdfg, tc_node.data, edge.data.subset)

        # Set non-tensor-core C++ expression based on memlet
        if edge.data.data == nontc_node.data:
            other_expr = cpp_array_expr(sdfg, edge.data)
        elif edge.data.other_subset is not None:
            offset_cppstr = cpp_offset_expr(nontc_desc, edge.data.other_subset)
            other_expr = "%s[%s]" % (nontc_node.data, offset_cppstr)
        else:
            other_expr = "%s[0]" % nontc_node.data
        #####################################################################

        # Emit copy code
        if "TensorCore" in dst_desc.storage.name:
            # GPU memory to Tensor Cores
            callsite_stream.write(
                "wmma::load_matrix_sync({tc}, &{other}, "
                "{stride});".format(
                    tc=ptr,
                    other=other_expr,
                    stride=src_desc.strides[0 if row_major else 1],
                ),
                cfg,
                state_id,
                [src_node, dst_node],
            )
        else:
            # Tensor Cores to GPU memory
            callsite_stream.write(
                "wmma::store_matrix_sync(&{other}, {tc}, "
                "{stride}, wmma::mem_{maj}_major);".format(
                    tc=ptr,
                    other=other_expr,
                    maj="row" if row_major else "col",
                    stride=dst_desc.strides[0 if row_major else 1],
                ),
                cfg,
                state_id,
                [src_node, dst_node],
            )

    def define_out_memlet(
        self,
        sdfg: dace.SDFG,
        cfg: ControlFlowRegion,
        dfg: StateSubgraphView,
        state_id: int,
        src_node: nodes.Node,
        dst_node: nodes.Node,
        edge: MultiConnectorEdge,
        function_stream: CodeIOStream,
        callsite_stream: CodeIOStream,
    ):
        # Output memlets that are directed at WMMA fragments can use the "auto"
        # keyword for simplicity.
        ptr = self._make_tensorcore_pointer(sdfg, edge.data.data, edge.data.subset)
        callsite_stream.write(f"auto& {edge.src_conn} = {ptr};")

    def _make_tensorcore_pointer(
        self, sdfg: dace.SDFG, tc_name: str, subset: dace.subsets.Subset
    ):
        if len(subset) > 2:
            desc = sdfg.arrays[tc_name]
            offset_cppstr = cpp_offset_expr(desc, subset)
        else:
            offset_cppstr = "0"
        return f"{tc_name}[{offset_cppstr}]"


############################################################################
# Tensor core frontend support

# Here we introduce two new functions that can be used in the Python frontend:
#   * frag_fill: Fill a TC fragment with a value (usually used to zero memory)
#   * wmma: The actual TC matrix-multiplication-addition


def _include_mma(sdfg: dace.SDFG):
    """
    Add the Tensor Core includes into global code, if not already included.
    """

    global_code = """
#ifdef __CUDACC__
#include <mma.h>
using namespace nvcuda;
#endif
"""
    # We append the global code only to the CUDA-generated file. Every
    # file generated by each code generators creates an entry in the SDFG
    # global code dictionary. The `None` key refers to global code that will
    # be added to every generated file.
    if "cuda" not in sdfg.global_code or "mma.h" not in sdfg.global_code["cuda"].code:
        sdfg.append_global_code(global_code, "cuda")


def frag_fill(frag, fill):
    # Define a tasklet with the appropriate input and output connectors.
    # Then we can directly emit CUDA for the tasklet.
    with dace.tasklet(dace.Language.CPP):
        val << fill
        out >> frag
        """
        wmma::fill_fragment(out, val);
        """


def wmma(a_frag, b_frag, c_frag):
    # We do the same here as we did with frag_fill. Since c_frag is used
    # as both an input and an output, we specify two separate variables
    # to be passed to mma_sync and declare c_frag as an input to one and
    # an output to the other. This ensures proper dataflow.
    with dace.tasklet(dace.Language.CPP):
        afrag << a_frag
        bfrag << b_frag
        cfrag << c_frag
        dfrag >> c_frag
        """
        wmma::mma_sync(dfrag, afrag, bfrag, cfrag);
        """


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
    TargetCodeGenerator.register(TensorCoreCodegen, name="tensorcore")


############################################################################
# Sample code that uses tensor cores

N = dace.symbol("N")

TILE_M = 256
TILE_N = 128
TILE_K = 32

NUM_WARPS = TILE_M // TC_SIZE_M
NUM_ACCUMULATORS = TILE_N // TC_SIZE_N


@dace.program
def hgemm_tiled(
    A: dace.float16[N, N] @ dace.StorageType.GPU_Global,
    B: dace.float16[N, N] @ dace.StorageType.GPU_Global,
    C: dace.float32[N, N] @ dace.StorageType.GPU_Global,
):
    for tile_i_offset, tile_j_offset in dace.map[0:N:TILE_M, 0:N:TILE_N]:
        # here I am a threadblock.x = tile_i, threadblock.y = tile_j

        # allocate shared memory buffers
        a_shmem = dace.ndarray(
            [TILE_M, TILE_K],
            dtype=dace.float16,
            storage=dace.StorageType.GPU_Shared,
        )
        b_shmem = dace.ndarray(
            [TILE_K, TILE_N],
            dtype=dace.float16,
            storage=dace.StorageType.GPU_Shared,
        )

        # allocate register tile for C accumulation:
        # - per threadblock: TILE_M x TILE_N
        # - per warp: TC_SIZE x TC_SIZE x (TILE_N // TC_SIZE)
        # - per thread:
        # TC_SIZE x TC_SIZE x (TILE_N // TC_SIZE) // 32
        ctile = dace.ndarray(
            [NUM_ACCUMULATORS, TC_SIZE_M, TC_SIZE_N],
            dtype=dace.float32,
            storage=dace.StorageType.TensorCore_Accumulator,
        )

        for frag_offset in dace.map[0:NUM_ACCUMULATORS] @ dace.ScheduleType.Sequential:
            frag_fill(ctile[frag_offset], 0.0)

        for step_k_offset in range(0, N, TILE_K):
            # load A and B tiles into shared mem
            a_shmem[:] = A[
                tile_i_offset : tile_i_offset + TILE_M,
                step_k_offset : step_k_offset + TILE_K,
            ]
            b_shmem[:] = B[
                step_k_offset : step_k_offset + TILE_K,
                tile_j_offset : tile_j_offset + TILE_N,
            ]

            for w_index, thread_index in (
                dace.map[0:NUM_WARPS, 0:32] @ dace.ScheduleType.GPU_ThreadBlock
            ):
                # we have 8 warps per threadblock, 32 threads per warp
                # now I am a warp w in the threadblock.x = tile_i, threadblock.y = tile_j

                for TC_frag_offset in (
                    dace.map[0:TILE_N:TC_SIZE_N] @ dace.ScheduleType.Sequential
                ):  # Warp map
                    for k_step in (
                        dace.map[0:TILE_K:TC_SIZE_K] @ dace.ScheduleType.Sequential
                    ):
                        atile = dace.ndarray(
                            [TC_SIZE_M, TC_SIZE_K],
                            dtype=dace.float16,
                            storage=dace.StorageType.TensorCore_A,
                        )
                        btile = dace.ndarray(
                            [TC_SIZE_K, TC_SIZE_N],
                            dtype=dace.float16,
                            storage=dace.StorageType.TensorCore_B,
                        )
                        atile[:] = a_shmem[
                            w_index * TC_SIZE_M : (w_index + 1) * TC_SIZE_M,
                            k_step : k_step + TC_SIZE_K,
                        ]
                        btile[:] = b_shmem[
                            k_step : k_step + TC_SIZE_K,
                            TC_frag_offset : TC_frag_offset + TC_SIZE_N,
                        ]
                        wmma(atile, btile, ctile[TC_frag_offset // TC_SIZE_N])

        # finally, store it back to global memory
        for w_index, thread_index in (
            dace.map[0:NUM_WARPS, 0:32] @ dace.ScheduleType.GPU_ThreadBlock
        ):
            for TC_frag_offset in (
                dace.map[0:TILE_N:TC_SIZE_N] @ dace.ScheduleType.Sequential
            ):  # Warp map
                C[
                    tile_i_offset
                    + w_index * TC_SIZE_M : tile_i_offset
                    + (w_index + 1) * TC_SIZE_M,
                    tile_j_offset
                    + TC_frag_offset : tile_j_offset
                    + TC_frag_offset
                    + TC_SIZE_N,
                ] = ctile[TC_frag_offset // TC_SIZE_N]


############################################################################
# Main function
import torch
import time


def benchmark_matmul(matmul_func, M, N, K, num_iterations=100):
    # Create random matrices
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(K, N, dtype=torch.float16, device="cuda")
    C = torch.randn(M, N, dtype=torch.float32, device="cuda")

    # Warm-up run
    if type(matmul_func) == dace.codegen.CompiledSDFG:
        matmul_func(A=A, B=B, C=C, N=M)
    else:
        matmul_func(A, B)

    # Synchronize CUDA
    torch.cuda.synchronize()

    # Benchmark
    start_time = time.perf_counter()
    for _ in tqdm.trange(num_iterations):
        if type(matmul_func) == dace.codegen.CompiledSDFG:
            matmul_func(A=A, B=B, C=C, N=M)
        else:
            matmul_func(A, B)
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    # Calculate total runtime and average runtime per iteration
    total_runtime = end_time - start_time
    avg_runtime = total_runtime / num_iterations

    # Calculate FLOPs
    flops = 2 * M * N * K
    tflops = (flops * num_iterations) / (total_runtime * 1e12)
    print(f"Total runtime: {avg_runtime:.4f} s, performance: {tflops:.2f} TFLOP/s")
    return total_runtime, avg_runtime, tflops


if __name__ == "__main__":
    extend_dace()

    N = 4096

    # Prerequisite for sample: CUDA compute capability >= 70
    dace.Config.set("compiler", "cuda", "cuda_arch", value="80")

    A = torch.randn(N, N, dtype=torch.float16, device="cuda")
    B = torch.randn(N, N, dtype=torch.float16, device="cuda")
    C = torch.randn(N, N, dtype=torch.float32, device="cuda")

    dace_matmul: dace.SDFG = hgemm_tiled.to_sdfg(simplify=True)

    # dace_matmul.view()
    #
    # exit()
    # Transform the code to run on the GPU, while ensuring that the warp map
    # in the example runs within a single thread-block.
    dace_matmul.apply_transformations(
        GPUTransformSDFG, options=dict(sequential_innermaps=False)
    )

    compiled = dace_matmul.compile()
    # print(f"\n\ntype of dace: {type(dace_matmul)}\n\n")

    A = torch.tensor(A, device="cuda", dtype=torch.float16)
    B = torch.tensor(B, device="cuda", dtype=torch.float16)
    C = torch.tensor(C, device="cuda", dtype=torch.float32)

    # torch.cuda.synchronize()

    iters = 2
    # print("")
    benchmark_matmul(compiled, N, N, N, num_iterations=iters)

    benchmark_matmul(torch.matmul, N, N, N, num_iterations=iters)

    compiled(A=A, B=B, C=C, N=N)

    diff = torch.linalg.norm(A @ B - C) / (N * N)
    print("Difference:", diff)
    exit(1 if diff > 1e-3 else 0)
