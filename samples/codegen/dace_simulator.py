# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
This sample shows how to extend the frontend and backend of DaCe by adding
NVIDIA Tensor Core storage type and code generation support.

Running the sample requires an NVIDIA GPU with Tensor Cores.
"""

import copy

# # General DaCe imports
# import dace
# from dace import data as dt
# import dace.codegen
# from dace.sdfg import nodes

# # Code generator imports and helpers
# from dace.codegen.targets.framecode import DaCeCodeGenerator
# from dace.codegen.targets.target import TargetCodeGenerator
# from dace.codegen.targets.cpp import cpp_array_expr, cpp_offset_expr

# # Frontend imports and helpers
# from dace.frontend.common.op_repository import replaces
# from dace.frontend.python.newast import ProgramVisitor

# # Transformations
# from dace.transformation.interstate import GPUTransformSDFG

# # Type hints
# from dace.sdfg.graph import MultiConnectorEdge
# from dace.sdfg.state import ControlFlowRegion, StateSubgraphView
# from dace.codegen.prettycode import CodeIOStream
# from dace.codegen.dispatcher import DefinedType
from typing import Any, List

# optimizations
# from dace.transformation.dataflow import (
#     DoubleBuffering,
#     MapCollapse,
#     MapExpansion,
#     MapReduceFusion,
#     StripMining,
#     InLocalStorage,
#     AccumulateTransient,
#     Vectorization,
# )

# Other imports
import itertools
import numpy as np
import torch
import tqdm


# fix random seed
np.random.seed(0)
torch.manual_seed(0)

############################################################################
# Tensor core code generator

# Define new storage types for Tensor Core locations
_TC_STORAGE_TYPES = ["TensorCore_A", "TensorCore_B", "TensorCore_Accumulator"]

TC_SIZE_M = 16
TC_SIZE_K = 16
TC_SIZE_N = 16


#####################################################################
# Data-centric optimization helpers



def frag_fill(frag, fill):
    # Define a tasklet with the appropriate input and output connectors.
    # Then we can directly emit CUDA for the tasklet.
    frag[:] = fill


def wmma(a_frag, b_frag, c_frag):
    # We do the same here as we did with frag_fill. Since c_frag is used
    # as both an input and an output, we specify two separate variables
    # to be passed to mma_sync and declare c_frag as an input to one and
    # an output to the other. This ensures proper dataflow.
    c_frag += a_frag @ b_frag


############################################################################
# Extend DaCe with new storage types and code generator



############################################################################
# Sample code that uses tensor cores
N = 32
TILE_M = 32
TILE_N = 32
TILE_K = 32

NUM_WARPS = TILE_M // TC_SIZE_M
NUM_ACCUMULATORS = TILE_N // TC_SIZE_N

def hgemm_tiled(
    A: torch.tensor ,
    B: torch.tensor,
    C: torch.tensor,
    N:int
):
    for tile_i_offset, tile_j_offset in zip(
        range(0,N,TILE_M), range(0,N,TILE_N)):
        # here I am a threadblock.x = tile_i, threadblock.y = tile_j

        # allocate shared memory buffers
        a_shmem = torch.tensor(
            [TILE_M, TILE_K]
        )
        b_shmem = torch.tensor(
            [TILE_K, TILE_N]
        )

        # allocate register tile for C accumulation:
        # - per threadblock: TILE_M x TILE_N
        # - per warp: TC_SIZE x TC_SIZE x (TILE_N // TC_SIZE)
        # - per thread:
        # TC_SIZE x TC_SIZE x (TILE_N // TC_SIZE) // 32
        atile = torch.tensor(
            [TC_SIZE_M, TC_SIZE_K]
        )
        btile = torch.tensor(
            [TC_SIZE_K, TC_SIZE_N]
        )
        ctile = torch.tensor(
            [NUM_ACCUMULATORS, TC_SIZE_M, TC_SIZE_N]
        )

        for frag_offset in range(0,NUM_ACCUMULATORS):
            frag_fill(ctile[frag_offset], 0.0)

        for step_k_offset in range(0,N,TILE_K):
            # load A and B tiles into shared mem
            a_shmem[:] = A[
                tile_i_offset : tile_i_offset + TILE_M,
                step_k_offset : step_k_offset + TILE_K,
            ]
            b_shmem[:] = B[
                step_k_offset : step_k_offset + TILE_K,
                tile_j_offset : tile_j_offset + TILE_N,
            ]

            for w_index, thread_index in zip(
                range(0,NUM_WARPS), range(0,32)
            ):
                # we have 8 warps per threadblock, 32 threads per warp
                # now I am a warp w in the threadblock.x = tile_i, threadblock.y = tile_j

                for TC_frag_offset in (
                    range(0,TILE_N,TC_SIZE_N)
                ):  # Warp map
                    for k_step in (
                        range(0,TILE_K,TC_SIZE_K)
                    ):
                        atile[:] = a_shmem[
                            w_index * TC_SIZE_M : (w_index + 1) * TC_SIZE_M,
                            k_step : k_step + TC_SIZE_K,
                        ]
                        btile[:] = b_shmem[
                            k_step : k_step + TC_SIZE_K,
                            TC_frag_offset : TC_frag_offset + TC_SIZE_N,
                        ]
                        wmma(atile, btile, ctile[TC_frag_offset // TC_SIZE_N])
                        # wmma(atile, btile, ctile[0])

        # finally, store it back to global memory
        for w_index, thread_index in zip(
            range(0,NUM_WARPS), range(0,32)
        ):
            for TC_frag_offset in (
                range(0,TILE_N,TC_SIZE_N)
            ):  # Warp map
                #     C[
                #     tile_i_offset
                #     + w_index * TC_SIZE_M : tile_i_offset
                #     + (w_index + 1) * TC_SIZE_M,
                #     tile_j_offset + TC_frag_offset : tile_j_offset + TC_frag_offset + TC_SIZE_N,
                # ] = ctile[0]
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
import time


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if __name__ == "__main__":


    N = 32
    # Prerequisite for sample: CUDA compute capability >= 70

    A = torch.randn(N, N, dtype=torch.float16, device=device)
    B = torch.randn(N, N, dtype=torch.float16, device=device)
    C = torch.randn(N, N, dtype=torch.float32, device=device)

    # A = torch.ones(N, N, dtype=torch.float16, device=device)
    # B = torch.ones(N, N, dtype=torch.float16, device=device)
    # C = torch.ones(N, N, dtype=torch.float32, device=device)


    # Transform the code to run on the GPU, while ensuring that the warp map
    # in the example runs within a single thread-block.
    # dace_matmul.apply_transformations(
    #     GPUTransformSDFG, options=dict(sequential_innermaps=False, register_trans=False)
    # )

    # dace_matmul.view()

    # exit()

    # smem_a = dace_matmul.arrays["a_shmem"]
    # smem_b = dace_matmul.arrays["b_shmem"]

    # def find_node_by_param(sdfg: dace.SDFG, pname: str) -> dace.nodes.MapEntry:
    #     return next(node for node in sdfg.data_nodes() if node.data == pname)

    # ktile = find_map_by_param(dace_matmul, "step_k_offset")
    # smem_a = find_node_by_param(dace_matmul, "a_shmem")
    # smem_b = find_node_by_param(dace_matmul, "b_shmem")

    # DoubleBuffering.apply_to(dace_matmul, map_entry=ktile, transient=smem_a)
    # DoubleBuffering.apply_to(dace_matmul, map_entry=ktile, transient=smem_b)

    hgemm_tiled(A=A, B=B, C=C, N=N)

    C_ref = A @ B



    diff = torch.linalg.norm(A @ B - C) / (N * N)
    if diff > 0.001:
        if N <= 64:
            import sys
            np.set_printoptions(precision=3, suppress=True, linewidth=3000, threshold=sys.maxsize)
            
            # set numpy printoptions to full linewidth. Currently, C print format is as follows:
#             [[-8.164  8.766 -6.466 ...  0.     0.     0.   ]
            #  [-4.277 -3.639  0.875 ...  0.     0.     0.   ]
            #  [ 9.489 -1.763 -2.317 ...  0.     0.     0.   ]
            #  ...
            #  [ 4.217 -2.018 -2.782 ...  0.     0.     0.   ]
            #  [-0.455 -3.837  0.513 ...  0.     0.     0.   ]
            #  [ 4.589  3.24   3.137 ...  0.     0.     0.   ]]
            # We don't want truncated format with ..., we want to print the full array.
            C_np = C.detach().cpu().numpy()
            C_ref_np = C_ref.detach().cpu().numpy()
            print(f"\nnumpy precision set. Type: {type(C_np)}\n")
            # print(f"DaCe matmul:\n{C_np}")
            # print(f"Reference matmul:\n{C_ref_np}")
            
            for i in range(0, N, N//2):
                line = ""
                for j in range(0, N, N//2):                                        
                   line += f"{C_np[i,j]:.3f}\t{C_ref_np[i,j]:.3f}\t\t\t"
                print(line)
                    # print(f"Reference matmul[{i}:{i+N//2},{j}://`{j+N//2}]:\n{C_ref[i:i+N//2,j:j+N//2]}")
    print("Difference:", diff)
    exit(1 if diff > 1e-3 else 0)
