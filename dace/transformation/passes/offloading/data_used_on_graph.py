import dace
import numpy as np
from typing import Dict, Set


# TODO: Implement by using DaCe's built-in read/write set analysis.
# You probably need to extend it 
# DaCe provides:
#     read_set, write_set = state.read_and_write_sets()
#
# This returns two sets of container names (strings) per state.
#
# For this analysis we need more than names: we need to know the
# *storage type* of each accessed container, and whether the access
# happens inside a GPU-scheduled map or a sequential/CPU scope.


def arrays_accessed_per_state(sdfg: dace.SDFG) -> Dict[dace.SDFGState, Dict[str, Set[str]]]:
    return dict()


@dace.program
def example(A: dace.float64[10], B: dace.float64[10], C: dace.float64[10],
            D: dace.float64[10], N: int, i: int):
    if i < N / 2:
        for j in range(1, N):
            A[j] = A[j - 1] + B[j] + C[j]
        for j in dace.map[0:N // 2] @ dace.dtypes.ScheduleType.GPU_Device:
            D[j] = B[j] * 2
    else:
        for j in dace.map[0:N] @ dace.dtypes.ScheduleType.GPU_Device:
            A[j] = B[j] * C[j]
        for j in dace.map[0:N // 2] @ dace.dtypes.ScheduleType.GPU_Device:
            D[j] = B[j] * 2


def test_example() -> dace.SDFG:
    # Validation intentionally skipped: the program declares A, B, C, D
    # with default (CPU) storage, but GPU_Device maps access them.
    # This mismatch is irrelevant for the *analysis* we're building —
    # the goal is to determine which arrays need to be on which device,
    # not to enforce it.
    sdfg = example.to_sdfg(validate=False)

    # NOTE: The SDFG viewer may fail to render GPU-scheduled maps when
    # arrays have CPU storage. To inspect the graph visually, save a
    # version without GPU schedule annotations:
    #
    #   @dace.program
    #   def example(...):
    #       if i < N / 2:
    #           for j in range(1, N):        # sequential (CPU)
    #               A[j] = A[j-1] + B[j] + C[j]
    #           for j in dace.map[0:N//2]:   # no GPU annotation
    #               D[j] = B[j] * 2
    #       else:
    #           for j in dace.map[0:N]:
    #               A[j] = B[j] * C[j]
    #           for j in dace.map[0:N//2]:
    #               D[j] = B[j] * 2

    sdfg.save("example.sdfg", compress=False)

    # Expected SDFG structure:
    #
    # ┌─────────────────────────────────────────────────────────┐
    # │ State 0 (init):                                         │
    # │   Computes N_div_2 = N // 2 (CPU scalar assignment).    │
    # │   N_div_2 becomes a symbol "tmp" used in the branch     |
    # |     condition.                                          │
    # └──────────────────────┬──────────────────────────────────┘
    #                        │
    #              ┌─────────┴─────────┐
    #              ▼                   ▼
    #     ┌── if (i < tmp) ──┐  ┌── else ──────────-┐
    #     │                  │  │                   │
    #     │ State 1:         │  │ State 3:          │
    #     │  Sequential loop │  │  GPU map over j   │
    #     │  A[j] = A[j-1]   │  │  A[j] = B[j]*C[j] │
    #     │       + B[j]     │  │  reads: A, B, C   │
    #     │       + C[j]     │  │  writes: A        │
    #     │  reads:  A,B,C   │  │                   │
    #     │  writes: A       │  │ State 4:          │
    #     │                  │  │  GPU map over j   │
    #     │ State 2:         │  │  D[j] = B[j] * 2  │
    #     │  GPU map over j  │  │  reads:  B        │
    #     │  D[j] = B[j] * 2 │  │  writes: D        │
    #     │  reads:  B       │  └───────────────────┘
    #     │  writes: D       │
    #     └──────────────────┘
    #              │                   │
    #              └─────────┬─────────┘
    #                        ▼
    #                      merge
    #
    # Analysis conclusion:
    #   D     → GPU only (both branches write D in a GPU map)
    #   B     → GPU and CPU (GPU maps read it; the if-branch's
    #           sequential loop also reads it on CPU)
    #   A, C  → GPU and CPU (same reasoning: the if-branch reads/writes
    #           them sequentially on CPU, the else-branch accesses them
    #           in a GPU map)
    #
    # This divergence is the interesting case: the same array may need
    # to reside on *both* devices depending on which branch executes.
    # A downstream pass would need to insert host↔device copies or
    # use unified memory to satisfy both paths.

    # Important: being able to offload this requires chaining the if else
    # which we can support much later when inserting copies:
    # ┌── if (i < tmp) ──┐
    # |                  |
    # └──────────────────┘
    #         -─┬─-
    #           ▼
    # ┌── if (i >= tmp) ─┐
    # |                  |
    # └──────────────────┘
    #         -─┬─-
    #           ▼
    # Then we can sequqntialize and insert necessary copies. 
    # I already have a construction utility that performs this
    # but this is a very late step in the pipeline, so we can ignore it for now.

    sdfg.validate()
    return sdfg


test_example().view()