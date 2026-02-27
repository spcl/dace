import dace
from typing import Dict, List

"""
for i in range(0, N):
    for j in range(0, M):
        A[i, j] = B[i, j] + C[i, j]

the loop nests should be represented as:
{ i : Range(0, N), j: Range(0, M) }

access subset should be represented as:
{
    A: [i,j],
    B: [i,j],
    C: [i,j],
}

block size for CPUs will be potentially 8 elements for doubles

We assume both N and M are multiple of the block size (=8)

Need to compute overlap between iterations, behavior
"""

def average_blocks_touched(
    state: dace.SDFGState,
    loop_nests: List[Dict[str, dace.subsets.Range]],
    access_subset: Dict[str, dace.subsets.Subset],
    block_size: int, # In elements
):
    pass