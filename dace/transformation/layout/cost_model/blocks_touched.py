import dace
from typing import Dict, List, Iterable, Optional, NamedTuple, Set
import copy

import dace
import sympy as sp

from dace.transformation.layout.cost_model.access_subsets import get_access_subsets

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


def static_overlap_blocks_touched(loop_ranges, access_subsets, block_size, sdfg):
    """Placeholder for static overlap model."""
    raise NotImplementedError("Static overlap analysis not yet implemented.")


def average_blocks_touched(
    state: dace.SDFGState,
    loop_nests: Set[dace.nodes.MapEntry], # Loop nests going from outer to inner
    loop_ranges: List[Dict[str, dace.subsets.Range]], # Same order list of param->range
    access_subsets: Dict[str, dace.subsets.Subset], # Dict mapping arrays to the subsets accessed
    block_size: int, # Block (cache line) size
    symbols_defined: Set[str], # Symbols available within the innermost nest
    overlap: bool = False, # Whether to use static overlap to assess the block numbers
) -> Dict[str, sp.Basic]:
    """
    Returns dict of array_name -> symbolic average new blocks per iteration.
    """
    sdfg = state.sdfg

    if overlap:
        return static_overlap_blocks_touched(loop_ranges, access_subsets, block_size, sdfg)

    print(loop_ranges, type(loop_ranges))
    params = list()
    # Collect all loop parameters
    for lnest in loop_ranges:
        for param in lnest:
            params.append(param)
    print("Params", params)

    # Flatten such that we can acess the range by just providing the symbol name
    flattened_loop_ranges = {}
    for lnest in loop_ranges:
        for param, lrange in lnest.items():
            assert param not in flattened_loop_ranges
            flattened_loop_ranges[param] = lrange

    sym = lambda s: dace.symbolic.pystr_to_symbolic(s)

    # Get iteration counts
    extents = {}
    for p in params:
        b, e, s = flattened_loop_ranges[p]
        extents[p] = sp.floor((sym(e) - sym(b)) / sym(s)) + 1

    # Acc. to total iteration coutns
    total_iters = sp.Mul(*[extents[p] for p in params])
    print(f"Total iterations: {total_iters}")
    print(f"Params: {params}")

    results = {}

    for arr, subset in access_subsets.items():
        if arr not in sdfg.arrays or not isinstance(subset, dace.subsets.Range):
            continue

        strides = sdfg.arrays[arr].strides
        indices = [sym(rb) for rb, _, _ in subset.ranges]
        addr = sum(idx * sym(st) for idx, st in zip(indices, strides))

        total_new = sp.Integer(1)  # iteration 0 always new

        for depth, p in enumerate(params):
            psym = sym(p)
            step_sym = sym(flattened_loop_ranges[p][2])
            stride = sp.simplify(addr.subs(psym, psym + step_sym) - addr)
            print(f"Step_sym: {step_sym}")
            print(f"Stride: {stride}")

            if p == params[-1]:  # innermost
                frac = sp.Min(1, sp.Abs(stride) / block_size)
            else:
                frac = sp.Integer(1)

            outer = sp.Mul(*[extents[params[k]] for k in range(depth)]) if depth else sp.Integer(1)
            total_new += outer * (extents[p] - 1) * frac

        results[arr] = sp.simplify(total_new / total_iters)

    return results


if __name__ == "__main__":
    N = dace.symbol("N")
    @dace.program
    def madd(A : dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
        for i, j in dace.map[0:N, 0:N]:
            C[i, j] = B[j, i] + A[i, j]
    
    sdfg = madd.to_sdfg()
    states = {s for s in sdfg.all_states()}
    assert len(states) == 1

    sdfg.save("s.sdfg")

    state = states.pop()

    loop_nests = {n for n in state.nodes() if isinstance(n, dace.nodes.MapEntry)}
    assert len(loop_nests) == 1

    loop_ranges = [
        {k: r for k, r in zip(loop_nest.map.params, loop_nest.map.range)}
        for loop_nest in loop_nests
    ]
    all_map_params = {s for lp in loop_nests for s in lp.params}
    print(all_map_params)

    symbols = set(all_map_params)
    for loop_nest in loop_nests:
        symbols |= state.symbols_defined_at(loop_nest).keys()

    print(symbols)
    print(loop_ranges)

    access_subsets = get_access_subsets(state, loop_nest)

    print(access_subsets)

    average_blocks_touched(
        state=state,
        loop_nests=loop_nests,
        loop_ranges=loop_ranges,
        access_subsets=access_subsets,
        block_size=2,
        symbols_defined=symbols,
    )


"""
for i in range(4):
    for j in range(4):
        # B = 2
        # j -> j + 1
        C[i, j] = A[i, j] + B[i, j]

        C[i, idx[j]] = A[i, idx[j]] + B[i, idx[j]]



for i in range(N):
    for j in range(N):
        C[i // 4, j // 4, i % 4, j % 4] = 
            A[i // 4, j // 4, i % 4, j % 4] + 
            B[i // 4, j // 4, i % 4, j % 4]

for i in range(N//4):
    for j in range(N//4):
        for ii in range(4):
            for jj in range(4):
                C[i, j, ii, jj] = 
                    A[i, j, ii, jj] + 
                    B[i, j, ii, jj]


for i in range(N):
    for j in range(N):
        C[i // 4 + i % 4, j // 4 + j % 4] = 
            A[i // 4 + i % 4, j // 4 + j % 4]+ 
            B[i // 4 + i % 4, j // 4 + j % 4]

for i in range(N//4):
    for j in range(N//4):
        for ii in range(4):
            for jj in range(4):
                C[i*4 + ii, j*4 + jj] = 
                    A[i*4 + ii, j*4 + jj] + 
                    B[i*4 + ii, j*4 + jj]
"""