# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A table of compile-time constants read by an index symbol is a premise, not an unknown.

CloudSC's melting step loops over the species ``jm`` and, for those whose phase table ``iphase`` says ice, moves mass
from ``zqxfg[jm]`` to ``zqxfg[imelt[jm]]``. The second index is a symbol the iteration binds from the table, so it
varies per iteration and the loop was refused. ``iphase`` and ``imelt`` are literal-filled tables, SDFG constants after
``FoldConstantTables``: every value is known, so the two writes can be compared exactly, and no iteration's melt target
is an ice species.
"""

import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.fold_constant_tables import FoldConstantTables
from dace.transformation.passes.parallelization.parallelize_loops import ParallelizeLoops

L = dace.symbol("L")


@dace.program
def melt_ice_into_liquid(q: dace.float64[5, L], x: dace.float64[5, L]):
    iphase = np.ndarray([5], dtype=np.int32)
    imelt = np.ndarray([5], dtype=np.int32)
    iphase[0] = 0
    iphase[1] = 1
    iphase[2] = 1
    iphase[3] = 2
    iphase[4] = 2
    imelt[0] = 0
    imelt[1] = 0
    imelt[2] = 0
    imelt[3] = 1
    imelt[4] = 2
    for jm in range(5):
        if iphase[jm] == 2:
            target = imelt[jm]
            for jl in dace.map[0:L]:
                if x[jm, jl] > 0.0:
                    q[jm, jl] = q[jm, jl] - x[jm, jl]
                    q[target, jl] = q[target, jl] + x[jm, jl]


@dace.program
def melt_ice_into_one_liquid(q: dace.float64[5, L], x: dace.float64[5, L]):
    iphase = np.ndarray([5], dtype=np.int32)
    imelt = np.ndarray([5], dtype=np.int32)
    iphase[0] = 0
    iphase[1] = 1
    iphase[2] = 1
    iphase[3] = 2
    iphase[4] = 2
    imelt[0] = 0
    imelt[1] = 0
    imelt[2] = 0
    imelt[3] = 1
    imelt[4] = 1
    for jm in range(5):
        if iphase[jm] == 2:
            target = imelt[jm]
            for jl in dace.map[0:L]:
                if x[jm, jl] > 0.0:
                    q[jm, jl] = q[jm, jl] - x[jm, jl]
                    q[target, jl] = q[target, jl] + x[jm, jl]


@dace.program
def melt_ice_into_liquid_read_inside_the_column_map(q: dace.float64[5, L], x: dace.float64[5, L]):
    iphase = np.ndarray([5], dtype=np.int32)
    imelt = np.ndarray([5], dtype=np.int32)
    iphase[0] = 0
    iphase[1] = 1
    iphase[2] = 1
    iphase[3] = 2
    iphase[4] = 2
    imelt[0] = 0
    imelt[1] = 0
    imelt[2] = 0
    imelt[3] = 1
    imelt[4] = 2
    for jm in range(5):
        for jl in dace.map[0:L]:
            if iphase[jm] == 2:
                target = imelt[jm]
                if x[jm, jl] > 0.0:
                    q[jm, jl] = q[jm, jl] - x[jm, jl]
                    q[target, jl] = q[target, jl] + x[jm, jl]


def folded_and_parallelized(program: dace.program = melt_ice_into_liquid) -> dace.SDFG:
    sdfg = program.to_sdfg(simplify=True)
    assert FoldConstantTables().apply_pass(sdfg, {}) == {"iphase", "imelt"}
    ParallelizeLoops().apply_pass(sdfg, {})
    return sdfg


def loops(sdfg: dace.SDFG) -> int:
    return sum(isinstance(r, LoopRegion) for r in sdfg.all_control_flow_regions(recursive=True))


def test_a_loop_whose_writes_go_through_constant_tables_is_a_map_and_computes_the_same():
    sdfg = folded_and_parallelized()
    assert loops(sdfg) == 0
    rng = np.random.default_rng(0)
    q, x = rng.standard_normal((5, 6)), rng.standard_normal((5, 6))
    expected = q.copy()
    for jm, target in ((3, 1), (4, 2)):
        gone = np.where(x[jm] > 0.0, x[jm], 0.0)
        expected[jm] -= gone
        expected[target] += gone
    sdfg(q=q, x=x, L=6)
    assert np.allclose(q, expected, rtol=0, atol=1e-15)


def test_tables_read_inside_the_column_map_still_let_the_species_loop_map_and_compute_the_same():
    sdfg = folded_and_parallelized(melt_ice_into_liquid_read_inside_the_column_map)
    assert loops(sdfg) == 0
    rng = np.random.default_rng(1)
    q, x = rng.standard_normal((5, 6)), rng.standard_normal((5, 6))
    expected = q.copy()
    for jm, target in ((3, 1), (4, 2)):
        gone = np.where(x[jm] > 0.0, x[jm], 0.0)
        expected[jm] -= gone
        expected[target] += gone
    sdfg(q=q, x=x, L=6)
    assert np.allclose(q, expected, rtol=0, atol=1e-15)


def test_a_loop_whose_two_ice_species_melt_into_one_liquid_stays_a_loop():
    assert loops(folded_and_parallelized(melt_ice_into_one_liquid)) == 1


if __name__ == "__main__":
    test_tables_read_inside_the_column_map_still_let_the_species_loop_map_and_compute_the_same()
    test_a_loop_whose_two_ice_species_melt_into_one_liquid_stays_a_loop()
    test_a_loop_whose_writes_go_through_constant_tables_is_a_map_and_computes_the_same()
