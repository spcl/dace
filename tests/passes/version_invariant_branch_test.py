# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""ParallelizeUnderConstraint versions a loop on the condition of the one branch that stores what kept it a loop, when
the loop leaves that condition unchanged: ``if (!cond) { parallel, without the branch } else { the loop }``.

CloudSC's column loop over ``jl`` stores ``ZVQX(NCLDQI)`` only under ``IF (LAERICESED .AND. JM == NCLDQI)``, a flag no
iteration changes; with the flag off, every iteration is independent.
"""

import numpy as np

import dace
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.parallelization.contracts.under_constraint import ParallelizeUnderConstraint

N = dace.symbol("N")


@dace.program
def flag_stores_shared(flag: dace.int32, j: dace.int32, a: dace.float64[N], v: dace.float64[4], b: dace.float64[N]):
    for i in range(N):
        if flag and j == 2:
            v[1] = 0.002 * a[i]
        b[i] = v[j - 1] * a[i]


def reference(flag, j, a, v, b):
    for i in range(a.shape[0]):
        if flag and j == 2:
            v[1] = 0.002 * a[i]
        b[i] = v[j - 1] * a[i]


def top_loops(sdfg: dace.SDFG) -> list[LoopRegion]:
    return [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion)]


# Reproducer: CloudSC's jl loop stayed sequential on zvqx, stored only under LAERICESED (dace-fortran CloudSC, 9 loops)
def test_a_loop_storing_only_under_an_invariant_flag_is_versioned_on_it():
    sdfg = flag_stores_shared.to_sdfg(simplify=True)
    loop = top_loops(sdfg)[0]
    xf = LoopToMap()
    xf.loop = loop
    assert not xf.can_be_applied(sdfg, 0, sdfg)

    assert ParallelizeUnderConstraint().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    versions = [b for b in sdfg.nodes() if isinstance(b, ConditionalBlock)]
    assert len(versions) == 1
    # The parallel arm holds a map and no loop; the fallback keeps the loop
    parallel, fallback = (branch for condition, branch in versions[0].branches)
    assert not any(isinstance(r, LoopRegion) for r in parallel.all_control_flow_regions())
    assert any(isinstance(r, LoopRegion) for r in fallback.all_control_flow_regions())

    rng = np.random.default_rng(0)
    for flag, j in ((0, 2), (1, 2), (1, 3)):
        a = rng.standard_normal(16)
        v, b = rng.standard_normal(4), np.zeros(16)
        v_ref, b_ref = v.copy(), b.copy()
        reference(flag, j, a, v_ref, b_ref)
        sdfg(flag=flag, j=j, a=a, v=v, b=b, N=16)
        assert np.allclose(b, b_ref) and np.allclose(v, v_ref), (flag, j)


M = dace.symbol("M")


@dace.program
def flag_stores_shared_in_a_map(flag: dace.int32, a: dace.float64[M, N], v: dace.float64[M, 4], b: dace.float64[M, N]):
    for r in dace.map[0:M]:
        for i in range(N):
            if flag:
                v[r, 1] = 0.002 * a[r, i]
            b[r, i] = v[r, 1] * a[r, i]


# Reproducer: the versioning looked the loop's containers up in the top-level SDFG, a KeyError for a loop nested in a
# map (velocity_tendencies, __gather8_o)
def test_a_loop_in_a_nested_sdfg_is_versioned_in_its_own_sdfg():
    sdfg = flag_stores_shared_in_a_map.to_sdfg(simplify=True)
    assert ParallelizeUnderConstraint().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    rng = np.random.default_rng(1)
    for flag in (0, 1):
        a = rng.standard_normal((3, 8))
        v, b = rng.standard_normal((3, 4)), np.zeros((3, 8))
        v_ref, b_ref = v.copy(), b.copy()
        for r in range(3):
            reference(flag, 2, a[r], v_ref[r], b_ref[r])
        sdfg(flag=flag, a=a, v=v, b=b, M=3, N=8)
        assert np.allclose(b, b_ref) and np.allclose(v, v_ref), flag


@dace.program
def flag_changes_in_loop(a: dace.float64[N], v: dace.float64[4], b: dace.float64[N]):
    flag = 0
    for i in range(N):
        if flag:
            v[1] = a[i]
        flag = i
        b[i] = v[1] * a[i]


def test_a_branch_on_a_condition_the_loop_changes_is_not_versioned():
    sdfg = flag_changes_in_loop.to_sdfg(simplify=True)
    assert ParallelizeUnderConstraint().apply_pass(sdfg, {}) is None


if __name__ == "__main__":
    test_a_loop_storing_only_under_an_invariant_flag_is_versioned_on_it()
    test_a_loop_in_a_nested_sdfg_is_versioned_in_its_own_sdfg()
    test_a_branch_on_a_condition_the_loop_changes_is_not_versioned()
