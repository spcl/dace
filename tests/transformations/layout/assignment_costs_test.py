# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""B2/B3-lite cost providers + the end-to-end greedy/global/oracle pipeline (D3 on the fixtures):
the model table finds conflict3's B trajectory (write row-major, flip to column-major for the two
transposed readers), the DP matches the enumeration oracle, the chosen assignment applies
bit-exactly, and the eval table drives the same pipeline from measured timings."""
import pytest

from dace.transformation.layout.apply_assignment import apply_assignment
from dace.transformation.layout.assignment_costs import (assignment_arrays, eval_costs, model_costs,
                                                         permutation_layouts)
from dace.transformation.layout.global_assign import (brute_force_trajectories, conflict_report, format_conflict_report,
                                                      greedy_assignment, per_array_dp, to_assignment)
from dace.transformation.layout.line_graph import kernel_per_state, line_graph
from dace.transformation.layout.prepare import prepare_for_layout

from tests.transformations.layout import multinest_programs as fixtures
from tests.transformations.layout.multinest_fixtures_test import run_and_check


def split_program(program_name):
    program, _, _ = fixtures.PROGRAMS[program_name]
    sdfg = program.to_sdfg(simplify=True)
    prepare_for_layout(sdfg)
    kernel_per_state(sdfg)
    return sdfg, line_graph(sdfg)


def test_permutation_layouts_identity_first():
    layouts = permutation_layouts(2)
    assert [l.tag for l in layouts] == ["identity", "perm10"]
    with pytest.raises(NotImplementedError, match="rank 4"):
        permutation_layouts(4)


def test_model_costs_find_conflict3_trajectory(n=256):
    """The model table on conflict3: B conflicts (producer wants row-major, both transposed readers
    want column-major), the DP picks the paid trajectory, matches the oracle, and the composed
    assignment is bit-exact end to end -- the greedy-vs-global figure in one test."""
    sdfg, kernels = split_program("conflict3")
    assert assignment_arrays(sdfg, kernels) == ["A", "B", "C", "D"]
    costs = model_costs(sdfg, kernels, symbols={"N": n})

    rows = {r.array: r for r in conflict_report(costs, len(kernels))}
    assert rows["B"].conflicting
    assert rows["B"].per_kernel_preference == ["identity", "perm10", "perm10"]
    assert not rows["A"].conflicting and rows["A"].chosen == ["identity"] * 3
    print(format_conflict_report(list(rows.values())))

    dp = per_array_dp(costs, len(kernels))
    oracle = brute_force_trajectories(costs, len(kernels))
    for array in dp:
        assert dp[array].cost == pytest.approx(oracle[array].cost), array
    assert dp["B"].tags == ["identity", "perm10", "perm10"]  # the paid boundary beats carrying
    single = per_array_dp(costs, len(kernels), allow_changes=False)
    assert dp["B"].cost < single["B"].cost  # the trajectory strictly beats every single layout

    # Apply the chosen assignment end to end -- bit-exact against the fixture oracle.
    assignment = to_assignment(dp, costs.layouts)
    assert "B" in assignment
    applied = apply_assignment(sdfg, kernels, assignment)
    assert len(applied.boundary_states) >= 1
    sdfg.validate()
    run_and_check(sdfg, "conflict3", 64, seed=23)


def test_model_costs_agree2_needs_no_changes(n=256):
    sdfg, kernels = split_program("agree2")
    costs = model_costs(sdfg, kernels, symbols={"N": n})
    dp = per_array_dp(costs, len(kernels))
    greedy = greedy_assignment(costs, len(kernels))
    for array in dp:
        assert dp[array].changes() == 0
        assert dp[array].tags == greedy[array].tags  # no conflict: greedy == global
    assert dp["B"].tags == ["identity", "identity"]


def test_eval_costs_drive_the_same_pipeline(n=48):
    """The measured table on conflict2: complete, solvable, and the chosen assignment applies
    bit-exactly. Rankings are not asserted (timing on a shared host is noisy) -- the contract is
    the pipeline, the numbers are advisory. The APPLY leg must not degrade to a no-op under that
    noise, so when the measured DP picks all-identity a non-identity trajectory is forced through
    instead (B flips for kernel 2, which reads B live-in -> an entry conversion always exists)."""
    sdfg, kernels = split_program("conflict2")
    inputs = fixtures.make_inputs(n, seed=29)
    chain = {"A": inputs["A"], **fixtures.conflict2_oracle(inputs["A"])}
    costs = eval_costs(sdfg, kernels, symbols={"N": n}, provided=chain, reps=3, warmup=1)

    for array in ("A", "B", "C"):
        for k in range(len(kernels)):
            for layout in costs.layouts[array]:
                assert (array, k, layout.tag) in costs.node_cost
    dp = per_array_dp(costs, len(kernels))
    oracle = brute_force_trajectories(costs, len(kernels))
    for array in dp:
        assert dp[array].cost == pytest.approx(oracle[array].cost), array

    assignment = to_assignment(dp, costs.layouts)
    if not assignment:
        assignment = {"B": [costs.layouts["B"][0], costs.layouts["B"][1]]}
    applied = apply_assignment(sdfg, kernels, assignment)
    assert len(applied.boundary_states) >= 1 or applied.exit_state is not None
    sdfg.validate()
    run_and_check(sdfg, "conflict2", n, seed=30)


if __name__ == "__main__":
    test_permutation_layouts_identity_first()
    test_model_costs_find_conflict3_trajectory()
    test_model_costs_agree2_needs_no_changes()
    test_eval_costs_drive_the_same_pipeline()
    print("assignment_costs tests PASS")
