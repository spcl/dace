# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Where the readable CPU generator declares a ``LoopRegion`` counter: in the loop's own for-init clause when nothing
reads it outside the loop, hoisted otherwise, as the legacy generator always does."""

import numpy as np

import dace
from tests.codegen.readable.common import (
    EXPERIMENTAL,
    LEGACY,
    assert_outputs_equivalent,
    generated_code,
    run_isolated,
    use_implementation,
)

N = dace.symbol("N")


@dace.program
def accumulate(A: dace.float64[N, 8], B: dace.float64[N]):
    for i in dace.map[0:N]:
        s = 0.0
        for k in range(8):
            s = s + A[i, k]
        B[i] = s


def code_of(sdfg: dace.SDFG, implementation: str) -> str:
    with use_implementation(implementation):
        return generated_code(sdfg)


def test_loop_counter_declared_in_for_init():
    code = code_of(accumulate.to_sdfg(simplify=True), EXPERIMENTAL)
    assert "int k;" not in code, code
    assert "for (int k = 0;" in code, code


def test_counter_used_after_loop_stays_hoisted():
    """A counter read after its loop cannot move into the for-init clause: the read would be out of scope."""
    sdfg = dace.SDFG("after_loop")
    sdfg.add_array("out", [1], dace.float64)
    loop = dace.sdfg.state.LoopRegion("lp", "j < 5", "j", "j = 0", "j = j + 1")
    sdfg.add_node(loop, is_start_block=True)
    loop.add_state("body", is_start_block=True)
    after = sdfg.add_state_after(loop, "after")
    tasklet = after.add_tasklet("use", {}, {"o"}, "o = j")
    after.add_edge(tasklet, "o", after.add_write("out"), None, dace.Memlet("out[0]"))
    sdfg.validate()

    code = code_of(sdfg, EXPERIMENTAL)
    assert "j;" in code, code
    assert "for (int64_t j = 0;" not in code and "for (long long j = 0;" not in code, code


def test_legacy_keeps_the_hoisted_loop_counter():
    code = code_of(accumulate.to_sdfg(simplify=True), LEGACY)
    assert "int k;" in code and "for (int k = 0;" not in code, code


def test_accumulator_matches_legacy():
    A = np.random.default_rng(0).random((4, 8))

    def run(implementation):
        def build_and_run():
            with use_implementation(implementation):
                sdfg = accumulate.to_sdfg(simplify=True)
                sdfg.name = f"loop_counter_decl_{implementation}"
                B = np.zeros(4)
                sdfg.compile()(A=A.copy(), B=B, N=4)
                return {"B": B}

        return run_isolated(build_and_run)

    legacy = run(LEGACY)
    assert np.allclose(legacy["B"], A.sum(axis=1))
    assert_outputs_equivalent(legacy, run(EXPERIMENTAL), "cpu", label="accumulate")


if __name__ == "__main__":
    test_loop_counter_declared_in_for_init()
    test_counter_used_after_loop_stays_hoisted()
    test_legacy_keeps_the_hoisted_loop_counter()
    test_accumulator_matches_legacy()
