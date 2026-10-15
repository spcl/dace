# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A persistent buffer of a nested SDFG every thread of a team calls is ONE buffer, allocated before the team.

``HoistParallelRegion`` outlines a sequential loop over parallel maps into a nested SDFG called by every thread of
a ``#pragma omp parallel`` region, and the maps inside become ``omp for``. Its scratch arrays are ``Persistent``:
the ordinary generators keep them in the state struct, one buffer shared by the team. CPF demotes that lifetime to
an allocation on entry of the declaring function, which for the outlined body meant one private copy per thread:
each thread filled its own ``omp for`` share and read the rest uninitialized. bfs is that shape (``frontier``,
``reach``, ``nxt`` per level); here a mirrored read makes every thread read another's share.
:func:`dace.codegen.cpf.lift_team_shared_transients` moves such buffers to the entry.
"""

import numpy as np

import dace
from dace.codegen import cpf
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from tests.codegen.cpf.conftest import assert_matches, build_standalone, call_standalone

N = dace.symbol("N")


@dace.program
def mirrored_sweep(level: dace.float64[N], T: dace.int64):
    for d in range(T):
        scratch = level * 2.0 + d
        for i in dace.map[0:N]:
            level[i] = scratch[N - 1 - i] * 0.5


def reference(level: np.ndarray, steps: int) -> np.ndarray:
    level = level.copy()
    for d in range(steps):
        scratch = level * 2.0 + d
        level[:] = scratch[::-1] * 0.5
    return level


def rendered(name: str) -> tuple[dace.SDFG, str]:
    sdfg = mirrored_sweep.to_sdfg(simplify=True)
    canonicalize(sdfg, target="cpu")
    finalize_for_target(sdfg, "cpu")
    sdfg.name = name
    return sdfg, cpf.render(sdfg, language="c++").code


def inputs(n: int = 4099) -> dict[str, np.ndarray]:
    return {"level": np.random.default_rng(5).random(n)}


def test_a_team_shares_one_allocation_of_each_persistent_buffer():
    sdfg, code = rendered("cpf_team_shared")
    team = code.index("#pragma omp parallel\n")
    called = code[team:].split("(", 1)[0].split()[-1]
    start = code.index(f"void {called}(")
    body = code[start : code.index("\n}\n", start)]
    # The function every thread of the team runs allocates nothing: its buffers come from the entry.
    assert " new " not in body.split("{", 1)[1], body


def test_a_team_shared_buffer_computes_the_reference():
    sdfg, code = rendered("cpf_team_shared_values")
    library = build_standalone(code, "cpf_team_shared_values")
    arguments = inputs()
    expected = reference(arguments["level"], 7)
    call_standalone(library, sdfg, {**arguments, "N": arguments["level"].shape[0], "T": 7})
    assert_matches({"level": expected}, {"level": arguments["level"]})


if __name__ == "__main__":
    test_a_team_shares_one_allocation_of_each_persistent_buffer()
    test_a_team_shared_buffer_computes_the_reference()
