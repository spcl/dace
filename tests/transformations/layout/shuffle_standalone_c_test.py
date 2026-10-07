# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A shuffle's floored modulo reaches CPF's standalone C unit, which builds and computes what the program does."""

import copy

import numpy as np

import dace
from dace.codegen.cpf import render
from dace.libraries.layout.shuffle import register_shuffle
from dace.transformation.layout.shuffle_elements import ShuffleElements
from tests.codegen.cpf.conftest import assert_standalone, build_standalone, call_standalone

N = dace.symbol("N")


@dace.program
def increment(A: dace.float64[N]):
    for i in dace.map[0:N]:
        A[i] = A[i] + 1.0


def test_a_shuffled_cyclic_shift_renders_as_standalone_c_and_computes_the_reference():
    register_shuffle("c_cyc", "(i + 1) % N", "(i + N - 1) % N")
    sdfg = copy.deepcopy(increment.to_sdfg(simplify=True))
    sdfg.name = "shuffle_standalone_c"
    ShuffleElements(shuffle_map={"A": ("c_cyc", 0)}).apply_pass(sdfg, {})
    rendering = render(sdfg, language="c")
    assert_standalone(rendering.code, sdfg.name, language="c")

    values = np.random.default_rng(3).random(8)
    arguments = {"A": values.copy(), "N": 8}
    call_standalone(build_standalone(rendering.code, sdfg.name, language="c"), rendering.sdfg, arguments)

    assert np.allclose(arguments["A"], values + 1.0)


if __name__ == "__main__":
    test_a_shuffled_cyclic_shift_renders_as_standalone_c_and_computes_the_reference()
