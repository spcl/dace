# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A point read through a strided range (``a[i*S + k : i*S + k : S]``) is one element, whatever the step.

Canonicalization reads ``a[k : k + N*S : S]`` element by element as the range ``i*S + k`` to ``i*S + k`` with step
``S``. Its size was ``ceiling(1/S)``, which a registry symbol (a bare name, no sign) does not fold to one, so the
tasklet's connector was typed as a pointer and the C unit added a pointer to a double
(``conv_transpose3d_avg_pool_clamp_softmax_multiply``'s average pool).
"""

import numpy as np

import dace
from dace import subsets, symbolic
from dace.codegen.cpf import render
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from tests.codegen.cpf.conftest import build_standalone, call_standalone

N = dace.symbol("N", dace.int64, positive=True)
S = dace.symbol("S", dace.int64, positive=True)


@dace.program
def strided_pool(a: dace.float64[N * S], b: dace.float64[N]):
    for k in range(S):
        b[:] = b + a[k : k + N * S : S]


def test_a_single_index_with_a_symbolic_step_has_one_element():
    i, step = symbolic.symbol("i"), symbolic.symbol("S")
    point = subsets.Range([(i * step + 1, i * step + 1, step)])
    assert point.size() == [1]
    assert point.num_elements() == 1


def test_a_strided_pool_renders_and_computes_what_numpy_does():
    sdfg = strided_pool.to_sdfg(simplify=True)
    sdfg.name = "cpf_strided_point_memlet"
    canonicalize(sdfg, target="cpu")
    finalize_for_target(sdfg, "cpu")
    rendering = render(sdfg, language="c")
    library = build_standalone(rendering.code, sdfg.name, language="c")
    n, s = 10, 3
    a = np.random.default_rng(0).standard_normal(n * s)
    arguments = {"a": a.copy(), "b": np.zeros(n), "N": n, "S": s}
    call_standalone(library, rendering.sdfg, arguments, rendering.arguments)
    assert np.allclose(arguments["b"], a.reshape(n, s).sum(axis=1))


if __name__ == "__main__":
    test_a_single_index_with_a_symbolic_step_has_one_element()
    test_a_strided_pool_renders_and_computes_what_numpy_does()
