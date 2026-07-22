# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``NormalizeFloorDivision``: residual sympy floor() must not reach codegen."""
import numpy as np
import pytest
import sympy

import dace
from dace.codegen.targets.cpp import sym2cpp
from dace.subsets import Indices, Range
from dace.symbolic import pystr_to_symbolic, symbol
from dace.transformation.passes.canonicalize.normalize_floor_division import NormalizeFloorDivision, normalize

N = dace.symbol("N")


def test_parsed_dace_source_already_uses_int_floor():
    """`//` written in a kernel is fine -- pystr_to_symbolic maps it onto int_floor. This pass exists
    for the OTHER spelling, python's `//` applied to a sympy object in transformation code."""
    assert type(pystr_to_symbolic("(N - 1) // 2")).__name__ == "__int_floor"
    assert isinstance(symbol("N") // 2, sympy.floor)


@pytest.mark.parametrize("expression", ["(N + 1) * 4 // 8", "(N - 1) // 2", "N // 2", "(2 * N + 3) // 4"])
def test_normalize_is_value_preserving_and_survives_codegen(expression):
    n = symbol("N")
    floored = eval(expression, {"N": n})  # python `//` on a sympy object -> sympy.floor
    assert floored.atoms(sympy.floor), f"{expression} did not produce a sympy floor"
    converted = normalize(floored)
    assert not converted.atoms(sympy.floor)
    for value in range(12):
        assert int(converted.subs({n: value})) == int(floored.subs({n: value})), value
    assert "1 / 2" not in sym2cpp(converted), sym2cpp(converted)


def test_pass_rewrites_map_ranges_memlets_and_shapes():
    """All three places a bad index reaches codegen from."""
    i = symbol("i")
    sdfg = dace.SDFG("residual_floor")
    sdfg.add_array("a", [(N + 1) // 2], dace.float64)
    state = sdfg.add_state()
    entry, exit_node = state.add_map("m", {"i": Range([(0, N - 1, 1)])})
    entry.map.range = Range([((i + 1) // 2, N - 1, 1)])
    tasklet = state.add_tasklet("t", {"inp"}, {"out"}, "out = inp + 1.0")
    read, write = state.add_read("a"), state.add_write("a")
    state.add_memlet_path(read, entry, tasklet, dst_conn="inp", memlet=dace.Memlet(data="a", subset=Indices([i // 2])))
    state.add_memlet_path(tasklet, exit_node, write, src_conn="out", memlet=dace.Memlet("a[i]"))

    assert NormalizeFloorDivision().apply_pass(sdfg, {}) >= 3
    for expr in (*sdfg.arrays["a"].shape, entry.map.range.ranges[0][0]):
        assert not sympy.sympify(expr).atoms(sympy.floor), expr
    for edge in state.edges():
        if edge.data is not None and edge.data.subset is not None:
            for bound in edge.data.subset.free_symbols:
                assert "floor" not in str(bound)


def test_pass_is_a_noop_on_a_clean_sdfg():
    """No floors -> no rewrites, so the pass never reports spurious modification."""

    @dace.program
    def clean(a: dace.float64[N], b: dace.float64[N]):
        for i in dace.map[0:N]:
            b[i] = a[i * 2 // 3]

    sdfg = clean.to_sdfg(simplify=True)
    assert NormalizeFloorDivision().apply_pass(sdfg, {}) is None


def test_normalized_kernel_computes_the_same_numbers():
    """End to end: a strided read whose index carries a residual floor still validates."""

    @dace.program
    def strided(a: dace.float64[N], b: dace.float64[N]):
        for i in dace.map[0:N]:
            b[i] = a[(i + 1) // 2]

    sdfg = strided.to_sdfg(simplify=True)
    NormalizeFloorDivision().apply_pass(sdfg, {})
    size = 16
    a = np.arange(size, dtype=np.float64)
    b = np.zeros(size)
    sdfg(a=a, b=b, N=size)
    np.testing.assert_allclose(b, a[(np.arange(size) + 1) // 2])
