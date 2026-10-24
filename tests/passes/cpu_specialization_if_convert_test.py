# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""IfConvertParallelMapBodies: per-element if/else in a parallel map body becomes ITE selects (``?:``)."""

import numpy as np

import dace
from dace.codegen.cpf import cpf
from dace.sdfg.state import ConditionalBlock
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize

N = dace.symbol("N", dace.int64)


@dace.program
def nested_unequal_arms(
    a: dace.float64[N],
    b: dace.float64[N],
    c: dace.float64[N],
    d: dace.float64[N],
    e: dace.float64[N],
    x: dace.float64[1],
):
    for i in range(N):
        if a[i] > b[i]:
            a[i] = a[i] + b[i] * d[i]
            if N > 10:
                c[i] = c[i] + d[i] * d[i]
            else:
                c[i] = d[i] * e[i] + 1.0
        else:
            b[i] = a[i] + e[i] * e[i]
            if x[0] > 0.0:
                c[i] = a[i] + d[i] * d[i]
            else:
                c[i] = c[i] + e[i] * e[i]


@dace.program
def guarded_gather(a: dace.float64[N], idx: dace.int64[N], b: dace.float64[N]):
    for i in range(N):
        if idx[i] >= 0:
            a[i] = b[idx[i]]


@dace.program
def exclusive_switch(
    a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N], e: dace.float64[N], k: dace.int32[N]
):
    """TSVC ``s442``: each arm reads an array no other path reads."""
    for i in range(N):
        if k[i] == 1:
            a[i] = a[i] + b[i] * b[i]
        elif k[i] == 2:
            a[i] = a[i] + c[i] * c[i]
        elif k[i] == 3:
            a[i] = a[i] + d[i] * d[i]
        elif k[i] == 4:
            a[i] = a[i] + e[i] * e[i]


def reference(a, b, c, d, e, x):
    a, b, c = a.copy(), b.copy(), c.copy()
    for i in range(a.size):
        if a[i] > b[i]:
            a[i] = a[i] + b[i] * d[i]
            c[i] = c[i] + d[i] * d[i] if a.size > 10 else d[i] * e[i] + 1.0
        else:
            b[i] = a[i] + e[i] * e[i]
            c[i] = a[i] + d[i] * d[i] if x[0] > 0.0 else c[i] + e[i] * e[i]
    return a, b, c


def canonical(program, name: str) -> dace.SDFG:
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = name
    canonicalize(sdfg, validate=True, validate_all=False, target="cpu")
    finalize_for_target(sdfg, "cpu")
    return sdfg


def conditionals(sdfg: dace.SDFG) -> int:
    return sum(
        isinstance(b, ConditionalBlock) for sd in sdfg.all_sdfgs_recursive() for b in sd.all_control_flow_blocks()
    )


def test_nested_unequal_arms_become_selects_and_compute_the_reference():
    sdfg = canonical(nested_unequal_arms, "ifconv_nested_unequal")
    assert conditionals(sdfg) == 0
    code = cpf(sdfg)
    assert " ? " in code and "ITE(" not in code
    rng = np.random.default_rng(0)
    arrays = {name: rng.random(1000) for name in "abcde"}
    x = np.array([-0.5])
    expected = reference(arrays["a"], arrays["b"], arrays["c"], arrays["d"], arrays["e"], x)
    sdfg(**arrays, x=x, N=1000)
    for name, value in zip("abc", expected):
        np.testing.assert_allclose(arrays[name], value, rtol=1e-14, atol=0.0)


def test_a_guarded_gather_keeps_its_branch():
    sdfg = canonical(guarded_gather, "ifconv_guarded_gather")
    assert conditionals(sdfg) == 1


def test_a_switch_whose_arms_read_their_own_arrays_keeps_its_branches():
    """As selects, every element would stream b, c, d and e and store a, where the branches touch one of them on
    the elements that take an arm (40.9 ms against 29.2 ms on s442 at preset L)."""
    sdfg = canonical(exclusive_switch, "ifconv_exclusive_switch")
    assert conditionals(sdfg) > 0
    rng = np.random.default_rng(0)
    arrays = {name: rng.random(64) for name in "abcde"}
    k = rng.integers(0, 6, 64).astype(np.int32)
    expected = arrays["a"].copy()
    for i in range(64):
        if 1 <= k[i] <= 4:
            other = arrays["bcde"[k[i] - 1]][i]
            expected[i] += other * other
    sdfg(**arrays, k=k, N=64)
    assert np.allclose(arrays["a"], expected)


if __name__ == "__main__":
    test_nested_unequal_arms_become_selects_and_compute_the_reference()
    test_a_guarded_gather_keeps_its_branch()
    test_a_switch_whose_arms_read_their_own_arrays_keeps_its_branches()
