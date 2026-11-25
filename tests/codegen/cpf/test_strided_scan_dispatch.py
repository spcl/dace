# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A lifted scan ``a[i] = a[i - K] + x[i]`` takes one arm per carry distance ``K``: a constant ``K`` its own arm, a
symbolic one a runtime check over ``K == 1``, ``K == 2``, ``3 <= K <= 8`` and ``K > 8``."""

import numpy as np
import pytest

import dace
from dace.codegen.cpf import render
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from tests.codegen.cpf.conftest import build_standalone, call_standalone

K = dace.symbol("K", dtype=dace.int64, positive=True)
N = dace.symbol("N", dtype=dace.int64, positive=True)
M = dace.symbol("M", dtype=dace.int64, positive=True)

#: Each distance arm, with a distance it covers, and lengths: not a multiple of the distance, past three chunks, and
#: shorter than the distance.
DISTANCES = (1, 2, 3, 8, 9)
LENGTHS = (37, 4096 * 3 + 5, 5)
SYMBOLIC_ARMS = (
    "the runtime check found K == 1: one chain, the unit-stride prefix scan",
    "the runtime check found K == 2: two independent chains, each a prefix scan at the literal distance 2",
    "the runtime check found 3 <= K <= 8: K independent chains, each a prefix scan carried in a register",
    "the runtime check found K > 8: K independent chains, each a prefix scan; a vector holds consecutive elements",
)
#: What the arm of each constant distance runs.
CONSTANT_ARMS = {
    2: "two independent chains, each a prefix scan at the literal distance 2",
    3: "3 independent chains, each a prefix scan carried in a register",
    8: "8 independent chains, each a prefix scan carried in a register",
    9: "9 independent chains, each a prefix scan; a vector holds consecutive elements",
}


@dace.program
def strided_scan_symbolic(a: dace.float64[N], x: dace.float64[N]):
    for i in range(K, N):
        a[i] = a[i - K] + x[i]


@dace.program
def strided_scan_rows(a: dace.float64[M, N], x: dace.float64[M, N]):
    for r in dace.map[0:M]:
        for i in range(K, N):
            a[r, i] = a[r, i - K] + x[r, i]


def constant_distance_program(distance: int):
    @dace.program
    def strided_scan_constant(a: dace.float64[N], x: dace.float64[N]):
        for i in range(distance, N):
            a[i] = a[i - distance] + x[i]

    return strided_scan_constant


def reference(a: np.ndarray, x: np.ndarray, distance: int) -> np.ndarray:
    a = a.copy()
    for i in range(distance, a.shape[-1]):
        a[..., i] = a[..., i - distance] + x[..., i]
    return a


def rendered(program, name: str, language: str) -> tuple[dace.SDFG, str]:
    sdfg = program.to_sdfg()
    sdfg.name = name
    canonicalize(sdfg, target="cpu")
    return sdfg, render(sdfg, language=language).code


def check(sdfg: dace.SDFG, code: str, language: str, shape: tuple[int, ...], distance: int, symbolic: bool) -> None:
    sut = build_standalone(code, sdfg.name, language=language)
    rng = np.random.default_rng(distance)
    a, x = rng.uniform(0.0, 1.0, shape), rng.uniform(0.0, 1.0, shape)
    want = reference(a, x, distance)
    arguments = {"a": a, "x": x, "N": shape[-1]}
    if len(shape) == 2:
        arguments["M"] = shape[0]
    if symbolic:
        arguments["K"] = distance
    call_standalone(sut, sdfg, arguments)
    np.testing.assert_allclose(a, want, rtol=1e-10, err_msg=f"K={distance} shape={shape}")


@pytest.mark.parametrize("language", ["c", "c++"])
# Reproducer: CPF: render and canonicalization fixes for v1.0.1 (6db8721f3d)
def test_a_symbolic_distance_is_dispatched_once_per_scan(language: str) -> None:
    sdfg, code = rendered(strided_scan_symbolic, f"scan_distance_symbolic_{language.replace('+', 'p')}", language)
    for arm in SYMBOLIC_ARMS:
        assert code.count("parallel -- scan, " + arm) == 1, code
    # ``K == 1`` is tested once, ahead of every loop of the scan; the rescan arms once per chunk of pass 2
    assert code.count("if ((K) == 1)") == 1, code
    assert code.index("if ((K) == 1)") < code.index("for (long cpf_"), code
    assert code.count("if (cpf_s == 2)") == 1, code
    for distance in DISTANCES:
        for size in LENGTHS:
            check(sdfg, code, language, (size,), distance, symbolic=True)


@pytest.mark.parametrize("distance", [2, 3, 8, 9])
# Reproducer: CPF: render and canonicalization fixes for v1.0.1 (6db8721f3d)
def test_a_constant_distance_takes_its_one_arm(distance: int) -> None:
    sdfg, code = rendered(constant_distance_program(distance), f"scan_distance_constant_{distance}", "c")
    assert f"parallel -- scan, the carry distance is {distance}: {CONSTANT_ARMS[distance]}" in code, code
    assert "runtime check" not in code, code
    for size in LENGTHS:
        check(sdfg, code, "c", (size,), distance, symbolic=False)


@pytest.mark.parametrize("language", ["c", "c++"])
# Reproducer: CPF: render and canonicalization fixes for v1.0.1 (6db8721f3d)
def test_a_scan_in_a_parallel_map_is_dispatched_once_per_row(language: str) -> None:
    sdfg, code = rendered(strided_scan_rows, f"scan_distance_rows_{language.replace('+', 'p')}", language)
    for arm in SYMBOLIC_ARMS:
        assert code.count("sequential -- scan, " + arm) == 1, code
    for distance in DISTANCES:
        for size in (37, 5):
            check(sdfg, code, language, (3, size), distance, symbolic=True)


if __name__ == "__main__":
    for language in ("c", "c++"):
        test_a_symbolic_distance_is_dispatched_once_per_scan(language)
        test_a_scan_in_a_parallel_map_is_dispatched_once_per_row(language)
    for distance in (2, 3, 8, 9):
        test_a_constant_distance_takes_its_one_arm(distance)
