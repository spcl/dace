# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``Reduce`` renders through its OpenMP CPU lowering: CPF spells ``dace::reduce`` as a ``reduction`` loop.

Rendered through ``pure`` instead, a full reduction is a parallel loop with a ``critical`` section per element,
which made canonicalized argmax_value take about half a second for six million doubles.
"""

import numpy as np
import pytest

import dace
from dace.codegen import cpf
from tests.codegen.cpf.conftest import build_standalone, call_standalone

N = dace.symbol("N", dace.int64)
M = dace.symbol("M", dace.int64)
B = dace.symbol("B", dace.int64)
L = dace.symbol("L", dace.int64)


@dace.program
def full_max(a: dace.float64[N], out: dace.float64[1]):
    out[0] = np.max(a)


@dace.program
def full_sum(a: dace.int32[N], out: dace.int32[1]):
    out[0] = np.sum(a)


@dace.program
def complex_sum(a: dace.complex128[N], out: dace.complex128[1]):
    out[0] = np.sum(a)


@dace.program
def row_min(a: dace.float64[M, N], out: dace.float64[M]):
    out[:] = np.min(a, axis=1)


@pytest.mark.parametrize("language", ["c", "c++"])
@pytest.mark.parametrize(
    ("program", "clause", "dtype", "reference"),
    [
        (full_max, "reduction(max:acc)", np.float64, lambda a: np.array([a.max()])),
        (full_sum, "reduction(+:acc)", np.int32, lambda a: np.array([a.sum()], dtype=np.int32)),
    ],
    ids=["max", "sum"],
)
def test_a_full_reduction_is_one_openmp_reduction_loop(language, program, clause, dtype, reference):
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = f"cpf_reduce_{program.name}_{'c' if language == 'c' else 'cpp'}"
    a = (np.random.default_rng(3).random(1000) * 1000).astype(dtype)
    out = np.zeros(1, dtype=dtype)

    rendering = cpf.render(sdfg, language=language)
    call_standalone(
        build_standalone(rendering.code, sdfg.name, language=language),
        rendering.sdfg,
        {"a": a, "out": out, "N": a.size},
    )

    assert clause in rendering.code and "critical" not in rendering.code, rendering.code
    assert "// reduction over the given axes with the given operator" in rendering.code, rendering.code
    assert np.array_equal(out, reference(a))


@pytest.mark.parametrize("language", ["c", "c++"])
def test_a_reduction_under_its_parallel_output_loop_folds_each_row_sequentially(language):
    sdfg = row_min.to_sdfg(simplify=True)
    sdfg.name = f"cpf_reduce_row_min_{'c' if language == 'c' else 'cpp'}"
    a = np.random.default_rng(4).random((7, 33))
    out = np.zeros(7)

    rendering = cpf.render(sdfg, language=language)
    call_standalone(
        build_standalone(rendering.code, sdfg.name, language=language),
        rendering.sdfg,
        {"a": a, "out": out, "M": 7, "N": 33},
    )

    assert "reduce_seq_min" in rendering.code and "critical" not in rendering.code, rendering.code
    assert np.array_equal(out, a.min(axis=1))


@dace.program
def slab_max(a: dace.float64[B, L, N], out: dace.float64[1]):
    out[0] = np.max(a[:, 2, :])


@dace.program
def slab_any(a: dace.bool_[B, L, N], out: dace.bool_[1]):
    out[0] = np.any(a[:, 2, :])


def slab(dtype):
    rng = np.random.default_rng(6)
    if dtype == np.bool_:
        a = np.zeros((40, 5, 33), dtype=np.bool_)
        a[37, 2, 31] = True
        return a
    return rng.random((40, 5, 33))


@pytest.mark.parametrize("language", ["c", "c++"])
@pytest.mark.parametrize(
    ("program", "clause", "dtype", "reference"),
    [
        (slab_max, "reduction(max:_acc)", np.float64, lambda a: a[:, 2, :].max()),
        (slab_any, "reduction(||:_acc)", np.bool_, lambda a: a[:, 2, :].any()),
    ],
    ids=["max", "any"],
)
def test_a_reduction_over_a_strided_slab_forks_once_over_its_rows(language, program, clause, dtype, reference):
    """The rows of ``a[:, 2, :]`` are not one contiguous run: the team shares the rows out, each row folds
    sequentially. A parallel entry per row forked once per row (ICON's levelmask, 27,000 regions per call)."""
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = f"cpf_reduce_{program.name}_{'c' if language == 'c' else 'cpp'}"
    a = slab(dtype)
    out = np.zeros(1, dtype=dtype)

    rendering = cpf.render(sdfg, language=language)
    call_standalone(
        build_standalone(rendering.code, sdfg.name, language=language),
        rendering.sdfg,
        {"a": a, "out": out, "B": a.shape[0], "L": a.shape[1], "N": a.shape[2]},
    )

    assert rendering.code.count("omp parallel") == 1 and clause in rendering.code, rendering.code
    assert "reduce_seq_" in rendering.code, rendering.code
    assert out[0] == reference(a)


def test_a_c_form_using_bool_builds_before_c23():
    """gcc 14 defaults to ``gnu17``, where ``bool`` needs ``<stdbool.h>``."""
    sdfg = slab_any.to_sdfg(simplify=True)
    sdfg.name = "cpf_reduce_slab_any_gnu17"
    a = slab(np.bool_)
    out = np.zeros(1, dtype=np.bool_)

    rendering = cpf.render(sdfg, language="c")
    library = build_standalone(rendering.code, sdfg.name, language="c", extra_flags=("-std=gnu17",))
    call_standalone(library, rendering.sdfg, {"a": a, "out": out, "B": 40, "L": 5, "N": 33})

    assert out[0]


@pytest.mark.parametrize("language", ["c", "c++"])
def test_a_complex_sum_reduces_through_the_combiner_it_declares(language):
    sdfg = complex_sum.to_sdfg(simplify=True)
    sdfg.name = f"cpf_reduce_complex_sum_{'c' if language == 'c' else 'cpp'}"
    rng = np.random.default_rng(5)
    a = rng.random(500) + 1j * rng.random(500)
    out = np.zeros(1, dtype=np.complex128)

    rendering = cpf.render(sdfg, language=language)
    call_standalone(
        build_standalone(rendering.code, sdfg.name, language=language),
        rendering.sdfg,
        {"a": a, "out": out, "N": a.size},
    )

    assert "reduction(cpf_cadd:acc)" in rendering.code and "critical" not in rendering.code, rendering.code
    assert np.allclose(out, [a.sum()], rtol=1e-12)


def test_a_c_reduction_is_vectorizable_and_keeps_its_value():
    """CPF's C reduce helper carries ``simd`` and a unit-stride loop: without them a strict-FP compiler keeps the
    source's order and runs the sum scalar, 2.6x slower than a raw spelling with split accumulators (E5 sum at one
    thread). The reduction clause is what lets it reassociate."""
    sdfg = full_sum.to_sdfg(simplify=True)
    sdfg.name = "cpf_reduce_simd_sum_c"
    a = (np.random.default_rng(5).random(1000) * 1000).astype(np.int32)
    out = np.zeros(1, dtype=np.int32)
    rendering = cpf.render(sdfg, language="c")
    call_standalone(
        build_standalone(rendering.code, sdfg.name, language="c"), rendering.sdfg, {"a": a, "out": out, "N": a.size}
    )
    assert "#pragma omp parallel for simd reduction(+:acc)" in rendering.code, rendering.code
    assert "if (stride == 1) {" in rendering.code, rendering.code
    assert np.array_equal(out, np.array([a.sum()], dtype=np.int32))


if __name__ == "__main__":
    for lang in ("c", "c++"):
        test_a_full_reduction_is_one_openmp_reduction_loop(
            lang, full_max, "reduction(max:acc)", np.float64, lambda a: np.array([a.max()])
        )
        test_a_full_reduction_is_one_openmp_reduction_loop(
            lang, full_sum, "reduction(+:acc)", np.int32, lambda a: np.array([a.sum()], dtype=np.int32)
        )
        test_a_reduction_under_its_parallel_output_loop_folds_each_row_sequentially(lang)
        test_a_reduction_over_a_strided_slab_forks_once_over_its_rows(
            lang, slab_max, "reduction(max:_acc)", np.float64, lambda a: a[:, 2, :].max()
        )
        test_a_reduction_over_a_strided_slab_forks_once_over_its_rows(
            lang, slab_any, "reduction(||:_acc)", np.bool_, lambda a: a[:, 2, :].any()
        )
        test_a_complex_sum_reduces_through_the_combiner_it_declares(lang)
    test_a_c_form_using_bool_builds_before_c23()
    test_a_c_reduction_is_vectorizable_and_keeps_its_value()
