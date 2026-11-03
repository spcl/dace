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
