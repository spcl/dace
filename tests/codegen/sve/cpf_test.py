# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CPF renders Arm SVE maps: a self-contained unit (``arm_sve.h`` only, no DaCe runtime) that an SVE compiler
accepts, in both dialects. The render's own compile check proves the last part: it builds every unit with an
SVE compiler (an aarch64 cross compiler off aarch64) and raises when the unit does not compile."""

import pytest

import dace
from dace.codegen import cpf
from tests.codegen.sve import common

N = dace.symbol("N")
M = dace.symbol("M")


@dace.program
def axpy(X: dace.float64[N], Y: dace.float64[N], A: dace.float64):
    for i in dace.map[0:N]:
        with dace.tasklet:
            a << A
            x << X[i]
            y << Y[i]
            out >> Y[i]
            out = a * x + y


@dace.program
def gathered_sum(idx: dace.uint32[N], x: dace.float32[M], total: dace.float32[1]):
    for j in dace.map[0:N]:
        with dace.tasklet:
            v << x[idx[j]]
            out >> total(1, lambda a, b: a + b)[0]
            out = 2 * v


@dace.program
def filtered(A: dace.float32[N], out: dace.float32[N]):
    S = dace.define_stream(dace.float32, N)
    for i in dace.map[0:N]:
        with dace.tasklet:
            a << A[i]
            s >> S(-1)
            if a > 0.5:
                s = a
    S >> out


def assert_self_contained(code: str) -> None:
    assert "#include <arm_sve.h>" in code
    assert "dace::" not in code and "dace/" not in code and "#define" not in code


@pytest.mark.parametrize("language", ["c++", "c"])
def test_an_sve_map_renders_as_a_predicated_loop_an_sve_compiler_accepts(language):
    sut = cpf.render(common.vectorize(axpy), language=language)

    assert_self_contained(sut.code)
    assert "svwhilele_b64" in sut.code and "svcntd()" in sut.code


@pytest.mark.parametrize("language", ["c++", "c"])
def test_a_gather_and_a_reduction_render_without_the_runtime(language):
    sut = cpf.render(common.vectorize(gathered_sum), language=language)

    assert_self_contained(sut.code)
    assert "svld1_gather_index" in sut.code
    assert "svaddv(" in sut.code, "the register folds into the sum through an SVE reduction"


def test_a_stream_is_refused_by_name():
    with pytest.raises(NotImplementedError, match="Stream"):
        cpf.render(common.vectorize(filtered))


def test_an_sve_map_is_seen_through_a_nested_sdfg():
    outer = dace.SDFG("outer")
    outer.add_state().add_nested_sdfg(common.vectorize(axpy), {}, {})

    assert cpf.uses_sve(outer) and not cpf.uses_sve(dace.SDFG("empty"))
