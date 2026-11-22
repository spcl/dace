# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A CPF unit carries no text an agent reads for nothing: a scalar one tasklet writes and one other reads is folded
into its reader, and a 1-D unit-stride array is indexed directly instead of through an identity ``<array>_idx``."""

import re

import numpy as np

import dace
from dace.codegen.cpf import render
from tests.codegen.cpf.conftest import build_standalone, call_standalone

N = dace.symbol("N", dtype=dace.int64)


@dace.program
def stencil_1d(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[1 : N - 1]:
        tmp = dace.define_local_scalar(dace.float64)
        with dace.tasklet:
            left << a[i - 1]
            mid << a[i]
            right << a[i + 1]
            curvature >> tmp
            curvature = left - 2.0 * mid + right
        with dace.tasklet:
            c << tmp
            m << a[i]
            out >> b[i]
            out = 0.25 * c + m


def test_single_use_temporaries_fold_into_their_reader_and_compute_the_same():
    sdfg = stencil_1d.to_sdfg(simplify=True)
    sdfg.name = "token_cuts_stencil"
    code = render(sdfg, language="c").code
    body = code[code.index("token_cuts_stencil(") :]
    assert not re.search(r"^\s*double \w+;$", body, re.MULTILINE), body
    rng = np.random.default_rng(3)
    for n in (3, 17):
        a, b = rng.random(n), np.zeros(n)
        call_standalone(build_standalone(code, "token_cuts_stencil", language="c"), sdfg, {"a": a, "b": b, "N": n})
        want = np.zeros(n)
        want[1:-1] = 0.25 * (a[:-2] - 2.0 * a[1:-1] + a[2:]) + a[1:-1]
        np.testing.assert_array_equal(b[1:-1], want[1:-1])


def test_a_unit_stride_vector_is_indexed_without_an_identity_helper():
    sdfg = stencil_1d.to_sdfg(simplify=True)
    sdfg.name = "token_cuts_identity"
    code = render(sdfg, language="c").code
    assert not re.search(r"_idx\(int64_t i0\) \{ return i0; \}", code), code
    assert re.search(r"\ba\[\(?i", code), code


if __name__ == "__main__":
    test_single_use_temporaries_fold_into_their_reader_and_compute_the_same()
    test_a_unit_stride_vector_is_indexed_without_an_identity_helper()
