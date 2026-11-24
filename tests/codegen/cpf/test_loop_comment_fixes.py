# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""One reproducer per wrong or missing fact a usability review found in the CPF loop comments.

Each test renders a few-line program and checks the comment string of the loop the finding is about."""

import dace
from tests.codegen.cpf.test_specialization_hints import named

N = dace.symbol("N")
T = dace.symbol("T")


def class_lines(code: str) -> list[str]:
    """The class comments of ``code``, without their indentation."""
    return [
        line.strip()
        for line in code.splitlines()
        if line.strip().startswith(("// parallel", "// sequential", "// unknown"))
    ]


def rendered(program: dace.program, name: str) -> str:
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = name
    return named(sdfg)


def test_a_sequential_loop_over_an_indirect_access_offers_no_wavefront():
    """The running example: ``x[col[j]]`` has no affine skew, so the wavefront alternative was false."""

    @dace.program
    def indirect(x: dace.float64[N], y: dace.float64[N], col: dace.int64[N]):
        for t in range(T):
            for j in dace.map[0:N]:
                y[j] = x[col[j]]
            for j in dace.map[0:N]:
                x[j] = y[j]

    outer = class_lines(rendered(indirect, "cpf_comment_wavefront"))[0]
    assert outer.startswith("// sequential -- proven dependence on x[0:N]"), outer
    assert "wavefront" not in outer and "alt:" not in outer, outer


if __name__ == "__main__":
    test_a_sequential_loop_over_an_indirect_access_offers_no_wavefront()
