# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""One reproducer per wrong or missing fact the second usability review found in the CPF loop comments.

Each test renders a few-line program the way the bench does and checks the line the finding is about."""

import dace
from tests.codegen.cpf.conftest import render_gpu

N = dace.symbol("N")


def comment_above(code: str, marker: str) -> list[str]:
    """The comment lines right above the first line of ``code`` that contains ``marker``, past pragmas and braces."""
    lines = code.splitlines()
    at = next(n for n, line in enumerate(lines) if marker in line)
    above = []
    while at > 0 and (lines[at - 1].strip().startswith(("//", "#pragma")) or lines[at - 1].strip() == "{"):
        at -= 1
        above.insert(0, lines[at].strip())
    return [line for line in above if line.startswith("//")]


@dace.program
def dot(a: dace.float64[N], b: dace.float64[N], r: dace.float64[1]):
    for i in range(N):
        r[0] += a[i] * b[i]


def test_a_kernel_library_loop_names_the_kernel_not_its_launch_bounds():
    """Finding 4: the function name was read off ``__launch_bounds__(512)``."""
    device = render_gpu(dot, "cpf_review2_bounds", "cuda").device_code
    assert "function __launch_bounds__" not in device, device
    assert "library loop (kernel cpf_review2_bounds_" in device, device


if __name__ == "__main__":
    test_a_kernel_library_loop_names_the_kernel_not_its_launch_bounds()
