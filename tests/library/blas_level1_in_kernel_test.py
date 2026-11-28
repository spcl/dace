# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``Axpy`` and ``Scal`` inside a GPU kernel run on the kernel's thread block, each lane on its share of the vector."""

import numpy as np
import pytest

from dace.libraries.blas import Axpy, Scal
from tests.library.blas_in_kernel_common import lowered_on_the_block, per_block, random_inputs, run_on_the_block

N = 300


def axpy_per_block(name: str):
    node = Axpy("axpy", a=2.5)
    sdfg, state = per_block(name, node, [("_x", "x", (N,), False), ("_y", "y", (N,), False), ("_res", "y", (N,), True)])
    return sdfg, state, node


def scal_per_block(name: str):
    node = Scal("scal", a=-3.0)
    sdfg, state = per_block(name, node, [("_x", "x", (N,), False), ("_res", "x", (N,), True)])
    return sdfg, state, node


def test_an_axpy_in_a_kernel_lowers_to_the_block():
    code = lowered_on_the_block(*axpy_per_block("axpy_block_lowering"))
    assert "__a * __x[" in code


def test_a_scal_in_a_kernel_lowers_to_the_block():
    code = lowered_on_the_block(*scal_per_block("scal_block_lowering"))
    assert "__a * __x[" in code


@pytest.mark.gpu
def test_an_axpy_on_the_block_matches_numpy():
    built = axpy_per_block("axpy_block_run")
    inputs = random_inputs({"x": (N,), "y": (N,)})
    out = run_on_the_block(built, inputs)
    np.testing.assert_allclose(out["y"], 2.5 * inputs["x"] + inputs["y"], rtol=1e-14)


@pytest.mark.gpu
def test_a_scal_on_the_block_matches_numpy():
    built = scal_per_block("scal_block_run")
    inputs = random_inputs({"x": (N,)})
    out = run_on_the_block(built, inputs)
    np.testing.assert_allclose(out["x"], -3.0 * inputs["x"], rtol=1e-14)


if __name__ == "__main__":
    test_an_axpy_in_a_kernel_lowers_to_the_block()
    test_a_scal_in_a_kernel_lowers_to_the_block()
    test_an_axpy_on_the_block_matches_numpy()
    test_a_scal_on_the_block_matches_numpy()
