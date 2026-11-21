# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A copy between two containers runs no code, so it does not decide which side of the machine either lives on."""

import numpy as np
import pytest

import dace
from dace.transformation.passes.canonicalize.finalize import apply_gpu_storage

N = dace.symbol("N")


@dace.program
def bump_and_return(x: dace.float64[N]):
    for t in range(3):
        for i in dace.map[0:N]:
            x[i] = x[i] + 1.0
    return x


def offloaded() -> dace.SDFG:
    sdfg = bump_and_return.to_sdfg(simplify=True)
    apply_gpu_storage(sdfg)
    sdfg.apply_gpu_transformations(validate=True)
    return sdfg


def test_a_returned_device_array_is_copied_on_the_device():
    """``return x`` after a loop of kernels is a top-level copy ``x -> __return``, both in device memory. Read
    as host code, it staged ``__return`` on the host: ``x`` went through ``buffer__return_host`` into
    ``__return`` and back again at exit, three whole-array copies for none, and a consumer looking for the
    return slot filled by a copy of an argument no longer found one (HB minife's GPU render)."""
    sdfg = offloaded()
    assert not any(name.startswith("buffer__return") for name in sdfg.arrays), sorted(sdfg.arrays)
    copies = [
        (edge.src.data, edge.dst.data)
        for state in sdfg.states()
        for edge in state.edges()
        if isinstance(edge.src, dace.nodes.AccessNode) and isinstance(edge.dst, dace.nodes.AccessNode)
    ]
    assert ("x", "__return") in copies, copies
    assert sdfg.arrays["__return"].storage == dace.StorageType.GPU_Global


@pytest.mark.gpu
def test_a_returned_device_array_computes_what_numpy_computes():
    import cupy  # GPU-only dependency; a CPU collection of this file must not need it

    sdfg = offloaded()
    x = cupy.arange(16, dtype=np.float64)
    result = sdfg(x=x, N=16)
    np.testing.assert_allclose(cupy.asnumpy(result), np.arange(16) + 3.0)


if __name__ == "__main__":
    test_a_returned_device_array_is_copied_on_the_device()
