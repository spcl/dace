# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The pure Dot expansion initializes a result in device memory on the device."""

import dace
import dace.libraries.blas as blas


def test_pure_dot_initializes_device_result_on_device():
    """After the GPU transformation ``_result`` lives in GPU global memory: a bare host tasklet zeroing it is
    rejected by validation ("stored as GPU_Global but accessed on host"); the init must run on the device."""

    @dace.program
    def dot(x: dace.float64[20], y: dace.float64[20]):
        return x @ y

    sdfg = dot.to_sdfg()
    sdfg.apply_gpu_transformations()
    previous = blas.default_implementation
    blas.default_implementation = "pure"
    try:
        sdfg.expand_library_nodes()
    finally:
        blas.default_implementation = previous
    sdfg.validate()


if __name__ == "__main__":
    test_pure_dot_initializes_device_result_on_device()
