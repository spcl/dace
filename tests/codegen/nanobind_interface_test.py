# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the nanobind-based `CompiledSDFG` interface (`compiler.interface=nanobind`)."""
import sys

import numpy as np

import dace
from dace.config import set_temporary


def test_axpy_nanobind_interface():
    """Stage-1 acceptance: an axpy-class SDFG runs end-to-end on the nanobind interface."""
    with set_temporary('compiler', 'interface', value='nanobind'):
        N = dace.symbol('N')

        @dace.program
        def axpy_nanobind(A: dace.float64[N], B: dace.float64[N], alpha: dace.float64):
            B[:] = alpha * A + B

        sdfg = axpy_nanobind.to_sdfg()
        csdfg = sdfg.compile()

        n = 32
        a = np.random.rand(n)
        b = np.random.rand(n)
        expected = 2.0 * a + b
        csdfg(A=a, B=b, alpha=np.float64(2.0), N=np.int32(n))

        assert np.allclose(b, expected)
        # The module is registered under the dace.generated.* namespace.
        assert f'dace.generated.{sdfg.name}' in sys.modules
        # The stub-based loader is not involved on this path.
        assert type(csdfg).__name__ == 'NanobindCompiledSDFG'


def test_nanobind_interface_wrong_dtype_raises():
    """A wrong-dtype array is rejected by the generated marshalling code with a typed error."""
    import pytest

    with set_temporary('compiler', 'interface', value='nanobind'):
        N = dace.symbol('N')

        @dace.program
        def axpy_nanobind_dtype(A: dace.float64[N], B: dace.float64[N], alpha: dace.float64):
            B[:] = alpha * A + B

        csdfg = axpy_nanobind_dtype.to_sdfg().compile()
        n = 8
        a = np.random.rand(n).astype(np.float32)  # wrong dtype
        b = np.random.rand(n)
        with pytest.raises(Exception):
            csdfg(A=a, B=b, alpha=np.float64(2.0), N=np.int32(n))


if __name__ == '__main__':
    test_axpy_nanobind_interface()
    test_nanobind_interface_wrong_dtype_raises()
