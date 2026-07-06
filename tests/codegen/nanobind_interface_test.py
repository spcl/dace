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
        assert isinstance(csdfg, dace.codegen.nanobind_compiled_sdfg.NanobindCompiledSDFG)

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
        assert isinstance(csdfg, dace.codegen.nanobind_compiled_sdfg.NanobindCompiledSDFG)
        n = 8
        a = np.random.rand(n).astype(np.float32)  # wrong dtype
        b = np.random.rand(n)
        with pytest.raises(Exception):
            csdfg(A=a, B=b, alpha=np.float64(2.0), N=np.int32(n))


def test_nanobind_interface_same_name_recompile():
    """Recompiling under an already-imported module name silently renames (sys.modules increment)."""
    with set_temporary('compiler', 'interface', value='nanobind'):
        N = dace.symbol('N')

        @dace.program
        def axpy_nanobind_rename(A: dace.float64[N], B: dace.float64[N], alpha: dace.float64):
            B[:] = alpha * A + B

        sdfg1 = axpy_nanobind_rename.to_sdfg()
        base_name = sdfg1.name
        csdfg1 = sdfg1.compile()
        assert isinstance(csdfg1, dace.codegen.nanobind_compiled_sdfg.NanobindCompiledSDFG)

        # Fresh SDFG with the same name: the module name is taken, so the
        # compile path must rename and recompile instead of silently reusing
        # the already-imported (stale) module.
        sdfg2 = axpy_nanobind_rename.to_sdfg()
        assert sdfg2.name == base_name
        csdfg2 = sdfg2.compile()
        assert isinstance(csdfg2, dace.codegen.nanobind_compiled_sdfg.NanobindCompiledSDFG)

        n = 16
        a = np.random.rand(n)
        b = np.random.rand(n)
        expected = 3.0 * a + b
        csdfg2(A=a, B=b, alpha=np.float64(3.0), N=np.int32(n))
        assert np.allclose(b, expected)
        assert f'dace.generated.{base_name}_0' in sys.modules


def test_nanobind_interface_return_value():
    """A program with a return array allocates it in Python and returns it."""
    with set_temporary('compiler', 'interface', value='nanobind'):
        N = dace.symbol('N')

        @dace.program
        def add_one_nanobind(A: dace.float64[N]):
            return A + 1.0

        csdfg = add_one_nanobind.to_sdfg().compile()
        assert isinstance(csdfg, dace.codegen.nanobind_compiled_sdfg.NanobindCompiledSDFG)
        n = 24
        a = np.random.rand(n)
        result = csdfg(A=a, N=np.int32(n))
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, a + 1.0)


def test_nanobind_interface_positional_and_extra_kwargs():
    """Positional calls work, and extra keyword arguments are absorbed (old-interface behavior)."""
    with set_temporary('compiler', 'interface', value='nanobind'):
        N = dace.symbol('N')

        @dace.program
        def axpy_nanobind_pos(A: dace.float64[N], B: dace.float64[N], alpha: dace.float64):
            B[:] = alpha * A + B

        csdfg = axpy_nanobind_pos.to_sdfg().compile()
        n = 16
        a = np.random.rand(n)
        b = np.random.rand(n)
        expected = 2.0 * a + b
        csdfg(a, b, np.float64(2.0), N=np.int32(n), unused_extra_argument=42)
        assert np.allclose(b, expected)


def test_nanobind_interface_has_gpu_code():
    """The handle and the shell expose has_gpu_code (False for a CPU-only SDFG)."""
    with set_temporary('compiler', 'interface', value='nanobind'):
        N = dace.symbol('N')

        @dace.program
        def axpy_nanobind_gpuq(A: dace.float64[N], B: dace.float64[N], alpha: dace.float64):
            B[:] = alpha * A + B

        csdfg = axpy_nanobind_gpuq.to_sdfg().compile()
        assert csdfg.has_gpu_code is False


def test_nanobind_interface_state_pointer():
    """state_pointer raises while the state is uninitialized or after finalize."""
    import pytest

    with set_temporary('compiler', 'interface', value='nanobind'):
        N = dace.symbol('N')

        @dace.program
        def axpy_nanobind_stateptr(A: dace.float64[N], B: dace.float64[N], alpha: dace.float64):
            B[:] = alpha * A + B

        csdfg = axpy_nanobind_stateptr.to_sdfg().compile()
        handle = csdfg._handle

        with pytest.raises(RuntimeError):
            handle.state_pointer  # not initialized yet

        n = 8
        a = np.random.rand(n)
        b = np.random.rand(n)
        csdfg(A=a, B=b, alpha=np.float64(2.0), N=np.int32(n))
        assert isinstance(csdfg, dace.codegen.nanobind_compiled_sdfg.NanobindCompiledSDFG)
        assert handle.state_pointer != 0

        csdfg.finalize()
        with pytest.raises(RuntimeError):
            handle.state_pointer  # finalized


def test_nanobind_interface_pyobject_rejected():
    """pyobject arguments/returns are rejected with a clear error at codegen time.

    pyobject support (including the PR#2206 bug-compatible decay of pyobject
    arrays) is deferred to part 2 of the port; until then the generator must
    refuse instead of emitting C++ that does not compile.
    """
    import pytest
    from dace import dtypes
    from dace.codegen.nanobind_bindings import generate_bindings_code

    sdfg = dace.SDFG('pyobject_reject_probe')
    sdfg.add_array('__return', [1], dtypes.pyobject())
    with pytest.raises(NotImplementedError, match='pyobject'):
        generate_bindings_code(sdfg)


if __name__ == '__main__':
    test_axpy_nanobind_interface()
    test_nanobind_interface_wrong_dtype_raises()
    test_nanobind_interface_same_name_recompile()
    test_nanobind_interface_return_value()
    test_nanobind_interface_positional_and_extra_kwargs()
    test_nanobind_interface_has_gpu_code()
    test_nanobind_interface_state_pointer()
    test_nanobind_interface_pyobject_rejected()
