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


def test_nanobind_interface_rename_own_build_folder():
    """A collision-renamed program is compiled into its own build folder, not in-place."""
    import os

    with set_temporary('compiler', 'interface', value='nanobind'):
        N = dace.symbol('N')

        @dace.program
        def axpy_nanobind_ownfolder(A: dace.float64[N], B: dace.float64[N], alpha: dace.float64):
            B[:] = alpha * A + B

        sdfg1 = axpy_nanobind_ownfolder.to_sdfg()
        base_name = sdfg1.name
        original_folder = sdfg1.build_folder
        sdfg1.compile()

        csdfg2 = axpy_nanobind_ownfolder.to_sdfg().compile()
        renamed = csdfg2.sdfg.name
        assert renamed == f'{base_name}_0'

        # Own folder, derived from the new name - no artifacts of the renamed
        # program inside the original build folder.
        renamed_folder = csdfg2.sdfg.build_folder
        assert os.path.basename(renamed_folder) == renamed
        assert os.path.isfile(os.path.join(renamed_folder, 'INTERFACE'))
        assert not os.path.isfile(os.path.join(original_folder, 'build', f'lib{renamed}.so'))


def test_nanobind_interface_name_collision_error():
    """With compiler.nanobind_name_collision=error, a taken name refuses to compile."""
    import pytest

    with set_temporary('compiler', 'interface', value='nanobind'):
        with set_temporary('compiler', 'nanobind_name_collision', value='error'):
            N = dace.symbol('N')

            @dace.program
            def axpy_nanobind_collerr(A: dace.float64[N], B: dace.float64[N], alpha: dace.float64):
                B[:] = alpha * A + B

            axpy_nanobind_collerr.to_sdfg().compile()
            with pytest.raises(Exception, match='already loaded'):
                axpy_nanobind_collerr.to_sdfg().compile()


def test_nanobind_interface_workspace():
    """External-memory workspace functions work on the nanobind interface."""
    with set_temporary('compiler', 'interface', value='nanobind'):
        N = dace.symbol('N')

        @dace.program
        def extmem_nanobind(a: dace.float64[N]):
            workspace = dace.ndarray([N], dace.float64, lifetime=dace.AllocationLifetime.External)
            workspace[:] = a
            workspace += 1
            a[:] = workspace

        csdfg = extmem_nanobind.to_sdfg().compile()

        n = 20
        a = np.random.rand(n)
        # Positional: `a` must map to the SDFG argument `a` (user-facing
        # order), NOT to the C++ initialize's first parameter `N`.
        csdfg.initialize(a, N=np.int32(n))
        sizes = csdfg.get_workspace_sizes()
        assert sizes == {dace.StorageType.CPU_Heap: n * 8}

        wsp = np.random.rand(n)
        csdfg.set_workspace(dace.StorageType.CPU_Heap, wsp)

        ref = a + 1
        csdfg(a=a, N=np.int32(n))
        assert np.allclose(a, ref)
        assert np.allclose(wsp, ref)

        # The state-struct field names are baked in at codegen time; the
        # external workspace pointer is one of them.
        fields = csdfg.state_fields()
        assert isinstance(fields, list) and len(fields) > 0
        assert any('workspace' in f for f in fields)

        # get_state_struct returns the raw state pointer on this interface,
        # unlike the ctypes path which returns a ctypes.Structure view.
        assert csdfg.get_state_struct() == csdfg._handle.state_pointer


def test_nanobind_interface_get_exported_function():
    """Arbitrary exported symbols stay reachable, with the wrapper as keep-alive."""
    with set_temporary('compiler', 'interface', value='nanobind'):
        N = dace.symbol('N')

        @dace.program
        def axpy_nanobind_expfun(A: dace.float64[N], B: dace.float64[N], alpha: dace.float64):
            B[:] = alpha * A + B

        csdfg = axpy_nanobind_expfun.to_sdfg().compile()
        func = csdfg.get_exported_function(f'__dace_exit_{csdfg.sdfg.name}')
        assert func is not None
        assert func.__compiled_sdfg__ is csdfg
        assert csdfg.get_exported_function('definitely_not_a_symbol') is None


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


def test_nanobind_interface_string_argument():
    """A ``dtypes.string`` scalar argument marshals a Python ``str`` (and ``None``) into the kernel.

    The kernel reads the first byte of the string, or writes -1 when the pointer
    is null - so passing a ``str`` observes the bytes, and passing ``None``
    observes the null-pointer path (matching the ctypes marshaller).
    """
    with set_temporary('compiler', 'interface', value='nanobind'):

        @dace.program
        def string_arg_nanobind(s: str, out: dace.int8[1]):

            @dace.tasklet('CPP')
            def read():
                sin << s
                o >> out[0]
                """
                o = (sin == nullptr) ? -1 : sin[0];
                """

        csdfg = string_arg_nanobind.to_sdfg().compile()
        assert isinstance(csdfg, dace.codegen.nanobind_compiled_sdfg.NanobindCompiledSDFG)

        out = np.zeros(1, dtype=np.int8)
        csdfg(s='A', out=out)
        assert out[0] == ord('A')  # 65 - the first byte reached the kernel
        csdfg(s='Zoo', out=out)
        assert out[0] == ord('Z')  # first byte only
        csdfg(s=None, out=out)
        assert out[0] == -1  # None arrived as a null pointer


def test_nanobind_interface_optional_array():
    """An optional (nullable) array accepts both a real array and ``None`` (a null pointer)."""
    from typing import Optional

    with set_temporary('compiler', 'interface', value='nanobind'):

        @dace.program
        def optional_arg_nanobind(a: Optional[dace.float64[1]], out: dace.float64[1]):
            if a is None:
                out[0] = -1.0
            else:
                out[0] = a[0]

        sdfg = optional_arg_nanobind.to_sdfg()
        assert sdfg.arrays['a'].optional is True
        csdfg = sdfg.compile()

        out = np.zeros(1)
        csdfg(a=np.array([3.5]), out=out)
        assert out[0] == 3.5  # a real array is passed by reference

        out = np.zeros(1)
        csdfg(a=None, out=out)
        assert out[0] == -1.0  # None arrived as a null pointer


def test_nanobind_interface_nullable_args_enable_none():
    """Nullable arguments emit ``nb::arg(...).none()`` in the generated bindings.

    nanobind rejects None-valued arguments unless ``.none()`` is set;
    ``std::optional`` accepts None implicitly only on some nanobind versions, so
    the generator must opt in explicitly. This is a version-independent guard
    (the end-to-end None tests pass regardless of .none() on lenient nanobinds).
    """
    from dace import dtypes
    from dace.codegen.nanobind_bindings import generate_bindings_code

    sdfg = dace.SDFG('nullable_probe')
    sdfg.add_scalar('s', dtypes.string)
    sdfg.add_array('opt', [1], dtypes.float64)
    sdfg.arrays['opt'].optional = True
    sdfg.add_array('req', [1], dtypes.float64)
    sdfg.arrays['req'].optional = False

    code = generate_bindings_code(sdfg)
    # String scalar and optional array must accept None.
    assert 'nb::arg("s").none()' in code
    assert 'nb::arg("opt").noconvert().none()' in code
    # A non-optional array must NOT accept None.
    assert 'nb::arg("req").noconvert()' in code
    assert 'nb::arg("req").noconvert().none()' not in code


if __name__ == '__main__':
    test_axpy_nanobind_interface()
    test_nanobind_interface_wrong_dtype_raises()
    test_nanobind_interface_same_name_recompile()
    test_nanobind_interface_return_value()
    test_nanobind_interface_positional_and_extra_kwargs()
    test_nanobind_interface_has_gpu_code()
    test_nanobind_interface_state_pointer()
    test_nanobind_interface_pyobject_rejected()
    test_nanobind_interface_string_argument()
    test_nanobind_interface_optional_array()
    test_nanobind_interface_nullable_args_enable_none()
