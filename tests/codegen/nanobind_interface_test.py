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
    """pyobject returns are dropped (arrays only); pyobject arguments are deferred to part 2.

    Both are rejected at codegen (the generator must refuse instead of emitting
    C++ that does not compile), but with distinct messages: a pyobject return is
    permanently unsupported (the nanobind interface returns arrays only), while a
    pyobject argument (callbacks) is deferred to part 2 of the port.
    """
    import pytest
    from dace import dtypes
    from dace.codegen.nanobind_bindings import generate_bindings_code

    # pyobject return value: dropped, arrays-only message.
    ret_sdfg = dace.SDFG('pyobject_return_reject_probe')
    ret_sdfg.add_array('__return', [1], dtypes.pyobject())
    with pytest.raises(NotImplementedError, match='arrays only'):
        generate_bindings_code(ret_sdfg)

    # pyobject argument: deferred to part 2 (callbacks).
    arg_sdfg = dace.SDFG('pyobject_arg_reject_probe')
    arg_sdfg.add_scalar('cb', dtypes.pyobject())
    with pytest.raises(NotImplementedError, match='part 2'):
        generate_bindings_code(arg_sdfg)


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


def test_nanobind_interface_load_reuses_same_artifact():
    """Loading the same artifact path again reuses the module (one module, many handles)."""
    from dace.codegen.compiler import load_nanobind_module

    with set_temporary('compiler', 'interface', value='nanobind'):
        N = dace.symbol('N')

        @dace.program
        def load_reuse_nanobind(A: dace.float64[N], B: dace.float64[N], alpha: dace.float64):
            B[:] = alpha * A + B

        csdfg = load_reuse_nanobind.to_sdfg().compile()
        module = load_nanobind_module(csdfg.module.__file__, csdfg.sdfg.name)
        assert module is csdfg.module


def test_nanobind_interface_load_distinct_artifact_raises():
    """A distinct artifact under an already-loaded generated name is a hard error.

    ``load_precompiled_sdfg`` loads a fixed prebuilt artifact and bypasses the
    compile-time rename, so a second, different artifact under a name already
    taken in ``sys.modules`` must fail loudly (extension modules cannot be
    re-imported) instead of silently returning the stale module.
    """
    import os
    import shutil
    import tempfile

    import pytest
    from dace.codegen.compiler import load_nanobind_module

    with set_temporary('compiler', 'interface', value='nanobind'):
        N = dace.symbol('N')

        @dace.program
        def load_distinct_nanobind(A: dace.float64[N], B: dace.float64[N], alpha: dace.float64):
            B[:] = alpha * A + B

        csdfg = load_distinct_nanobind.to_sdfg().compile()
        # A copy of the .so at a different path is a distinct artifact.
        copied = os.path.join(tempfile.mkdtemp(), os.path.basename(csdfg.module.__file__))
        shutil.copy(csdfg.module.__file__, copied)
        with pytest.raises(ValueError, match='already loaded'):
            load_nanobind_module(copied, csdfg.sdfg.name)


def test_nanobind_interface_safe_call():
    """safe_call runs the SDFG in a subprocess: it forwards in/out output, and a
    crash (writing to a null pointer) surfaces as an exception instead of killing
    the calling process."""
    import pytest

    with set_temporary('compiler', 'interface', value='nanobind'):

        @dace.program
        def safe_call_nanobind(A: dace.float64[5], B: dace.float64[5], ub: dace.int64):
            for i in range(5):
                with dace.tasklet('CPP'):
                    b << B[i]
                    u << ub
                    a >> A[i]
                    """
                    if (u == 0) { *((double*)nullptr) = 42.0; }
                    a = b + 1;
                    """

        csdfg = safe_call_nanobind.to_sdfg().compile()

        A = np.zeros(5)
        B = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        csdfg.safe_call(A, B, 5)
        assert np.allclose(A, B + 1)  # in/out array forwarded back from the subprocess

        # The null write in the subprocess must raise here, not crash the test.
        A = np.zeros(5)
        B = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.raises(RuntimeError):
            csdfg.safe_call(A, B, 0)


def test_nanobind_interface_safe_call_kwargs():
    """safe_call accepts the keyword-argument call form."""
    with set_temporary('compiler', 'interface', value='nanobind'):

        @dace.program
        def safe_call_kwargs_nanobind(A: dace.float64[5], B: dace.float64[5], ub: dace.int64):
            for i in range(5):
                with dace.tasklet('CPP'):
                    b << B[i]
                    u << ub
                    a >> A[i]
                    """
                    a = b + 1;
                    """

        csdfg = safe_call_kwargs_nanobind.to_sdfg().compile()
        A = np.zeros(5)
        B = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        csdfg.safe_call(A=A, B=B, ub=5)
        assert np.allclose(A, B + 1)


def test_nanobind_interface_safe_call_return_rejected():
    """safe_call does not support return values (parity with the ctypes path)."""
    import pytest

    with set_temporary('compiler', 'interface', value='nanobind'):

        @dace.program
        def safe_call_return_nanobind(A: dace.float64[5]):
            return A + 1

        csdfg = safe_call_return_nanobind.to_sdfg().compile()
        with pytest.raises(NotImplementedError):
            csdfg.safe_call(np.zeros(5))


def _build_csr_to_dense(name, nested):
    """Builds a CSR-to-dense SDFG whose input ``A`` is a (optionally nested) Structure.

    Returns ``(sdfg, csr_obj, wrapper_obj_or_None)``.
    """
    M, N, nnz = (dace.symbol(s) for s in ('M', 'N', 'nnz'))
    csr_obj = dace.data.Structure(dict(indptr=dace.int32[M + 1], indices=dace.int32[nnz], data=dace.float32[nnz]),
                                  name='CSRMatrix')
    wrapper_obj = dace.data.Structure(dict(csr=csr_obj), name='Wrapper') if nested else None

    sdfg = dace.SDFG(name)
    sdfg.add_datadesc('A', wrapper_obj if nested else csr_obj)
    sdfg.add_array('B', [M, N], dace.float32)

    spmat = wrapper_obj.members['csr'] if nested else csr_obj
    prefix = 'A.csr' if nested else 'A'
    sdfg.add_view('vindptr', spmat.members['indptr'].shape, spmat.members['indptr'].dtype)
    sdfg.add_view('vindices', spmat.members['indices'].shape, spmat.members['indices'].dtype)
    sdfg.add_view('vdata', spmat.members['data'].shape, spmat.members['data'].dtype)

    state = sdfg.add_state()
    A = state.add_access('A')
    B = state.add_access('B')
    indptr = state.add_access('vindptr')
    indices = state.add_access('vindices')
    data = state.add_access('vdata')

    state.add_edge(A, None, indptr, 'views', dace.Memlet.from_array(f'{prefix}.indptr', spmat.members['indptr']))
    state.add_edge(A, None, indices, 'views', dace.Memlet.from_array(f'{prefix}.indices', spmat.members['indices']))
    state.add_edge(A, None, data, 'views', dace.Memlet.from_array(f'{prefix}.data', spmat.members['data']))

    ime, imx = state.add_map('i', dict(i='0:M'))
    jme, jmx = state.add_map('idx', dict(idx='start:stop'))
    jme.add_in_connector('start')
    jme.add_in_connector('stop')
    t = state.add_tasklet('indirection', {'j', '__val'}, {'__out'}, '__out[i, j] = __val')

    state.add_memlet_path(indptr, ime, jme, memlet=dace.Memlet(data='vindptr', subset='i'), dst_conn='start')
    state.add_memlet_path(indptr, ime, jme, memlet=dace.Memlet(data='vindptr', subset='i+1'), dst_conn='stop')
    state.add_memlet_path(indices, ime, jme, t, memlet=dace.Memlet(data='vindices', subset='idx'), dst_conn='j')
    state.add_memlet_path(data, ime, jme, t, memlet=dace.Memlet(data='vdata', subset='idx'), dst_conn='__val')
    state.add_memlet_path(t, jmx, imx, B, memlet=dace.Memlet(data='B', subset='0:M, 0:N', volume=1), src_conn='__out')
    return sdfg, csr_obj, wrapper_obj


def _csr_example():
    """A small CSR matrix (M=2, N=3, nnz=3) and its dense form, as contiguous numpy arrays."""
    indptr = np.array([0, 2, 3], dtype=np.int32)
    indices = np.array([0, 2, 1], dtype=np.int32)
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    expected = np.array([[1, 0, 2], [0, 3, 0]], dtype=np.float32)
    return indptr, indices, data, expected


def test_nanobind_interface_structure_argument():
    """A flat Structure argument is passed as a pointer to a user-built ctypes.Structure."""
    with set_temporary('compiler', 'interface', value='nanobind'):
        sdfg, csr_obj, _ = _build_csr_to_dense('csr_struct_nanobind', nested=False)
        csdfg = sdfg.compile()

        indptr, indices, data, expected = _csr_example()
        B = np.zeros((2, 3), dtype=np.float32)
        inpA = csr_obj.dtype._typeclass.as_ctypes()(indptr=indptr.__array_interface__['data'][0],
                                                    indices=indices.__array_interface__['data'][0],
                                                    data=data.__array_interface__['data'][0])
        csdfg(A=inpA, B=B, M=2, N=3, nnz=3)
        assert np.allclose(B, expected)


def test_nanobind_interface_nested_structure_argument():
    """A nested Structure argument (Wrapper(csr=...)) works via the same pointer passthrough."""
    import ctypes

    with set_temporary('compiler', 'interface', value='nanobind'):
        sdfg, csr_obj, wrapper_obj = _build_csr_to_dense('nested_csr_struct_nanobind', nested=True)
        csdfg = sdfg.compile()

        indptr, indices, data, expected = _csr_example()
        B = np.zeros((2, 3), dtype=np.float32)
        inpCSR = csr_obj.dtype._typeclass.as_ctypes()(indptr=indptr.__array_interface__['data'][0],
                                                      indices=indices.__array_interface__['data'][0],
                                                      data=data.__array_interface__['data'][0])
        inpW = wrapper_obj.dtype._typeclass.as_ctypes()(csr=ctypes.pointer(inpCSR))
        csdfg(A=inpW, B=B, M=2, N=3, nnz=3)
        assert np.allclose(B, expected)


def test_nanobind_interface_container_array_read():
    """ContainerArray argument (array of structures) on the nanobind interface.

    NOTE: verbatim copy of
    ``tests/sdfg/data/container_array_test.py::test_read_struct_array``, run
    under ``compiler.interface=nanobind`` - kept so the ContainerArray behaviour
    is validated against a known-good ctypes test with no doubt. Remove it once
    the ContainerArray tests are parametrized over both interfaces.
    """
    import ctypes

    import pytest
    sparse = pytest.importorskip('scipy.sparse')

    with set_temporary('compiler', 'interface', value='nanobind'):
        L, M, N, nnz = (dace.symbol(s) for s in ('L', 'M', 'N', 'nnz'))
        csr_obj = dace.data.Structure(dict(indptr=dace.int32[M + 1], indices=dace.int32[nnz], data=dace.float32[nnz]),
                                      name='CSRMatrix')

        sdfg = dace.SDFG('array_of_csr_to_dense')

        sdfg.add_datadesc('A', csr_obj[L])
        sdfg.add_array('B', [L, M, N], dace.float32)

        sdfg.add_datadesc_view('vcsr', csr_obj)
        sdfg.add_view('vindptr', csr_obj.members['indptr'].shape, csr_obj.members['indptr'].dtype)
        sdfg.add_view('vindices', csr_obj.members['indices'].shape, csr_obj.members['indices'].dtype)
        sdfg.add_view('vdata', csr_obj.members['data'].shape, csr_obj.members['data'].dtype)

        state = sdfg.add_state()

        A = state.add_access('A')
        B = state.add_access('B')

        bme, bmx = state.add_map('b', dict(b='0:L'))
        bme.map.schedule = dace.ScheduleType.Sequential

        vcsr = state.add_access('vcsr')
        indptr = state.add_access('vindptr')
        indices = state.add_access('vindices')
        data = state.add_access('vdata')

        state.add_memlet_path(A, bme, vcsr, dst_conn='views', memlet=dace.Memlet(data='A', subset='b'))
        state.add_edge(vcsr,
                       None,
                       indptr,
                       'views',
                       memlet=dace.Memlet.from_array('vcsr.indptr', csr_obj.members['indptr']))
        state.add_edge(vcsr,
                       None,
                       indices,
                       'views',
                       memlet=dace.Memlet.from_array('vcsr.indices', csr_obj.members['indices']))
        state.add_edge(vcsr, None, data, 'views', memlet=dace.Memlet.from_array('vcsr.data', csr_obj.members['data']))

        ime, imx = state.add_map('i', dict(i='0:M'))
        jme, jmx = state.add_map('idx', dict(idx='start:stop'))
        jme.add_in_connector('start')
        jme.add_in_connector('stop')
        t = state.add_tasklet('indirection', {'j', '__val'}, {'__out'}, '__out[i, j] = __val')

        state.add_memlet_path(indptr, ime, jme, memlet=dace.Memlet(data='vindptr', subset='i'), dst_conn='start')
        state.add_memlet_path(indptr, ime, jme, memlet=dace.Memlet(data='vindptr', subset='i+1'), dst_conn='stop')
        state.add_memlet_path(indices, ime, jme, t, memlet=dace.Memlet(data='vindices', subset='idx'), dst_conn='j')
        state.add_memlet_path(data, ime, jme, t, memlet=dace.Memlet(data='vdata', subset='idx'), dst_conn='__val')
        state.add_memlet_path(t,
                              jmx,
                              imx,
                              bmx,
                              B,
                              memlet=dace.Memlet(data='B', subset='b, 0:M, 0:N', volume=1),
                              src_conn='__out')

        func = sdfg.compile()

        rng = np.random.default_rng(42)
        A = np.ndarray((10, ), dtype=sparse.csr_matrix)
        dace_A = np.ndarray((10, ), dtype=ctypes.c_void_p)
        B = np.zeros((10, 20, 20), dtype=np.float32)

        ctypes_A = []
        for b in range(10):
            A[b] = sparse.random(20, 20, density=0.1, format='csr', dtype=np.float32, random_state=rng)
            ctypes_obj = csr_obj.dtype._typeclass.as_ctypes()(indptr=A[b].indptr.__array_interface__['data'][0],
                                                              indices=A[b].indices.__array_interface__['data'][0],
                                                              data=A[b].data.__array_interface__['data'][0])
            ctypes_A.append(ctypes_obj)  # This is needed to keep the object alive ...
            dace_A[b] = ctypes.addressof(ctypes_obj)

        func(A=dace_A, B=B, L=A.shape[0], M=A[0].shape[0], N=A[0].shape[1], nnz=A[0].nnz)
        ref = np.ndarray((10, 20, 20), dtype=np.float32)
        for b in range(10):
            ref[b] = A[b].toarray()

        assert np.allclose(B, ref)


def test_nanobind_interface_complex_array():
    """A complex128 array argument compiles and runs (dace::complex128 resolves via the dace type header)."""
    with set_temporary('compiler', 'interface', value='nanobind'):
        N = dace.symbol('N')

        @dace.program
        def complex_scale_nanobind(A: dace.complex128[N], B: dace.complex128[N]):
            B[:] = A + A

        csdfg = complex_scale_nanobind.to_sdfg().compile()
        assert isinstance(csdfg, dace.codegen.nanobind_compiled_sdfg.NanobindCompiledSDFG)

        n = 16
        a = (np.random.rand(n) + 1j * np.random.rand(n)).astype(np.complex128)
        b = np.zeros(n, dtype=np.complex128)
        csdfg(A=a, B=b, N=np.int32(n))
        assert np.allclose(b, a + a)


def test_nanobind_interface_includes_dace_type_headers():
    """The generated TU includes the dace runtime type headers (so dace:: scalar names resolve).

    Version-independent guard for the type-header fix: complex/unsigned ndarray
    scalar types are dace:: aliases of nanobind-supported scalars, but only once
    the header is included.
    """
    from dace import dtypes
    from dace.codegen.nanobind_bindings import generate_bindings_code

    sdfg = dace.SDFG('dace_type_header_probe')
    sdfg.add_array('c', [4], dtypes.complex128)
    code = generate_bindings_code(sdfg)
    assert '#include <dace/types.h>' in code
    assert '#include <dace/vector.h>' in code
    assert '#include <nanobind/stl/complex.h>' in code


def test_nanobind_interface_vector_array():
    """A vector (veclen) array binds as its base scalar and copies correctly.

    Reproduces the BLAS veclen failures: the ndarray scalar must be the base
    type (float), while the pointer handed to the kernel stays dace::vec<float,2>*.

    Both buffers are wrapped in sentinel padding on either side and only the
    interior is passed, so any access past the intended N vectors is caught as a
    corrupted guard region. The aligned vector type is wider than a plain scalar,
    so this keeps a future size miscalculation from silently over-reading or
    over-writing.
    """
    with set_temporary('compiler', 'interface', value='nanobind'):
        N = dace.symbol('N')
        vtype = dace.vector(dace.float32, 2)

        sdfg = dace.SDFG('vec_copy_nanobind')
        sdfg.add_array('x', [N], vtype)
        sdfg.add_array('y', [N], vtype)
        state = sdfg.add_state()
        state.add_edge(state.add_access('x'), None, state.add_access('y'), None, dace.Memlet('x[0:N]'))

        csdfg = sdfg.compile()
        assert isinstance(csdfg, dace.codegen.nanobind_compiled_sdfg.NanobindCompiledSDFG)

        n = 8
        floats = 2 * n  # a veclen-2 array of N vectors is 2*N base scalars
        pad = 8  # guard scalars on each side
        sentinel = np.float32(-999.0)

        x_buf = np.full(pad + floats + pad, sentinel, dtype=np.float32)
        x_buf[pad:pad + floats] = np.arange(floats, dtype=np.float32)
        y_buf = np.full(pad + floats + pad, sentinel, dtype=np.float32)

        x = x_buf[pad:pad + floats].reshape(n, 2)  # contiguous interior view
        y = y_buf[pad:pad + floats].reshape(n, 2)
        csdfg(x=x, y=y, N=np.int32(n))

        assert np.allclose(y, x)  # data copied
        assert np.all(y_buf[:pad] == sentinel)  # no under-write
        assert np.all(y_buf[pad + floats:] == sentinel)  # no over-write


def test_nanobind_interface_vector_uses_base_scalar():
    """A vector array's nb::ndarray uses the base scalar; the cast target stays dace::vec.

    Version-independent guard: nb::ndarray needs a real scalar type, not the
    dace::vec struct, but the pointer passed to the kernel must remain dace::vec*.
    """
    from dace.codegen.nanobind_bindings import generate_bindings_code

    sdfg = dace.SDFG('vec_scalar_probe')
    sdfg.add_array('v', [4], dace.vector(dace.float32, 2))
    code = generate_bindings_code(sdfg)
    assert 'nb::ndarray<float' in code  # base scalar in the ndarray type
    assert 'nb::ndarray<dace::vec' not in code  # never the struct
    assert 'reinterpret_cast<dace::vec<float, 2> *>' in code  # true pointer type kept


def test_nanobind_interface_float16_rejected():
    """float16 arrays are refused with a clear error (dace::half is not a valid nanobind dtype).

    Out of scope for now (absent from the CI failures); rejecting explicitly beats
    an opaque compile error. A future slice can map it.
    """
    import pytest
    from dace import dtypes
    from dace.codegen.nanobind_bindings import generate_bindings_code

    sdfg = dace.SDFG('float16_reject_probe')
    sdfg.add_array('h', [4], dtypes.float16)
    with pytest.raises(NotImplementedError, match='float16'):
        generate_bindings_code(sdfg)


def test_nanobind_interface_filename():
    """`filename` returns the resolved absolute path to the built .so (parity with CompiledSDFG)."""
    import pathlib

    with set_temporary('compiler', 'interface', value='nanobind'):
        N = dace.symbol('N')

        @dace.program
        def axpy_nanobind_filename(A: dace.float64[N], B: dace.float64[N], alpha: dace.float64):
            B[:] = alpha * A + B

        csdfg = axpy_nanobind_filename.to_sdfg().compile()
        assert isinstance(csdfg, dace.codegen.nanobind_compiled_sdfg.NanobindCompiledSDFG)

        expected = str(pathlib.Path(csdfg.module.__file__).resolve())
        assert csdfg.filename == expected
        p = pathlib.Path(csdfg.filename)
        assert p.is_absolute()
        assert p.exists()
        assert csdfg.filename.endswith('.so')


def test_nanobind_interface_struct_element_return():
    """A return array of a dace.struct (dtypes.struct element) round-trips (argmax-style)."""
    with set_temporary('compiler', 'interface', value='nanobind'):
        pair = dace.struct('pair', idx=dace.int32, val=dace.float64)

        @dace.program
        def argmax_nanobind(x: dace.float64[1024]):
            result = np.ndarray([1], dtype=pair)
            with dace.tasklet:
                init >> result[0]
                init.idx = -1
                init.val = -1e38

            for i in dace.map[0:1024]:
                with dace.tasklet:
                    inp << x[i]
                    out >> result(1, lambda x, y: pair(val=max(x.val, y.val), idx=(x.idx if x.val > y.val else y.idx)))
                    out = pair(idx=i, val=inp)

            return result

        csdfg = argmax_nanobind.to_sdfg().compile()
        assert isinstance(csdfg, dace.codegen.nanobind_compiled_sdfg.NanobindCompiledSDFG)

        A = np.random.rand(1024)
        result = csdfg(x=A)
        assert result[0][0] == np.argmax(A)


def test_nanobind_interface_struct_element_array_forward_declared():
    """A dtypes.struct-element array forward-declares the struct and binds an untyped ndarray."""
    from dace.codegen.nanobind_bindings import generate_bindings_code

    pair = dace.struct('pair', idx=dace.int32, val=dace.float64)
    sdfg = dace.SDFG('struct_elem_probe')
    sdfg.add_array('p', [4], pair)
    code = generate_bindings_code(sdfg)
    assert 'struct pair;' in code  # forward-declared
    assert 'reinterpret_cast<pair *>' in code  # cast to the struct pointer
    assert 'nb::ndarray<pair' not in code  # never the struct as ndarray scalar


def test_nanobind_interface_struct_element_input():
    """A dtypes.struct-element array passed as an input is byte-view marshalled and copies correctly."""
    with set_temporary('compiler', 'interface', value='nanobind'):
        pair = dace.struct('pair', idx=dace.int32, val=dace.float64)

        sdfg = dace.SDFG('copy_struct_input_nanobind')
        sdfg.add_array('A', [4], pair)  # input array of struct
        sdfg.add_array('B', [4], pair)  # output array of struct
        state = sdfg.add_state()
        state.add_edge(state.add_access('A'), None, state.add_access('B'), None, dace.Memlet('A[0:4]'))

        csdfg = sdfg.compile()
        assert isinstance(csdfg, dace.codegen.nanobind_compiled_sdfg.NanobindCompiledSDFG)

        A = np.zeros(4, dtype=pair.as_numpy_dtype())
        for i in range(4):
            A[i]['idx'] = i * 10
            A[i]['val'] = float(i)
        B = np.zeros(4, dtype=pair.as_numpy_dtype())
        csdfg(A=A, B=B)
        assert np.array_equal(B['idx'], A['idx'])
        assert np.array_equal(B['val'], A['val'])


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
    test_nanobind_interface_load_reuses_same_artifact()
    test_nanobind_interface_load_distinct_artifact_raises()
    test_nanobind_interface_safe_call()
    test_nanobind_interface_safe_call_kwargs()
    test_nanobind_interface_safe_call_return_rejected()
    test_nanobind_interface_structure_argument()
    test_nanobind_interface_nested_structure_argument()
    test_nanobind_interface_container_array_read()
    test_nanobind_interface_complex_array()
    test_nanobind_interface_includes_dace_type_headers()
    test_nanobind_interface_vector_array()
    test_nanobind_interface_vector_uses_base_scalar()
    test_nanobind_interface_float16_rejected()
    test_nanobind_interface_filename()
    test_nanobind_interface_struct_element_return()
    test_nanobind_interface_struct_element_array_forward_declared()
    test_nanobind_interface_struct_element_input()
