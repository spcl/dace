# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the nanobind-based `CompiledSDFG` interface (`compiler.interface=nanobind`)."""
import sys

import numpy as np
import pytest

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
        # The module is registered under dace.generated.<folder magic>.<name>.
        from dace.codegen.compiler import nanobind_qualified_module_name
        qualname = nanobind_qualified_module_name(csdfg.sdfg.build_folder, sdfg.name)
        assert qualname.startswith('dace.generated.')
        assert qualname.endswith(f'.{sdfg.name}')
        assert qualname in sys.modules
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
        assert csdfg2.sdfg.name == f'{base_name}_0'
        from dace.codegen.compiler import nanobind_qualified_module_name
        assert nanobind_qualified_module_name(csdfg2.sdfg.build_folder, f'{base_name}_0') in sys.modules


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


def test_nanobind_interface_return_override_forbidden_by_default():
    """By default the nanobind interface refuses a caller-provided __return buffer."""
    import pytest
    with set_temporary('compiler', 'interface', value='nanobind'):

        @dace.program
        def double_ret_default(A: dace.float64[20]):
            return A * 2

        csdfg = double_ret_default.to_sdfg().compile()
        a = np.random.rand(20)
        out = np.empty(20, dtype=np.float64)
        with pytest.raises(ValueError, match='nanobind_allow_return_override'):
            csdfg(A=a, __return=out)


def test_nanobind_interface_return_override_allowed():
    """With the option on, a caller-provided __return buffer is written in place and returned."""
    with set_temporary('compiler', 'interface', value='nanobind'):
        with set_temporary('compiler', 'nanobind_allow_return_override', value=True):

            @dace.program
            def double_ret_ovr(A: dace.float64[20]):
                return A * 2

            csdfg = double_ret_ovr.to_sdfg().compile()
            a = np.random.rand(20)
            out = np.zeros(20, dtype=np.float64)
            result = csdfg(A=a, __return=out)
            assert result is out  # the caller's buffer is returned
            assert np.allclose(out, a * 2)  # ...and written in place


def test_nanobind_interface_return_override_wrong_dtype_rejected_by_binding():
    """With the option on, no Python-side type check is imposed: a buffer the
    nanobind binding cannot accept (wrong dtype) is rejected by the binding."""
    import pytest
    with set_temporary('compiler', 'interface', value='nanobind'):
        with set_temporary('compiler', 'nanobind_allow_return_override', value=True):

            @dace.program
            def double_ret_dtype(A: dace.float64[20]):
                return A * 2

            csdfg = double_ret_dtype.to_sdfg().compile()
            a = np.random.rand(20)
            wrong = np.zeros(20, dtype=np.float32)  # binding expects float64
            with pytest.raises(Exception):
                csdfg(A=a, __return=wrong)


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


def test_nanobind_interface_get_state_struct_parity():
    """get_state_struct exposes the same leading state-struct pointer fields as
    the ctypes interface, as a live ctypes.Structure overlay of state memory."""
    import ctypes

    def build_and_fields(interface):
        with set_temporary('compiler', 'interface', value=interface):
            N = dace.symbol('N')
            sdfg = dace.SDFG(f'ststruct_parity_{interface}')
            sdfg.add_array('A', [N], dace.float64)
            # A persistent transient becomes a named pointer field in the state struct.
            sdfg.add_transient('buf', [N], dace.float64, lifetime=dace.AllocationLifetime.Persistent)
            st = sdfg.add_state()
            st.add_nedge(st.add_read('A'), st.add_write('buf'), dace.Memlet('A[0:N]'))
            st.add_nedge(st.add_read('buf'), st.add_write('A'), dace.Memlet('buf[0:N]'))
            csdfg = sdfg.compile()
            csdfg(A=np.ones(8), N=np.int32(8))  # initialize the state
            struct = csdfg.get_state_struct()
            assert isinstance(struct, ctypes.Structure)
            names = [name for name, _ in struct._fields_]
            csdfg.finalize()
            return names

    nb_fields = build_and_fields('nanobind')
    ct_fields = build_and_fields('ctypes')
    assert nb_fields == ct_fields
    assert any('buf' in f for f in nb_fields)  # the persistent transient's pointer field


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
        # program inside the original build folder. DaCe may append the SDFG
        # hash to the build-folder basename to disambiguate a pre-existing
        # cache directory (as happens on CI), so match the prefix, not equality.
        renamed_folder = csdfg2.sdfg.build_folder
        assert os.path.basename(renamed_folder).startswith(renamed)
        assert os.path.isfile(os.path.join(renamed_folder, 'INTERFACE'))
        assert not os.path.isfile(os.path.join(original_folder, 'build', f'lib{renamed}.so'))


def test_nanobind_interface_rename_explicit_folder_stays(tmp_path):
    """An explicitly-set build folder is the user's contract: a collision-renamed
    program builds in place inside it (the fixed-folder regime, same behaviour
    as cache mode 'single') instead of re-deriving its own folder."""
    import os

    with set_temporary('compiler', 'interface', value='nanobind'):
        N = dace.symbol('N')

        @dace.program
        def axpy_nanobind_explfolder(A: dace.float64[N], B: dace.float64[N], alpha: dace.float64):
            B[:] = alpha * A + B

        folder = str(tmp_path / 'pinned')

        sdfg1 = axpy_nanobind_explfolder.to_sdfg()
        base_name = sdfg1.name
        sdfg1.build_folder = folder
        csdfg1 = sdfg1.compile()

        sdfg2 = axpy_nanobind_explfolder.to_sdfg()
        sdfg2.build_folder = folder
        csdfg2 = sdfg2.compile()

        # Renamed, but still in the pinned folder - never relocated.
        assert csdfg2.sdfg.name == f'{base_name}_0'
        assert os.path.realpath(csdfg2.sdfg.build_folder) == os.path.realpath(folder)
        assert os.path.realpath(os.path.dirname(csdfg2.filename)).startswith(os.path.realpath(folder))

        n = 16
        a = np.random.rand(n)
        b = np.random.rand(n)
        expected = 3.0 * a + b
        csdfg2(A=a, B=b, alpha=np.float64(3.0), N=np.int32(n))
        assert np.allclose(b, expected)
        b2 = np.random.rand(n)
        expected2 = 2.0 * a + b2
        csdfg1(A=a, B=b2, alpha=np.float64(2.0), N=np.int32(n))
        assert np.allclose(b2, expected2)


def test_nanobind_interface_rename_third_compile_consistent():
    """Three same-named compiles yield base, _0, _1 - the collision probe must
    track the folder each candidate actually builds into, or the third compile
    would silently reuse the stale _0 module."""
    with set_temporary('compiler', 'interface', value='nanobind'):
        N = dace.symbol('N')

        @dace.program
        def axpy_nanobind_thrice(A: dace.float64[N], B: dace.float64[N], alpha: dace.float64):
            B[:] = alpha * A + B

        sdfg1 = axpy_nanobind_thrice.to_sdfg()
        base_name = sdfg1.name
        csdfg1 = sdfg1.compile()
        csdfg2 = axpy_nanobind_thrice.to_sdfg().compile()
        csdfg3 = axpy_nanobind_thrice.to_sdfg().compile()

        assert csdfg2.sdfg.name == f'{base_name}_0'
        assert csdfg3.sdfg.name == f'{base_name}_1'

        # All three dispatch into live, correct code.
        n = 16
        a = np.random.rand(n)
        for csdfg, alpha in ((csdfg1, 2.0), (csdfg2, 3.0), (csdfg3, 4.0)):
            b = np.random.rand(n)
            expected = alpha * a + b
            csdfg(A=a, B=b, alpha=np.float64(alpha), N=np.int32(n))
            assert np.allclose(b, expected)


def test_nanobind_interface_report_follows_rename():
    """Instrumentation reports of a collision-renamed program are found via the
    compiled handle's sdfg (the renamed compile copy, which knows its own
    folder). The ORIGINAL object keeps looking in its identity-derived folder
    and finds nothing - the accepted limitation behind refusing
    SDFG.safe_call() on nanobind."""
    with set_temporary('compiler', 'interface', value='nanobind'):
        N = dace.symbol('N')

        @dace.program
        def axpy_nanobind_report(A: dace.float64[N], B: dace.float64[N], alpha: dace.float64):
            B[:] = alpha * A + B

        # First compile occupies the (folder, name) identity.
        axpy_nanobind_report.to_sdfg().compile()

        # The instrumented recompile of the same name is renamed away.
        sdfg = axpy_nanobind_report.to_sdfg()
        sdfg.instrument = dace.InstrumentationType.Timer
        csdfg = sdfg.compile()
        assert csdfg.sdfg.name == f'{sdfg.name}_0'

        n = 8
        a = np.random.rand(n)
        b = np.random.rand(n)
        csdfg(A=a, B=b, alpha=np.float64(2.0), N=np.int32(n))
        csdfg.finalize()  # __dace_exit is what saves the report

        # Diagnostics for the CI-only failure of the first assert (passes
        # locally standalone, under the full CI env, and in full-file runs):
        # where did the report go?
        import os
        folder = str(csdfg.sdfg.build_folder)
        perf = os.path.join(folder, 'perf')
        diag = (f'cwd="{os.getcwd()}", build_folder="{folder}" '
                f'(exists={os.path.isdir(folder)}), perf exists={os.path.isdir(perf)}, '
                f'perf content={os.listdir(perf) if os.path.isdir(perf) else "n/a"}, '
                f'original folder="{sdfg.build_folder}" (exists={os.path.isdir(str(sdfg.build_folder))})')
        assert csdfg.sdfg.get_latest_report() is not None, diag
        assert sdfg.get_latest_report() is None, diag


def test_nanobind_interface_perf_folder_only_when_instrumented():
    """The perf/ report folder is created exactly when the SDFG is
    instrumented, in BOTH folder modes. Production mode used to skip it
    entirely, silently dropping every report (the runtime's report.save()
    neither creates directories nor reports a failed open); uninstrumented
    folders stay lean in both modes."""
    import os

    with set_temporary('compiler', 'interface', value='nanobind'):
        N = dace.symbol('N')

        @dace.program
        def axpy_nanobind_perfdir(A: dace.float64[N], B: dace.float64[N], alpha: dace.float64):
            B[:] = alpha * A + B

        # Production mode (env must yield to set_temporary, see gotcha):
        # no perf/ without instrumentation, perf/ with it.
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv('DACE_compiler_build_folder_mode', raising=False)
            with set_temporary('compiler', 'build_folder_mode', value='production'):
                csdfg = axpy_nanobind_perfdir.to_sdfg().compile()
                assert not os.path.isdir(os.path.join(csdfg.sdfg.build_folder, 'perf'))

                sdfg2 = axpy_nanobind_perfdir.to_sdfg()
                sdfg2.instrument = dace.InstrumentationType.Timer
                csdfg2 = sdfg2.compile()  # collision-renamed - irrelevant here
                assert os.path.isdir(os.path.join(csdfg2.sdfg.build_folder, 'perf'))

            # Development mode: uninstrumented folders are lean here too
            # (previously perf/ was created unconditionally).
            with set_temporary('compiler', 'build_folder_mode', value='development'):
                sdfg3 = axpy_nanobind_perfdir.to_sdfg()
                csdfg3 = sdfg3.compile()  # renamed again - fresh folder
                assert not os.path.isdir(os.path.join(csdfg3.sdfg.build_folder, 'perf'))


def test_nanobind_interface_sdfg_safe_call_refused():
    """SDFG.safe_call() is refused on the nanobind interface: it compiles
    internally and hides the compiled object, so after a collision rename any
    post-call query on the original SDFG (e.g. get_latest_report()) would
    silently look in the wrong folder. compile() + CompiledSDFG.safe_call()
    is the supported route."""
    with set_temporary('compiler', 'interface', value='nanobind'):
        sdfg = dace.SDFG('sdfg_safe_call_refuse_probe')
        sdfg.add_array('A', [4], dace.float64)
        with pytest.raises(NotImplementedError, match='safe_call'):
            sdfg.safe_call(A=np.zeros(4))


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
        # Symbol values are never stored on the handle; the workspace entry
        # points take them per call. Any subset of the __call__ arguments is
        # accepted - the binding picks the ones it needs.
        sizes = csdfg.get_workspace_sizes(N=np.int32(n))
        assert sizes == {dace.StorageType.CPU_Heap: n * 8}

        wsp = np.random.rand(n)
        # The full __call__-style argument set (including arrays) is accepted.
        csdfg.set_workspace(dace.StorageType.CPU_Heap, wsp, a=a, N=np.int32(n))

        ref = a + 1
        csdfg(a=a, N=np.int32(n))
        assert np.allclose(a, ref)
        assert np.allclose(wsp, ref)

        # The state-struct field names are baked in at codegen time; the
        # external workspace pointer is one of them.
        fields = csdfg.state_fields()
        assert isinstance(fields, list) and len(fields) > 0
        assert any('workspace' in f for f in fields)

        # get_state_struct returns a live, mutable ctypes.Structure overlay of
        # the state memory - parity with the ctypes interface.
        import ctypes
        struct = csdfg.get_state_struct()
        assert isinstance(struct, ctypes.Structure)
        assert [name for name, _ in struct._fields_] == fields
        # The structure aliases the actual state memory at state_pointer.
        assert ctypes.addressof(struct) == csdfg._handle.state_pointer
        # The workspace pointer set via set_workspace is readable through it.
        wsp_field = next(f for f in fields if 'workspace' in f)
        assert getattr(struct, wsp_field) is not None


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
    """pyobject returns are dropped (arrays only); pyobject ARRAY arguments are
    not supported (only scalars are).

    Both are rejected at codegen (the generator must refuse instead of emitting
    C++ that does not compile), with distinct messages.
    """
    import pytest
    from dace import dtypes
    from dace.codegen.nanobind_bindings import generate_bindings_code

    # pyobject return value: dropped, arrays-only message.
    ret_sdfg = dace.SDFG('pyobject_return_reject_probe')
    ret_sdfg.add_array('__return', [1], dtypes.pyobject())
    with pytest.raises(NotImplementedError, match='arrays only'):
        generate_bindings_code(ret_sdfg)

    # pyobject array argument: only scalars pass through.
    arg_sdfg = dace.SDFG('pyobject_arr_reject_probe')
    arg_sdfg.add_array('objs', [4], dtypes.pyobject())
    with pytest.raises(NotImplementedError, match='scalar'):
        generate_bindings_code(arg_sdfg)


def test_nanobind_interface_pyobject_scalar_binding():
    """A pyobject scalar argument binds as nb::object and forwards the raw
    PyObject* (reinterpret_cast to the opaque `pyobject` typedef)."""
    from dace import dtypes
    from dace.codegen.nanobind_bindings import generate_bindings_code

    sdfg = dace.SDFG('pyobject_scalar_bind_probe')
    sdfg.add_scalar('obj', dtypes.pyobject())
    sdfg.add_array('A', [4], dace.float64)

    code = generate_bindings_code(sdfg)
    assert 'nb::object obj' in code
    assert 'reinterpret_cast<pyobject>(obj.ptr())' in code
    assert 'nb::arg("obj")' in code


def test_nanobind_interface_pyobject_scalar_arg_e2e():
    """A pyobject scalar argument passes through as an opaque PyObject* and
    arrives at a callback as the very same object (identity preserved)."""
    from dace import dtypes

    with set_temporary('compiler', 'interface', value='nanobind'):
        sdfg = dace.SDFG('pyobject_passthrough')
        sdfg.add_scalar('obj', dtypes.pyobject())
        sdfg.add_array('A', [4], dace.float64)
        sdfg.add_symbol('consume', dace.callback(None, dtypes.pyobject()))
        state = sdfg.add_state()
        t = state.add_tasklet('t', {'o_in'}, {'a_out'}, 'consume(o_in)\na_out = 1.0')
        state.add_edge(state.add_read('obj'), None, t, 'o_in', dace.Memlet('obj[0]'))
        state.add_edge(t, 'a_out', state.add_write('A'), None, dace.Memlet('A[0]'))

        csdfg = sdfg.compile()

        class Payload:
            pass

        payload = Payload()
        received = []
        a = np.zeros(4)
        csdfg(obj=payload, A=a, consume=lambda x: received.append(x))
        assert a[0] == 1.0
        assert len(received) == 1
        assert received[0] is payload


def test_nanobind_interface_lowp_dtypes_rejected():
    """bfloat16/float8 SCALARS are refused at codegen (they would need value
    type-casters, like float16 scalars); arrays pass through the
    __array_interface__ pointer extraction instead (see the binding test)."""
    from dace.codegen.nanobind_bindings import generate_bindings_code

    sdfg = dace.SDFG('lowp_reject_scalar')
    sdfg.add_scalar('x', dace.bfloat16)
    with pytest.raises(NotImplementedError, match='ctypes'):
        generate_bindings_code(sdfg)


def test_nanobind_interface_lowp_array_binding():
    """A bfloat16/float8 array binds as nb::object: numpy cannot export
    ml_dtypes arrays via DLPack or the buffer protocol, but it does expose
    __array_interface__ - the raw pointer is extracted from there (the same
    protocol the ctypes marshaller uses), with a typestr itemsize check as
    the one sanity guard. GPU arrays read __cuda_array_interface__."""
    from dace.codegen.nanobind_bindings import generate_bindings_code

    sdfg = dace.SDFG('lowp_array_bind_probe')
    sdfg.add_array('A', [4], dace.bfloat16)
    sdfg.add_array('B', [4], dace.float8_e4m3fn)
    code = generate_bindings_code(sdfg)
    sig = code.split('void call(')[1].split(') {')[0]
    assert 'nb::object A' in sig
    assert '__array_interface__' in code
    # Itemsize-only guard: the typestr kind letter varies across ml_dtypes.
    assert 'expected a bfloat16 array (itemsize 2)' in code
    assert 'expected a float8_e4m3fn array (itemsize 1)' in code
    assert 'reinterpret_cast<dace::bfloat16 *>' in code
    assert 'reinterpret_cast<dace::float8_e4m3fn *>' in code

    # GPU storage reads the CUDA flavor of the protocol.
    sdfg = dace.SDFG('lowp_gpu_bind_probe')
    sdfg.add_array('A', [4], dace.bfloat16, storage=dace.StorageType.GPU_Global)
    code = generate_bindings_code(sdfg)
    assert '__cuda_array_interface__' in code

    # Not eligible for user_args (needs setup statements, outside the fast
    # path's initial scope).
    sdfg = dace.SDFG('lowp_uargs_probe')
    sdfg.add_array('A', [4], dace.bfloat16)
    sdfg.user_args = ['A']
    with pytest.raises(ValueError, match='not supported'):
        generate_bindings_code(sdfg)


def test_nanobind_interface_callback_binding():
    """A callback argument binds as a function-pointer address (std::uintptr_t + reinterpret_cast)."""
    from dace.codegen.nanobind_bindings import generate_bindings_code

    sdfg = dace.SDFG('callback_bind_probe')
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_symbol('cb', dace.callback(dace.float64, dace.float64))
    state = sdfg.add_state()
    t = state.add_tasklet('t', {}, {'o'}, 'o = cb(1.0)')
    state.add_edge(t, 'o', state.add_write('A'), None, dace.Memlet('A[0]'))

    code = generate_bindings_code(sdfg)
    assert 'std::uintptr_t cb__addr' in code  # shadow parameter
    assert 'reinterpret_cast<double (*)(double)>(cb__addr)' in code  # typed local via setup
    assert 'nb::arg("cb")' in code  # keyword stays the real name
    assert 'm_sym_' not in code  # symbol values are never stored on the handle


def test_nanobind_interface_handle_metadata_binding():
    """The handle exposes the codegen-time call metadata - return-array names,
    the single-value-return convention, and callback names - so the Python
    wrapper does not re-derive them from naming conventions."""
    from dace.codegen.nanobind_bindings import generate_bindings_code

    # Multi-value returns: names in order, not the single-value convention.
    sdfg = dace.SDFG('metadata_probe_rets')
    sdfg.add_array('__return_0', [4], dace.float64)
    sdfg.add_array('__return_1', [4], dace.float64)
    code = generate_bindings_code(sdfg)
    assert 'nb::make_tuple("__return_0", "__return_1")' in code
    assert '"is_single_value_ret", [](DaceHandle_metadata_probe_rets &) { return false; }' in code

    # Single-value return plus a callback.
    sdfg = dace.SDFG('metadata_probe_single')
    sdfg.add_array('__return', [4], dace.float64)
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_symbol('cb', dace.callback(dace.float64, dace.float64))
    state = sdfg.add_state()
    t = state.add_tasklet('t', {}, {'o'}, 'o = cb(1.0)')
    state.add_edge(t, 'o', state.add_write('A'), None, dace.Memlet('A[0]'))
    code = generate_bindings_code(sdfg)
    assert 'nb::make_tuple("__return")' in code
    assert '"is_single_value_ret", [](DaceHandle_metadata_probe_single &) { return true; }' in code
    assert '"callback_names", [](DaceHandle_metadata_probe_single &) { return nb::make_tuple("cb"); }' in code

    # No returns, no callbacks: empty tuples.
    sdfg = dace.SDFG('metadata_probe_empty')
    sdfg.add_array('A', [10], dace.float64)
    code = generate_bindings_code(sdfg)
    assert '"return_names", [](DaceHandle_metadata_probe_empty &) { return nb::make_tuple(); }' in code
    assert '"callback_names", [](DaceHandle_metadata_probe_empty &) { return nb::make_tuple(); }' in code


def test_nanobind_interface_symbol_inference_binding():
    """A size symbol not in arg_names binds as an optional parameter with a shape-derived fallback."""
    from dace.codegen.nanobind_bindings import generate_bindings_code

    N = dace.symbol('N')
    M = dace.symbol('M')

    @dace.program
    def sym_infer_probe(A: dace.float64[N + 1], B: dace.float64[10]):
        B[0] = A[0] * N + M  # M appears in no shape: not inferable

    # Explicit simplify=True: the simplify pipeline marks the arrays
    # non-nullable (optional=False), which this test's unguarded-fallback
    # asserts rely on; the unsimplified path has its own test below.
    code = generate_bindings_code(sym_infer_probe.to_sdfg(simplify=True))
    # N: optional, inferred from A's shape (inverted expression N = shape - 1).
    assert 'N__opt' in code
    assert 'nb::arg("N") = nb::none()' in code
    assert 'A.shape(0)' in code
    # M: optional too, but omitting it is a clear error naming the symbol.
    assert 'nb::arg("M") = nb::none()' in code
    assert "missing argument 'M'" in code


def test_nanobind_interface_symbol_inference():
    """Omitted size symbols are inferred from array shapes; explicit values still win."""
    with set_temporary('compiler', 'interface', value='nanobind'):
        N = dace.symbol('N')

        @dace.program
        def infer_e2e_nanobind(A: dace.float64[N], B: dace.float64[N]):
            B[:] = A + B

        csdfg = infer_e2e_nanobind.to_sdfg().compile()
        n = 12
        a = np.random.rand(n)
        b = np.random.rand(n)
        expected = a + b
        csdfg(A=a, B=b)  # no N: inferred from A.shape
        assert np.allclose(b, expected)

        # Explicit N still wins (and behaves as before).
        a2 = np.random.rand(n)
        b2 = np.random.rand(n)
        expected2 = a2 + b2
        csdfg(A=a2, B=b2, N=np.int32(n))
        assert np.allclose(b2, expected2)


def test_nanobind_interface_symbol_inference_unsimplified_binding():
    """Inference works on an unsimplified SDFG (``optional=None`` arrays):
    nullability is opt-in, so the arrays bind plain and the shape fallback
    needs no guard."""
    from dace.codegen.nanobind_bindings import generate_bindings_code

    N = dace.symbol('N')

    @dace.program
    def sym_infer_unsimplified(A: dace.float64[N + 1], B: dace.float64[10]):
        B[0] = A[0] * N

    code = generate_bindings_code(sym_infer_unsimplified.to_sdfg(simplify=False))
    assert 'N__opt' in code
    assert 'nb::arg("N") = nb::none()' in code
    assert 'A.shape(0)' in code


def test_nanobind_interface_symbol_inference_unsimplified():
    """E2E: omitting a size symbol works on an unsimplified SDFG too (the CI
    legs run with automatic simplification off)."""
    with set_temporary('compiler', 'interface', value='nanobind'):
        N = dace.symbol('N')

        @dace.program
        def infer_e2e_unsimplified(A: dace.float64[N], B: dace.float64[N]):
            B[:] = A + B

        csdfg = infer_e2e_unsimplified.to_sdfg(simplify=False).compile()
        n = 9
        a = np.random.rand(n)
        b = np.random.rand(n)
        expected = a + b
        csdfg(A=a, B=b)  # no N: inferred from A.shape through the guard
        assert np.allclose(b, expected)


def test_nanobind_interface_symbol_inference_stride_binding():
    """A symbol appearing only in an array's strides is inferable too - DaCe
    descriptor strides and DLPack (nb::ndarray::stride) both count elements."""
    from dace.codegen.nanobind_bindings import generate_bindings_code

    S = dace.symbol('S')
    sdfg = dace.SDFG('sym_infer_stride_probe')
    sdfg.add_array('A', [4], dace.float64, strides=[S], total_size=4 * S)
    sdfg.arg_names = ['A']

    code = generate_bindings_code(sdfg)
    assert 'S__opt' in code
    assert 'nb::arg("S") = nb::none()' in code
    assert 'A.stride(0)' in code


def test_nanobind_interface_symbol_inference_stride():
    """E2E: a stride symbol is inferred from the passed array's actual stride."""
    with set_temporary('compiler', 'interface', value='nanobind'):
        S = dace.symbol('S')
        sdfg = dace.SDFG('sym_infer_stride_e2e')
        sdfg.add_array('A', [4], dace.float64, strides=[S], total_size=4 * S)
        sdfg.add_array('B', [4], dace.float64)
        sdfg.arg_names = ['A', 'B']
        state = sdfg.add_state()
        state.add_mapped_tasklet('copy',
                                 dict(i='0:4'),
                                 dict(inp=dace.Memlet('A[i]')),
                                 'out = inp',
                                 dict(out=dace.Memlet('B[i]')),
                                 external_edges=True)

        csdfg = sdfg.compile()
        base = np.arange(12.0)
        a = base[::3]  # stride of 3 elements, shape (4,)
        b = np.zeros(4)
        csdfg(A=a, B=b)  # no S: inferred from A.stride(0)
        assert np.allclose(b, base[::3])


def test_nanobind_interface_symbol_inference_return_requires_symbol():
    """Symbolic-shaped returns require the symbol explicitly: their shapes are
    evaluated in Python, and inference happens only in compiled code (a
    per-call sympy inference in Python would be slow - by choice)."""
    import pytest
    with set_temporary('compiler', 'interface', value='nanobind'):
        N = dace.symbol('N')

        @dace.program
        def infer_ret_nanobind(A: dace.float64[N]):
            return A + 1.0

        csdfg = infer_ret_nanobind.to_sdfg().compile()
        a = np.random.rand(16)
        with pytest.raises(Exception, match='N'):
            csdfg(A=a)  # the return shape needs N in Python
        result = csdfg(A=a, N=np.int32(16))
        assert result.shape == (16, )
        assert np.allclose(result, a + 1.0)


def test_nanobind_interface_symbol_inference_missing():
    """An omitted symbol that cannot be inferred raises an error naming it."""
    import pytest
    with set_temporary('compiler', 'interface', value='nanobind'):
        M = dace.symbol('M')

        @dace.program
        def infer_missing_nanobind(A: dace.float64[10], B: dace.float64[10]):
            B[:] = A * M

        csdfg = infer_missing_nanobind.to_sdfg().compile()
        a = np.random.rand(10)
        b = np.zeros(10)
        with pytest.raises(Exception, match="missing argument 'M'.*not inferable"):
            csdfg(A=a, B=b)
        csdfg(A=a, B=b, M=np.int32(3))  # explicit still works
        assert np.allclose(b, a * 3)


def test_nanobind_interface_arg_names_symbol_not_inferred():
    """A symbol listed in ``SDFG.arg_names`` is an explicit parameter of the
    user-facing signature: it binds as a plain *required* scalar (no ``__opt``
    optional, no ``nb::none()`` default, no shape-derived fallback). Only its
    absence from ``arg_names`` makes a symbol an inference candidate -
    membership alone drives the split."""
    from dace.codegen.nanobind_bindings import generate_bindings_code

    N = dace.symbol('N')

    @dace.program
    def argname_symbol_probe(A: dace.float64[N]):
        A[:] = A + 1.0

    # The frontend's arg_names is ['A']: N is an artifact symbol,
    # optional + shape-inferred.
    artifact = argname_symbol_probe.to_sdfg(simplify=True)
    assert 'N' not in artifact.arg_names
    code = generate_bindings_code(artifact)
    assert 'N__opt' in code
    assert 'nb::arg("N") = nb::none()' in code
    assert 'A.shape(0)' in code

    # The same program with N promoted into arg_names: a plain required
    # scalar parameter, and no fallback is generated at all.
    explicit = argname_symbol_probe.to_sdfg(simplify=True)
    explicit.arg_names = explicit.arg_names + ['N']
    code = generate_bindings_code(explicit)
    assert 'N__opt' not in code
    assert 'nb::arg("N") = nb::none()' not in code
    assert 'nb::arg("N")' in code
    assert 'A.shape(0)' not in code


def test_nanobind_interface_symbol_inference_cross_symbol():
    """``A[a + b]`` with ``b`` promised as an explicit parameter: ``a`` is
    inferable as ``A.shape(0) - b``. A dim expression may reference further
    symbols besides the target, as long as each is itself listed in
    ``arg_names`` (a plain required parameter, in scope at the fallback).

    Hand-built via the SDFG API: the Python frontend rejects a parameter named
    like a symbol (``def testee(A: dace.float64[a + b], b: dace.int32)`` dies
    with ``FileExistsError: Cannot create symbol "b"``), so the explicit-``b``
    promise is expressed by listing it in ``arg_names``. The memlet must use
    the symbols - a shape-only symbol does not enter ``arglist()``."""
    from dace.codegen.nanobind_bindings import generate_bindings_code

    def build():
        a = dace.symbol('a')
        b = dace.symbol('b')
        sdfg = dace.SDFG('cross_sym_infer_probe')
        sdfg.add_array('A', [a + b], dace.float64)
        state = sdfg.add_state()
        tasklet = state.add_tasklet('set_last', {}, {'o'}, 'o = 1.0')
        state.add_edge(tasklet, 'o', state.add_write('A'), None, dace.Memlet('A[a + b - 1]'))
        sdfg.arg_names = ['A', 'b']
        return sdfg

    # Binding: 'a' is optional with the cross-symbol fallback; 'b' is a plain
    # required parameter the fallback references by name.
    code = generate_bindings_code(build())
    assert 'a__opt' in code
    assert 'A.shape(0) - b' in code
    assert "missing argument 'a'" not in code

    with set_temporary('compiler', 'interface', value='nanobind'):
        csdfg = build().compile()
        A = np.zeros(10)
        csdfg(A=A, b=np.int32(4))  # no 'a': inferred as A.shape(0) - b = 6
        assert A[9] == 1.0

        # An explicit 'a' still wins over the inference.
        A2 = np.zeros(10)
        csdfg(A=A2, b=np.int32(4), a=np.int32(6))
        assert A2[9] == 1.0


def test_nanobind_interface_scalar_callback():
    """A scalar callback is invoked from the GIL-released kernel and its result lands in the output."""
    with set_temporary('compiler', 'interface', value='nanobind'):
        cscale = dace.symbol('cscale', dace.callback(dace.float64, dace.float64))

        @dace.program
        def cb_prog_nanobind(A: dace.float64[10], B: dace.float64[10]):

            @dace.map(_[0:10])
            def index(i):
                a << A[i]
                b >> B[i]
                b = cscale(a)

        csdfg = cb_prog_nanobind.to_sdfg().compile()
        A = np.random.rand(10)
        B = np.zeros(10)
        csdfg(A=A, B=B, cscale=lambda x: x * 3.0)
        assert np.allclose(B, A * 3.0)


def test_nanobind_interface_bool_scalar_binds_as_uint8():
    """A bool scalar arg binds as uint8_t (nanobind's bool caster rejects numpy.bool_)."""
    from dace.codegen.nanobind_bindings import generate_bindings_code

    sdfg = dace.SDFG('bool_scalar_bind_probe')
    sdfg.add_scalar('flag', dace.bool)
    sdfg.add_array('__return', [1], dace.bool)
    state = sdfg.add_state()
    r = state.add_read('flag')
    w = state.add_write('__return')
    state.add_edge(r, None, w, None, sdfg.make_array_memlet('flag'))

    code = generate_bindings_code(sdfg)
    # The nanobind binding param is uint8_t (nanobind's `bool` caster rejects
    # numpy.bool_), cast back to bool for the kernel call. The kernel's own
    # extern-C declaration still takes `bool flag` - that is correct - so this
    # asserts the binding param specifically, not the whole TU.
    assert 'uint8_t flag' in code
    assert 'static_cast<bool>(flag)' in code
    assert 'bool flag' not in code.split('void call(')[1].split(') {')[0]


def test_nanobind_interface_bool_scalar_numpy_input():
    """A numpy.bool_ scalar argument is accepted end-to-end on the nanobind interface."""
    import numpy as np
    with set_temporary('compiler', 'interface', value='nanobind'):
        sdfg = dace.SDFG('bool_scalar_numpy_e2e')
        sdfg.add_scalar('flag', dace.bool)
        sdfg.add_array('__return', [1], dace.bool)
        state = sdfg.add_state()
        r = state.add_read('flag')
        w = state.add_write('__return')
        state.add_edge(r, None, w, None, sdfg.make_array_memlet('flag'))

        csdfg = sdfg.compile()
        assert csdfg(flag=np.True_)[0]  # numpy.bool_ (the previously-rejected case)
        assert not csdfg(flag=np.False_)[0]
        assert csdfg(flag=True)[0]  # python bool still works


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


def test_nanobind_interface_must_pass_symbols_extracted_first():
    """Symbols that must be passed explicitly (no inference source, so they are
    never legitimately None) are extracted at the top of ``call()``, before the
    inferable-symbol deductions. This fixes the evaluation order so a deduction
    may later reference an explicitly-passed symbol (e.g. shape ``(a + b,)``
    with ``a`` promised by the caller and ``b`` deduced from the shape)."""
    from dace.codegen.nanobind_bindings import generate_bindings_code

    a = dace.symbol('a')  # inferable from A's shape
    z = dace.symbol('z')  # appears in no shape/stride: must be passed

    @dace.program
    def must_pass_first_probe(A: dace.float64[a], B: dace.float64[10]):
        B[0] = A[0] * a + z

    code = generate_bindings_code(must_pass_first_probe.to_sdfg(simplify=True))
    call_body = code.split('void call(')[1]
    # 'z' (must-pass) is unwrapped before 'a' (deduced) - plain arglist order
    # would put 'a' first.
    assert "missing argument 'z'" in call_body
    z_extract = call_body.index('const int z = *z__opt;')
    a_deduce = call_body.index('const int a = a__opt.has_value()')
    assert z_extract < a_deduce


def test_nanobind_interface_load_reuses_same_artifact():
    """Loading the same artifact path again reuses the module (one module, many handles)."""
    from dace.codegen.compiler import load_nanobind_module

    with set_temporary('compiler', 'interface', value='nanobind'):
        N = dace.symbol('N')

        @dace.program
        def load_reuse_nanobind(A: dace.float64[N], B: dace.float64[N], alpha: dace.float64):
            B[:] = alpha * A + B

        csdfg = load_reuse_nanobind.to_sdfg().compile()
        module = load_nanobind_module(csdfg.module.__file__, csdfg.sdfg.name, csdfg.sdfg.build_folder)
        assert module is csdfg.module


def test_nanobind_interface_load_distinct_artifact_coexists():
    """A distinct artifact under an already-loaded generated name loads as its
    own module: registration is ``dace.generated.<magic>.<name>`` with
    ``<magic>`` derived from the SDFG's resolved build folder, so only the
    same (folder, name) pair reuses a loaded module - artifacts in different
    folders coexist."""
    import os
    import shutil
    import tempfile

    from dace.codegen.compiler import nanobind_qualified_module_name, load_nanobind_module
    from dace.codegen.nanobind_compiled_sdfg import NanobindCompiledSDFG

    with set_temporary('compiler', 'interface', value='nanobind'):
        N = dace.symbol('N')

        @dace.program
        def load_distinct_nanobind(A: dace.float64[N], B: dace.float64[N], alpha: dace.float64):
            B[:] = alpha * A + B

        csdfg = load_distinct_nanobind.to_sdfg().compile()
        # A copy of the .so in a different folder is a distinct artifact.
        copied_folder = tempfile.mkdtemp()
        copied = os.path.join(copied_folder, os.path.basename(csdfg.module.__file__))
        shutil.copy(csdfg.module.__file__, copied)
        # Identical content means identical C++ type identity (same content
        # hash in the type namespace), so nanobind warns about the duplicate
        # registration - expected and harmless here, the code is the same.
        import warnings as warnings_mod
        with warnings_mod.catch_warnings():
            warnings_mod.simplefilter('ignore', RuntimeWarning)
            module = load_nanobind_module(copied, csdfg.sdfg.name, copied_folder)
        assert module is not csdfg.module
        assert nanobind_qualified_module_name(copied_folder, csdfg.sdfg.name) in sys.modules
        assert (nanobind_qualified_module_name(copied_folder, csdfg.sdfg.name)
                != nanobind_qualified_module_name(csdfg.sdfg.build_folder, csdfg.sdfg.name))

        # A handle minted from the copy works (and the original still does).
        n = 8
        a = np.random.rand(n)
        b = np.random.rand(n)
        expected = 2.0 * a + b
        csdfg2 = NanobindCompiledSDFG(csdfg.sdfg, module, csdfg.sdfg.arg_names)
        csdfg2(A=a, B=b, alpha=np.float64(2.0), N=np.int32(n))
        assert np.allclose(b, expected)


def test_nanobind_interface_load_magic_collision_detected():
    """A folder-magic hash collision (same registry key, different file) is a
    hard error instead of silently aliasing two artifacts."""
    import types

    from dace.codegen.compiler import nanobind_qualified_module_name, load_nanobind_module

    bogus_folder = '/nonexistent/collision_probe'
    bogus = f'{bogus_folder}/libcollision_probe.so'
    key = nanobind_qualified_module_name(bogus_folder, 'collision_probe')
    # Plant a module under the key that claims to come from a different file,
    # as a hash collision between two folders would produce.
    sys.modules[key] = types.SimpleNamespace(__file__='/some/other/artifact.so')
    try:
        with pytest.raises(ValueError, match='collision'):
            load_nanobind_module(bogus, 'collision_probe', bogus_folder)
    finally:
        del sys.modules[key]


def test_nanobind_interface_load_unverifiable_module_rejected():
    """A registry hit whose module has no ``__file__`` cannot be verified as
    ours, so it must not be reused silently - it is rejected like a mismatch."""
    import types

    from dace.codegen.compiler import nanobind_qualified_module_name, load_nanobind_module

    folder = '/nonexistent/unverifiable_probe'
    library = f'{folder}/libunverifiable_probe.so'
    key = nanobind_qualified_module_name(folder, 'unverifiable_probe')
    sys.modules[key] = types.SimpleNamespace()
    try:
        with pytest.raises(ValueError, match='collision'):
            load_nanobind_module(library, 'unverifiable_probe', folder)
    finally:
        del sys.modules[key]


def test_nanobind_interface_same_name_different_programs_coexist():
    """Two different programs sharing an SDFG name load side by side and each
    executes its own code: the sys.modules key carries the build-folder magic, and the
    generated C++ namespace carries a content hash, so nanobind's process-wide
    type registry (keyed by type name) cannot conflate their handle types and
    silently dispatch one program's handle into the other's methods."""

    def make(addend):
        sdfg = dace.SDFG('coexist_tester')
        sdfg.add_array('A', [4], dace.float64)
        state = sdfg.add_state()
        tasklet = state.add_tasklet('t', {'i'}, {'o'}, f'o = i + {addend}')
        state.add_edge(state.add_read('A'), None, tasklet, 'i', dace.Memlet('A[0]'))
        state.add_edge(tasklet, 'o', state.add_write('A'), None, dace.Memlet('A[0]'))
        return sdfg

    with set_temporary('compiler', 'interface', value='nanobind'):
        # Distinct build folders per content: the default 'name' cache would
        # collide the folders and trigger the same-path rename instead. The
        # env var must be dropped first - a DACE_cache export (the nanobind CI
        # sets 'unique', under which same-named programs in one process share
        # a folder and rename) takes precedence over set_temporary.
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv('DACE_cache', raising=False)
            with set_temporary('cache', value='hash'):
                csdfg1 = make(1.0).compile()
                csdfg2 = make(2.0).compile()

        # Neither was renamed: same-name coexistence, not the rename loop.
        assert csdfg1.sdfg.name == 'coexist_tester', f'Build folders `csdfg1({csdfg1.sdfg.name}) = "{csdfg1.filename}"`, `csdfg2({csdfg2.sdfg.name}) = "{csdfg2.filename}"`'
        assert csdfg2.sdfg.name == 'coexist_tester', f'Build folders `csdfg1({csdfg1.sdfg.name}) = "{csdfg1.filename}"`, `csdfg2({csdfg2.sdfg.name}) = "{csdfg2.filename}"`'

        a = np.zeros(4)
        csdfg1(A=a)
        assert a[0] == 1.0
        b = np.zeros(4)
        csdfg2(A=b)
        assert b[0] == 2.0
        # The first handle still dispatches into its own program.
        c = np.zeros(4)
        csdfg1(A=c)
        assert c[0] == 1.0


def test_nanobind_interface_type_namespace_carries_content_hash():
    """The generated C++ namespace is <name>_<content hash>: same-named but
    different SDFGs get distinct type identities (nanobind conflates
    identically-named types across modules). Only disambiguation is required
    of the hash - hash_sdfg() carries no cross-version stability promise."""
    from dace.codegen.nanobind_bindings import generate_bindings_code

    def make(addend):
        sdfg = dace.SDFG('nshash_tester')
        sdfg.add_array('A', [4], dace.float64)
        state = sdfg.add_state()
        tasklet = state.add_tasklet('t', {'i'}, {'o'}, f'o = i + {addend}')
        state.add_edge(state.add_read('A'), None, tasklet, 'i', dace.Memlet('A[0]'))
        state.add_edge(tasklet, 'o', state.add_write('A'), None, dace.Memlet('A[0]'))
        return sdfg

    def type_namespace(code):
        marker = 'namespace dace { namespace generated { namespace '
        return code.split(marker)[1].split(' ')[0]

    ns_a = type_namespace(generate_bindings_code(make(1.0)))
    ns_b = type_namespace(generate_bindings_code(make(2.0)))
    assert ns_a.startswith('nshash_tester_')
    assert ns_a != ns_b  # distinct per content - the property that matters


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


def test_nanobind_interface_single_element_tuple_return():
    """A single-element tuple return comes back as a 1-tuple, not a bare array.

    DaCe names a single value ``__return`` but a one-element tuple ``__return_0``,
    so the wrapper must distinguish them (a bare ``len == 1`` check would collapse
    the 1-tuple to the array).
    """
    with set_temporary('compiler', 'interface', value='nanobind'):
        N = dace.symbol('N')

        @dace.program
        def one_tuple_nanobind(A: dace.float64[N]):
            return (A + 1.0, )

        csdfg = one_tuple_nanobind.to_sdfg().compile()
        n = 8
        a = np.random.rand(n)
        result = csdfg(A=a, N=np.int32(n))
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert np.allclose(result[0], a + 1.0)


def test_nanobind_interface_non_array_return_rejected():
    """A non-array return value is refused at codegen (returns are arrays only)."""
    import pytest
    from dace import dtypes
    from dace.codegen.nanobind_bindings import generate_bindings_code

    sdfg = dace.SDFG('non_array_return_probe')
    sdfg.add_scalar('__return', dtypes.float64)
    with pytest.raises(NotImplementedError, match='arrays only'):
        generate_bindings_code(sdfg)


def test_nanobind_interface_many_return_values():
    """More than ten return values keep their numeric order (not lexicographic `sorted`).

    With `sorted`, `__return_10` would precede `__return_2`, permuting the tuple.
    """
    with set_temporary('compiler', 'interface', value='nanobind'):

        @dace.program
        def many_returns_nanobind():
            return 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12

        csdfg = many_returns_nanobind.to_sdfg().compile()
        result = csdfg()
        assert isinstance(result, tuple)
        assert len(result) == 12
        assert tuple(int(r[0]) for r in result) == tuple(range(1, 13))


def test_nanobind_interface_struct_input_binds_as_object():
    """A struct-element array input binds as nb::object; the pointer is extracted in C++."""
    from dace.codegen.nanobind_bindings import generate_bindings_code

    pair = dace.struct('pair', idx=dace.int32, val=dace.float64)
    sdfg = dace.SDFG('struct_input_object_probe')
    sdfg.add_array('p', [4], pair)  # struct-element array input
    sdfg.arrays['p'].optional = False  # non-nullable, for the deterministic form
    code = generate_bindings_code(sdfg)
    assert 'nb::object p' in code  # bound as a generic Python object
    assert 'PyObject_GetBuffer' in code  # pointer pulled via the buffer protocol in C++
    assert '_DacePyBuffer p_buf(p.ptr())' in code  # via the RAII helper
    assert 'reinterpret_cast<pair *>(p_buf.data())' in code  # cast to the struct pointer
    assert 'nb::ndarray<uint8_t, nb::device::cpu> p' not in code  # no Python-side byte-view form


def test_nanobind_interface_optional_struct_array_binding():
    """A struct-element array binds as nb::object; nullable -> guarded optional + None->null, non-nullable -> unconditional."""
    from dace.codegen.nanobind_bindings import generate_bindings_code

    pair = dace.struct('pair', idx=dace.int32, val=dace.float64)
    sdfg = dace.SDFG('opt_struct_probe')
    sdfg.add_array('p', [4], pair)
    sdfg.arrays['p'].optional = True
    sdfg.add_array('q', [4], pair)
    sdfg.arrays['q'].optional = False
    code = generate_bindings_code(sdfg)

    # both bind as nb::object
    assert 'nb::object p' in code
    assert 'nb::object q' in code
    # nullable p: guarded optional buffer + None -> null pointer
    assert 'std::optional<_DacePyBuffer> p_buf' in code
    assert 'p.is_none()' in code
    # non-nullable q: unconditional buffer extraction
    assert '_DacePyBuffer q_buf(q.ptr())' in code
    # no old byte-view ndarray form remains
    assert 'nb::ndarray<uint8_t' not in code


def test_nanobind_interface_optional_struct_array_input():
    """An optional struct-element array accepts a record array (read by reference) and None (null pointer)."""
    from typing import Optional

    with set_temporary('compiler', 'interface', value='nanobind'):
        pair = dace.struct('pair', idx=dace.int32, val=dace.float64)

        @dace.program
        def optional_struct_arg(a: Optional[pair[1]], out: dace.int32[1]):
            if a is None:
                out[0] = -1
            else:
                out[0] = 1

        sdfg = optional_struct_arg.to_sdfg()
        assert sdfg.arrays['a'].optional is True
        csdfg = sdfg.compile()

        A = np.zeros(1, dtype=pair.as_numpy_dtype())
        out = np.zeros(1, dtype=np.int32)
        csdfg(a=A, out=out)
        assert out[0] == 1  # a real (non-null) array is passed by reference

        out = np.zeros(1, dtype=np.int32)
        csdfg(a=None, out=out)
        assert out[0] == -1  # None arrived as a null pointer


def test_nanobind_interface_gpu_array_binding():
    """A GPU_Global array binds as a device ndarray (cuda/rocm per the
    configured backend); CPU arrays, scalars and container arrays stay on
    nb::device::cpu."""
    from dace.codegen.nanobind_bindings import generate_bindings_code

    def build():
        sdfg = dace.SDFG('gpu_array_probe')
        sdfg.add_array('A', [10], dace.float64, storage=dace.StorageType.GPU_Global)
        sdfg.arrays['A'].optional = False
        sdfg.add_array('B', [10], dace.float64)
        sdfg.arrays['B'].optional = False
        # Nullable GPU array: the device annotation applies inside the optional.
        sdfg.add_array('C', [10], dace.float64, storage=dace.StorageType.GPU_Global)
        sdfg.arrays['C'].optional = True
        return sdfg

    with set_temporary('compiler', 'cuda', 'backend', value='cuda'):
        code = generate_bindings_code(build())
    assert 'nb::ndarray<double, nb::device::cuda> A' in code
    assert 'nb::ndarray<double, nb::device::cpu> B' in code
    assert 'std::optional<nb::ndarray<double, nb::device::cuda>> C' in code

    with set_temporary('compiler', 'cuda', 'backend', value='hip'):
        code = generate_bindings_code(build())
    assert 'nb::ndarray<double, nb::device::rocm> A' in code
    assert 'nb::ndarray<double, nb::device::cpu> B' in code


@pytest.mark.gpu
def test_nanobind_interface_gpu_arrays():
    """E2E: CuPy arrays pass directly to GPU-storage parameters; a host numpy
    array for a GPU parameter is rejected at dispatch."""
    cp = pytest.importorskip('cupy')

    with set_temporary('compiler', 'interface', value='nanobind'):
        N = dace.symbol('N')

        # The explicit GPU_Global annotations keep the *arguments* on the
        # device - apply_gpu_transformations alone would leave them on the
        # host and insert copies, binding CPU parameters.
        @dace.program
        def gpu_axpy_nanobind(A: dace.float64[N] @ dace.StorageType.GPU_Global,
                              B: dace.float64[N] @ dace.StorageType.GPU_Global):
            B[:] = A + B

        sdfg = gpu_axpy_nanobind.to_sdfg()
        sdfg.apply_gpu_transformations()
        csdfg = sdfg.compile()

        n = 32
        a = cp.random.rand(n)
        b = cp.random.rand(n)
        expected = cp.asnumpy(a + b)
        csdfg(A=a, B=b, N=np.int32(n))
        assert np.allclose(cp.asnumpy(b), expected)

        with pytest.raises(TypeError):
            csdfg(A=np.random.rand(n), B=b, N=np.int32(n))


def test_nanobind_interface_gpu_return_binding():
    """A GPU_Global __return binds as a device ndarray (pin: the GPU-argument
    device selection must keep covering return arrays)."""
    from dace.codegen.nanobind_bindings import generate_bindings_code

    sdfg = dace.SDFG('gpu_return_probe')
    sdfg.add_array('A', [10], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.arrays['A'].optional = False
    sdfg.add_array('__return', [10], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.arrays['__return'].optional = False

    with set_temporary('compiler', 'cuda', 'backend', value='cuda'):
        code = generate_bindings_code(sdfg)
    assert 'nb::ndarray<double, nb::device::cuda> __return' in code


def test_nanobind_interface_gpu_return_allocation_requires_cupy():
    """Allocating a GPU_Global return without CuPy raises NotImplementedError
    instead of silently handing the device binding a host array."""
    import importlib.util
    import types
    from dace.codegen.nanobind_compiled_sdfg import NanobindCompiledSDFG

    if importlib.util.find_spec('cupy') is not None:
        pytest.skip('cupy is installed; the missing-cupy error path cannot trigger')

    sdfg = dace.SDFG('gpu_return_alloc_probe')
    sdfg.add_array('__return', [10], dace.float64, storage=dace.StorageType.GPU_Global)
    stub_handle = types.SimpleNamespace(has_gpu_code=True,
                                        return_names=('__return', ),
                                        is_single_value_ret=True,
                                        callback_names=())
    stub_module = types.SimpleNamespace(make_compiled_sdfg=lambda: stub_handle, __file__='<stub>')
    wrapper = NanobindCompiledSDFG(sdfg, stub_module, [])
    with pytest.raises(NotImplementedError, match='cupy'):
        wrapper._allocate_return_arrays({})


@pytest.mark.gpu
def test_nanobind_interface_gpu_return_values():
    """E2E: a GPU program's return value comes back as a CuPy array."""
    cp = pytest.importorskip('cupy')

    with set_temporary('compiler', 'interface', value='nanobind'):

        @dace.program
        def gpu_ret_nanobind(A: dace.float64[10] @ dace.StorageType.GPU_Global):
            return A + 1.0

        sdfg = gpu_ret_nanobind.to_sdfg()
        sdfg.apply_gpu_transformations()
        csdfg = sdfg.compile()

        a = cp.random.rand(10)
        result = csdfg(A=a)
        assert isinstance(result, cp.ndarray)
        assert np.allclose(cp.asnumpy(result), cp.asnumpy(a) + 1.0)


def test_nanobind_interface_gpu_error_check(monkeypatch):
    """After a call on a GPU SDFG the GPU runtime's last error is checked
    (ctypes parity: fast_call's do_gpu_check): an error raises, a
    runtime-lookup failure only warns."""
    import types
    import warnings as warnings_mod
    from dace.codegen import common
    from dace.codegen.nanobind_compiled_sdfg import NanobindCompiledSDFG

    sdfg = dace.SDFG('gpu_error_probe')
    sdfg.add_array('A', [10], dace.float64, storage=dace.StorageType.GPU_Global)

    calls = []

    class FakeHandle:
        has_gpu_code = True
        return_names = ()
        is_single_value_ret = False
        callback_names = ()

        def __call__(self, *args, **kwargs):
            calls.append(kwargs)

    handle = FakeHandle()
    stub_module = types.SimpleNamespace(make_compiled_sdfg=lambda: handle, __file__='<stub>')
    csdfg = NanobindCompiledSDFG(sdfg, stub_module, ['A'])

    class FakeRuntime:

        def __init__(self, err):
            self._err = err

        def get_last_error_string(self):
            return self._err

    # No pending error: the call goes through.
    monkeypatch.setattr(common, 'get_gpu_runtime', lambda: FakeRuntime(None))
    csdfg(A=object())
    assert len(calls) == 1

    # A pending error raises, naming the error and the syncdebug hint.
    monkeypatch.setattr(common, 'get_gpu_runtime', lambda: FakeRuntime('illegal memory access'))
    with pytest.raises(RuntimeError, match='illegal memory access'):
        csdfg(A=object())

    # Failure to obtain the runtime degrades to a warning, not an error.
    def broken_runtime():
        raise RuntimeError('no runtime available')

    monkeypatch.setattr(common, 'get_gpu_runtime', broken_runtime)
    with warnings_mod.catch_warnings(record=True) as caught:
        warnings_mod.simplefilter('always')
        csdfg(A=object())
    assert any('Could not get last error' in str(w.message) for w in caught)


def test_nanobind_interface_gpu_error_check_disabled(monkeypatch):
    """gpu_error_check disables the post-call GPU error check (replacing ctypes'
    fast_call(do_gpu_check=False)): it defaults to the constructor argument and
    is settable through the property."""
    import types
    from dace.codegen import common
    from dace.codegen.nanobind_compiled_sdfg import NanobindCompiledSDFG

    sdfg = dace.SDFG('gpu_error_disable_probe')
    sdfg.add_array('A', [10], dace.float64, storage=dace.StorageType.GPU_Global)

    class FakeHandle:
        has_gpu_code = True
        return_names = ()
        is_single_value_ret = False
        callback_names = ()

        def __call__(self, *args, **kwargs):
            pass

    stub_module = types.SimpleNamespace(make_compiled_sdfg=lambda: FakeHandle(), __file__='<stub>')

    class FakeRuntime:

        def get_last_error_string(self):
            return 'illegal memory access'

    monkeypatch.setattr(common, 'get_gpu_runtime', lambda: FakeRuntime())

    # Default (constructor arg True): a pending error raises.
    csdfg = NanobindCompiledSDFG(sdfg, stub_module, ['A'])
    assert csdfg.gpu_error_check is True
    with pytest.raises(RuntimeError, match='illegal memory access'):
        csdfg(A=object())

    # Disabled through the property: the same pending error is ignored.
    csdfg.gpu_error_check = False
    assert csdfg.gpu_error_check is False
    csdfg(A=object())  # does not raise

    # Disabled through the constructor argument.
    csdfg2 = NanobindCompiledSDFG(sdfg, stub_module, ['A'], gpu_error_check=False)
    assert csdfg2.gpu_error_check is False
    csdfg2(A=object())  # does not raise


def test_nanobind_interface_finalize_exit_code_binding():
    """The generated finalize() hands the __dace_exit code back to Python
    instead of throwing - only the wrapper can translate GPU error codes
    (it needs the Python-side GPU runtime)."""
    from dace.codegen.nanobind_bindings import generate_bindings_code

    sdfg = dace.SDFG('finalize_code_probe')
    sdfg.add_array('A', [10], dace.float64)

    code = generate_bindings_code(sdfg)
    assert 'int finalize()' in code
    assert 'An error was detected after running' not in code


def test_nanobind_interface_finalize_error_translation(monkeypatch):
    """A nonzero __dace_exit code raises from the wrapper's finalize(); with
    GPU code the numeric code is translated through the GPU runtime and the
    syncdebug hint is appended (ctypes parity: CompiledSDFG.finalize /
    _get_error_text)."""
    import types
    from dace.codegen import common
    from dace.codegen.nanobind_compiled_sdfg import NanobindCompiledSDFG

    class FakeHandle:
        has_gpu_code = False
        return_names = ()
        is_single_value_ret = False
        callback_names = ()
        exit_code = 0

        def finalize(self):
            return self.exit_code

    def make_wrapper(handle):
        stub_module = types.SimpleNamespace(make_compiled_sdfg=lambda: handle, __file__='<stub>')
        return NanobindCompiledSDFG(dace.SDFG('finalize_error_probe'), stub_module, [])

    # Exit code 0: nothing raises.
    make_wrapper(FakeHandle()).finalize()

    # CPU-only: the raw code is reported as-is (no GPU runtime involved).
    handle = FakeHandle()
    handle.exit_code = 1
    with pytest.raises(RuntimeError, match='An error was detected after running "finalize_error_probe": 1'):
        make_wrapper(handle).finalize()

    # GPU code: the code goes through the runtime's get_error_string.
    class GpuHandle(FakeHandle):
        has_gpu_code = True

    class FakeRuntime:

        def get_error_string(self, code):
            assert code == 700
            return 'an illegal memory access was encountered'

    monkeypatch.setattr(common, 'get_gpu_runtime', lambda: FakeRuntime())
    handle = GpuHandle()
    handle.exit_code = 700
    with pytest.raises(RuntimeError, match='an illegal memory access was encountered. Consider'):
        make_wrapper(handle).finalize()


def test_nanobind_interface_float16_array_binding():
    """A float16 array binds as nb::ndarray<dace::float16> and the TU carries a
    dtype_traits specialization teaching nanobind that dace::float16 is a
    16-bit float (its own dtype detection uses std::is_floating_point, false
    for the dace::half struct)."""
    from dace.codegen.nanobind_bindings import generate_bindings_code

    sdfg = dace.SDFG('float16_array_probe')
    sdfg.add_array('A', [10], dace.float16)

    code = generate_bindings_code(sdfg)
    assert 'nb::ndarray<dace::float16' in code
    assert 'struct dtype_traits<dace::float16>' in code
    assert 'dlpack::dtype_code::Float' in code


def test_nanobind_interface_float16_trait_only_when_needed():
    """A float16-free module does not emit the dtype_traits specialization."""
    from dace.codegen.nanobind_bindings import generate_bindings_code

    sdfg = dace.SDFG('float16_absent_probe')
    sdfg.add_array('A', [10], dace.float64)

    code = generate_bindings_code(sdfg)
    assert 'dtype_traits<dace::float16>' not in code


def test_nanobind_interface_float16_scalar_still_rejected():
    """A float16 *scalar* argument stays unsupported (it would need a value
    type-caster, not an ndarray dtype trait) and points at the ctypes interface."""
    from dace.codegen.nanobind_bindings import generate_bindings_code

    sdfg = dace.SDFG('float16_scalar_probe')
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_scalar('s', dace.float16)

    with pytest.raises(NotImplementedError, match='float16'):
        generate_bindings_code(sdfg)


@pytest.mark.skipif(sys.platform == 'win32', reason='half compile probe uses g++')
def test_nanobind_interface_float16_end_to_end():
    """E2E: a float16[N] in / float16[N] out program round-trips through the
    real compiled nanobind module, and a passed float16 array is by-reference."""
    with set_temporary('compiler', 'interface', value='nanobind'):
        N = dace.symbol('N')

        @dace.program
        def add_half_nanobind(A: dace.float16[N], B: dace.float16[N]):
            B[:] = A + dace.float16(1.0)

        csdfg = add_half_nanobind.to_sdfg().compile()
        assert isinstance(csdfg, dace.codegen.nanobind_compiled_sdfg.NanobindCompiledSDFG)

        n = 16
        a = (np.arange(n) * 0.5).astype(np.float16)
        b = np.zeros(n, dtype=np.float16)
        csdfg(A=a, B=b, N=np.int32(n))
        assert b.dtype == np.float16
        assert np.allclose(b.astype(np.float32), a.astype(np.float32) + 1.0)


def test_nanobind_interface_float16_return_value():
    """E2E: a float16 return array comes back as a numpy float16 array."""
    with set_temporary('compiler', 'interface', value='nanobind'):
        N = dace.symbol('N')

        @dace.program
        def add_one_half_nanobind(A: dace.float16[N]):
            return A + dace.float16(1.0)

        csdfg = add_one_half_nanobind.to_sdfg().compile()
        n = 12
        a = (np.arange(n) * 0.25).astype(np.float16)
        result = csdfg(A=a, N=np.int32(n))
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float16
        assert np.allclose(result.astype(np.float32), a.astype(np.float32) + 1.0)


def test_nanobind_interface_strict_scalar_cast_binding():
    """The strict option adds .noconvert() to a numeric scalar arg; default does not."""
    from dace.codegen.nanobind_bindings import generate_bindings_code

    def build():
        sdfg = dace.SDFG('strict_scalar_probe')
        sdfg.add_scalar('a', dace.int32)
        return sdfg

    with set_temporary('compiler', 'nanobind_strict_scalar_cast', value=True):
        strict_code = generate_bindings_code(build())
    with set_temporary('compiler', 'nanobind_strict_scalar_cast', value=False):
        loose_code = generate_bindings_code(build())

    assert 'nb::arg("a").noconvert()' in strict_code
    assert 'nb::arg("a").noconvert()' not in loose_code
    assert 'nb::arg("a")' in loose_code  # present, just without .noconvert()


def test_nanobind_interface_strict_scalar_cast_runtime():
    """Strict off allows a safe widening scalar cast (int -> double); strict on rejects it."""
    import pytest

    with set_temporary('compiler', 'interface', value='nanobind'):
        # Default (off): a Python int widens to the double parameter.
        @dace.program
        def widen_off_prog(a: dace.float64):
            return a + 1.0

        off = widen_off_prog.to_sdfg().compile()
        result = off(2)  # Python int -> double (widening), accepted
        assert np.isclose(result[0], 3.0)
        result = off(np.float64(2.0))  # exact-width numpy scalar, accepted
        assert np.isclose(result[0], 3.0)

        # On: the same widening is rejected. A distinct SDFG name keeps the
        # .dacecache entry separate from the off build (the cache key is the SDFG
        # hash, which does not capture the config option).
        with set_temporary('compiler', 'nanobind_strict_scalar_cast', value=True):

            @dace.program
            def widen_on_prog(a: dace.float64):
                return a + 1.0

            on = widen_on_prog.to_sdfg().compile()
            with pytest.raises(Exception):
                on(2)  # Python int -> double rejected under .noconvert()
            # .noconvert() also disables the __index__ path, so numpy scalars
            # are rejected even at the exact width: strict means built-in
            # Python scalar types only.
            with pytest.raises(Exception):
                on(np.float64(2.0))
            result = on(2.0)  # a genuine Python float still passes
            assert np.isclose(result[0], 3.0)


def _uargs_axpy_sdfg(simplify=True):
    """Shared probe for the user_args tests: arglist A, B, alpha + size symbol N."""
    N = dace.symbol('N')

    @dace.program
    def uargs_axpy(A: dace.float64[N], B: dace.float64[N], alpha: dace.float64):
        B[:] = alpha * A + B

    return uargs_axpy.to_sdfg(simplify=simplify)


def test_nanobind_interface_user_args_binding():
    """A non-empty ``user_args`` generates ``user_call``: tuple entries bind as
    ``nb::tuple`` with per-element extraction, plain entries bind like call()'s
    parameters, no kwargs absorber exists, and unlisted inferable symbols are
    plain const locals (no optional machinery)."""
    from dace.codegen.nanobind_bindings import generate_bindings_code

    sdfg = _uargs_axpy_sdfg()
    sdfg.user_args = [('A', 'B'), 'alpha']
    code = generate_bindings_code(sdfg)

    assert '.def("user_call"' in code
    sig = code.split('void user_call(')[1].split(') {')[0]
    body = code.split('void user_call(')[1]
    assert 'nb::tuple arg1' in sig
    assert 'double alpha' in sig
    assert 'kwargs' not in sig  # no kwargs at all: everything listed or inferable
    # Per-element extraction: length check + convert-controlled casts.
    assert 'nb::len(arg1)' in body
    assert 'try_cast' in body
    # N is unlisted and inferable: a plain const local, no std::optional dance.
    assert 'const int N = ' in body
    assert 'A.shape(0)' in body
    assert 'N__opt' not in body.split('void call(')[0]


def test_nanobind_interface_user_args_not_generated_when_empty():
    """Without user_args nothing is generated - the feature is fully opt-in."""
    from dace.codegen.nanobind_bindings import generate_bindings_code

    code = generate_bindings_code(_uargs_axpy_sdfg())
    assert 'user_call' not in code


def test_nanobind_interface_user_args_validation():
    """user_args is validated at codegen time with clear errors."""
    from dace import dtypes
    from dace.codegen.nanobind_bindings import generate_bindings_code

    def probe(user_args):
        sdfg = _uargs_axpy_sdfg()
        sdfg.user_args = user_args
        return generate_bindings_code(sdfg)

    with pytest.raises(ValueError, match="unknown argument 'nope'"):
        probe([('A', 'nope'), 'B', 'alpha'])
    with pytest.raises(ValueError, match="'A'.*more than once"):
        probe([('A', 'A'), 'B', 'alpha'])
    with pytest.raises(ValueError, match='empty tuple'):
        probe([('A', ()), 'B', 'alpha'])
    with pytest.raises(ValueError, match="missing argument"):
        probe([('A', 'B')])  # alpha unlisted and not inferable

    # A string scalar is outside the initial primitive-only scope.
    sdfg = dace.SDFG('uargs_string_probe')
    sdfg.add_array('A', [1], dace.float64)
    sdfg.add_scalar('s', dtypes.string)
    sdfg.user_args = ['A', 's']
    with pytest.raises(ValueError, match="'s'.*not supported"):
        generate_bindings_code(sdfg)

    # A nullable array is outside the initial scope.
    sdfg = dace.SDFG('uargs_nullable_probe')
    sdfg.add_array('A', [1], dace.float64)
    sdfg.arrays['A'].optional = True
    sdfg.user_args = ['A']
    with pytest.raises(ValueError, match="'A'.*not supported"):
        generate_bindings_code(sdfg)

    # Return-value SDFGs are refused (the fast path allocates nothing).
    N = dace.symbol('N')

    @dace.program
    def uargs_ret(A: dace.float64[10]):
        return A + 1.0

    sdfg = uargs_ret.to_sdfg(simplify=True)
    sdfg.user_args = ['A']
    with pytest.raises(ValueError, match='return'):
        generate_bindings_code(sdfg)


def test_nanobind_interface_user_args_e2e():
    """E2E: structured call through user_bind_call, by-reference semantics kept."""
    with set_temporary('compiler', 'interface', value='nanobind'):
        sdfg = _uargs_axpy_sdfg()
        sdfg.user_args = [('A', 'B'), 'alpha']
        csdfg = sdfg.compile()

        n = 16
        a = np.random.rand(n)
        b = np.random.rand(n)
        expected = 2.0 * a + b
        csdfg.user_bind_call((a, b), 2.0)  # N inferred from A.shape(0)
        assert np.allclose(b, expected)

        # A float32 array element must be rejected, never silently copied.
        with pytest.raises(Exception):
            csdfg.user_bind_call((np.zeros(n, dtype=np.float32), b), 2.0)


def test_nanobind_interface_user_args_position_name_collision():
    """An SDFG argument literally named like a synthesized positional
    parameter (arg1, arg2, ...) must not be shadowed by it: the synthesized
    C++ names are mangled away from real argument names (trailing '_')."""
    from dace.codegen.nanobind_bindings import generate_bindings_code

    with set_temporary('compiler', 'interface', value='nanobind'):
        sdfg = dace.SDFG('uargs_argname_clash')
        sdfg.add_array('arg1', [4], dace.float64)
        sdfg.add_array('B', [4], dace.float64)
        state = sdfg.add_state()
        t = state.add_tasklet('t', {'i'}, {'o'}, 'o = i + 1.0')
        state.add_edge(state.add_read('arg1'), None, t, 'i', dace.Memlet('arg1[0]'))
        state.add_edge(t, 'o', state.add_write('B'), None, dace.Memlet('B[0]'))
        sdfg.user_args = [('arg1', 'B')]

        # The tuple parameter at position 1 yields its name to the real
        # argument 'arg1' listed inside it.
        sig = generate_bindings_code(sdfg).split('void user_call(')[1].split(') {')[0]
        assert 'nb::tuple arg1_' in sig

        # The real proof is that it compiles and runs (RED: C++ shadowing).
        csdfg = sdfg.compile()
        a = np.ones(4)
        b = np.zeros(4)
        csdfg.user_bind_call((a, b))
        assert b[0] == 2.0


def test_nanobind_interface_user_args_pyobject_scalar_binding():
    """A pyobject scalar may be listed in user_args: top-level it binds as
    nb::object, nested it arrives via try_cast<nb::object>; the raw PyObject*
    is forwarded either way."""
    from dace import dtypes
    from dace.codegen.nanobind_bindings import generate_bindings_code

    def make():
        sdfg = dace.SDFG('uargs_pyobj_probe')
        sdfg.add_scalar('obj', dtypes.pyobject())
        sdfg.add_array('A', [4], dace.float64)
        return sdfg

    sdfg = make()
    sdfg.user_args = ['obj', 'A']
    ucall = generate_bindings_code(sdfg).split('void user_call(')[1]
    assert 'nb::object obj' in ucall.split(') {')[0]
    assert 'reinterpret_cast<pyobject>(obj.ptr())' in ucall

    sdfg = make()
    sdfg.user_args = [('obj', 'A')]
    ucall = generate_bindings_code(sdfg).split('void user_call(')[1]
    assert 'try_cast<nb::object>' in ucall
    assert 'reinterpret_cast<pyobject>(obj.ptr())' in ucall


def test_nanobind_interface_user_args_pyobject_e2e():
    """E2E: a pyobject rides through user_bind_call in a nested position
    without disturbing its neighbors."""
    from dace import dtypes

    with set_temporary('compiler', 'interface', value='nanobind'):
        sdfg = _uargs_axpy_sdfg()
        sdfg.add_scalar('obj', dtypes.pyobject())
        sdfg.user_args = [('A', 'obj'), 'B', 'alpha']
        csdfg = sdfg.compile()

        n = 16
        a = np.random.rand(n)
        b = np.random.rand(n)
        expected = 2.0 * a + b
        csdfg.user_bind_call((a, object()), b, 2.0)
        assert np.allclose(b, expected)


def test_nanobind_interface_user_args_ignore_slots_binding():
    """'' entries are ignored placeholder slots: top-level they are nb::object
    parameters accepting anything (incl. None) and never read; nested they are
    counted by the tuple length check but never extracted."""
    from dace.codegen.nanobind_bindings import generate_bindings_code

    sdfg = _uargs_axpy_sdfg()
    sdfg.user_args = ['', ('A', '', 'B'), 'alpha']
    ucall = generate_bindings_code(sdfg).split('void user_call(')[1]
    sig = ucall.split(') {')[0]
    body = ucall

    assert 'nb::object' in sig  # the top-level ignored slot
    assert 'nb::arg("arg1").none()' in body  # placeholder accepts None too
    assert 'nb::len(arg2) != 3' in body  # the nested ignored slot is counted...
    assert 'arg2[0]' in body and 'arg2[2]' in body
    assert 'arg2[1]' not in body  # ...but never extracted


def test_nanobind_interface_user_args_ignore_slots_e2e():
    """E2E: ignored slots swallow arbitrary values (None, dicts) while the real
    entries around them keep working; the tuple length check still counts them."""
    with set_temporary('compiler', 'interface', value='nanobind'):
        sdfg = _uargs_axpy_sdfg()
        sdfg.user_args = ['', ('A', '', 'B'), 'alpha']
        csdfg = sdfg.compile()

        n = 16
        a = np.random.rand(n)
        b = np.random.rand(n)
        expected = 2.0 * a + b
        csdfg.user_bind_call(None, (a, {'junk': 1}, b), 2.0)
        assert np.allclose(b, expected)

        # The ignored nested slot still counts toward the tuple length.
        with pytest.raises(Exception):
            csdfg.user_bind_call(None, (a, b), 2.0)

        # Wrong tuple length is a clear error.
        with pytest.raises(Exception):
            csdfg.user_bind_call((a, ), 2.0)


def test_nanobind_interface_user_args_nested_e2e():
    """E2E: nested tuples destructure (the idea.md example shape)."""
    with set_temporary('compiler', 'interface', value='nanobind'):
        sdfg = _uargs_axpy_sdfg()
        sdfg.user_args = [('A', ('B', ), 'alpha')]
        csdfg = sdfg.compile()

        n = 8
        a = np.random.rand(n)
        b = np.random.rand(n)
        expected = 3.0 * a + b
        csdfg.user_bind_call((a, (b, ), 3.0))
        assert np.allclose(b, expected)


def test_nanobind_interface_user_args_cross_symbol_inference():
    """A listed symbol is promised, so an unlisted one is inferable through a
    multi-symbol shape: A[a + b] with 'b' listed infers a = A.shape(0) - b."""
    from dace.codegen.nanobind_bindings import generate_bindings_code

    a = dace.symbol('a')
    b = dace.symbol('b')
    sdfg = dace.SDFG('uargs_cross_sym_probe')
    sdfg.add_array('A', [a + b], dace.float64)
    state = sdfg.add_state()
    tasklet = state.add_tasklet('set_last', {}, {'o'}, 'o = 1.0')
    state.add_edge(tasklet, 'o', state.add_write('A'), None, dace.Memlet('A[a + b - 1]'))
    sdfg.user_args = ['A', 'b']

    code = generate_bindings_code(sdfg)
    body = code.split('void user_call(')[1]
    assert 'const int a = ' in body
    assert 'A.shape(0) - b' in body


def test_nanobind_interface_user_args_serialization_and_hash():
    """user_args is a serialized SDFG property: it survives a JSON roundtrip
    and changes the SDFG hash (so the build cache rebuilds on change)."""
    sdfg = _uargs_axpy_sdfg()
    hash_without = sdfg.hash_sdfg()

    sdfg.user_args = [('A', 'B'), 'alpha']
    restored = dace.SDFG.from_json(sdfg.to_json())
    # JSON has no tuples; entries come back as sequences with the same nesting.
    assert [list(e) if not isinstance(e, str) else e for e in restored.user_args] \
        == [['A', 'B'], 'alpha']
    assert sdfg.hash_sdfg() != hash_without


def test_nanobind_interface_user_bind_call_requires_user_args():
    """user_bind_call on a module compiled without user_args raises clearly."""
    with set_temporary('compiler', 'interface', value='nanobind'):
        csdfg = _uargs_axpy_sdfg().compile()
        with pytest.raises(ValueError, match='user_args'):
            csdfg.user_bind_call((np.zeros(4), np.zeros(4)), 1.0)


def test_nanobind_interface_user_bind_call_gpu_error_check(monkeypatch):
    """user_bind_call keeps the GPU last-error check, gated by the
    gpu_error_check property like every other call path."""
    import types
    from dace.codegen import common
    from dace.codegen.nanobind_compiled_sdfg import NanobindCompiledSDFG

    sdfg = dace.SDFG('uargs_gpu_error_probe')
    sdfg.add_array('A', [10], dace.float64, storage=dace.StorageType.GPU_Global)

    class FakeHandle:
        has_gpu_code = True
        return_names = ()
        is_single_value_ret = False
        callback_names = ()

        def user_call(self, *args):
            pass

    stub_module = types.SimpleNamespace(make_compiled_sdfg=lambda: FakeHandle(), __file__='<stub>')

    class FakeRuntime:

        def get_last_error_string(self):
            return 'illegal memory access'

    monkeypatch.setattr(common, 'get_gpu_runtime', lambda: FakeRuntime())

    csdfg = NanobindCompiledSDFG(sdfg, stub_module, ['A'])
    with pytest.raises(RuntimeError, match='illegal memory access'):
        csdfg.user_bind_call((object(), ))

    csdfg.gpu_error_check = False
    csdfg.user_bind_call((object(), ))  # does not raise


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
    test_nanobind_interface_load_distinct_artifact_coexists()
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
    test_nanobind_interface_single_element_tuple_return()
    test_nanobind_interface_non_array_return_rejected()
    test_nanobind_interface_many_return_values()
    test_nanobind_interface_optional_struct_array_binding()
    test_nanobind_interface_optional_struct_array_input()
    test_nanobind_interface_strict_scalar_cast_binding()
    test_nanobind_interface_strict_scalar_cast_runtime()
