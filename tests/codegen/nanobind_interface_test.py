# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the nanobind-based `CompiledSDFG` interface (`compiler.interface=nanobind`)."""
import sys

import numpy as np
import pytest

import dace
from dace.config import set_temporary


@pytest.fixture
def nanobind_interface(monkeypatch):
    """Pins ``compiler.interface`` to nanobind for the duration of a test.

    Not ``autouse``: a good third of this file only calls
    ``generate_bindings_code`` and compiles nothing, and those tests should not
    depend on a config change they never read.

    ``DACE_compiler_interface`` takes precedence over ``set_temporary``, so it
    is dropped first. That is what makes these tests self-sufficient: they
    exercise the nanobind interface whatever interface the surrounding CI leg
    selects, so the ctypes legs and the GPU job produce real nanobind coverage.
    """
    monkeypatch.delenv('DACE_compiler_interface', raising=False)
    with set_temporary('compiler', 'interface', value='nanobind'):
        yield


def test_axpy_nanobind_interface(nanobind_interface):
    """Stage-1 acceptance: an axpy-class SDFG runs end-to-end on the nanobind interface."""
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


def test_nanobind_interface_wrong_dtype_raises(nanobind_interface):
    """A wrong-dtype array is rejected by the generated marshalling code with a typed error."""
    import pytest

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


def test_nanobind_interface_same_name_recompile(nanobind_interface):
    """Recompiling under an already-imported module name silently renames (sys.modules increment)."""
    # Pins the rename machinery; reuse has its own test.
    with set_temporary('compiler', 'nanobind_reuse_loaded', value=False):
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


def test_nanobind_interface_return_value(nanobind_interface):
    """A program with a return array allocates it in Python and returns it."""
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


def test_nanobind_interface_return_override_forbidden_by_default(nanobind_interface):
    """By default the nanobind interface refuses a caller-provided __return buffer."""
    import pytest

    @dace.program
    def double_ret_default(A: dace.float64[20]):
        return A * 2

    csdfg = double_ret_default.to_sdfg().compile()
    a = np.random.rand(20)
    out = np.empty(20, dtype=np.float64)
    with pytest.raises(ValueError, match='nanobind_allow_return_override'):
        csdfg(A=a, __return=out)


def test_nanobind_interface_return_override_allowed(nanobind_interface):
    """With the option on, a caller-provided __return buffer is written in place and returned."""
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


def test_nanobind_interface_return_override_wrong_dtype_rejected_by_binding(nanobind_interface):
    """With the option on, no Python-side type check is imposed: a buffer the
    nanobind binding cannot accept (wrong dtype) is rejected by the binding."""
    import pytest
    with set_temporary('compiler', 'nanobind_allow_return_override', value=True):

        @dace.program
        def double_ret_dtype(A: dace.float64[20]):
            return A * 2

        csdfg = double_ret_dtype.to_sdfg().compile()
        a = np.random.rand(20)
        wrong = np.zeros(20, dtype=np.float32)  # binding expects float64
        with pytest.raises(Exception):
            csdfg(A=a, __return=wrong)


def test_nanobind_interface_return_override_too_small_rejected(nanobind_interface):
    """With the option on, a caller-provided buffer SMALLER than the
    symbol-derived return size is rejected (the program writes through the
    descriptor's shape and strides - a too-small buffer means out-of-bounds
    writes). A LARGER buffer is a legitimate pattern and passes: the program
    fills its prefix and the tail stays untouched (the contract
    local_storage_test's test_uneven relies on)."""
    import pytest
    with set_temporary('compiler', 'nanobind_allow_return_override', value=True):

        @dace.program
        def double_ret_shape(A: dace.float64[20]):
            return A * 2

        csdfg = double_ret_shape.to_sdfg().compile()
        a = np.random.rand(20)
        with pytest.raises(Exception, match='shape'):
            csdfg(A=a, __return=np.zeros(16, dtype=np.float64))  # too small
        big = np.ones(25, dtype=np.float64)
        result = csdfg(A=a, __return=big)
        assert result is big
        assert np.allclose(big[:20], a * 2)  # prefix written...
        assert np.allclose(big[20:], 1.0)  # ...tail untouched


def test_nanobind_interface_positional_and_extra_kwargs(nanobind_interface):
    """Positional calls work, and extra keyword arguments are absorbed (old-interface behavior)."""
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


def test_nanobind_interface_has_gpu_code(nanobind_interface):
    """The handle and the shell expose has_gpu_code (False for a CPU-only SDFG)."""
    N = dace.symbol('N')

    @dace.program
    def axpy_nanobind_gpuq(A: dace.float64[N], B: dace.float64[N], alpha: dace.float64):
        B[:] = alpha * A + B

    csdfg = axpy_nanobind_gpuq.to_sdfg().compile()
    assert csdfg.has_gpu_code is False


def test_nanobind_interface_state_pointer(nanobind_interface):
    """state_pointer raises while the state is uninitialized or after finalize."""
    import pytest

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


def test_nanobind_interface_get_state_struct_parity(monkeypatch):
    """get_state_struct exposes the same leading state-struct pointer fields as
    the ctypes interface, as a live ctypes.Structure overlay of state memory."""
    import ctypes

    # Builds under BOTH interfaces, so it cannot take the nanobind_interface
    # fixture - but it needs the same delenv: DACE_compiler_interface overrides
    # set_temporary, and a leg that pins nanobind would make the ctypes half a
    # second nanobind build and the comparison vacuous.
    monkeypatch.delenv('DACE_compiler_interface', raising=False)
    # The ctypes half needs DEVELOPMENT folder mode: its get_state_struct
    # recovers the layout by parsing the generated src/cpu/<name>.cpp, and
    # production mode trims the sources away (it also moves the .so up out of
    # build/). The nanobind half bakes the field names in at codegen time and
    # does not care. Env first - it overrides set_temporary.
    monkeypatch.delenv('DACE_compiler_build_folder_mode', raising=False)

    def build_and_fields(interface):
        with set_temporary('compiler', 'interface', value=interface), \
                set_temporary('compiler', 'build_folder_mode', value='development'):
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


def test_nanobind_interface_rename_own_build_folder(nanobind_interface):
    """A collision-renamed program is compiled into its own build folder, not in-place."""
    import os

    # Pins the rename machinery; reuse has its own test.
    with set_temporary('compiler', 'nanobind_reuse_loaded', value=False):
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


def test_nanobind_interface_reuse_unchanged_module(nanobind_interface):
    """Recompiling an UNCHANGED SDFG whose identity is already loaded reuses
    the loaded module - no rename, no rebuild: the module bakes the source
    SDFG's pre-codegen content hash (`source_sdfg_hash`) and compile()
    compares against it before entering the rename loop. A CHANGED SDFG
    still renames-and-rebuilds (the module cannot be reloaded). Gated by
    ``compiler.nanobind_reuse_loaded`` (default: enabled)."""
    import sympy

    from dace import symbolic

    # The comparison rides on hash_sdfg(); clear the known cross-test sympy
    # cache pollution so an unrelated prior test cannot flip a hash bit
    # (see the symbolic-serialization flake).
    sympy.core.cache.clear_cache()
    symbolic.deserialize_symbolic.cache_clear()

    N = dace.symbol('N')

    @dace.program
    def axpy_nanobind_reuse(A: dace.float64[N], B: dace.float64[N], alpha: dace.float64):
        B[:] = alpha * A + B

    sdfg = axpy_nanobind_reuse.to_sdfg()
    base_name = sdfg.name
    c1 = sdfg.compile()
    assert hasattr(c1.module, 'source_sdfg_hash')

    c2 = sdfg.compile()
    assert c2.sdfg.name == base_name  # not renamed...
    assert c2.module is c1.module  # ...the very same loaded module

    n = 16
    a = np.random.rand(n)
    b = np.random.rand(n)
    expected = 2.0 * a + b
    c2(A=a, B=b, alpha=np.float64(2.0), N=np.int32(n))
    assert np.allclose(b, expected)

    # A changed SDFG must still rename-and-rebuild.
    sdfg.instrument = dace.InstrumentationType.Timer
    c3 = sdfg.compile()
    assert c3.sdfg.name == f'{base_name}_0'


def test_nanobind_interface_external_nested_sdfg(nanobind_interface):
    """An SDFG that still carries an UNRESOLVED external nested SDFG
    (``NestedSDFG.sdfg is None``; the content is only loaded from
    ``ext_sdfg_path``) compiles, runs, and takes part in content reuse:
    the reuse key is computed on a resolved deepcopy - ``to_json()``
    cannot recurse into a missing nested SDFG, and ``self`` must stay
    unresolved."""
    import os
    import tempfile

    import sympy

    from dace import symbolic

    # The reuse assertions ride on hash_sdfg(); clear the known cross-test
    # sympy cache pollution (see test_nanobind_interface_reuse_unchanged_module).
    sympy.core.cache.clear_cache()
    symbolic.deserialize_symbolic.cache_clear()

    inner = dace.SDFG('nb_ext_inner')
    inner.add_array('xin', [1], dace.float64)
    inner.add_array('xout', [1], dace.float64)
    istate = inner.add_state()
    itask = istate.add_tasklet('inc', {'a'}, {'b'}, 'b = a + 1')
    istate.add_edge(istate.add_read('xin'), None, itask, 'a', dace.Memlet('xin[0]'))
    istate.add_edge(itask, 'b', istate.add_write('xout'), None, dace.Memlet('xout[0]'))

    fd, filename = tempfile.mkstemp(suffix='.sdfg')
    try:
        inner.save(filename)

        outer = dace.SDFG('nb_ext_outer')
        outer.add_array('A', [1], dace.float64)
        outer.add_array('B', [1], dace.float64)
        state = outer.add_state()
        nsdfg = state.add_nested_sdfg(None, {'xin'}, {'xout'}, name='nb_ext_inner', external_path=filename)
        state.add_edge(state.add_read('A'), None, nsdfg, 'xin', dace.Memlet('A[0]'))
        state.add_edge(nsdfg, 'xout', state.add_write('B'), None, dace.Memlet('B[0]'))

        a = np.array([2.0])
        b = np.zeros(1)
        outer(A=a, B=b)
        assert b[0] == 3.0
        # The caller's object stays unresolved (only copies are loaded).
        assert nsdfg.sdfg is None

        # The hash rides on the resolved copy, so an unchanged external
        # SDFG reuses the loaded module (no rename, no rebuild).
        c1 = outer.compile()
        c2 = outer.compile()
        assert c2.sdfg.name == outer.name
        assert c2.module is c1.module
    finally:
        os.close(fd)
        os.unlink(filename)


def test_nanobind_interface_handle_sdfg_isolated(nanobind_interface):
    """The handle's ``sdfg`` is isolated from the caller's object on EVERY
    compile() return path - freshly built, content-reuse, use_cache
    cached-binary, and the regenerate_code=False branch. Passing ``self``
    (the old behavior of the non-codegen paths) leaked later mutations of
    the original into the handle."""
    with pytest.MonkeyPatch.context() as mp, \
            set_temporary('compiler', 'build_folder_mode', value='development'):
        # The else-branch leg (c4) rebuilds from the generated sources, which
        # only exist in development folder mode (production trims them) - and
        # an existing folder's FOLDER_MODE marker overrides the config, so the
        # WHOLE test pins development mode. The env var must be dropped first
        # (it overrides set_temporary, see the CI-env gotcha).
        mp.delenv('DACE_compiler_build_folder_mode', raising=False)
        N = dace.symbol('N')

        @dace.program
        def axpy_nanobind_isolated(A: dace.float64[N], B: dace.float64[N], alpha: dace.float64):
            B[:] = alpha * A + B

        sdfg = axpy_nanobind_isolated.to_sdfg()
        base_name = sdfg.name

        c1 = sdfg.compile()  # freshly built (codegen deepcopy)
        assert c1.sdfg is not sdfg

        c2 = sdfg.compile()  # content-reuse path
        assert c2.sdfg is not sdfg

        with set_temporary('compiler', 'use_cache', value=True):
            c3 = sdfg.compile()  # cached-binary path
            assert c3.sdfg is not sdfg

        # regenerate_code=False + existing folder: the else branch. Reuse
        # must be off, or it would win first.
        with set_temporary('compiler', 'nanobind_reuse_loaded', value=False):
            sdfg.regenerate_code = False
            try:
                c4 = sdfg.compile()
            finally:
                sdfg.regenerate_code = True
            assert c4.sdfg is not sdfg

        # Mutating the original afterwards does not leak into any handle.
        sdfg.name = f'{base_name}_mutation_probe'
        try:
            for c in (c1, c2, c3, c4):
                assert c.sdfg.name != sdfg.name
        finally:
            sdfg.name = base_name


def test_nanobind_interface_rename_explicit_folder_stays(nanobind_interface, tmp_path):
    """An explicitly-set build folder is the user's contract: a collision-renamed
    program builds in place inside it (the fixed-folder regime, same behaviour
    as cache mode 'single') instead of re-deriving its own folder."""
    import os

    # Pins the rename machinery; reuse has its own test.
    with set_temporary('compiler', 'nanobind_reuse_loaded', value=False):
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


def test_nanobind_interface_rename_third_compile_consistent(nanobind_interface):
    """Three same-named compiles yield base, _0, _1 - the collision probe must
    track the folder each candidate actually builds into, or the third compile
    would silently reuse the stale _0 module."""
    # Pins the rename machinery; reuse has its own test.
    with set_temporary('compiler', 'nanobind_reuse_loaded', value=False):
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


def test_nanobind_interface_report_follows_rename(nanobind_interface):
    """Instrumentation reports of a collision-renamed program are found via the
    compiled handle's sdfg (the renamed compile copy, which knows its own
    folder). The ORIGINAL object keeps looking in its identity-derived folder
    and finds nothing - the accepted limitation behind refusing
    SDFG.safe_call() on nanobind."""
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


def test_nanobind_interface_perf_folder_only_when_instrumented(nanobind_interface):
    """The perf/ report folder is created exactly when the SDFG is
    instrumented, in BOTH folder modes. Production mode used to skip it
    entirely, silently dropping every report (the runtime's report.save()
    neither creates directories nor reports a failed open); uninstrumented
    folders stay lean in both modes."""
    import os

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


def test_nanobind_interface_sdfg_safe_call_refused(nanobind_interface):
    """SDFG.safe_call() is refused on the nanobind interface: it compiles
    internally and hides the compiled object, so after a collision rename any
    post-call query on the original SDFG (e.g. get_latest_report()) would
    silently look in the wrong folder. compile() + CompiledSDFG.safe_call()
    is the supported route."""
    sdfg = dace.SDFG('sdfg_safe_call_refuse_probe')
    sdfg.add_array('A', [4], dace.float64)
    with pytest.raises(NotImplementedError, match='safe_call'):
        sdfg.safe_call(A=np.zeros(4))


def test_nanobind_interface_name_collision_error(nanobind_interface):
    """With compiler.nanobind_name_collision=error, a taken name refuses to compile."""
    import pytest

    # Pins the rename machinery; reuse has its own test.
    with set_temporary('compiler', 'nanobind_reuse_loaded', value=False):
        with set_temporary('compiler', 'nanobind_name_collision', value='error'):
            N = dace.symbol('N')

            @dace.program
            def axpy_nanobind_collerr(A: dace.float64[N], B: dace.float64[N], alpha: dace.float64):
                B[:] = alpha * A + B

            axpy_nanobind_collerr.to_sdfg().compile()
            with pytest.raises(Exception, match='already loaded'):
                axpy_nanobind_collerr.to_sdfg().compile()


def test_nanobind_interface_workspace(nanobind_interface):
    """External-memory workspace functions work on the nanobind interface."""
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


def test_nanobind_interface_get_exported_function(nanobind_interface):
    """Arbitrary exported symbols stay reachable, with the wrapper as keep-alive."""
    N = dace.symbol('N')

    @dace.program
    def axpy_nanobind_expfun(A: dace.float64[N], B: dace.float64[N], alpha: dace.float64):
        B[:] = alpha * A + B

    csdfg = axpy_nanobind_expfun.to_sdfg().compile()
    func = csdfg.get_exported_function(f'__dace_exit_{csdfg.sdfg.name}')
    assert func is not None
    assert func.__compiled_sdfg__ is csdfg
    assert csdfg.get_exported_function('definitely_not_a_symbol') is None


def test_nanobind_interface_pyobject_array_and_return_binding():
    """pyobject ARRAY arguments and pyobject RETURNS both bind through the
    array-interface dict, not through nb::ndarray.

    DLPack refuses object arrays outright ("DLPack only supports signed/unsigned
    integers, float and complex dtypes"), exactly as it refuses the ml_dtypes-backed
    low-precision types, so both take the same __array_interface__ route: read the
    dict, check the typestr kind letter, cast the raw pointer. A pyobject return is
    additionally decayed to the single contained object on the way out, which is what
    the ctypes interface does (``_return_arrays[i].item()`` whenever the return is a
    pyobject).
    """
    from dace import dtypes
    from dace.codegen.nanobind_bindings import generate_bindings_code

    # Array argument: nb::object parameter, pointer from the interface dict.
    arg_sdfg = dace.SDFG('pyobject_arr_bind_probe')
    arg_sdfg.add_array('objs', [4], dtypes.pyobject())
    code = generate_bindings_code(arg_sdfg)
    assert 'nb::object objs' in code
    assert '__array_interface__' in code
    assert "objs__ts[1] != 'O'" in code  # object arrays are typestr '|O' - no size suffix
    assert 'reinterpret_cast<pyobject *>(objs__ptr)' in code

    # Return value: allocated as a numpy object array, decayed via .item().
    ret_sdfg = dace.SDFG('pyobject_return_bind_probe')
    ret_sdfg.add_array('__return', [1], dtypes.pyobject())
    code = generate_bindings_code(ret_sdfg)
    assert '__mod.attr("dtype")("object")' in code
    assert 'return __return__obj.attr("item")();' in code


def test_nanobind_interface_pyobject_array_arg_e2e(nanobind_interface):
    """E2E: every element of an object array reaches a callback as the very same
    Python object. The caller's array owns the references and the nb::object
    parameter keeps it alive across the call, so the slots stay valid while the
    program dereferences them - the same lifetime contract as ctypes, which also
    hands out pointers into the caller's buffer."""
    from dace import dtypes

    sdfg = dace.SDFG('pyobject_array_passthrough')
    sdfg.add_array('objs', [3], dtypes.pyobject())
    sdfg.add_array('A', [3], dace.float64)
    sdfg.add_symbol('consume', dace.callback(None, dtypes.pyobject()))
    state = sdfg.add_state()
    me, mx = state.add_map('m', dict(i='0:3'))
    t = state.add_tasklet('t', {'o_in'}, {'a_out'}, 'consume(o_in)\na_out = 1.0')
    state.add_memlet_path(state.add_read('objs'), me, t, dst_conn='o_in', memlet=dace.Memlet('objs[i]'))
    state.add_memlet_path(t, mx, state.add_write('A'), src_conn='a_out', memlet=dace.Memlet('A[i]'))

    csdfg = sdfg.compile()

    class Payload:

        def __init__(self, tag):
            self.tag = tag

    payloads = [Payload(0), Payload(1), Payload(2)]
    objs = np.empty(3, dtype=object)
    objs[:] = payloads
    seen = []
    a = np.zeros(3)
    csdfg(objs=objs, A=a, consume=lambda x: seen.append(x))

    assert len(seen) == 3
    assert sorted(o.tag for o in seen) == [0, 1, 2]
    assert all(s is p for s, p in zip(sorted(seen, key=lambda o: o.tag), payloads))  # identity, not copies
    assert np.allclose(a, 1.0)


def test_nanobind_interface_pyobject_return_e2e(nanobind_interface):
    """E2E: a pyobject return comes back as the object itself, not as a 1-element
    object array. Asserted against the ctypes interface in the same test, since
    the decay-to-single-object convention is what is being matched."""
    from dace import dtypes

    def build_and_run(interface):
        with set_temporary('compiler', 'interface', value=interface):
            sdfg = dace.SDFG(f'pyobj_ret_parity_{interface}')
            sdfg.add_array('A', [4], dace.float64)
            sdfg.arrays['A'].optional = False
            sdfg.add_array('__return', [1], dtypes.pyobject())
            sdfg.add_symbol('produce', dace.callback(dtypes.pyobject()))
            sdfg.arg_names = ['A']
            st = sdfg.add_state()
            t = st.add_tasklet('t', {}, {'o'}, 'o = produce()')
            st.add_edge(t, 'o', st.add_write('__return'), None, dace.Memlet('__return[0]'))
            csdfg = sdfg.compile()
            payload = {'tag': 'the-object'}
            return csdfg(A=np.zeros(4), produce=lambda: payload), payload

    nb_result, nb_payload = build_and_run('nanobind')
    assert nb_result is nb_payload  # the object itself, and the very same one

    ct_result, ct_payload = build_and_run('ctypes')
    assert ct_result is ct_payload
    assert type(nb_result) is type(ct_result)


def test_nanobind_interface_lowp_return_binding():
    """A bfloat16/float8 RETURN allocates through ml_dtypes and takes its pointer
    from the array-interface dict.

    numpy cannot resolve the dtype NAME without ml_dtypes imported (np.dtype('bfloat16')
    raises TypeError), and nb::ndarray cannot ingest the result, so the allocation
    imports ml_dtypes and the extraction skips the nb::cast. The storage is NOT
    special-cased: the dtype goes to CuPy for a GPU return exactly as the ctypes
    allocator hands it over, and CuPy succeeds or raises on its own.
    """
    from dace.codegen.nanobind_bindings import generate_bindings_code

    sdfg = dace.SDFG('lowp_return_bind_probe')
    sdfg.add_array('A', [8], dace.bfloat16)
    sdfg.add_array('__return', [8], dace.bfloat16)
    sdfg.arrays['__return'].optional = False
    code = generate_bindings_code(sdfg)
    assert 'nb::module_::import_("ml_dtypes")' in code
    assert '__mod.attr("dtype")("bfloat16")' in code
    assert '__return__ai' in code and '__array_interface__' in code
    assert 'reinterpret_cast<dace::bfloat16 *>(__return__ptr)' in code
    assert 'nb::cast<nb::ndarray<' not in code.split('__return__obj')[1][:400]  # no DLPack view

    # GPU storage takes the CUDA flavour of the protocol and allocates via CuPy.
    gpu_sdfg = dace.SDFG('lowp_return_gpu_bind_probe')
    gpu_sdfg.add_array('__return', [8], dace.bfloat16, storage=dace.StorageType.GPU_Global)
    gpu_sdfg.arrays['__return'].optional = False
    with set_temporary('compiler', 'cuda', 'backend', value='cuda'):
        gpu_code = generate_bindings_code(gpu_sdfg)
    assert 'nb::module_::import_("cupy")' in gpu_code
    assert '__cuda_array_interface__' in gpu_code


def test_nanobind_interface_lowp_return_e2e(nanobind_interface):
    """E2E: a bfloat16 return array is allocated in the binding and comes back as a
    numpy array of the right dtype and values."""
    ml_dtypes = pytest.importorskip('ml_dtypes')

    sdfg = dace.SDFG('bf16_return_e2e')
    sdfg.add_array('A', [8], dace.bfloat16)
    sdfg.add_array('__return', [8], dace.bfloat16)
    sdfg.arrays['A'].optional = False
    sdfg.arrays['__return'].optional = False
    sdfg.arg_names = ['A']
    st = sdfg.add_state()
    st.add_mapped_tasklet('copy',
                          dict(i='0:8'),
                          dict(inp=dace.Memlet('A[i]')),
                          'out = inp',
                          dict(out=dace.Memlet('__return[i]')),
                          external_edges=True)

    csdfg = sdfg.compile()
    a = np.arange(8, dtype=np.float32).astype(ml_dtypes.bfloat16)
    result = csdfg(A=a)
    assert result.dtype == ml_dtypes.bfloat16
    assert np.array_equal(result.astype(np.float32), a.astype(np.float32))


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


def test_nanobind_interface_pyobject_scalar_arg_e2e(nanobind_interface):
    """A pyobject scalar argument passes through as an opaque PyObject* and
    arrives at a callback as the very same object (identity preserved)."""
    from dace import dtypes

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
    sig = code.split('nb::object call(')[1].split(') {')[0]
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


def test_nanobind_interface_unset_workspace_refused(nanobind_interface):
    """An SDFG with external memory must refuse to run before set_workspace was
    called - running anyway dereferences a null workspace pointer (silent UB
    under ctypes). The requirement is per state: finalize() drops the
    association, so a re-initialized handle must be refused again."""
    N = dace.symbol('N')

    @dace.program
    def extmem_guard_probe(a: dace.float64[N]):
        workspace = dace.ndarray([N], dace.float64, lifetime=dace.AllocationLifetime.External)
        workspace[:] = a
        workspace += 1
        a[:] = workspace

    csdfg = extmem_guard_probe.to_sdfg().compile()

    n = 20
    a = np.random.rand(n)
    with pytest.raises(RuntimeError, match='[Ee]xternal memory .* was not set'):
        csdfg(a=a, N=np.int32(n))

    csdfg.initialize(a, N=np.int32(n))
    wsp = np.random.rand(n)
    csdfg.set_workspace(dace.StorageType.CPU_Heap, wsp, a=a, N=np.int32(n))
    ref = a + 1
    csdfg(a=a, N=np.int32(n))
    assert np.allclose(a, ref)

    # finalize() frees the state the workspace was set on: a fresh state
    # must demand a fresh set_workspace.
    csdfg.finalize()
    with pytest.raises(RuntimeError, match='[Ee]xternal memory .* was not set'):
        csdfg(a=a, N=np.int32(n))


def test_nanobind_interface_workspace_guard_only_with_external_memory():
    """The workspace guard is compiled in exactly when the SDFG has external
    memory - a plain SDFG carries neither the flags nor the check."""
    from dace.codegen.nanobind_bindings import generate_bindings_code

    plain = dace.SDFG('ws_guard_free_probe')
    plain.add_array('A', [10], dace.float64)
    code = generate_bindings_code(plain)
    assert 'was not set' not in code
    assert 'm_ws_set_' not in code

    ext = dace.SDFG('ws_guard_probe')
    ext.add_array('A', [10], dace.float64)
    ext.add_array('wsp', [10],
                  dace.float64,
                  transient=True,
                  storage=dace.StorageType.CPU_Heap,
                  lifetime=dace.AllocationLifetime.External)
    code = generate_bindings_code(ext)
    assert 'm_ws_set_CPU_Heap' in code
    assert 'was not set' in code


def test_nanobind_interface_raw_code_objects_build_ctypes_folder(tmp_path):
    """generate_program_folder with sdfg=None (raw helper code objects, as
    parse_state_struct_test's cuda_helper builds) can generate no bindings, so
    the folder must be a plain ctypes-style artifact regardless of
    compiler.interface - and must not be refused as an unknown interface."""
    import ctypes
    import os
    from dace.codegen import codeobject, targets, compiler as comp
    from dace.codegen.ctypes_compiled_sdfg import ReloadableDLL

    helper_code = '''
    #include <dace/dace.h>
    extern "C" {
        DACE_EXPORTED int the_answer() { return 42; }
    }
    '''
    program = codeobject.CodeObject('nb_raw_helper', helper_code, 'cpp', targets.cpu.CPUCodeGen, 'RawHelper')
    path = str(tmp_path / 'raw_helper')
    with set_temporary('compiler', 'interface', value='nanobind'):
        comp.generate_program_folder(None, [program], path)
        comp.configure_and_compile(path)
    with open(os.path.join(path, 'INTERFACE')) as fh:
        assert fh.read().strip() == 'ctypes'
    # configure_and_compile derives the program name from the folder name.
    dll = ReloadableDLL(comp.get_binary_name(path, 'raw_helper'))
    dll.load()
    fn = dll.get_symbol('the_answer')
    fn.restype = ctypes.c_int
    assert fn() == 42
    dll.unload()


def test_nanobind_interface_initialize_returns_state_handle(nanobind_interface):
    """initialize() returns the state pointer (ctypes-interface parity): it is
    the value functions from get_exported_function take as their state
    argument - SDFG.call_with_instrumented_data passes it to the compiled
    report setter, and a None return reaches C as a null state and crashes."""

    @dace.program
    def init_handle_probe(A: dace.float64[10]):
        A += 1.0

    sdfg = init_handle_probe.to_sdfg()
    csdfg = sdfg.compile()
    handle = csdfg.initialize(np.zeros(10))
    import ctypes
    assert isinstance(handle, ctypes.c_void_p)
    assert handle.value == csdfg._handle.state_pointer
    assert handle.value  # non-null


def test_nanobind_interface_handle_metadata_binding():
    """The handle exposes the codegen-time call metadata - return-array names
    and callback names - so the Python wrapper does not re-derive them from
    naming conventions. The single-vs-tuple return convention is compiled into
    call()'s return statement, not exposed."""
    from dace.codegen.nanobind_bindings import generate_bindings_code

    # Multi-value returns: names in order, returned as a tuple by call().
    sdfg = dace.SDFG('metadata_probe_rets')
    sdfg.add_array('__return_0', [4], dace.float64)
    sdfg.add_array('__return_1', [4], dace.float64)
    code = generate_bindings_code(sdfg)
    assert 'nb::make_tuple("__return_0", "__return_1")' in code
    assert 'return nb::make_tuple(__return_0__obj, __return_1__obj);' in code

    # Single-value return plus a callback: the bare array is returned.
    sdfg = dace.SDFG('metadata_probe_single')
    sdfg.add_array('__return', [4], dace.float64)
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_symbol('cb', dace.callback(dace.float64, dace.float64))
    state = sdfg.add_state()
    t = state.add_tasklet('t', {}, {'o'}, 'o = cb(1.0)')
    state.add_edge(t, 'o', state.add_write('A'), None, dace.Memlet('A[0]'))
    code = generate_bindings_code(sdfg)
    assert 'nb::make_tuple("__return")' in code
    assert 'return __return__obj;' in code
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


def test_nanobind_interface_symbol_inference(nanobind_interface):
    """Omitted size symbols are inferred from array shapes; explicit values still win."""
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


def test_nanobind_interface_symbol_inference_unsimplified(nanobind_interface):
    """E2E: omitting a size symbol works on an unsimplified SDFG too (the CI
    legs run with automatic simplification off)."""
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


def test_nanobind_interface_symbol_inference_stride(nanobind_interface):
    """E2E: a stride symbol is inferred from the passed array's actual stride."""
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


def test_nanobind_interface_return_shape_from_inferred_symbol(nanobind_interface):
    """A symbolic-shaped return no longer requires its symbols explicitly:
    allocation happens in the binding AFTER compiled symbol inference, so an
    inferred symbol can size the return array (previously the Python-side
    allocation demanded the symbol per call)."""
    N = dace.symbol('N')

    @dace.program
    def infer_ret_nanobind(A: dace.float64[N]):
        return A + 1.0

    csdfg = infer_ret_nanobind.to_sdfg().compile()
    a = np.random.rand(16)
    result = csdfg(A=a)  # N inferred from A's shape sizes the return
    assert result.shape == (16, )
    assert np.allclose(result, a + 1.0)


def test_nanobind_interface_symbol_inference_missing(nanobind_interface):
    """An omitted symbol that cannot be inferred raises an error naming it."""
    import pytest
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


def test_nanobind_interface_symbol_inference_cross_symbol(nanobind_interface):
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

    csdfg = build().compile()
    A = np.zeros(10)
    csdfg(A=A, b=np.int32(4))  # no 'a': inferred as A.shape(0) - b = 6
    assert A[9] == 1.0

    # An explicit 'a' still wins over the inference.
    A2 = np.zeros(10)
    csdfg(A=A2, b=np.int32(4), a=np.int32(6))
    assert A2[9] == 1.0


def test_nanobind_interface_scalar_callback(nanobind_interface):
    """A scalar callback is invoked from the GIL-released kernel and its result lands in the output."""
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


def test_nanobind_interface_bool_scalar_binds_via_caster():
    """A bool scalar arg binds through the emitted dace_bool caster: nanobind's
    own bool caster accepts only an exact Python bool, and the integer-caster
    detour (uint8_t) died with nanobind 2.14 (integer conversion narrowed to
    __index__) x numpy 2.5 (numpy.bool_ lost __index__). The caster is emitted
    only when a bool scalar is actually bound."""
    from dace.codegen.nanobind_bindings import generate_bindings_code

    sdfg = dace.SDFG('bool_scalar_bind_probe')
    sdfg.add_scalar('flag', dace.bool)
    sdfg.add_array('__return', [1], dace.bool)
    state = sdfg.add_state()
    r = state.add_read('flag')
    w = state.add_write('__return')
    state.add_edge(r, None, w, None, sdfg.make_array_memlet('flag'))

    code = generate_bindings_code(sdfg)
    # The binding param goes through the custom caster, cast back to bool for
    # the kernel call. The kernel's own extern-C declaration still takes
    # `bool flag` - that is correct - so this asserts the binding param
    # specifically, not the whole TU.
    assert 'dace_bool flag' in code
    assert 'static_cast<bool>(flag.value)' in code
    assert 'struct type_caster<dace_bool>' in code

    # The call() signature itself must not take the parameter as a plain bool
    # (nanobind's bool caster would then reject numpy.bool_ again).
    call_signature = code.split('nb::object call(')[1].split(') {')[0]
    assert 'dace_bool flag' in call_signature
    assert ' bool flag' not in call_signature

    # No bool scalar bound -> no caster emitted.
    plain = dace.SDFG('bool_scalar_free_probe')
    plain.add_array('A', [10], dace.float64)
    assert 'dace_bool' not in generate_bindings_code(plain)


def test_nanobind_interface_bool_symbol_condition(nanobind_interface):
    """A bool SYMBOL (used in an interstate condition) is an init argument: the
    dace_bool binding parameter is passed by its raw name into init_impl and
    the program call, so the type must convert to bool wherever the name is
    used - the explicit cast only covers the program-call argument list."""
    import numpy as np
    sdfg = dace.SDFG('bool_symbol_cond_probe')
    sdfg.add_symbol('cond', dace.bool)
    sdfg.add_array('__return', [1], dace.float64)
    start = sdfg.add_state(is_start_block=True)
    s_true = sdfg.add_state()
    s_false = sdfg.add_state()
    for state, val in ((s_true, 1.0), (s_false, 2.0)):
        t = state.add_tasklet('write', {}, {'out'}, f'out = {val}')
        w = state.add_write('__return')
        state.add_edge(t, 'out', w, None, dace.Memlet('__return[0]'))
    sdfg.add_edge(start, s_true, dace.InterstateEdge(condition='cond'))
    sdfg.add_edge(start, s_false, dace.InterstateEdge(condition='not cond'))

    csdfg = sdfg.compile()
    assert csdfg(cond=np.True_)[0] == 1.0  # numpy.bool_ on the symbol path
    assert csdfg(cond=False)[0] == 2.0


def test_nanobind_interface_bool_scalar_numpy_input(nanobind_interface):
    """A numpy.bool_ scalar argument is accepted end-to-end on the nanobind interface."""
    import numpy as np
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


def test_nanobind_interface_string_argument(nanobind_interface):
    """A ``dtypes.string`` scalar argument marshals a Python ``str`` (and ``None``) into the kernel.

    The kernel reads the first byte of the string, or writes -1 when the pointer
    is null - so passing a ``str`` observes the bytes, and passing ``None``
    observes the null-pointer path (matching the ctypes marshaller).
    """

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


def test_nanobind_interface_optional_array(nanobind_interface):
    """An optional (nullable) array accepts both a real array and ``None`` (a null pointer)."""
    from typing import Optional

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
    call_body = code.split('nb::object call(')[1]
    # 'z' (must-pass) is unwrapped before 'a' (deduced) - plain arglist order
    # would put 'a' first.
    assert "missing argument 'z'" in call_body
    z_extract = call_body.index('const int z = *z__opt;')
    a_deduce = call_body.index('const int a = a__opt.has_value()')
    assert z_extract < a_deduce


def test_nanobind_interface_load_reuses_same_artifact(nanobind_interface):
    """Loading the same artifact path again reuses the module (one module, many handles)."""
    from dace.codegen.compiler import load_nanobind_module

    N = dace.symbol('N')

    @dace.program
    def load_reuse_nanobind(A: dace.float64[N], B: dace.float64[N], alpha: dace.float64):
        B[:] = alpha * A + B

    csdfg = load_reuse_nanobind.to_sdfg().compile()
    module = load_nanobind_module(csdfg.module.__file__, csdfg.sdfg.name, csdfg.sdfg.build_folder)
    assert module is csdfg.module


def test_nanobind_interface_load_distinct_artifact_coexists(nanobind_interface):
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


def test_nanobind_interface_same_name_different_programs_coexist(nanobind_interface):
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


def test_nanobind_interface_safe_call(nanobind_interface):
    """safe_call runs the SDFG in a subprocess: it forwards in/out output, and a
    crash (writing to a null pointer) surfaces as an exception instead of killing
    the calling process."""
    import pytest

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


def test_nanobind_interface_safe_call_kwargs(nanobind_interface):
    """safe_call accepts the keyword-argument call form."""

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


def test_nanobind_interface_safe_call_return_rejected(nanobind_interface):
    """safe_call does not support return values (parity with the ctypes path)."""
    import pytest

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


def test_nanobind_interface_structure_argument(nanobind_interface):
    """A flat Structure argument is passed as a pointer to a user-built ctypes.Structure."""
    sdfg, csr_obj, _ = _build_csr_to_dense('csr_struct_nanobind', nested=False)
    csdfg = sdfg.compile()

    indptr, indices, data, expected = _csr_example()
    B = np.zeros((2, 3), dtype=np.float32)
    inpA = csr_obj.dtype._typeclass.as_ctypes()(indptr=indptr.__array_interface__['data'][0],
                                                indices=indices.__array_interface__['data'][0],
                                                data=data.__array_interface__['data'][0])
    csdfg(A=inpA, B=B, M=2, N=3, nnz=3)
    assert np.allclose(B, expected)


def test_nanobind_interface_nested_structure_argument(nanobind_interface):
    """A nested Structure argument (Wrapper(csr=...)) works via the same pointer passthrough."""
    import ctypes

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


def test_nanobind_interface_container_array_read(nanobind_interface):
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
    state.add_edge(vcsr, None, indptr, 'views', memlet=dace.Memlet.from_array('vcsr.indptr', csr_obj.members['indptr']))
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


def test_nanobind_interface_complex_array(nanobind_interface):
    """A complex128 array argument compiles and runs (dace::complex128 resolves via the dace type header)."""
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


def test_nanobind_interface_vector_array(nanobind_interface):
    """A vector (veclen) array binds as its base scalar and copies correctly.

    Reproduces the BLAS veclen failures: the ndarray scalar must be the base
    type (float), while the pointer handed to the kernel stays dace::vec<float,2>*.

    Both buffers are wrapped in sentinel padding on either side and only the
    interior is passed, so any access past the intended N vectors is caught as a
    corrupted guard region. The aligned vector type is wider than a plain scalar,
    so this keeps a future size miscalculation from silently over-reading or
    over-writing.
    """
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


def test_nanobind_interface_filename(nanobind_interface):
    """`filename` returns the resolved absolute path to the built .so (parity with CompiledSDFG)."""
    import pathlib

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


def test_nanobind_interface_struct_element_return(nanobind_interface):
    """A return array of a dace.struct (dtypes.struct element) round-trips (argmax-style)."""
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


def test_nanobind_interface_struct_element_input(nanobind_interface):
    """A dtypes.struct-element array passed as an input is byte-view marshalled and copies correctly."""
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


def test_nanobind_interface_single_element_tuple_return(nanobind_interface):
    """A single-element tuple return comes back as a 1-tuple, not a bare array.

    DaCe names a single value ``__return`` but a one-element tuple ``__return_0``,
    so the wrapper must distinguish them (a bare ``len == 1`` check would collapse
    the 1-tuple to the array).
    """
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


def test_nanobind_interface_many_return_values(nanobind_interface):
    """More than ten return values keep their numeric order (not lexicographic `sorted`).

    With `sorted`, `__return_10` would precede `__return_2`, permuting the tuple.
    """

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


def test_nanobind_interface_optional_struct_array_input(nanobind_interface):
    """An optional struct-element array accepts a record array (read by reference) and None (null pointer)."""
    from typing import Optional

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
def test_nanobind_interface_gpu_arrays(nanobind_interface):
    """E2E: CuPy arrays pass directly to GPU-storage parameters; a host numpy
    array for a GPU parameter is rejected at dispatch."""
    cp = pytest.importorskip('cupy')

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
    # The return binds as a defaulted nb::object; the setup casts it (or the
    # in-binding allocation) to a DEVICE ndarray view for the pointer.
    assert 'nb::arg("__return").none() = nb::none()' in code
    assert 'nb::cast<nb::ndarray<double, nb::device::cuda>>(__return__obj, false)' in code


def test_nanobind_interface_gpu_error_record_binding():
    """With a GPU target present, the binding reads the SDFG's OWN error record
    (``__dace_gpu_last_error``, present exactly when the CUDA target emitted its
    init/exit pair) after each program call, instead of consulting the
    process-global CUDA last-error slot - that slot is per-host-thread and
    shared with every other GPU user in the process, so it can carry
    third-party state (mirrors the ctypes interface's mechanism)."""
    from dace.codegen.nanobind_bindings import generate_bindings_code

    sdfg = dace.SDFG('gpu_error_record_probe')
    sdfg.add_array('A', [10], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.arrays['A'].optional = False

    with set_temporary('compiler', 'cuda', 'backend', value='cuda'):
        code = generate_bindings_code(sdfg, gpu_backend='cuda')
    assert f'int __dace_gpu_last_error(' in code
    assert code.count('check_gpu_error();') >= 1, 'the record is never consulted after a program call'
    assert 'cudaGetErrorString' in code
    assert 'def_prop_rw("gpu_error_check"' in code


def test_nanobind_interface_gpu_error_record_absent_on_cpu():
    """A CPU-only module must not reference the GPU error record: the symbol
    only exists when a GPU target emitted its init/exit pair, so an undue
    declaration would be a link error. The ``gpu_error_check`` toggle is
    uniformly present (inert without GPU code)."""
    from dace.codegen.nanobind_bindings import generate_bindings_code

    sdfg = dace.SDFG('cpu_no_gpu_error_probe')
    sdfg.add_array('A', [10], dace.float64)
    code = generate_bindings_code(sdfg)
    assert '__dace_gpu_last_error' not in code
    assert 'GetErrorString' not in code
    assert 'def_prop_rw("gpu_error_check"' in code


def test_nanobind_interface_gpu_error_check_toggle(nanobind_interface):
    """E2E (CPU): the wrapper's ``gpu_error_check`` is backed by the compiled
    handle, so the compiled code - not Python - honors the toggle."""

    @dace.program
    def gpu_check_toggle_probe(A: dace.float64[10]):
        A += 1.0

    sdfg = gpu_check_toggle_probe.to_sdfg()
    csdfg = sdfg.compile()
    assert csdfg.gpu_error_check is True
    assert csdfg._handle.gpu_error_check is True
    csdfg.gpu_error_check = False
    assert csdfg._handle.gpu_error_check is False
    a = np.ones(10)
    csdfg(A=a)
    assert np.allclose(a, 2.0)


def test_nanobind_interface_return_allocation_module_choice():
    """The in-binding allocation imports CuPy only for GPU_Global returns and
    NumPy for host returns - a CPU-only module must never touch cupy (its
    absence then surfaces naturally as an import error at call time)."""
    from dace.codegen.nanobind_bindings import generate_bindings_code

    cpu = dace.SDFG('ret_alloc_cpu_probe')
    cpu.add_array('__return', [10], dace.float64)
    cpu_code = generate_bindings_code(cpu)
    assert 'nb::module_::import_("numpy")' in cpu_code
    assert '"cupy"' not in cpu_code

    gpu = dace.SDFG('ret_alloc_gpu_probe')
    gpu.add_array('__return', [10], dace.float64, storage=dace.StorageType.GPU_Global)
    gpu.arrays['__return'].optional = False
    with set_temporary('compiler', 'cuda', 'backend', value='cuda'):
        gpu_code = generate_bindings_code(gpu)
    assert 'nb::module_::import_("cupy")' in gpu_code


@pytest.mark.gpu
def test_nanobind_interface_gpu_return_values(nanobind_interface):
    """E2E: a GPU program's return value comes back as a CuPy array."""
    cp = pytest.importorskip('cupy')

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


@pytest.mark.gpu
def test_nanobind_interface_gpu_has_gpu_code(nanobind_interface):
    """``has_gpu_code`` is True for a GPU SDFG. The property is backed by the
    compiled handle, and only its False case is covered end-to-end (on a
    CPU-only program, in test_nanobind_interface_has_gpu_code)."""
    cp = pytest.importorskip('cupy')
    N = dace.symbol('N')

    @dace.program
    def gpu_has_code_nanobind(A: dace.float64[N] @ dace.StorageType.GPU_Global):
        A += 1.0

    sdfg = gpu_has_code_nanobind.to_sdfg()
    sdfg.apply_gpu_transformations()
    csdfg = sdfg.compile()

    assert csdfg.has_gpu_code is True
    assert csdfg._handle.has_gpu_code is True

    n = 8
    a = cp.ones(n)
    csdfg(A=a, N=np.int32(n))
    assert np.allclose(cp.asnumpy(a), 2.0)


@pytest.mark.gpu
def test_nanobind_interface_gpu_workspace(nanobind_interface):
    """External-memory workspace in DEVICE memory: the workspace buffer is a
    CuPy array, and its device pointer is what reaches the state struct.

    ``set_workspace`` binds the buffer as an unconstrained ``nb::ndarray<>``
    (no device annotation), so this is the test that a device buffer actually
    survives that path - test_nanobind_interface_workspace only covers
    ``CPU_Heap``."""
    cp = pytest.importorskip('cupy')
    N = dace.symbol('N')

    @dace.program
    def extmem_gpu_nanobind(a: dace.float64[N] @ dace.StorageType.GPU_Global):
        workspace = dace.ndarray([N],
                                 dace.float64,
                                 storage=dace.StorageType.GPU_Global,
                                 lifetime=dace.AllocationLifetime.External)
        workspace[:] = a
        workspace += 1
        a[:] = workspace

    sdfg = extmem_gpu_nanobind.to_sdfg()
    sdfg.apply_gpu_transformations()
    csdfg = sdfg.compile()

    n = 20
    a = cp.random.rand(n)
    csdfg.initialize(a, N=np.int32(n))
    sizes = csdfg.get_workspace_sizes(N=np.int32(n))
    assert sizes == {dace.StorageType.GPU_Global: n * 8}

    wsp = cp.zeros(n, dtype=cp.float64)
    csdfg.set_workspace(dace.StorageType.GPU_Global, wsp, a=a, N=np.int32(n))

    ref = cp.asnumpy(a) + 1
    csdfg(a=a, N=np.int32(n))
    assert np.allclose(cp.asnumpy(a), ref)
    assert np.allclose(cp.asnumpy(wsp), ref)  # the caller's device buffer was the workspace

    # The external workspace pointer is a named field of the state struct.
    fields = csdfg.state_fields()
    assert any('workspace' in f for f in fields)


@pytest.mark.gpu
def test_nanobind_interface_gpu_callback(nanobind_interface):
    """A callback that receives a DEVICE array, on the nanobind interface.

    The callback's array is rebuilt from the raw pointer by
    ``make_reference_from_descriptor``, which chooses CuPy over NumPy from the
    descriptor's storage - so the callback must see a ``cupy.ndarray`` aliasing
    device memory, and a write through it must be visible to the caller.

    Note this currently pins behaviour SHARED with ctypes:
    ``NanobindCompiledSDFG._process_callbacks`` delegates to the same
    ``cbtype.get_trampoline`` the ctypes interface uses. It is here as the
    regression guard for moving that processing into the nanobind binding,
    where the device-array reconstruction would become nanobind's own.

    Adapted from
    ``tests/python_frontend/callback_autodetect_test.py::test_gpu_callback``.
    """
    cp = pytest.importorskip('cupy')

    seen = []

    def cb_with_gpu(arr):
        seen.append(arr)
        arr *= 2

    # ``A`` stays unannotated on purpose: the frontend takes its descriptor
    # (and so GPU_Global storage) from the CuPy argument, which is what makes
    # the callback's own array descriptor device-resident too.
    @dace.program
    def gpu_callback_nanobind(A):
        tmp = dace.ndarray([20], dace.float64, storage=dace.StorageType.GPU_Global)
        tmp[:] = A
        cb_with_gpu(tmp)
        A[:] = tmp

    a = cp.random.rand(20)
    expected = cp.asnumpy(a) * 2
    with pytest.warns(UserWarning, match='Automatically creating callback'):
        gpu_callback_nanobind(a)

    assert np.allclose(cp.asnumpy(a), expected)  # the in-place write reached the caller
    assert len(seen) == 1
    assert isinstance(seen[0], cp.ndarray)  # device memory, not a host copy


def test_nanobind_interface_gpu_error_check(monkeypatch):
    """The GPU error check lives in the compiled binding (it reads the SDFG's
    own error record there, mirroring the ctypes mechanism): its exception
    propagates unchanged through the wrapper, and the wrapper NEVER consults
    the process-global GPU runtime slot - that slot is shared with every other
    GPU user in the process and can carry third-party state."""
    import types
    from dace.codegen import common
    from dace.codegen.nanobind_compiled_sdfg import NanobindCompiledSDFG

    sdfg = dace.SDFG('gpu_error_probe')
    sdfg.add_array('A', [10], dace.float64, storage=dace.StorageType.GPU_Global)

    calls = []

    class FakeHandle:
        has_gpu_code = True
        return_names = ()
        callback_names = ()
        pending_error = None

        def __call__(self, *args, **kwargs):
            calls.append(kwargs)
            if self.pending_error is not None:
                # What the compiled check_gpu_error() throw surfaces as.
                raise RuntimeError(f'An error was detected when calling "gpu_error_probe": {self.pending_error}')

    handle = FakeHandle()
    stub_module = types.SimpleNamespace(make_compiled_sdfg=lambda: handle, __file__='<stub>')
    csdfg = NanobindCompiledSDFG(sdfg, stub_module, ['A'])

    # The wrapper must never reach for the process-global slot on any path.
    def forbidden_runtime():
        raise AssertionError('the wrapper consulted the process-global GPU runtime slot')

    monkeypatch.setattr(common, 'get_gpu_runtime', forbidden_runtime)

    # No recorded error: the call goes through, without touching the runtime.
    csdfg(A=object())
    assert len(calls) == 1

    # A recorded error raises from the binding and propagates unchanged.
    handle.pending_error = 'illegal memory access'
    with pytest.raises(RuntimeError, match='illegal memory access'):
        csdfg(A=object())


def test_nanobind_interface_gpu_error_check_disabled():
    """The gpu_error_check toggle (replacing ctypes' fast_call(do_gpu_check))
    is honored by the compiled binding, so the wrapper's only job is to keep
    the handle's property in sync: the constructor argument and the property
    setter must both write through."""
    import types
    from dace.codegen.nanobind_compiled_sdfg import NanobindCompiledSDFG

    sdfg = dace.SDFG('gpu_error_disable_probe')
    sdfg.add_array('A', [10], dace.float64, storage=dace.StorageType.GPU_Global)

    class FakeHandle:
        has_gpu_code = True
        return_names = ()
        callback_names = ()
        gpu_error_check = None  # written by the wrapper

        def __call__(self, *args, **kwargs):
            pass

    handle = FakeHandle()
    stub_module = types.SimpleNamespace(make_compiled_sdfg=lambda: handle, __file__='<stub>')

    # Default (constructor arg True) reaches the handle.
    csdfg = NanobindCompiledSDFG(sdfg, stub_module, ['A'])
    assert handle.gpu_error_check is True
    assert csdfg.gpu_error_check is True

    # The property writes through and reads back from the handle.
    csdfg.gpu_error_check = False
    assert handle.gpu_error_check is False
    assert csdfg.gpu_error_check is False

    # The constructor argument reaches the handle too.
    handle2 = FakeHandle()
    stub_module2 = types.SimpleNamespace(make_compiled_sdfg=lambda: handle2, __file__='<stub>')
    csdfg2 = NanobindCompiledSDFG(sdfg, stub_module2, ['A'], gpu_error_check=False)
    assert handle2.gpu_error_check is False
    assert csdfg2.gpu_error_check is False


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
def test_nanobind_interface_float16_end_to_end(nanobind_interface):
    """E2E: a float16[N] in / float16[N] out program round-trips through the
    real compiled nanobind module, and a passed float16 array is by-reference."""
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


def test_nanobind_interface_float16_return_value(nanobind_interface):
    """E2E: a float16 return array comes back as a numpy float16 array."""
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


def test_nanobind_interface_strict_scalar_cast_runtime(nanobind_interface):
    """Strict off allows a safe widening scalar cast (int -> double); strict on rejects it."""
    import pytest

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
    assert 'N__opt' not in body.split('nb::object call(')[0]


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

    # A pyobject ARRAY is outside the initial scope: like the low-precision arrays it takes
    # its pointer from __array_interface__ through a setup statement, and the fast path has
    # no setup scope. Without this the generator hits `assert not setup` instead of a usable
    # error. A pyobject SCALAR stays eligible - it forwards `.ptr()` inline.
    sdfg = dace.SDFG('uargs_pyobject_array_probe')
    sdfg.add_array('objs', [4], dtypes.pyobject())
    sdfg.add_array('A', [4], dace.float64)
    sdfg.arrays['A'].optional = False
    sdfg.user_args = ['objs', 'A']
    with pytest.raises(ValueError, match="'objs'.*not supported"):
        generate_bindings_code(sdfg)

    sdfg = dace.SDFG('uargs_pyobject_scalar_probe')
    sdfg.add_scalar('obj', dtypes.pyobject())
    sdfg.add_array('A', [4], dace.float64)
    sdfg.arrays['A'].optional = False
    sdfg.user_args = ['obj', 'A']
    generate_bindings_code(sdfg)  # eligible

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


def test_nanobind_interface_user_args_e2e(nanobind_interface):
    """E2E: structured call through user_bind_call, by-reference semantics kept."""
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


def test_nanobind_interface_user_args_position_name_collision(nanobind_interface):
    """An SDFG argument literally named like a synthesized positional
    parameter (arg1, arg2, ...) must not be shadowed by it: the synthesized
    C++ names are mangled away from real argument names (trailing '_')."""
    from dace.codegen.nanobind_bindings import generate_bindings_code

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


def test_nanobind_interface_user_args_pyobject_e2e(nanobind_interface):
    """E2E: a pyobject rides through user_bind_call in a nested position
    without disturbing its neighbors."""
    from dace import dtypes

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


def test_nanobind_interface_user_args_ignore_slots_e2e(nanobind_interface):
    """E2E: ignored slots swallow arbitrary values (None, dicts) while the real
    entries around them keep working; the tuple length check still counts them."""
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


def test_nanobind_interface_user_args_nested_e2e(nanobind_interface):
    """E2E: nested tuples destructure (the idea.md example shape)."""
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


def test_nanobind_interface_user_bind_call_requires_user_args(nanobind_interface):
    """user_bind_call on a module compiled without user_args raises clearly."""
    csdfg = _uargs_axpy_sdfg().compile()
    with pytest.raises(ValueError, match='user_args'):
        csdfg.user_bind_call((np.zeros(4), np.zeros(4)), 1.0)


def test_nanobind_interface_user_bind_call_gpu_error_check(monkeypatch):
    """user_bind_call keeps the GPU error-record check: it runs inside the
    compiled user_call (gated there by the gpu_error_check toggle), so its
    exception propagates through the wrapper, which itself never consults the
    process-global GPU runtime slot."""
    import types
    from dace.codegen import common
    from dace.codegen.nanobind_compiled_sdfg import NanobindCompiledSDFG

    sdfg = dace.SDFG('uargs_gpu_error_probe')
    sdfg.add_array('A', [10], dace.float64, storage=dace.StorageType.GPU_Global)

    class FakeHandle:
        has_gpu_code = True
        return_names = ()
        callback_names = ()
        gpu_error_check = None  # written by the wrapper
        pending_error = None

        def user_call(self, *args):
            if self.gpu_error_check and self.pending_error is not None:
                raise RuntimeError(f'An error was detected when calling "uargs_gpu_error_probe": '
                                   f'{self.pending_error}')

    handle = FakeHandle()
    stub_module = types.SimpleNamespace(make_compiled_sdfg=lambda: handle, __file__='<stub>')

    def forbidden_runtime():
        raise AssertionError('the wrapper consulted the process-global GPU runtime slot')

    monkeypatch.setattr(common, 'get_gpu_runtime', forbidden_runtime)

    csdfg = NanobindCompiledSDFG(sdfg, stub_module, ['A'])
    handle.pending_error = 'illegal memory access'
    with pytest.raises(RuntimeError, match='illegal memory access'):
        csdfg.user_bind_call((object(), ))

    # The toggle reaches the handle, where the compiled check honors it.
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
