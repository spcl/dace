# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Equivalence + inspection tests for native (C++ / library-node) tasklet connector
inlining in the experimental (readable) code generator.

Native tasklets (BLAS/cuBLAS gemm, fill, small ternary tasklets) are emitted
by ``ExperimentalCPUCodeGen`` with their connectors inlined at codegen time:

* scalar connectors become direct ``<array>_idx(...)`` accesses, and
* pointer / whole-subset connectors become base-pointer expressions.

Each kernel is generated + run under both ``compiler.cpu.implementation`` =
``legacy`` and ``experimental`` on identical inputs; the outputs must be
bit-exact (the runtime compiles with ``-ffp-contract=off``). Inputs are generated
once and deep-copied per run. The BLAS cases exercise the pointer-vs-scalar
connector decision across three different library lowerings (OpenBLAS external
call, cuBLAS device call, and the pure/naive nested-tasklet expansion).
"""
import copy
import ctypes.util
import re
import os
import shutil

import numpy as np
import pytest

import dace
import dace.libraries.blas as blas
from dace.codegen.exceptions import CompilationError, CompilerConfigurationError
from dace.config import Config
from dace.libraries.standard.nodes.fill import FillLibraryNode
from dace.sdfg import nodes

from tests.codegen.readable.conftest import EXPERIMENTAL, LEGACY

N, M, K = (dace.symbol(s) for s in ('N', 'M', 'K'))


@dace.program
def mm(A: dace.float64[M, K], B: dace.float64[K, N], C: dace.float64[M, N]):
    C[:] = A @ B


@pytest.fixture(autouse=True)
def restore_implementation():
    """``_set_impl`` writes the GLOBAL config, so without this every test in this module would leak its
    selection into the rest of the suite (there is no other fixture restoring it)."""
    old = Config.get('compiler', 'cpu', 'implementation')
    yield
    Config.set('compiler', 'cpu', 'implementation', value=old)


def _set_impl(impl):
    Config.set('compiler', 'cpu', 'implementation', value=impl)


def _join_code(programs):
    return '\n'.join(p.clean_code for p in programs)


# -- SDFG builders ------------------------------------------------------------


def _build_ternary(name):
    """ A CPP scalar-connector tasklet: ``_out = (i < 2) ? 0.0 : _inp;``. """
    sdfg = dace.SDFG(name)
    sdfg.add_array('arr', [8], dace.float64)
    sdfg.add_array('arr2', [8], dace.float64)
    st = sdfg.add_state('main')
    me, mx = st.add_map('m', dict(i='0:8'))
    t = st.add_tasklet('tern', {'_inp'}, {'_out'}, '_out = (i < 2) ? 0.0 : _inp;', language=dace.dtypes.Language.CPP)
    ra = st.add_access('arr')
    wb = st.add_access('arr2')
    st.add_memlet_path(ra, me, t, dst_conn='_inp', memlet=dace.Memlet('arr[i]'))
    st.add_memlet_path(t, mx, wb, src_conn='_out', memlet=dace.Memlet('arr2[i]'))
    sdfg.validate()
    return sdfg


def _build_shadowing_ternary(name, body):
    """``_build_ternary``'s shape with a caller-chosen body, so a body that DECLARES one of the
    names the inlined access text carries (``arr[arr_idx(i)]`` -> ``arr``, ``arr_idx``, ``i``) can be
    checked to keep its connector copy."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('arr', [8], dace.float64)
    sdfg.add_array('arr2', [8], dace.float64)
    st = sdfg.add_state('main')
    me, mx = st.add_map('m', dict(i='0:8'))
    t = st.add_tasklet('tern', {'_inp'}, {'_out'}, body, language=dace.dtypes.Language.CPP)
    st.add_memlet_path(st.add_access('arr'), me, t, dst_conn='_inp', memlet=dace.Memlet('arr[i]'))
    st.add_memlet_path(t, mx, st.add_access('arr2'), src_conn='_out', memlet=dace.Memlet('arr2[i]'))
    sdfg.validate()
    return sdfg


def _build_ptr_scalar(name):
    """ A CPP tasklet with two pointer connectors and one scalar connector. """
    sdfg = dace.SDFG(name)
    sdfg.add_array('src', [4], dace.float64)
    sdfg.add_array('dst', [4], dace.float64)
    sdfg.add_array('s', [1], dace.float64)
    st = sdfg.add_state('main')
    t = st.add_tasklet('cpy', {'_src', '_scal'}, {'_dst'},
                       'for (int _k = 0; _k < 4; _k++) { _dst[_k] = _src[_k] + _scal; }',
                       language=dace.dtypes.Language.CPP)
    rsrc = st.add_access('src')
    rs = st.add_access('s')
    wdst = st.add_access('dst')
    st.add_edge(rsrc, None, t, '_src', dace.Memlet('src[0:4]'))
    st.add_edge(rs, None, t, '_scal', dace.Memlet('s[0]'))
    st.add_edge(t, '_dst', wdst, None, dace.Memlet('dst[0:4]'))
    sdfg.validate()
    return sdfg


def _build_offset_ptr(name):
    """A CPP tasklet whose pointer connectors start at a NONZERO offset and are SUBSCRIPTED.

    The inlined connector is a compound expression (``src + 1``), so the body's ``_src[_k]`` becomes
    ``src + 1[_k]`` unless the base is parenthesized -- which C++ parses as ``src + (1[_k])`` and
    rejects with ``invalid types 'int[int]' for array subscript``. The zero-offset ``_build_ptr_scalar``
    case cannot catch this: its connectors inline to a bare name.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array('src', [5], dace.float64)
    sdfg.add_array('dst', [5], dace.float64)
    st = sdfg.add_state('main')
    t = st.add_tasklet('shift', {'_src'}, {'_dst'},
                       'for (int _k = 0; _k < 4; _k++) { _dst[_k] = _src[_k] * 2.0; }',
                       language=dace.dtypes.Language.CPP)
    st.add_edge(st.add_access('src'), None, t, '_src', dace.Memlet('src[1:5]'))
    st.add_edge(t, '_dst', st.add_access('dst'), None, dace.Memlet('dst[1:5]'))
    sdfg.validate()
    return sdfg


def _build_fill(name):
    """ A ``FillLibraryNode`` expanded to a CPP ``std::fill_n(...)`` tasklet. """
    sdfg = dace.SDFG(name)
    sdfg.add_array('out', [8], dace.float64)
    st = sdfg.add_state('main')
    node = FillLibraryNode('memzero')
    node.implementation = 'CPU'
    st.add_node(node)
    wout = st.add_access('out')
    st.add_edge(node, FillLibraryNode.OUTPUT_CONNECTOR_NAME, wout, None, dace.Memlet('out[0:8]'))
    sdfg.validate()
    sdfg.expand_library_nodes()
    return sdfg


def _build_mixed_cpu_gpu(name):
    """
    Same array element-accessed on the host (a host map) AND inside a GPU kernel,
    so the generated ``A_idx`` index function is referenced from both the ``.cpp``
    (Frame) and ``.cu`` (CUDA) output files.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [8], dace.float64)
    sdfg.add_array('B', [8], dace.float64)
    sdfg.add_array('gA', [8], dace.float64, storage=dace.dtypes.StorageType.GPU_Global, transient=True)
    # Host state: B[i] = A[i] + 1 -> A_idx / B_idx referenced in the .cpp file.
    s0 = sdfg.add_state('host')
    me, mx = s0.add_map('hm', dict(i='0:8'))
    t = s0.add_tasklet('h', {'a'}, {'b'}, 'b = a + 1.0')
    s0.add_memlet_path(s0.add_access('A'), me, t, dst_conn='a', memlet=dace.Memlet('A[i]'))
    s0.add_memlet_path(t, mx, s0.add_access('B'), src_conn='b', memlet=dace.Memlet('B[i]'))
    # Host -> device copy, then a GPU kernel: gA[i] *= 2 -> gA_idx in the .cu file.
    s1 = sdfg.add_state_after(s0, 'h2d')
    s1.add_edge(s1.add_access('A'), None, s1.add_access('gA'), None, dace.Memlet('gA[0:8]'))
    s2 = sdfg.add_state_after(s1, 'dev')
    dme, dmx = s2.add_map('dm', dict(i='0:8'), schedule=dace.dtypes.ScheduleType.GPU_Device)
    dt_ = s2.add_tasklet('d', {'x'}, {'y'}, 'y = x * 2.0')
    s2.add_memlet_path(s2.add_access('gA'), dme, dt_, dst_conn='x', memlet=dace.Memlet('gA[i]'))
    s2.add_memlet_path(dt_, dmx, s2.add_access('gA'), src_conn='y', memlet=dace.Memlet('gA[i]'))
    sdfg.validate()
    return sdfg


def _build_gemm(impl_flag, blas_impl, name, gpu=False):
    """ ``C = A @ B`` expanded with a specific BLAS ``Gemm`` implementation. """
    _set_impl(impl_flag)
    sdfg = mm.to_sdfg(simplify=True)
    sdfg.name = name
    if gpu:
        sdfg.apply_gpu_transformations()
    # MatMul only has the 'specialize' expansion -> a Gemm library node.
    for n, st in list(sdfg.all_nodes_recursive()):
        if isinstance(n, nodes.LibraryNode) and type(n).__name__ == 'MatMul':
            n.expand(st, 'specialize')
    # Pin the concrete BLAS implementation on the resulting Gemm node.
    for n, st in list(sdfg.all_nodes_recursive()):
        if isinstance(n, nodes.LibraryNode) and blas_impl in n.implementations:
            n.implementation = blas_impl
    sdfg.expand_library_nodes()
    return sdfg


def _gemm_equiv(blas_impl, gpu=False):
    """ Build + run ``C = A @ B`` under legacy and experimental; assert bit-exact. """
    a = np.random.rand(6, 5)
    b = np.random.rand(5, 7)
    base_c = np.zeros((6, 7))

    cl = copy.deepcopy(base_c)
    _build_gemm('legacy', blas_impl, 'mm_%s_leg' % blas_impl.lower(), gpu).compile()(A=copy.deepcopy(a),
                                                                                     B=copy.deepcopy(b),
                                                                                     C=cl,
                                                                                     M=6,
                                                                                     K=5,
                                                                                     N=7)
    ce = copy.deepcopy(base_c)
    _build_gemm(EXPERIMENTAL, blas_impl, 'mm_%s_exp' % blas_impl.lower(), gpu).compile()(A=copy.deepcopy(a),
                                                                                         B=copy.deepcopy(b),
                                                                                         C=ce,
                                                                                         M=6,
                                                                                         K=5,
                                                                                         N=7)
    assert np.array_equal(cl, ce), '%s: experimental output differs from legacy' % blas_impl
    return cl


# -- tests --------------------------------------------------------------------


def test_cpp_ternary_scalar_inline():
    # Bit-exact legacy vs experimental.
    base = dict(arr=np.arange(8, dtype=np.float64) + 1.0, arr2=np.zeros(8))
    _set_impl(LEGACY)
    leg = copy.deepcopy(base)
    _build_ternary('tern_legacy').compile()(**leg)
    _set_impl(EXPERIMENTAL)
    exp = copy.deepcopy(base)
    _build_ternary('tern_experimental').compile()(**exp)
    assert np.array_equal(leg['arr2'], exp['arr2'])

    # Scalar connectors inlined via <array>_idx, with no copy-in temporary.
    _set_impl(EXPERIMENTAL)
    code = _build_ternary('tern_codegen').generate_code()[0].clean_code
    assert 'arr[arr_idx(' in code
    assert 'arr2[arr2_idx(' in code
    assert 'double _inp =' not in code


def test_cpp_body_declaration_blocks_inlining():
    """A connector is inlined as raw text into a body this generator does not parse, so a body that
    DECLARES one of the names that text carries must keep the connector copy -- otherwise the
    inlined name binds to the body's local. Only a declaration shadows: reading the map's own ``i``
    (``test_cpp_ternary_scalar_inline``) must still inline."""
    _set_impl(EXPERIMENTAL)

    # Declares the ARRAY name carried by _inp's access text `arr[arr_idx(i)]`; _out's `arr2[...]`
    # names nothing the body declares, so it still inlines.
    sdfg = _build_shadowing_ternary('tern_shadow_arr', 'double arr = 0.5; _out = _inp + arr;')
    code = sdfg.generate_code()[0].clean_code
    assert 'double _inp =' in code, 'connector copy dropped although the body declares `arr`'
    assert 'arr[arr_idx(' not in code
    assert 'arr2[arr2_idx(' in code, '_out names nothing the body declares and must stay inlined'

    # Declares the MAP SYMBOL both access texts carry, so both connectors keep their copy.
    sdfg = _build_shadowing_ternary('tern_shadow_sym', 'int i = 3; _out = _inp + i;')
    code = sdfg.generate_code()[0].clean_code
    assert 'double _inp =' in code and 'double _out;' in code
    assert 'arr[arr_idx(' not in code and 'arr2[arr2_idx(' not in code


def test_cpp_pointer_and_scalar_unit():
    # Inspect the generated experimental body: pointer connectors stay base
    # pointers, the scalar connector becomes an <array>_idx access, no copy-ins.
    _set_impl(EXPERIMENTAL)
    code = _build_ptr_scalar('ptrscalar_codegen').generate_code()[0].clean_code
    bodies = [l for l in code.splitlines() if '_k' in l and 'for' in l]
    assert bodies, 'tasklet body not found in generated code'
    body = bodies[0]
    assert 's[s_idx(' in body, body  # scalar connector inlined
    assert 'src[_k]' in body and 'dst[_k]' in body, body  # pointer connectors -> base pointers
    assert 'src_idx' not in code and 'dst_idx' not in code, 'pointer connector wrongly indexed'
    assert 'double* _src' not in code and 'double *_src' not in code, 'pointer copy-in emitted'
    assert 'double _scal' not in code, 'scalar copy-in emitted'

    # Bit-exact legacy vs experimental.
    src = np.arange(4, dtype=np.float64) + 1.0
    sc = np.array([10.0])
    _set_impl(LEGACY)
    leg_dst = np.zeros(4)
    _build_ptr_scalar('ptrscalar_legacy').compile()(src=copy.deepcopy(src), dst=leg_dst, s=copy.deepcopy(sc))
    _set_impl(EXPERIMENTAL)
    exp_dst = np.zeros(4)
    _build_ptr_scalar('ptrscalar_experimental').compile()(src=copy.deepcopy(src), dst=exp_dst, s=copy.deepcopy(sc))
    assert np.array_equal(leg_dst, exp_dst)


def test_offset_pointer_connector_is_parenthesized_before_subscript():
    """An offset base pointer must reach the body parenthesized, so a body that subscripts the
    connector still binds the base first."""
    _set_impl(EXPERIMENTAL)
    code = _build_offset_ptr('offsetptr_codegen').generate_code()[0].clean_code
    subscripts = [l.strip() for l in code.splitlines() if '_k]' in l and '*' in l]
    assert subscripts, 'no subscripted pointer-connector line emitted'
    line = subscripts[0]
    assert '_src' not in line and '_dst' not in line, 'pointer connector not inlined: %s' % line
    assert '(src + 1)[' in line and '(dst + 1)[' in line, 'offset base not parenthesized: %s' % line

    # Bit-exact legacy vs experimental -- and it has to COMPILE, which is the actual regression.
    base = dict(src=np.arange(5, dtype=np.float64) + 1.0, dst=np.zeros(5, dtype=np.float64))
    _set_impl(LEGACY)
    leg = copy.deepcopy(base)
    _build_offset_ptr('offsetptr_legacy').compile()(**leg)
    _set_impl(EXPERIMENTAL)
    exp = copy.deepcopy(base)
    _build_offset_ptr('offsetptr_experimental').compile()(**exp)
    assert np.array_equal(leg['dst'], exp['dst'])
    assert np.array_equal(exp['dst'], np.array([0.0, 4.0, 6.0, 8.0, 10.0]))


def test_fill_pointer_connector():
    # The fill output pointer connector stays a base pointer. The CPU expansion spells the fill
    # as ``std::fill_n`` (both gcc and clang lower it to a memset when the value allows it).
    _set_impl(EXPERIMENTAL)
    code = _build_fill('fill_codegen').generate_code()[0].clean_code
    calls = [l.strip() for l in code.splitlines() if l.strip().startswith('std::fill_n(')]
    assert calls, 'no fill call emitted'
    call = calls[0]
    assert FillLibraryNode.OUTPUT_CONNECTOR_NAME not in call, 'pointer connector not inlined: %s' % call
    assert '_idx(' not in call, 'pointer connector wrongly turned into single-element index: %s' % call
    assert re.search(r'std::fill_n\(\s*out\b', call), call

    # Bit-exact legacy vs experimental.
    base = dict(out=np.arange(8, dtype=np.float64) + 1.0)
    _set_impl(LEGACY)
    leg = copy.deepcopy(base)
    _build_fill('fill_legacy').compile()(**leg)
    _set_impl(EXPERIMENTAL)
    exp = copy.deepcopy(base)
    _build_fill('fill_experimental').compile()(**exp)
    assert np.array_equal(leg['out'], exp['out'])
    assert np.array_equal(exp['out'], np.zeros(8))


def _defined_index_functions(code):
    """Names of ``<name>_idx`` functions DEFINED (``constexpr`` lines) in ``code``."""
    names = set()
    for line in code.splitlines():
        if 'constexpr' in line:
            names.update(re.findall(r'(\w+_idx)\s*\(', line))
    return names


def _called_index_functions(code):
    """Names of ``<name>_idx`` functions CALLED (non-definition, non-preprocessor lines)."""
    names = set()
    for line in code.splitlines():
        # Skip definition lines and the `#ifndef/#define` guard directives (whose
        # DACE_IDXFN_<name>_idx macro would otherwise read as a spurious call).
        if 'constexpr' in line or line.lstrip().startswith('#'):
            continue
        names.update(re.findall(r'\b(\w+_idx)\b', line))
    return names


def test_index_functions_defined_in_every_file():
    # A mixed CPU+GPU SDFG emits index functions into two separate output files
    # (the host .cpp Frame and the device .cu CUDA file), sharing one codegen
    # instance. The flush is per output stream, so each file carries the
    # definitions it references; a global dedup would emit each definition in
    # only one file, leaving the other with an undefined `<array>_idx`.
    if shutil.which('nvcc') is None:
        pytest.skip('nvcc not available (no CUDA backend for the .cu file)')
    _set_impl(EXPERIMENTAL)
    try:
        programs = _build_mixed_cpu_gpu('idx_mixed').generate_code()
    except (CompilationError, CompilerConfigurationError) as e:
        pytest.skip('mixed CPU+GPU codegen unavailable: %s' % e)

    # Safety invariant: every index function CALLED in a file is DEFINED in it.
    for prog in programs:
        code = prog.clean_code
        missing = _called_index_functions(code) - _defined_index_functions(code)
        assert not missing, 'file %r calls undefined index functions %s' % (prog.title, missing)

    # NOTE: this deliberately does NOT assert that some index function is re-emitted across two files.
    # Device code carries no ``<array>_idx`` helpers at all (the .cu file defines none), so the .cpp and
    # the .cu can never share one and such an assertion is unsatisfiable here rather than a regression
    # guard. The per-file re-emission mechanism is guarded where it is actually observable -- across the
    # per-nest host TUs -- by ``test_split_readable_index_helpers_are_per_tu`` in
    # tests/codegen/split_nsdfg_translation_units_test.py, which asserts every TU that USES ``A_idx``
    # also DEFINES it exactly once.


def test_gemm_pure_naive():
    # Pure/naive lowering expands to nested Python tasklets (no external call).
    if 'pure' not in blas.Gemm.implementations:
        pytest.skip('pure Gemm implementation not registered')
    _gemm_equiv('pure')


def test_gemm_openblas_pointers():
    # OpenBLAS lowering is an external cblas_dgemm(...) call whose array
    # connectors must stay base pointers (not <array>_idx single elements).
    if 'OpenBLAS' not in blas.Gemm.implementations:
        pytest.skip('OpenBLAS Gemm implementation not registered')

    _set_impl(EXPERIMENTAL)
    code = _build_gemm(EXPERIMENTAL, 'OpenBLAS', 'mm_ob_codegen').generate_code()[0].clean_code
    calls = [l.strip() for l in code.splitlines() if 'cblas_dgemm' in l]
    assert calls, 'no cblas_dgemm call emitted'
    call = calls[0]
    assert '_idx(' not in call, 'pointer connector wrongly turned into single-element index: %s' % call
    for conn in ('_a', '_b', '_c'):
        assert not re.search(r'\b%s\b' % conn, call), 'connector %s not inlined into base pointer: %s' % (conn, call)
    assert re.search(r'\bA\b', call) and re.search(r'\bB\b', call) and re.search(r'\bC\b', call), call

    # Link + run bit-exact (requires the OpenBLAS/BLAS runtime).
    if ctypes.util.find_library('openblas') is None and ctypes.util.find_library('blas') is None:
        pytest.skip('OpenBLAS/BLAS runtime library not found for linking')
    try:
        _gemm_equiv('OpenBLAS')
    except (CompilationError, CompilerConfigurationError) as e:
        pytest.skip('OpenBLAS build/link unavailable: %s' % e)


def test_gemm_cublas_gpu():
    # cuBLAS lowering is an external device gemm call inside the GPU path; its
    # array connectors must stay base pointers.
    if shutil.which('nvcc') is None:
        pytest.skip('nvcc not available')
    if 'cuBLAS' not in blas.Gemm.implementations:
        pytest.skip('cuBLAS Gemm implementation not registered')

    _set_impl(EXPERIMENTAL)
    try:
        sdfg = _build_gemm(EXPERIMENTAL, 'cuBLAS', 'mm_cub_codegen', gpu=True)
        code = _join_code(sdfg.generate_code())
    except (CompilationError, CompilerConfigurationError) as e:
        pytest.skip('cuBLAS codegen unavailable: %s' % e)
    match = re.search(r'cublas\w*[Gg]emm\w*\([^;]*\);', code, re.DOTALL)
    assert match, 'no cublas gemm call emitted'
    call = match.group(0)
    assert '_idx(' not in call, 'pointer connector wrongly turned into single-element index: %s' % call
    for conn in ('_a', '_b', '_c'):
        assert not re.search(r'\b%s\b' % conn, call), 'connector %s not inlined into base pointer: %s' % (conn, call)
    # The two offloaders spell a device copy differently -- OffloadToAccelerator names it ``A_gpu``,
    # the transformation that preceded the offloading named it ``gpu_A`` -- and this assertion is about the connector
    # being inlined to a base pointer, not about which one ran. Read the names off the SDFG.
    for operand in ('A', 'B', 'C'):
        copies = [n for n in sdfg.arrays if n != operand and n.replace('_gpu', '').replace('gpu_', '') == operand]
        assert copies, f'no device copy of {operand} in {list(sdfg.arrays)}'
        assert any(re.search(r'\b%s\b' % re.escape(c), call) for c in copies), (operand, copies, call)

    # Compile + run bit-exact (requires a GPU device). The driver being installed is not the same
    # as a device being VISIBLE -- a CPU-only run sets CUDA_VISIBLE_DEVICES empty, and then the
    # build succeeds and the runtime raises on init instead.
    if shutil.which('nvidia-smi') is None:
        pytest.skip('no GPU device (nvidia-smi missing)')
    if os.environ.get('CUDA_VISIBLE_DEVICES', None) == '':
        pytest.skip('no GPU device (CUDA_VISIBLE_DEVICES is empty)')
    try:
        _gemm_equiv('cuBLAS', gpu=True)
    except (CompilationError, CompilerConfigurationError, RuntimeError) as e:
        pytest.skip('cuBLAS build/run unavailable: %s' % e)


if __name__ == '__main__':
    test_cpp_ternary_scalar_inline()
    test_cpp_body_declaration_blocks_inlining()
    test_cpp_pointer_and_scalar_unit()
    test_fill_pointer_connector()
    test_index_functions_defined_in_every_file()
    test_gemm_pure_naive()
    test_gemm_openblas_pointers()
    test_gemm_cublas_gpu()
    print('ok')
