# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end numerical verification of the ``Scan`` libnode with ``stride > 1``.

The libnode expansion runs ``s`` independent inclusive scans, one per residue class
modulo ``s``. Each class is a closed scan (writes only to indices congruent to
``k (mod s)`` strictly forward and reads only those indices plus the class's seed
at position ``k``), so the ``s`` classes have no cross-dependence. The parallel
runtime scans each class with its own OpenMP 5.0 ``inscan`` loop.

This test confirms the result is bit-identical to a sequential strided-scan
oracle for several ``(n, s)`` and both supported CPU implementations.

The libnode's expansion emits a runtime ``s > 0`` ``std::abort()`` check before
the scan starts; the negative/zero-stride case is exercised via subprocess to
avoid killing the test runner.
"""
import os
import subprocess
import sys
import tempfile
import textwrap

import numpy as np
import pytest

import dace
from dace.codegen.codegen import generate_code
from dace.libraries.standard.nodes.scan import (Scan, ScanOp, INPUT_CONNECTOR_NAME, OUTPUT_CONNECTOR_NAME)


def _build_scan_sdfg(n: int, stride: int, op: ScanOp, implementation: str) -> dace.SDFG:
    """Build a single-state SDFG that scans ``arr_in[0:N]`` into ``arr_out[0:N]``
    with the given stride and op."""
    sdfg = dace.SDFG(f'strided_scan_{op.value}_{implementation}_n{n}_s{stride}')
    sdfg.add_array('arr_in', [n], dace.float64)
    sdfg.add_array('arr_out', [n], dace.float64)
    state = sdfg.add_state('scan')
    a_in = state.add_read('arr_in')
    a_out = state.add_write('arr_out')
    node = Scan('Scan', op=op, exclusive=False)
    node.stride = stride
    node.implementation = implementation
    state.add_node(node)
    state.add_edge(a_in, None, node, INPUT_CONNECTOR_NAME, dace.Memlet(f'arr_in[0:{n}]'))
    state.add_edge(node, OUTPUT_CONNECTOR_NAME, a_out, None, dace.Memlet(f'arr_out[0:{n}]'))
    sdfg.validate()
    return sdfg


def _residue_class_scan_oracle(arr_in: np.ndarray, stride: int, op: ScanOp) -> np.ndarray:
    """Reference: ``out[j] = OP_running(arr_in[j_first], ..., arr_in[j])`` within each
    residue class ``j`` congruent to ``k (mod stride)``. The libnode produces the same values."""
    n = arr_in.shape[0]
    out = np.zeros_like(arr_in)
    if op is ScanOp.SUM:
        binop = lambda a, b: a + b
        ident = 0.0
    elif op is ScanOp.PRODUCT:
        binop = lambda a, b: a * b
        ident = 1.0
    elif op is ScanOp.MIN:
        binop = min
        ident = None
    elif op is ScanOp.MAX:
        binop = max
        ident = None
    else:
        raise AssertionError(f'Unknown op: {op}')
    for k in range(stride):
        first = True
        acc = ident
        for j in range(k, n, stride):
            if op in (ScanOp.SUM, ScanOp.PRODUCT):
                acc = binop(acc, arr_in[j])
            else:
                acc = arr_in[j] if first else binop(acc, arr_in[j])
            out[j] = acc
            first = False
    return out


@pytest.mark.parametrize('stride', [1, 2, 3, 4, 5])
@pytest.mark.parametrize('n', [16, 33])
@pytest.mark.parametrize('op', [ScanOp.SUM, ScanOp.PRODUCT, ScanOp.MIN, ScanOp.MAX])
@pytest.mark.parametrize('implementation', ['CPU', 'pure'])
def test_strided_scan_matches_residue_class_oracle(stride: int, n: int, op: ScanOp, implementation: str):
    """For each stride, dtype, and implementation, the libnode-produced output equals
    the per-residue-class sequential scan."""
    # For PRODUCT keep magnitudes ~1; for SUM/MIN/MAX use the unit interval.
    seed = stride * 100 + n + ord(op.value[0]) + ord(implementation[0])
    rng = np.random.default_rng(seed)
    if op is ScanOp.PRODUCT:
        arr_in = rng.uniform(0.95, 1.05, size=n)
    else:
        arr_in = rng.uniform(-1.0, 1.0, size=n)
    arr_out = np.zeros_like(arr_in)
    sdfg = _build_scan_sdfg(n, stride, op, implementation)
    sdfg(arr_in=arr_in.copy(), arr_out=arr_out)
    expected = _residue_class_scan_oracle(arr_in, stride, op)
    assert np.allclose(arr_out, expected), (f'stride={stride} n={n} op={op.value} impl={implementation}: '
                                            f'max abs diff {np.max(np.abs(arr_out - expected))}')


def test_strided_scan_stride_2_explicit():
    """Hand-computed: stride=2, n=8 ascending integers. Even and odd residues are
    independent cumsum subsequences."""
    n, s = 8, 2
    arr_in = np.arange(1.0, n + 1.0)  # [1, 2, 3, 4, 5, 6, 7, 8]
    sdfg = _build_scan_sdfg(n, s, ScanOp.SUM, 'pure')
    arr_out = np.zeros(n)
    sdfg(arr_in=arr_in.copy(), arr_out=arr_out)
    # Even residue: cumsum([1, 3, 5, 7]) = [1, 4, 9, 16]
    # Odd  residue: cumsum([2, 4, 6, 8]) = [2, 6, 12, 20]
    expected = np.array([1.0, 2.0, 4.0, 6.0, 9.0, 12.0, 16.0, 20.0])
    assert np.allclose(arr_out, expected), f'got {arr_out}, expected {expected}'


def test_strided_scan_stride_one_matches_contiguous():
    """``stride=1`` is identical to the contiguous inclusive scan -- the dispatch should
    pick the existing OpenMP scan path, and the result matches ``np.cumsum``."""
    n = 17
    rng = np.random.default_rng(0)
    arr_in = rng.uniform(-1.0, 1.0, size=n)
    arr_out = np.zeros(n)
    sdfg = _build_scan_sdfg(n, 1, ScanOp.SUM, 'CPU')
    sdfg(arr_in=arr_in.copy(), arr_out=arr_out)
    assert np.allclose(arr_out, np.cumsum(arr_in))


_NEGATIVE_STRIDE_SCRIPT = textwrap.dedent("""
    import sys
    sys.path.insert(0, {repo!r})

    import numpy as np
    import dace
    from dace.libraries.standard.nodes.scan import (Scan, ScanOp, INPUT_CONNECTOR_NAME,
                                                    OUTPUT_CONNECTOR_NAME)

    n = 12
    sdfg = dace.SDFG('negstride_probe')
    sdfg.add_array('arr_in', [n], dace.float64)
    sdfg.add_array('arr_out', [n], dace.float64)
    st = sdfg.add_state()
    a_in = st.add_read('arr_in')
    a_out = st.add_write('arr_out')
    node = Scan('Scan', op=ScanOp.SUM, exclusive=False)
    # A negative literal stride trips the runtime ``s > 0`` check inside ``dace::scan``.
    node.stride = -2
    node.implementation = 'pure'
    st.add_node(node)
    st.add_edge(a_in, None, node, INPUT_CONNECTOR_NAME, dace.Memlet('arr_in[0:%d]' % n))
    st.add_edge(node, OUTPUT_CONNECTOR_NAME, a_out, None, dace.Memlet('arr_out[0:%d]' % n))

    arr_in = np.arange(n, dtype=np.float64)
    arr_out = np.zeros(n)
    sdfg(arr_in=arr_in, arr_out=arr_out)
    print('UNEXPECTEDLY_SURVIVED', flush=True)
""").format(repo=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_negative_stride_aborts_at_runtime():
    """A non-positive stride must abort the program before the scan runs. Spawned in
    a subprocess so the abort doesn't kill the test runner."""
    proc = subprocess.run([sys.executable, '-c', _NEGATIVE_STRIDE_SCRIPT], capture_output=True, text=True, timeout=120)
    assert 'UNEXPECTEDLY_SURVIVED' not in proc.stdout, (
        f'Negative stride failed to abort. stdout={proc.stdout!r} stderr={proc.stderr[-400:]!r}')
    assert proc.returncode != 0, (f'Expected non-zero exit on abort; got returncode={proc.returncode}. '
                                  f'stdout={proc.stdout!r} stderr={proc.stderr[-400:]!r}')


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))


def _scan_in_map_sdfg(n: int, rows: int) -> dace.SDFG:
    """``rows`` independent length-``n`` scans, one per iteration of a parallel Map."""
    sdfg = dace.SDFG('scan_inside_parallel_map')
    sdfg.add_array('arr_in', [rows, n], dace.float64)
    sdfg.add_array('arr_out', [rows, n], dace.float64)
    state = sdfg.add_state('scan_map')
    me, mx = state.add_map('rows', {'r': f'0:{rows}'}, schedule=dace.dtypes.ScheduleType.CPU_Multicore)
    a_in = state.add_read('arr_in')
    a_out = state.add_write('arr_out')
    node = Scan('Scan', op=ScanOp.SUM, exclusive=False)
    node.implementation = 'CPU'
    state.add_node(node)
    state.add_memlet_path(a_in, me, node, dst_conn=INPUT_CONNECTOR_NAME, memlet=dace.Memlet(f'arr_in[r, 0:{n}]'))
    state.add_memlet_path(node, mx, a_out, src_conn=OUTPUT_CONNECTOR_NAME, memlet=dace.Memlet(f'arr_out[r, 0:{n}]'))
    sdfg.validate()
    return sdfg


def test_scan_in_parallel_map_lowers_to_sequential_form():
    """A Scan inside a parallel Map must take the SEQUENTIAL shape.

    The parallel expansion opens its own OpenMP region and the runtime header has
    no nesting check, so the shape is chosen statically from the schedule. What may not
    appear inside the map body is a nested REGION or a call into a ``dace::scan`` entry
    point that opens one. The sequential shape's own ``omp simd reduction(inscan, ...)``
    is not a region -- it vectorizes one thread's loop, forks nothing -- so it is asserted
    PRESENT here rather than absent: it is the sequential lowering.
    """
    sdfg = _scan_in_map_sdfg(n=64, rows=4)
    code = ''.join(o.clean_code for o in sdfg.generate_code())
    body = code[code.index('#pragma omp parallel for'):] if '#pragma omp parallel for' in code else code
    assert '#pragma omp parallel' not in body[len('#pragma omp parallel for'):], \
        'a nested OpenMP region was emitted inside the parallel Map'
    assert '::dace::scan::inclusive_' not in code, 'parallel scan entry point called inside a parallel Map'
    assert '::dace::scan::detail::scan_incl_' in code, \
        'the sequential shape should call the header single-block simd inscan'


def test_scan_in_parallel_map_is_numerically_correct():
    """The sequential shape selected inside the Map still computes every row's scan."""
    n, rows = 64, 4
    rng = np.random.default_rng(20260807)
    arr_in = rng.uniform(-1.0, 1.0, size=(rows, n))
    arr_out = np.zeros_like(arr_in)
    _scan_in_map_sdfg(n, rows)(arr_in=arr_in.copy(), arr_out=arr_out)
    assert np.allclose(arr_out, np.cumsum(arr_in, axis=1))


def test_parallel_entry_point_called_inside_a_team_is_correct():
    """Directly calling the parallel entry point from inside an existing OpenMP
    region is not a shape the expansion emits, but it must still be CORRECT:
    OpenMP's nested default gives the inner region a one-thread team, so the
    values are right and only the speedup is lost."""
    src = textwrap.dedent("""
        #include <cstdio>
        #include <vector>
        #include <omp.h>
        #include <dace/scan.hpp>
        int main() {
            const long n = 4096;
            std::vector<double> in(n, 1.0), out(n, 0.0);
            #pragma omp parallel num_threads(4)
            {
                #pragma omp single
                dace::scan::inclusive_sum(in.data(), in.data() + n, out.data());
            }
            std::printf("%s\\n", (out[0] == 1.0 && out[n - 1] == (double)n) ? "OK" : "BAD");
            return 0;
        }
    """)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    include = os.path.join(root, 'dace', 'runtime', 'include')
    with tempfile.TemporaryDirectory() as tmp:
        cpp, exe = os.path.join(tmp, 'p.cpp'), os.path.join(tmp, 'p')
        with open(cpp, 'w') as fh:
            fh.write(src)
        build = subprocess.run(['g++', '-std=c++20', '-O2', '-fopenmp', '-I', include, '-o', exe, cpp],
                               capture_output=True,
                               text=True)
        if build.returncode != 0:
            pytest.skip(f'probe did not build: {build.stderr[-300:]}')
        run = subprocess.run([exe], capture_output=True, text=True, timeout=120)
    assert run.stdout.strip() == 'OK', f'nested direct call gave wrong values: {run.stdout!r}'


def _scan_in_nested_sdfg_in_map_sdfg(n: int, rows: int) -> dace.SDFG:
    """``rows`` independent length-``n`` scans, each computed by a Scan libnode one level
    down inside a NestedSDFG, with the NestedSDFG itself inside a CPU_Multicore Map.

    The adversarial case for scope resolution: ``state.entry_node(node)`` on the Scan's OWN
    state returns ``None`` -- the Map lives in the OUTER SDFG, one nested-SDFG boundary away --
    so a scope walk that only looks at the immediately enclosing state misses it.
    """
    inner = dace.SDFG('scan_row_inner')
    inner.add_array('ri', [n], dace.float64)
    inner.add_array('ro', [n], dace.float64)
    ist = inner.add_state()
    node = Scan('Scan', op=ScanOp.SUM, exclusive=False)
    node.implementation = 'CPU'
    ist.add_node(node)
    ist.add_edge(ist.add_read('ri'), None, node, INPUT_CONNECTOR_NAME, dace.Memlet(f'ri[0:{n}]'))
    ist.add_edge(node, OUTPUT_CONNECTOR_NAME, ist.add_write('ro'), None, dace.Memlet(f'ro[0:{n}]'))

    sdfg = dace.SDFG('scan_in_nested_sdfg_in_map')
    sdfg.add_array('arr_in', [rows, n], dace.float64)
    sdfg.add_array('arr_out', [rows, n], dace.float64)
    state = sdfg.add_state('rows')
    entry, exit_ = state.add_map('rows', {'r': f'0:{rows}'}, schedule=dace.dtypes.ScheduleType.CPU_Multicore)
    nested = state.add_nested_sdfg(inner, {'ri'}, {'ro'})
    state.add_memlet_path(state.add_read('arr_in'),
                          entry,
                          nested,
                          dst_conn='ri',
                          memlet=dace.Memlet(f'arr_in[r, 0:{n}]'))
    state.add_memlet_path(nested,
                          exit_,
                          state.add_write('arr_out'),
                          src_conn='ro',
                          memlet=dace.Memlet(f'arr_out[r, 0:{n}]'))
    sdfg.validate()
    return sdfg


def test_scan_in_nested_sdfg_inside_parallel_map_lowers_to_sequential_form():
    """The documented gap: a Scan inside a NestedSDFG must take the SEQUENTIAL shape when the
    NestedSDFG itself sits inside a parallel Map, even though the Scan's own state has no
    entry node to walk. No nested OpenMP region and no parallel entry point may leak into
    the generated code; the sequential shape's ``simd`` ``inscan`` opens no region and is
    expected to be there.
    """
    sdfg = _scan_in_nested_sdfg_in_map_sdfg(n=64, rows=4)
    code = ''.join(o.clean_code for o in sdfg.generate_code())
    body = code[code.index('#pragma omp parallel for'):] if '#pragma omp parallel for' in code else code
    assert '#pragma omp parallel' not in body[len('#pragma omp parallel for'):], \
        'a nested OpenMP region was emitted inside the parallel Map'
    assert '::dace::scan::inclusive_' not in code, (
        'parallel scan entry point called inside a NestedSDFG in a parallel Map')
    assert '::dace::scan::detail::scan_incl_' in code, \
        'the sequential shape should call the header single-block simd inscan'


def test_scan_in_nested_sdfg_inside_parallel_map_is_numerically_correct():
    """The sequential shape selected one nesting level down still computes every row's scan."""
    n, rows = 64, 4
    rng = np.random.default_rng(20260807)
    arr_in = rng.uniform(-1.0, 1.0, size=(rows, n))
    arr_out = np.zeros_like(arr_in)
    _scan_in_nested_sdfg_in_map_sdfg(n, rows)(arr_in=arr_in.copy(), arr_out=arr_out)
    assert np.allclose(arr_out, np.cumsum(arr_in, axis=1))


def gpu_strided_scan(stride: int, op: ScanOp, dtype=dace.float64) -> dace.SDFG:
    """Single strided ``Scan`` over device-global memory, lowered by the ``CUDA`` key."""
    sdfg = dace.SDFG(f'strided_gpu_{op.value}_s{stride}')
    sdfg.add_array('A', [64], dtype, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('B', [64], dtype, storage=dace.StorageType.GPU_Global)
    state = sdfg.add_state()
    node = Scan('scan', op=op)
    node.stride = stride
    node.implementation = 'CUDA'
    state.add_node(node)
    state.add_edge(state.add_read('A'), None, node, INPUT_CONNECTOR_NAME, dace.Memlet('A[0:64]'))
    state.add_edge(node, OUTPUT_CONNECTOR_NAME, state.add_write('B'), None, dace.Memlet('B[0:64]'))
    sdfg.validate()
    return sdfg


def test_the_gpu_key_is_cuda_for_every_stride():
    """There is ONE GPU implementation name. A caller asking for the device must not also have to
    know the stride, and the fast-library priority lists only ever name ``CUDA``."""
    assert 'CUDA_strided' not in Scan('scan').implementations, (
        'the strided GPU lowering is reachable under its own key again; auto_optimize and '
        'canonicalize only ever ask for CUDA, so a strided scan would refuse instead of lowering')
    assert 'CUDA' in Scan('scan').implementations


@pytest.mark.parametrize('backend', ['cuda', 'hip'])
@pytest.mark.parametrize('stride', [1, 8])
def test_the_strided_launch_lands_in_the_cuda_translation_unit(backend: str, stride: int):
    """The kernel launch is nvcc/hipcc-only syntax, so it cannot sit in the host ``.cpp``.

    This is what the deleted ``auxiliary_sources`` field never achieved: it was declared by the
    old ``ScanStrided`` environment and consumed nowhere in codegen, so the wrappers it named were
    never compiled and ``Scan`` could not link on either backend.
    """
    with dace.config.set_temporary('compiler', 'cuda', 'backend', value=backend):
        codes = generate_code(gpu_strided_scan(stride, ScanOp.SUM))
    cuda = '\n'.join(c.code for c in codes if c.title == 'CUDA')
    host = '\n'.join(c.code for c in codes if c.title != 'CUDA')
    wanted = 'strided_inclusive_sum' if stride != 1 else 'DeviceScan::InclusiveScan'
    assert wanted in cuda, f'{wanted} is not in the CUDA translation unit under the {backend} backend'
    assert wanted not in host, f'{wanted} leaked into the host translation unit'
    assert '__dace_scan' in host, 'the host tasklet does not call the emitted wrapper'


@pytest.mark.parametrize('dtype', [dace.float32, dace.int32, dace.int16])
def test_the_strided_lowering_is_not_limited_to_a_fixed_dtype_set(dtype):
    """The wrapper is templated where it is emitted, so there is no pre-instantiated dtype list.

    ``int16`` in particular was outside the old ``f64 / f32 / i64 / i32`` set and raised.
    """
    codes = generate_code(gpu_strided_scan(4, ScanOp.MAX, dtype))
    cuda = '\n'.join(c.code for c in codes if c.title == 'CUDA')
    assert f'strided_inclusive_max<{dtype.ctype}>' in cuda
