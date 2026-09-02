# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Element-type coverage for the ``dace::reduce`` runtime and the ``Scan`` CPU lowering.

THE SUPPORTED SET, which this file is the executable statement of: the builtin arithmetic types,
``complex64`` / ``complex128`` under ``+`` and ``*`` ONLY, and the low-precision floats
``float16`` / ``bfloat16`` / ``float8_e4m3fn`` / ``float8_e5m2`` under every real op. Anything
outside it must be refused where it can be seen -- a named ``static_assert`` in ``reduction.h``, a
``ValueError`` out of the expansion -- and never quietly return a number.

Each reduce cell is run through BOTH entry points, which are different code: ``dace::reduce::<op>``
(an OpenMP ``reduction`` clause, reassociating) for a node that is the outermost parallel work, and
``dace::reduce::seq::<op>`` (a fixed, thread-count independent association) for one re-entered by an
enclosing parallel map -- the ``libnode_is_sequential`` path. Both run at 64 and at 100000 elements off
one compiled SDFG with a symbolic length.

TOLERANCES are the dtype's epsilon times a small factor, and the inputs are built so the fold is
EXACT in every dtype under test (see :func:`_input`), so the tolerance is headroom rather than a
budget being spent. A low-precision reduction has no accuracy to spare otherwise: the accumulator is
the OUTPUT dtype (see :func:`test_reduce_accumulates_in_the_output_dtype`), so an fp16 sum of 100000
ones saturates at 2048 -- correct per the contract, useless against a float64 reference.
"""
import os
import shutil
import subprocess

import numpy as np
import pytest

import dace
from dace import memlet as mm
from dace.libraries.standard.nodes.scan import Scan, ScanOp, INPUT_CONNECTOR_NAME, OUTPUT_CONNECTOR_NAME

N = dace.symbol('N')

#: Relative tolerance per dtype, as a small multiple of that dtype's epsilon.
#: float16   eps = 2**-10 = 9.8e-4  -> 1e-2  (~10 eps)
#: bfloat16  eps = 2**-7  = 7.8e-3  -> 1e-1  (~13 eps)
#: complex64 eps = 2**-23 = 1.2e-7  -> 1e-6  (~8 eps)
#: complex128 eps = 2**-52 = 2.2e-16 -> 1e-12 (headroom for the parallel shape's reassociation)
_RTOL = {'float16': 1e-2, 'bfloat16': 1e-1, 'complex64': 1e-6, 'complex128': 1e-12}

_SIZES = (64, 100000)

_LOWP = (dace.float16, dace.bfloat16)
_COMPLEX = (dace.complex64, dace.complex128)

_INCLUDE = os.path.join(os.path.dirname(os.path.abspath(dace.__file__)), 'runtime', 'include')

#: op -> (WCR lambda, numpy ufunc). ``min`` / ``max`` are real-only; see :data:`_REAL_ONLY`.
_OPS = {
    'sum': ('lambda x, y: x + y', np.add),
    'product': ('lambda x, y: x * y', np.multiply),
    'min': ('lambda x, y: min(x, y)', np.minimum),
    'max': ('lambda x, y: max(x, y)', np.maximum),
}

_REAL_ONLY = ('min', 'max')


def _tol(dtype):
    return _RTOL[dtype.to_string()]


def _identity(op, np_t):
    """Reduction identity, ROUND-TRIPPED through the dtype so the oracle folds the same value.

    ``min`` / ``max`` seed from a bound well outside the data range ([-3, 3] by construction) rather
    than from an infinity, which has no portable literal spelling in the emitted C++.
    """
    if op == 'sum':
        return 0.0
    if op == 'product':
        return 1.0
    rounded = np.asarray(1e4, dtype=np.float64).astype(np_t)
    bound = float(rounded.real) if np.iscomplexobj(rounded) else float(rounded.astype(np.float64))
    return bound * (1.0 if op == 'min' else -1.0)


def _input(op, n, dtype):
    """Inputs whose fold is exactly representable in every dtype under test, at every thread count.

    The base pattern is dyadic (multiples of 0.25) and its period sums to zero, so a running sum
    stays inside +-0.75 whatever prefix a thread is handed. ``sum`` adds 16 unit spikes on top, so
    every partial sum is a multiple of 0.25 below 17 -- 67 quarter-units, inside bfloat16's 8-bit
    significand (256) and so inside every wider one. ``product`` is ones with four exact power-of-two
    factors, ``min`` / ``max`` are the base pattern with one unique extremum each.
    """
    np_t = dtype.as_numpy_dtype()
    base = 0.25 * ((np.arange(n) % 5) - 2)
    if op == 'sum':
        v = base.copy()
        v[::max(n // 16, 1)] += 1.0
    elif op == 'product':
        v = np.ones(n)
        v[1] = v[3] = v[7] = 2.0
        v[5] = 0.25
    else:
        v = base.copy()
        v[n // 3] = -3.0
        v[2 * n // 3] = 3.0
    if dtype in _COMPLEX:
        # A product must stay bounded: one unit-modulus factor, never a per-element imaginary ramp
        # whose modulus would compound to overflow over 100000 elements.
        v = v.astype(np.complex128)
        if op == 'product':
            v[2] = 1j
        else:
            v = v + 0.25j * ((np.arange(n) % 3) - 1)
    return v.astype(np_t)


def _oracle(op, arr, identity):
    """numpy in float64 / complex128 -- the reference the low-precision fold is measured against."""
    wide = arr.astype(np.complex128 if np.iscomplexobj(arr) else np.float64)
    return _OPS[op][1].reduce(wide, initial=identity)


def _parallel_sdfg(dtype, op):
    """Full reduction at the top of the SDFG: the node IS the outermost parallel work, so the
    expansion emits ``::dace::reduce::<op>``."""
    sdfg = dace.SDFG('red_par_%s_%s' % (op, dtype.to_string()))
    sdfg.add_array('A', [N], dtype)
    sdfg.add_array('B', [1], dtype)
    state = sdfg.add_state()
    red = state.add_reduce(_OPS[op][0], None, _identity(op, dtype.as_numpy_dtype()))
    red.implementation = 'OpenMP'
    red.schedule = dace.ScheduleType.CPU_Multicore
    state.add_edge(state.add_read('A'), None, red, '_in', mm.Memlet('A[0:N]'))
    state.add_edge(red, '_out', state.add_write('B'), None, mm.Memlet('B[0]'))
    sdfg.validate()
    return sdfg


def _sequential_sdfg(dtype, op):
    """The same reduction re-entered by an enclosing ``CPU_Multicore`` map, which is what
    ``libnode_is_sequential`` recognises -- the expansion must emit ``::dace::reduce::seq::<op>``."""
    inner = dace.SDFG('red_seq_inner_%s_%s' % (op, dtype.to_string()))
    inner.add_array('ri', [N], dtype)
    inner.add_array('ro', [1], dtype)
    istate = inner.add_state()
    red = istate.add_reduce(_OPS[op][0], None, _identity(op, dtype.as_numpy_dtype()))
    red.implementation = 'OpenMP'
    istate.add_edge(istate.add_read('ri'), None, red, '_in', mm.Memlet('ri[0:N]'))
    istate.add_edge(red, '_out', istate.add_write('ro'), None, mm.Memlet('ro[0]'))

    sdfg = dace.SDFG('red_seq_%s_%s' % (op, dtype.to_string()))
    sdfg.add_array('A', [2, N], dtype)
    sdfg.add_array('B', [2], dtype)
    state = sdfg.add_state()
    entry, exit_ = state.add_map('rows', {'i': '0:2'}, schedule=dace.ScheduleType.CPU_Multicore)
    nested = state.add_nested_sdfg(inner, {'ri': None}, {'ro': None})
    state.add_memlet_path(state.add_read('A'), entry, nested, dst_conn='ri', memlet=mm.Memlet('A[i, 0:N]'))
    state.add_memlet_path(nested, exit_, state.add_write('B'), src_conn='ro', memlet=mm.Memlet('B[i]'))
    sdfg.validate()
    return sdfg


def _emitted_call(sdfg):
    code = '\n'.join(c.clean_code for c in sdfg.generate_code() if c.language == 'cpp')
    return [ln.strip() for ln in code.splitlines() if '::dace::reduce::' in ln]


@pytest.mark.parametrize('dtype', _LOWP + _COMPLEX, ids=lambda d: d.to_string())
@pytest.mark.parametrize('op', list(_OPS))
def test_reduce_parallel_entry_matches_numpy(op, dtype):
    """{sum, product, min, max} x {float16, bfloat16, complex64, complex128}, parallel entry.

    The ordering ops on a complex element type are a REJECTED cell, not a computed one: the
    expansion has to refuse them by name before any C++ exists.
    """
    if op in _REAL_ONLY and dtype in _COMPLEX:
        with pytest.raises(ValueError, match='not defined for complex data type'):
            _parallel_sdfg(dtype, op).generate_code()
        return
    np_t = dtype.as_numpy_dtype()
    sdfg = _parallel_sdfg(dtype, op)
    assert any('::dace::reduce::seq::' not in c for c in _emitted_call(sdfg)), 'expected the parallel entry'
    csdfg = sdfg.compile()
    for n in _SIZES:
        arr = _input(op, n, dtype)
        out = np.zeros(1, np_t)
        csdfg(A=arr, B=out, N=n)
        want = _oracle(op, arr, _identity(op, np_t))
        assert np.allclose(out[0].astype(np.complex128), want, rtol=_tol(dtype),
                           atol=_tol(dtype)), (op, dtype, n, out[0], want)


@pytest.mark.parametrize('dtype', _LOWP + _COMPLEX, ids=lambda d: d.to_string())
@pytest.mark.parametrize('op', list(_OPS))
def test_reduce_sequential_entry_matches_numpy(op, dtype):
    """The same matrix through the ``libnode_is_sequential`` path (one thread, fixed association)."""
    if op in _REAL_ONLY and dtype in _COMPLEX:
        with pytest.raises(ValueError, match='not defined for complex data type'):
            _sequential_sdfg(dtype, op).generate_code()
        return
    np_t = dtype.as_numpy_dtype()
    sdfg = _sequential_sdfg(dtype, op)
    calls = _emitted_call(sdfg)
    assert calls and all('::dace::reduce::seq::' in c for c in calls), calls
    csdfg = sdfg.compile()
    for n in _SIZES:
        arr = np.stack([_input(op, n, dtype), _input(op, n, dtype)[::-1].copy()])
        out = np.zeros(2, np_t)
        csdfg(A=arr, B=out, N=n)
        for row in range(2):
            want = _oracle(op, arr[row], _identity(op, np_t))
            assert np.allclose(out[row].astype(np.complex128), want, rtol=_tol(dtype),
                               atol=_tol(dtype)), (op, dtype, n, row, out[row], want)


_WARNINGS = ['-Wall', '-Wextra', '-Wpedantic', '-Wshadow', '-Wconversion']


@pytest.mark.parametrize('cxx', ['g++', 'clang++'])
def test_reduction_header_instantiates_the_supported_set_warning_free(cxx, tmp_path):
    """Every in-policy op x element-type pair, including the fp8 formats and the mixed-width pairs.

    Only diagnostics pointing INTO ``reduction.h`` count -- the surrounding headers have their own
    ``-Wconversion`` noise, which this test neither owns nor hides.
    """
    if shutil.which(cxx) is None:
        pytest.skip(f'{cxx} not installed')
    real = ['dace::float16', 'dace::bfloat16', 'dace::float8_e4m3fn', 'dace::float8_e5m2', 'float', 'double']
    cplx = ['dace::complex64', 'dace::complex128']
    lines = ['#include <dace/reduction.h>']
    for ns in ('dace::reduce', 'dace::reduce::seq'):
        for op in ('sum', 'product'):
            for t in real + cplx:
                lines.append(f'template {t} {ns}::{op}<{t}, {t}>(const {t} *, long, long, {t});')
        # ``logical_and`` / ``logical_or`` are left to the integer instantiations in
        # ``reduce_runtime_test.py``: ``acc && x`` on any floating-point operand is a float-to-bool
        # conversion clang reports under ``-Wconversion``, which is not a low-precision matter.
        for op in ('min', 'max'):
            for t in real:
                lines.append(f'template {t} {ns}::{op}<{t}, {t}>(const {t} *, long, long, {t});')
        # Mixed accumulator/element: a low-precision or narrow-complex array folded into a wider one,
        # which is how a caller asks for a promoted accumulator.
        for t, u in (('float', 'dace::float16'), ('double', 'dace::bfloat16'), ('float', 'dace::float8_e4m3fn'),
                     ('dace::complex128', 'dace::complex64')):
            lines.append(f'template {t} {ns}::sum<{t}, {u}>(const {u} *, long, long, {t});')
            lines.append(f'template {t} {ns}::product<{t}, {u}>(const {u} *, long, long, {t});')
    src = tmp_path / 'supported.cpp'
    src.write_text('\n'.join(lines) + '\n')
    proc = subprocess.run(
        [cxx, '-std=c++20', '-fopenmp', '-O2', '-I', _INCLUDE, *_WARNINGS, '-c',
         str(src), '-o', '/dev/null'],
        capture_output=True,
        text=True)
    assert proc.returncode == 0, proc.stderr[:4000]
    offending = [
        ln for ln in proc.stderr.splitlines() if 'reduction.h:' in ln and (': warning' in ln or ': error' in ln)
    ]
    assert not offending, '\n'.join(offending)


#: (instantiation, expected static-assert substring). Each is a supported-set violation the runtime
#: has to name rather than let the op's operator noise speak for it.
_REJECTED = [
    ('template dace::complex64 dace::reduce::min<dace::complex64, dace::complex64>'
     '(const dace::complex64 *, long, long, dace::complex64);', 'need a real element type'),
    ('template dace::complex128 dace::reduce::seq::max<dace::complex128, dace::complex128>'
     '(const dace::complex128 *, long, long, dace::complex128);', 'need a real element type'),
    ('template dace::complex64 dace::reduce::seq::logical_and<dace::complex64, dace::complex64>'
     '(const dace::complex64 *, long, long, dace::complex64);', 'need a real element type'),
    ('template dace::float16 dace::reduce::seq::bitwise_and<dace::float16, dace::float16>'
     '(const dace::float16 *, long, long, dace::float16);', 'need an integral element type'),
    ('template double dace::reduce::bitwise_or<double, double>(const double *, long, long, double);',
     'need an integral element type'),
]


@pytest.mark.parametrize('source,message',
                         _REJECTED,
                         ids=[m.split()[-3] + str(i) for i, (_, m) in enumerate(_REJECTED)])
def test_reduction_header_names_the_rejected_element_types(source, message, tmp_path):
    """Out-of-policy op/dtype pairs fail to compile with the runtime's OWN diagnostic."""
    if shutil.which('g++') is None:
        pytest.skip('g++ not installed')
    src = tmp_path / 'rejected.cpp'
    src.write_text('#include <dace/reduction.h>\n' + source + '\n')
    proc = subprocess.run(['g++', '-std=c++20', '-fopenmp', '-I', _INCLUDE, '-c',
                           str(src), '-o', '/dev/null'],
                          capture_output=True,
                          text=True)
    assert proc.returncode != 0, 'out-of-policy instantiation compiled'
    assert message in proc.stderr, proc.stderr[:2000]


@pytest.mark.parametrize('dtype', _LOWP + _COMPLEX, ids=lambda d: d.to_string())
def test_reduce_accumulates_in_the_output_dtype(dtype):
    """THE CONTRACT: the accumulator is ``seed``'s type, i.e. the OUTPUT dtype -- no hidden promotion.

    A low-precision output therefore stops accumulating once the running sum outruns the significand
    (fp16 ones saturate at 2048, bfloat16 at 256), which is the honest answer for a reduction asked
    to produce that dtype. The promotion a caller may want is spelled in the SDFG, by giving the
    output a wider dtype: that instantiates ``dace::reduce::sum<float, T>`` and is exact. This pins
    both halves so neither can drift into the other.
    """
    n = 100000
    np_t = dtype.as_numpy_dtype()
    ones = np.ones(n, np_t)

    narrow = _parallel_sdfg(dtype, 'sum')
    out = np.zeros(1, np_t)
    narrow(A=ones, B=out, N=n)
    saturated = complex(out[0])
    assert abs(saturated) <= n, 'accumulating in %s cannot exceed the exact sum' % dtype
    if dtype in _LOWP:
        assert abs(saturated) < n, 'a %s accumulator must saturate well below %d' % (dtype, n)

    wide_t = dace.complex128 if dtype in _COMPLEX else dace.float32
    sdfg = dace.SDFG('red_promoted_%s' % dtype.to_string())
    sdfg.add_array('A', [N], dtype)
    sdfg.add_array('B', [1], wide_t)
    state = sdfg.add_state()
    red = state.add_reduce(_OPS['sum'][0], None, 0.0)
    red.implementation = 'OpenMP'
    red.schedule = dace.ScheduleType.CPU_Multicore
    state.add_edge(state.add_read('A'), None, red, '_in', mm.Memlet('A[0:N]'))
    state.add_edge(red, '_out', state.add_write('B'), None, mm.Memlet('B[0]'))
    sdfg.validate()
    wide_out = np.zeros(1, wide_t.as_numpy_dtype())
    sdfg(A=ones, B=wide_out, N=n)
    # float32 holds 100000 exactly (< 2**24), so the promoted fold is the exact sum at any width.
    assert complex(wide_out[0]) == complex(n)


_LIMITS_DRIVER = r'''
#include <dace/types.h>

#include <cstdio>
#include <limits>

template <typename T>
static void show(const char *name) {
  using L = std::numeric_limits<T>;
  printf("%s %d %d %.9g %.9g %.9g %.9g %d\n", name, (int)L::is_specialized, (int)L::has_infinity, (double)(float)L::max(),
         (double)(float)L::lowest(), (double)(float)L::min(), (double)(float)L::epsilon(), L::digits);
}

int main() {
  show<dace::float16>("float16");
  show<dace::bfloat16>("bfloat16");
  show<dace::float8_e5m2>("float8_e5m2");
  show<dace::float8_e4m3fn>("float8_e4m3fn");
  return 0;
}
'''

#: dtype -> (max, lowest, smallest normal, epsilon, significand bits). The IEEE binary16 / bfloat16 /
#: OCP e5m2 / OCP e4m3fn constants; ``e4m3fn`` is the finite format and has no infinity.
_LIMITS = {
    'float16': (65504.0, -65504.0, 6.103515625e-05, 9.765625e-04, 11, 1),
    'bfloat16': (3.38953139e+38, -3.38953139e+38, 1.17549435e-38, 7.8125e-03, 8, 1),
    'float8_e5m2': (57344.0, -57344.0, 6.103515625e-05, 2.5e-01, 3, 1),
    'float8_e4m3fn': (448.0, -448.0, 1.5625e-02, 1.25e-01, 4, 0),
}


def test_numeric_limits_of_the_low_precision_types(tmp_path):
    """``std::numeric_limits`` must be SPECIALIZED for the low-precision structs.

    The primary template answers ``max() == lowest() == infinity() == T()`` for a class type, i.e.
    zero -- and every consumer that derives a min/max identity that way (``dace/scan.hpp``,
    ``libraries/tileops``) then reduces silently into zero instead of failing. This is the guard on
    that: real bounds, correct significand widths, and no infinity for the finite fp8 format.
    """
    if shutil.which('g++') is None:
        pytest.skip('g++ not installed')
    src = tmp_path / 'limits.cpp'
    src.write_text(_LIMITS_DRIVER)
    exe = tmp_path / 'limits'
    subprocess.run(['g++', '-std=c++20', '-O1', '-I', _INCLUDE, str(src), '-o', str(exe)], check=True)
    lines = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout.splitlines()
    got = {p[0]: p[1:] for p in (ln.split() for ln in lines)}
    for name, (mx, lo, mn, eps, digits, has_inf) in _LIMITS.items():
        spec, inf, gmx, glo, gmn, geps, gdig = got[name]
        assert spec == '1', '%s: numeric_limits is not specialized' % name
        assert inf == str(has_inf), name
        assert gdig == str(digits), name
        assert float(gmx) == pytest.approx(mx, rel=1e-7), name
        assert float(glo) == pytest.approx(lo, rel=1e-7), name
        assert float(gmn) == pytest.approx(mn, rel=1e-7), name
        assert float(geps) == pytest.approx(eps, rel=1e-7), name


def _scan_sdfg(dtype, op, implementation, n):
    sdfg = dace.SDFG('scan_%s_%s_%s' % (op.value, implementation, dtype.to_string().replace(':', '_')))
    sdfg.add_array('arr_in', [n], dtype)
    sdfg.add_array('arr_out', [n], dtype)
    state = sdfg.add_state()
    node = Scan('Scan', op=op, exclusive=False)
    node.implementation = implementation
    state.add_node(node)
    state.add_edge(state.add_read('arr_in'), None, node, INPUT_CONNECTOR_NAME, mm.Memlet('arr_in[0:%d]' % n))
    state.add_edge(node, OUTPUT_CONNECTOR_NAME, state.add_write('arr_out'), None, mm.Memlet('arr_out[0:%d]' % n))
    sdfg.validate()
    return sdfg


_SCAN_ORACLE = {
    ScanOp.SUM: np.cumsum,
    ScanOp.PRODUCT: np.cumprod,
    ScanOp.MIN: np.minimum.accumulate,
    ScanOp.MAX: np.maximum.accumulate,
}

#: (op, dtype, implementation) triples the CPU scan lowering supports today. ``CPU`` is the
#: blocked parallel shape, ``pure`` the sequential naked loop.
_SCAN_CELLS = ([(op, dt, 'pure') for op in (ScanOp.SUM, ScanOp.MIN, ScanOp.MAX)
                for dt in _LOWP] + [(ScanOp.SUM, dt, 'pure') for dt in _COMPLEX] + [(ScanOp.PRODUCT, dt, 'pure')
                                                                                    for dt in _COMPLEX] +
               [(ScanOp.SUM, dt, 'CPU') for dt in _LOWP] + [(ScanOp.MIN, dt, 'CPU') for dt in _LOWP])


@pytest.mark.parametrize('op,dtype,implementation',
                         _SCAN_CELLS,
                         ids=lambda v: v.value if isinstance(v, ScanOp) else str(v))
def test_scan_low_precision_and_complex(op, dtype, implementation):
    """Scan over the low-precision and complex element types, both shapes.

    ``MIN`` on the parallel shape is the regression guard for the identity: the blocked scan seeds a
    block from ``std::numeric_limits<T>``, so before those were specialized for the low-precision
    structs this cell returned an array of ZEROS rather than the running minimum.
    """
    n = 64
    np_t = dtype.as_numpy_dtype()
    arr = _input({ScanOp.SUM: 'sum', ScanOp.PRODUCT: 'product'}.get(op, 'min'), n, dtype)
    out = np.zeros(n, np_t)
    _scan_sdfg(dtype, op, implementation, n)(arr_in=arr.copy(), arr_out=out)
    wide = arr.astype(np.complex128 if dtype in _COMPLEX else np.float64)
    want = _SCAN_ORACLE[op](wide)
    assert np.allclose(out.astype(np.complex128), want, rtol=_tol(dtype), atol=_tol(dtype)), (out[:8], want[:8])
