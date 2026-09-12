# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the ``dace::reduce`` runtime facility and the CPU expansion that calls it.

Three gates. (1) ``dace/runtime/include/dace/reduction.h`` compiles warning-free under the
full warning set on both g++ and clang++. (2) Every entry point -- each op, contiguous and
strided, parallel and sequential shape -- matches numpy at 1, 4 and 16 OpenMP threads, with
the NaN behaviour of min/max pinned. (3) The ``OpenMP`` expansion of the ``Reduce`` libnode
emits a ``dace::reduce`` call and no per-element atomic.

Floating-point results are compared at ``1e-12``: the parallel shape reassociates the fold
across threads, which is the sanctioned reordering INSIDE the reduction's combining
operation. Integer results are compared exactly.
"""
import os
import shutil
import subprocess

import numpy as np
import pytest

import dace
from dace import memlet as mm
from dace.codegen.codegen import inline_host_nested_sdfgs
from dace.libraries.standard.nodes.reduce import Reduce
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.loop_to_reduce import LoopToReduce

N = dace.symbol("N")

#: Sanctioned association tolerance (see module docstring).
_TOL = 1e-12

#: Elements in the generated test data. Large enough that the OpenMP team really splits it.
_NELEM = 4096

_WARNINGS = ['-Wall', '-Wextra', '-Wpedantic', '-Wshadow', '-Wconversion']

_HEADER = os.path.join(os.path.dirname(os.path.abspath(dace.__file__)), 'runtime', 'include', 'dace', 'reduction.h')
_INCLUDE = os.path.dirname(os.path.dirname(_HEADER))

#: Double-typed entry points -> (numpy ufunc, seed).
_REAL_OPS = {
    'sum': (np.add, 0.0),
    'product': (np.multiply, 1.0),
    'min': (np.minimum, np.inf),
    'max': (np.maximum, -np.inf),
}

#: Integer-typed entry points -> (numpy ufunc, seed).
_INT_OPS = {
    'bitwise_and': (np.bitwise_and, -1),
    'bitwise_or': (np.bitwise_or, 0),
    'bitwise_xor': (np.bitwise_xor, 0),
    'logical_and': (np.logical_and, 1),
    'logical_or': (np.logical_or, 0),
    'min': (np.minimum, np.iinfo(np.int64).max),
    'max': (np.maximum, np.iinfo(np.int64).min),
}

_DRIVER = r'''
#include <dace/reduction.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const long NELEM = @NELEM@;

template <typename T>
static std::vector<T> load(const char *path) {
  std::vector<T> v(static_cast<size_t>(NELEM));
  FILE *f = fopen(path, "rb");
  if (f == nullptr) std::exit(2);
  if (fread(v.data(), sizeof(T), static_cast<size_t>(NELEM), f) != static_cast<size_t>(NELEM)) std::exit(3);
  fclose(f);
  return v;
}

#define RUN_REAL(OP, SEED)                                                                              \
  for (long s = 1; s <= 3; s += 2) {                                                                    \
    const long n = NELEM / s;                                                                           \
    printf(#OP " par %ld %.17g\n", s, dace::reduce::OP(d.data(), n, s, static_cast<double>(SEED)));      \
    printf(#OP " seq %ld %.17g\n", s, dace::reduce::seq::OP(d.data(), n, s, static_cast<double>(SEED))); \
  }

#define RUN_INT(OP, SEED)                                                                                        \
  for (long s = 1; s <= 3; s += 2) {                                                                             \
    const long n = NELEM / s;                                                                                    \
    printf("i" #OP " par %ld %lld\n", s,                                                                         \
           static_cast<long long>(dace::reduce::OP(v.data(), n, s, static_cast<int64_t>(SEED))));                 \
    printf("i" #OP " seq %ld %lld\n", s,                                                                         \
           static_cast<long long>(dace::reduce::seq::OP(v.data(), n, s, static_cast<int64_t>(SEED))));            \
  }

int main(int argc, char **argv) {
  if (argc < 3) return 2;
  const std::vector<double> d = load<double>(argv[1]);
  const std::vector<int64_t> v = load<int64_t>(argv[2]);

  RUN_REAL(sum, 0.0)
  RUN_REAL(product, 1.0)
  RUN_REAL(min, HUGE_VAL)
  RUN_REAL(max, -HUGE_VAL)
  RUN_INT(bitwise_and, -1)
  RUN_INT(bitwise_or, 0)
  RUN_INT(bitwise_xor, 0)
  RUN_INT(logical_and, 1)
  RUN_INT(logical_or, 0)
  RUN_INT(min, INT64_MAX)
  RUN_INT(max, INT64_MIN)

  // NaN pinning: a NaN among the DATA loses every comparison and is swallowed; a NaN SEED
  // propagates through the sequential fold. An empty range returns the seed untouched.
  std::vector<double> withnan = d;
  withnan[7] = std::nan("");
  const double quiet = std::nan("");
  printf("nandata_min par %.17g\n", dace::reduce::min(withnan.data(), NELEM, 1L, HUGE_VAL));
  printf("nandata_min seq %.17g\n", dace::reduce::seq::min(withnan.data(), NELEM, 1L, HUGE_VAL));
  printf("nandata_max par %.17g\n", dace::reduce::max(withnan.data(), NELEM, 1L, -HUGE_VAL));
  printf("nandata_max seq %.17g\n", dace::reduce::seq::max(withnan.data(), NELEM, 1L, -HUGE_VAL));
  printf("nanseed_min seq %d\n", std::isnan(dace::reduce::seq::min(d.data(), NELEM, 1L, quiet)) ? 1 : 0);
  printf("nanseed_max seq %d\n", std::isnan(dace::reduce::seq::max(d.data(), NELEM, 1L, quiet)) ? 1 : 0);
  printf("empty par %.17g\n", dace::reduce::sum(d.data(), 0L, 1L, 17.5));
  printf("empty seq %.17g\n", dace::reduce::seq::min(d.data(), 0L, 1L, 17.5));

  // Association, the reason the sequential sum is a tree and not a running total. One huge element
  // ahead of many small ones: a left-to-right fold has outgrown every later addend by its second
  // step and returns the huge element unchanged, while a tree adds the small ones to each other
  // first and carries their total up at a magnitude the final addition can still represent.
  std::vector<double> stiff(1 << 16, 1.0);
  stiff[0] = 1e16;
  double running = 0.0;
  for (double x : stiff) running += x;
  printf("stiff running %.17g\n", running);
  printf("stiff seq %.17g\n", dace::reduce::seq::sum(stiff.data(), static_cast<long>(stiff.size()), 1L, 0.0));
  // Same shape strided, so the tree's index walk is exercised on the non-contiguous path too.
  printf("stiff seq3 %.17g\n",
         dace::reduce::seq::sum(stiff.data(), static_cast<long>(stiff.size()) / 3, 3L, 0.0));

  // No entry point carries a nesting check; an external caller already inside a team gets
  // OpenMP's nested default (one thread). Still the RIGHT value, which is what this pins.
  int wrong = 0;
  const double want = dace::reduce::seq::sum(d.data(), NELEM, 1L, 0.0);
#pragma omp parallel num_threads(4) reduction(+ : wrong)
  {
    const double got = dace::reduce::sum(d.data(), NELEM, 1L, 0.0);
    if (!(std::fabs(got - want) <= 1e-9 * std::fabs(want))) wrong = 1;
  }
  printf("inteam wrong %d\n", wrong);
  return 0;
}
'''.replace('@NELEM@', str(_NELEM))


def _cxx(name):
    """Path to a working ``name`` compiler, or ``None``."""
    return shutil.which(name)


def _make_data(tmp_path):
    """Deterministic real and integer inputs, written as raw binaries for the driver.

    The reals sit in ``[0.5, 1.5]`` so a 4096-element product stays far from overflow and
    underflow. The integers carry zeros ONLY at indices ``1 mod 3``, so ``logical_and`` is
    false over the contiguous range and true over the stride-3 one -- the two shapes cannot
    pass by returning the same trivial answer.
    """
    rng = np.random.default_rng(20260807)
    reals = rng.uniform(0.5, 1.5, _NELEM)
    ints = rng.integers(1, 1 << 20, _NELEM, dtype=np.int64)
    ints[1::3] = 0
    real_path = tmp_path / 'reals.bin'
    int_path = tmp_path / 'ints.bin'
    real_path.write_bytes(reals.tobytes())
    int_path.write_bytes(ints.tobytes())
    return reals, ints, str(real_path), str(int_path)


def _build_driver(cxx, tmp_path):
    src = tmp_path / 'reduce_driver.cpp'
    src.write_text(_DRIVER)
    exe = tmp_path / 'reduce_driver'
    subprocess.run([cxx, '-std=c++20', '-fopenmp', '-O2', '-I', _INCLUDE, str(src), '-o', str(exe)], check=True)
    return str(exe)


def _run_driver(exe, real_path, int_path, threads):
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    out = subprocess.run([exe, real_path, int_path], check=True, capture_output=True, text=True, env=env)
    parsed = {}
    for line in out.stdout.splitlines():
        parts = line.split()
        parsed[tuple(parts[:-1])] = float(parts[-1])
    return parsed


@pytest.mark.parametrize('cxx', ['g++', 'clang++'])
def test_reduction_header_compiles_warning_free(cxx, tmp_path):
    """Every ``dace::reduce`` instantiation the expansions can emit, built with warnings on.

    Only diagnostics pointing INTO ``reduction.h`` count -- the surrounding DaCe headers have
    their own ``-Wconversion`` noise, which this test neither owns nor hides.
    """
    if _cxx(cxx) is None:
        pytest.skip(f'{cxx} not installed')
    ordered = [
        'float', 'double', 'int', 'long', 'short', 'signed char', 'unsigned int', 'bool', 'dace::float16',
        'dace::bfloat16'
    ]
    # Complex has ``+`` and ``*`` only -- ``min``/``max`` have no meaning on a complex operand
    # and no OpenMP reduction, so the entry points are deliberately not instantiable for them.
    complexes = ['dace::complex64', 'dace::complex128']
    lines = ['#include <dace/reduction.h>']
    for ns in ('dace::reduce', 'dace::reduce::seq'):
        for op in ('sum', 'product'):
            for t in ordered + complexes:
                lines.append(f'template {t} {ns}::{op}<{t}, {t}>(const {t} *, long, long, {t});')
        for op in ('min', 'max'):
            for t in ordered:
                lines.append(f'template {t} {ns}::{op}<{t}, {t}>(const {t} *, long, long, {t});')
        for op in ('bitwise_and', 'bitwise_or', 'bitwise_xor', 'logical_and', 'logical_or'):
            for t in ('int', 'long', 'short', 'unsigned int', 'bool'):
                lines.append(f'template {t} {ns}::{op}<{t}, {t}>(const {t} *, long, long, {t});')
        # Mixed accumulator/element types: an integer array reduced into a real output.
        lines.append(f'template double {ns}::sum<double, int>(const int *, long, long, double);')
        lines.append(f'template double {ns}::min<double, float>(const float *, long, long, double);')
    src = tmp_path / 'instantiations.cpp'
    src.write_text('\n'.join(lines) + '\n')
    proc = subprocess.run([
        cxx, '-std=c++20', '-fopenmp', '-O2', '-I', _INCLUDE, *_WARNINGS, '-c',
        str(src), '-o',
        str(tmp_path / 'instantiations.o')
    ],
                          capture_output=True,
                          text=True)
    assert proc.returncode == 0, proc.stderr
    offending = [
        ln for ln in proc.stderr.splitlines() if 'reduction.h:' in ln and (': warning' in ln or ': error' in ln)
    ]
    assert not offending, '\n'.join(offending)


@pytest.mark.parametrize('threads', [1, 4, 16])
def test_reduce_entry_points_match_numpy(threads, tmp_path):
    """Each op x {contiguous, stride 3} x {parallel, sequential} against numpy."""
    if _cxx('g++') is None:
        pytest.skip('g++ not installed')
    reals, ints, real_path, int_path = _make_data(tmp_path)
    got = _run_driver(_build_driver('g++', tmp_path), real_path, int_path, threads)

    for stride in (1, 3):
        count = _NELEM // stride
        real_slice = reals[:count * stride:stride]
        int_slice = ints[:count * stride:stride]
        for op, (ufunc, seed) in _REAL_OPS.items():
            want = ufunc.reduce(real_slice, initial=seed)
            for shape in ('par', 'seq'):
                # Sanctioned association tolerance: the parallel shape reassociates the fold.
                assert np.isclose(got[(op, shape, str(stride))], want, rtol=_TOL, atol=_TOL), (op, shape, stride)
        for op, (ufunc, seed) in _INT_OPS.items():
            want = int(ufunc.reduce(int_slice, initial=seed))
            for shape in ('par', 'seq'):
                assert got[('i' + op, shape, str(stride))] == want, (op, shape, stride)
    # A parallel entry called from inside a caller's own team still returns the right value; it has
    # no nesting check, it just gets OpenMP's nested default of one thread.
    assert got[('inteam', 'wrong')] == 0


@pytest.mark.parametrize('threads', [1, 4, 16])
def test_reduce_min_max_nan_and_empty_semantics(threads, tmp_path):
    """The NaN and empty-range behaviour ``dace::reduce`` documents, pinned at every team size."""
    if _cxx('g++') is None:
        pytest.skip('g++ not installed')
    reals, _ints, real_path, int_path = _make_data(tmp_path)
    got = _run_driver(_build_driver('g++', tmp_path), real_path, int_path, threads)

    without_nan = np.delete(reals, 7)
    for shape in ('par', 'seq'):
        # A NaN among the data loses every comparison, so it never reaches the accumulator.
        assert got[('nandata_min', shape)] == pytest.approx(without_nan.min(), rel=_TOL, abs=_TOL)
        assert got[('nandata_max', shape)] == pytest.approx(without_nan.max(), rel=_TOL, abs=_TOL)
    # A NaN seed propagates through the sequential fold (the parallel combiner is the runtime's).
    assert got[('nanseed_min', 'seq')] == 1
    assert got[('nanseed_max', 'seq')] == 1
    # An empty range returns the seed untouched, which is what makes an identity unnecessary.
    assert got[('empty', 'par')] == 17.5
    assert got[('empty', 'seq')] == 17.5


def test_the_sequential_sum_associates_pairwise(tmp_path):
    """The sequential sum is a tree, and the tree is why it still answers on a stiff range.

    A running total is the obvious spelling and the wrong one: once the accumulator is 1e16 the
    next 65535 additions of 1.0 each round back to where they started, and the reduction returns
    the first element. The assertion is on the ERROR, not on a bit pattern -- any association that
    keeps the partials comparable in size passes, a running total cannot.
    """
    if _cxx('g++') is None:
        pytest.skip('g++ not installed')
    _reals, _ints, real_path, int_path = _make_data(tmp_path)
    got = _run_driver(_build_driver('g++', tmp_path), real_path, int_path, 1)

    exact = 1e16 + float((1 << 16) - 1)
    running_error = abs(got[('stiff', 'running')] - exact)
    tree_error = abs(got[('stiff', 'seq')] - exact)
    assert got[('stiff', 'running')] == 1e16, ('the running total was expected to swallow every addend; if it no '
                                               'longer does, this data no longer separates the two associations')
    # What the tree still loses, and why the bound is not zero: 1e16 sits in one of the block's eight
    # chains, and the addends that land in THAT chain -- one in eight of its 128 -- are added to a
    # number too large to move. Everything else is summed among peers and arrives intact.
    assert tree_error <= 16.0, f'sequential sum lost more than the block it had to: got {got[("stiff", "seq")]!r}'
    assert tree_error * 1000 < running_error, (f'tree error {tree_error} is not decisively better than the running '
                                               f'total error {running_error}')
    assert got[('stiff', 'seq')] == pytest.approx(np.sum(np.concatenate([[1e16], np.ones((1 << 16) - 1)])), abs=2.0), (
        'the shape is numpy pairwise; a difference here means the block or the lane count drifted apart from it')

    strided_exact = 1e16 + float((1 << 16) // 3 - 1)
    assert abs(got[('stiff', 'seq3')] - strided_exact) <= 16.0, 'the strided walk did not reach the same answer'


def _sum_loop_sdfg():
    """``for i in range(N): A[0] = A[0] + B[i]`` -- the shape ``LoopToReduce`` lifts."""
    sdfg = dace.SDFG('reduce_call_shape')
    sdfg.add_array('A', [1], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    loop = LoopRegion('loop', condition_expr='i < N', loop_var='i', initialize_expr='i = 0', update_expr='i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state('body', is_start_block=True)
    task = body.add_tasklet('add', {'in_a', 'in_b'}, {'out'}, 'out = in_a + in_b')
    body.add_edge(body.add_read('A'), None, task, 'in_a', mm.Memlet('A[0]'))
    body.add_edge(body.add_read('B'), None, task, 'in_b', mm.Memlet('B[i]'))
    body.add_edge(task, 'out', body.add_write('A'), None, mm.Memlet('A[0]'))
    sdfg.validate()
    return sdfg


def _row_reduce_sdfg(schedule):
    """``B[i] = sum(A[i, :])`` with the Reduce inside a CPU_Multicore map, pinned to ``schedule``."""
    inner = dace.SDFG('row_reduce_inner')
    inner.add_array('ri', [16], dace.float64)
    inner.add_array('ro', [1], dace.float64)
    ist = inner.add_state()
    red = ist.add_reduce('lambda x, y: x + y', None, 0.0)
    ist.add_edge(ist.add_read('ri'), None, red, '_in', mm.Memlet('ri[0:16]'))
    ist.add_edge(red, '_out', ist.add_write('ro'), None, mm.Memlet('ro[0]'))
    sdfg = dace.SDFG('row_reduce')
    sdfg.add_array('A', [8, 16], dace.float64)
    sdfg.add_array('B', [8], dace.float64)
    state = sdfg.add_state()
    entry, exit_ = state.add_map('rows', {'i': '0:8'}, schedule=dace.ScheduleType.CPU_Multicore)
    nested = state.add_nested_sdfg(inner, {'ri'}, {'ro'})
    state.add_memlet_path(state.add_read('A'), entry, nested, dst_conn='ri', memlet=mm.Memlet('A[i, 0:16]'))
    state.add_memlet_path(nested, exit_, state.add_write('B'), src_conn='ro', memlet=mm.Memlet('B[i]'))
    sdfg.validate()
    red.schedule = schedule
    return sdfg, red


@pytest.mark.parametrize('schedule', [dace.ScheduleType.CPU_Multicore, dace.ScheduleType.Sequential])
def test_reduce_inside_parallel_map_never_emits_the_parallel_entry(schedule):
    """SCOPE, not ``node.schedule``, picks the shape -- the entry points have no nesting check.

    ``CPU_Multicore`` is the adversarial case: a library node's schedule is derived from neighbouring
    STORAGE, so a Reduce re-entered by a parallel map really does arrive carrying it.
    """
    sdfg, _red = _row_reduce_sdfg(schedule)
    code = '\n'.join(c.clean_code for c in sdfg.generate_code() if c.language == 'cpp')
    assert '::dace::reduce::sum(' not in code, 'parallel entry emitted inside a parallel map'
    assert 'reduce_atomic' not in code
    if schedule == dace.ScheduleType.CPU_Multicore:
        assert '::dace::reduce::seq::sum(' in code
    a = np.arange(8 * 16, dtype=np.float64).reshape(8, 16)
    b = np.zeros(8)
    sdfg(A=a.copy(), B=b)
    assert np.allclose(b, a.sum(axis=1), rtol=_TOL, atol=_TOL)


def test_sequential_expansion_connectors_survive_inlining_beside_same_named_arrays():
    """The sequential expansion returns a nested SDFG, and codegen inlines it -- its tasklets then
    live in the PARENT SDFG, where a connector sharing a name with an array is invalid. So the
    connectors must be namespaced, not the bare ``a`` / ``b`` / ``o`` a user array really carries
    (npbench ``channel_flow`` reduces ``u`` in an SDFG holding ``b``).
    """
    sdfg = dace.SDFG('reduce_beside_colliding_names')
    sdfg.add_array('a', [_NELEM], dace.float64)
    sdfg.add_array('b', [1], dace.float64)
    sdfg.add_array('o', [1], dace.float64)
    state = sdfg.add_state()
    red = state.add_reduce('lambda x, y: x + y', None, 0.0)
    state.add_edge(state.add_read('a'), None, red, '_in', mm.Memlet(f'a[0:{_NELEM}]'))
    state.add_edge(red, '_out', state.add_write('b'), None, mm.Memlet('b[0]'))
    sdfg.validate()
    red.schedule = dace.ScheduleType.Sequential

    sdfg.expand_library_nodes()
    inline_host_nested_sdfgs(sdfg)
    sdfg.validate()

    tasklets = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet)]
    assert tasklets, 'the sequential expansion emits tasklets; none survived the inline'
    connectors = {c for t in tasklets for c in t.in_connectors.keys() | t.out_connectors.keys()}
    assert not (connectors & set(sdfg.arrays)), f'connectors shadow array names: {connectors & set(sdfg.arrays)}'

    a = np.arange(_NELEM, dtype=np.float64)
    b = np.zeros(1)
    sdfg(a=a.copy(), b=b, o=np.zeros(1))
    assert np.allclose(b, a.sum(), rtol=_TOL, atol=_TOL)


def test_openmp_expansion_calls_dace_reduce():
    """A lifted sum loop lowers to a ``dace::reduce`` call, with no atomic anywhere."""
    sdfg = _sum_loop_sdfg()
    assert LoopToReduce(prefer='reduce-libnode').apply_pass(sdfg, {}) == 1
    reduces = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Reduce)]
    assert len(reduces) == 1
    reduces[0].implementation = 'OpenMP'
    reduces[0].schedule = dace.ScheduleType.CPU_Multicore

    code = '\n'.join(c.clean_code for c in sdfg.generate_code() if c.language == 'cpp')
    assert '::dace::reduce::sum(' in code
    assert 'reduce_atomic' not in code
    # The ``reduction`` clause is the header's, not the tasklet's -- that IS the two-shape
    # design, so assert it where it lives rather than in the generated file.
    header = open(_HEADER).read()
    assert 'reduction(+ : acc)' in header
    # No runtime nesting check exists any more: SCOPE picks the shape statically in the expansion.
    assert 'omp_in_parallel' not in header
