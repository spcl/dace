# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Validation of the work-depth (compute) and total-volume (memory) analyses on the canonical
PolyBench kernels, reusing the kernels under ``tests/polybench`` rather than redefining them. The
two analyses together approximate the *best-case operational complexity* of each kernel -- the flops
needed and the minimum global-memory traffic assuming on-chip reuse within a nest -- as opposed to a
cache-size-parametric I/O analysis (cf. IOLB, a future extension).

Where the numbers come from:

* **Compute (flops):** :func:`work_depth.analyze_sdfg`, the actual arithmetic-operation count of the
  generated SDFG. This is intentionally the realised count, not a textbook formula: e.g. ``gemm`` is
  ``3*NI*NJ*NK + NI*NJ`` (the ``alpha`` scaling is not folded), whereas the textbook leading term is
  ``2*NI*NJ*NK``. Published per-kernel flop counts for cross-reference: PolyBench/Python (Abella et
  al., CC 2021, https://inria.hal.science/hal-03153351/document) and the PolyBench/C
  ``polybench_set_program_flops`` values. The leading-order textbook term is noted in :data:`EXPECTED`.
* **Memory (bytes):** :func:`total_volume.analyze_sdfg` with ``optimize=False``. See its module
  docstring for the cost model: each accessed region is counted once per enclosing map nest (reuse
  within a parallel nest) and multiplied by the trip count of every enclosing sequential loop (no
  reuse across loop iterations or non-nested scopes). So stencils scale by ``tsteps`` and the
  triangular solvers pay for re-reads across their sequential loops.

The expected values pin the analysis output at a fixed problem size (:data:`_SIZES`) as a regression
guard. The auto-optimized memory path (``optimize=True``) is exercised separately to confirm it stays
valid (its exact value is implementation-dependent, so it is not pinned).

This file also covers the library-node counters (Cholesky/Inv/Solve), the compute-vs-address
classification of interstate-edge arithmetic, the loop-carried integer-compute case (counting the
work that lives in interstate-edge assignments), and the data-dependent (sparse matrix-vector) known
limitation.
"""

import importlib.util
import pathlib

import numpy as np
import pytest
import sympy as sp

import dace
from dace.sdfg import nodes as nd
from dace.sdfg.performance_evaluation import total_volume, work_depth
from dace.sdfg.performance_evaluation.helpers import get_uuid
from dace.symbolic import pystr_to_symbolic

_POLYBENCH_DIR = pathlib.Path(__file__).resolve().parents[1] / 'polybench'

# Problem size at which the symbolic results are evaluated for comparison.
_SIZES = {
    'N': 13,
    'M': 11,
    'NI': 7,
    'NJ': 8,
    'NK': 9,
    'NL': 10,
    'NM': 12,
    'NP': 5,
    'NQ': 6,
    'NR': 4,
    'NX': 7,
    'NY': 8,
    'TMAX': 3,
    'tsteps': 4,
    'H': 6,
    'W': 5
}

# kernel file stem -> the dace.program defined in it.
_KERNEL_FUNCS = {
    '2mm': 'k2mm',
    '3mm': 'k3mm',
    'adi': 'adi',
    'atax': 'atax',
    'bicg': 'bicg',
    'cholesky': 'cholesky',
    'correlation': 'correlation',
    'covariance': 'covariance',
    'deriche': 'deriche',
    'doitgen': 'doitgen',
    'durbin': 'durbin',
    'fdtd-2d': 'fdtd2d',
    'floyd-warshall': 'floyd_warshall',
    'gemm': 'gemm',
    'gemver': 'gemver',
    'gesummv': 'gesummv',
    'gramschmidt': 'gramschmidt',
    'heat-3d': 'heat3d',
    'jacobi-1d': 'jacobi1d',
    'jacobi-2d': 'jacobi2d',
    'lu': 'lu',
    'ludcmp': 'ludcmp',
    'mvt': 'mvt',
    'nussinov': 'nussinov',
    'seidel-2d': 'seidel2d',
    'symm': 'symm',
    'syr2k': 'syr2k',
    'syrk': 'syrk',
    'trisolv': 'trisolv',
    'trmm': 'trmm'
}

# kernel -> (compute work, read bytes, write bytes) evaluated at _SIZES.
# Leading-order references: gemm ~2*NI*NJ*NK, cholesky ~N**3/3, lu/ludcmp ~2*N**3/3,
# floyd-warshall ~N**3, the stencils ~tsteps*flops_per_point*interior, the BLAS-2 kernels ~N**2.
# The triangular-access kernels (cholesky/lu/ludcmp) now produce a closed-form volume too.
EXPECTED = {
    '2mm': (2702, 2744, 2016),
    '3mm': (4048, 3896, 1648),
    'adi': (18432, 139264, 53904),
    'atax': (572, 3520, 1336),
    'bicg': (572, 1336, 280),
    'cholesky': (1183, 12272, 1456),
    'correlation': (2750, 5816, 4064),
    'covariance': (2145, 17864, 4488),
    'deriche': (960, 2640, 2708),
    'doitgen': (1200, 6920, 1920),
    'durbin': (386, 3176, 1752),
    'fdtd-2d': (1503, 8376, 3528),
    'floyd-warshall': (2197, 676, 676),
    'gemm': (1568, 1544, 896),
    'gemver': (1703, 4904, 1664),
    'gesummv': (715, 3032, 312),
    'gramschmidt': (4732, 31720, 17056),
    'heat-3d': (159720, 140608, 85184),
    'jacobi-1d': (264, 832, 704),
    'jacobi-2d': (4840, 10816, 7744),
    'lu': (2028, 266968, 4160),
    'ludcmp': (2509, 312624, 18720),
    'mvt': (676, 1560, 208),
    'nussinov': (442, 5928, 2080),
    'seidel-2d': (4356, 34848, 3872),
    'symm': (4433, 22720, 8008),
    'syr2k': (6097, 3656, 2704),
    'syrk': (3094, 2512, 2704),
    'trisolv': (247, 1560, 312),
    'trmm': (1716, 15760, 2288),
}


def _load_kernel(stem: str):
    """Import the PolyBench kernel module by path (the file names are not valid module names) and
    return its ``dace.program``."""
    spec = importlib.util.spec_from_file_location('polybench_' + stem.replace('-', '_'), _POLYBENCH_DIR / f'{stem}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, _KERNEL_FUNCS[stem])


def _value(expr):
    """Evaluate a symbolic analysis result at :data:`_SIZES`; return ``None`` if it stays symbolic."""
    expr = pystr_to_symbolic(expr)
    subs = {s: _SIZES[s.name] for s in expr.free_symbols if s.name in _SIZES}
    value = sp.simplify(expr.subs(subs).doit())
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _compute_work(sdfg) -> sp.Expr:
    w_d_map = {}
    work_depth.analyze_sdfg(sdfg, w_d_map, work_depth.get_tasklet_work_depth, [], False)
    return w_d_map[get_uuid(sdfg)][0]


@pytest.mark.parametrize('stem', sorted(_KERNEL_FUNCS))
def test_polybench_compute(stem):
    """The compute work of each PolyBench kernel matches its pinned (leading-order) value."""
    work = _compute_work(_load_kernel(stem).to_sdfg(simplify=True))
    assert _value(work) == EXPECTED[stem][0]


@pytest.mark.parametrize('optimize', [False, True])
@pytest.mark.parametrize('stem', sorted(_KERNEL_FUNCS))
def test_polybench_memory(stem, optimize):
    """The un-optimized memory volume matches the pinned working-set bytes; the optimized form must
    still yield a valid, non-negative volume."""
    read, write = total_volume.analyze_sdfg(_load_kernel(stem).to_sdfg(simplify=True), optimize=optimize)
    if optimize:
        # The auto-optimized form is implementation-dependent (fusion, vectorization tiling), so we
        # only require a valid, non-negative result rather than a pinned value.
        for volume in (read, write):
            value = _value(volume)
            assert value is None or value >= 0
        return
    assert _value(read) == EXPECTED[stem][1]
    assert _value(write) == EXPECTED[stem][2]


# Library-node counters (exercised directly on the linalg solver nodes).

N = dace.symbol('N')


@dace.program
def _cholesky_kernel(A: dace.float64[N, N]):
    return np.linalg.cholesky(A)


@dace.program
def _inv_kernel(A: dace.float64[N, N]):
    return np.linalg.inv(A)


@dace.program
def _solve_kernel(A: dace.float64[N, N], b: dace.float64[N]):
    return np.linalg.solve(A, b)


@pytest.mark.parametrize('program, expected_work', [
    (_cholesky_kernel, N**3 / 3),
    (_inv_kernel, 2 * N**3),
    (_solve_kernel, 2 * N**3 / 3 + 2 * N**2),
])
def test_linalg_library_node_work(program, expected_work):
    """Cholesky/Inv/Solve library nodes report their leading-order factorization flop count."""
    sdfg = program.to_sdfg(simplify=True)
    assert any(isinstance(n, nd.LibraryNode) for n, _ in sdfg.all_nodes_recursive())
    work = pystr_to_symbolic(_compute_work(sdfg))
    # Compare at a concrete size (divisible by 3); the analysis and the reference use distinct N
    # symbol instances, so substitute by name rather than subtracting symbolically.
    work_n = work.subs({s: 12 for s in work.free_symbols if s.name == 'N'})
    expected_n = pystr_to_symbolic(expected_work).subs(
        {s: 12
         for s in pystr_to_symbolic(expected_work).free_symbols if s.name == 'N'})
    assert work_n == expected_n


def _make_unary_func_sdfg(func_name: str) -> dace.SDFG:
    """An SDFG that applies ``func_name`` element-wise over N values (one call per iteration)."""
    sdfg = dace.SDFG(f'{func_name}_kernel')
    sdfg.add_array('x', [N], dace.float64)
    sdfg.add_array('y', [N], dace.float64)
    state = sdfg.add_state()
    entry, exit_node = state.add_map('m', {'i': '0:N'})
    tasklet = state.add_tasklet('t', {'inp'}, {'out'}, f'out = {func_name}(inp)')
    state.add_memlet_path(state.add_read('x'), entry, tasklet, dst_conn='inp', memlet=dace.Memlet('x[i]'))
    state.add_memlet_path(tasklet, exit_node, state.add_write('y'), src_conn='out', memlet=dace.Memlet('y[i]'))
    return sdfg


def test_user_defined_function_flop_cost(monkeypatch):
    """A user can override a function's flop cost: e.g. declaring sin to cost 65 flops instead of 1
    (the costs live in the public work_depth.PYFUNC_TO_ARITHMETICS registry)."""
    sdfg = _make_unary_func_sdfg('sin')
    n = _SIZES['N']
    assert _value(_compute_work(sdfg)) == n  # default: one sin per element, sin = 1 flop
    monkeypatch.setitem(work_depth.PYFUNC_TO_ARITHMETICS, 'sin', 65)
    assert _value(_compute_work(sdfg)) == 65 * n  # user-provided cost: sin = 65 flops


# Compute vs. address classification of interstate-edge arithmetic.


def test_interstate_edge_compute_vs_address():
    """Arithmetic assigned to an addressing-only symbol is excluded from work; arithmetic assigned
    to a value read by a tasklet is counted."""
    sdfg = dace.SDFG('iedge_split')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_array('B', [20], dace.float64)
    s0 = sdfg.add_state('s0')
    s1 = sdfg.add_state('s1')
    # idx (3 ops) is only an index; y (1 op) is read by the tasklet.
    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={'idx': '1 + 2 * 3 + 4', 'y': '5 + 6'}))
    t = s1.add_tasklet('t', {'inp'}, {'out'}, 'out = inp + y')
    s1.add_edge(s1.add_read('A'), None, t, 'inp', dace.Memlet('A[idx]'))
    s1.add_edge(t, 'out', s1.add_write('B'), None, dace.Memlet('B[idx]'))

    assert work_depth.compute_symbols(sdfg) == {'y'}
    # Work = 1 (tasklet '+') + 1 (the compute edge assignment 'y = 5 + 6'); idx's arithmetic is address.
    assert _value(_compute_work(sdfg)) == 2


# Loop-carried integer compute (the work lives in interstate-edge assignments).


@dace.program
def _bitwise_accumulate(data: dace.uint32[N]):
    acc = dace.uint32(0)
    for i in range(N):
        acc = (acc << 1) ^ (data[i] & 255)
    return acc


def test_loop_carried_integer_compute():
    """A pure-integer kernel whose bitwise work lives on interstate-edge assignments still reports
    non-zero, size-dependent work."""
    work = _compute_work(_bitwise_accumulate.to_sdfg(simplify=True))
    assert work != 0
    assert N in pystr_to_symbolic(work).free_symbols


# Data-dependent sparse matrix-vector (known limitation).

H = dace.symbol('H')
W = dace.symbol('W')
NNZ = dace.symbol('NNZ')


@dace.program
def _spmv(rowptr: dace.int32[H + 1], colidx: dace.int32[NNZ], vals: dace.float64[NNZ], x: dace.float64[W],
          y: dace.float64[H]):
    for i in range(H):
        for j in range(rowptr[i], rowptr[i + 1]):
            y[i] += vals[j] * x[colidx[j]]


def test_spmv_data_dependent_known_limitation():
    """For sparse matrix-vector the inner trip count is data-dependent (it depends on ``rowptr``
    indexed by the loop variable), so the analyses cannot produce a closed form in the size symbols
    alone -- the result legitimately depends on a per-iteration execution symbol. We accept and
    verify that the analyses run and return such a symbolic result rather than a wrong constant."""
    from dace import symbolic
    sdfg = _spmv.to_sdfg(simplify=True)
    work = pystr_to_symbolic(_compute_work(sdfg))
    read, _write = total_volume.analyze_sdfg(sdfg, optimize=False)
    read = pystr_to_symbolic(read)
    size_symbols = {'H', 'W', 'NNZ'}

    def depends_on_data_or_iterator(expr) -> bool:
        # Either an array value (``rowptr[i]``, a Subscript) leaks into the result, or a loop
        # iterator (a non-size free symbol) was never summed away.
        return bool(expr.atoms(symbolic.Subscript)) or any(s.name not in size_symbols for s in expr.free_symbols)

    # Work depends on the row-pointer values (rowptr[H] - rowptr[0] = number of non-zeros).
    assert depends_on_data_or_iterator(work)
    # Read volume depends on the loop iterator i (the inner trip count rowptr[i+1] - rowptr[i]).
    assert depends_on_data_or_iterator(read)
    assert any(s.name == 'i' for s in read.free_symbols)


if __name__ == '__main__':
    pytest.main([__file__])
