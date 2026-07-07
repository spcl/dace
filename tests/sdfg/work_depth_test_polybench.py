# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Validation of the work-depth analysis on the canonical PolyBench kernels.
"""

import importlib.util
import sys
from contextlib import contextmanager
import pathlib
import pytest
import sympy as sp

from dace.sdfg.performance_evaluation import work_depth
from dace.sdfg.performance_evaluation.helpers import get_uuid
from dace.symbolic import pystr_to_symbolic, simplify

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

EXPECTED = {
    '2mm': ('NI*(3*NJ*NK + 2*NJ*NL + NL)', 2702),
    '3mm': ('2*NJ*(NI*NK + NI*NL + NL*NM)', 4048),
    'adi': ('38*tsteps*(N - 2)**2 + 40', 18432),
    'atax': ('4*M*N', 572),
    'bicg': ('4*M*N', 572),
    'cholesky': ('N**2*(N + 1)/2', 1183),
    'correlation': ('M*(M*N + 8*N + 3)', 2750),
    'covariance': ('M*(M*N + M + 3*N + 2)', 2145),
    'deriche': ('32*H*W', 960), 
    'doitgen': ('2*NP**2*NQ*NR', 1200),
    'durbin': ('2*N**2 + 4*N - 4', 386),
    'fdtd-2d': ('TMAX*(11*NX*NY - 8*NX - 8*NY + 5)', 1503),
    'floyd-warshall': ('N**3', 2197),
    'gemm': ('NI*NJ*(3*NK + 1)', 1568),
    'gemver': ('N*(10*N + 1)', 1703),
    'gesummv': ('N*(4*N + 3)', 715),
    'gramschmidt': ('N*(5*M*N + M + 2)/2', 4732),
    'heat-3d': ('30*tsteps*(N - 2)**3', 159720),
    'jacobi-1d': ('6*tsteps*(N - 2)', 264),
    'jacobi-2d': ('10*tsteps*(N - 2)**2', 4840),
    'lu': ('N**2*(N - 1)', 2028),
    'ludcmp': ('N*(N**2 + 2*N - 2)', 2509), 
    'mvt': ('4*N**2', 676), 
    'nussinov': ('N*(N**2 + 3*N - 4)/6', 442), 
    'seidel-2d': ('9*tsteps*(N - 2)**2', 4356),
    'symm': ('M*N*(5*M + 7)/2', 4433),
    'syr2k': ('N*(6*M + 1)*(N + 1)/2', 6097), 
    'syrk': ('N*(3*M + 1)*(N + 1)/2', 3094),
    'trisolv': ('N*(3*N - 1)/2', 247),
    'trmm': ('M*N*(M + 1)', 1716),
    }


@contextmanager
def _on_path(directory):
    path_str = str(directory)
    added = path_str not in sys.path
    if added:
        sys.path.insert(0, path_str)
    try:
        yield
    finally:
        if added:
            sys.path.remove(path_str)

def _load_kernel(stem: str):
    """Import the PolyBench kernel module by path (the file names are not valid module names) and
    return its ``dace.program``."""
    with _on_path(_POLYBENCH_DIR):
        spec = importlib.util.spec_from_file_location('polybench_' + stem.replace('-', '_'), _POLYBENCH_DIR / f'{stem}.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, _KERNEL_FUNCS[stem])


def _value(expr):
    """Evaluate a symbolic analysis result at :data:`_SIZES`; return ``None`` if it stays symbolic."""
    expr = pystr_to_symbolic(expr)
    subs = {s: _SIZES[s.name] for s in expr.free_symbols if s.name in _SIZES}
    value = simplify(expr.subs(subs).doit())
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _assert_matches(analysis, expected):
    """Check an analysis result against its pinned ``(symbolic, value-at-_SIZES)`` reference: the
    closed forms must be symbolically equal, and the pinned value must be that form evaluated at
    :data:`_SIZES` (so the two side-by-side columns are kept in sync)."""
    symbolic, value = expected
    assert simplify(pystr_to_symbolic(analysis) - pystr_to_symbolic(symbolic)) == 0
    assert _value(symbolic) == value


def _compute_work(sdfg) -> sp.Expr:
    w_d_map = {}
    work_depth.analyze_sdfg(sdfg, w_d_map, work_depth.get_tasklet_work_depth, [], False)
    return w_d_map[get_uuid(sdfg)][0]


@pytest.mark.parametrize('stem', sorted(_KERNEL_FUNCS))
def test_polybench_compute(stem):
    """The compute work of each PolyBench kernel matches its pinned closed form and value."""
    work = _compute_work(_load_kernel(stem).to_sdfg(simplify=True))
    _assert_matches(work, EXPECTED[stem])

if __name__ == '__main__':
    pytest.main([__file__])
