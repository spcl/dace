"""Tests for the ``compiler.tree_reduction`` codegen gate and the perf pipelines
that drive it.

Two things are covered:

1. ``test_tree_reduction_config_gates_codegen`` -- single process: the flag decides
   whether a parallel WCR scalar reduction lowers to an OpenMP ``reduction(op:var)``
   clause (tree reduction) or to a plain atomic WCR. Deterministic; no compiler needed
   (only C++ codegen is inspected).

2. ``test_two_mpi_ranks_get_different_tree_reduction_config`` -- launches 2 MPI ranks
   as 2 processes via ``mpirun -n 2``; rank 0 runs the canonicalize pipeline (tree
   reduction ON) and rank 1 the auto_opt pipeline (OFF). Each rank asserts its own
   process-global ``compiler.tree_reduction`` value, and the parent asserts the two
   ranks ended up with DIFFERENT config -- i.e. the Config is per-process and each
   pipeline sets it correctly with no cross-rank leakage.

Run the whole file under pytest; when invoked as ``python3 test_tree_reduction_config.py
--worker`` it instead runs the single-rank body used by the MPI test.
"""
import os

# MPI anti-hang defaults must be set before dace is imported anywhere (mirrors engine.py
# and every entry-point script); the mpirun workers inherit these too.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MPI4PY_RC_INITIALIZE', '0')
os.environ.setdefault('OMPI_MCA_pml', 'ob1')
os.environ.setdefault('OMPI_MCA_btl', 'self,vader')
os.environ.setdefault('UCX_VFS_ENABLE', 'n')

import copy
import shutil
import subprocess
import sys

import pytest

import engine

_HERE = os.path.dirname(os.path.abspath(__file__))


def _reduction_sdfg():
    """A raw parallel scalar reduction ``s = sum(a)`` (before any perf pipeline)."""
    import dace

    @dace.program
    def sumred(a: dace.float64[256], s: dace.float64[1]):
        for i in dace.map[0:256]:
            s[0] += a[i]

    return sumred.to_sdfg(simplify=True)


def _cpu_code(sdfg):
    """Generated C++ for `sdfg` (all targets concatenated). Generation only -- no
    compile -- so a fresh deepcopy is generated each time to avoid codegen mutating a
    shared SDFG."""
    from dace.codegen import codegen
    return '\n'.join(co.clean_code for co in codegen.generate_code(copy.deepcopy(sdfg)))


def _has_omp_reduction_clause(code):
    return 'reduction(' in code


def test_tree_reduction_config_gates_codegen():
    """Same SDFG, flag flipped -> tree reduction present iff the flag is on.

    The naive ``s[0] += a[i]`` accumulates into the *external* argument ``s`` (never
    eligible for a ``reduction()`` clause -- only a LOCAL scalar can be privatized), so
    we canonicalize first to get the eligible local-accumulator shape; then the ONLY
    difference between the two generated variants is the flag. (pipeline_canon turns the
    flag on as a side effect; we override it explicitly below.)"""
    import dace
    sdfg = engine.pipeline_canon(_reduction_sdfg(), 'cpu')

    dace.Config.set('compiler', 'tree_reduction', value=True)
    on = _cpu_code(sdfg)
    dace.Config.set('compiler', 'tree_reduction', value=False)
    off = _cpu_code(sdfg)

    assert _has_omp_reduction_clause(on), 'tree_reduction ON should emit an OpenMP reduction() clause'
    assert not _has_omp_reduction_clause(off), 'tree_reduction OFF must not emit a reduction() clause'
    # OFF still has to resolve the conflict -- just via the (contended) atomic path.
    assert 'reduce_atomic' in off, 'tree_reduction OFF should fall back to the atomic WCR path'


# --------------------------------------------------------------------------
# MPI worker: rank 0 == canon (tree ON), rank 1 == auto_opt (tree OFF). Prints one
# machine-parsable line; the parent test (below) launches this under mpirun -n 2.
# --------------------------------------------------------------------------
_EXPECTED = {0: ('canon', True), 1: ('auto_opt', False)}


def _worker_main():
    import dace
    rank = engine.get_world_rank()
    pipeline, expect = _EXPECTED[rank % 2]

    sdfg = _reduction_sdfg()
    engine.PIPELINES[pipeline](sdfg, 'cpu')  # sets compiler.tree_reduction for this process
    got = dace.Config.get_bool('compiler', 'tree_reduction')

    ok = (got == expect)
    # A parse-friendly, single line per rank.
    print(f'TREE_REDUCTION_RANK rank={rank} pipeline={pipeline} tree_reduction={got} ok={ok}', flush=True)
    return 0 if ok else 1


@pytest.mark.mpi
def test_two_mpi_ranks_get_different_tree_reduction_config():
    """Two real MPI ranks (2 processes) end up with DIFFERENT tree_reduction config:
    the canon rank ON, the auto_opt rank OFF -- proving the Config is per-process and
    each pipeline sets it correctly."""
    mpirun = shutil.which('mpirun') or shutil.which('mpiexec')
    if mpirun is None:
        pytest.skip('no mpirun/mpiexec on PATH')

    env = dict(os.environ)
    env.setdefault('PYTHONUNBUFFERED', '1')
    cmd = [mpirun, '-n', '2', '--oversubscribe', sys.executable, os.path.abspath(__file__), '--worker']
    proc = subprocess.run(cmd, cwd=_HERE, env=env, capture_output=True, text=True, timeout=600)

    lines = [l for l in proc.stdout.splitlines() if l.startswith('TREE_REDUCTION_RANK')]
    assert proc.returncode == 0, f'workers failed (rc={proc.returncode}).\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}'
    assert len(lines) == 2, f'expected 2 rank lines, got {len(lines)}:\n{proc.stdout}\n{proc.stderr}'

    by_rank = {}
    for line in lines:
        fields = dict(tok.split('=', 1) for tok in line.split()[1:])
        by_rank[int(fields['rank'])] = fields['tree_reduction'] == 'True'

    assert set(by_rank) == {0, 1}, f'expected ranks 0 and 1, got {sorted(by_rank)}'
    assert by_rank[0] is True, 'canon rank (0) must have tree_reduction ON'
    assert by_rank[1] is False, 'auto_opt rank (1) must have tree_reduction OFF'
    assert by_rank[0] != by_rank[1], 'the two ranks must hold different tree_reduction config'


if __name__ == '__main__':
    if '--worker' in sys.argv:
        sys.exit(_worker_main())
    sys.exit(pytest.main([os.path.abspath(__file__), '-v']))
