# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end tests for lowering a CONDITIONAL reduction via a tree-reduction.

A guarded accumulator ``if cond(i): acc OP= x`` lowers, by default, to a
per-passing-thread guarded atomic (``reduce_atomic`` on CPU, a per-thread
``atomicAdd`` on GPU). :class:`~dace.transformation.passes.canonicalize.
loop_to_conditional_reduce.LoopToConditionalReduce` folds the guard into the
accumulated value using the reduction identity (``masked = x if cond else
IDENTITY(OP)``) and splits it into a mask tasklet + a plain accumulate, so the
``reduction_to_wcr_map`` canonicalize stage lifts it to the fast tree reduction:
an OpenMP ``reduction(OP:acc)`` clause on CPU, a block/warp reduce on GPU.

These tests pin BOTH halves of the contract: the value is preserved AND the
generated code takes the tree-reduce path (reduction clause present on CPU,
block reduce on GPU) with the guarded atomic gone.
"""
import os

# Steer Open MPI off UCX before dace's lazy ``from mpi4py import MPI`` so
# ``to_sdfg`` does not stall on MPI_Init (mirrors the corpus tests).
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")

import re

import numpy as np
import pytest

import dace
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.canonicalize.finalize import finalize_for_target

N = dace.symbol('N')
K = dace.symbol('K')


def _cpu_code(sdfg) -> str:
    finalize_for_target(sdfg, 'cpu')
    return "\n".join(c.clean_code for c in sdfg.generate_code())


def _gpu_code(sdfg) -> str:
    finalize_for_target(sdfg, 'gpu')
    return "\n".join(c.clean_code for c in sdfg.generate_code())


# ---------------------------------------------------------------------------
# Kernels: guarded scalar reductions.
# ---------------------------------------------------------------------------


@dace.program
def condsum(a: dace.float64[N], thresh: dace.float64, out: dace.float64[1]):
    s = 0.0
    for i in range(N):
        if a[i] > thresh:
            s = s + a[i]
    out[0] = s


@dace.program
def condsum_sym(a: dace.float64[N], out: dace.float64[1]):
    s = 0.0
    for i in range(N):
        if a[i] > K:
            s = s + a[i]
    out[0] = s


@dace.program
def condprod(a: dace.float64[N], out: dace.float64[1]):
    p = 1.0
    for i in range(N):
        if a[i] > 1.0:
            p = p * a[i]
    out[0] = p


# ---------------------------------------------------------------------------
# CPU: reduction clause present, guarded atomic gone, value preserved.
# ---------------------------------------------------------------------------


def test_condsum_cpu_emits_reduction_clause_and_no_atomic():
    """``if a[i] > thresh: s += a[i]`` -> OMP ``reduction(+:...)`` on CPU, with
    NO ``reduce_atomic`` for the accumulation."""
    sdfg = condsum.to_sdfg(simplify=True)
    canonicalize(sdfg,
                 target='cpu',
                 peel_limit=4,
                 break_anti_dependence=True,
                 interchange_carry_with_map=True,
                 scatter_to_guarded_maps=True)
    code = _cpu_code(sdfg)
    assert re.search(r'#pragma omp[^\n]*reduction\(', code), \
        "conditional sum should lower to an OpenMP reduction clause"
    assert 'reduce_atomic' not in code, "the guarded atomic should be gone"


def test_condsum_cpu_value_preserving():
    """The tree-reduced conditional sum matches the numpy oracle (allclose;
    the parallel reduction reorders the fp sum vs numpy's pairwise ``.sum()``)."""
    n = 4096
    rng = np.random.default_rng(3111)
    a = rng.standard_normal(n)
    thresh = 0.25
    expected = float(a[a > thresh].sum())

    sdfg = condsum.to_sdfg(simplify=True)
    canonicalize(sdfg,
                 target='cpu',
                 peel_limit=4,
                 break_anti_dependence=True,
                 interchange_carry_with_map=True,
                 scatter_to_guarded_maps=True)
    out = np.zeros(1)
    sdfg(a=a, thresh=thresh, out=out, N=n)
    assert np.allclose(out[0], expected, rtol=1e-9, atol=1e-9, equal_nan=True), \
        f"got {out[0]!r}, expected {expected!r}"


def test_condsum_symbolic_threshold_cpu():
    """Symbolic-threshold sibling (``if a[i] > K``): the mask is computed at
    runtime, still lowering to a reduction clause with a preserved value."""
    n = 2048
    rng = np.random.default_rng(4)
    a = rng.integers(-5, 5, size=n).astype(np.float64)
    k = 1
    expected = float(a[a > k].sum())

    sdfg = condsum_sym.to_sdfg(simplify=True)
    canonicalize(sdfg,
                 target='cpu',
                 peel_limit=4,
                 break_anti_dependence=True,
                 interchange_carry_with_map=True,
                 scatter_to_guarded_maps=True)
    code = _cpu_code(sdfg)
    assert re.search(r'#pragma omp[^\n]*reduction\(', code)
    assert 'reduce_atomic' not in code

    out = np.zeros(1)
    free = {str(s) for s in sdfg.free_symbols}
    for s in free:
        if s not in sdfg.symbols:
            sdfg.add_symbol(s, dace.int64)
    sdfg(a=a, out=out, N=n, K=k)
    assert np.allclose(out[0], expected, rtol=1e-9, atol=1e-9)


def test_condprod_cpu_uses_multiplicative_identity():
    """A guarded PRODUCT reduction masks with the multiplicative identity ``1``
    (from ``reduction_identity``), lowering to ``reduction(*:...)`` and staying
    value-preserving."""
    n = 300
    rng = np.random.default_rng(7)
    a = np.abs(rng.standard_normal(n)) + 0.5  # some > 1, some < 1
    expected = float(np.prod(a[a > 1.0]))

    sdfg = condprod.to_sdfg(simplify=True)
    canonicalize(sdfg,
                 target='cpu',
                 peel_limit=4,
                 break_anti_dependence=True,
                 interchange_carry_with_map=True,
                 scatter_to_guarded_maps=True)
    code = _cpu_code(sdfg)
    assert re.search(r'#pragma omp[^\n]*reduction\(\*', code), "expected a multiplicative reduction clause"
    assert 'reduce_atomic' not in code

    out = np.zeros(1)
    sdfg(a=a, out=out, N=n)
    assert np.allclose(out[0], expected, rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# GPU: structural only (no run) -- block reduce present, guarded atomic gone.
# ---------------------------------------------------------------------------


def test_condsum_gpu_emits_block_reduce():
    """On GPU the conditional sum lowers to a block/warp tree-reduce
    (``BlockReduce``), dropping the per-passing-thread ``atomicAdd``. The single
    residual ``reduce_atomic`` is the per-block combine (identical to a plain
    reduction); assert the block reduce fired and no per-thread atomicAdd."""
    sdfg = condsum.to_sdfg(simplify=True)
    canonicalize(sdfg,
                 target='gpu',
                 peel_limit=4,
                 break_anti_dependence=True,
                 interchange_carry_with_map=False,
                 scatter_to_guarded_maps=True)
    code = _gpu_code(sdfg)
    assert 'BlockReduce' in code, "conditional sum should lower to a GPU block reduce"
    assert 'atomicAdd' not in code, "no per-passing-thread atomicAdd should remain"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
