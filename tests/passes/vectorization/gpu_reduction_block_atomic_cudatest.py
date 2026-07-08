# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU thread-block reduction for the tile-op vectorizer's map-exit WCR.

Scalar reduction (``acc += A[i]``, ``max``, ``min``) on CUDA: fold half2 partials
per thread (``TileReduce``), then lift map-exit WCR to one ``cub::BlockReduce`` per
block + ONE ``reduce_atomic`` from thread 0 (GPU mirror of CPU OpenMP
``reduction(op:var)``), not one atomic per thread. Per-thread partial = thread-local
register: a shared/global partial would have every thread write+read back the SAME
element (race that only "works" by nvcc ``__restrict__`` register-caching luck), and
cub needs a register input anyway.

``min``/``max`` reach this path only via ``AugAssignToWCR`` converting their
loop-carried ``acc = f(acc, A[i])`` (frontend combine-then-copyback subgraph) into a
WCR write so ``LoopToMap`` parallelizes.

Assert CUDA shape without a GPU; compile with nvcc; run+check exact fp16 with a GPU.
Inputs are order-independent (exact small ints for sum; associative max/min) and vary
per element so each thread's partial is distinct across many blocks -- a race or
dropped-block atomic would corrupt the result.
"""
import os

os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import shutil

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.transformation.interstate import LoopToMap
from dace.transformation.dataflow.wcr_conversion import AugAssignToWCR
from dace.transformation.passes.vectorization.vectorize_gpu import VectorizeGPU
from dace.libraries.tileops import TileReduce

_HAS_NVCC = shutil.which("nvcc") is not None
N = dace.symbol("N")


@dace.program
def _vsum16(A: dace.float16[N], out: dace.float16[1]):
    acc = dace.float16(0.0)
    for i in dace.map[0:N]:
        acc += A[i]
    out[0] = acc


@dace.program
def _vmax16(A: dace.float16[N], out: dace.float16[1]):
    acc = dace.float16(-1.0e4)
    for i in range(N):
        acc = max(acc, A[i])
    out[0] = acc


@dace.program
def _vmin16(A: dace.float16[N], out: dace.float16[1]):
    acc = dace.float16(1.0e4)
    for i in range(N):
        acc = min(acc, A[i])
    out[0] = acc


# (program, cub reduction-type suffix as emitted in ``dace::ReductionType::<...>``)
_PROGRAMS = {"sum": (_vsum16, "Sum"), "max": (_vmax16, "Max"), "min": (_vmin16, "Min")}


def _vectorized(prog):
    """@dace.program -> simplify + AugAssignToWCR + LoopToMap + VectorizeGPU (half2 GPU)."""
    sdfg = prog.to_sdfg(simplify=True)
    # min/max loop-carried -> WCR writes here; sum already map+WCR from frontend, unaffected.
    sdfg.apply_transformations_repeated(AugAssignToWCR)
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.simplify()
    VectorizeGPU().apply_pass(sdfg, {})
    return sdfg


def _device_code(sdfg):
    return "\n".join(c.clean_code for c in sdfg.generate_code() if c.language == "cu")


@pytest.mark.parametrize("kind", list(_PROGRAMS))
def test_partial_is_thread_local_register(kind):
    """The interposed reduction access node ``_nmr_out`` is a single-element Register
    transient (per-thread), never the shared global accumulator."""
    sdfg = _vectorized(_PROGRAMS[kind][0])
    parts = [(k, d) for s in sdfg.all_sdfgs_recursive() for k, d in s.arrays.items() if k.startswith("_nmr_out")]
    assert parts, "expected an interposed _nmr_out reduction partial"
    for k, d in parts:
        assert d.storage == dtypes.StorageType.Register, f"{k} partial must be Register, got {d.storage}"
        assert d.total_size == 1, f"{k} reduction partial must fold onto a single element, got {d.total_size}"


@pytest.mark.parametrize("kind", list(_PROGRAMS))
def test_half2_tile_reduce_fires(kind):
    """The within-thread half2->half fold is a ``TileReduce`` of width 2."""
    sdfg = _vectorized(_PROGRAMS[kind][0])
    reds = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TileReduce)]
    assert len(reds) == 1 and list(reds[0].widths) == [2], \
        f"expected one width-2 TileReduce; got {[r.widths for r in reds]}"


@pytest.mark.parametrize("kind", list(_PROGRAMS))
def test_emits_block_reduce_and_single_atomic(kind):
    """The device TU folds the block with ``cub::BlockReduce`` and commits ONE atomic
    from thread 0 with the op's reduction functor; the per-thread atomic is suppressed."""
    cu = _device_code(_vectorized(_PROGRAMS[kind][0]))
    suffix = _PROGRAMS[kind][1]
    assert "cub::BlockReduce<dace::float16, 32>" in cu, "block reduce not typed to the 32-thread block"
    assert ".Reduce(" in cu, "cub block Reduce call missing"
    assert f"dace::ReductionType::{suffix}" in cu, f"block reduce not using the {suffix} functor"
    assert "reduce_atomic" in cu, "thread-0 atomic to the global accumulator missing"
    assert "threadIdx.x == 0" in cu, "atomic not guarded to a single thread per block"
    assert "__shared__" in cu, "block-reduce temp storage not in shared memory"
    assert "folded by GPU block reduction" in cu, "per-thread atomic not suppressed"


@pytest.mark.skipif(not _HAS_NVCC, reason="nvcc not available; compile check skipped")
@pytest.mark.parametrize("kind", list(_PROGRAMS))
def test_compiles(kind):
    sdfg = _vectorized(_PROGRAMS[kind][0])
    sdfg.name = f"gpu_block_reduction_compile_{kind}"
    shutil.rmtree(os.path.join(".dacecache", sdfg.name), ignore_errors=True)
    sdfg.compile()


def _run_inputs(kind, nval):
    if kind == "sum":
        return (np.arange(nval) % 3).astype(np.float16), lambda a: np.float16(a.sum())
    rng = np.random.default_rng(nval)
    a = (rng.permutation(nval) % 101 - 50).astype(np.float16)
    return a, (np.max if kind == "max" else np.min)


@pytest.mark.gpu
@pytest.mark.parametrize("kind", list(_PROGRAMS))
def test_runs_exact_multiblock(kind):
    sdfg = _vectorized(_PROGRAMS[kind][0])
    sdfg.name = f"gpu_block_reduction_run_{kind}"
    shutil.rmtree(os.path.join(".dacecache", sdfg.name), ignore_errors=True)
    csdfg = sdfg.compile()
    for nval in (64, 257, 1000, 4000):  # up to ~63 blocks (int_ceil(N, 64))
        a, ref = _run_inputs(kind, nval)
        out = np.zeros(1, dtype=np.float16)
        csdfg(A=a.copy(), out=out, N=nval)
        exp = np.float16(ref(a))
        assert out[0] == exp, f"{kind} N={nval}: {float(out[0])} != {float(exp)}"


if __name__ == "__main__":
    for _kind in _PROGRAMS:
        test_partial_is_thread_local_register(_kind)
        test_half2_tile_reduce_fires(_kind)
        test_emits_block_reduce_and_single_atomic(_kind)
    print("codegen ok")
