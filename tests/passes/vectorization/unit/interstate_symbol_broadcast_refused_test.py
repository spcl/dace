# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A tile operand an interstate assignment ties to the iter_var must not be broadcast.

``ConvertTaskletsToTileOps`` inlines a tasklet operand as a uniform ``Symbol`` when the operand
does not SPELL a tile iter_var. That test is not the question it is used to answer: the frontend
routes an array element into a tasklet through an interstate assignment, so a compaction-mask body
reads a bare ``b_index`` (defined by ``b_index = b[i]`` on the edge into the state) that names no
iter_var yet holds a different value in every lane. Inlining it splats lane 0 across the tile --
TSVC s341/s342/s343 drifted by ~1.5-1.8 in absolute value. "Names no iter_var" is not "is
loop-invariant"; the assignment chain has to be resolved before the operand can be believed.

Pack/expand is not vectorizable by broadcast in any case: ``if (b[i] > 0) { a[++j] = b[i]; }``
advances ``j`` under a data-dependent predicate, so a lane-parallel form needs a real prefix sum of
the mask. Canonicalization does build that scan; the tile widener then miscompiled the MASK map
feeding it. The contract pinned here is the honest one -- refuse, leave the kernel scalar and
correct -- not a silent wrong answer.

The over-refusal control matters as much as the refusal: an interstate assignment that reads a
LOOP-INVARIANT element (``alpha = c[0]``) is a genuine broadcast and must still tile, or the guard
would have bought correctness by disabling vectorization wholesale.
"""
import os

os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import numpy as np
import pytest

import dace
from dace.libraries.tileops._dispatch import detect_host_isa
from dace.sdfg import nodes as nd
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import RemainderStrategy, BranchMode
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

N = dace.symbol('N')
W = 8
#: Host's best runnable SIMD ISA -- vectorization enforces arch-native, so a pinned AVX-512 would
#: SIGILL on an AVX2-only or ARM host.
_HOST_ISA = detect_host_isa()


@dace.program
def pack_kernel(a: dace.float64[N], b: dace.float64[N]):
    """TSVC s341: conditional compress. ``j`` advances only on the taken lanes."""
    j = -1
    for i in range(N):
        if b[i] > 0.0:
            j = j + 1
            a[j] = b[i]


@dace.program
def expand_kernel(a: dace.float64[N], b: dace.float64[N]):
    """TSVC s342: conditional expand -- the gather index is the compaction rank."""
    j = -1
    for i in range(N):
        if a[i] > 0.0:
            j = j + 1
            a[i] = b[j]


@dace.program
def invariant_symbol_kernel(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    """Over-refusal control: ``alpha`` is read ONCE at a fixed index, so it is uniform across the
    tile and the map must still be strided."""
    alpha = c[0]
    for i in range(N):
        a[i] = a[i] + alpha * b[i]


def pack_reference(a, b, n):
    out = a.copy()
    j = -1
    for i in range(n):
        if b[i] > 0.0:
            j = j + 1
            out[j] = b[i]
    return out


def expand_reference(a, b, n):
    out = a.copy()
    j = -1
    for i in range(n):
        if out[i] > 0.0:
            j = j + 1
            out[i] = b[j]
    return out


def invariant_reference(a, b, c, n):
    return a + c[0] * b


def vectorized(prog, tag):
    """Canonicalize then vectorize ``prog``. A refusal is internal (the pass warns and restores),
    so the returned SDFG is always valid -- the caller distinguishes the two outcomes by whether
    the maps were strided."""
    sdfg = prog.to_sdfg(simplify=True)
    sdfg.name = tag
    canonicalize(sdfg, validate=True, peel_limit=4, break_anti_dependence=True)
    VectorizeCPUMultiDim(
        VectorizeConfig(widths=(W, ),
                        validate_all=True,
                        target_isa=_HOST_ISA,
                        remainder_strategy=RemainderStrategy.FULL_MASK,
                        branch_mode=BranchMode.MERGE)).apply_pass(sdfg, {})
    sdfg.validate()
    return sdfg


def map_steps(sdfg):
    """Step of every map in ``sdfg``, innermost dim. ``'1'`` everywhere == nothing was tiled."""
    return [str(m.map.range[-1][2]) for m, _ in sdfg.all_nodes_recursive() if isinstance(m, nd.MapEntry)]


def run(sdfg, kwargs, buf, ref, label):
    """Compile + run in-process and compare.

    The sibling ``tile_param_dependent_branch_refused_test`` forks first, to contain an
    out-of-bounds masked-tail read. Not needed here and deliberately not copied: none of these
    cases reads past an array (the refused kernels emit scalar code, and the control's n=64 is
    tile-aligned, so no masked tail exists), and ``os.fork()`` around ``sdfg.compile()`` deadlocks
    the child on an inherited lock once other tests in this directory have run -- a hang is a
    worse failure mode than the crash it would be guarding against.
    """
    sdfg.compile()(**kwargs)
    assert np.allclose(buf, ref, rtol=1e-12, atol=1e-12), \
        f'{label}: max|diff|={np.nanmax(np.abs(np.asarray(buf) - np.asarray(ref))):.3e}'


@pytest.mark.parametrize('n', [64, 61])
def test_pack_matches_numpy(n):
    """Numeric contract. ``n=61`` is deliberately NOT a multiple of ``W``: the tile base then walks
    off the lane-0 element the broadcast used, so a regression drifts on the ragged case too."""
    rng = np.random.default_rng(1234)
    a, b = rng.random(n), rng.random(n) - 0.5
    ref = pack_reference(a, b, n)
    sdfg = vectorized(pack_kernel, f'pack_{n}')
    work = a.copy()
    run(sdfg, dict(a=work, b=b.copy(), N=n), work, ref, f'pack n={n}')


@pytest.mark.parametrize('n', [64, 61])
def test_expand_matches_numpy(n):
    rng = np.random.default_rng(4321)
    a, b = rng.random(n) - 0.5, rng.random(n)
    ref = expand_reference(a, b, n)
    sdfg = vectorized(expand_kernel, f'expand_{n}')
    work = a.copy()
    run(sdfg, dict(a=work, b=b.copy(), N=n), work, ref, f'expand n={n}')


def test_pack_is_refused_rather_than_broadcast():
    """Structural half: the mask map must stay unstrided. Before the guard it was strided by 8 and
    its predicate read ``b`` at the TILE BASE only, so all 8 lanes shared lane 0's verdict."""
    assert set(map_steps(vectorized(pack_kernel, 'pack_struct'))) == {'1'}, \
        'pack kernel was tiled; its mask predicate broadcasts the tile-base element across the lanes'


def test_invariant_interstate_symbol_still_tiles():
    """Over-refusal control: ``alpha = c[0]`` names no iter_var and its interstate definition does
    not reach one, so it is a genuine uniform broadcast and the map must still be strided."""
    steps = map_steps(vectorized(invariant_symbol_kernel, 'invariant_sym'))
    assert steps and str(W) in steps, f'loop-invariant interstate symbol blocked tiling; map steps were {steps}'


def test_invariant_interstate_symbol_matches_numpy():
    n = 64
    rng = np.random.default_rng(99)
    a, b, c = rng.random(n), rng.random(n), rng.random(n)
    ref = invariant_reference(a, b, c, n)
    sdfg = vectorized(invariant_symbol_kernel, 'invariant_sym_num')
    work = a.copy()
    run(sdfg, dict(a=work, b=b.copy(), c=c.copy(), N=n), work, ref, 'invariant symbol')


if __name__ == '__main__':
    test_pack_matches_numpy(64)
    test_pack_matches_numpy(61)
    test_expand_matches_numpy(64)
    test_expand_matches_numpy(61)
    test_pack_is_refused_rather_than_broadcast()
    test_invariant_interstate_symbol_still_tiles()
    test_invariant_interstate_symbol_matches_numpy()
