# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the skewed-TILING lowering of :class:`WavefrontSkew`.

The pass emits a sequential tile-diagonal loop over a parallel tile-column Map over
two sequential intra-tile loops, the innermost at unit stride. These tests pin the
emitted shape per domain shape, the bit-exactness of the result against the same nest
run sequentially, the fallback to the element-granularity lowering when a dependence
outruns a tile, and determinism.
"""
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.canonicalize.wavefront_skew import (WavefrontSkew, _SKEW_P_PREFIX, _SKEW_T_PREFIX,
                                                                    tiling_legal, Dependence)

N = dace.symbol('N')


@dace.program
def wf_north_west(a: dace.float64[N, N]):
    """tsvc_2_5 ``wf_north_west``: rectangular sum-diagonal wavefront."""
    for i in range(1, N):
        for j in range(1, N):
            a[i, j] = a[i, j] + a[i - 1, j] + a[i, j - 1]


@dace.program
def wf_triangular(a: dace.float64[N, N]):
    """tsvc_2_5 ``wf_triangular``: the same recurrence over the upper triangle ``j >= i``."""
    for i in range(1, N):
        for j in range(i, N):
            a[i, j] = a[i, j] + a[i - 1, j] + a[i, j - 1]


@dace.program
def wavefront2d(a: dace.float64[N, N]):
    """tsvc_2_5 ``wavefront2d`` (s2111): north + west + the extra ``(1, 1)`` corner dep."""
    for i in range(1, N):
        for j in range(1, N):
            a[i, j] = 0.25 * (a[i, j] + a[i - 1, j] + a[i, j - 1] + a[i - 1, j - 1])


@dace.program
def far_carry(a: dace.float64[N, N]):
    """A wavefront whose ``u`` distance is 3: neither axis is parallel on its own, so the
    pass skews, but the tiling is only legal once a tile is at least 3 rows tall."""
    for i in range(3, N):
        for j in range(1, N):
            a[i, j] = a[i, j] + a[i - 3, j] + a[i, j - 1]


@dace.program
def gauss_seidel_backward(a: dace.float64[N, N]):
    """The steep ``tau = (2, 1)`` Gauss-Seidel nest. Its flow distance ``(-1, +1)`` is legal
    element-wise but NOT once tiled -- the pair can straddle the ``v`` boundary alone."""
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            a[i, j] = (a[i, j - 1] + a[i - 1, j] + a[i - 1, j - 1] + a[i - 1, j + 1]) / 4.0


ALL_SHAPES = [wf_north_west, wf_triangular, wavefront2d]


def skewed(prog, bi=32, bj=32, target='cpu'):
    """``prog``'s SDFG after an isolated :class:`WavefrontSkew` with the given tile extents."""
    sdfg = prog.to_sdfg(simplify=True)
    xf = WavefrontSkew(target=target)
    xf.tile_i, xf.tile_j = bi, bj
    fired = xf.apply_pass(sdfg, {})
    sdfg.validate()
    return sdfg, fired


def loops(sdfg):
    return [
        r for sd in sdfg.all_sdfgs_recursive() for r in sd.all_control_flow_regions()
        if isinstance(r, LoopRegion) and r.loop_variable
    ]


def structure(sdfg):
    """A stable signature of every emitted loop plus the Map parameters."""
    sig = [(r.loop_variable, r.init_statement.as_string, r.loop_condition.as_string, r.update_statement.as_string,
            r.pinned_sequential) for r in loops(sdfg)]
    return sig, [tuple(n.map.params) for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]


@pytest.mark.parametrize('prog', ALL_SHAPES, ids=lambda p: p.name)
def test_tiled_wavefront_emits_pinned_diagonal_over_map_over_unit_stride_nest(prog):
    """Rectangular, triangular and extra-diagonal shapes all lower to the SAME four-level
    shape: pinned tile diagonal > parallel tile-column Map > two pinned intra-tile loops,
    the innermost of unit stride and holding no further loop."""
    sdfg, fired = skewed(prog)
    assert fired == 1, f'{prog.name}: the tiled wavefront must fire'

    emitted = loops(sdfg)
    assert len(emitted) == 3, f'{prog.name}: expected T + intra-tile i + j; got {[l.loop_variable for l in emitted]}'
    diag = [l for l in emitted if l.loop_variable.startswith(_SKEW_T_PREFIX)]
    assert len(diag) == 1 and diag[0].pinned_sequential, 'the tile diagonal must be the one sequential-pinned axis'

    maps = [n.map for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]
    assert len(maps) == 1 and maps[0].params[0].startswith(_SKEW_P_PREFIX), \
        f'{prog.name}: expected one parallel tile-column Map; got {[m.params for m in maps]}'

    intra = [l for l in emitted if l is not diag[0]]
    assert all(l.pinned_sequential for l in intra), 'intra-tile loops carry the dependences and must stay sequential'
    innermost = [
        l for l in intra if not any(isinstance(c, LoopRegion) for c in l.all_control_flow_regions() if c is not l)
    ]
    assert len(innermost) == 1, f'{prog.name}: expected exactly one innermost loop'
    assert loop_analysis.get_loop_stride(innermost[0]) == 1, 'the innermost tile loop must be unit stride'


@pytest.mark.parametrize('prog', ALL_SHAPES, ids=lambda p: p.name)
def test_tiled_wavefront_is_bit_exact_against_the_sequential_nest(prog):
    """N=96 with 32x32 tiles is a 3x3 tile grid, so every tile boundary is exercised. The
    tiled result must equal the SAME kernel run sequentially bit for bit -- each cell is
    written once and every summand it reads is the final value from a strictly earlier tile
    diagonal (or from the same tile, in the original order), so nothing is reassociated."""
    n = 96
    rng = np.random.default_rng(20260807)
    a0 = rng.standard_normal((n, n))

    reference = prog.to_sdfg(simplify=True)
    ref = a0.copy()
    reference(a=ref, N=n)

    sdfg, fired = skewed(prog)
    assert fired == 1
    got = a0.copy()
    sdfg(a=got, N=n)
    assert np.array_equal(got, ref), f'{prog.name}: max abs diff {np.max(np.abs(got - ref)):.3e}'


def test_dependence_outrunning_the_tile_falls_back_to_the_untiled_lowering():
    """``a[i - 3, j]`` reaches three rows back. With 2-row tiles the tile-index distance is
    no longer a clamped sign, so the pass must keep the element-granularity diagonal (one
    loop, one Map); with 4-row tiles the very same nest tiles. The pair is what makes this a
    test of the guard rather than of the nest."""
    small, _ = skewed(far_carry, bi=2, bj=64)
    assert len(loops(small)) == 1 and loops(small)[0].loop_variable.startswith(_SKEW_T_PREFIX), \
        f'a distance longer than the tile must keep the untiled diagonal; got {[l.loop_variable for l in loops(small)]}'

    big, fired = skewed(far_carry, bi=4, bj=64)
    assert fired == 1 and len(loops(big)) == 3, \
        f'non-vacuity: the same nest must tile once the tile is tall enough; got {[l.loop_variable for l in loops(big)]}'


@pytest.mark.parametrize('prog', ALL_SHAPES, ids=lambda p: p.name)
def test_both_targets_block_and_the_gpu_also_skews_the_tile(prog):
    """Both targets block, and they block for different reasons, so they emit different shapes.

    A CPU tile is cache blocking: a parallel tile-column Map over two SEQUENTIAL interior loops,
    the innermost at unit stride. Three loops, one Map.

    A GPU tile is what turns one kernel launch per element anti-diagonal into one per TILE
    anti-diagonal. That alone would leave each block running its whole tile on one thread, so the
    interior is skewed a second time: a sequential intra-tile diagonal over a parallel Map across
    the block's threads. Two loops, two Maps -- and the inner Map carries ``is_warp_tile``, the
    request :class:`PromoteWarpTiles` redeems after the device offload has sequentialized it.

    The pair is what makes this a test of the target gate rather than of the nest: both targets
    must still skew, and to the SAME answer, which the numeric check below pins."""
    cpu, cpu_fired = skewed(prog, target='cpu')
    gpu, gpu_fired = skewed(prog, target='gpu')
    assert cpu_fired == 1 and gpu_fired == 1, f'{prog.name}: the wavefront must be skewed on both targets'

    assert len(loops(cpu)) == 3, f'{prog.name}: cpu tiles; got {[l.loop_variable for l in loops(cpu)]}'
    cpu_maps = [n.map for n, _ in cpu.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]
    assert len(cpu_maps) == 1 and not cpu_maps[0].is_warp_tile, \
        f'{prog.name}: the cpu interior stays sequential; got {[m.params for m in cpu_maps]}'

    gpu_loops = loops(gpu)
    assert len(gpu_loops) == 2 and all(l.loop_variable.startswith(_SKEW_T_PREFIX) for l in gpu_loops), \
        f'{prog.name}: gpu runs a tile diagonal over an intra-tile diagonal; ' \
        f'got {[l.loop_variable for l in gpu_loops]}'

    gpu_maps = [n.map for n, _ in gpu.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]
    assert len(gpu_maps) == 2, f'{prog.name}: grid Map over thread-block Map; got {[m.params for m in gpu_maps]}'
    assert all(m.params[0].startswith(_SKEW_P_PREFIX) for m in gpu_maps), \
        f'{prog.name}: both Maps are skewed parallel axes; got {[m.params for m in gpu_maps]}'
    assert sum(m.is_warp_tile for m in gpu_maps) == 1, \
        f'{prog.name}: exactly the interior Map is tagged; got {[(m.params, m.is_warp_tile) for m in gpu_maps]}'

    n = 96
    rng = np.random.default_rng(20260824)
    a0 = rng.standard_normal((n, n))
    reference = prog.to_sdfg(simplify=True)
    ref = a0.copy()
    reference(a=ref, N=n)
    for label, sdfg in (('cpu', cpu), ('gpu', gpu)):
        got = a0.copy()
        sdfg(a=got, N=n)
        assert np.array_equal(got, ref), f'{prog.name}/{label}: max abs diff {np.max(np.abs(got - ref)):.3e}'


def test_steep_gauss_seidel_diagonal_is_refused_by_the_tiling_guard():
    """``tau = (2, 1)`` with the flow distance ``(-1, +1)``: legal element-wise, illegal
    tiled (the pair can straddle only the ``v`` boundary, giving the tile distance
    ``(0, +1)`` and ``tau . (0, 1) = +1 > 0``). Checked both on the predicate and on the
    nest the pass actually emits."""
    p = dace.symbolic.pystr_to_symbolic
    deps = [Dependence(p(du), p(dv), [], 'flow') for du, dv in ((0, -1), (-1, 0), (-1, -1), (-1, 1))]
    assert tiling_legal(deps, (1, 1), 64, 64) is False, 'sanity: (1, 1) is not even legal element-wise here'
    assert tiling_legal(deps, (2, 1), 64, 64) is False, 'the straddling (0, +1) tile distance must refuse (2, 1)'
    assert tiling_legal(deps[:2], (2, 1), 64, 64) is True, 'non-vacuity: without (-1, +1) the same tau tiles'

    sdfg, fired = skewed(gauss_seidel_backward)
    assert fired == 1
    assert len(loops(sdfg)) == 1, f'the steep diagonal must stay untiled; got {[l.loop_variable for l in loops(sdfg)]}'


@pytest.mark.parametrize('prog', ALL_SHAPES, ids=lambda p: p.name)
def test_tiled_wavefront_is_idempotent(prog):
    """A second run of the pass finds no wavefront left and changes nothing -- the emitted
    tile diagonal is pinned sequential and its intra-tile loops are not two-level wavefronts."""
    sdfg, fired = skewed(prog)
    assert fired == 1
    before = structure(sdfg)
    again = WavefrontSkew()
    again.tile_i, again.tile_j = 32, 32
    assert again.apply_pass(sdfg, {}) is None, f'{prog.name}: the pass must not re-skew its own output'
    assert structure(sdfg) == before, f'{prog.name}: a no-match run must leave the SDFG untouched'


def test_tiled_wavefront_is_deterministic_under_a_randomised_hash_seed():
    """The emitted bounds must not depend on ``PYTHONHASHSEED``: the pass sorts every symbol
    set it walks, and the tile counts come from an ISL projection, not from a dict order."""
    src = textwrap.dedent('''
        import dace
        from dace.sdfg import nodes
        from dace.sdfg.state import LoopRegion
        from dace.transformation.passes.canonicalize.wavefront_skew import WavefrontSkew
        N = dace.symbol('N')

        @dace.program
        def prog(a: dace.float64[N, N]):
            for i in range(1, N):
                for j in range(i, N):
                    a[i, j] = a[i, j] + a[i - 1, j] + a[i, j - 1]

        sdfg = prog.to_sdfg(simplify=True)
        xf = WavefrontSkew()
        xf.tile_i, xf.tile_j = 32, 32
        assert xf.apply_pass(sdfg, {}) == 1
        sig = []
        for sd in sdfg.all_sdfgs_recursive():
            for r in sd.all_control_flow_regions():
                if isinstance(r, LoopRegion) and r.loop_variable:
                    sig.append((r.loop_variable, r.init_statement.as_string, r.loop_condition.as_string))
        sig.append([n.map.params for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)])
        print('SIG', sig, flush=True)
    ''')
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        script = os.path.join(tmp, 'wavefront_tiling_seed_child.py')
        with open(script, 'w') as fh:
            fh.write(src)
        sigs = []
        for seed in ('0', '1'):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            res = subprocess.run([sys.executable, script], capture_output=True, timeout=600, env=env)
            lines = [l for l in res.stdout.decode().splitlines() if l.startswith('SIG')]
            assert lines, f'child with PYTHONHASHSEED={seed} produced no signature: {res.stderr[-500:]!r}'
            sigs.append(lines[0])
    assert sigs[0] == sigs[1], f'hash-seed dependent lowering:\n{sigs[0]}\n{sigs[1]}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
