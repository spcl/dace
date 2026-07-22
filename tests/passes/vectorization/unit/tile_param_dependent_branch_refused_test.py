# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A conditional guarding a TILED map's own param keeps that map scalar only when the arms NEED it.

Striding rebinds a map's params from per-iteration index to TILE BASE, so a guard over one that
SURVIVES to codegen is evaluated once per tile -- lane 0 decides for all W lanes. That is the
divergence this file exists to pin.

Surviving is the operative word. A guard whose arms are in range regardless of it only SELECTS a
value, and if-conversion turns it into a per-lane blend that no longer consults the tile base:
``if i + 1 < mid: a[i] = a[i] + b[i]*c[i] else: a[i] = a[i] + b[i]*d[i]`` indexes ``[i]`` in both
arms, so evaluating both and blending per lane is exact, and the map tiles. A guard the arms need
for their range is the real refusal -- drop it and ``b[i + 1]`` runs off the end at ``i = N-1`` --
and that map stays scalar.

The refusal is likewise scoped to the map's OWN params: a guard over an enclosing scope's symbol is
uniform across the tiled lanes, the cloudsc ``for jk: for jl: if jk > 1`` shape.
"""
import os

os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")

import numpy as np
import pytest

import dace
from dace.sdfg import nodes as nd
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA, RemainderStrategy, BranchMode
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

M = dace.symbol('M')
N = dace.symbol('N')
W = 8


@dace.program
def range_guard_kernel(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    """Guard the arms NEED for their range: without it, lane ``i = N-1`` reads ``b[i + 1]`` off the
    end. Cannot be if-converted, so it survives as a per-tile predicate -> the map must stay
    scalar."""
    for i in range(N):
        if i + 1 < N:
            a[i] = a[i] + b[i + 1] * c[i]


@dace.program
def index_guard_kernel(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
    """Guard on the TILED param, but both arms index ``[i]`` -- in range for every ``i`` the map
    runs, so it only SELECTS a value and if-converts to a per-lane blend."""
    mid = N // 2
    for i in range(N):
        if i + 1 < mid:
            a[i] = a[i] + b[i] * c[i]
        else:
            a[i] = a[i] + b[i] * d[i]


@dace.program
def outer_param_guard_kernel(a: dace.float64[M, N], b: dace.float64[M, N], c: dace.float64[M, N]):
    """Guard on the OUTER map's param -> uniform across the tiled lanes -> must still tile."""
    for j in range(M):
        for i in range(N):
            if j > 0:
                a[j, i] = a[j, i] + b[j, i] * c[j, i]


def fork_run(sdfg, kwargs, outputs) -> None:
    """Compile + run in a forked child; assert every output matches its ref. Forked so a
    would-be out-of-bounds access (the N=61 masked tail reads a full 8-lane tile off a
    61-element buffer) cannot take the pytest process down, and because the child mutates the
    buffers in its own copy-on-write space -- so the comparison happens IN the child and the
    verdict travels back as an exit code.

    ``allclose``, not ``array_equal``: g++ defaults to ``-ffp-contract=fast``, so the compiled
    ``a[i] + b[i]*c[i]`` contracts to one FMA while the numpy oracle rounds twice -- a ~1 ulp
    delta that is correct lowering, not a bug.
    """
    pid = os.fork()
    if pid == 0:
        code = 0
        try:
            sdfg.compile()(**kwargs)
            code = 0 if all(np.allclose(buf, ref, rtol=1e-12, atol=1e-12) for buf, ref in outputs) else 3
        except BaseException:  # noqa: BLE001 -- child reports any failure via exit code
            code = 4
        os._exit(code)
    _, status = os.waitpid(pid, 0)
    assert os.WIFEXITED(status), f'kernel died on signal {os.WTERMSIG(status) if os.WIFSIGNALED(status) else "?"}'
    assert os.WEXITSTATUS(status) == 0, f'forked kernel failed with exit code {os.WEXITSTATUS(status)}'


def vectorized(prog, tag):
    sdfg = prog.to_sdfg(simplify=True)
    sdfg.name = tag
    canonicalize(sdfg, validate=True, peel_limit=4, break_anti_dependence=True)
    VectorizeCPUMultiDim(
        VectorizeConfig(widths=(W, ),
                        validate_all=True,
                        target_isa=ISA.AVX512,
                        remainder_strategy=RemainderStrategy.FULL_MASK,
                        branch_mode=BranchMode.MERGE)).apply_pass(sdfg, {})
    sdfg.validate()
    return sdfg


def sole_map(sdfg, nparams):
    """The single MapEntry with ``nparams`` params. Pins identity rather than scanning every map,
    so a fixture that silently stops producing the map under test fails loudly instead of passing
    vacuously. Canonicalization renames the source index (``i`` -> ``_loop_it_0``) and may merge a
    nest into one N-D map, so identify by arity, not by the name written in the kernel.
    """
    found = [m for m, _ in sdfg.all_nodes_recursive() if isinstance(m, nd.MapEntry) and len(m.map.params) == nparams]
    assert len(
        found) == 1, f'expected exactly one {nparams}-param map, found {len(found)}: {[m.map.params for m in found]}'
    return found[0]


def step_of(map_entry):
    return str(map_entry.map.range[-1][2])


def inputs_1d(n, seed=1234):
    rng = np.random.default_rng(seed)
    return rng.random(n), rng.random(n), rng.random(n), rng.random(n)


def index_guard_reference(a, b, c, d, n):
    out = a.copy()
    mid = n // 2
    for i in range(n):
        out[i] = a[i] + b[i] * (c[i] if i + 1 < mid else d[i])
    return out


@pytest.mark.parametrize('n', [64, 61])
def test_index_guard_kernel_matches_numpy(n):
    """The flip point (i = n//2 - 1) is deliberately NOT tile-aligned, so a per-tile predicate
    mis-predicates. n=64 -> mid=32, flip at i=31, the last lane of tile [24:32) -> exactly one
    wrong index. n=61 -> mid=30, flip at i=29, mid-tile -> lanes 29..31 wrong, and 61 = 7*8+5
    also exercises the 5-element masked remainder."""
    a, b, c, d = inputs_1d(n)
    ref = index_guard_reference(a, b, c, d, n)
    sdfg = vectorized(index_guard_kernel, f'index_guard_{n}')
    fork_run(sdfg, dict(a=a.copy(), b=b, c=c, d=d, N=n), [])
    work = a.copy()
    fork_run(sdfg, dict(a=work, b=b, c=c, d=d, N=n), [(work, ref)])


def test_range_protecting_guard_leaves_the_map_scalar():
    """Structural half: an arm that needs the guard for its range keeps the i-map unstrided."""
    sdfg = vectorized(range_guard_kernel, 'range_guard_struct')
    assert step_of(sole_map(sdfg, 1)) == '1', 'map was tiled despite a guard its arms need for range'


def test_range_protecting_guard_survives_as_a_scalar_loop_in_the_emitted_cpp():
    """The bug is defined in terms of emitted C++ -- a scalar ``if`` over the tile base -- so pin
    the C++. Catches the inverse regression the numeric test cannot: the map going scalar because
    some earlier pass started bailing, leaving the guard correct but this predicate dead."""
    sdfg = vectorized(range_guard_kernel, 'range_guard_cpp')
    code = sdfg.generate_code()[0].clean_code
    assert 'tile_mask_gen' not in code, 'the guarded map was tiled: tile ops emitted around a per-tile predicate'
    assert '+= 8' not in code, 'a map was strided by the tile width despite the guard'


def test_value_selecting_guard_is_if_converted_and_tiles():
    """The other side of the discriminator: both arms in range, so the guard becomes a per-lane
    blend and the map tiles. Without this, refusing every param guard would also pass."""
    sdfg = vectorized(index_guard_kernel, 'index_guard_blend')
    assert step_of(sole_map(sdfg, 1)) == str(W), 'a value-selecting guard must not keep the map scalar'


def test_outer_param_guard_still_tiles():
    """OVER-REFUSAL CONTROL: ``if j > 0`` guards an ENCLOSING map's param.

    ``j`` is uniform across the tiled i-lanes, so the i-map must still tile. This is the cloudsc
    ``for jk: for jl: if jk > 1`` shape.

    Deliberately makes no claim about a ``ConditionalBlock`` surviving to the gate. This arm is
    in range for every ``(j, i)``, so it if-converts to a masked write like any other -- asserting
    the block survives pins the lowering route rather than the contract, and the route changed.
    """
    mm, nn = 4, 64
    rng = np.random.default_rng(7)
    a, b, c = rng.random((mm, nn)), rng.random((mm, nn)), rng.random((mm, nn))
    ref = a.copy()
    ref[1:, :] = a[1:, :] + b[1:, :] * c[1:, :]

    sdfg = vectorized(outer_param_guard_kernel, 'outer_param_guard')
    assert step_of(sole_map(sdfg, 2)) == str(W), 'lane-uniform outer guard wrongly refused the tiled dim'

    work = a.copy()
    fork_run(sdfg, dict(a=work, b=b, c=c, M=mm, N=nn), [(work, ref)])


if __name__ == '__main__':
    test_index_guard_kernel_matches_numpy(64)
    test_index_guard_kernel_matches_numpy(61)
    test_range_protecting_guard_leaves_the_map_scalar()
    test_range_protecting_guard_survives_as_a_scalar_loop_in_the_emitted_cpp()
    test_value_selecting_guard_is_if_converted_and_tiles()
    test_outer_param_guard_still_tiles()
