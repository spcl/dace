# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""G5 regression tests for :class:`SplitMapForTileRemainder` (design section 8.2).

Verifies the K-boundary peel produces exactly K+1 regions (1 interior +
K boundary slabs) for every K in {1, 2, 3} with non-divisible bounds.
This is the load-bearing region-count invariant; a Cartesian split
would produce 2^K regions which is wrong (section 8.2 algorithm).
"""
import copy
import signal

import numpy as np

import dace
from dace.libraries.tileops._dispatch import detect_host_isa
from dace.sdfg.nodes import MapEntry, Tasklet
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.split_map_for_tile_remainder import (
    SplitMapForTileRemainder,
    TILE_MAIN_MARKER,
)
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim
from tests.helpers.isolation import exit_code
from tests.passes.vectorization.tile_assertions import assert_tiled

N = dace.symbol("N")
M = dace.symbol("M")


def _count_range_guards(sdfg):
    """Count ``assume_even`` runtime-guard states and their side-effect trap tasklets.

    :param sdfg: The SDFG to inspect.
    :returns: ``(number of tile_even_range_check states, number of side-effect trap tasklets)``.
    """
    guards = 0
    traps = 0
    for state in sdfg.all_states():
        if "tile_even_range_check" not in state.label:
            continue
        guards += 1
        for node in state.nodes():
            if isinstance(node, Tasklet) and node.side_effects:
                traps += 1
    return guards, traps


def _count_map_regions(sdfg, kernel_label):
    """Count interior + boundary MapEntry nodes whose label starts with kernel_label.

    Interior = the one (if any) MapEntry whose label ends with TILE_MAIN_MARKER. All other
    matching MapEntries are boundary slabs (in masked mode they keep their original label
    so we can't filter by suffix; in scalar / tile_k1 modes they get explicit markers).
    """
    total = 0
    interior = 0
    for node, _ in sdfg.all_nodes_recursive():
        if not isinstance(node, MapEntry):
            continue
        lbl = node.map.label
        if not lbl.startswith(kernel_label):
            continue
        total += 1
        if lbl.endswith(TILE_MAIN_MARKER):
            interior += 1
    boundary = total - interior
    return interior, boundary


def _build_kernel(K, bounds, widths, kernel_label="k"):
    """Build a hand-rolled SDFG with one innermost K-dim map at the given bounds.

    The map's body is a no-op tasklet so the split pass has something to peel.
    """
    sdfg = dace.SDFG(f"split_K{K}")
    sdfg.add_array("A", bounds, dace.float64, transient=False)
    state = sdfg.add_state("s")
    a = state.add_access("A")
    map_dims = {f"d{p}": f"0:{bounds[p]}" for p in range(K)}
    me, mx = state.add_map(kernel_label, map_dims)
    tasklet = state.add_tasklet("body", set(), {"_out"}, "_out = 0.0")
    subset = ", ".join(f"d{p}" for p in range(K))
    state.add_memlet_path(me, tasklet, memlet=dace.Memlet())
    state.add_memlet_path(tasklet, mx, a, src_conn="_out", memlet=dace.Memlet(f"A[{subset}]"))
    return sdfg, state, me


def test_K1_non_divisible_produces_2_regions():
    """K=1 with N % W != 0 -> 1 interior + 1 boundary = 2 regions."""
    sdfg, state, me = _build_kernel(K=1, bounds=(17, ), widths=(4, ))
    SplitMapForTileRemainder(widths=(4, ), tail_mode="masked").apply_pass(sdfg, {})
    interior, boundary = _count_map_regions(sdfg, "k")
    assert interior == 1
    assert boundary == 1
    assert interior + boundary == 2


def test_K1_divisible_stays_one_region():
    """K=1 with N % W == 0 -> just the interior; no boundary (provably divisible)."""
    sdfg, state, me = _build_kernel(K=1, bounds=(16, ), widths=(4, ))
    SplitMapForTileRemainder(widths=(4, ), tail_mode="masked").apply_pass(sdfg, {})
    interior, boundary = _count_map_regions(sdfg, "k")
    assert interior == 1
    assert boundary == 0


def test_K2_both_non_divisible_produces_3_regions():
    """K=2 with non-divisible bounds on both dims -> 1 interior + 2 boundary = 3 regions
    (NOT 4 = 2^K, which would be a Cartesian corner split)."""
    sdfg, state, me = _build_kernel(K=2, bounds=(17, 13), widths=(4, 8))
    SplitMapForTileRemainder(widths=(4, 8), tail_mode="masked").apply_pass(sdfg, {})
    interior, boundary = _count_map_regions(sdfg, "k")
    assert interior == 1
    assert boundary == 2
    assert interior + boundary == 3


def test_K2_one_dim_divisible_one_not_produces_2_regions():
    """K=2 with one dim divisible, one not -> 1 interior + 1 boundary = 2 regions."""
    sdfg, state, me = _build_kernel(K=2, bounds=(16, 13), widths=(4, 8))
    SplitMapForTileRemainder(widths=(4, 8), tail_mode="masked").apply_pass(sdfg, {})
    interior, boundary = _count_map_regions(sdfg, "k")
    assert interior == 1
    assert boundary == 1


def test_K3_all_non_divisible_produces_4_regions():
    """K=3 with non-divisible bounds on all 3 dims -> 1 interior + 3 boundary = 4 regions
    (NOT 8 = 2^K)."""
    sdfg, state, me = _build_kernel(K=3, bounds=(17, 13, 11), widths=(4, 8, 4))
    SplitMapForTileRemainder(widths=(4, 8, 4), tail_mode="masked").apply_pass(sdfg, {})
    interior, boundary = _count_map_regions(sdfg, "k")
    assert interior == 1
    assert boundary == 3
    assert interior + boundary == 4


def test_K3_all_divisible_stays_one_region():
    """K=3 with all bounds divisible -> 1 interior, no boundaries."""
    sdfg, state, me = _build_kernel(K=3, bounds=(16, 16, 8), widths=(4, 8, 4))
    SplitMapForTileRemainder(widths=(4, 8, 4), tail_mode="masked").apply_pass(sdfg, {})
    interior, boundary = _count_map_regions(sdfg, "k")
    assert interior == 1
    assert boundary == 0


def test_assume_even_range_check_traps_symbolic_extent():
    """``assume_even`` peels no boundary, but a NOT-provably-divisible symbolic extent gets a
    host-side runtime guard: one ``tile_even_range_check`` state with a side-effect trap tasklet."""
    sdfg, _, _ = _build_kernel(K=1, bounds=(N, ), widths=(4, ))
    SplitMapForTileRemainder(widths=(4, ), assume_even=True).apply_pass(sdfg, {})
    interior, boundary = _count_map_regions(sdfg, "k")
    assert interior == 1 and boundary == 0, "assume_even marks the interior and peels no slab"
    guards, traps = _count_range_guards(sdfg)
    assert guards == 1 and traps == 1, f"expected one guard + trap, got {guards} guards / {traps} traps"


def test_assume_even_provably_divisible_no_trap():
    """A provably-divisible extent (``4*M % 4 == 0``) needs no runtime check -> no guard."""
    sdfg, _, _ = _build_kernel(K=1, bounds=(4 * M, ), widths=(4, ))
    SplitMapForTileRemainder(widths=(4, ), assume_even=True).apply_pass(sdfg, {})
    guards, traps = _count_range_guards(sdfg)
    assert guards == 0 and traps == 0, "a provably-even extent must not emit a runtime guard"


def test_assume_even_range_check_disabled_no_trap():
    """``range_check=False`` suppresses the guard even for a non-divisible symbolic extent."""
    sdfg, _, _ = _build_kernel(K=1, bounds=(N, ), widths=(4, ))
    SplitMapForTileRemainder(widths=(4, ), assume_even=True, range_check=False).apply_pass(sdfg, {})
    guards, traps = _count_range_guards(sdfg)
    assert guards == 0 and traps == 0, "range_check=False must not emit a guard"


def test_assume_even_range_check_aborts_at_runtime():
    """The emitted host-side guard actually traps: a non-divisible extent aborts (SIGABRT) before
    the map runs; a divisible extent completes. CPU-only (the guard is device-agnostic host code),
    and run in a throwaway process so ``abort`` does not kill pytest.

    Spawned, not forked -- an ``os.fork()`` child deadlocks on the OpenMP team this process already
    holds, see :mod:`tests.helpers.isolation`."""
    sdfg, _, _ = _build_kernel(K=1, bounds=(N, ), widths=(4, ), kernel_label="rt")
    SplitMapForTileRemainder(widths=(4, ), assume_even=True).apply_pass(sdfg, {})

    def isolated(n):
        return exit_code(sdfg, dict(A=np.zeros(n, np.float64), N=n))

    assert isolated(16) == 0, "a divisible extent must run cleanly"  # 16 % 4 == 0 -> guard passes
    code = isolated(15)  # 15 % 4 != 0 -> guard aborts (SIGABRT = 6)
    assert code == -signal.SIGABRT, \
        f"a non-divisible extent must trap via abort (SIGABRT); got child exit code {code}"


def level_loop_writing_state_array_sdfg() -> dace.SDFG:
    """CloudSC's ``zqsmix`` / ``zfoeew`` shape, over ``for k in 1:klev: for i in map(0:klon)``::

        fwd = min(a[k, i], e[k - 1, i]); e[k, i] = fwd; q[k, i] = fwd; f[k, i] = fwd
        q[k, i] = q[k, i] / (1 + f[k, i])

    then ``out[1:] = q[1:]; g[1:] = f[1:]`` in a later state. ``q`` and ``f`` are 2-D transients written
    and re-read inside the map body. ``q`` leaves the map through its exit; ``f`` is written by an
    AccessNode inside the body and reaches the later state only by its name. The ``e[k - 1]`` read keeps
    the level loop a loop.
    """
    klev, klon = dace.symbol('klev'), dace.symbol('klon')
    sdfg = dace.SDFG('level_loop_writing_state_array')
    for name in ('a', 'e', 'out', 'g'):
        sdfg.add_array(name, [klev, klon], dace.float64)
    for name in ('q', 'f'):
        sdfg.add_array(name, [klev, klon], dace.float64, transient=True)
    for name in ('fwd', 'qk', 'qj'):
        sdfg.add_array(name, [1], dace.float64, transient=True)
    sdfg.add_scalar('qn', dace.float64, transient=True)
    loop = LoopRegion('levels', 'k < klev', 'k', 'k = 1', 'k = k + 1')
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state('column_body', is_start_block=True)
    copy_out = sdfg.add_state('copy_out')
    sdfg.add_edge(loop, copy_out, dace.InterstateEdge())
    entry, exit_ = body.add_map('columns', dict(i='0:klon'))

    def assign(label: str) -> Tasklet:
        return body.add_tasklet(label, dict.fromkeys(['_in']), dict.fromkeys(['_out']), '_out = _in')

    fmin = body.add_tasklet('fmin', dict.fromkeys(['_a', '_b']), dict.fromkeys(['_o']), '_o = min(_a, _b)')
    body.add_memlet_path(body.add_read('a'), entry, fmin, dst_conn='_a', memlet=dace.Memlet('a[k, i]'))
    body.add_memlet_path(body.add_read('e'), entry, fmin, dst_conn='_b', memlet=dace.Memlet('e[k - 1, i]'))
    fwd = body.add_access('fwd')
    body.add_edge(fmin, '_o', fwd, None, dace.Memlet('fwd[0]'))
    to_e = assign('to_e')
    body.add_edge(fwd, None, to_e, '_in', dace.Memlet('fwd[0]'))
    body.add_memlet_path(to_e, exit_, body.add_write('e'), src_conn='_out', memlet=dace.Memlet('e[k, i]'))
    to_q = assign('to_q')
    body.add_edge(fwd, None, to_q, '_in', dace.Memlet('fwd[0]'))
    q_inner = body.add_access('q')
    body.add_edge(to_q, '_out', q_inner, None, dace.Memlet('q[k, i]'))
    to_f = assign('to_f')
    body.add_edge(fwd, None, to_f, '_in', dace.Memlet('fwd[0]'))
    f_inner = body.add_access('f')
    body.add_edge(to_f, '_out', f_inner, None, dace.Memlet('f[k, i]'))
    div = body.add_tasklet('div', dict.fromkeys(['_x', '_y']), dict.fromkeys(['_o']), '_o = _x / (1.0 + _y)')
    for name, conn, read_from, source in (('qk', '_x', q_inner, 'q'), ('qj', '_y', f_inner, 'f')):
        read_q = assign(f'read_{name}')
        body.add_edge(read_from, None, read_q, '_in', dace.Memlet(f'{source}[k, i]'))
        staged = body.add_access(name)
        body.add_edge(read_q, '_out', staged, None, dace.Memlet(f'{name}[0]'))
        body.add_edge(staged, None, div, conn, dace.Memlet(f'{name}[0]'))
    qn = body.add_access('qn')
    body.add_edge(div, '_o', qn, None, dace.Memlet('qn[0]'))
    store = assign('store')
    body.add_edge(qn, None, store, '_in', dace.Memlet('qn[0]'))
    body.add_memlet_path(store, exit_, body.add_write('q'), src_conn='_out', memlet=dace.Memlet('q[k, i]'))
    for source, target in (('q', 'out'), ('f', 'g')):
        copy_out.add_nedge(copy_out.add_read(source), copy_out.add_write(target),
                           dace.Memlet(data=source, subset='1:klev, 0:klon', other_subset='1:klev, 0:klon'))
    return sdfg


def test_remainder_writes_the_state_arrays_it_shares_with_later_states():
    """The remainder copy of the map writes ``q`` and ``f`` themselves, not renamed copies of them.

    Both are read after the map. Renaming ``f`` left the remainder columns of ``f`` unwritten for the later
    state (a silent wrong result). Renaming ``q`` made a 2-D transient only the remainder used; nesting kept
    it inside the body, the tile passes left its ``(klev, klon)`` shape alone, and the lowered copy from the
    ``(8,)`` tile into it failed validation. A temporary only the map uses is still renamed.
    """
    sdfg = level_loop_writing_state_array_sdfg()
    SplitMapForTileRemainder(widths=(8, ), tail_mode="masked").apply_pass(sdfg, {})
    interior, boundary = _count_map_regions(sdfg, "columns")
    assert (interior, boundary) == (1, 1)
    two_dim_transients = [name for name, desc in sdfg.arrays.items() if desc.transient and len(desc.shape) == 2]
    assert two_dim_transients == ['q',
                                  'f'], f"the remainder must write q and f, not renamed copies: {two_dim_transients}"
    assert 'fwd_0' in sdfg.arrays, "the map-local temporary fwd must still get its own name in the remainder"


def test_level_loop_writing_state_array_matches_numpy_with_a_remainder():
    """``klon = 13`` leaves a 5-column remainder at width 8; the vectorized kernel compiles and matches NumPy."""
    sdfg = level_loop_writing_state_array_sdfg()
    untransformed = copy.deepcopy(sdfg)
    VectorizeCPUMultiDim(VectorizeConfig(widths=(8, ), target_isa=detect_host_isa(),
                                         validate=True)).apply_pass(sdfg, {})
    assert_tiled(sdfg, untransformed, 'level_loop_writing_state_array')

    klev, klon = 4, 13
    rng = np.random.default_rng(7)
    a = rng.random((klev, klon))
    e = rng.random((klev, klon))
    out = np.zeros((klev, klon))
    g = np.zeros((klev, klon))
    expected_e = e.copy()
    expected_out = np.zeros((klev, klon))
    expected_g = np.zeros((klev, klon))
    for k in range(1, klev):
        expected_e[k] = np.minimum(a[k], expected_e[k - 1])
        expected_g[k] = expected_e[k]
        expected_out[k] = expected_e[k] / (1.0 + expected_e[k])

    sdfg.compile()(a=a, e=e, out=out, g=g, klev=klev, klon=klon)
    np.testing.assert_allclose(e, expected_e, rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(g, expected_g, rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(out, expected_out, rtol=1e-14, atol=0.0)


if __name__ == "__main__":
    test_K1_non_divisible_produces_2_regions()
    test_K1_divisible_stays_one_region()
    test_K2_both_non_divisible_produces_3_regions()
    test_K2_one_dim_divisible_one_not_produces_2_regions()
    test_K3_all_non_divisible_produces_4_regions()
    test_K3_all_divisible_stays_one_region()
    test_assume_even_range_check_traps_symbolic_extent()
    test_assume_even_provably_divisible_no_trap()
    test_assume_even_range_check_disabled_no_trap()
    test_assume_even_range_check_aborts_at_runtime()
    test_remainder_writes_the_state_arrays_it_shares_with_later_states()
    test_level_loop_writing_state_array_matches_numpy_with_a_remainder()
