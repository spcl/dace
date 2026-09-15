# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`~dace.libraries.standard.nodes.scan.Scan`.

Covers inclusive / exclusive scans, all supported ops (``SUM``, ``PRODUCT``,
``MIN``, ``MAX``), several integer and floating dtypes, and the ``CPU`` /
``pure`` implementations. ``CUDA`` is not exercised in this CPU-only file; it
would need a GPU test environment.
"""
import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes.scan import (Scan, ScanOp, INPUT_CONNECTOR_NAME, OUTPUT_CONNECTOR_NAME, in_connector,
                                                init_connector, out_connector)


def _safe_label(s: str) -> str:
    return s.replace(':', '_').replace(' ', '_')


def _build_scan_sdfg(dace_dtype: dace.dtypes.typeclass,
                     n: int,
                     op: ScanOp,
                     exclusive: bool,
                     implementation: str,
                     identity=None) -> dace.SDFG:
    """Build a single-state SDFG that scans ``arr_in[0:N]`` into ``arr_out[0:N]``."""
    name = f"scan_{op.value}_{int(exclusive)}_{implementation}_{_safe_label(dace_dtype.to_string())}_{n}"
    sdfg = dace.SDFG(name)
    sdfg.add_array('arr_in', [n], dace_dtype)
    sdfg.add_array('arr_out', [n], dace_dtype)
    state = sdfg.add_state('scan')
    a_in = state.add_read('arr_in')
    a_out = state.add_write('arr_out')
    node = Scan('Scan', op=op, exclusive=exclusive, identity=identity)
    node.implementation = implementation
    state.add_node(node)
    state.add_edge(a_in, None, node, INPUT_CONNECTOR_NAME, dace.Memlet(f'arr_in[0:{n}]'))
    state.add_edge(node, OUTPUT_CONNECTOR_NAME, a_out, None, dace.Memlet(f'arr_out[0:{n}]'))
    sdfg.validate()
    return sdfg


def _numpy_inclusive(arr, op: ScanOp):
    if op is ScanOp.SUM:
        return np.cumsum(arr)
    if op is ScanOp.PRODUCT:
        return np.cumprod(arr)
    if op is ScanOp.MIN:
        return np.minimum.accumulate(arr)
    if op is ScanOp.MAX:
        return np.maximum.accumulate(arr)
    raise AssertionError(f'Unknown op: {op}')


def _numpy_exclusive(arr, op: ScanOp, identity):
    """Exclusive scan: out[0] = identity, out[k] = identity OP in[0] OP ... OP in[k-1]."""
    out = np.empty_like(arr)
    out[0] = identity
    if len(arr) == 1:
        return out
    incl = _numpy_inclusive(arr, op)
    out[1:] = incl[:-1] if op is ScanOp.SUM else incl[:-1]
    if op is ScanOp.SUM:
        out[1:] = identity + incl[:-1]
    elif op is ScanOp.PRODUCT:
        out[1:] = identity * incl[:-1]
    elif op is ScanOp.MIN:
        out[1:] = np.minimum(identity, incl[:-1])
    elif op is ScanOp.MAX:
        out[1:] = np.maximum(identity, incl[:-1])
    return out


@pytest.mark.parametrize('implementation', ['CPU', 'pure'])
@pytest.mark.parametrize('op', [ScanOp.SUM, ScanOp.PRODUCT, ScanOp.MIN, ScanOp.MAX])
def test_scan_inclusive_matches_numpy(op: ScanOp, implementation: str):
    """Inclusive scan over float64 matches numpy's cum* / accumulate."""
    n = 64
    rng = np.random.default_rng(int(op.value.encode().hex(), 16) & 0xFFFF)
    if op is ScanOp.PRODUCT:
        arr_in = rng.uniform(0.95, 1.05, size=n)  # keep magnitudes finite
    else:
        arr_in = rng.uniform(-1.0, 1.0, size=n)
    arr_out = np.zeros_like(arr_in)
    sdfg = _build_scan_sdfg(dace.float64, n, op, exclusive=False, implementation=implementation)
    sdfg(arr_in=arr_in.copy(), arr_out=arr_out)
    expected = _numpy_inclusive(arr_in, op).astype(np.float64)
    assert np.allclose(arr_out, expected), (f'{op.value} inclusive scan mismatch on {implementation}; '
                                            f'max diff {np.max(np.abs(arr_out - expected))}.')


@pytest.mark.parametrize('implementation', ['CPU', 'pure'])
def test_scan_inclusive_sum_int32(implementation: str):
    """Integer dtype inclusive sum scan -- exact equality, no floating tolerance."""
    n = 50
    arr_in = np.arange(1, n + 1, dtype=np.int32)
    arr_out = np.zeros(n, dtype=np.int32)
    sdfg = _build_scan_sdfg(dace.int32, n, ScanOp.SUM, exclusive=False, implementation=implementation)
    sdfg(arr_in=arr_in.copy(), arr_out=arr_out)
    assert np.array_equal(arr_out, np.cumsum(arr_in))


@pytest.mark.parametrize('implementation', ['CPU', 'pure'])
def test_scan_exclusive_sum_with_seed(implementation: str):
    """Exclusive sum scan with a non-zero seed identity -- prefix-then-shift semantics."""
    n = 32
    arr_in = np.arange(1, n + 1, dtype=np.float64)
    arr_out = np.zeros(n, dtype=np.float64)
    seed = 10.0
    sdfg = _build_scan_sdfg(dace.float64, n, ScanOp.SUM, exclusive=True, implementation=implementation, identity=seed)
    sdfg(arr_in=arr_in.copy(), arr_out=arr_out)
    expected = _numpy_exclusive(arr_in, ScanOp.SUM, seed)
    assert np.allclose(arr_out, expected)


@pytest.mark.parametrize('implementation', ['CPU', 'pure'])
def test_scan_single_element(implementation: str):
    """A length-1 inclusive scan returns the single element; exclusive returns the seed.
    The libnode expansion degenerates to a copy / identity tasklet for this case."""
    arr_in = np.array([3.5], dtype=np.float64)
    arr_out = np.zeros(1, dtype=np.float64)
    inc_sdfg = _build_scan_sdfg(dace.float64, 1, ScanOp.SUM, exclusive=False, implementation=implementation)
    inc_sdfg(arr_in=arr_in.copy(), arr_out=arr_out)
    assert arr_out[0] == arr_in[0]

    arr_out = np.zeros(1, dtype=np.float64)
    exc_sdfg = _build_scan_sdfg(dace.float64,
                                1,
                                ScanOp.SUM,
                                exclusive=True,
                                implementation=implementation,
                                identity=7.0)
    exc_sdfg(arr_in=arr_in.copy(), arr_out=arr_out)
    assert arr_out[0] == 7.0


@pytest.mark.parametrize('implementation', ['CPU', 'pure'])
def test_scan_two_elements(implementation: str):
    """A length-2 inclusive scan: ``out[0] = in[0], out[1] = in[0] + in[1]`` (smallest
    non-degenerate input). Scans of length 1 are a degenerate case the libnode does
    not promise to support -- the contract is "scan an *array*", and single-element
    subsets are silently inferred as scalars by the codegen which conflicts with the
    libnode's iterator-based call shape."""
    arr_in = np.array([3.5, 2.0], dtype=np.float64)
    arr_out = np.zeros(2, dtype=np.float64)
    inc_sdfg = _build_scan_sdfg(dace.float64, 2, ScanOp.SUM, exclusive=False, implementation=implementation)
    inc_sdfg(arr_in=arr_in.copy(), arr_out=arr_out)
    assert np.allclose(arr_out, [3.5, 5.5])

    arr_out = np.zeros(2, dtype=np.float64)
    exc_sdfg = _build_scan_sdfg(dace.float64,
                                2,
                                ScanOp.SUM,
                                exclusive=True,
                                implementation=implementation,
                                identity=7.0)
    exc_sdfg(arr_in=arr_in.copy(), arr_out=arr_out)
    assert np.allclose(arr_out, [7.0, 10.5])


def _build_multi_chain_sdfg(n: int,
                            chains: int,
                            op: ScanOp,
                            implementation: str,
                            seeded: bool,
                            dtype: dace.dtypes.typeclass = dace.float64) -> dace.SDFG:
    """One ``Scan`` carrying ``chains`` independent scans over row ``c`` of a 2-D
    in/out pair, optionally seeded from ``seed[c]`` through ``_scan_init_c``."""
    sdfg = dace.SDFG(f'multi_{op.value}_{chains}_{implementation}_{int(seeded)}_'
                     f'{_safe_label(dtype.to_string())}_{n}')
    sdfg.add_array('arr_in', [chains, n], dtype)
    sdfg.add_array('arr_out', [chains, n], dtype)
    if seeded:
        sdfg.add_array('seed', [chains], dtype)
    state = sdfg.add_state('scan')
    node = Scan('Scan', op=op, chains=chains)
    node.implementation = implementation
    state.add_node(node)
    a_out = state.add_write('arr_out')
    for c in range(chains):
        state.add_edge(state.add_read('arr_in'), None, node, in_connector(c), dace.Memlet(f'arr_in[{c}, 0:{n}]'))
        state.add_edge(node, out_connector(c), a_out, None, dace.Memlet(f'arr_out[{c}, 0:{n}]'))
        if seeded:
            node.add_in_connector(init_connector(c))
            state.add_edge(state.add_read('seed'), None, node, init_connector(c), dace.Memlet(f'seed[{c}]'))
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize('implementation', ['CPU', 'pure'])
@pytest.mark.parametrize('op', [ScanOp.SUM, ScanOp.PRODUCT, ScanOp.MIN, ScanOp.MAX])
@pytest.mark.parametrize('seeded', [False, True])
def test_multi_chain_scan_matches_numpy(op: ScanOp, implementation: str, seeded: bool):
    """``chains > 1``: K independent scans over one index range. Every chain must
    reproduce its own single-chain result, and no chain may leak into another."""
    n, chains = 4096, 5
    rng = np.random.default_rng(0xC0FFEE + int(seeded))
    arr_in = rng.uniform(0.98, 1.02, (chains, n)) if op is ScanOp.PRODUCT else rng.uniform(-1.0, 1.0, (chains, n))
    seed = rng.uniform(0.5, 1.5, chains)
    arr_out = np.zeros_like(arr_in)
    sdfg = _build_multi_chain_sdfg(n, chains, op, implementation, seeded)
    kwargs = {'seed': seed.copy()} if seeded else {}
    sdfg(arr_in=arr_in.copy(), arr_out=arr_out, **kwargs)
    for c in range(chains):
        row = np.concatenate(([seed[c]], arr_in[c])) if seeded else arr_in[c]
        expected = _numpy_inclusive(row, op)[1:] if seeded else _numpy_inclusive(row, op)
        assert np.allclose(arr_out[c], expected, rtol=1e-12,
                           atol=1e-12), (f'{op.value} chain {c} mismatch on {implementation} (seeded={seeded}); '
                                         f'max diff {np.max(np.abs(arr_out[c] - expected))}.')


def test_multi_chain_scan_opens_one_parallel_region():
    """The point of ``chains > 1``: ONE fork/join and ONE ``omp scan`` directive
    listing every accumulator, instead of one region per chain."""
    import re
    sdfg = _build_multi_chain_sdfg(256, 4, ScanOp.SUM, 'CPU', seeded=False)
    src = "\n".join(c.clean_code for c in sdfg.generate_code())
    assert len(re.findall(r'#\s*pragma\s+omp\s+parallel', src)) == 1, src
    items = re.findall(r'#\s*pragma\s+omp\s+scan\s+inclusive\(([^)]*)\)', src)
    assert len(items) == 1 and len(items[0].split(',')) == 4, items


def test_multi_chain_scan_refuses_unequal_chain_lengths():
    """Chains share one loop, so an in-edge spanning a different element count is
    rejected at validation rather than silently scanning past the end."""
    sdfg = dace.SDFG('multi_unequal')
    sdfg.add_array('arr_in', [2, 16], dace.float64)
    sdfg.add_array('arr_out', [2, 16], dace.float64)
    state = sdfg.add_state('scan')
    node = Scan('Scan', op=ScanOp.SUM, chains=2)
    state.add_node(node)
    a_out = state.add_write('arr_out')
    for c, span in enumerate((16, 8)):
        state.add_edge(state.add_read('arr_in'), None, node, in_connector(c), dace.Memlet(f'arr_in[{c}, 0:{span}]'))
        state.add_edge(node, out_connector(c), a_out, None, dace.Memlet(f'arr_out[{c}, 0:{span}]'))
    with pytest.raises(Exception):
        sdfg.validate()


def test_scan_refuses_dtype_mismatch():
    """Validation rejects an output array whose dtype differs from the input's."""
    sdfg = dace.SDFG('scan_refuses_mismatch')
    sdfg.add_array('arr_in', [8], dace.float64)
    sdfg.add_array('arr_out', [8], dace.float32)
    state = sdfg.add_state('scan')
    a_in = state.add_read('arr_in')
    a_out = state.add_write('arr_out')
    node = Scan('Scan', op=ScanOp.SUM)
    state.add_node(node)
    state.add_edge(a_in, None, node, INPUT_CONNECTOR_NAME, dace.Memlet('arr_in[0:8]'))
    state.add_edge(node, OUTPUT_CONNECTOR_NAME, a_out, None, dace.Memlet('arr_out[0:8]'))
    with pytest.raises(Exception):
        sdfg.compile()


#: Big enough that a realistic team splits it into several per-thread blocks, each
#: still inside one tile. The sizes above (32 - 4096) leave most of the team with an
#: empty block, so they never exercise the cross-thread prefix; these two put the
#: per-thread block folds, the cross-thread offset prefix and the barriers under test.
_MULTI_BLOCK_N = 1 << 14
#: Past one tile (``TILE_BYTES`` per thread), so the tile loop runs several times
#: and the carry from one tile to the next is exercised as well.
_MULTI_TILE_N = 300000


@pytest.mark.parametrize('n', [_MULTI_BLOCK_N, _MULTI_TILE_N])
@pytest.mark.parametrize('op', [ScanOp.SUM, ScanOp.MIN, ScanOp.MAX])
def test_blocked_parallel_scan_is_exact_on_integers(op: ScanOp, n: int):
    """Above the parallel threshold the CPU scan splits into per-thread blocks and
    tiles. On int64 the block fold, the offset prefix and the tile carry are all
    exact, so a misplaced block boundary shows up as an inequality instead of
    hiding inside a floating-point tolerance."""
    rng = np.random.default_rng(0x51CA11)
    arr_in = rng.integers(-1000, 1000, size=n, dtype=np.int64)
    arr_out = np.zeros(n, dtype=np.int64)
    sdfg = _build_scan_sdfg(dace.int64, n, op, exclusive=False, implementation='CPU')
    sdfg(arr_in=arr_in.copy(), arr_out=arr_out)
    assert np.array_equal(arr_out, _numpy_inclusive(arr_in, op))


@pytest.mark.parametrize('n', [_MULTI_BLOCK_N, _MULTI_TILE_N])
def test_blocked_parallel_exclusive_scan_is_exact_on_integers(n: int):
    """The exclusive shape shifts inside each block, so its block boundaries are a
    separate risk from the inclusive one's; int64 pins them exactly."""
    rng = np.random.default_rng(0x51CA12)
    arr_in = rng.integers(-1000, 1000, size=n, dtype=np.int64)
    arr_out = np.zeros(n, dtype=np.int64)
    sdfg = _build_scan_sdfg(dace.int64, n, ScanOp.SUM, exclusive=True, implementation='CPU', identity=17)
    sdfg(arr_in=arr_in.copy(), arr_out=arr_out)
    assert np.array_equal(arr_out, _numpy_exclusive(arr_in, ScanOp.SUM, 17))


@pytest.mark.parametrize('op', [ScanOp.SUM, ScanOp.PRODUCT, ScanOp.MIN, ScanOp.MAX])
def test_blocked_parallel_scan_matches_numpy_float64(op: ScanOp):
    """Same shape in float64. Blocking reassociates within the scan's own op, so the
    gate is scale-relative rather than bit-exact -- but it shortens the dependent
    chains, so it does not lose accuracy against the sequential fold."""
    n = _MULTI_TILE_N
    rng = np.random.default_rng(0xB10C)
    arr_in = rng.uniform(1.0 - 1e-6, 1.0 + 1e-6, n) if op is ScanOp.PRODUCT else rng.uniform(0.5, 1.5, n)
    arr_out = np.zeros_like(arr_in)
    sdfg = _build_scan_sdfg(dace.float64, n, op, exclusive=False, implementation='CPU')
    sdfg(arr_in=arr_in.copy(), arr_out=arr_out)
    expected = _numpy_inclusive(arr_in, op)
    worst = np.max(np.abs(arr_out - expected)) / np.max(np.abs(expected))
    assert worst <= 1e-12, f'{op.value} blocked scan drifted {worst:.3e} scale-relative.'


@pytest.mark.parametrize('op', [ScanOp.SUM, ScanOp.MIN, ScanOp.MAX])
def test_blocked_parallel_seeded_scan_matches_numpy(op: ScanOp):
    """A wired ``_scan_init`` takes the SAME blocked parallel path as an unseeded
    scan. It used to fall back to a sequential ``std::inclusive_scan``, so this is
    the size at which that fallback was costing the whole speedup."""
    n = _MULTI_TILE_N
    rng = np.random.default_rng(0x5EED)
    arr_in = rng.uniform(0.5, 1.5, (1, n))
    seed = rng.uniform(0.5, 1.5, 1)
    arr_out = np.zeros_like(arr_in)
    sdfg = _build_multi_chain_sdfg(n, 1, op, 'CPU', seeded=True)
    sdfg(arr_in=arr_in.copy(), arr_out=arr_out, seed=seed.copy())
    expected = _numpy_inclusive(np.concatenate(([seed[0]], arr_in[0])), op)[1:]
    worst = np.max(np.abs(arr_out[0] - expected)) / np.max(np.abs(expected))
    assert worst <= 1e-12, f'{op.value} seeded blocked scan drifted {worst:.3e} scale-relative.'


@pytest.mark.parametrize('seeded', [False, True])
@pytest.mark.parametrize('op', [ScanOp.SUM, ScanOp.MIN, ScanOp.MAX])
def test_multi_chain_blocked_parallel_matches_numpy(op: ScanOp, seeded: bool):
    """``chains > 1`` above the parallel threshold: the blocked shape carries K block
    totals per thread, so it has K ways to leak one chain's offset into another."""
    n, chains = _MULTI_TILE_N, 3
    rng = np.random.default_rng(0xC0FFEE + int(seeded))
    arr_in = rng.uniform(0.5, 1.5, (chains, n))
    seed = rng.uniform(0.5, 1.5, chains)
    arr_out = np.zeros_like(arr_in)
    sdfg = _build_multi_chain_sdfg(n, chains, op, 'CPU', seeded)
    sdfg(arr_in=arr_in.copy(), arr_out=arr_out, **({'seed': seed.copy()} if seeded else {}))
    for c in range(chains):
        row = np.concatenate(([seed[c]], arr_in[c])) if seeded else arr_in[c]
        expected = _numpy_inclusive(row, op)[1:] if seeded else _numpy_inclusive(row, op)
        worst = np.max(np.abs(arr_out[c] - expected)) / np.max(np.abs(expected))
        assert worst <= 1e-12, f'{op.value} chain {c} drifted {worst:.3e} scale-relative (seeded={seeded}).'


@pytest.mark.parametrize('n', [_MULTI_BLOCK_N, _MULTI_TILE_N])
@pytest.mark.parametrize('chains', [2, 5])
@pytest.mark.parametrize('seeded', [False, True])
def test_multi_chain_blocked_parallel_is_exact_on_integers(chains: int, n: int, seeded: bool):
    """The multi-chain blocked path, pinned EXACTLY rather than to a tolerance.

    ``chains > 1`` publishes K block totals per thread and walks a K-wide offset prefix
    over them, so a chain that seeds from a neighbour's total, a carry that only
    propagates chain 0, or a block boundary off by one are all reachable bugs -- and on
    float64 they hide under the reassociation the blocking is allowed to do. On int64
    every phase is exact, so the sequential result is the ONLY correct answer.

    Five chains is the widest the corpus asks for (TSVC-2.5 ``scan_multi_5carry``); the
    tile cap is per chain, so K also decides how many tiles the ``__base`` loop takes and
    thus whether the tile-to-tile carry runs at all.
    """
    rng = np.random.default_rng(0x51CA13 + chains)
    arr_in = rng.integers(-1000, 1000, size=(chains, n), dtype=np.int64)
    seed = rng.integers(-50, 50, size=chains, dtype=np.int64)
    arr_out = np.zeros_like(arr_in)
    sdfg = _build_multi_chain_sdfg(n, chains, ScanOp.SUM, 'CPU', seeded, dace.int64)
    sdfg(arr_in=arr_in.copy(), arr_out=arr_out, **({'seed': seed.copy()} if seeded else {}))
    expected = np.cumsum(arr_in, axis=1) + (seed[:, None] if seeded else 0)
    for c in range(chains):
        assert np.array_equal(
            arr_out[c], expected[c]), (f'chain {c} of {chains} diverged at n={n} (seeded={seeded}); first wrong index '
                                       f'{int(np.argmax(arr_out[c] != expected[c]))}.')


@pytest.mark.parametrize('dtype', [dace.complex64, dace.complex128])
@pytest.mark.parametrize('op', [ScanOp.SUM, ScanOp.PRODUCT])
def test_multi_chain_complex_scan_matches_numpy(op: ScanOp, dtype):
    """Complex element types on the multi-chain parallel path.

    OpenMP has no built-in reduction for a class type, and a user-defined one is found by
    unqualified lookup from the point of use -- so ``dace/scan.hpp``'s declarations, which
    live in ``dace::scan::detail``, are invisible to the pragmas this expansion splices
    into the generated program. Every complex chain past the first failed to compile with
    "user defined reduction not found"; the tasklet now carries its own block-scope
    declaration. Single-chain complex went through the header and always worked, so K=1 is
    not a substitute for this.

    The inputs are exactly representable (integer real/imaginary parts, unit-modulus
    phases for the product), so blocking cannot move the answer and the gate is equality.
    """
    n, chains = _MULTI_TILE_N, 3
    npdt = dtype.as_numpy_dtype()
    rng = np.random.default_rng(0xC0DE + int(op is ScanOp.SUM))
    if op is ScanOp.PRODUCT:
        arr_in = np.array([1, 1j, -1, -1j], dtype=npdt)[rng.integers(0, 4, size=(chains, n))]
        expected = np.multiply.accumulate(arr_in.astype(np.complex128), axis=1).astype(npdt)
    else:
        arr_in = (rng.integers(-100, 100, (chains, n)) + 1j * rng.integers(-100, 100, (chains, n))).astype(npdt)
        expected = np.add.accumulate(arr_in.astype(np.complex128), axis=1).astype(npdt)
    arr_out = np.zeros_like(arr_in)
    sdfg = _build_multi_chain_sdfg(n, chains, op, 'CPU', False, dtype)
    sdfg(arr_in=arr_in.copy(), arr_out=arr_out)
    for c in range(chains):
        assert np.array_equal(arr_out[c],
                              expected[c]), (f'{dtype.to_string()} {op.value} chain {c} diverged; '
                                             f'first wrong index {int(np.argmax(arr_out[c] != expected[c]))}.')


def test_seeded_scan_calls_the_parallel_runtime_entry_point():
    """Regression guard: the seeded CPU scan emits the blocked runtime overload, not
    the sequential ``std::inclusive_scan`` it used to fall back to even when the
    node was scheduled in parallel."""
    import re
    sdfg = _build_multi_chain_sdfg(1024, 1, ScanOp.SUM, 'CPU', seeded=True)
    src = "\n".join(c.clean_code for c in sdfg.generate_code())
    # The codegen substitutes the connectors, so the seed shows up as the ``seed``
    # array rather than as ``_scan_init``; what matters is that it is an argument of
    # the runtime call and not the initial value of a sequential ``<numeric>`` one.
    assert re.search(r'::dace::scan::inclusive_sum\([^;]*\bseed\b[^;]*\);', src), src
    assert 'std::inclusive_scan' not in src, src


def test_cpu_scan_does_not_emit_the_composite_inscan_loop():
    """``#pragma omp parallel for simd reduction(inscan, ...)`` is the shape GCC
    lowers to a per-thread scratch ``malloc`` plus a scratch-to-``out`` second pass,
    which measured slower than a sequential scan at every size. Both the
    single-chain and the multi-chain expansions must emit the hand-blocked region
    instead."""
    for chains in (1, 3):
        sdfg = _build_multi_chain_sdfg(256, chains, ScanOp.SUM, 'CPU', seeded=False)
        src = "\n".join(c.clean_code for c in sdfg.generate_code())
        assert 'parallel for simd' not in src, src


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
