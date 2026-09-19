# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Canonicalize on an indirect ACCUMULATE whose index array repeats: ``bins[ip[i]] += src[i]``.

The scatter guard exists for a plain indirect WRITE, where two iterations landing on one slot
race and the surviving value is undefined; it proves the index is a permutation at run time and
``std::abort()``s when it is not. An accumulate is a different program. Colliding iterations fold
into the slot through an atomic combine, so any ``ip`` at all is correct, and asking it for a
permutation aborts on the inputs the kernel exists to handle -- a histogram, a sparse
scatter-add, a particle deposition all repeat indices by construction.

The trap is content-driven, not size-driven: ONE repeated entry is enough, at any length. The
tests below pin the value against a numpy oracle (the thing a structural assertion cannot see),
and pin the structure so a future guard cannot creep back in: an accumulate must reach codegen as
a WCR write with no tag buffer and no trap.

The last two tests pin the MECHANISM rather than this kernel. What cost the WCR was not the
scatter at all -- it was two interstate symbols carrying one address, which makes
``AugAssignToWCR``'s syntactic same-slot test answer "different slots". Any pass that mints a
second name for an expression that already has one re-creates that, so the invariant is stated
directly ("two names for one address must still become a WCR") together with the idempotence that
lets ``SymbolDedup`` sit at every stage boundary instead of one chosen point.

Results here are ``allclose``, NOT bit-exact: an atomic fold reassociates the additions, so two
runs of the same input can differ in the last ulp (measured 1.2e-15 relative, 16 threads).
"""
import numpy as np

import dace
from dace import symbolic
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize.symbol_dedup import SymbolDedup
from dace.transformation.passes.scatter_to_guarded_maps import (detect_scatter_idx_arrays,
                                                                indirect_write_needs_injectivity)

#: Every descriptor the conflict guard allocates shares this infix (tag buffer + count scalar).
GUARD_INFIX = 'scatter_guard'

LEN_1D = dace.symbol('LEN_1D', dtype=dace.int64, positive=True)


@dace.program
def _scatter_accum(bins: dace.float64[LEN_1D], src: dace.float64[LEN_1D], ip: dace.int32[LEN_1D]):
    for i in range(LEN_1D):
        bins[ip[i]] = bins[ip[i]] + src[i]


@dace.program
def _scatter_store(dst: dace.float64[LEN_1D], src: dace.float64[LEN_1D], ip: dace.int32[LEN_1D]):
    for i in range(LEN_1D):
        dst[ip[i]] = src[i]


def _oracle(bins: np.ndarray, src: np.ndarray, ip: np.ndarray) -> np.ndarray:
    out = bins.copy()
    for i in range(ip.size):
        out[ip[i]] += src[i]
    return out


def _inputs(n: int, duplicates: bool):
    rng = np.random.default_rng(20260902)
    bins = rng.uniform(1.0, 1000.0, n)
    src = rng.uniform(0.5, 2.0, n)
    ip = rng.integers(0, n, n, dtype=np.int32) if duplicates else rng.permutation(n).astype(np.int32)
    return bins, src, ip


def _canonicalized():
    sdfg = _scatter_accum.to_sdfg(simplify=False)
    canonicalize(sdfg, target='cpu')
    return sdfg


def _guard_descriptors(sdfg: dace.SDFG):
    return sorted(name for sd in sdfg.all_sdfgs_recursive() for name in sd.arrays if GUARD_INFIX in name)


def test_duplicate_indices_run_and_match_oracle():
    """The kernel the guard used to abort on: eight repeats in a 1024-element index.

    ``std::abort()`` inside the compiled kernel is a SIGABRT that takes the pytest process with
    it, so a regression here shows up as a crashed worker rather than a failed assertion.
    """
    bins, src, ip = _inputs(1024, duplicates=True)
    assert np.unique(ip).size < ip.size, 'the draw must repeat an index or the test proves nothing'
    got = bins.copy()
    _canonicalized()(bins=got, src=src.copy(), ip=ip.copy(), LEN_1D=1024)
    assert np.allclose(got, _oracle(bins, src, ip))


def test_single_duplicate_is_enough():
    """One repeated entry at a tiny length: the trap keys on the index CONTENT, not the size."""
    n = 8
    bins = np.arange(1.0, n + 1.0)
    src = np.full(n, 0.5)
    ip = np.arange(n, dtype=np.int32)
    ip[n - 1] = 0
    got = bins.copy()
    _canonicalized()(bins=got, src=src.copy(), ip=ip.copy(), LEN_1D=n)
    assert np.allclose(got, _oracle(bins, src, ip))


def test_permutation_indices_still_match_oracle():
    """The case that survived the guard keeps working -- the fix is not a special case for dups."""
    bins, src, ip = _inputs(1024, duplicates=False)
    got = bins.copy()
    _canonicalized()(bins=got, src=src.copy(), ip=ip.copy(), LEN_1D=1024)
    assert np.allclose(got, _oracle(bins, src, ip))


def test_accumulate_lifts_to_a_wcr_map_without_a_guard():
    """Structure: a parallel Map whose indirect write carries the WCR, no tag buffer, no loop.

    Without the WCR the same Map is a write-write race that only the removed trap concealed, so
    the WCR assertion is the one that keeps the parallelization honest.
    """
    sdfg = _canonicalized()
    assert not _guard_descriptors(sdfg), 'an accumulate must not allocate a conflict-check buffer'
    assert not [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]
    assert [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]
    wcr_writes = [
        e.data.wcr for sd in sdfg.all_sdfgs_recursive() for st in sd.all_states() for e in st.edges()
        if e.data is not None and e.data.data == 'bins' and e.data.wcr is not None
    ]
    assert wcr_writes, 'the indirect accumulate must reach codegen as a WCR write'


def test_plain_indirect_store_is_still_guarded():
    """The guard is not disabled: a plain ``dst[ip[i]] = src[i]`` still needs its permutation."""
    sdfg = _scatter_store.to_sdfg(simplify=False)
    canonicalize(sdfg, target='cpu')
    assert _guard_descriptors(sdfg), 'a plain indirect store must keep its conflict check'


def test_detector_ignores_accumulating_writes():
    """Unit-level mirror of the two tests above, on the detector itself."""
    assert detect_scatter_idx_arrays(_scatter_store.to_sdfg(simplify=True)) == {'ip'}
    assert indirect_write_needs_injectivity(dace.Memlet(data='bins', subset='0')) is True
    assert indirect_write_needs_injectivity(dace.Memlet(data='bins', subset='0', wcr='lambda a, b: a + b')) is False


def _alias_the_store_subscript(sdfg: dace.SDFG) -> str:
    """Give the indirect STORE its own symbol name for the address the LOAD already has one for.

    The frontend happens to do this by itself today (``ip_index`` for the load, ``bins_slice`` for
    the store, both bound to ``ip[i]`` on one edge), but the invariant under test is not "the
    frontend emits two names" -- it is "two names for one address must not cost the WCR". Minting
    the alias here states the premise outright, so the test keeps testing the mechanism even if
    the frontend stops producing a duplicate on its own.

    :param sdfg: An unsimplified or simplified frontend SDFG.
    :returns: The alias symbol name.
    """
    for sd in sdfg.all_sdfgs_recursive():
        for region in sd.all_control_flow_regions():
            if not isinstance(region, LoopRegion):
                continue
            for iedge in region.edges():
                for sym, rhs in list(iedge.data.assignments.items()):
                    for state in region.all_states():
                        for edge in state.edges():
                            if not isinstance(edge.dst, nodes.AccessNode) or edge.dst.data != 'bins':
                                continue
                            subset = edge.data.dst_subset if edge.data.dst_subset is not None else edge.data.subset
                            if subset is None or sym not in {str(s) for s in subset.free_symbols}:
                                continue
                            alias = f'_alias_{sym}'
                            dtype = sd.symbols[sym]
                            iedge.data.assignments[alias] = rhs
                            sd.add_symbol(alias, dtype, find_new_name=False)
                            # A dace ``symbol``, never a bare string: ``Subset.free_symbols`` counts
                            # only dace symbols, so a raw sympy Symbol (what sympifying a str gives)
                            # is invisible to it and every later rename silently skips this subset.
                            edge.data.replace({sym: symbolic.symbol(alias, dtype)})
                            return alias
    raise AssertionError('no indirect store into bins to alias -- the premise of this test is gone')


def test_two_names_for_one_address_still_becomes_a_wcr():
    """The MECHANISM, not the symptom: an accumulate must not lose its WCR to a renamed address.

    ``AugAssignToWCR`` decides "same slot" by comparing the load and store subsets syntactically,
    so a second name for one address reads as a second location and the accumulate stays a plain
    write -- which the scatter stage then guards, and the guard aborts. Any future pass that mints
    a duplicate symbol re-creates that, so the invariant is pinned here rather than at the one
    producer that happened to trip it.
    """
    sdfg = _scatter_accum.to_sdfg(simplify=True)
    alias = _alias_the_store_subscript(sdfg)
    assert alias in {s for sd in sdfg.all_sdfgs_recursive() for s in sd.symbols}
    canonicalize(sdfg, target='cpu')
    assert not _guard_descriptors(sdfg)
    assert [
        e for sd in sdfg.all_sdfgs_recursive() for st in sd.all_states() for e in st.edges()
        if e.data is not None and e.data.data == 'bins' and e.data.wcr is not None
    ], 'the accumulate lost its WCR to the aliased address'


def test_symbol_dedup_is_idempotent():
    """A second consecutive run must merge nothing and leave the SDFG bit-identical.

    This is what makes ``SymbolDedup`` free to place at every stage boundary rather than at one
    chosen point; without it, repeating the pass would be a source of churn.
    """
    sdfg = _scatter_accum.to_sdfg(simplify=True)
    _alias_the_store_subscript(sdfg)
    first = SymbolDedup().apply_pass(sdfg, {})
    assert first, 'the aliased address must give SymbolDedup something to merge'
    settled = sdfg.to_json()
    assert SymbolDedup().apply_pass(sdfg, {}) is None
    assert sdfg.to_json() == settled
