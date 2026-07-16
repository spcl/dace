# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for the ``OutlineTopLevelNests`` pass: each top-level loop nest of the root SDFG becomes its
    own ``no_inline`` nested SDFG, structurally loss-free and codegen-agnostic. """
import copy

import numpy
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.utils import inline_sdfgs
from dace.transformation.passes.outline_top_level_nests import OutlineTopLevelNests, outline_top_level_nests

N = dace.symbol('N')


@dace.program
def two_maps(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N], D: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] + B[i]
    for i in dace.map[0:N]:
        D[i] = C[i] * 2.0


@dace.program
def seq_loop(A: dace.float64[N], B: dace.float64[N]):
    # Carried dependency keeps this a sequential LoopRegion (exercises the nest_sdfg_subgraph path
    # and the loop-symbol pre-declaration), not a map.
    B[0] = A[0]
    for i in range(1, N):
        B[i] = B[i - 1] + A[i]


def top_level_nsdfgs(sdfg):
    return [n for state in sdfg.states() for n in state.nodes() if isinstance(n, nodes.NestedSDFG)]


def test_two_top_level_maps_are_outlined():
    sdfg = two_maps.to_sdfg(simplify=True)
    assert not top_level_nsdfgs(sdfg), 'fixture should start with no nested SDFGs'

    count = OutlineTopLevelNests().apply_pass(sdfg, {})

    assert count == 2
    nsdfgs = top_level_nsdfgs(sdfg)
    assert len(nsdfgs) == 2
    for nsdfg in nsdfgs:
        assert nsdfg.no_inline is True
        assert nsdfg.unique_name
    assert len({n.unique_name for n in nsdfgs}) == 2, 'unique_name must be distinct per nest'


def test_connectors_and_symbols_preserved():
    sdfg = two_maps.to_sdfg(simplify=True)
    outline_top_level_nests(sdfg)

    interfaces = {frozenset(n.in_connectors): frozenset(n.out_connectors) for n in top_level_nsdfgs(sdfg)}
    # nest 1 reads A,B writes C; nest 2 reads C writes D.
    assert frozenset({'A', 'B'}) in interfaces and interfaces[frozenset({'A', 'B'})] == frozenset({'C'})
    assert frozenset({'C'}) in interfaces and interfaces[frozenset({'C'})] == frozenset({'D'})
    for nsdfg in top_level_nsdfgs(sdfg):
        assert 'N' in nsdfg.symbol_mapping


def test_idempotent():
    sdfg = two_maps.to_sdfg(simplify=True)
    assert outline_top_level_nests(sdfg) == 2
    # After outlining, the top-level scopes are single-nsdfg states; nothing new to wrap.
    assert outline_top_level_nests(sdfg) == 0


def test_no_op_on_flat_sdfg():
    sdfg = dace.SDFG('flat')
    sdfg.add_array('A', [1], dace.float64)
    state = sdfg.add_state()
    state.add_edge(state.add_read('A'), None, state.add_write('A'), None, dace.Memlet('A[0]'))
    assert OutlineTopLevelNests().apply_pass(sdfg, {}) is None


def test_inlining_back_reproduces_the_original():
    """The wrap is loss-free: clearing no_inline and inlining recovers a nest-free SDFG. """
    sdfg = two_maps.to_sdfg(simplify=True)
    outline_top_level_nests(sdfg)
    assert top_level_nsdfgs(sdfg)

    for nsdfg in top_level_nsdfgs(sdfg):
        nsdfg.no_inline = False  # InlineSDFG refuses no_inline nodes
    inline_sdfgs(sdfg)
    assert not top_level_nsdfgs(sdfg), 'every outlined nest should inline back'


def test_sequential_loop_region_is_outlined():
    """The nest_sdfg_subgraph path: a top-level CFG loop region becomes its own no_inline nest,
    with its loop index pre-declared as a parent symbol so the outliner can type its symbolic out."""
    sdfg = seq_loop.to_sdfg(simplify=True)
    from dace.sdfg.state import LoopRegion
    assert any(isinstance(b, LoopRegion) for b in sdfg.nodes()), 'fixture must have a top-level LoopRegion'

    count = outline_top_level_nests(sdfg)

    assert count >= 1
    nsdfgs = top_level_nsdfgs(sdfg)
    assert nsdfgs
    for nsdfg in nsdfgs:
        assert nsdfg.no_inline is True and nsdfg.unique_name
    # No un-outlined top-level LoopRegion should remain.
    assert not any(isinstance(b, LoopRegion) for b in sdfg.nodes())
    sdfg.validate()
    # The loop index must NOT leak out as a required argument: the outliner pre-declares it and maps
    # it (sym: sym), and unless that inbound mapping is dropped it becomes a spurious root free symbol.
    assert 'i' not in {str(s) for s in sdfg.free_symbols}, 'loop index leaked as a required symbol'
    for nsdfg in nsdfgs:
        assert 'i' not in nsdfg.symbol_mapping, 'loop index left in the nest symbol_mapping'


@pytest.mark.parametrize('outline', [False, True])
def test_loop_region_numerically_equivalent(outline):
    """The loop-region path must RUN, not just validate: a leaked loop-index argument makes the built
    SDFG demand an `i` no caller supplies. seq_loop is a carried-dependency prefix sum."""
    sdfg = seq_loop.to_sdfg(simplify=True)
    if outline:
        assert outline_top_level_nests(sdfg) >= 1
    A = numpy.random.default_rng(0).random(37)
    B = numpy.zeros(37)
    sdfg(A=A, B=B, N=37)
    assert numpy.allclose(B, numpy.cumsum(A))


@pytest.mark.parametrize('outline', [False, True])
def test_numerically_equivalent(outline):
    """ Outlined and non-outlined builds must produce identical results. """
    sdfg = two_maps.to_sdfg(simplify=True)
    if outline:
        assert outline_top_level_nests(sdfg) == 2

    rng = numpy.random.default_rng(0)
    A = rng.random(64)
    B = rng.random(64)
    C = numpy.zeros(64)
    D = numpy.zeros(64)
    sdfg(A=A, B=B, C=C, D=D, N=64)
    assert numpy.allclose(C, A + B)
    assert numpy.allclose(D, (A + B) * 2.0)


if __name__ == '__main__':
    test_two_top_level_maps_are_outlined()
    test_connectors_and_symbols_preserved()
    test_idempotent()
    test_no_op_on_flat_sdfg()
    test_inlining_back_reproduces_the_original()
    test_numerically_equivalent(False)
    test_numerically_equivalent(True)
