# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Guards for the comparison hazard where one operand carries a symbol's DECLARED assumptions and
the other holds a bare instance reparsed from a string.

``dace.symbolic.symbol._hashable_content`` folds the DaCe dtype -- and sympy folds the assumptions --
into symbol identity, so ``N`` declared as ``dace.symbol('N', dtype=dace.int64, positive=True)`` and
``N`` recovered by ``pystr_to_symbolic`` from a ``CodeBlock`` are two independent sympy variables.
Subtracting them yields ``N - N`` rather than ``0`` and every gate written as
``simplify(bound - (shape - 1)) != 0`` then refuses, silently, with no exception and no warning.

Every test here MUST mint its symbol WITH assumptions: a bare-symbol fixture makes both instances
identical and the guard cannot fail, which is why the corpus never caught the original defect.
"""
import numpy as np
import pytest

import dace
from dace import data, subsets, symbolic
from dace.codegen.targets.cpu import _contiguous_element_count
from dace.libraries.linalg.nodes.inv import Inv
from dace.libraries.standard.nodes.symmetrize import ExpandSymmetrizeCUDA, Symmetrize
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.dataflow.wcr_conversion import WCRToAugAssign
from dace.transformation.layout.apply_assignment import covers_dimension, covers_full_array
from dace.transformation.layout.permute_dimensions import covers_full_array as permute_covers_full_array
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.analysis.analysis import covers_full_extent
from dace.transformation.passes.canonicalize.lift_inv import LiftInv
from dace.transformation.passes.canonicalize.loop_to_transpose import LoopToTranspose
from dace.transformation.passes.canonicalize.untile_loops import _diff_is_zero
from dace.transformation.passes.scatter_conflict_guard import scatter_index_is_provably_injective

# The declared spelling: dtype and assumptions are exactly what a benchmark emitter or a real
# frontend attaches, and exactly what the reparsed spelling below cannot recover.
DECLARED_N = dace.symbol('N', dtype=dace.int64, positive=True)


def bare(name: str) -> symbolic.SymbolicType:
    """The assumption-free instance a string round-trip yields, as loop bounds and memlets hold it."""
    return symbolic.pystr_to_symbolic(name)


def test_declared_and_reparsed_symbols_do_not_cancel():
    """The premise of every other test here: without equalization the two spellings never cancel."""
    assert symbolic.simplify(bare('N') - DECLARED_N) != 0
    lhs, rhs = symbolic.equalize_symbols_across(bare('N'), DECLARED_N)
    assert symbolic.simplify(lhs - rhs) == 0


def test_covers_full_extent_accepts_reparsed_bound():
    desc = data.Array(dace.float64, (DECLARED_N, ))
    reparsed = subsets.Range([(0, bare('N') - 1, 1)])
    assert covers_full_extent(reparsed, desc) is True


def test_apply_assignment_covers_dimension_accepts_reparsed_bound():
    assert covers_dimension(0, bare('N') - 1, 1, DECLARED_N) is True


def test_apply_assignment_covers_full_array_accepts_reparsed_memlet():
    desc = data.Array(dace.float64, (DECLARED_N, ))
    memlet = dace.Memlet(data='a', subset=subsets.Range([(0, bare('N') - 1, 1)]))
    assert covers_full_array(memlet, desc) is True


def test_permute_dimensions_covers_full_array_accepts_reparsed_memlet():
    desc = data.Array(dace.float64, (DECLARED_N, ))
    memlet = dace.Memlet(data='a', subset=subsets.Range([(0, bare('N') - 1, 1)]))
    assert permute_covers_full_array(memlet, desc) is True


def test_contiguous_element_count_accepts_reparsed_total_size():
    """``total_size`` is reassigned from reparsed map extents by MapFission and friends."""
    desc = data.Array(dace.float64, (DECLARED_N, ))
    desc.total_size = bare('N')
    assert _contiguous_element_count(desc) is not None


def test_diff_is_zero_accepts_reparsed_operand():
    assert _diff_is_zero(DECLARED_N - 1, bare('N') - 1) is True


def test_symmetrize_canonical_window_accepts_reparsed_bounds():
    """The window bounds are string properties, so they ALWAYS reach the check reparsed."""
    desc = data.Array(dace.float64, (DECLARED_N, DECLARED_N))
    node = Symmetrize('sym', row_lo='0', row_hi='N - 1', col_offset=1, col_hi='N')
    assert ExpandSymmetrizeCUDA.canonical_window(node, desc) is True


def affine_index_sdfg() -> dace.SDFG:
    """``for j in range(0, N): idx[j] = 2 * j`` over an array declared with assumptions.

    The loop bounds live in ``CodeBlock`` strings, so ``loop_analysis`` hands them back bare while
    ``idx``'s shape keeps the declared ones -- the shape the scatter-guard gate compares them to.
    """
    sdfg = dace.SDFG('affine_index')
    sdfg.add_array('idx', (DECLARED_N, ), dace.int64, transient=True)
    loop = LoopRegion('loop', 'j < N', 'j', 'j = 0', 'j = j + 1')
    sdfg.add_node(loop, is_start_block=True)
    state = loop.add_state('body', is_start_block=True)
    tasklet = state.add_tasklet('assign', {}, {'o'}, 'o = 2 * j')
    write = state.add_access('idx')
    state.add_edge(tasklet, 'o', write, None, dace.Memlet('idx[j]'))
    return sdfg


def test_scatter_index_injectivity_accepts_reparsed_loop_end():
    assert scatter_index_is_provably_injective(affine_index_sdfg(), 'idx') is True


DECLARED_M = dace.symbol('M', dtype=dace.int64, positive=True)


@dace.program
def declared_transpose(A: dace.float64[DECLARED_N, DECLARED_M], B: dace.float64[DECLARED_M, DECLARED_N]):
    for i in range(DECLARED_M):
        for j in range(DECLARED_N):
            B[i, j] = A[j, i]


def test_loop_to_transpose_sees_the_whole_array():
    """A whole-array permutation must lift to plain memlets, not to strided Views.

    The loop ranges are rebuilt from the reparsed loop bounds, the shapes carry the declared
    assumptions; unequalized, the pass concludes the access is a sub-grid and wraps both operands
    in Views that exist only to describe an offset that is not there.
    """
    sdfg = declared_transpose.to_sdfg(simplify=True)
    Pipeline([LoopToTranspose()]).apply_pass(sdfg, {})
    assert any(isinstance(n, nodes.LibraryNode) for n, _ in sdfg.all_nodes_recursive())
    assert not [name for name, desc in sdfg.arrays.items() if isinstance(desc, data.View)]


@dace.program
def declared_solve_eye(A: dace.float64[DECLARED_N, DECLARED_N], out: dace.float64[DECLARED_N, DECLARED_N]):
    out[:] = np.linalg.solve(A, np.eye(DECLARED_N))


def test_lift_inv_recognises_a_declared_size_identity():
    """``solve(A, eye(N))`` must still lift to ``Inv`` when N is declared with assumptions.

    Three gates compare across mintings here: A's shape against the output's, A's against the eye
    transient's, and the identity map's reparsed range against A's shape.
    """
    sdfg = declared_solve_eye.to_sdfg(simplify=True)
    Pipeline([LiftInv()]).apply_pass(sdfg, {})
    assert sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Inv)) == 1


def test_wcr_to_aug_assign_accepts_an_independently_minted_volume():
    """``volume`` is stored beside the subset, not derived from it on every read.

    Propagation recomputes it and deserialization reparses it, so its symbols need not be the
    subset's instances of the same names; the ``num_elements() - volume`` guard then never
    resolves and a conflict-free WCR copy keeps its atomics.
    """
    sdfg = dace.SDFG('wcr_copy')
    sdfg.add_array('A', (DECLARED_N, ), dace.float64)
    sdfg.add_array('src', (DECLARED_N, ), dace.float64)
    state = sdfg.add_state()
    memlet = dace.Memlet(data='A', subset=subsets.Range([(0, DECLARED_N - 1, 1)]), wcr='lambda a, b: a + b')
    memlet.volume = bare('N')
    state.add_edge(state.add_access('src'), None, state.add_access('A'), None, memlet)
    assert sdfg.apply_transformations_repeated(WCRToAugAssign) == 1


if __name__ == '__main__':
    pytest.main([__file__])
