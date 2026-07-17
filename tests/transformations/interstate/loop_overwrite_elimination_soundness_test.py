# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Soundness regressions for ``LoopOverwriteElimination``.

The pass replaces a loop that overwrites the same location every iteration with just its last
iteration. Three independent defects made it eliminate loops it must not, each producing WRONG
NUMBERS silently:

1. **The guard validated a different index than the rewrite used.** ``can_be_applied``'s
   loop-carried-dependency check substituted ``get_loop_end()`` -- the inclusive bound implied by the
   loop CONDITION (``i < 10`` -> 9) -- while ``apply`` substituted the last ATTAINED iterate
   (``range(0, 10, 2)`` -> 8). For a stride that does not divide the trip range the guard therefore
   examined an index the body was never pinned to, and a real dependency at the true last iterate went
   unchecked. Both now derive the index from one helper, ``_last_iteration``.

2. **``//`` instead of ``int_floor``.** ``start + (end - start) // stride * stride`` on sympy operands
   builds ``sympy.floor``, which the code generator lowers to C ``/`` -- truncation toward zero rather
   than floor -- so with a symbolic bound the emitted index disagreed with the analyzed one.

3. **The dependency guard failed open on an undecidable comparison.** ``dace.subsets.intersects`` is
   three-valued (True / False / None); None means it could not decide (a non-unit stride or tile, or an
   undecidable symbolic relation -- the module-level wrapper converts that TypeError into None). ``None``
   is falsy, so ``if any(intersects(...))`` fell through and ACCEPTED. The sibling check at the
   unique-data guard already got this right with ``not intersects(...)``.

Defect 3 was found while disproving a fourth suspicion (that the TypeError escaped ``can_be_applied``);
it cannot, because line 11 imports the module-level wrapper, which swallows it.

The pass is exported from ``dace.transformation.interstate`` but is not wired into any pipeline, so
these are latent for anyone who requests the transformation explicitly.
"""
import copy
import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MPI4PY_RC_INITIALIZE', '0')
os.environ.setdefault('OMPI_MCA_pml', 'ob1')
os.environ.setdefault('OMPI_MCA_btl', 'self,vader')
os.environ.setdefault('UCX_VFS_ENABLE', 'n')

import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate import LoopOverwriteElimination


def _loops(sdfg):
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, LoopRegion)]


def test_nondividing_stride_readback_is_not_eliminated():
    """``for i in range(0, 10, 2): A[8] = A[i] * 2`` carries a dependency and must survive.

    The read reaches the invariant write ``A[8]`` at the real last iterate ``i == 8``. The old guard
    tested ``i := end == 9``, missed the overlap, and eliminated the loop.
    """

    @dace.program
    def tester(A: dace.float64[10]):
        for i in range(0, 10, 2):
            A[8] = A[i] * 2.0

    sdfg = tester.to_sdfg(simplify=True)
    assert len(_loops(sdfg)) == 1

    applied = sdfg.apply_transformations_repeated(LoopOverwriteElimination)

    assert applied == 0, 'eliminated a loop whose last iterate reads the location it overwrites'
    assert len(_loops(sdfg)) == 1


def test_nondividing_stride_readback_is_value_preserving():
    """The same kernel, compared BIT-EXACTLY against the untransformed SDFG.

    Integers-as-floats with an exact reference: ``np.testing.assert_equal`` is an exact comparison,
    so no tolerance is invented here. Before the fix: got 18.0, expected 28.0 at index 8.
    """

    @dace.program
    def tester(A: dace.float64[10]):
        for i in range(0, 10, 2):
            A[8] = A[i] * 2.0

    ref_sdfg = tester.to_sdfg(simplify=True)
    xf_sdfg = tester.to_sdfg(simplify=True)
    xf_sdfg.apply_transformations_repeated(LoopOverwriteElimination)

    base = np.arange(1.0, 11.0, dtype=np.float64)
    ref, got = base.copy(), base.copy()
    ref_sdfg(A=ref)
    xf_sdfg(A=got)

    np.testing.assert_equal(got, ref)


def test_nondividing_stride_without_readback_still_eliminates():
    """Over-refusal guard for fix 1: ``A[9]`` is never read back, so elimination stays legal.

    ``i`` never attains 9, so the old code's ``end``-based guard ALSO erred conservatively here and
    refused a safe loop. The shared ``_last_iteration`` must recover this elimination, not just block
    the unsound one -- otherwise the fix would be indistinguishable from "always refuse".
    """

    @dace.program
    def tester(A: dace.float64[10]):
        for i in range(0, 10, 2):
            A[9] = A[i] * 2.0

    ref_sdfg = tester.to_sdfg(simplify=True)
    xf_sdfg = tester.to_sdfg(simplify=True)

    base = np.arange(1.0, 11.0, dtype=np.float64)
    ref, got = base.copy(), base.copy()
    ref_sdfg(A=ref)

    applied = xf_sdfg.apply_transformations_repeated(LoopOverwriteElimination)
    assert applied == 1, 'refused a safe elimination: i never attains 9'
    assert len(_loops(xf_sdfg)) == 0

    xf_sdfg(A=got)
    np.testing.assert_equal(got, ref)


def test_symbolic_bound_nonunit_stride_is_value_preserving():
    """Fix 2: a SYMBOLIC bound with a non-unit stride must pin the body to the right iterate.

    ``//`` produced ``floor(N/2 - 1/2)``, which lowered to a truncating C ``/`` and selected the wrong
    element. ``int_floor`` keeps the symbolic and the emitted index in agreement. Swept over N so both
    parities of ``(end - start) / stride`` are covered.
    """
    N = dace.symbol('N')

    @dace.program
    def tester(A: dace.float64[64], B: dace.float64[1]):
        for i in range(0, N, 2):
            B[0] = A[i]

    for n in (8, 9, 15, 16):
        A = (np.arange(64.0, dtype=np.float64) * 1.25).copy()
        expected = np.array([A[max(range(0, n, 2))]], dtype=np.float64)

        xf_sdfg = tester.to_sdfg(simplify=True)
        xf_sdfg.apply_transformations_repeated(LoopOverwriteElimination)
        got = np.zeros(1, dtype=np.float64)
        xf_sdfg(A=A.copy(), B=got, N=n)

        np.testing.assert_equal(got, expected, err_msg=f'wrong last iterate pinned for N={n}')


def test_reparenting_does_not_collide_block_names():
    """Fix 3 (structural): reparenting the body must not leave two blocks sharing one name.

    ``apply`` added only ``start_block`` and let ``add_edge`` auto-add the rest, which bypasses unique
    naming. A loop body is its own name scope, so a label unique inside the loop can already be taken
    in the destination. ``TrivialLoopElimination`` had this exact defect.
    """
    sdfg = dace.SDFG('collide')
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_symbol('i', dace.int64)

    # A parent state whose label the loop body also uses.
    outer = sdfg.add_state('body', is_start_block=True)

    loop = LoopRegion('loop', 'i < 4', 'i', 'i = 0', 'i = i + 1', sdfg=sdfg)
    sdfg.add_node(loop)
    sdfg.add_edge(outer, loop, dace.InterstateEdge())
    inner = loop.add_state('body', is_start_block=True)
    t = inner.add_tasklet('set', {}, {'out'}, 'out = 1.0')
    inner.add_edge(t, 'out', inner.add_write('A'), None, dace.Memlet(data='A', subset='0'))

    sdfg.apply_transformations_repeated(LoopOverwriteElimination)

    for region in sdfg.all_control_flow_regions(recursive=True):
        labels = [b.label for b in region.nodes()]
        assert len(labels) == len(set(labels)), \
            f'region {region.label!r} holds duplicate block names: {labels}'
    sdfg.validate()


if __name__ == '__main__':
    pytest.main([__file__, '-x', '-q'])
