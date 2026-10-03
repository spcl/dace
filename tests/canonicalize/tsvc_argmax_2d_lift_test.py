# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The 2-D argmax nest of TSVC ``s3110`` / ``s13110``, asserted structurally and numerically.

``for i: for j: if aa[i, j] > maxv: maxv = aa[i, j]; xindex = i; yindex = j`` scans the whole
contiguous array, so :class:`~dace.transformation.passes.canonicalize.arg_max_lift.ArgMaxLift`
folds the nest into ONE flat ``ArgReduce`` and decomposes the flat position back into the two
indices.

The assertions are on what ``canonicalize`` PRODUCES -- the libnode, no surviving sequential
loop, and an OpenMP-parallel lowering OF that libnode -- because a numeric comparison alone
passes just as happily on the untouched sequential nest, and "it parallelized" is the property at
stake. A libnode also hides its schedule from every pragma or map counter (the parallel and the
serial expansion are the same node until ``finalize_for_target`` picks one), so the emitted C++ is
the only honest place to read that half.

The numeric half is what stops the lift from being made to fire by breaking the tie rule: the
guard is strict, so the sequential nest keeps the FIRST maximum, and ``bb[0, 0]`` folds both
indices into its checksum and would move if the flat arg-reduce disagreed.

Contiguity is what makes the fold legal, and proving it means subtracting each loop bound against
the matching array extent. Both come from ``LEN_2D``, but the bound is reparsed from a
``CodeBlock`` string while the extent carries whatever the shape symbol was DECLARED with, so a
declaration like ``dace.symbol('N', dtype=dace.int64, positive=True)`` puts two sympy instances of
one name in that subtraction, ``N - N`` does not cancel, and ``_match_2d`` refuses a nest it fully
supports. The corpus declares ``LEN_2D`` bare, which is why the two kernels above lift either way
and cannot express that failure; ``test_2d_argmax_lifts_when_the_shape_symbol_carries_assumptions``
covers it on the declaration real frontends emit.
"""
import os

os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes import ArgReduce
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import finalize
from dace.transformation.passes.canonicalize.pipeline import canonicalize

from tests.corpus.tsvc import tsvc
from tests.corpus.tsvc.tsvc_numpy import REFERENCES

KERNELS = ('s3110_d_single', 's13110_d_single')


def canonicalized(name, tag):
    """``(kernel, sdfg)`` for one TSVC kernel put through the production canonicalize recipe."""
    kernel = tsvc.collect(name=name)[0]
    sdfg = tsvc.to_sdfg(kernel, tag, simplify=True)
    canonicalize(sdfg, validate=True, peel_limit=4)
    return kernel, sdfg


def residual_loops(sdfg):
    """The sequential ``LoopRegion`` s canonicalize did not turn into parallel work."""
    return [
        r for sd in sdfg.all_sdfgs_recursive() for r in sd.all_control_flow_regions()
        if isinstance(r, LoopRegion) and r.loop_variable
    ]


def num_argreduce(sdfg):
    return sum(1 for node, _ in sdfg.all_nodes_recursive() if isinstance(node, ArgReduce))


def assert_lowers_to_a_parallel_reduction(sdfg, name):
    """The libnode must reach the C++ as a parallel arg-reduction, not a sequential scan."""
    finalize.finalize_for_target(sdfg, 'cpu')
    src = '\n'.join(o.clean_code for o in sdfg.generate_code())
    assert '#pragma omp parallel for reduction' in src, f'{name}: ArgReduce must lower to a parallel reduction'
    assert '#pragma omp declare reduction' in src, f'{name}: the arg-reduction needs its (value, index) combiner'


def assert_matches_reference(kernel, sdfg):
    """The canonicalized kernel must reproduce the numpy reference element for element.

    Structure is read off ``sdfg`` BEFORE this runs: compiling is one-way, and the SDFG a
    ``CompiledSDFG`` still points at is not the object to inspect afterwards.
    """
    arrays, call_kwargs = tsvc.make_inputs(kernel)
    ref = {n: a.copy() for n, a in arrays.items()}
    REFERENCES[kernel.name](**ref, **call_kwargs)
    got = {n: a.copy() for n, a in arrays.items()}
    sdfg.compile()(**got, **call_kwargs)
    for n, arr in arrays.items():
        if np.issubdtype(arr.dtype, np.integer):
            continue
        assert np.allclose(ref[n], got[n], equal_nan=True), f'{kernel.name}: value mismatch on {n}'


@pytest.mark.parametrize('name', KERNELS)
def test_2d_argmax_nest_lifts_to_one_flat_argreduce(name):
    """The whole nest becomes a single flat ``ArgReduce``, leaving no sequential loop behind."""
    kernel, sdfg = canonicalized(name, 'lift')
    assert num_argreduce(sdfg) == 1, f'{name}: the 2-D nest must lift to exactly one flat ArgReduce'
    assert residual_loops(sdfg) == [], f'{name}: no loop of the nest may survive the lift'
    assert_matches_reference(kernel, sdfg)


@pytest.mark.parametrize('name', KERNELS)
def test_2d_argmax_argreduce_lowers_to_an_openmp_reduction(name):
    """The lift only pays off if the libnode lowers PARALLEL; a counter cannot see that."""
    _kernel, sdfg = canonicalized(name, 'omp')
    assert_lowers_to_a_parallel_reduction(sdfg, name)


def scaling_transients(sdfg):
    """Transients whose allocation grows with a program symbol -- i.e. a copy of the input."""
    return sorted(name for name, desc in sdfg.arrays.items()
                  if desc.transient and dace.symbolic.symlist(desc.total_size))


@pytest.mark.parametrize('name', KERNELS)
def test_2d_argmax_lift_allocates_nothing_of_problem_size(name):
    """The flat arg-reduce must read ``aa`` itself, not a staged copy of it.

    An arg-reduction consumes a sequence and answers two scalars, so anything of problem size in
    between is a full extra write plus a full extra read for nothing -- 4.2 GB each way at the
    corpus's XL rung. Asserted structurally because it is a property of the canonical form, and a
    numeric check passes just as happily with the copy in place.
    """
    _kernel, sdfg = canonicalized(name, 'nobuf')
    assert scaling_transients(sdfg) == [], f'{name}: the lift allocated a problem-sized buffer'
    node, state = next((n, st) for n, st in sdfg.all_nodes_recursive() if isinstance(n, ArgReduce))
    in_edge = next(e for e in state.in_edges(node) if e.dst_conn == '_in')
    assert in_edge.data.data == 'aa', f'{name}: the arg-reduce must read the input array directly'


def test_2d_argmax_ties_resolve_to_the_first_maximum_in_row_major_order():
    """Both index carriers must name the FIRST maximum, not just A maximum.

    The guard is strict, so the sequential nest never updates on a tie: with the maximum repeated
    the earliest ``(i, j)`` in row-major order wins. The lift folds the nest into a flat scan, so
    this checks two things at once -- that the flat scan keeps the lowest flat position, and that
    the decomposition ``(flat // ncols, flat % ncols)`` puts it back on the right pair of indices.
    Random draws never tie; these values are placed.
    """
    M = dace.symbol('argmax2d_tie_M')

    @dace.program
    def argmax2d_tie(aa: dace.float64[M, M], out: dace.float64[3]):
        maxv = aa[0, 0]
        xindex = 0
        yindex = 0
        for i in range(M):
            for j in range(M):
                if aa[i, j] > maxv:
                    maxv = aa[i, j]
                    xindex = i
                    yindex = j
        out[0] = maxv
        out[1] = float(xindex)
        out[2] = float(yindex)

    sdfg = argmax2d_tie.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, peel_limit=4)
    assert num_argreduce(sdfg) == 1
    assert scaling_transients(sdfg) == [], 'the 2-D lift allocated a problem-sized buffer'

    n = 5
    # A plateau of equal maxima spread over three rows, so a scan that split the array differently
    # -- or decomposed the flat index wrongly -- lands on one of the later ones.
    aa = np.full((n, n), -1.0)
    for i, j in ((1, 3), (2, 0), (2, 4), (4, 4)):
        aa[i, j] = 7.0
    out = np.zeros(3)
    sdfg(aa=aa.copy(), out=out, argmax2d_tie_M=n)

    flat = int(np.argmax(aa))  # numpy also returns the first occurrence in row-major order
    assert (flat // n, flat % n) == (1, 3)  # sanity: the fixture's first maximum
    assert out[0] == 7.0, f'value: got {out[0]}'
    assert (int(out[1]),
            int(out[2])) == (flat // n, flat %
                             n), (f'ties must resolve to the first maximum (1, 3); got ({int(out[1])}, {int(out[2])})')


def test_2d_argmax_lifts_when_the_shape_symbol_carries_assumptions():
    """The same nest, with the shape symbol declared the way real frontends declare it.

    ``dtype`` and assumptions are folded into sympy symbol identity, so this declaration is what
    makes the contiguity check's ``N - N`` fail to cancel. Structurally identical to
    ``s3110_d_single``, and checked against numpy's 2-D argmax rather than ``REFERENCES`` because
    the corpus has no assumption-carrying twin of that kernel to name.
    """
    N = dace.symbol('argmax2d_assumed_N', dtype=dace.int64, positive=True)

    @dace.program
    def argmax2d_assumed(aa: dace.float64[N, N], out: dace.float64[3]):
        maxv = aa[0, 0]
        xindex = 0
        yindex = 0
        for i in range(N):
            for j in range(N):
                if aa[i, j] > maxv:
                    maxv = aa[i, j]
                    xindex = i
                    yindex = j
        out[0] = maxv
        out[1] = float(xindex)
        out[2] = float(yindex)

    sdfg = argmax2d_assumed.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, peel_limit=4)
    assert num_argreduce(sdfg) == 1, 'declared symbol assumptions must not hide the contiguity of the nest'
    assert residual_loops(sdfg) == [], 'no loop of the nest may survive the lift'
    assert_lowers_to_a_parallel_reduction(sdfg, 'argmax2d_assumed')

    n = 7
    aa = np.random.default_rng(311013110).standard_normal((n, n))
    out = np.zeros(3)
    sdfg(aa=aa, out=out, argmax2d_assumed_N=n)
    flat = int(np.argmax(aa))
    assert np.isclose(out[0], aa.flat[flat]), f'value: got {out[0]}, expected {aa.flat[flat]}'
    assert (int(out[1]), int(out[2])) == (flat // n, flat % n), f'index: got ({out[1]}, {out[2]})'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
