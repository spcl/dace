# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for the ``Symmetrize`` library node and the ``LoopToSymmetrize``
canonicalization pass.

The pass lifts a triangular in-place matrix-symmetrization loop nest
(``for i: for j in i+1:M: X[j,i] = X[i,j]``) to a ``Symmetrize`` node whose pure
expansion is a parallel triangular copy -- turning a nest that ``LoopToMap``
refuses (in-place symmetric read/write false-dependence) into a fully parallel
form.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.libraries.standard.nodes import Symmetrize
from dace.transformation.passes.canonicalize.pipeline import canonicalize

M = dace.symbol('M')


@dace.program
def symmetrize_upper(X: dace.float64[M, M]):
    for i in range(0, M - 1):
        for j in range(i + 1, M):
            X[j, i] = X[i, j]


@dace.program
def symmetrize_lower(X: dace.float64[M, M]):
    for i in range(0, M - 1):
        for j in range(i + 1, M):
            X[i, j] = X[j, i]


@dace.program
def not_symmetrize(A: dace.float64[M, M], B: dace.float64[M, M]):
    """Cross-array copy (not in-place) -- must NOT lift to Symmetrize."""
    for i in range(0, M - 1):
        for j in range(i + 1, M):
            A[j, i] = B[i, j]


def _nsym(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Symmetrize))


def _nloops(sdfg):
    return sum(1 for r in sdfg.all_control_flow_regions(recursive=True)
               if isinstance(r, LoopRegion) and r.loop_variable)


def _mirror(X, source_upper):
    exp = X.copy()
    m = X.shape[0]
    for i in range(m):
        for j in range(i + 1, m):
            if source_upper:
                exp[j, i] = exp[i, j]
            else:
                exp[i, j] = exp[j, i]
    return exp


def test_node_expands_and_runs():
    """A Symmetrize node builds, expands to a parallel triangular copy, and runs."""
    sdfg = dace.SDFG('sym_node')
    sdfg.add_array('X', [M, M], dace.float64)
    st = sdfg.add_state()
    node = Symmetrize('sym', row_lo='0', row_hi='M', col_offset=1, col_hi='M', source_upper=True)
    st.add_node(node)
    st.add_edge(st.add_read('X'), None, node, '_in', dace.Memlet('X[0:M, 0:M]'))
    st.add_edge(node, '_out', st.add_write('X'), None, dace.Memlet('X[0:M, 0:M]'))
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()

    m = 7
    rng = np.random.default_rng(0)
    X = rng.standard_normal((m, m))
    got = X.copy()
    sdfg(X=got, M=m)
    assert np.allclose(got, _mirror(X, True))
    assert np.allclose(got, got.T)


@pytest.mark.parametrize('prog,source_upper', [(symmetrize_upper, True), (symmetrize_lower, False)])
def test_lifts_and_parallelizes(prog, source_upper):
    """The triangular symmetrization nest lifts to one Symmetrize node, leaves no
    sequential loop, and stays value-correct."""
    sdfg = prog.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    assert _nsym(sdfg) == 1, 'the symmetrization nest must lift to exactly one Symmetrize node'
    assert _nloops(sdfg) == 0, 'no sequential loop should remain'
    sdfg.validate()

    m = 9
    rng = np.random.default_rng(1)
    X = rng.standard_normal((m, m))
    got = X.copy()
    sdfg(X=got, M=m)
    assert np.allclose(got, _mirror(X, source_upper))


def test_cross_array_copy_not_lifted():
    """A cross-array (not in-place) triangular copy must NOT lift to Symmetrize --
    it is already a false-dependence-free parallel copy. Canonicalize leaves no
    Symmetrize node and stays value-correct."""
    sdfg = not_symmetrize.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    assert _nsym(sdfg) == 0

    m = 8
    rng = np.random.default_rng(2)
    A0, B = rng.standard_normal((m, m)), rng.standard_normal((m, m))
    got = A0.copy()
    sdfg(A=got, B=B.copy(), M=m)
    exp = A0.copy()
    for i in range(m):
        for j in range(i + 1, m):
            exp[j, i] = B[i, j]
    assert np.allclose(got, exp)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


def cuda_launch_dimensions(sdfg: dace.SDFG) -> list[str]:
    """The ``dim3(...)`` grid and block arguments of every kernel launch the CUDA unit emits."""
    dims: list[str] = []
    for obj in sdfg.generate_code():
        if obj.language not in ('cu', 'cpp'):
            continue
        for line in obj.clean_code.splitlines():
            dims += [chunk.split(')')[0] for chunk in line.split('dim3(')[1:]]
    return dims


def test_the_triangular_column_map_never_reaches_a_launch_dimension():
    """A block dimension is host code and has to be uniform across the grid; a triangular row
    length is neither. The column map's extent depends on ``__i``, so scheduling it
    ``GPU_ThreadBlock`` emitted ``dim3(((M - __i) - 1), 1, 1)`` and nvcc answered
    ``identifier "__i" is undefined`` -- polybench correlation, an empty kernel away from a wrong
    number if it had compiled."""
    sdfg = symmetrize_upper.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    assert _nsym(sdfg) == 1, 'the nest did not lift, so this asserts nothing'
    sdfg.apply_gpu_transformations()

    dims = cuda_launch_dimensions(sdfg)
    assert dims, 'no kernel was launched, so the launch dimensions assert nothing'
    assert not [d for d in dims if '__i' in d], f'a launch dimension reads a map parameter: {dims}'


@pytest.mark.parametrize('source_upper', [True, False])
def test_the_bounding_box_expansion_walks_a_rectangle_and_computes_the_same_mirror(source_upper):
    """The fallback GPU lowering trades the triangle for its bounding box to keep BOTH axes parallel.

    This is what a non-canonical window (a sub-block, a non-square target) gets, since the tiled
    kernel indexes one extent only.

    Correctness rests on the clamp: a thread past the end of a short row redoes that row's last
    element instead of running off it. That is only safe because the write is
    ``X[mirror] = X[source]``, the source triangle is never written, and every duplicate writes the
    same value to the same address -- so this runs the expansion and compares against the oracle
    rather than asserting the index expression. No GPU is involved: the shape of the iteration
    space is what is under test, and it is the same on either target.
    """
    sdfg = dace.SDFG(f'sym_gpu_{int(source_upper)}')
    sdfg.add_array('X', [M, M], dace.float64)
    st = sdfg.add_state()
    node = Symmetrize('sym', row_lo='0', row_hi='M - 1', col_offset=1, col_hi='M', source_upper=source_upper)
    st.add_node(node)
    st.add_edge(st.add_read('X'), None, node, '_in', dace.Memlet('X[0:M, 0:M]'))
    st.add_edge(node, '_out', st.add_write('X'), None, dace.Memlet('X[0:M, 0:M]'))
    node.implementation = 'bounding_box'
    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()

    maps = [n.map for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.sdfg.nodes.MapEntry)]
    assert len(maps) == 1, f'the bounding box has to be ONE map for the tiler to split it: {[m.label for m in maps]}'
    assert len(maps[0].params) == 2, f'both axes have to reach the launch: {maps[0].params}'
    outer = maps[0].params[0]
    inner_bounds = str(maps[0].range[1])
    assert outer not in inner_bounds, (f'the column extent still names the row parameter ({outer}), so it would '
                                       f'become a per-block launch dimension: {inner_bounds}')

    m = 9
    rng = np.random.default_rng(1)
    X = rng.standard_normal((m, m))
    got = X.copy()
    sdfg(X=got, M=m)
    assert np.allclose(got, _mirror(X, source_upper))
    assert np.allclose(got, got.T)


@pytest.mark.gpu
@pytest.mark.parametrize('source_upper', [True, False])
@pytest.mark.parametrize('m', [9, 32, 33, 64, 100])
def test_the_tiled_cuda_kernel_mirrors_every_size(source_upper, m):
    """The tiled kernel is guarded per element, so the ragged sizes are the ones that matter.

    ``m`` deliberately straddles the 32-wide tile: a size that is not a multiple of it leaves
    partial tiles on the last row and column, and the diagonal tiles hold both source and
    destination elements at once. Values, not structure -- an off-by-one in those guards writes the
    wrong triangle and no error is raised.
    """
    sdfg = dace.SDFG(f'sym_cuda_{int(source_upper)}_{m}')
    # Host signature: ``apply_gpu_transformations`` stages it to the device and schedules the node
    # there, which is the shape the pipeline actually produces.
    sdfg.add_array('X', [M, M], dace.float64)
    st = sdfg.add_state()
    node = Symmetrize('sym', row_lo='0', row_hi='M - 1', col_offset=1, col_hi='M', source_upper=source_upper)
    node.implementation = 'CUDA'
    st.add_node(node)
    st.add_edge(st.add_read('X'), None, node, '_in', dace.Memlet('X[0:M, 0:M]'))
    st.add_edge(node, '_out', st.add_write('X'), None, dace.Memlet('X[0:M, 0:M]'))
    sdfg.validate()
    sdfg.apply_gpu_transformations(validate=False, simplify=False)

    rng = np.random.default_rng(m)
    X = rng.standard_normal((m, m))
    got = X.copy()
    sdfg(X=got, M=m)
    assert np.allclose(got, _mirror(X, source_upper))
    assert np.allclose(got, got.T)
