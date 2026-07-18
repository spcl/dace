# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for indirect tasklet memlets (``in_x << x[A_col[j]]``) in the
next-generation Python frontend: inner data reads become synthetic
``__ind<N>`` input connectors, the outer array becomes a full-array
``<conn>__arr`` connector, and the element access moves into the tasklet
code. Together with dynamic map ranges and mapscope desugaring, this makes
the classic spmv program lower with zero callbacks.
"""
import numpy as np
import scipy.sparse as sp

import dace
from dace.frontend.python import nextgen
from dace.sdfg.analysis.schedule_tree import treenodes as tn

W = dace.symbol('W')
H = dace.symbol('H')
nnz = dace.symbol('nnz')
N = dace.symbol('N')


def _nodes_of_type(root: tn.ScheduleTreeRoot, node_type):
    return [node for node in root.preorder_traversal() if isinstance(node, node_type)]


@dace.program
def spmv(A_row: dace.uint32[H + 1], A_col: dace.uint32[nnz], A_val: dace.float32[nnz], x: dace.float32[W],
         b: dace.float32[H]):

    @dace.mapscope(_[0:H])
    def compute_row(i):

        @dace.map(_[A_row[i]:A_row[i + 1]])
        def compute(j):
            a << A_val[j]
            in_x << x[A_col[j]]
            out >> b(1, lambda x, y: x + y)[i]

            out = a * in_x


def test_spmv_structure():
    tree = nextgen.parse_program(spmv)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)

    # Two dynamic map-range inputs read A_row directly (no scalar temps).
    dyn_copies = _nodes_of_type(tree, tn.DynScopeCopyNode)
    assert len(dyn_copies) == 2
    assert all(node.memlet.data == 'A_row' for node in dyn_copies)

    tasklets = _nodes_of_type(tree, tn.TaskletNode)
    assert len(tasklets) == 1
    tasklet = tasklets[0]
    assert set(tasklet.in_memlets) == {'a', 'in_x__arr', '__ind0'}
    # The index read is the affine A_col[j] element; x arrives whole.
    assert tasklet.in_memlets['__ind0'].data == 'A_col'
    assert tasklet.in_memlets['in_x__arr'].data == 'x'
    assert str(tasklet.in_memlets['in_x__arr'].subset) == '0:W'
    # The element access moved into the tasklet code.
    assert 'in_x = in_x__arr[__ind0]' in tasklet.node.code.as_string
    # The WCR output stays a plain affine memlet.
    assert tasklet.out_memlets['out'].data == 'b'
    assert tasklet.out_memlets['out'].wcr is not None


def test_spmv_execution():
    tree = nextgen.parse_program(spmv)
    func = tree.as_sdfg().compile()

    rng = np.random.default_rng(42)
    matrix = sp.random(20, 25, density=0.3, format='csr', dtype=np.float32, random_state=rng)
    x = rng.random(25, dtype=np.float32)
    b = np.zeros(20, dtype=np.float32)
    func(A_row=matrix.indptr.astype(np.uint32),
         A_col=matrix.indices.astype(np.uint32),
         A_val=matrix.data,
         x=x,
         b=b,
         W=25,
         H=20,
         nnz=matrix.nnz)
    assert np.allclose(b, matrix @ x, rtol=1e-5, atol=1e-5)


def test_indirect_read_plain_map():

    @dace.program
    def gather(x: dace.float64[N], idx: dace.int32[N], out: dace.float64[N]):
        for j in dace.map[0:N]:
            with dace.tasklet:
                v << x[idx[j]]
                o >> out[j]
                o = v

    tree = nextgen.parse_program(gather)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)

    func = tree.as_sdfg().compile()
    rng = np.random.default_rng(7)
    n = 12
    x = rng.random(n)
    idx = rng.integers(0, n, size=n).astype(np.int32)
    out = np.zeros(n)
    func(x=x, idx=idx, out=out, N=n)
    assert np.allclose(out, x[idx])


def test_indirect_nonaffine_fallback():

    @dace.program
    def double_gather(x: dace.float64[N], idx: dace.int32[N], out: dace.float64[N]):
        for j in dace.map[0:N]:
            with dace.tasklet:
                v << x[idx[idx[j]]]
                o >> out[j]
                o = v

    # Nested indirection is not supported; the loop must fall back to a
    # callback without crashing.
    tree = nextgen.parse_program(double_gather)
    assert len(_nodes_of_type(tree, tn.PythonCallbackNode)) >= 1


if __name__ == '__main__':
    test_spmv_structure()
    test_spmv_execution()
    test_indirect_read_plain_map()
    test_indirect_nonaffine_fallback()
