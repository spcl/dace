# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

import dace
from dace.sdfg import utils as sdutil
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.array_elimination import ArrayElimination


def test_redundant_simple():

    @dace.program
    def tester(A: dace.float64[20], B: dace.float64[20]):
        e = dace.ndarray([20], dace.float64)
        f = dace.ndarray([20], dace.float64)
        g = dace.ndarray([20], dace.float64)
        h = dace.ndarray([20], dace.float64)
        c = A + 1
        d = A + 2
        e[:] = c
        f[:] = d
        g[:] = f
        h[:] = d
        B[:] = g + e

    sdfg = tester.to_sdfg(simplify=False)
    sdutil.inline_sdfgs(sdfg)
    sdutil.fuse_states(sdfg)
    sdutil.inline_sdfgs(sdfg)
    sdutil.fuse_states(sdfg)
    Pipeline([ArrayElimination()]).apply_pass(sdfg, {})
    assert len(sdfg.arrays) == 4


def test_merge_simple():
    sdfg = dace.SDFG('tester')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_array('B', [20], dace.float64)

    state = sdfg.add_state()
    a1 = state.add_read('A')
    a2 = state.add_read('A')
    b1 = state.add_write('B')
    b2 = state.add_write('B')
    t1 = state.add_tasklet('doit1', {'a'}, {'b'}, 'b = a')
    t2 = state.add_tasklet('doit2', {'a'}, {'b'}, 'b = a')
    state.add_edge(a1, None, t1, 'a', dace.Memlet('A[0]'))
    state.add_edge(a2, None, t2, 'a', dace.Memlet('A[1]'))
    state.add_edge(t1, 'b', b1, None, dace.Memlet('B[0]'))
    state.add_edge(t2, 'b', b2, None, dace.Memlet('B[1]'))

    Pipeline([ArrayElimination()]).apply_pass(sdfg, {})
    assert len(state.data_nodes()) == 2


def test_remove_unused_scalars() -> None:
    sdfg = dace.SDFG("tester")
    sdfg.add_array("A", [10], dace.float32)
    sdfg.add_array("B", [10], dace.float32)
    sdfg.add_scalar("tmp", dace.int32, transient=True)

    state = sdfg.add_state()
    state.add_mapped_tasklet("copy", {"i": dace.subsets.Range.from_string("0:10")},
                             code="a = b; t=1",
                             inputs={"a": dace.Memlet("A[i]")},
                             outputs={
                                 "b": dace.Memlet("B[i]"),
                                 "t": dace.Memlet("tmp[0]")
                             },
                             external_edges=True)
    sdfg.simplify()
    assert "tmp" not in sdfg.arrays


if __name__ == '__main__':
    test_redundant_simple()
    test_merge_simple()
    test_remove_unused_scalars()
