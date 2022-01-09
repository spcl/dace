# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

from dace.transformation import dataflow, interstate, subgraph
from dace.transformation.interstate import InlineSDFG
from dace.transformation.transformation import simplification_transformations

xforms = simplification_transformations()
xforms.remove(InlineSDFG)


def test_inconn_self_copy():
    @dace.program
    def loop_body(A: dace.int32[5, 5], B: dace.int32[5]):
        A[1] = A[0]
        B[0] = np.sum(A[1])

    @dace.program
    def inconn_self_copy(A: dace.int32[5, 5]):
        B = np.ndarray(5, dtype=np.int32)
        loop_body(A, B)
        return B

    sdfg = inconn_self_copy.to_sdfg(simplify=False)
    sdfg.apply_transformations_repeated(xforms)
    sdfg.save('test_pre_is.sdfg')
    sdfg.apply_transformations(InlineSDFG)
    sdfg.save('test_post_is.sdfg')

    A = np.zeros((5, 5), dtype=np.int32)
    A[0] = 1
    refA = np.copy(A)
    B = sdfg(A=A)
    refB = inconn_self_copy.f(refA)

    assert (np.allclose(A, refA))
    assert (np.allclose(B[0], refB[0]))


def test_outconn_self_copy():
    @dace.program
    def loop_body(A: dace.int32[5, 5], B: dace.int32[5, 5]):
        A[1] = A[0]
        B[0] = A[2]
        B[1] = B[0]

    @dace.program
    def outconn_self_copy(A: dace.int32[5, 5]):
        B = np.ndarray((5, 5), dtype=np.int32)
        loop_body(A, B)
        return B

    sdfg = outconn_self_copy.to_sdfg(simplify=False)
    sdfg.apply_transformations_repeated(xforms)
    sdfg.save('test_pre_is.sdfg')
    sdfg.apply_transformations(InlineSDFG)
    sdfg.save('test_post_is.sdfg')

    A = np.zeros((5, 5), dtype=np.int32)
    A[0] = 1
    A[2] = 2
    refA = np.copy(A)
    B = sdfg(A=A)
    refB = outconn_self_copy.f(refA)

    assert (np.allclose(A, refA))
    assert (np.allclose(B[0:2], refB[0:2]))


def test_in_out_inconn_copy():
    @dace.program
    def loop_body(A: dace.int32[5, 5], B: dace.int32[5, 5]):
        B[1] = A[0]
        A[2] = B[3]

    @dace.program
    def in_out_inconn_copy(A: dace.int32[5, 5]):
        B = np.ndarray((5, 5), dtype=np.int32)
        B[3] = 3
        loop_body(A, B)
        return B

    sdfg = in_out_inconn_copy.to_sdfg(simplify=False)
    sdfg.apply_transformations_repeated(xforms)
    sdfg.save('test_pre_is.sdfg')
    sdfg.apply_transformations(InlineSDFG)
    sdfg.save('test_post_is.sdfg')

    A = np.zeros((5, 5), dtype=np.int32)
    A[0] = 1
    refA = np.copy(A)
    B = sdfg(A=A)
    refB = in_out_inconn_copy.f(refA)

    assert (np.allclose(A, refA))
    assert (np.allclose(B[1], refB[1]))
    assert (np.allclose(B[3], refB[3]))


def test_intermediate_copies():
    @dace.program
    def loop_body(A: dace.int32[5, 5], B: dace.int32[5, 5]):
        B[1] = A[0]
        tmp1 = B[2] + 5
        B[3] = tmp1
        A[2] = B[3]
        tmp2 = A[1] + 5
        A[4] = tmp2
        B[4] = A[4]

    @dace.program
    def intermediate_copies(A: dace.int32[5, 5]):
        B = np.ndarray((5, 5), dtype=np.int32)
        B[2] = 2
        B[3] = 3
        loop_body(A, B)
        return B

    sdfg = intermediate_copies.to_sdfg(simplify=False)
    sdfg.apply_transformations_repeated(xforms)
    sdfg.save('test_pre_is.sdfg')
    sdfg.apply_transformations(InlineSDFG)
    sdfg.save('test_post_is.sdfg')

    A = np.zeros((5, 5), dtype=np.int32)
    A[0] = 1
    A[1] = 2
    A[2] = 3
    A[3] = 4
    A[4] = 5
    refA = np.copy(A)
    B = sdfg(A=A)
    refB = intermediate_copies.f(refA)

    assert (np.allclose(A, refA))
    assert (np.allclose(B[1:], refB[1:]))


if __name__ == "__main__":
    test_inconn_self_copy()
    test_outconn_self_copy()
    test_in_out_inconn_copy()
    test_intermediate_copies()
