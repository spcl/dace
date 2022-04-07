# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import pytest
import dace
import numpy as np

from dace.sdfg.validation import InvalidSDFGEdgeError

W = dace.symbol('W')

number = 42


@dace.program
def duplicate_naming_inner(A, number):
    @dace.map(_[0:W])
    def bla(i):
        inp << A[i]
        out >> number[i]
        out = 2 * inp


@dace.program
def duplicate_naming(A, B):
    no = dace.define_local([number], dace.float32)
    number = dace.define_local([W], dace.float32)

    duplicate_naming_inner(A, number)

    @dace.map(_[0:W])
    def bla2(i):
        inp << number[i]
        out >> B[i]
        out = 2 * inp


def test():
    W.set(3)

    A = dace.ndarray([W])
    B = dace.ndarray([W])

    A[:] = np.mgrid[0:W.get()]
    B[:] = dace.float32(0.0)

    duplicate_naming(A, B, W=W)

    diff = np.linalg.norm(4 * A - B) / W.get()
    print("Difference:", diff)
    assert diff <= 1e-5


def test_duplicate_object():
    sdfg = dace.SDFG('shouldfail')
    sdfg.add_array('A', [20], dace.float64)
    state = sdfg.add_state()
    a = state.add_read('A')
    b = state.add_write('A')
    memlet = dace.Memlet('A[0]')
    state.add_nedge(a, b, memlet)
    state.add_nedge(a, b, memlet)

    with pytest.raises(InvalidSDFGEdgeError):
        sdfg.validate()


if __name__ == "__main__":
    # test()
    test_duplicate_object()
