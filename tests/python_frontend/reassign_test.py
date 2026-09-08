# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest

from dace.frontend.python.common import DaceSyntaxError


def test_reassign():
    """A name rebound to a differently-shaped container in each branch, where
    every branch returns.

    Nothing after the chain can read ``tmp``, so the two shapes never have to
    be reconciled -- only the returns do, and both hand back 20 elements.

    Nextgen-only: the classic frontend (``DACE_frontend_use_nextgen=0``)
    refuses any conditional rebinding that changes shape, live or not.
    """

    @dace.program
    def dead_after_return(A: dace.float64[20], B: dace.float64[30], selector: dace.int32):
        if selector == 0:
            tmp = np.empty_like(A)
            tmp[:] = A
            return tmp
        else:
            tmp = np.empty_like(B)
            tmp[:] = B
            return tmp[0:20]

    A = np.arange(20, dtype=np.float64)
    B = np.arange(100, 130, dtype=np.float64)
    assert np.allclose(dead_after_return(A.copy(), B.copy(), np.int32(0)), A)
    assert np.allclose(dead_after_return(A.copy(), B.copy(), np.int32(1)), B[0:20])


def test_reassign_samesize():

    @dace.program
    def samesize(A: dace.float64[20], B: dace.float64[30], selector: dace.int32):
        if selector == 0:
            tmp = np.empty_like(A)
            tmp[:] = A
            return tmp
        else:
            tmp = np.empty_like(A)
            tmp[:] = B[:20]
            return tmp[0:20]

    samesize.to_sdfg()


def test_reassign_retval():

    @dace.program
    def shouldfail_retval(A: dace.float64[20], B: dace.float64[30], selector: dace.int32):
        if selector == 0:
            tmp = np.empty_like(A)
            tmp[:] = A
            return tmp
        else:
            tmp2 = np.empty_like(B)
            tmp2[:] = B
            return tmp2

    with pytest.raises(DaceSyntaxError, match='Return'):
        shouldfail_retval.to_sdfg()


if __name__ == '__main__':
    test_reassign()
    test_reassign_samesize()
    test_reassign_retval()
