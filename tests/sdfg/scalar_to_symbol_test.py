# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests the scalar to symbol promotion functionality. """
import dace
from dace.sdfg.analysis import scalar_to_symbol


def test_find_promotable():
    """ Find promotable and non-promotable symbols. """
    @dace.program
    def testprog(A: dace.float32[20, 20]):
        tmp = dace.ndarray([20, 20], dtype=dace.float32)
        i = 1
        i = 2
        j = 0
        while j < 5:
            tmp[:] = A + j
            j += 1
            i += j

    sdfg: dace.SDFG = testprog.to_sdfg()
    scalars = scalar_to_symbol.find_promotable_scalars(sdfg)
    assert scalars == {'i', 'j'}


def test_promote_simple():
    """ Simple promotion. """
    pass


def test_promote_loops():
    """ Nested loops. """
    pass


def test_promote_indirection():
    """ Indirect access in promotion. """
    pass


if __name__ == '__main__':
    test_find_promotable()
    test_promote_simple()
    test_promote_loops()
    test_promote_indirection()
