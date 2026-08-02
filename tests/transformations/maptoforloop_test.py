# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" A test for the MapToForLoop transformation. """

import dace
import numpy as np
from dace.sdfg.state import LoopRegion
from dace.transformation.dataflow import MapExpansion, MapToForLoop


@dace.program
def map2for(A: dace.float64[20, 20, 20]):
    for k in range(1, 19):
        for i, j in dace.map[0:20, 0:20]:
            with dace.tasklet:
                inp << A[i, j, k]
                inp2 << A[i, j, k - 1]
                out >> A[i, j, k + 1]
                out = inp + inp2


@dace.program
def map2for_scalar_range(A: dace.float64[20], lo: dace.int64, hi: dace.int64):
    for i in dace.map[lo:hi]:
        A[i] = A[i] + 1.0


def test_map2for_overlap():
    A = np.random.rand(20, 20, 20)
    expected = np.copy(A)
    for k in range(1, 19):
        expected[:, :, k + 1] = expected[:, :, k] + expected[:, :, k - 1]

    sdfg = map2for.to_sdfg()
    assert sdfg.apply_transformations([MapExpansion, MapToForLoop]) == 2
    sdfg(A=A)
    assert np.allclose(A, expected)


def test_map2for_scalar_dynamic_range():
    """A dynamic map range read from a ``Scalar`` parameter lowers to that scalar's VALUE.

    ``MapToForLoop`` splices the range into the LoopRegion's init / condition / update statements,
    which are PYTHON source over SDFG names. A ``Scalar`` is emitted as a by-value variable, so
    subscripting it there (``lo[0]``) is not lowerable: the C++ indexes an ``int64_t`` and the
    build fails with ``invalid types 'int64_t[int]' for array subscript``. The statement text is
    asserted too -- a ``Subscript(lo, 0)`` term also poisons every symbolic expression a later
    pass derives from the range, including array shapes, long before any compiler sees it.
    """
    sdfg = map2for_scalar_range.to_sdfg(simplify=True)
    assert sdfg.apply_transformations(MapToForLoop) == 1

    loops = [b for s in sdfg.all_sdfgs_recursive() for b in s.all_control_flow_blocks() if isinstance(b, LoopRegion)]
    assert len(loops) == 1
    statements = [
        loops[0].init_statement.as_string, loops[0].loop_condition.as_string, loops[0].update_statement.as_string
    ]
    assert not any('lo[' in s or 'hi[' in s for s in statements), statements

    A = np.random.rand(20)
    expected = np.copy(A)
    expected[3:17] += 1.0
    sdfg(A=A, lo=3, hi=17)
    assert np.allclose(A, expected)


if __name__ == '__main__':
    test_map2for_overlap()
    test_map2for_scalar_dynamic_range()
