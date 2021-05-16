import numpy as np

import dace


def test_numpy_bool_input():

    sdfg = dace.SDFG("test_numpy_bool_input")
    state = sdfg.add_state()

    sdfg.add_scalar("in_bool", dace.bool)
    sdfg.add_array("__return", [1], dace.bool)

    r = state.add_read("in_bool")
    w = state.add_write("__return")
    state.add_edge(r, None, w, None, sdfg.make_array_memlet("in_bool"))

    # test python bool
    result = sdfg(in_bool=True)
    assert result[0]

    # test numpy.bool_
    result = sdfg(in_bool=result[0])
    assert result[0]

    # test numpy.bool (which is just bool)
    result = sdfg(in_bool=bool(True))
    assert result[0]


if __name__ == "__main__":
    test_numpy_bool_input()
