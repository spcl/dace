# # Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from dace.memlet import Memlet
from dace.libraries.standard.nodes import std_nodes 


def pure_graph(implementation, dtype, size):
    sdfg_name = f"fill_{implementation}_{dtype.ctype}_w{size}"
    sdfg = dace.SDFG(sdfg_name)

    state = sdfg.add_state("fill")

    value = dace.symbol("value")
    sdfg.add_array("r", [size], dtype)
    result = state.add_write("r")

    fill_node = std_nodes.Fill("fill")
    fill_node.implementation = implementation
    fill_node.value = value

    # how to initialize memlet here?
    state.add_memlet_path(fill_node, result, src_conn="_output", memlet=Memlet())

    return sdfg


def run_test(target, size, value):
    if target == "pure":
        sdfg = pure_graph("pure", dace.float32, size)
        # expand the nested sdfg returned by fill node
        sdfg.expand_library_nodes()
    else:
        print(f"Unsupported target: {target}")
        exit(-1)

    # we get the function we can call
    fill = sdfg.compile()

    # supposed to be filled
    result = np.ndarray(size, dtype=np.float32)

    # the parameters are all the symbols defined in the sdfg
    fill(value=value, r=result)
    for val in result:
        if val != value:
            raise ValueError(f"expected {value}, found {val}")
    return sdfg


def test_fill_pure():
    # should not return a value error
    assert isinstance(run_test("pure", 64, 1), dace.SDFG)


if __name__ == "__main__":
    test_fill_pure()
