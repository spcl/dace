# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import pytest
import numpy as np

import addlib
import dace
import dace.library
from dace.sdfg import infer_types


@pytest.mark.gpu
@pytest.mark.parametrize('input_array', [True, False])
@pytest.mark.parametrize('output_array', [True, False])
@pytest.mark.parametrize('expand_first', [True, False])
def test_gpu(input_array, output_array, expand_first):

    sdfg = dace.SDFG("test_gpu_scalars")
    state = sdfg.add_state()

    if input_array:
        # input_arr is an unsqueezed scalar
        sdfg.add_array("input_arr", [1], dace.float32)
    else:
        sdfg.add_scalar("input_arr", dace.float32)

    if output_array:
        # output_arr is an unsqueezed scalar
        sdfg.add_array("output_arr", [1], dace.float32)
    else:
        sdfg.add_scalar("transient_output_arr", dace.float32, transient=True)
        sdfg.add_array("output_arr", [1], dace.float32)

    inp = state.add_access("input_arr")
    addnode = addlib.AddNode("add")
    state.add_node(addnode)
    outp = state.add_access("output_arr")

    state.add_edge(inp, None, addnode, "_a", dace.Memlet("input_arr"))

    if output_array:
        state.add_edge(addnode, "_b", outp, None, dace.Memlet("output_arr"))
    else:
        transient_outp = state.add_access("transient_output_arr")
        state.add_edge(addnode, "_b", transient_outp, None, sdfg.make_array_memlet("transient_output_arr"))
        state.add_edge(transient_outp, None, outp, None, sdfg.make_array_memlet("transient_output_arr"))

    sdfg.apply_gpu_transformations()

    if expand_first:
        sdfg.expand_library_nodes()
        infer_types.infer_connector_types(sdfg)
    else:
        infer_types.infer_connector_types(sdfg)
        sdfg.expand_library_nodes()
        infer_types.infer_connector_types(sdfg)

    input_arr = np.array([1]).astype(np.float32) if input_array else 1
    output_arr = np.array([0]).astype(np.float32)
    sdfg(input_arr=input_arr, output_arr=output_arr)
    assert output_arr[0] == 2


if __name__ == '__main__':
    test_gpu(True, True, False)
    test_gpu(True, True, True)
    test_gpu(False, False, False)
    test_gpu(False, False, True)
