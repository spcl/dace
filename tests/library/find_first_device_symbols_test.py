# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The device search's predicate functor receives the symbols the predicate reads.

The functor is emitted into the device translation unit, where the host program's symbols are not in scope: a
predicate reading ``KFIND`` failed to compile (``use of undeclared identifier 'KFIND'``, tsvc_2_5 ext_break). Each
symbol of ``FindFirst.symbols`` is now a by-value member, like a scalar input.
"""

import dace
from dace.libraries.standard.nodes import FindFirst
from dace.libraries.standard.nodes.find_first import INDEX_NAME, OUTPUT_CONNECTOR_NAME, ExpandFindFirstCUDA

N, KFIND = dace.symbol("N"), dace.symbol("KFIND", dace.int32)


def device_search(name: str) -> tuple[dace.SDFG, dace.SDFGState, FindFirst]:
    """``out[0] = first i with a[i] > KFIND`` over a device array."""
    sdfg = dace.SDFG(name)
    sdfg.add_symbol("KFIND", dace.int32)
    sdfg.add_array("a", [N], dace.int32, storage=dace.StorageType.GPU_Global)
    sdfg.add_array("out", [1], dace.int64)
    state = sdfg.add_state("search", is_start_block=True)
    node = FindFirst("search", predicate=f"__r_a[{INDEX_NAME}] > KFIND", begin=0, end=N)
    node.symbols = ["KFIND"]
    node.implementation = "CUDA"
    state.add_node(node)
    node.add_in_connector("__r_a")
    state.add_edge(state.add_read("a"), None, node, "__r_a", dace.Memlet("a[0:N]"))
    state.add_edge(node, OUTPUT_CONNECTOR_NAME, state.add_write("out"), None, dace.Memlet("out[0]"))
    return sdfg, state, node


def test_the_device_predicate_takes_the_symbols_it_reads():
    sdfg, state, node = device_search("find_first_device_symbols")
    tasklet = ExpandFindFirstCUDA.expansion(node, state, sdfg)
    device = sdfg.global_code["cuda"].as_string
    assert "int KFIND;" in device, device  # a functor member
    assert "(int KFIND, const int *__r_a" in device, device  # a wrapper parameter
    assert "(KFIND, __r_a, (0)" in tasklet.code.as_string, tasklet.code.as_string  # passed at the call site


if __name__ == "__main__":
    test_the_device_predicate_takes_the_symbols_it_reads()
