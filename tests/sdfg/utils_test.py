# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import dace

from dace.sdfg import utils


def test_traverse_sdfg_with_defined_symbols():
    sdfg = dace.SDFG("tester")
    sdfg.add_symbol("my_symbol", dace.int32)

    start = sdfg.add_state("start", is_start_block=True)
    start.add_tasklet("noop", set(), set(), "")
    sdfg.add_state_after(start, "next")

    for _state, _node, defined_symbols in utils.traverse_sdfg_with_defined_symbols(sdfg):
        assert "my_symbol" in defined_symbols


if __name__ == "__main":
    test_traverse_sdfg_with_defined_symbols()
