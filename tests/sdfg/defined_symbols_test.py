# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``SDFGState.defined_symbols`` types interstate assignments from one environment kept per call."""

from unittest import mock

import dace
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import LoopRegion


def assigning_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG("assigning")
    sdfg.add_symbol("N", dace.int64)
    sdfg.add_array("A", [dace.symbol("N")], dace.float32)
    sdfg.add_scalar("k", dace.float64, transient=True)
    first = sdfg.add_state("first", is_start_block=True)
    loop = LoopRegion("loop", "i < N", "i", "i = 0", "i = i + 1")
    loop.add_state("body", is_start_block=True)
    sdfg.add_node(loop)
    last = sdfg.add_state("last")
    # ``m`` is typed from ``N``, ``w`` from the array element, ``k`` shadows the scalar descriptor and
    # ``z`` is typed from ``k`` -- which must resolve to the DESCRIPTOR's dtype, not the symbol's.
    sdfg.add_edge(first, loop, InterstateEdge(assignments={"m": "N + 1", "w": "A[0]"}))
    sdfg.add_edge(loop, last, InterstateEdge(assignments={"k": "m * 2", "z": "k + 1"}))
    return sdfg


def test_defined_symbols_builds_no_type_environment_per_edge():
    """``new_symbols(sdfg, ...)`` copies the symbols and every array's dtype per assigning edge:
    2.4 s per call on warpx_field_gather, once per LoopToMap lift that nests a body."""
    sdfg = assigning_sdfg()
    state = next(s for s in sdfg.states() if s.label == "last")
    with mock.patch.object(
        InterstateEdge, "new_symbols", autospec=True, side_effect=InterstateEdge.new_symbols
    ) as new_symbols:
        state.defined_symbols()
    assert new_symbols.called
    assert [c for c in new_symbols.call_args_list if c.args[1] is not None] == []


if __name__ == "__main__":
    test_defined_symbols_builds_no_type_environment_per_edge()
