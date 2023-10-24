# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.sdfg import utils as sdutils


def test_loop_inlining_regular_for():
    sdfg = dace.SDFG('inlining')
    state0 = sdfg.add_state('state0', is_start_block=True)
    loop1 = LoopRegion('i', 'i = 0', 'i < 10', 'i = i + 1', 'loop1')
    sdfg.add_node(loop1)
    state1 = loop1.add_state('state1', is_start_block=True)
    state2 = loop1.add_state('state2')
    loop1.add_edge(state1, state2, dace.InterstateEdge())
    state3 = sdfg.add_state('state3')
    sdfg.add_edge(state0, loop1, dace.InterstateEdge())
    sdfg.add_edge(loop1, state3, dace.InterstateEdge())

    sdutils.inline_loop_blocks(sdfg)

    print(sdfg)


if __name__ == '__main__':
    test_loop_inlining_regular_for()