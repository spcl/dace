# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Fusing away a no-op START block must not re-resolve pattern nodes after removal.

``BlockFusion.apply`` removes the empty first block and then re-pins the start block on the
surviving second block. Pattern-node handles resolve by NODE ID on every attribute access, and
removing a node shifts the ids of every node inserted after it -- so touching
``self.second_block`` AFTER ``remove_node`` raised ``NodeNotFoundError`` on exactly the shape
below (an empty start state followed by the real one, as ``StateFusionExtended`` leaves behind
on the fibonacci consume sample).
"""
import numpy as np
import pytest

import dace
from dace.transformation.interstate import BlockFusion


def build_chain() -> dace.SDFG:
    sdfg = dace.SDFG('noop_start_chain')
    sdfg.add_array('a', [4], dace.float64)
    empty = sdfg.add_state('empty_start', is_start_block=True)
    work = sdfg.add_state('work')
    tasklet = work.add_tasklet('w', {}, {'out'}, 'out = 7.0')
    work.add_edge(tasklet, 'out', work.add_write('a'), None, dace.Memlet('a[0]'))
    sdfg.add_edge(empty, work, dace.InterstateEdge())
    return sdfg


def test_fusing_away_the_noop_start_block():
    sdfg = build_chain()
    applied = sdfg.apply_transformations_repeated(BlockFusion, validate=True)
    assert applied == 1, 'BlockFusion refused the empty start block'
    assert sdfg.number_of_nodes() == 1
    assert sdfg.start_block.label == 'work', 'start block must move to the surviving block'
    sdfg.validate()

    a = np.zeros(4)
    sdfg(a=a)
    assert a[0] == 7.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
