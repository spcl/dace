# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""MapToForLoop leaves the CFG list exactly as a fresh rebuild would, and rebuilds it only where its
own steps do not: the inline that flattens the loop already ends with a rebuild."""
from unittest import mock

import dace
from dace.sdfg.state import AbstractControlFlowRegion, LoopRegion
from dace.transformation.dataflow import MapToForLoop


def two_map_sdfg() -> dace.SDFG:
    """Two states, each with a 1-D map writing ``a``."""
    sdfg = dace.SDFG('maptoforloop_cfg_list')
    sdfg.add_array('a', [10], dace.float64)
    previous = None
    for index in range(2):
        state = sdfg.add_state(f's{index}', is_start_block=(previous is None))
        if previous is not None:
            sdfg.add_edge(previous, state, dace.InterstateEdge())
        previous = state
        state.add_mapped_tasklet(f'm{index}',
                                 map_ranges={'i': '0:10'},
                                 inputs={},
                                 outputs={'__out': dace.Memlet('a[i]')},
                                 code=f'__out = {index + 1}.0',
                                 external_edges=True)
    sdfg.validate()
    return sdfg


def fresh_cfg_list(sdfg: dace.SDFG) -> list:
    return list(sdfg.all_control_flow_regions(recursive=True))


def test_cfg_list_is_current_after_each_lowering():
    sdfg = two_map_sdfg()
    for _ in range(2):
        assert sdfg.apply_transformations(MapToForLoop, validate=False) == 1
        assert sdfg.cfg_list == fresh_cfg_list(sdfg)
        assert all(cfg.cfg_list is sdfg.cfg_list for cfg in sdfg.cfg_list)
    loops = [cfg for cfg in sdfg.cfg_list if isinstance(cfg, LoopRegion)]
    assert len(loops) == 2 and all(loop.parent_graph is sdfg for loop in loops)
    sdfg.validate()
    a = dace.ndarray([10], dace.float64)
    a[:] = 0
    sdfg(a=a)
    assert (a == 2.0).all()


def test_inlined_lowering_does_not_rebuild_after_the_inline():
    sdfg = two_map_sdfg()
    original = AbstractControlFlowRegion.reset_cfg_list
    with mock.patch.object(AbstractControlFlowRegion, 'reset_cfg_list', autospec=True, side_effect=original) as spy:
        assert sdfg.apply_transformations(MapToForLoop, validate=False) == 1
    # nest_state_subgraph (nested SDFG added + its own rebuild), the loop region added to the nested
    # SDFG, and the region the inline hoists plus the inline's closing rebuild; nothing after that.
    assert spy.call_count <= 5, spy.call_count
    assert sdfg.cfg_list == fresh_cfg_list(sdfg)


if __name__ == '__main__':
    test_cfg_list_is_current_after_each_lowering()
    test_inlined_lowering_does_not_rebuild_after_the_inline()
