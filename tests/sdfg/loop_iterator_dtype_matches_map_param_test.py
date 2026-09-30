# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" LoopToMap declares the nested iterator at the loop iterator's own dtype, so one name keeps one dtype. """
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate import LoopToMap


def counting_loop_sdfg(bound_dtype: dace.typeclass) -> dace.SDFG:
    sdfg = dace.SDFG('counting_loop')
    sdfg.add_symbol('M', bound_dtype)
    sdfg.add_array('a', [dace.symbol('M', bound_dtype)], dace.float64)
    loop = LoopRegion('walk', 'i < M', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    state = loop.add_state('body', is_start_block=True)
    tasklet = state.add_tasklet('t', {'__in'}, {'__out'}, '__out = __in + 1.0')
    state.add_edge(state.add_read('a'), None, tasklet, '__in', dace.Memlet('a[i]'))
    state.add_edge(tasklet, '__out', state.add_write('a'), None, dace.Memlet('a[i]'))
    return sdfg


@pytest.mark.parametrize('bound_dtype', [dace.int32, dace.int64])
def test_loop_to_map_declares_the_nested_iterator_at_the_loop_iterator_dtype(bound_dtype):
    sdfg = counting_loop_sdfg(bound_dtype)
    loop = next(r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion))
    loop_dtype = loop.new_symbols(dict(sdfg.symbols))['i']
    assert sdfg.apply_transformations(LoopToMap) == 1
    declared = {s.name: s.symbols['i'] for s in sdfg.all_sdfgs_recursive() if 'i' in s.symbols}
    assert declared and all(dtype == loop_dtype for dtype in declared.values()), declared
    sdfg.validate()


if __name__ == '__main__':
    for dtype in (dace.int32, dace.int64):
        test_loop_to_map_declares_the_nested_iterator_at_the_loop_iterator_dtype(dtype)
