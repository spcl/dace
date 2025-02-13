# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.sdfg import InvalidSDFGError
import pytest

sdfg = dace.SDFG('const_access_test')
sdfg.add_array('A', [1], dace.float64)
state = sdfg.add_state()

nsdfg = dace.SDFG('nested')
nsdfg.add_array('a', [1], dace.float64)
nstate = nsdfg.add_state()
t = nstate.add_tasklet('add', {'inp'}, {'out'}, 'out = inp + inp')
nstate.add_edge(nstate.add_read('a'), None, t, 'inp', dace.Memlet.simple('a', '0'))
nstate.add_edge(t, 'out', nstate.add_write('a'), None, dace.Memlet.simple('a', '0'))

nsdfg_node = state.add_nested_sdfg(nsdfg, None, {'a'}, {})
state.add_edge(state.add_read('A'), None, nsdfg_node, 'a', dace.Memlet.simple('A', '0'))


def test():
    with pytest.raises(InvalidSDFGError):
        sdfg.validate()

if __name__ == '__main__':
    test()
