# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Replacing a scalar with a constant keeps the scalar's type.

``replace_dict`` turns a read of a non-transient scalar whose value is known into a tasklet writing
that value into a container of its own. Under the nested SDFG contract (see
``dace.sdfg.dealias.integrate_nested_sdfg``) a connector reading that scalar is the container it is
connected to, so a container typed after the constant rather than after the scalar leaves the two
disagreeing.
"""
import numpy as np

import dace
from dace import symbolic


def _adding_body():
    """``o[i] = s + 1`` over a ``float32`` scalar connector."""
    sdfg = dace.SDFG('body')
    sdfg.add_scalar('s', dace.float32)
    sdfg.add_array('o', [4], dace.float32)
    sdfg.add_symbol('i', dace.int64)
    state = sdfg.add_state()
    tasklet = state.add_tasklet('t', {'x'}, {'y'}, 'y = x + 1')
    state.add_edge(state.add_read('s'), None, tasklet, 'x', dace.Memlet('s[0]'))
    state.add_edge(tasklet, 'y', state.add_write('o'), None, dace.Memlet('o[i]'))
    return sdfg


def test_replaced_scalar_keeps_its_type():
    sdfg = dace.SDFG('replace_constant_scalar')
    sdfg.add_scalar('s', dace.float32)
    sdfg.add_array('O', [4], dace.float32)
    state = sdfg.add_state()
    entry, exit_ = state.add_map('m', dict(i='0:4'))
    node = state.add_nested_sdfg(_adding_body(), {'s'}, {'o'}, {'i': 'i'})
    state.add_memlet_path(state.add_read('s'), entry, node, dst_conn='s', memlet=dace.Memlet('s[0]'))
    state.add_memlet_path(node, exit_, state.add_write('O'), src_conn='o', memlet=dace.Memlet('O[i]'))
    sdfg.validate()

    sdfg.replace_dict({}, symrepl={symbolic.pystr_to_symbolic('s'): symbolic.pystr_to_symbolic('5')})

    edge = next(e for e in state.in_edges(node) if e.dst_conn == 's')
    assert sdfg.arrays[edge.data.data].dtype == dace.float32
    assert node.sdfg.arrays['s'].is_equivalent(sdfg.arrays[edge.data.data])
    sdfg.validate()

    O = np.zeros(4, dtype=np.float32)
    # ``s`` is still an argument of the SDFG; the constant is what the body actually reads.
    sdfg(s=np.float32(0), O=O)
    assert np.allclose(O, 6.0)


if __name__ == '__main__':
    test_replaced_scalar_keeps_its_type()
