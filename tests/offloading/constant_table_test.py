# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A table filled by tasklets and read by kernels computes what numpy computes once offloaded.

The shape of CloudSC's ``imelt[0:5] = 2, 3, 4, 3, -99``: filled by input-less tasklets in one state, read by
kernels in later states.
"""
from typing import Optional

import numpy as np
import pytest

import dace

from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.offloading import OffloadToAccelerator

N = 16
TABLE = (1.0, 2.0, 0.5, -3.0)


def fill_then_kernel(host_read: Optional[str] = None, scalar_entry: bool = False) -> dace.SDFG:
    """``table`` filled by tasklets in one state, read as ``table[i % 4]`` by a map in the next.

    ``host_read`` adds a host read: ``interstate`` of ``table[1]`` on the edge between the states, ``tasklet`` of
    ``table[2]`` by a host tasklet after the kernel. ``scalar_entry`` fills
    one element from the scalar ``s`` instead of a literal.
    """
    sdfg = dace.SDFG(f'fill_then_kernel_{host_read}_{scalar_entry}')
    sdfg.add_array('x', [N], dace.float64)
    sdfg.add_array('y', [N], dace.float64)
    sdfg.add_array('first', [1], dace.float64)
    sdfg.add_scalar('s', dace.float64)
    sdfg.add_array('table', [len(TABLE)], dace.float64, transient=True)

    fill = sdfg.add_state('fill', is_start_block=True)
    table = fill.add_write('table')
    for index, value in enumerate(TABLE):
        if scalar_entry and index == 2:
            tasklet = fill.add_tasklet(f'fill_{index}', {'inp'}, {'out'}, 'out = inp')
            fill.add_edge(fill.add_read('s'), None, tasklet, 'inp', dace.Memlet('s[0]'))
        else:
            tasklet = fill.add_tasklet(f'fill_{index}', {}, {'out'}, f'out = {value}')
        fill.add_edge(tasklet, 'out', table, None, dace.Memlet(f'table[{index}]'))

    use = sdfg.add_state('use')
    use.add_mapped_tasklet('scale', {'i': f'0:{N}'}, {
        'x_in': dace.Memlet('x[i]'),
        't_in': dace.Memlet(f'table[i % {len(TABLE)}]')
    },
                           'o = x_in * t_in', {'o': dace.Memlet('y[i]')},
                           external_edges=True)
    sdfg.add_edge(fill, use, dace.InterstateEdge(assignments={'k': 'table[1]'} if host_read == 'interstate' else {}))
    if host_read == 'tasklet':
        after = sdfg.add_state_after(use, 'after')
        tasklet = after.add_tasklet('peek', {'inp'}, {'out'}, 'out = inp')
        after.add_edge(after.add_read('table'), None, tasklet, 'inp', dace.Memlet('table[2]'))
        after.add_edge(tasklet, 'out', after.add_write('first'), None, dace.Memlet('first[0]'))
    sdfg.validate()
    return sdfg


def offloaded(**options) -> dace.SDFG:
    sdfg = fill_then_kernel(**options)
    ppl.Pipeline([OffloadToAccelerator()]).apply_pass(sdfg, {})
    sdfg.validate()
    return sdfg


@pytest.mark.gpu
@pytest.mark.parametrize('host_read', [None, 'interstate', 'tasklet'])
def test_a_constant_table_read_in_a_kernel_computes_what_numpy_computes(host_read):
    sdfg = offloaded(host_read=host_read)
    x = np.random.default_rng(3).random(N)
    y = np.zeros(N)
    first = np.zeros(1)
    sdfg(x=x, y=y, s=0.5, first=first)
    np.testing.assert_array_equal(y, x * np.array(TABLE)[np.arange(N) % len(TABLE)])
    if host_read == 'tasklet':
        np.testing.assert_array_equal(first, [TABLE[2]])


@pytest.mark.gpu
def test_a_table_filled_from_a_scalar_computes_what_numpy_computes():
    sdfg = offloaded(scalar_entry=True)
    x = np.random.default_rng(3).random(N)
    y = np.zeros(N)
    sdfg(x=x, y=y, s=0.5, first=np.zeros(1))
    table = np.array([0.5 if index == 2 else value for index, value in enumerate(TABLE)])
    np.testing.assert_array_equal(y, x * table[np.arange(N) % len(TABLE)])
