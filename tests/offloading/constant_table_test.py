# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A table filled once with literals and read by kernels becomes an SDFG constant, so no copy ships it down.

The shape of CloudSC's ``imelt[0:5] = 2, 3, 4, 3, -99``. A constant is declared on both sides.
"""

import sys

import numpy as np
import pytest

import dace

from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.offloading import OffloadToAccelerator

N = 16
TABLE = (1.0, 2.0, 0.5, -3.0)


def fill_then_kernel(host_read: str | None = None, looped: bool = False, scalar_entry: bool = False) -> dace.SDFG:
    """``table`` filled by tasklets in one state, read as ``table[i % 4]`` by a map in the next.

    ``host_read`` adds a host read of ``table[1]`` on the edge between the states (``interstate``) or of ``table[2]``
    by a tasklet after the kernel (``tasklet``); ``looped`` fills inside a loop; ``scalar_entry`` fills one element
    from the scalar ``s``.
    """
    sdfg = dace.SDFG(f'fill_then_kernel_{host_read}_{looped}_{scalar_entry}')
    sdfg.add_array('x', [N], dace.float64)
    sdfg.add_array('y', [N], dace.float64)
    sdfg.add_array('first', [1], dace.float64)
    sdfg.add_scalar('s', dace.float64)
    sdfg.add_array('table', [len(TABLE)], dace.float64, transient=True)

    if looped:
        region = LoopRegion('fill_loop', 'r < 2', 'r', 'r = 0', 'r = r + 1')
        sdfg.add_node(region, is_start_block=True)
        fill = region.add_state('fill', is_start_block=True)
    else:
        region = fill = sdfg.add_state('fill', is_start_block=True)
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
    sdfg.add_edge(region, use, dace.InterstateEdge(assignments={'k': 'table[1]'} if host_read == 'interstate' else {}))
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


def table_copies(sdfg: dace.SDFG) -> list:
    return [
        (edge.src.data, edge.dst.data) for state in sdfg.all_states() for edge in state.edges()
        if isinstance(edge.src, dace.nodes.AccessNode) and isinstance(edge.dst, dace.nodes.AccessNode) and 'table' in (
            edge.src.data + edge.dst.data)
    ]


HOST_READS = [None, 'interstate', 'tasklet']


@pytest.mark.parametrize('host_read', HOST_READS)
def test_a_literal_table_is_a_constant_and_never_copied(host_read):
    sdfg = offloaded(host_read=host_read)
    assert list(sdfg.constants['table']) == list(TABLE)
    assert not table_copies(sdfg), table_copies(sdfg)
    fill = next(state for state in sdfg.all_states() if state.label == 'fill')
    assert not [node for node in fill.nodes() if isinstance(node, dace.nodes.Tasklet)], 'the fill still runs'


UNFOLDED_OPTIONS = [{'looped': True}, {'scalar_entry': True}]


@pytest.mark.parametrize('options', UNFOLDED_OPTIONS)
def test_a_table_not_filled_once_with_literals_stays_an_array(options):
    """A fill that runs more than once, or one that reads a runtime value, is not a compile-time constant."""
    sdfg = offloaded(**options)
    assert 'table' not in sdfg.constants
    assert table_copies(sdfg), 'the kernel needs the host-filled table copied down'


@pytest.mark.gpu
@pytest.mark.parametrize('host_read', HOST_READS)
def test_a_constant_table_read_in_a_kernel_computes_what_numpy_computes(host_read):
    sdfg = offloaded(host_read=host_read)
    x = np.random.default_rng(3).random(N)
    y = np.zeros(N)
    first = np.zeros(1)
    sdfg(x=x, y=y, s=0.5, first=first)
    np.testing.assert_array_equal(y, x * np.array(TABLE)[np.arange(N) % len(TABLE)])
    if host_read == 'tasklet':
        np.testing.assert_array_equal(first, [TABLE[2]])


def test_a_constant_table_is_declared_once_per_side_and_never_passed_to_a_kernel():
    """Host code and the kernel each declare the table once, and the kernel takes it as no argument."""
    sdfg = offloaded()
    objects = sdfg.generate_code()
    host = next(obj.clean_code for obj in objects if '__program_' in obj.clean_code and obj.language == 'cpp')
    device = next(obj.clean_code for obj in objects if obj.language == 'cu')
    declaration = 'double table[4] = {'
    assert host.count(declaration) == 1, host
    kernel = device[device.index('__global__'):]
    assert kernel.count(declaration) == 1, kernel
    signature = kernel[:kernel.index('{')]
    assert 'table' not in signature, signature


@pytest.mark.gpu
def test_a_table_filled_from_a_scalar_computes_what_numpy_computes():
    sdfg = offloaded(scalar_entry=True)
    x = np.random.default_rng(3).random(N)
    y = np.zeros(N)
    sdfg(x=x, y=y, s=0.5, first=np.zeros(1))
    table = np.array([0.5 if index == 2 else value for index, value in enumerate(TABLE)])
    np.testing.assert_array_equal(y, x * table[np.arange(N) % len(TABLE)])


if __name__ == '__main__':
    for host_read in HOST_READS:
        test_a_literal_table_is_a_constant_and_never_copied(host_read)
    for options in UNFOLDED_OPTIONS:
        test_a_table_not_filled_once_with_literals_stays_an_array(options)
    test_a_constant_table_is_declared_once_per_side_and_never_passed_to_a_kernel()
    if len(sys.argv) > 1 and sys.argv[1] == 'gpu':
        for host_read in HOST_READS:
            test_a_constant_table_read_in_a_kernel_computes_what_numpy_computes(host_read)
        test_a_table_filled_from_a_scalar_computes_what_numpy_computes()
