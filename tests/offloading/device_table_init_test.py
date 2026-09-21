# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A small table the host fills and kernels read is filled on the device, not copied there mid-run.

The shape of CloudSC's ``imelt[0:5] = 2, 3, 4, 3, -99``: filled by free tasklets in one state, read by
kernels in later states. No state holds both the host write and a device read, so the hybrid resolution
never lifted the fill, and the copy analysis shipped the finished table down between two kernels.
CloudSC's ``iphase`` and ``zvqx`` are the same fill with interstate reads on top, so the host keeps its
own fill and the device fills a twin.
"""
from typing import Optional

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.offloading import OffloadToAccelerator

N = 16
#: What the fill writes, ``None`` standing for the by-value scalar ``s``.
TABLE = (1.0, 2.0, None, -3.0)


def fill_then_kernel(host_read: Optional[str] = None, looped: bool = False) -> dace.SDFG:
    """``table`` filled by free tasklets in one state, read as ``table[i % 4]`` by a map in the next.

    ``host_read`` adds a host read: ``interstate`` of ``table[1]`` in the edge between the states
    (CloudSC's ``zvqx``), ``tasklet`` of ``table[2]`` by a host tasklet after the kernel. ``looped`` puts
    the fill inside a loop.
    """
    sdfg = dace.SDFG(f'fill_then_kernel_{host_read}_{looped}')
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
        if value is None:
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


def offloaded(host_read: Optional[str] = None, looped: bool = False) -> dace.SDFG:
    sdfg = fill_then_kernel(host_read, looped)
    ppl.Pipeline([OffloadToAccelerator()]).apply_pass(sdfg, {})
    sdfg.validate()
    return sdfg


def table_copies(sdfg: dace.SDFG) -> list[tuple[str, str]]:
    """``(source, destination)`` of every host/device copy of the table."""
    found = []
    for state in sdfg.all_states():
        for edge in state.edges():
            if not (isinstance(edge.src, dace.nodes.AccessNode) and isinstance(edge.dst, dace.nodes.AccessNode)):
                continue
            if not edge.src.data.startswith('table'):
                continue
            sides = [n.desc(state.sdfg).storage in GPU_RESIDENT_STORAGES for n in (edge.src, edge.dst)]
            if sides[0] != sides[1]:
                found.append((edge.src.data, edge.dst.data))
    return found


def fill_scopes(sdfg: dace.SDFG) -> list:
    """The scope entry of every tasklet in the fill state: None for host code."""
    fill = next(state for state in sdfg.all_states() if state.label == 'fill')
    scopes = fill.scope_dict()
    return [scopes[node] for node in fill.nodes() if isinstance(node, dace.nodes.Tasklet)]


@pytest.mark.parametrize('host_read', [None, 'interstate', 'tasklet'])
def test_a_table_kernels_read_is_never_copied_to_the_device(host_read):
    copies = table_copies(offloaded(host_read))
    assert not copies, copies


@pytest.mark.parametrize('host_read', [None, 'interstate', 'tasklet'])
def test_the_device_fill_is_one_kernel_launch(host_read):
    """One size-1 kernel per state replaces the copy; a launch per element would trade one transfer for four."""
    kernels = {scope for scope in fill_scopes(offloaded(host_read)) if scope is not None}
    assert len(kernels) == 1, kernels
    assert next(iter(kernels)).map.schedule == dtypes.ScheduleType.GPU_Device


@pytest.mark.parametrize('host_read', ['interstate', 'tasklet'])
def test_a_table_the_host_reads_keeps_its_host_fill(host_read):
    """The host reads its own copy; filling only the device one would need the copy back."""
    host = [scope for scope in fill_scopes(offloaded(host_read)) if scope is None]
    assert len(host) == len(TABLE), host


def test_a_fill_inside_a_loop_stays_host_code():
    """A kernel there would launch once per iteration to replace one copy after the loop."""
    scopes = fill_scopes(offloaded(looped=True))
    assert scopes and all(scope is None for scope in scopes), scopes


@pytest.mark.gpu
@pytest.mark.parametrize('host_read', [None, 'interstate', 'tasklet'])
def test_the_table_filled_on_the_device_computes_what_numpy_computes(host_read):
    sdfg = fill_then_kernel(host_read)
    sdfg.apply_gpu_transformations()
    x = np.random.default_rng(3).random(N)
    s = 0.5
    y = np.zeros(N)
    first = np.zeros(1)
    sdfg(x=x, y=y, s=s, first=first)
    table = np.array([s if value is None else value for value in TABLE])
    np.testing.assert_array_equal(y, x * table[np.arange(N) % len(TABLE)])
    if host_read == 'tasklet':
        np.testing.assert_array_equal(first, [s])
