# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Maps the offloading keeps on the HOST, and the things it must never offload at all.

Structural tests: they assert which schedule each map came out with, so they do not need a GPU.
The numerical companion lives in ``offload_to_accelerator_graphs_test.py``.
"""
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.passes.offloading import OffloadToAccelerator
from dace.transformation.passes.offloading.host_maps import host_maps
from dace.transformation.passes.offloading.offloading_helpers import is_callback_tasklet

NB = dace.symbol("NB")
NLEV = dace.symbol("NLEV")
NPROMA = dace.symbol("NPROMA")


def dace_inhibitor(f):
    """Marks a function the frontend must NOT parse, so the call becomes a real callback."""
    return f


@dace.program
def icon_zekinh_gather(
    e_bln: dace.float64[NB, 3, NPROMA],
    edge_idx: dace.int32[NB, NPROMA, 3],
    edge_blk: dace.int32[NB, NPROMA, 3],
    z_kin_hor_e: dace.float64[NB, NLEV, NPROMA],
    z_ekinh: dace.float64[NB, NLEV, NPROMA],
):
    """ICON ``velocity_zekinh_block``: cell-from-edges bilinear interpolation.

    The canonical "parent map only launches" shape -- ``jb`` over blocks exists to launch the
    ``jk``/``jc`` work under it, and its body's extents do not mention ``jb``.
    """
    for jb in dace.map[0:NB]:
        for jk, jc in dace.map[0:NLEV, 0:NPROMA]:
            z_ekinh[jb, jk, jc] = (e_bln[jb, 0, jc] * z_kin_hor_e[edge_blk[jb, jc, 0], jk, edge_idx[jb, jc, 0]] +
                                   e_bln[jb, 1, jc] * z_kin_hor_e[edge_blk[jb, jc, 1], jk, edge_idx[jb, jc, 1]] +
                                   e_bln[jb, 2, jc] * z_kin_hor_e[edge_blk[jb, jc, 2], jk, edge_idx[jb, jc, 2]])


def map_schedules(sdfg: dace.SDFG) -> dict:
    """Every map in ``sdfg`` by label -> schedule, nested SDFGs included."""
    return {
        node.map.label: node.map.schedule
        for node, _ in sdfg.all_nodes_recursive() if isinstance(node, nodes.MapEntry)
    }


def outer_map_label(sdfg: dace.SDFG) -> str:
    """The label of the map over ``jb`` -- the one whose body is the kernel."""
    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.MapEntry) and parent.entry_node(node) is None and 'jb' in node.map.params:
            return node.map.label
    raise AssertionError('no top-level jb map in the gather kernel')


def zekinh_sdfg() -> dace.SDFG:
    return icon_zekinh_gather.to_sdfg(simplify=False)


def test_without_host_maps_the_outer_map_is_the_kernel():
    """The control: with nothing named, the outer map is offloaded and its body is sequential.

    Without this the pinning tests below could pass while doing nothing, because they would be
    asserting the schedule the pass produces anyway.
    """
    sdfg = zekinh_sdfg()
    outer = outer_map_label(sdfg)
    OffloadToAccelerator().apply_pass(sdfg, {})

    schedules = map_schedules(sdfg)
    assert schedules[outer] == dace.ScheduleType.GPU_Device
    inner = [s for label, s in schedules.items() if label != outer]
    assert inner, 'the gather kernel has maps under jb'
    assert all(s == dace.ScheduleType.Sequential for s in inner), schedules


def test_a_named_host_map_keeps_the_host_and_its_body_becomes_the_kernel():
    """The ICON shape: ``jb`` launches, so it stays host and ``jk``/``jc`` become the kernels."""
    sdfg = zekinh_sdfg()
    outer = outer_map_label(sdfg)
    OffloadToAccelerator(host_maps=[outer]).apply_pass(sdfg, {})

    schedules = map_schedules(sdfg)
    assert schedules[outer] != dace.ScheduleType.GPU_Device, schedules
    inner = [s for label, s in schedules.items() if label != outer]
    assert inner and all(s == dace.ScheduleType.GPU_Device for s in inner), schedules


def test_a_map_entry_object_pins_the_same_map_as_its_label():
    """``host_maps`` takes the node as readily as its name, and the two agree."""
    by_label = zekinh_sdfg()
    outer = outer_map_label(by_label)
    OffloadToAccelerator(host_maps=[outer]).apply_pass(by_label, {})

    by_node = zekinh_sdfg()
    entry = next(n for n, p in by_node.all_nodes_recursive() if isinstance(n, nodes.MapEntry) and n.map.label == outer)
    OffloadToAccelerator(host_maps=[entry]).apply_pass(by_node, {})

    assert map_schedules(by_label) == map_schedules(by_node)


def test_auto_finds_the_launching_map_that_the_default_offloads():
    """``host_maps=True`` derives the same answer the ICON shape has to be given by hand."""
    sdfg = zekinh_sdfg()
    outer = outer_map_label(sdfg)
    OffloadToAccelerator(host_maps=True).apply_pass(sdfg, {})

    schedules = map_schedules(sdfg)
    assert schedules[outer] != dace.ScheduleType.GPU_Device, schedules
    inner = [s for label, s in schedules.items() if label != outer]
    assert inner and all(s == dace.ScheduleType.GPU_Device for s in inner), schedules


def test_auto_declines_a_map_that_does_its_own_work():
    """The negative control: a parent map that COMPUTES is not a launcher, so auto leaves it alone.

    Without this, ``test_auto_finds_...`` would also pass an implementation that called every outer
    map a host map.
    """

    @dace.program
    def computes_at_the_top(A: dace.float64[16, 8], B: dace.float64[16, 8]):
        for i in dace.map[0:16]:
            A[i, 0] = A[i, 0] + 1.0  # real work in the outer scope, not a launch
            for j in dace.map[0:8]:
                B[i, j] = A[i, j] * 2.0

    sdfg = computes_at_the_top.to_sdfg(simplify=False)
    outer = next(n.map.label for n, p in sdfg.all_nodes_recursive()
                 if isinstance(n, nodes.MapEntry) and p.entry_node(n) is None and 'i' in n.map.params)
    assert host_maps(sdfg, True) == host_maps(sdfg, None), 'a computing map is not auto-detected'

    OffloadToAccelerator(host_maps=True).apply_pass(sdfg, {})
    assert map_schedules(sdfg)[outer] == dace.ScheduleType.GPU_Device


def test_the_spellings_that_name_no_host_maps_agree() -> None:
    """``False`` is the default and runs no heuristics; ``None`` and ``[]`` say the same thing.

    ``[]`` matters on its own: a caller that computes the list and finds it empty must get "none",
    not "derive them for me".
    """
    sdfg = zekinh_sdfg()
    for spec in (False, None, []):
        assert not host_maps(sdfg, spec), f'{spec!r} must name no host maps'
    assert host_maps(sdfg, True), 'True runs the heuristics'


def test_host_maps_rejects_anything_that_is_not_a_label_or_a_map():
    sdfg = zekinh_sdfg()
    with pytest.raises(TypeError, match='map labels or MapEntry nodes'):
        host_maps(sdfg, [object()])
    with pytest.raises(TypeError, match='None, a bool or a list'):
        host_maps(sdfg, 'nblks')


def test_apply_gpu_transformations_forwards_host_maps() -> None:
    """The public entry point reaches the same decision the pass does.

    ``apply_gpu_transformations`` is how callers outside this package offload, so a ``host_maps``
    the pass honours but the method drops is a feature nobody can use. Asserting the two agree keeps
    the parameter plumbed.
    """
    through_pass = zekinh_sdfg()
    outer = outer_map_label(through_pass)
    OffloadToAccelerator(host_maps=[outer]).apply_pass(through_pass, {})

    through_method = zekinh_sdfg()
    through_method.apply_gpu_transformations(host_maps=[outer], validate=False, simplify=False)

    assert map_schedules(through_method) == map_schedules(through_pass)
    assert map_schedules(through_method)[outer] != dace.ScheduleType.GPU_Device


def test_apply_gpu_transformations_without_host_maps_offloads_the_outer_map() -> None:
    """The control: the parameter defaults to naming nothing, so the outer map is still the kernel."""
    sdfg = zekinh_sdfg()
    outer = outer_map_label(sdfg)
    sdfg.apply_gpu_transformations(validate=False, simplify=False)
    assert map_schedules(sdfg)[outer] == dace.ScheduleType.GPU_Device


def test_a_frontend_callback_is_never_offloaded():
    """Neither kind of callback can run on the device, so nothing around one is offloaded.

    This is what ``host_data=['__pystate']`` and ``exclude_tasklets`` were pinning by hand against
    the transformation this pass replaces: the pass derives it from the callback itself.
    """

    @dace_inhibitor
    def host_only(x):
        return x * 2.0

    @dace.program
    def calls_back(A: dace.float64[8], B: dace.float64[8]):
        for i in range(8):
            B[i] = host_only(A[i])

    sdfg = calls_back.to_sdfg(simplify=False)
    assert any(isinstance(stype, dace.dtypes.callback)
               for stype in sdfg.symbols.values()), 'the fixture must produce a real callback'
    callbacks = [
        node for node, parent in sdfg.all_nodes_recursive()
        if isinstance(node, nodes.Tasklet) and is_callback_tasklet(node, sdfg)
    ]
    assert callbacks, 'the detector must recognise a frontend-generated callback'

    OffloadToAccelerator().apply_pass(sdfg, {})
    sdfg.validate()

    for label, schedule in map_schedules(sdfg).items():
        assert schedule != dace.ScheduleType.GPU_Device, f'map {label} around a callback was offloaded'


def callback_in_a_map_sdfg() -> dace.SDFG:
    """A map whose body calls back into Python, wired the way the frontend wires one.

    Built by hand because the frontend declines to create a callback whose result is assigned
    inside a ``dace.map`` -- the shape still has to be handled, however it is reached.
    """
    sdfg = dace.SDFG('callback_in_a_map')
    sdfg.add_array('A', [8], dace.float64)
    sdfg.add_array('B', [8], dace.float64)
    sdfg.add_scalar('__pystate', dace.int32, transient=True)
    sdfg.add_symbol('host_only', dace.dtypes.callback(dace.float64, dace.float64))

    state = sdfg.add_state('body', is_start_block=True)
    entry, exit_ = state.add_map('over_i', {'i': '0:8'})
    task = state.add_tasklet('call_host', {'__inp', '__istate'}, {'__out', '__ostate'}, '__out = host_only(__inp)')
    state.add_memlet_path(state.add_read('A'), entry, task, dst_conn='__inp', memlet=dace.Memlet('A[i]'))
    state.add_memlet_path(task, exit_, state.add_write('B'), src_conn='__out', memlet=dace.Memlet('B[i]'))
    state.add_edge(state.add_read('__pystate'), None, task, '__istate', dace.Memlet('__pystate[0]'))
    state.add_edge(task, '__ostate', state.add_write('__pystate'), None, dace.Memlet('__pystate[0]'))
    sdfg.validate()
    return sdfg


def test_a_map_around_a_callback_stays_on_the_host():
    """A kernel cannot issue a callback, so the map holding one is host code, not a launch."""
    sdfg = callback_in_a_map_sdfg()
    entry = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))
    assert entry in host_maps(sdfg, None), 'a map around a callback is host code without being named'

    OffloadToAccelerator().apply_pass(sdfg, {})
    sdfg.validate()
    assert map_schedules(sdfg)['over_i'] != dace.ScheduleType.GPU_Device


def test_a_sequential_scan_in_a_loop_region_is_not_offloaded():
    """``__1d_scan``: a scan carries its value across iterations, so it has no parallel map in it.

    The transformation this pass replaces wrapped the loop body's tasklet in a size-1 GPU map, which
    is one kernel launch per scan step. Nothing here may become a map at all.
    """
    sdfg = dace.SDFG('scan_1d')
    sdfg.add_array('out', [16], dace.float64)
    sdfg.add_scalar('carry', dace.float64, transient=True)

    init = sdfg.add_state('init', is_start_block=True)
    seed = init.add_tasklet('seed', {}, {'c'}, 'c = 0.0')
    init.add_edge(seed, 'c', init.add_write('carry'), None, dace.Memlet('carry[0]'))

    loop = dace.sdfg.state.LoopRegion('scan', 'k < 16', 'k', 'k = 0', 'k = k + 1')
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge())
    body = loop.add_state('scan_compute', is_start_block=True)
    step = body.add_tasklet('accumulate', {'c_in'}, {'c_out', 'o'}, 'c_out = c_in + 1.0\no = c_in + 1.0')
    body.add_edge(body.add_read('carry'), None, step, 'c_in', dace.Memlet('carry[0]'))
    body.add_edge(step, 'c_out', body.add_write('carry'), None, dace.Memlet('carry[0]'))
    body.add_edge(step, 'o', body.add_write('out'), None, dace.Memlet('out[k]'))
    sdfg.validate()

    OffloadToAccelerator().apply_pass(sdfg, {})
    sdfg.validate()

    maps = [n.map.label for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]
    assert not maps, f'a sequential scan must not gain a map, got {maps}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
