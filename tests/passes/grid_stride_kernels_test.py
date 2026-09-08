# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The launch shape a canonicalized GPU graph lands on, and the two answers to it.

``GridStrideKernels`` is the only pass that compares a kernel's grid against the DEVICE, and its
gate is CONTENTION, not size: a wide grid is folded only when every block of it updates the same
accumulator. That gate came out of measurement rather than taste (the module docstring carries the
numbers), so the tests below pin the shape of the gate itself as hard as they pin the rewrite -- a
kernel with no contention, and one whose conflict resolution scatters, must both come out untouched.

The second answer is a note. Where the kernel sits in a sequential loop the grid is one front and
there is nothing to fold; what would remove that cost is a persistent kernel with a grid-wide
barrier, which this pass does not do and records instead. The recording half is tested as carefully
as the rewriting half on purpose: a hint is the only trace a finding leaves, so a hint that stops
being written, or that is written twice, loses or corrupts the finding with nothing failing.
"""
import pathlib
import tempfile

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.dataflow.add_threadblock_map import to_3d_dims, validate_block_size_limits
from dace.transformation.passes.gpu_specialization import grid_stride_kernels
from dace.transformation.passes.gpu_specialization.grid_stride_kernels import (DEVICE_GRID_BLOCKS,
                                                                               PERSISTENT_KERNEL_LEAD,
                                                                               GridStrideKernels)

N = dace.symbol('N', dtype=dace.int64)
BLOCK = 256
SUM = 'lambda x, y: x + y'


def tiled_kernel(state, extent, block: int = BLOCK, accumulate: str = 'acc[0]', step: int = 1):
    """A ``(GPU_Device, GPU_ThreadBlock)`` pair in the shape ``AddThreadBlockMaps`` leaves behind.

    The outer map strides by exactly one block and the inner map covers one block, which is what
    lets the pass read the block extent off the pair instead of choosing a new one.
    ``accumulate`` is where the body writes: an ``acc`` subset with a conflict resolution is the
    contention the pass gates on, and ``'b[i]'`` is the plain streaming write that is not.

    ``step`` is the step of the map BEFORE strip-mining, which one thread still covers afterwards:
    the outer then strides ``block * step`` index units for ``block`` threads.
    """
    span = block * step
    outer_e, outer_x = state.add_map('grid', {'bi': f'0:{extent}:{span}'}, schedule=dtypes.ScheduleType.GPU_Device)
    inner_e, inner_x = state.add_map('block', {'i': f'bi:Min({extent} - 1, bi + {span - 1}) + 1:{step}'},
                                     schedule=dtypes.ScheduleType.GPU_ThreadBlock)
    tasklet = state.add_tasklet('t', {'__in'}, {'__out'}, '__out = __in * 2.0')
    state.add_memlet_path(state.add_read('a'), outer_e, inner_e, tasklet, dst_conn='__in', memlet=dace.Memlet('a[i]'))
    written = accumulate.split('[')[0]
    memlet = dace.Memlet(accumulate, wcr=SUM) if written == 'acc' else dace.Memlet(accumulate)
    state.add_memlet_path(tasklet, inner_x, outer_x, state.add_write(written), src_conn='__out', memlet=memlet)
    return outer_e


def straight_line(extent, block: int = BLOCK, accumulate: str = 'acc[0]', step: int = 1) -> dace.SDFG:
    """One tiled kernel in a single state, nothing around it."""
    sdfg = dace.SDFG('gs_straight')
    for name in ('a', 'b'):
        sdfg.add_array(name, [extent], dace.float64)
    sdfg.add_array('acc', [BLOCK], dace.float64)
    state = sdfg.add_state('s', is_start_block=True)
    tiled_kernel(state, extent, block, accumulate, step)
    sdfg.validate()
    return sdfg


def inside_loop(extent, trips: str = 'N') -> dace.SDFG:
    """The launch-bound shape: one accumulating tiled kernel re-entered by a sequential loop."""
    sdfg = dace.SDFG('gs_looped')
    for name in ('a', 'b'):
        sdfg.add_array(name, [extent], dace.float64)
    sdfg.add_array('acc', [BLOCK], dace.float64)
    loop = LoopRegion('front', f't < {trips}', 't', 't = 0', 't = t + 1')
    sdfg.add_node(loop, is_start_block=True)
    tiled_kernel(loop.add_state('body', is_start_block=True), extent)
    sdfg.validate()
    return sdfg


def kernels(sdfg: dace.SDFG):
    """Every ``GPU_Device`` map entry."""
    return [
        n for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.MapEntry) and n.map.schedule == dtypes.ScheduleType.GPU_Device
    ]


def loops(sdfg: dace.SDFG):
    return [
        b for cfg in sdfg.all_control_flow_regions(recursive=True) for b in cfg.nodes() if isinstance(b, LoopRegion)
    ]


def test_a_contended_grid_far_larger_than_the_device_is_folded_onto_the_device():
    """The measured win: 2e6 blocks atomically updating one accumulator become a device-sized grid."""
    sdfg = straight_line(N)
    assert GridStrideKernels().apply_pass(sdfg, {}) == (1, 0)
    grid = kernels(sdfg)[0].map.range.num_elements()
    assert dace.symbolic.evaluate(grid, {N: 100_000_000}) == DEVICE_GRID_BLOCKS
    # The point of the fold: a grid that no longer grows with the domain, so neither does the number
    # of atomic contributions to the accumulator.
    assert dace.symbolic.evaluate(grid, {N: 10 * 100_000_000}) == DEVICE_GRID_BLOCKS
    sequential = [
        n for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.MapEntry) and n.map.schedule == dtypes.ScheduleType.Sequential
    ]
    assert len(sequential) == 2, 'the fold must leave a strided loop over the original domain'
    sdfg.validate()


def test_a_kernel_with_nothing_to_contend_for_is_left_alone():
    """A plain streaming write is HBM-bound; block dispatch was never what it waited for. MEASURED
    at 0.99x to 1.19x when the fold was applied to this family anyway: nothing to gain, and five of
    twenty-one materially slower."""
    sdfg = straight_line(N, accumulate='b[i]')
    before = str(kernels(sdfg)[0].map.range)
    assert GridStrideKernels().apply_pass(sdfg, {}) is None
    assert str(kernels(sdfg)[0].map.range) == before


def test_a_conflict_resolution_that_scatters_is_not_contention():
    """``out[ix[i]] += ...`` contends with nothing in particular: ``scatter_accum_dup`` has one and
    measured 1.01x, so a wcr over a wide subset must not be read as an accumulator."""
    sdfg = straight_line(N, accumulate='b[0:N]')
    entry = kernels(sdfg)[0]
    exit_node = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapExit))
    for state in sdfg.states():
        for edge in state.in_edges(exit_node) + state.out_edges(exit_node):
            edge.data.wcr = SUM
    before = str(entry.map.range)
    assert GridStrideKernels().apply_pass(sdfg, {}) is None
    assert str(kernels(sdfg)[0].map.range) == before


def test_a_grid_smaller_than_the_device_keeps_its_own_shape():
    """A kernel that already fits the device once has nothing to fold, so nothing is folded."""
    small = DEVICE_GRID_BLOCKS * BLOCK // 2
    sdfg = straight_line(small)
    before = str(kernels(sdfg)[0].map.range)
    assert GridStrideKernels().apply_pass(sdfg, {}) is None
    assert str(kernels(sdfg)[0].map.range) == before


def test_a_domain_shorter_than_one_block_still_gets_a_grid():
    """The folded grid is a CEILING of blocks. A floor would round a short domain to zero blocks and
    the codegen's ``grid <= 0`` guard would skip the launch, computing nothing at all."""
    sdfg = straight_line(N)
    GridStrideKernels().apply_pass(sdfg, {})
    grid = kernels(sdfg)[0].map.range.num_elements()
    assert dace.symbolic.evaluate(grid, {N: BLOCK // 4}) == 1


def test_the_block_extent_survives_the_fold():
    """Block size is the block-size selector's decision; this pass only changes the grid."""
    sdfg = straight_line(N, block=512)
    GridStrideKernels().apply_pass(sdfg, {})
    block = next(n for n, _ in sdfg.all_nodes_recursive()
                 if isinstance(n, nodes.MapEntry) and n.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock)
    assert dace.symbolic.evaluate(block.map.range.num_elements(), {}) == 512


def test_a_stepped_kernel_map_keeps_its_thread_count():
    """A map of step ``s`` strip-mined by ``t`` threads leaves the outer stepping ``s * t``, so the
    outer step is the block's span in index units and NOT its thread count.

    Reading it as a thread count gave the step-4 ``tsvc/s31111`` a block of 2048 at 512 threads, and
    codegen refused the launch: past the 1024 threads per block every device allows.
    """
    sdfg = straight_line(N, block=512, step=4)
    assert GridStrideKernels().apply_pass(sdfg, {}) == (1, 0)
    block = next(n for n, _ in sdfg.all_nodes_recursive()
                 if isinstance(n, nodes.MapEntry) and n.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock)
    assert dace.symbolic.evaluate(block.map.range.num_elements(), {}) == 512
    validate_block_size_limits(block, to_3d_dims(list(block.map.range.size())))
    # The fold is only a schedule change, so the strided loop under the thread block must still
    # advance a whole block's SPAN -- 512 threads four elements apart -- or elements go unvisited.
    strided = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)
                   and n.map.schedule == dtypes.ScheduleType.Sequential and n.map.label == 'block')
    assert dace.symbolic.evaluate(strided.map.range[0][2], {}) == 512 * 4
    sdfg.validate()


def test_a_pair_that_is_not_a_strip_mining_of_one_step_is_left_alone():
    """An outer step that is not a whole number of inner steps is not a strip-mining, and the
    quotient that would be the thread count does not exist. Left alone rather than rounded."""
    sdfg = straight_line(N, block=512, step=4)
    kernels(sdfg)[0].map.range = dace.subsets.Range([(0, N - 1, 2050)])
    before = str(kernels(sdfg)[0].map.range)
    assert GridStrideKernels().apply_pass(sdfg, {}) is None
    assert str(kernels(sdfg)[0].map.range) == before


def test_a_library_expansion_that_chose_its_own_block_shape_is_left_alone():
    """A ``(grid, thread_block)`` pair out of a reduction expansion is not a strip-mining: its inner
    map starts at zero, not at the outer index, and its block width is not the outer step. Reading
    the outer step as a block width there would rewrite the kernel to one thread per block."""
    sdfg = dace.SDFG('gs_expansion')
    for name in ('a', 'acc'):
        sdfg.add_array(name, [N], dace.float64)
    state = sdfg.add_state('s', is_start_block=True)
    outer_e, outer_x = state.add_map('grid', {'o': '0:N'}, schedule=dtypes.ScheduleType.GPU_Device)
    inner_e, inner_x = state.add_map('thread_block', {'tid': '0:64'}, schedule=dtypes.ScheduleType.GPU_ThreadBlock)
    tasklet = state.add_tasklet('t', {'__in'}, {'__out'}, '__out = __in * 2.0')
    state.add_memlet_path(state.add_read('a'), outer_e, inner_e, tasklet, dst_conn='__in', memlet=dace.Memlet('a[o]'))
    state.add_memlet_path(tasklet,
                          inner_x,
                          outer_x,
                          state.add_write('acc'),
                          src_conn='__out',
                          memlet=dace.Memlet('acc[0]', wcr=SUM))
    sdfg.validate()
    before = str(kernels(sdfg)[0].map.range)
    assert GridStrideKernels().apply_pass(sdfg, {}) is None
    assert str(kernels(sdfg)[0].map.range) == before


def test_a_kernel_someone_else_already_shaped_is_left_alone():
    """Two inner scopes is not the ``AddThreadBlockMaps`` shape, and re-tiling it would overrule a
    decision this pass did not make."""
    sdfg = dace.SDFG('gs_two_inner')
    for name in ('a', 'b', 'c'):
        sdfg.add_array(name, [N], dace.float64)
    state = sdfg.add_state('s', is_start_block=True)
    outer_e, outer_x = state.add_map('grid', {'bi': f'0:N:{BLOCK}'}, schedule=dtypes.ScheduleType.GPU_Device)
    for src, dst in (('a', 'b'), ('a', 'c')):
        inner_e, inner_x = state.add_map(f'block_{dst}', {f'i_{dst}': f'bi:Min(N - 1, bi + {BLOCK - 1}) + 1'},
                                         schedule=dtypes.ScheduleType.GPU_ThreadBlock)
        tasklet = state.add_tasklet(f't_{dst}', {'__in'}, {'__out'}, '__out = __in * 2.0')
        state.add_memlet_path(state.add_read(src),
                              outer_e,
                              inner_e,
                              tasklet,
                              dst_conn='__in',
                              memlet=dace.Memlet(f'{src}[i_{dst}]'))
        state.add_memlet_path(tasklet,
                              inner_x,
                              outer_x,
                              state.add_write(dst),
                              src_conn='__out',
                              memlet=dace.Memlet(f'{dst}[0]', wcr=SUM))
    sdfg.validate()
    before = str(kernels(sdfg)[0].map.range)
    assert GridStrideKernels().apply_pass(sdfg, {}) is None
    assert str(kernels(sdfg)[0].map.range) == before


def test_a_kernel_inside_a_sequential_loop_is_not_folded():
    """Its grid is one front, measured at 39 to 90 blocks: there is nothing for a stride to fold,
    and folding it anyway measured 1.02x to 1.07x slower."""
    sdfg = inside_loop(N)
    before = str(kernels(sdfg)[0].map.range)
    strided, hinted = GridStrideKernels().apply_pass(sdfg, {})
    assert strided == 0
    assert hinted == 1
    assert str(kernels(sdfg)[0].map.range) == before


def test_the_persistent_kernel_is_recorded_where_it_was_not_taken():
    """The finding must name the shape, its price, and the rewrite that would remove it."""
    sdfg = inside_loop(N)
    GridStrideKernels().apply_pass(sdfg, {})
    hint = kernels(sdfg)[0].specialization_hint
    assert PERSISTENT_KERNEL_LEAD in hint
    assert 'N launches per call' in hint, 'the launch count is the whole point of the note'
    assert 'persistent kernel' in hint and 'grid-wide barrier' in hint
    assert 'Not taken here' in hint, 'a hint that does not say it was declined reads as a directive'
    # Recorded on the loop as well, because that is where a reader meets it: the CUDA target emits no
    # map hints, so the loop's copy is the one that reaches the rendered form.
    assert PERSISTENT_KERNEL_LEAD in loops(sdfg)[0].specialization_hint


def test_the_note_is_written_once():
    """Re-running must not stack the same finding, on the map or on the loop."""
    sdfg = inside_loop(N)
    GridStrideKernels().apply_pass(sdfg, {})
    first = (kernels(sdfg)[0].specialization_hint, loops(sdfg)[0].specialization_hint)
    assert GridStrideKernels().apply_pass(sdfg, {}) is None
    assert (kernels(sdfg)[0].specialization_hint, loops(sdfg)[0].specialization_hint) == first


def test_the_note_keeps_what_the_loop_already_said():
    """``AnnotateLoopKinds`` says what KIND of loop this is; that is a different fact, not a stale
    version of this one, so it must survive."""
    sdfg = inside_loop(N)
    kind = 'sequential -- a loop-carried dependence was PROVEN, so this iteration order is required.'
    loops(sdfg)[0].specialization_hint = kind
    GridStrideKernels().apply_pass(sdfg, {})
    hint = loops(sdfg)[0].specialization_hint
    assert kind in hint and PERSISTENT_KERNEL_LEAD in hint


def test_the_note_reaches_the_rendered_form_as_a_comment():
    """A finding nobody can read is not recorded. The standalone rendering is where a hint becomes
    text, so the hint must survive ``hint_comment`` intact, line for line."""
    from dace import cpf_lowering
    sdfg = inside_loop(N)
    GridStrideKernels().apply_pass(sdfg, {})
    with cpf_lowering.dialect_scope(cpf_lowering.Dialect.STANDALONE_HIP):
        rendered = cpf_lowering.hint_comment(loops(sdfg)[0].specialization_hint)
    assert all(line.startswith('// ') for line in rendered.splitlines())
    assert PERSISTENT_KERNEL_LEAD in rendered and 'grid-wide barrier' in rendered
    # A runtime build has a target already, so the note would be noise in code nobody reads.
    with cpf_lowering.dialect_scope(cpf_lowering.Dialect.RUNTIME):
        assert cpf_lowering.hint_comment(loops(sdfg)[0].specialization_hint) == ''


def test_the_note_survives_a_round_trip_through_the_file_format():
    """A finding that a save and load drops is a finding lost between the pass and the reader."""
    sdfg = inside_loop(N)
    GridStrideKernels().apply_pass(sdfg, {})
    path = pathlib.Path(tempfile.mkdtemp()) / 'hinted.sdfgz'
    sdfg.save(str(path), compress=True)
    back = dace.SDFG.from_file(str(path))
    assert PERSISTENT_KERNEL_LEAD in (kernels(back)[0].specialization_hint or '')
    assert PERSISTENT_KERNEL_LEAD in (loops(back)[0].specialization_hint or '')


def test_a_short_loop_is_not_worth_a_note():
    """Below the floor the launches are not what this kernel costs, and a note there is noise."""
    sdfg = inside_loop(N, trips=str(grid_stride_kernels.LAUNCH_HINT_FLOOR // 2))
    assert GridStrideKernels().apply_pass(sdfg, {}) is None
    assert not kernels(sdfg)[0].specialization_hint


def test_a_kernel_with_no_loop_around_it_is_not_hinted():
    """The note is about re-launching. A kernel launched once has nothing to say."""
    sdfg = straight_line(DEVICE_GRID_BLOCKS * BLOCK // 2)
    GridStrideKernels().apply_pass(sdfg, {})
    assert not kernels(sdfg)[0].specialization_hint


@pytest.mark.gpu
def test_the_folded_kernel_computes_the_same_thing():
    """A grid stride is a schedule change and nothing else: every element still reaches the
    accumulator exactly once, including the ones past the folded grid that only the strided loop
    reaches.

    Driven through the real offload and ``AddThreadBlockMap`` rather than the hand-built shape
    above, so the numbers are checked against the pair the pipeline actually hands this pass.
    """
    from dace.transformation.dataflow.add_threadblock_map import AddThreadBlockMap
    size = DEVICE_GRID_BLOCKS * BLOCK * 3 + 17  # a ragged tail past three full strides

    @dace.program
    def total(a: dace.float64[size], out: dace.float64[1]):
        for i in dace.map[0:size]:
            out[0] += a[i]

    sdfg = total.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations()
    sdfg.apply_transformations_once_everywhere(AddThreadBlockMap)
    assert GridStrideKernels().apply_pass(sdfg, {})[0] == 1
    sdfg.validate()

    a = np.random.rand(size)
    out = np.zeros(1)
    sdfg(a=a, out=out)
    assert np.allclose(out[0], a.sum())


@pytest.mark.gpu
def test_the_folded_stepped_kernel_computes_the_same_thing():
    """The ``s31111`` shape: a map of step 4 whose body reads the four elements the step covers.

    Strip-mining leaves the outer stepping ``4 * block``, and the thread count read off it was four
    times too large. This is the same assertion as above with a step, because a block width that is
    wrong by a factor of the step is either rejected by codegen or visits the wrong elements.
    """
    from dace.transformation.dataflow.add_threadblock_map import AddThreadBlockMap
    size = DEVICE_GRID_BLOCKS * BLOCK * 4 + 20  # whole strides of a step-4 block, then a ragged tail

    @dace.program
    def total4(a: dace.float64[size], out: dace.float64[1]):
        for i in dace.map[0:size:4]:
            out[0] += a[i] + a[i + 1] + a[i + 2] + a[i + 3]

    sdfg = total4.to_sdfg(simplify=True)
    sdfg.apply_gpu_transformations()
    sdfg.apply_transformations_once_everywhere(AddThreadBlockMap)
    assert GridStrideKernels().apply_pass(sdfg, {})[0] == 1
    sdfg.validate()

    a = np.random.rand(size)
    out = np.zeros(1)
    sdfg(a=a, out=out)
    assert np.allclose(out[0], a.sum())
