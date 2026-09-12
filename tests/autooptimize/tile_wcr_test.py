# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests write-conflict resolution tiling """
import dace
from dace.transformation.auto import auto_optimize as aopt
import numpy as np

N = dace.symbol('N')


def _runtest(sdfg: dace.SDFG, n: int, add_symbol: bool = True):
    A = np.random.rand(n).astype(np.float32)
    output = np.zeros([1], dtype=np.float32)
    if add_symbol:
        sdfg(A=A, output=output, N=n)
    else:
        sdfg(A=A, output=output)
    assert np.allclose(output, np.sum(A))


def _runtest2d(sdfg: dace.SDFG, n: int, m: int):
    A = np.random.rand(n, m).astype(np.float32)
    output = np.zeros([m], dtype=np.float32)
    sdfg(A=A, output=output, N=n)
    assert np.allclose(output, np.sum(A, axis=0))


def test_shortmap():

    @dace.program
    def sum(A: dace.float32[4], output: dace.float32[1]):
        for i in dace.map[0:4]:
            output += A[i]

    sdfg = sum.to_sdfg()
    aopt.auto_optimize(sdfg, dace.DeviceType.CPU)
    assert 'atomic' not in sdfg.generate_code()[0].code
    _runtest(sdfg, 4, False)
    del sdfg


def test_symmap():

    @dace.program
    def sum(A: dace.float32[N], output: dace.float32[1]):
        for i in dace.map[0:N]:
            output += A[i]

    sdfg = sum.to_sdfg()
    aopt.auto_optimize(sdfg, dace.DeviceType.CPU)
    code: str = sdfg.generate_code()[0].code
    # One atomic for the whole map -- that is what tiling the conflict buys. The tile-local
    # half used to read as a ``wcr_fixed::reduce`` into the tile scalar; the sequential tile
    # loop now accumulates in a register and stores once, so assert the register instead of
    # the memory round-trip it replaced.
    assert '__acc_' in code and code.count('atomic') == 1
    _runtest(sdfg, 257)
    del sdfg


def test_libnode():

    @dace.program
    def sum(A: dace.float32[N], output: dace.float32[1]):
        dace.reduce(lambda a, b: a + b, A, output, identity=0)

    sdfg = sum.to_sdfg()
    # This test is about TILING a write-conflict reduction, so it needs the expansion that
    # actually produces a WCR edge.  Reduce's default dispatches on schedule and lands on the
    # OpenMP expansion here, which emits a ``reduction()`` clause instead -- nothing for
    # TileWCR to act on -- so ask for the pure lowering explicitly rather than depending on
    # whichever one happens to be the default.
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, dace.libraries.standard.nodes.Reduce):
            n.implementation = 'pure'
    sdfg.expand_library_nodes()
    aopt.auto_optimize(sdfg, dace.DeviceType.CPU)
    code: str = sdfg.generate_code()[0].code
    # One atomic for the whole map -- that is what tiling the conflict buys. The tile-local
    # half used to read as a ``wcr_fixed::reduce`` into the tile scalar; the sequential tile
    # loop now accumulates in a register and stores once, so assert the register instead of
    # the memory round-trip it replaced.
    assert '__acc_' in code and code.count('atomic') == 1
    _runtest(sdfg, 257)
    del sdfg


def test_block_reduction():

    @dace.program
    def sum(A: dace.float32[N, N], output: dace.float32[N]):
        for i, j in dace.map[0:N, 0:N]:
            output[j] += A[i, j]

    sdfg = sum.to_sdfg()
    aopt.auto_optimize(sdfg, dace.DeviceType.CPU)
    code: str = sdfg.generate_code()[0].code
    if dace.Config.get_bool('optimizer', 'autotile_partial_parallelism'):
        assert 'reduce(' in code and code.count('atomic') == 0
    _runtest2d(sdfg, 257, 257)
    del sdfg


def test_block_reduction_short():

    @dace.program
    def sum(A: dace.float32[N, 2], output: dace.float32[2]):
        for i, j in dace.map[0:N, 0:2]:
            output[j] += A[i, j]

    sdfg = sum.to_sdfg()
    aopt.auto_optimize(sdfg, dace.DeviceType.CPU)
    code: str = sdfg.generate_code()[0].code
    assert 'reduce(' in code and code.count('atomic') == 1
    _runtest2d(sdfg, 257, 2)
    del sdfg


def _map_schedules(sdfg: dace.SDFG):
    """Every map in the SDFG tree, as ``label -> schedule``, nested SDFGs included."""
    from dace.sdfg import nodes as nd
    return {
        n.map.label: n.map.schedule
        for sd in sdfg.all_sdfgs_recursive()
        for st in sd.states()
        for n in st.nodes() if isinstance(n, nd.MapEntry)
    }


def test_the_tiled_conflict_body_is_sequential_on_gpu():
    """Tiling a write-conflicted map splits it into a parallel tile map and a body that accumulates
    into the per-tile transient. The body is sequential by construction -- that accumulation is what
    removes the atomics -- but MapTiling copies the original schedule onto both halves, so on GPU it
    came out ``GPU_Device``: a kernel inside a kernel. The codegen refuses that shape, and the pass
    that flattens it has to hoist the body's range into the grid, where it names the outer map's own
    parameter and no longer compiles."""

    @dace.program
    def argmax(x: dace.float64[N], r: dace.int64[1]):
        r[0] = np.argmax(x)

    sdfg = argmax.to_sdfg(simplify=True)
    aopt.auto_optimize(sdfg, dace.DeviceType.GPU)

    schedules = _map_schedules(sdfg)
    bodies = [lbl for lbl, sched in schedules.items() if lbl.endswith('_map') and 'init' not in lbl]
    assert bodies, f'nothing was tiled; schedules were {schedules}'
    device = [lbl for lbl, sched in schedules.items() if sched == dace.dtypes.ScheduleType.GPU_Device]
    assert device, f'no kernel map survived; schedules were {schedules}'
    # The body of a tiled conflict is never a second kernel.
    kernels = 0
    for node, state in sdfg.all_nodes_recursive():
        if not isinstance(node, dace.sdfg.nodes.MapEntry):
            continue
        if node.map.schedule != dace.dtypes.ScheduleType.GPU_Device:
            continue
        kernels += 1
        assert state.entry_node(node) is None, f'{node.map.label} is a GPU_Device map inside another map scope'
    assert kernels, 'no kernel map was produced, so the scope check above asserted nothing'


def test_the_gpu_grid_names_no_kernel_parameter():
    """The outer map's range sizes the grid on the HOST, so it may not name a parameter of the kernel
    itself. It did: the flattening pass hoisted the tile body's range, which is written in terms of
    the tile index, and the generated launch referenced an identifier that exists only per block."""

    @dace.program
    def argmax(x: dace.float64[N], r: dace.int64[1]):
        r[0] = np.argmax(x)

    sdfg = argmax.to_sdfg(simplify=True)
    aopt.auto_optimize(sdfg, dace.DeviceType.GPU)
    sdfg.generate_code()  # the launch geometry is only built here; a bad range fails at C++ compile

    kernels = [
        n for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, dace.sdfg.nodes.MapEntry) and n.map.schedule == dace.dtypes.ScheduleType.GPU_Device
    ]
    assert kernels, 'no kernel map was produced'
    for kernel in kernels:
        own = set(kernel.map.params)
        named = {str(sym) for b, e, _ in kernel.map.range for sym in dace.symbolic.pystr_to_symbolic(b).free_symbols}
        named |= {str(sym) for _, e, _ in kernel.map.range for sym in dace.symbolic.pystr_to_symbolic(e).free_symbols}
        assert not (named & own), \
            f'{kernel.map.label} sizes its grid from its own parameters {sorted(named & own)}: {kernel.map.range}'


if __name__ == '__main__':
    test_symmap()
    test_shortmap()
    test_libnode()
    test_block_reduction()
    test_block_reduction_short()
    test_the_tiled_conflict_body_is_sequential_on_gpu()
    test_the_gpu_grid_names_no_kernel_parameter()
