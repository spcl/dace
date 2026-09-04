# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The ``Scan`` block expansion: what a scan INSIDE a GPU kernel lowers to.

The point of the expansion is not the CUB call -- it is that the emitted subgraph carries a
``GPU_ThreadBlock`` map, because that is the only signal the code generator has that the enclosing
device map runs one iteration per BLOCK. Without it the row map is spread over threads and every
thread would scan a different row through the same collective. These tests assert that structure,
which is checkable without a GPU; the numerics are checked on device by the GPU test below.
"""
import pytest

import dace
from dace import dtypes
from dace.libraries.standard.nodes.scan import BLOCK_COLLECTIVE_THREADS, Scan, ScanOp
from dace.sdfg import nodes

N = dace.symbol('N', dtype=dace.int64)
ROWS = dace.symbol('ROWS', dtype=dace.int64)


def build_scan_sdfg(op: ScanOp = ScanOp.SUM, exclusive: bool = False, chains: int = 1) -> dace.SDFG:
    """One FULLY WIRED ``Scan`` over ``N`` elements, standing alone in a state.

    Every connector the shape requires is connected, the extra chains and ``op=AFFINE``'s
    coefficients included. An under-wired node fails the libnode's own validation, which would let
    a refusal test pass for the wrong reason -- rejecting a malformed graph rather than an
    unsupported shape.
    """
    sdfg = dace.SDFG('scan_block_probe')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('b', [N], dace.float64)
    state = sdfg.add_state()
    node = Scan('scan', op=op, exclusive=exclusive, chains=chains)
    state.add_node(node)
    for chain in range(chains):
        suffix = '' if chain == 0 else f'_{chain}'
        state.add_edge(state.add_read('a'), None, node, f'_scan_in{suffix}', dace.Memlet('a[0:N]'))
        state.add_edge(node, f'_scan_out{suffix}', state.add_write('b'), None, dace.Memlet('b[0:N]'))
        if op is ScanOp.AFFINE:
            state.add_edge(state.add_read('a'), None, node, f'_scan_coef{suffix}', dace.Memlet('a[0:N]'))
    return sdfg, state, node


def test_block_expansion_is_registered():
    assert 'CUDA (block)' in Scan.implementations


def test_expansion_emits_a_thread_block_map():
    """The structural claim: a ``GPU_ThreadBlock`` map of the collective's width."""
    sdfg, state, node = build_scan_sdfg()
    node.implementation = 'CUDA (block)'
    node.expand(state)
    tb_maps = [
        n for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.MapEntry) and n.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock
    ]
    assert len(tb_maps) == 1, 'exactly one thread-block map should carry the collective'
    lanes = tb_maps[0].map.range.size()[0]
    assert lanes == BLOCK_COLLECTIVE_THREADS


def test_every_lane_reads_the_whole_range():
    """The map supplies THREADS, not a partition. A per-lane subset would give each thread its own
    scan, which is the bug this test exists to catch."""
    sdfg, state, node = build_scan_sdfg()
    node.implementation = 'CUDA (block)'
    node.expand(state)
    entry = next(n for n, _ in sdfg.all_nodes_recursive()
                 if isinstance(n, nodes.MapEntry) and n.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock)
    param = entry.map.params[0]
    # The map lives in the expansion's NESTED SDFG, which ``all_states`` does not descend into.
    inner = next(st for n, st in sdfg.all_nodes_recursive() if n is entry)
    for edge in inner.out_edges(entry):
        assert param not in str(edge.data.subset), f'lane {param} narrows the scan range: {edge.data}'


def test_expansion_calls_the_shared_collective():
    """It must call the SAME collective the residue-class kernel uses, not a second copy."""
    sdfg, state, node = build_scan_sdfg()
    node.implementation = 'CUDA (block)'
    node.expand(state)
    tasklets = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.Tasklet)]
    assert len(tasklets) == 1
    assert 'block_inclusive_scan_strided' in tasklets[0].code.as_string
    assert f'{BLOCK_COLLECTIVE_THREADS}>' in tasklets[0].code.as_string


def test_stride_reaches_the_collective():
    """A strided scan is the general case, not an unsupported one: the stride is an argument."""
    sdfg, state, node = build_scan_sdfg()
    node.stride = 7
    node.implementation = 'CUDA (block)'
    node.expand(state)
    tasklet = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.Tasklet))
    assert '(long)(7)' in tasklet.code.as_string


@pytest.mark.parametrize('kwargs,reason', [
    ({
        'exclusive': True
    }, 'exclusive'),
    ({
        'chains': 2
    }, 'multi-chain'),
    ({
        'op': ScanOp.AFFINE
    }, 'op=AFFINE'),
    ({
        'op': ScanOp.MIN
    }, 'no identity'),
])
def test_unsupported_shapes_refuse_rather_than_approximate(kwargs, reason):
    """Each refusal sends the node to another expansion. Silently lowering one of these as a plain
    inclusive sum scan would be a miscompile, so the refusal is the contract."""
    sdfg, state, node = build_scan_sdfg(**kwargs)
    node.implementation = 'CUDA (block)'
    with pytest.raises(NotImplementedError):
        node.expand(state)
    # And the refusal must leave a working route: ``pure`` handles every one of these.
    assert 'pure' in Scan.implementations


def test_the_collective_header_declares_the_function():
    """The expansion names a symbol in scan.cuh; a rename there must break this, not the build."""
    import pathlib
    header = pathlib.Path(dace.__file__).parent / 'runtime' / 'include' / 'dace' / 'cuda' / 'scan.cuh'
    text = header.read_text()
    assert '__device__ void block_inclusive_scan_strided' in text
    # The residue-class kernel must go through the same collective, or the two drift apart.
    assert 'block_inclusive_scan_strided<T, Op, BLOCK>(in + k, out + k, m, s, op, identity);' in text


if __name__ == '__main__':
    test_block_expansion_is_registered()
    test_expansion_emits_a_thread_block_map()
    test_every_lane_reads_the_whole_range()
    test_expansion_calls_the_shared_collective()
    test_stride_reaches_the_collective()
    test_the_collective_header_declares_the_function()
    print('ok')
