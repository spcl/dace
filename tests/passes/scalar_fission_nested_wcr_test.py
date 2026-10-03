# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Scalar fission keeps a sum resolved inside a NestedSDFG on the container its initializer wrote.

LoopToMap outlines a reduction loop's body into a NestedSDFG, so the ``CR: Sum`` sits on the inner edge and
the MapExit path out to the accumulator carries none. Fission read that write as a fresh definition and
renamed it, the ``s = 0.0`` before the loop became dead and was removed, and on the GPU TSVC s3111's
conditional sum accumulated into a persistent device scalar: 1x, 2x, 3x over repeated calls.
"""
import numpy as np

import dace
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.scalar_fission import ScalarFission

N = dace.symbol('N')


def outlined_body() -> dace.SDFG:
    """The body LoopToMap outlines: ``acc += x`` as a WCR write into the connector array."""
    body = dace.SDFG('body')
    body.add_scalar('x', dace.float64)
    body.add_scalar('acc', dace.float64)
    state = body.add_state('accumulate')
    tasklet = state.add_tasklet('add', ['i'], ['o'], 'o = i')
    state.add_edge(state.add_read('x'), None, tasklet, 'i', dace.Memlet('x[0]'))
    state.add_edge(tasklet, 'o', state.add_write('acc'), None, dace.Memlet('acc[0]', wcr='lambda a, b: a + b'))
    return body


def zero_then_map_accumulate() -> dace.SDFG:
    """``s = 0; for i in map: body(a[i], s); out[0] = s``, the shape of stage ``reduction_to_wcr_map``."""
    sdfg = dace.SDFG('zero_then_map_accumulate')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('out', [1], dace.float64)
    sdfg.add_scalar('s', dace.float64, transient=True)
    init = sdfg.add_state('init', is_start_block=True)
    zero = init.add_tasklet('zero', [], ['y'], 'y = 0.0')
    init.add_edge(zero, 'y', init.add_write('s'), None, dace.Memlet('s[0]'))
    fold = sdfg.add_state_after(init, 'fold')
    entry, exit_ = fold.add_map('loop', {'i': '0:N'})
    nested = fold.add_nested_sdfg(outlined_body(), {'x'}, {'acc'})
    fold.add_memlet_path(fold.add_read('a'), entry, nested, dst_conn='x', memlet=dace.Memlet('a[i]'))
    fold.add_memlet_path(nested, exit_, fold.add_write('s'), src_conn='acc', memlet=dace.Memlet('s[0]'))
    copy = sdfg.add_state_after(fold, 'copy')
    tasklet = copy.add_tasklet('copy', ['x'], ['y'], 'y = x')
    copy.add_edge(copy.add_read('s'), None, tasklet, 'x', dace.Memlet('s[0]'))
    copy.add_edge(tasklet, 'y', copy.add_write('out'), None, dace.Memlet('out[0]'))
    sdfg.validate()
    return sdfg


def map_outputs(sdfg: dace.SDFG) -> list:
    """Containers a MapExit writes: the accumulators a parallel map folds into."""
    return [
        edge.data.data for node, state in sdfg.all_nodes_recursive() if isinstance(node, dace.nodes.MapExit)
        for edge in state.out_edges(node) if isinstance(edge.dst, dace.nodes.AccessNode)
    ]


def zero_writes(sdfg: dace.SDFG) -> list:
    """Containers a no-input tasklet writes ``0.0`` into."""
    return [
        state.out_edges(node)[0].data.data for node, state in sdfg.all_nodes_recursive()
        if isinstance(node, dace.nodes.Tasklet) and not node.in_connectors and '0.0' in node.code.as_string
    ]


def test_fission_does_not_split_a_nested_wcr_accumulator_from_its_initializer():
    sdfg = zero_then_map_accumulate()
    Pipeline([ScalarFission()]).apply_pass(sdfg, {})
    sdfg.validate()
    assert set(map_outputs(sdfg)) <= set(zero_writes(sdfg)), (map_outputs(sdfg), zero_writes(sdfg))


@dace.program
def s3111(a: dace.float64[N], b: dace.float64[2]):
    sum_val = 0.0
    for i in range(N):
        if a[i] > 0.0:
            sum_val = sum_val + a[i]
    b[0] = sum_val


def test_a_canonicalized_conditional_sum_keeps_the_initializer_its_map_accumulates_into():
    sdfg = s3111.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    outputs = map_outputs(sdfg)
    assert outputs, 'the loop was expected to become a map'
    assert set(outputs) <= set(zero_writes(sdfg)), (outputs, zero_writes(sdfg))
    a = np.random.default_rng(0).standard_normal(37)
    b = np.zeros(2)
    for _ in range(3):
        sdfg(a=a, b=b, N=37)
        np.testing.assert_allclose(b[0], a[a > 0].sum(), rtol=1e-12, atol=0.0)
