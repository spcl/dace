# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for ``recompute_fuse_for_gpu`` on a producer whose map body is a NESTED SDFG.

That shape is what ``LoopToMap`` leaves behind when it cannot inline a loop body, so it is the
common case on the GPU path rather than an exotic one, and the producer body then NAMES the map
parameter: the nested SDFG carries it in ``symbol_mapping``. Fusion has to keep that binding intact
while it renames the producer's parameters and replicates the body inside the consumer, and it has
to decline outright when the producer's write does not pin the parameter down at all.
"""
import numpy as np

import dace
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.finalize import recompute_fuse_for_gpu

N = dace.symbol('N')


def scaling_body(name: str, param: str, whole_range: bool) -> dace.SDFG:
    """Body SDFG for one map: ``dst = src * (param + 1)``, i.e. a body that names the parameter.

    ``whole_range`` picks the two memlet shapes a nested body appears with: a body whose connectors
    carry the WHOLE array (``LoopToMap``'s form -- the outer memlet is ``t[0:N]`` and the parameter
    does the indexing inside), or one whose connectors carry the single element the iteration
    touches (``t[param]``, indexed by the outer memlet).
    """
    body = dace.SDFG(name)
    body.add_symbol(param, dace.int64)
    shape = [N] if whole_range else [1]
    body.add_array('src', shape, dace.float64)
    body.add_array('dst', shape, dace.float64)
    state = body.add_state('single_state_body')
    tasklet = state.add_tasklet(name, {'inp'}, {'res'}, f'res = inp * ({param} + 1)')
    index = param if whole_range else '0'
    state.add_edge(state.add_access('src'), None, tasklet, 'inp', Memlet(f'src[{index}]'))
    state.add_edge(tasklet, 'res', state.add_access('dst'), None, Memlet(f'dst[{index}]'))
    return body


def producer_consumer_sdfg(whole_range: bool) -> dace.SDFG:
    """``t = a * (i + 1)`` then ``out = t * (j + 1)``, both map bodies nested SDFGs."""
    sdfg = dace.SDFG(f'nested_producer_{"whole" if whole_range else "element"}')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('out', [N], dace.float64)
    sdfg.add_transient('t', [N], dace.float64)
    state = sdfg.add_state('state')
    a_node, t_node, out_node = (state.add_access(name) for name in ('a', 't', 'out'))

    for param, src, dst, src_node, dst_node in (('i', 'a', 't', a_node, t_node), ('j', 't', 'out', t_node, out_node)):
        entry, exit_node = state.add_map(f'{dst}_map', {param: f'0:{N}'})
        body = state.add_nested_sdfg(scaling_body(f'{dst}_body', param, whole_range), {'src'}, {'dst'},
                                     symbol_mapping={
                                         param: param,
                                         'N': N
                                     })
        subset = f'0:{N}' if whole_range else param
        state.add_memlet_path(src_node, entry, body, dst_conn='src', memlet=Memlet(f'{src}[{subset}]'))
        state.add_memlet_path(body, exit_node, dst_node, src_conn='dst', memlet=Memlet(f'{dst}[{subset}]'))

    sdfg.validate()
    return sdfg


def assert_symbols_bound(sdfg: dace.SDFG) -> None:
    """Every nested SDFG binds each symbol its body reads -- the invariant ``validate`` checks."""
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.NestedSDFG):
            missing = [s for s in node.sdfg.free_symbols if s not in node.symbol_mapping]
            assert not missing, f'nested SDFG {node.label} reads unbound {missing}'


def materializes_intermediate(sdfg: dace.SDFG) -> bool:
    """Whether ``t`` is still written to memory (the descriptor outlives the fusion either way)."""
    return any(
        isinstance(node, nodes.AccessNode) and node.data == 't' for state in sdfg.all_states()
        for node in state.nodes())


def run(sdfg: dace.SDFG, n: int = 24) -> None:
    """Compile and check the numbers: ``out[i] = a[i] * (i + 1)^2``."""
    rng = np.random.default_rng(3)
    a = rng.random(n)
    out = np.zeros(n)
    sdfg(a=a, out=out, N=n)
    assert np.allclose(out, a * (np.arange(n) + 1.0)**2)


def test_element_wise_nested_producer_is_fused():
    """The write ``t[i]`` pins ``i`` to the consumer's ``j``, so the body is replicated inline.

    Renaming the producer's parameter has to move the nested SDFG's ``symbol_mapping`` KEY with the
    symbol it renames inside the body; renaming only one of the two left the replicated body reading
    a symbol its own node no longer bound.
    """
    sdfg = producer_consumer_sdfg(whole_range=False)
    assert recompute_fuse_for_gpu(sdfg) == 1
    assert_symbols_bound(sdfg)
    sdfg.validate()
    assert not materializes_intermediate(sdfg)
    run(sdfg)


def test_whole_range_nested_producer_is_not_fused():
    """A body whose connectors carry the whole array writes ``t[0:N]`` from EVERY iteration, so no
    consumer read can say which producer iteration to recompute -- the parameter would survive in
    the replicated body with nothing left to bind it."""
    sdfg = producer_consumer_sdfg(whole_range=True)
    assert recompute_fuse_for_gpu(sdfg) == 0
    assert_symbols_bound(sdfg)
    sdfg.validate()
    assert materializes_intermediate(sdfg)
    run(sdfg)


if __name__ == '__main__':
    test_element_wise_nested_producer_is_fused()
    test_whole_range_nested_producer_is_not_fused()
