# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from dace import nodes
from dace.sdfg import propagation


def test_decreasing_propagation():

    q = np.random.randn(19, 19)
    ref = q.copy()

    @dace.program
    def copy_nw_corner(q: dace.float64[19, 19]):
        for j in dace.map[15:19]:
            for i in dace.map[0:3]:
                q[i, j] = q[j - 12, 17 - i]

    sdfg = copy_nw_corner.to_sdfg()
    me = None
    state = None
    for n, s in sdfg.all_nodes_recursive():
        if isinstance(n, nodes.MapEntry) and n.map.params[0] == 'j':
            me = n
            state = s
            break
    assert (me)
    assert (state)
    edges = state.in_edges(me)
    assert (len(edges) == 1)
    subset = edges[0].data.src_subset
    assert (subset.ranges == [(3, 6, 1), (15, 17, 1)])

    copy_nw_corner(q)
    copy_nw_corner.f(ref)
    assert (np.allclose(q, ref))


def test_decreasing_propagation_with_a_symbolic_bound():
    """The same decreasing access, with a bound that is a symbol rather than a literal.

    Constant bounds hide this: ``Range`` normalises a begin that came out above its end, so the
    literal case above answers correctly even when the propagation put them the wrong way round. A
    symbolic one cannot be normalised, and the wrong order stays -- ``a[N - i:N]`` over ``i`` in
    ``0:N`` propagated to ``(N, N - 1)``, an EMPTY range where the union is ``(1, N - 1)``. An empty
    read at a scope boundary copies nothing, so the kernel reads whatever was already there.

    npbench ludcmp reaches this through its back-substitution loop, whose row shrinks by one each
    iteration.
    """
    N = dace.symbol('N')

    sdfg = dace.SDFG('decreasing_symbolic')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('b', [N], dace.float64)
    state = sdfg.add_state('s')
    entry, exit_ = state.add_map('m', {'i': '0:N'})
    tasklet = state.add_tasklet('first', {'inp'}, {'out'}, 'out = inp[0]')
    state.add_memlet_path(state.add_read('a'), entry, tasklet, dst_conn='inp', memlet=dace.Memlet('a[N - i:N]'))
    state.add_memlet_path(tasklet, exit_, state.add_write('b'), src_conn='out', memlet=dace.Memlet('b[i]'))

    propagation.propagate_memlets_sdfg(sdfg)

    read = next(e.data for e in state.in_edges(entry) if e.data.data == 'a')
    begin, end, _ = read.subset.ranges[0]
    assert (end - begin + 1).simplify() != 0, f'propagated an empty read: a{read.subset}'
    assert read.subset.ranges == [(1, N - 1, 1)], f'expected a[1:N], got a{read.subset}'


def test_decreasing_propagation_reads_every_element_it_should():
    """The numbers behind the range above, at a size small enough to check by hand.

    The kernel writes the LAST element of the suffix it is given, so every output is ``a[N - 1]``
    -- but only if the boundary memlet actually brought the suffix in. An empty one leaves the
    kernel reading uninitialised memory, which is a wrong answer and not a crash.
    """
    N = dace.symbol('N')

    @dace.program
    def suffix_last(a: dace.float64[N], b: dace.float64[N]):
        for i in dace.map[1:N]:
            b[i] = a[N - i]

    n = 6
    a = np.arange(1.0, n + 1.0)
    b = np.zeros(n)
    suffix_last(a=a, b=b, N=n)

    expected = np.zeros(n)
    for i in range(1, n):
        expected[i] = a[n - i]
    assert np.allclose(b, expected), f'{b} != {expected}'


if __name__ == '__main__':
    test_decreasing_propagation()
    test_decreasing_propagation_with_a_symbolic_bound()
    test_decreasing_propagation_reads_every_element_it_should()
