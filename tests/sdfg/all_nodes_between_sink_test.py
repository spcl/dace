# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``all_nodes_between`` discards its whole walk when the region holds a sink.

The docstring now says so, and these pin it. The behaviour is easy to read as "omit the dead-ending
node" and it is not: a single node with no out-edges empties the result even when every other node
reached ``end``. A predicate written over that result then reports "nothing found" without having
inspected anything, which is how a scope gate comes to approve a body it never looked at.
"""
import dace
from dace.sdfg import nodes


def map_over_a_scaled_copy(with_sink: bool) -> tuple[dace.SDFG, dace.SDFGState, nodes.MapEntry, nodes.MapExit]:
    """A one-dimensional map whose body scales ``A`` into ``B``, optionally beside a scratch scalar.

    The scratch is the shape that trips the walk in practice: a transient of extent 1 that is
    written and never read, so it carries an in-edge and no out-edge.
    """
    sdfg = dace.SDFG(f'scaled_copy_{"with" if with_sink else "without"}_sink')
    sdfg.add_array('A', [16], dace.float64)
    sdfg.add_array('B', [16], dace.float64)
    state = sdfg.add_state(is_start_block=True)
    entry, exit_node = state.add_map('outer', ndrange={'i': '0:16'})
    scale = state.add_tasklet('scale', {'a'}, {'b'}, 'b = a * 2.0')
    state.add_memlet_path(state.add_read('A'), entry, scale, dst_conn='a', memlet=dace.Memlet('A[i]'))
    state.add_memlet_path(scale, exit_node, state.add_write('B'), src_conn='b', memlet=dace.Memlet('B[i]'))
    if with_sink:
        sdfg.add_scalar('scratch', dace.float64, transient=True, lifetime=dace.dtypes.AllocationLifetime.Scope)
        write_only = state.add_tasklet('stash', {}, {'s'}, 's = 1.0')
        state.add_edge(entry, None, write_only, None, dace.Memlet())
        state.add_edge(write_only, 's', state.add_access('scratch'), None, dace.Memlet('scratch[0]'))
    return sdfg, state, entry, exit_node


def test_a_body_without_a_sink_is_walked_normally():
    _, state, entry, exit_node = map_over_a_scaled_copy(with_sink=False)

    walked = state.all_nodes_between(entry, exit_node)

    assert sorted(type(n).__name__ for n in walked) == ['Tasklet']


def test_one_write_only_scratch_scalar_empties_the_whole_walk():
    _, state, entry, exit_node = map_over_a_scaled_copy(with_sink=True)

    walked = state.all_nodes_between(entry, exit_node)

    # Not "the scratch is omitted" -- the scale tasklet reached the exit and is discarded with it.
    assert walked == set()


def test_the_scope_subgraph_still_sees_every_body_node_beside_the_scratch():
    _, state, entry, _ = map_over_a_scaled_copy(with_sink=True)

    scoped = state.scope_subgraph(entry, include_entry=False, include_exit=False).nodes()

    assert sorted(n.label if isinstance(n, nodes.Tasklet) else n.data for n in scoped) == ['scale', 'scratch', 'stash']


def test_an_empty_walk_cannot_be_told_apart_from_an_empty_scope():
    _, populated_state, populated_entry, populated_exit = map_over_a_scaled_copy(with_sink=True)
    empty_sdfg = dace.SDFG('empty_scope')
    empty_state = empty_sdfg.add_state(is_start_block=True)
    empty_entry, empty_exit = empty_state.add_map('outer', ndrange={'i': '0:16'})
    empty_state.add_edge(empty_entry, None, empty_exit, None, dace.Memlet())

    populated = populated_state.all_nodes_between(populated_entry, populated_exit)
    genuinely_empty = empty_state.all_nodes_between(empty_entry, empty_exit)

    # This equality is the defect: the caller has no way to distinguish the two.
    assert populated == genuinely_empty == set()
