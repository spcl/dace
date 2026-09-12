# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from dace.transformation.dataflow import MapInterchange


@dace.program
def miprog(A: dace.float64[20, 30, 40], B: dace.float64[40, 30, 20]):
    for i in dace.map[0:20]:
        for j in dace.map[0:30]:
            for k in dace.map[0:40]:
                with dace.tasklet:
                    a << A[i, j, k]
                    b >> B[k, j, i]
                    b = a + 5


def test_map_interchange():
    A = np.random.rand(20, 30, 40)
    B = np.random.rand(40, 30, 20)
    expected = np.transpose(A, axes=(2, 1, 0)) + 5

    oldval = dace.Config.get_bool('experimental', 'validate_undefs')
    dace.Config.set('experimental', 'validate_undefs', value=True)

    sdfg = miprog.to_sdfg()
    sdfg.simplify()
    sdfg.validate()

    # Apply map interchange
    state = sdfg.node(0)
    ome = next(n for n in state.nodes() if isinstance(n, dace.nodes.MapEntry) and n.map.params[0] == 'j')
    ime = next(n for n in state.nodes() if isinstance(n, dace.nodes.MapEntry) and n.map.params[0] == 'k')
    MapInterchange.apply_to(sdfg, outer_map_entry=ome, inner_map_entry=ime)

    # Validate memlets
    sdfg.validate()

    dace.Config.set('experimental', 'validate_undefs', value=oldval)

    # Validate correctness
    sdfg(A=A, B=B)
    assert np.allclose(B, expected)


def test_map_interchange_with_dynamic_map_inputs():
    # Three nested maps where the innermost one ranges between two values read from an array at
    # the outermost map's index -- the shape a sparse iteration over a CSR-style position array
    # takes. The bounds therefore depend on the enclosing map but not on the map being
    # interchanged with, which is the case MapInterchange has to reason about.
    #
    # The state is built here rather than parsed from a program. Under the nested SDFG contract
    # the reads that produce the bounds stay inside the enclosing map, so the two maps a program
    # like this lowers to are no longer siblings in one state and the pattern never appears at
    # the top level. That is a property of the lowering, not of the transformation, and this test
    # is about the transformation.
    N = dace.symbol('N')
    sdfg = dace.SDFG('sched_sddmm0compute')
    sdfg.add_array('B2_pos', [N + 1], dace.int32)
    sdfg.add_array('A_vals', [N], dace.float64)
    state = sdfg.add_state()

    pos = state.add_read('B2_pos')
    ime_i, imx_i = state.add_map('i', dict(i='0:N'))
    j_entry, j_exit = state.add_map('j', dict(j='0:N'))
    kb_entry, kb_exit = state.add_map('kB', dict(kB='__kB_b:__kB_e'))
    kb_entry.add_in_connector('__kB_b')
    kb_entry.add_in_connector('__kB_e')
    state.add_memlet_path(pos, ime_i, j_entry, kb_entry, dst_conn='__kB_b', memlet=dace.Memlet('B2_pos[i]'))
    state.add_memlet_path(pos, ime_i, j_entry, kb_entry, dst_conn='__kB_e', memlet=dace.Memlet('B2_pos[i + 1]'))

    tasklet = state.add_tasklet('compute', {}, {'out'}, 'out = 1.0')
    state.add_memlet_path(kb_entry, tasklet, memlet=dace.Memlet())
    state.add_memlet_path(tasklet,
                          kb_exit,
                          j_exit,
                          imx_i,
                          state.add_write('A_vals'),
                          src_conn='out',
                          memlet=dace.Memlet('A_vals[kB]'))
    sdfg.validate()

    # Find MapEntries of Maps over 'j' and 'kB'
    ome, ime = None, None
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.sdfg.nodes.MapEntry):
                if node.map.params[0] == 'j':
                    ome = node
                elif node.map.params[0] == 'kB':
                    ime = node

    # Assert the pattern MapEntry[j] -> MapEntry[kB] exists
    assert ome is not None and ime is not None
    state = sdfg.states()[0]
    assert len(list(state.edges_between(ome, ime))) > 0
    assert len(list(state.edges_between(ime, ome))) == 0

    # Interchange the Maps
    MapInterchange.apply_to(sdfg, outer_map_entry=ome, inner_map_entry=ime)
    ome, ime = None, None
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.sdfg.nodes.MapEntry):
                if node.map.params[0] == 'j':
                    ome = node
                elif node.map.params[0] == 'kB':
                    ime = node

    # Assert the pattern MapEntry[kB] -> MapEntry[j] exists
    assert ome is not None and ime is not None
    state = sdfg.states()[0]
    assert len(list(state.edges_between(ome, ime))) == 0
    assert len(list(state.edges_between(ime, ome))) > 0


if __name__ == '__main__':
    test_map_interchange()
    test_map_interchange_with_dynamic_map_inputs()
