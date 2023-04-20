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

    C1_dimension = dace.symbol('C1_dimension')
    C2_dimension = dace.symbol('C2_dimension')
    D1_dimension = dace.symbol('D1_dimension')
    D2_dimension = dace.symbol('D2_dimension')
    size_A_vals = dace.symbol('size_A_vals')
    size_B2_crd = dace.symbol('size_B2_crd')
    size_B2_pos = dace.symbol('size_B2_pos')
    size_B_vals = dace.symbol('size_B_vals')
    size_C_vals = dace.symbol('size_C_vals')
    size_D_vals = dace.symbol('size_D_vals')

    @dace.program
    def sched_sddmm0compute(A_vals: dace.float64[size_A_vals], B2_crd: dace.int32[size_B2_crd],
                            B2_pos: dace.int32[size_B2_pos], B_vals: dace.float64[size_B_vals],
                            C_vals: dace.float64[size_C_vals], D_vals: dace.float64[size_D_vals]):

        for i in dace.map[0:C1_dimension:1]:
            for j in dace.map[0:D1_dimension:1]:
                jC = i * C2_dimension + j
                for kB in dace.map[B2_pos[i]:B2_pos[(i + 1)]:1]:
                    k = B2_crd[kB]
                    kD = j * D2_dimension + k
                    A_vals[kB] = A_vals[kB] + (B_vals[kB] * C_vals[jC]) * D_vals[kD]

    sdfg = sched_sddmm0compute.to_sdfg()

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
