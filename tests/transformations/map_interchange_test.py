# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
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
    sdfg.coarsen_dataflow()
    sdfg.validate()

    # Apply map interchange
    state = sdfg.node(0)
    ome = next(n for n in state.nodes()
               if isinstance(n, dace.nodes.MapEntry) and n.map.params[0] == 'j')
    ime = next(n for n in state.nodes()
               if isinstance(n, dace.nodes.MapEntry) and n.map.params[0] == 'k')
    MapInterchange.apply_to(sdfg, outer_map_entry=ome, inner_map_entry=ime)

    # Validate memlets
    sdfg.validate()

    dace.Config.set('experimental', 'validate_undefs', value=oldval)

    # Validate correctness
    sdfg(A=A, B=B)
    assert np.allclose(B, expected)


if __name__ == '__main__':
    test_map_interchange()
