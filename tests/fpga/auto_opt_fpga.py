# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" FPGA Tests for Auto Optimization """

import dace
import numpy as np
from dace.transformation.interstate import FPGATransformSDFG
from dace.transformation.auto import fpga as aopt

N = dace.symbol('N')


def test_global_to_local(size: int):
    '''
    Tests global_to_local optimization
    :return:
    '''
    @dace.program
    def global_to_local(A: dace.float32[N], B: dace.float32[N]):
        for i in range(N):
            tmp = A[i]
            B[i] = tmp + 1

    A = np.random.rand(size).astype(np.float32)
    B = np.random.rand(size).astype(np.float32)

    sdfg = global_to_local.to_sdfg()
    sdfg.apply_transformations([FPGATransformSDFG])

    aopt.fpga_global_to_local(sdfg)

    # Check that transformation has been actually applied
    # There should be only one transient among the sdfg arrays and it must have Local Storage Type
    candidate = None
    for name, array in sdfg.arrays.items():
        if array.transient:
            assert array.storage == dace.dtypes.StorageType.FPGA_Local
            candidate = name
            break

    assert candidate is not None

    # Check that all access nodes that refer to this container have also been updated
    for node, graph in sdfg.all_nodes_recursive():
        if isinstance(node, dace.sdfg.nodes.AccessNode):
            trace = dace.sdfg.utils.trace_nested_access(node, graph,
                                                        graph.parent)

            for (_, acc_node), memlet_trace, state_trace, sdfg_trace in trace:
                if acc_node is not None and acc_node.data == candidate:
                    nodedesc = node.desc(graph)
                    assert nodedesc.storage == dace.dtypes.StorageType.FPGA_Local

    sdfg(A=A, B=B, N=size)
    assert np.allclose(A + 1, B)


if __name__ == "__main__":
    test_global_to_local(8)
