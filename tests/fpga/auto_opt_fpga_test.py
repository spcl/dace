# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" FPGA Tests for Auto Optimization """

import dace
import numpy as np
from dace.fpga_testing import fpga_test
from dace.transformation.interstate import FPGATransformSDFG
from dace.transformation.auto import auto_optimize as aopt
from dace.transformation.auto import fpga as fpga_auto_opt

N = dace.symbol('N')


@fpga_test()
def test_global_to_local():
    """
    Tests global_to_local optimization
    """
    @dace.program
    def global_to_local(alpha: dace.float32, B: dace.float32[N]):
        tmp = alpha / 2
        return tmp * B

    size = 8

    alpha = 0.5
    B = np.random.rand(size).astype(np.float32)

    sdfg = global_to_local.to_sdfg()

    aopt.auto_optimize(sdfg, dace.DeviceType.FPGA)

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
            trace = dace.sdfg.utils.trace_nested_access(node, graph, graph.parent)

            for (_, acc_node), memlet_trace, state_trace, sdfg_trace in trace:
                if acc_node is not None and acc_node.data == candidate:
                    nodedesc = node.desc(graph)
                    assert nodedesc.storage == dace.dtypes.StorageType.FPGA_Local

    C = sdfg(alpha=alpha, B=B, N=size)
    ref = alpha / 2 * B
    assert np.allclose(ref, C)

    return sdfg


@fpga_test()
def test_rr_interleave():
    """
        Tests RR interleaving of containers to memory banks
    """
    @dace.program
    def rr_interleave(A: dace.float32[8], B: dace.float32[8], C: dace.float32[8]):
        return A + B + C

    A = np.random.rand(8).astype(np.float32)
    B = np.random.rand(8).astype(np.float32)
    C = np.random.rand(8).astype(np.float32)

    sdfg = rr_interleave.to_sdfg()
    sdfg.apply_transformations([FPGATransformSDFG])

    #specifically run the the interleave transformation
    allocated = fpga_auto_opt.fpga_rr_interleave_containers_to_banks(sdfg)

    # There will be 5 arrays (one is a temporary containing A + B)
    assert allocated == [2, 1, 1, 1]

    R = sdfg(A=A, B=B, C=C)
    assert np.allclose(A + B + C, R)

    return sdfg


if __name__ == "__main__":
    test_global_to_local(8)
    test_rr_interleave()
