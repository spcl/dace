# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

# Tests reduce expansions for FPGA
import dace
import numpy as np
from dace.fpga_testing import fpga_test


def create_reduce_sdfg(wcr_str, reduction_axis, sdfg_name, input_data, output_data, dtype):
    '''
    Build an SDFG that perform the given reduction along the given axis
    :param wcr_str: reduction operation to perform
    :param reduction_axis: the axis on which operate
    :param sdfg_name:
    :param input_data:
    :param output_data:
    :return:
    '''
    sdfg = dace.SDFG(sdfg_name)

    ###########################################################################
    # Copy data to FPGA

    copy_in_state = sdfg.add_state("copy_to_device")
    input_data_shape = input_data.shape
    output_data_shape = output_data.shape

    sdfg.add_array('A', input_data_shape, dtype)

    in_host_A = copy_in_state.add_read('A')

    sdfg.add_array("device_A",
                   shape=input_data_shape,
                   dtype=dtype,
                   storage=dace.dtypes.StorageType.FPGA_Global,
                   transient=True)

    in_device_A = copy_in_state.add_write("device_A")

    copy_in_memlet = dace.Memlet("A[{}]".format(",".join([f"0:{i}" for i in input_data_shape])))

    copy_in_state.add_memlet_path(in_host_A, in_device_A, memlet=copy_in_memlet)

    ###########################################################################
    # Copy data from FPGA

    copy_out_state = sdfg.add_state("copy_from_device")
    sdfg.add_array("B", output_data_shape, dtype)
    sdfg.add_array("device_B",
                   shape=output_data_shape,
                   dtype=dtype,
                   storage=dace.dtypes.StorageType.FPGA_Global,
                   transient=True)

    out_device = copy_out_state.add_read("device_B")
    out_host = copy_out_state.add_write("B")
    copy_out_memlet = dace.Memlet("B[{}]".format(",".join([f"0:{i}" for i in output_data_shape])))
    copy_out_state.add_memlet_path(out_device, out_host, memlet=copy_out_memlet)

    ########################################################################
    # FPGA State

    fpga_state = sdfg.add_state("fpga_state")
    r = fpga_state.add_read("device_A")
    w = fpga_state.add_write("device_B")
    red = fpga_state.add_reduce(wcr_str, reduction_axis, 0, schedule=dace.dtypes.ScheduleType.FPGA_Device)

    fpga_state.add_nedge(r, red, dace.Memlet(data="device_A"))
    fpga_state.add_nedge(red, w, dace.Memlet(data="device_B"))

    ######################################
    # Interstate edges
    sdfg.add_edge(copy_in_state, fpga_state, dace.sdfg.sdfg.InterstateEdge())
    sdfg.add_edge(fpga_state, copy_out_state, dace.sdfg.sdfg.InterstateEdge())

    sdfg.validate()

    return sdfg


@fpga_test(assert_ii_1=False)
def test_reduce_sum_one_axis():
    A = np.random.rand(8, 8).astype(np.float32)
    B = np.random.rand(8).astype(np.float32)
    sdfg = create_reduce_sdfg("lambda a,b: a+b", [0], "reduction_sum_one_axis", A, B, dace.float32)
    from dace.libraries.standard import Reduce
    Reduce.default_implementation = "FPGAPartialReduction"
    sdfg.expand_library_nodes()
    sdfg(A=A, B=B)
    assert np.allclose(B, np.sum(A, axis=0))
    return sdfg


@fpga_test()
def test_reduce_sum_all_axis():
    A = np.random.rand(4, 4).astype(np.float32)
    B = np.random.rand(1).astype(np.float32)
    sdfg = create_reduce_sdfg("lambda a,b: a+b", (0, 1), "reduction_sum_all_axis", A, B, dace.float32)
    from dace.libraries.standard import Reduce
    Reduce.default_implementation = "FPGAPartialReduction"
    sdfg.expand_library_nodes()
    sdfg(A=A, B=B)
    assert np.allclose(B, np.sum(A, axis=(0, 1)))
    return sdfg


@fpga_test()
def test_reduce_sum_4D():
    A = np.random.rand(4, 4, 4, 12).astype(np.float64)
    B = np.random.rand(4, 4).astype(np.float64)
    sdfg = create_reduce_sdfg("lambda a,b: a+b", [2, 3], "reduction_sum_4D", A, B, dace.float64)
    from dace.libraries.standard import Reduce
    Reduce.default_implementation = "FPGAPartialReduction"
    sdfg.expand_library_nodes()
    sdfg(A=A, B=B)
    assert np.allclose(B, np.sum(A, axis=(2, 3)))
    return sdfg


@fpga_test(assert_ii_1=False)
def test_reduce_max():
    A = np.random.rand(4, 4).astype(np.float32)
    B = np.random.rand(4).astype(np.float32)
    sdfg = create_reduce_sdfg("lambda a,b: max(a,b)", [1], "reduction_max", A, B, dace.float32)
    from dace.libraries.standard import Reduce
    Reduce.default_implementation = "FPGAPartialReduction"
    sdfg.expand_library_nodes()
    sdfg(A=A, B=B)
    assert np.allclose(B, np.max(A, axis=1))
    return sdfg


if __name__ == "__main__":
    test_reduce_sum_one_axis(None)
    test_reduce_sum_all_axis(None)
    test_reduce_sum_4D(None)
    test_reduce_max(None)
