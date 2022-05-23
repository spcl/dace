# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Type inference test with annotated types. """

import dace
from dace.fpga_testing import fpga_test
from dace.transformation.interstate import FPGATransformSDFG
import numpy as np

N = dace.symbol("N")
CONSTANT_ARRAY = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32)
CONSTANT_VALUE = float(1)


def make_sdfg():
    sdfg = dace.SDFG("constant_type_inference")

    sdfg.add_array("output", shape=[N], dtype=dace.float32)

    sdfg.add_array("device_output",
                   shape=[N],
                   dtype=dace.float32,
                   storage=dace.dtypes.StorageType.FPGA_Global,
                   transient=True)

    ###########################################################################
    # Copy data from FPGA
    copy_out_state = sdfg.add_state("copy_to_host")

    device_output = copy_out_state.add_read("device_output")
    host_output = copy_out_state.add_write("output")

    copy_out_state.add_memlet_path(device_output, host_output, memlet=dace.Memlet(f"{host_output}[0:N]"))

    ########################################################################
    # FPGA, First State

    # increment constant array of 1 elements

    fpga_state = sdfg.add_state("fpga_state")

    out = fpga_state.add_write("device_output")
    map_entry, map_exit = fpga_state.add_map("increment_map", {"i": "0:N"}, schedule=dace.ScheduleType.FPGA_Device)

    # Force type inference for constant array
    tasklet = fpga_state.add_tasklet("increment_tasklet", {}, {"out"}, "incr = constant_value\n"
                                     "tmp = constant_array[i]\n"
                                     "out = tmp + incr")

    fpga_state.add_memlet_path(map_entry, tasklet, memlet=dace.Memlet())
    fpga_state.add_memlet_path(tasklet, map_exit, out, src_conn="out", memlet=dace.Memlet("device_output[i]"))

    sdfg.add_edge(fpga_state, copy_out_state, dace.sdfg.sdfg.InterstateEdge())
    sdfg.fill_scope_connectors()
    sdfg.validate()
    return sdfg


@dace.program
def type_inference(x: dace.float32[N], y: dace.float32[N]):
    @dace.map
    def comp(i: _[0:N]):
        in_x << x[i]
        in_y << y[i]
        out >> y[i]

        # computes y[i]=(int)x[i] + ((int)y[i])*2.1
        var1 = int(in_x)
        var2: int = in_y
        var3 = 2.1 if (i > 1 and i < 10) else 2.1  # Just for the sake of testing
        res = var1 + var3 * var2
        out = res


@fpga_test()
def test_type_inference_fpga():

    N.set(24)

    # Initialize vector: X
    X = np.random.uniform(-10, 0, N.get()).astype(dace.float32.type)
    Y = np.random.uniform(-10, 0, N.get()).astype(dace.float32.type)
    # compute expected result
    Z = np.zeros(N.get())
    for i in range(0, N.get()):
        Z[i] = int(X[i]) + int(Y[i]) * 2.1

    sdfg = type_inference.to_sdfg()
    sdfg.apply_transformations(FPGATransformSDFG)
    sdfg(x=X, y=Y, N=N)

    diff = np.linalg.norm(Z - Y) / N.get()

    assert diff <= 1e-5

    return sdfg


@fpga_test()
def test_constant_type_inference_fpga():
    sdfg = make_sdfg()
    sdfg.add_constant('constant_array', CONSTANT_ARRAY)
    sdfg.add_constant('constant_value', CONSTANT_VALUE)

    out = dace.ndarray([CONSTANT_ARRAY.size], dtype=dace.float32)
    sdfg(N=CONSTANT_ARRAY.size, output=out)
    ref = CONSTANT_ARRAY + CONSTANT_VALUE
    diff = np.linalg.norm(ref - out) / CONSTANT_ARRAY.size
    assert diff <= 1e-5
    return sdfg


if __name__ == "__main__":
    test_type_inference_fpga(None)
    test_constant_type_inference_fpga(None)
