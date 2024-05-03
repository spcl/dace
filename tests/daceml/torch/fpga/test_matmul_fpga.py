# Tests for Matmul Node Expansion: many of these can be implemented by using einsum

# TODO:
# - some deadlock for small matrices, such as (2, 16, 8) (2, 8, 8), not clear why. I suspect some problem with draining conditions

from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG

import torch
import torch.nn as nn
import numpy as np
from dace.frontend.python.module import DaceModule
import pytest
import dace
import argparse
from dace.util import utils
from multiprocessing import Process, Queue


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        # equivalent to np.einsum('bik,bkj->bij', A, B)
        z = torch.matmul(x, y)
        return z


def run(x_shape: tuple, y_shape: tuple, vec_width=1, queue=None):
    '''
    Evaluates the given configuration
    :param x_shape:
    :param y_shape:
    :param vec_width:
    :param execute_cpu_dace:
    :param queue:
    :return:
    '''

    import dace.libraries.onnx as donnx

    ptmodel = Model()

    x = torch.rand(x_shape, dtype=torch.float32)
    y = torch.rand(y_shape, dtype=torch.float32)
    torch_output = ptmodel(x, y)

    dace_model = DaceModule(ptmodel, auto_optimize=False)
    with dace.library.change_default(donnx.ONNXMatMul, "pure"):
        dace_output = dace_model(x, y)

    assert np.allclose(torch_output.detach().numpy(), dace_output, atol=1e-06)

    ##########################################
    # Transform to FPGA

    def TransformToFPGA(dace_module):
        '''
        Transforms the given module to run on FPGA.
        This includes vectorization and library node expansions.
        :param dace_module:
        :return:
        '''
        sdfg = dace_module.sdfg
        sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG])

        # Vectorize container (if needed)
        if vec_width > 1:
            vec_type = dace.vector(dace.float32, vec_width)
            input_data_name = sdfg.states()[0].source_nodes()[1].data
            output_data_name = sdfg.states()[0].sink_nodes()[0].data
            # vectorize input B
            utils.vectorize_array_and_memlet(sdfg, input_data_name, vec_type)
            # vectorize output B
            utils.vectorize_array_and_memlet(sdfg, output_data_name, vec_type)

        sdfg.expand_library_nodes()
        sdfg.apply_transformations_repeated([InlineSDFG])

    # Reset the SDFG
    dace_model.reset_sdfg()

    # Append transformation hook
    dace_model.append_post_onnx_hook("TransformToFPGA", TransformToFPGA)

    # Execute Module with FPGA expansion
    with dace.library.change_default(donnx.ONNXMatMul, "fpga"):
        dace_output_fpga = dace_model(x, y)

    dace_output_fpga_reshaped = dace_output_fpga.numpy().reshape(
        torch_output.detach().numpy().shape)
    diff = np.linalg.norm(torch_output.detach().numpy() -
                          dace_output_fpga_reshaped) / np.linalg.norm(
                              torch_output.detach().numpy())
    print("Difference: ", diff)

    if queue is not None:
        # we are testing
        queue.put(diff)
    else:
        if diff > 1e-6:
            import pdb
            pdb.set_trace()
            assert (False)

    del dace_model, ptmodel, x


@pytest.mark.fpga
def test(extensive=False):
    '''
    Evaluates multiple combination of Matmul/input size
    :return:
    '''

    print(
        f"----------- Testing Batched Matmul (3Dx3D tensor) (extensive: {extensive}) ---------------"
    )

    # Run FPGA tests in a different process to avoid issues with Intel OpenCL tools
    # (But not in parallel)

    # each position of this lists contains a test configuration
    if extensive:
        vec_width = [1, 1, 1, 1, 2, 4]
        x_shapes = [(4, 8, 16), (8, 16, 32), (8, 16, 16), (8, 16, 8),
                    (8, 16, 32), (8, 32, 64)]
        y_shapes = [(4, 16, 4), (8, 32, 64), (8, 16, 8), (8, 8, 16),
                    (8, 32, 64), (8, 64, 16)]
    else:
        vec_width = [1, 1, 4]
        x_shapes = [(4, 8, 16), (8, 16, 32), (8, 32, 64)]
        y_shapes = [(4, 16, 4), (8, 32, 64), (8, 64, 16)]

    for i in range(0, len(vec_width)):
        print("##########################################################")
        print(
            f"# Configuration: vw={vec_width[i]}, x_shape={x_shapes[i]}, y_shape={y_shapes[i]}"
        )
        print("##########################################################")
        queue = Queue()
        p = Process(target=run,
                    args=(x_shapes[i], y_shapes[i], vec_width[i], queue))
        p.start()
        p.join()
        assert (queue.get() < 1e-6)

    print(
        f"----------- Testing Matmul (3Dx2D tensor) (extensive: {extensive}) ---------------"
    )

    if extensive:
        vec_width = [1, 1, 1, 2, 4]
        x_shapes = [(4, 8, 16), (8, 16, 32), (2, 16, 32), (16, 2, 32),
                    (16, 2, 32), (16, 2, 32)]
        y_shapes = [(4, 16, 4), (32, 64), (32, 16), (32, 32), (32, 64),
                    (32, 16)]
    else:
        vec_width = [1, 1, 4]
        x_shapes = [(4, 8, 16), (8, 16, 32), (16, 2, 32)]
        y_shapes = [(4, 16, 4), (32, 64), (32, 64)]

    for i in range(0, len(vec_width)):
        print("##########################################################")
        print(
            f"# Configuration: vw={vec_width[i]}, x_shape={x_shapes[i]}, y_shape={y_shapes[i]}"
        )
        print("##########################################################")
        queue = Queue()
        p = Process(target=run,
                    args=(x_shapes[i], y_shapes[i], vec_width[i], queue))
        p.start()
        p.join()
        assert (queue.get() < 1e-6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("W",
                        type=int,
                        nargs="?",
                        default=1,
                        help="Vectorization width")
    parser.add_argument("-test",
                        action="store_true",
                        default=False,
                        help="Perform tests (USE ONLY WITH EMULATION)")

    args = vars(parser.parse_args())
    vec_width = args["W"]
    t = args["test"]

    if t:
        test(extensive=True)
    else:
        data_shape_1 = (16, 2, 32)
        data_shape_2 = (32, 32)
        run(data_shape_1, data_shape_2, vec_width)
