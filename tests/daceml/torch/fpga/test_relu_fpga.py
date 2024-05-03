# Tests Relu Expansion

from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dace.frontend.python.module import DaceModule
import dace
import argparse
from dace.util import utils
from multiprocessing import Process, Queue
import pytest


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return F.relu(x)


def run(data_shape: tuple, vec_width=1, queue=None):
    '''
    Evaluates a specific configuration
    :param data_shape:
    :param vec_width:
    :param queue:
    :return:
    '''

    ptmodel = Model()
    x = torch.rand(data_shape) - 0.5
    dace_model = DaceModule(ptmodel, auto_optimize=False)
    import dace.libraries.onnx as donnx
    with dace.library.change_default(donnx.ONNXRelu, "pure"):
        dace_output = dace_model(x)

    torch_output = ptmodel(x)

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
            utils.vectorize_array_and_memlet(sdfg, "fpga_x", vec_type)
            utils.vectorize_array_and_memlet(sdfg, "fpga_ONNX_1", vec_type)

        sdfg.expand_library_nodes()
        sdfg.apply_transformations_repeated([InlineSDFG])

    # Reset the SDFG
    dace_model.reset_sdfg()

    # Append transformation hook
    dace_model.append_post_onnx_hook("TransformToFPGA", TransformToFPGA)

    # Execute Module with FPGA expansion
    with dace.library.change_default(donnx.ONNXRelu, "fpga"):
        dace_output_fpga = dace_model(x)

    dace_output_fpga = dace_output_fpga.reshape(data_shape)
    diff = np.linalg.norm(torch_output.detach().numpy() -
                          dace_output_fpga.numpy()) / np.linalg.norm(
                              torch_output.detach().numpy())
    print("Difference: ", diff)
    if queue is not None:
        # we are testing
        queue.put(diff)
    else:
        assert diff < 1e-6
    del dace_model, ptmodel, x


@pytest.mark.fpga
def test(extensive=False):
    '''
    Evaluates multiple combination of input size/vecwidth
    '''

    print(f"----------- Testing Relu (extensive: {extensive} ---------------")
    if extensive:
        vec_width = [1, 1, 2, 4]
        data_shapes = [(4, 8, 16), (100, 4, 16, 32), (8, 16, 16),
                       (1000, 4, 32, 32)]
    else:
        vec_width = [1, 4]
        data_shapes = [(4, 8, 16), (1000, 4, 32, 32)]
    for i in range(0, len(vec_width)):
        print(
            "###############################################################")
        print(
            f"# Configuration: vw={vec_width[i]}, data_shape={data_shapes[i]}")
        print(
            "###############################################################")
        queue = Queue()
        p = Process(target=run, args=(data_shapes[i], vec_width[i], queue))
        p.start()
        p.join()
        assert (queue.get() < 1e-6)
    print("Success!")


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
        run((1000, 4, 32, 32), vec_width)
