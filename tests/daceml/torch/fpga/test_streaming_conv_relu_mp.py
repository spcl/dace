# Simple test for evaluating Conv-Relu-Maxpool in streaming composition

from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from dace.transformation.onnx import InputToConstant

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import dace
from dace.frontend.python.module import DaceModule

from dace.util import utils
from dace.transformation.dataflow import streaming_memory as sm
from dace.transformation.interstate import InlineSDFG
import argparse
import pytest
from multiprocessing import Process, Queue


class Model(nn.Module):
    def __init__(self, input_to_constant=False):
        super(Model, self).__init__()
        #first conv
        self.conv = nn.Conv2d(1, 6, 5)
        #second conv
        # self.conv = nn.Conv2d(6, 16, 5)
        if input_to_constant:
            #fix the weight otherwise everytime they are randomized
            self.conv.weight.data.fill_(0.1)
            self.conv.bias.data.fill_(1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv(x)), 2)
        return x


def run(data_shape, vec_width=1, input_to_constant=False, queue=None):

    ptmodel = Model(input_to_constant)

    x = torch.rand(data_shape)
    dace_model = DaceModule(ptmodel, auto_optimize=False)
    import dace.libraries.onnx as donnx
    with dace.library.change_default(donnx.ONNXConv,
                                     "pure"), dace.library.change_default(
                                         donnx.ONNXRelu,
                                         "pure"), dace.library.change_default(
                                             donnx.ONNXMaxPool, "pure"):
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
            utils.vectorize_array_and_memlet(sdfg, "ONNX_3", vec_type)
            utils.vectorize_array_and_memlet(sdfg, "ONNX_4", vec_type)

        sdfg.expand_library_nodes()
        sdfg.apply_transformations_repeated([InlineSDFG])

        if input_to_constant:
            sdfg.apply_transformations_repeated([InputToConstant],
                                                print_report=True)
        sdfg.apply_transformations_repeated(
            [InlineSDFG, sm.StreamingComposition],
            [{}, {
                "storage": dace.StorageType.FPGA_Local
            }])

    # Reset the SDFG
    dace_model.reset_sdfg()

    # Append transformation hook
    dace_model.append_post_onnx_hook("TransformToFPGA", TransformToFPGA)

    # Execute Module with FPGA expansion
    with dace.library.change_default(donnx.ONNXConv,
                                     "fpga"), dace.library.change_default(
                                         donnx.ONNXRelu,
                                         "fpga"), dace.library.change_default(
                                             donnx.ONNXMaxPool, "fpga"):

        dace_output_fpga = dace_model(torch.clone(x))

    dace_output_fpga = dace_output_fpga.reshape(dace_output.shape)

    torch_output_numpy = torch_output.detach().numpy()
    diff = np.linalg.norm(torch_output_numpy - dace_output_fpga.numpy()
                          ) / np.linalg.norm(torch_output_numpy)

    print("Difference: ", diff)
    if queue is not None:
        queue.put(diff)
    else:
        assert (diff < 1e-6)
    del ptmodel, dace_model, x


@pytest.mark.fpga
def test(vec_width=1, input_to_constant=False):
    data_shape = (100, 1, 28, 28)
    # Multiprocess is needed for testing otherwise Intel Compiler mess up with threads
    queue = Queue()
    p = Process(target=run, args=(data_shape, 1, False, queue))
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
    parser.add_argument("-input_to_constant",
                        action="store_true",
                        default=False,
                        help="Apply InputToConstant")

    args = vars(parser.parse_args())
    vec_width = args["W"]
    input_to_constant = args["input_to_constant"]
    # first conv
    data_shape = (100, 1, 28, 28)
    # second conv
    # data_shape = (100, 6, 12, 12)
    run(data_shape, vec_width, input_to_constant)
