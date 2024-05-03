# Tests Naive convolutions for FPGA

from dace.transformation.interstate import FPGATransformSDFG

import torch
import torch.nn as nn
import argparse
import numpy as np

from dace.frontend.python.module import DaceModule
import pytest
import dace
from dace.util import utils
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from dace.transformation.onnx import InputToConstant
from dace.transformation.dataflow import streaming_memory as sm
from dace.transformation.dataflow import PruneConnectors
from multiprocessing import Process, Queue


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 input_to_constant):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size)
        if input_to_constant:
            #fix the weight otherwise everytime they are randomized
            self.conv.weight.data.fill_(0.1)
            self.conv.bias.data.fill_(1)

    def forward(self, x):
        return self.conv(x)


def evaluate(in_channels,
             out_channels,
             kernel_size,
             data_shape: tuple,
             input_to_constant: bool,
             execute_cpu_dace: bool = False,
             queue=None):
    '''
    This function is used to evaluate a given model.
    It will build the pytorch model, transform it to a DaCe Model, apply transformation and execute on FPGA
    :return: returns if the result is correct
    '''
    # create pytorch model
    ptmodel = Model(in_channels, out_channels, kernel_size, input_to_constant)

    #create data
    x = torch.rand(data_shape)

    #evaluate pytorch model
    torch_output = ptmodel(x)

    #create dace model
    dace_model = DaceModule(ptmodel, dummy_inputs=(x, ), auto_optimize=False)

    if execute_cpu_dace:
        dace_output = dace_model(x)

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

        if input_to_constant:
            sdfg.apply_transformations_repeated([InputToConstant],
                                                print_report=True)

        sdfg.expand_library_nodes()
        sdfg.apply_transformations_repeated([InlineSDFG])

    # Reset the SDFG
    dace_model.reset_sdfg()
    # Append transformation hook
    dace_model.append_post_onnx_hook("TransformToFPGA", TransformToFPGA)

    # Execute
    import dace.libraries.onnx as donnx
    with dace.library.change_default(donnx.ONNXConv, "naive_fpga"):
        dace_output_fpga = dace_model(torch.clone(x))
    dace_output_fpga = dace_output_fpga.detach().numpy().reshape(
        torch_output.shape)

    diff = np.linalg.norm(torch_output.detach().numpy() -
                          dace_output_fpga) / np.linalg.norm(
                              torch_output.detach().numpy())
    print("Difference: ", diff)
    if queue is not None:
        # we are testing
        queue.put(diff)
    else:
        assert (diff < 1e-6)

    del dace_model, ptmodel, x


def run(input_to_constant):
    '''
    Execute the program, in hardware if required, with a fixed input size
    :return:
    '''
    # Example: second convolutional layer in Lenet
    evaluate(1, 6, 5, (100, 1, 28, 28), input_to_constant, False)


@pytest.mark.fpga
@pytest.mark.pure
def test(input_to_constant=False):
    '''
    Evaluates multiple combination of Convolution/input size
    :return:
    '''
    print("----------- Testing Naive Convolution ---------------")

    # Run FPGA tests in a different process to avoid issues with Intel OpenCL tools
    # (But not in parallel)

    ####
    # No vect
    queue = Queue()
    p = Process(target=evaluate,
                args=(1, 6, 5, (100, 1, 28, 28), input_to_constant, False,
                      queue))
    p.start()
    p.join()
    assert (queue.get() < 1e-6)

    p = Process(target=evaluate,
                args=(10, 1, 5, (100, 10, 20, 20), input_to_constant, False,
                      queue))
    p.start()
    p.join()
    assert (queue.get() < 1e-6)

    print("----------- Success! ---------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_to_constant",
                        action="store_true",
                        default=False,
                        help="Apply InputToConstant")

    parser.add_argument("-test",
                        action="store_true",
                        default=False,
                        help="Perform tests (USE ONLY WITH EMULATION)")

    args = vars(parser.parse_args())
    input_to_constant = args["input_to_constant"]
    t = args["test"]

    if t:
        test(input_to_constant)
    else:
        run(input_to_constant)
