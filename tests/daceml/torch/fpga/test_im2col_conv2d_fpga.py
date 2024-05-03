# Tests for Im2Col 2D convolutions for FPGA

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
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 input_to_constant,
                 padding=0):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=padding)
        if input_to_constant:
            #fix the weight otherwise everytime they are randomized
            self.conv.weight.data.fill_(0.1)
            self.conv.bias.data.fill_(1)

    def forward(self, x):
        return self.conv(x)


def evaluate(in_channels,
             out_channels,
             kernel_size,
             vec_width,
             data_shape: tuple,
             input_to_constant: bool,
             execute_cpu_dace: bool = False,
             queue=None,
             padding: int = 0,
             expansion="fpga",
             activation=None):
    '''
    This function is used to evaluate a given model.
    It will build the pytorch model, transform it to a DaCe Model, apply transformation and execute on FPGA
    :return: returns if the result is correct
    '''
    # create pytorch model
    ptmodel = Model(in_channels, out_channels, kernel_size, input_to_constant,
                    padding)

    #create data
    x = torch.rand(data_shape)

    #evaluate pytorch model
    torch_output = ptmodel(x)

    #create dace model
    dace_model = DaceModule(ptmodel, dummy_inputs=(x, ), auto_optimize=False)

    import dace.libraries.onnx as donnx
    if execute_cpu_dace:
        with dace.library.change_default(donnx.ONNXConv, "pure"):
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

        # Vectorize container (if needed)
        if vec_width > 1:
            vec_type = dace.vector(dace.float32, vec_width)
            utils.vectorize_array_and_memlet(sdfg, "fpga_ONNX_3", vec_type)

            # vectorize input too for tiled implementation
            if "tile" in expansion:
                utils.vectorize_array_and_memlet(sdfg, "fpga_input", vec_type)

        # pass expansion parameters for tiled implementation
        if "tile" in expansion:
            node, state = utils.get_library_node_by_name(sdfg, "Conv_0")
            node.expand(sdfg,
                        state,
                        tiles=(torch_output.shape[3] * 8),
                        pe=4,
                        activation=activation)
        else:
            sdfg.expand_library_nodes()
        sdfg.apply_transformations_repeated([InlineSDFG])
        # # Input to constant
        if input_to_constant:
            sdfg.apply_transformations_repeated([InputToConstant],
                                                print_report=True)

    # Reset the SDFG
    dace_model.reset_sdfg()
    # Append transformation hook
    dace_model.append_post_onnx_hook("TransformToFPGA", TransformToFPGA)

    # Execute Module with FPGA expansion
    with dace.library.change_default(donnx.ONNXConv, expansion):
        dace_output_fpga = dace_model(torch.clone(x))

    dace_output_fpga = dace_output_fpga.detach().numpy().reshape(
        torch_output.shape)

    torch_output_numpy = torch_output.detach().numpy()

    # perform additional activation if operator was
    # configured to perform activation before writing result
    if activation is not None and activation == "relu":
        torch_output_numpy = np.maximum(0, torch_output_numpy)
    diff = np.linalg.norm(torch_output_numpy - dace_output_fpga
                          ) / np.linalg.norm(torch_output_numpy)
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
    evaluate(1,
             6,
             5,
             1, (100, 1, 28, 28),
             input_to_constant,
             False,
             padding=0,
             expansion="fpga")


@pytest.mark.fpga
def test(input_to_constant=False, extensive=False):
    '''
    Evaluates multiple combination of Convolution/input size
    :param extensive: True for extensive tests
    :return:
    '''
    print(
        f"----------- Testing Convolution (extensive: {extensive}) ---------------"
    )
    # Run FPGA tests in a different process to avoid issues with Intel OpenCL tools
    # (But not in parallel)

    ####
    # No vect
    queue = Queue()
    p = Process(target=evaluate,
                args=(1, 6, 5, 1, (100, 1, 28, 28), input_to_constant, False,
                      queue))
    p.start()
    p.join()
    assert (queue.get() < 1e-6)

    if extensive:
        p = Process(target=evaluate,
                    args=(10, 1, 5, 1, (100, 10, 20, 20), input_to_constant,
                          False, queue))
        p.start()
        p.join()
        assert (queue.get() < 1e-6)

        p = Process(target=evaluate,
                    args=(14, 8, 3, 1, (100, 14, 20, 20), input_to_constant,
                          False, queue))
        p.start()
        p.join()
        assert (queue.get() < 1e-6)

    # With Vectorization
    # The first two are from Lenet
    p = Process(target=evaluate,
                args=(1, 6, 5, 8, (100, 1, 28, 28), input_to_constant, False,
                      queue))
    p.start()
    p.join()
    assert (queue.get() < 1e-6)

    p = Process(target=evaluate,
                args=(6, 16, 5, 8, (100, 6, 12, 12), input_to_constant, False,
                      queue))
    p.start()
    p.join()
    assert (queue.get() < 1e-6)

    if extensive:

        p = Process(target=evaluate,
                    args=(6, 4, 5, 4, (100, 6, 12, 12), input_to_constant,
                          False, queue))
        p.start()
        p.join()
        assert (queue.get() < 1e-6)

        p = Process(target=evaluate,
                    args=(3, 3, 3, 16, (100, 3, 34, 34), input_to_constant,
                          False, queue))
        p.start()
        p.join()
        assert (queue.get() < 1e-6)

    # With padding
    queue = Queue()
    p = Process(target=evaluate,
                args=(1, 6, 3, 1, (100, 1, 28, 28), input_to_constant, False,
                      queue, 1))
    p.start()
    p.join()
    assert (queue.get() < 1e-6)

    queue = Queue()
    p = Process(target=evaluate,
                args=(1, 6, 3, 4, (100, 1, 28, 28), input_to_constant, False,
                      queue, 1))
    p.start()
    p.join()
    assert (queue.get() < 1e-6)

    if extensive:

        queue = Queue()
        p = Process(target=evaluate,
                    args=(1, 6, 5, 1, (100, 1, 12, 12), input_to_constant,
                          False, queue, 2))
        p.start()
        p.join()
        assert (queue.get() < 1e-6)

        queue = Queue()
        p = Process(target=evaluate,
                    args=(1, 6, 5, 2, (100, 1, 12, 12), input_to_constant,
                          False, queue, 1))
        p.start()
        p.join()
        assert (queue.get() < 1e-6)

    print("----------- Success! ---------------")


@pytest.mark.fpga
def test_tiled(input_to_constant=False):
    '''
    Evaluates multiple combination of Convolution/input size
    using the tiled and vectori aligned implementation
    :param extensive: True for extensive tests
    :return:
    '''
    print(
        f"----------- Testing Convolution (tiled and aligned vectors) ---------------"
    )

    # Run FPGA tests in a different process to avoid issues with Intel OpenCL tools
    # (But not in parallel)

    # with tiling
    queue = Queue()
    p = Process(target=evaluate,
                args=(1, 6, 5, 1, (100, 1, 28, 28), input_to_constant, False,
                      queue, 0, "fpga_tiled"))
    p.start()
    p.join()
    assert (queue.get() < 1e-6)

    p = Process(target=evaluate,
                args=(14, 8, 3, 1, (100, 14, 12, 12), input_to_constant, False,
                      queue, 0, "fpga_tiled"))
    p.start()
    p.join()
    assert (queue.get() < 1e-6)

    # With Vectorization
    p = Process(target=evaluate,
                args=(2, 6, 5, 2, (100, 2, 28, 28), input_to_constant, False,
                      queue, 0, "fpga_tiled"))
    p.start()
    p.join()
    assert (queue.get() < 1e-6)

    p = Process(target=evaluate,
                args=(6, 4, 5, 4, (100, 6, 32, 32), input_to_constant, False,
                      queue, 0, "fpga_tiled"))
    p.start()
    p.join()
    assert (queue.get() < 1e-6)

    # With padding
    queue = Queue()
    p = Process(target=evaluate,
                args=(1, 6, 3, 4, (100, 1, 28, 28), input_to_constant, False,
                      queue, 1, "fpga_tiled"))
    p.start()
    p.join()
    assert (queue.get() < 1e-6)

    # with relu activation on tiled implementation
    queue = Queue()
    p = Process(target=evaluate,
                args=(1, 6, 3, 4, (100, 1, 28, 28), input_to_constant, False,
                      queue, 1, "fpga_tiled", "relu"))
    p.start()
    p.join()
    assert (queue.get() < 1e-6)

    print("----------- Success! ---------------")


@pytest.mark.fpga
@pytest.mark.xilinx
def test_tiled_xilinx(input_to_constant=False):
    '''
    Test Xilinx specific features:
        - Prefetching of bias values
    '''
    print(f"----------- Testing Convolution (Xilinx) ---------------")

    queue = Queue()
    p = Process(target=evaluate,
                args=(14, 8, 5, 1, (100, 14, 12, 12), input_to_constant, False,
                      queue, 0, "fpga_tiled"))
    p.start()
    p.join()
    assert (queue.get() < 1e-6)

    p = Process(target=evaluate,
                args=(3, 6, 3, 4, (100, 3, 28, 28), input_to_constant, False,
                      queue, 1, "fpga_tiled"))
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
        test(input_to_constant, extensive=True)
        test_tiled()
        # test_tiled_xilinx()
    else:
        run(input_to_constant)
