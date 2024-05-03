# MaxPool expansion, simple testing

# TODO: add more testing

import torch
import torch.nn as nn
import torch.nn.functional as F
import dace
import pytest
import numpy as np
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from dace.util import utils

import dace.libraries.onnx as donnx
from dace.frontend.python.module import DaceModule
import copy
import argparse
from multiprocessing import Process, Queue


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return F.max_pool2d(x, 2)


def run(data_shape: tuple, vec_width=1, queue=None, vec_width_out=1):
    '''
    Evaluates specific configurations
    :param data_shape:
    :param vec_width:
    :param queue:
    :return:
    '''

    ptmodel = Model()
    x = torch.rand(data_shape)

    dace_model = DaceModule(ptmodel, auto_optimize=False)
    import dace.libraries.onnx as donnx
    with dace.library.change_default(donnx.ONNXMaxPool, "pure"):
        dace_output = dace_model(x)
    torch_output = ptmodel(x)

    ##################################
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
            utils.vectorize_array_and_memlet(sdfg, "fpga_ONNX_0", vec_type)

        if vec_width_out > 1:
            vec_type = dace.vector(dace.float32, vec_width_out)
            utils.vectorize_array_and_memlet(sdfg, "fpga_ONNX_1", vec_type)

        sdfg.expand_library_nodes()
        sdfg.apply_transformations_repeated([InlineSDFG])

    # Reset the SDFG
    dace_model.reset_sdfg()
    # Append transformation hook
    dace_model.append_post_onnx_hook("TransformToFPGA", TransformToFPGA)

    # Execute Module with FPGA expansion
    with dace.library.change_default(donnx.ONNXMaxPool, "fpga"):
        dace_output_fpga = dace_model(torch.clone(x))
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
@pytest.mark.xilinx
def test():
    '''
       TODO: add more testing
    '''
    # Multiprocess is needed for testing otherwise Intel Compiler mess up with threads
    queue = Queue()
    vendor = dace.config.Config.get("compiler", "fpga_vendor")
    print("[MAXPOOL]", vendor)

    # Baseline case
    data_shape = (1000, 6, 32, 32)
    p = Process(target=run, args=(data_shape, 1, queue, 1))
    p.start()
    p.join()
    assert (queue.get() < 1e-6)

    # Vectorized input and output
    data_shape = (1000, 6, 32, 32)
    p = Process(target=run, args=(data_shape, 2, queue, 2))
    p.start()
    p.join()
    assert (queue.get() < 1e-6)

    # Vectorized input and output, output vectorized by in_vec / kernel
    # thus in_vec: 8 --> out_vec: 4 for testing kernel of s
    data_shape = (1000, 6, 32, 32)
    p = Process(target=run, args=(data_shape, 8, queue, 4))
    p.start()
    p.join()
    assert (queue.get() < 1e-6)

    # Intel only tests
    if vendor != "xilinx":
        # Vectorized input with unrolled writes (Intel)
        data_shape = (1000, 6, 32, 32)
        p = Process(target=run, args=(data_shape, 4, queue, 1))
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
        test()
    else:
        data_shape = (1000, 6, 32, 32)
        run(data_shape, vec_width)
