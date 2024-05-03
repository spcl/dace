# Simple test for softmax for FPGA

# NOTE: for the moment being it supports only the last axis

# TODO: add more tests

from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from dace.frontend.python.module import DaceModule
import argparse
import pytest
from multiprocessing import Process, Queue
import dace


class Model(nn.Module):
    def __init__(self, axis):
        super(Model, self).__init__()
        self.axis = axis

    def forward(self, x):
        x = F.softmax(x, dim=self.axis)
        return x


def run(data_shape: tuple, axis, queue=None):

    ptmodel = Model(axis)
    x = torch.rand(data_shape, )

    dace_model = DaceModule(ptmodel, auto_optimize=False)
    import dace.libraries.onnx as donnx
    with dace.library.change_default(donnx.ONNXSoftmax, "pure"):
        dace_output = dace_model(x)

    torch_output = ptmodel(x)

    assert np.allclose(torch_output.detach().numpy(), dace_output, atol=1e-06)

    ##########################################
    # Transform to FPGA

    def TransformToFPGA(dace_module):
        '''
        Transforms the given module to run on FPGA.
        This includes library node expansions.
        :param dace_module:
        :return:
        '''
        sdfg = dace_module.sdfg
        sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG])

        sdfg.expand_library_nodes()
        sdfg.apply_transformations_repeated([InlineSDFG])

    # Reset the SDFG
    dace_model.reset_sdfg()

    # Append transformation hook
    dace_model.append_post_onnx_hook("TransformToFPGA", TransformToFPGA)

    # Execute Module with FPGA expansion
    with dace.library.change_default(donnx.ONNXSoftmax, "fpga"):
        dace_output_fpga = dace_model(torch.clone(x)).numpy()

    diff = np.linalg.norm(torch_output.detach().numpy() -
                          dace_output_fpga) / dace_output_fpga.size

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
def test():
    data_shape = (1000, 10, 10)
    # Multiprocess is needed for testing otherwise Intel Compiler mess up with threads
    queue = Queue()
    p = Process(target=run, args=(data_shape, 2, queue))
    p.start()
    p.join()
    assert (queue.get() < 1e-6)
    #TODO: add more tests


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
        data_shape = (1000, 10, 10)
        run(data_shape, 2)
