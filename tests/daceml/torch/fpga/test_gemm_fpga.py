# Tests for the GEMM FPGA expansions

from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pytest
from dace.frontend.python.module import DaceModule
from dace.util import utils
from dace.transformation.onnx import InputToConstant

import dace
import copy
import argparse
from multiprocessing import Process, Queue


class Model(nn.Module):
    def __init__(self,
                 input_to_constant,
                 in_features=120,
                 out_features=84,
                 bias=None,
                 weights=None):
        super(Model, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        if input_to_constant:
            #otherwise everytime they are randomized
            self.fc.weight.data.fill_(0.1)
            self.fc.bias.data.fill_(1)
        else:
            if bias is not None:
                self.fc.bias.data = torch.from_numpy(bias)
            if weights is not None:
                self.fc.weight.data = torch.from_numpy(weights)

    def forward(self, x):
        return self.fc(x)


def run(vec_width,
        input_to_constant,
        batch_size=1000,
        input_features=120,
        output_features=84,
        execute_cpu_dace: bool = True,
        queue=None):
    '''
    Evaluates the given configuration
    :param vec_width: vectorization widht
    :param input_to_constant: true if InputToConstant transformation must be applied
    :param batch_size, input_features, output_features: data size
    :param execute_cpu_dace:
    :param queue: needed to run multiple configurations
    :return:
    '''

    x = torch.rand(batch_size, input_features, dtype=torch.float32)
    # build the DaCe model from the pytorch model
    ptmodel = Model(input_to_constant,
                    in_features=input_features,
                    out_features=output_features)
    dace_model = DaceModule(ptmodel, dummy_inputs=(x, ), auto_optimize=False)

    torch_output = ptmodel(x)
    import dace.libraries.onnx as donnx

    if execute_cpu_dace:
        with dace.library.change_default(donnx.ONNXGemm, "pure"):
            dace_output = dace_model(x)
        diff = np.linalg.norm(torch_output.detach().numpy() -
                              dace_output.numpy()) / np.linalg.norm(
                                  torch_output.detach().numpy())
        print("Difference: ", diff)
        assert np.allclose(torch_output.detach().numpy(),
                           dace_output,
                           atol=1e-06)

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
            output_data_name = sdfg.states()[0].sink_nodes()[0].data
            utils.vectorize_array_and_memlet(sdfg, output_data_name, vec_type)

        if input_to_constant:
            sdfg.apply_transformations_repeated([InputToConstant],
                                                print_report=True)

        sdfg.expand_library_nodes()
        sdfg.apply_transformations_repeated([InlineSDFG])

    # Reset the SDFG
    dace_model.reset_sdfg()
    # Append transformation hook
    dace_model.append_post_onnx_hook("TransformToFPGA", TransformToFPGA)

    # Execute Module with FPGA expansion
    with dace.library.change_default(donnx.ONNXGemm, "fpga"):
        dace_output_fpga = dace_model(torch.clone(x))
    # reshape if vec_width is different than 1
    dace_output_fpga = dace_output_fpga.detach().numpy().reshape(
        torch_output.shape)
    torch_output_np = torch_output.detach().numpy()
    diff = np.linalg.norm(torch_output_np -
                          dace_output_fpga) / np.linalg.norm(torch_output_np)
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
def test(input_to_constant=False, extensive=False):
    '''
    Evaluates multiple combination of Convolution/input size
    :param extensive: True for extensive tests
    :return:
    '''
    print(f"----------- Testing GEMM (extensive: {extensive}) ---------------")

    # Run FPGA tests in a different process to avoid issues with Intel OpenCL tools
    # (But not in parallel)

    # each position of this lists contains a test configuration
    if extensive:
        vec_width = [1, 4, 8]
        batch_size = [1000, 1000, 400]
        in_features = [120, 120, 256]
        out_features = [84, 84, 120]
    else:
        vec_width = [4]
        batch_size = [1000]
        in_features = [120]
        out_features = [84]

    for i in range(0, len(vec_width)):
        print("##########################################################")
        print(
            f"# Configuration: vw={vec_width[i]}, bs={batch_size[i]}, in_f={in_features[i]}, out_f={out_features[i]}"
        )
        print("##########################################################")
        queue = Queue()
        p = Process(target=run,
                    args=(vec_width[i], input_to_constant, batch_size[i],
                          in_features[i], out_features[i], False, queue))
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
    parser.add_argument("-input_to_constant",
                        action="store_true",
                        default=False,
                        help="Apply InputToConstant")
    parser.add_argument("-test",
                        action="store_true",
                        default=False,
                        help="Perform tests (USE ONLY WITH EMULATION)")

    args = vars(parser.parse_args())
    vec_width = args["W"]
    input_to_constant = args["input_to_constant"]
    t = args["test"]
    if t:
        test(input_to_constant, extensive=True)
    else:
        run(vec_width, input_to_constant=input_to_constant)
