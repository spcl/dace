# Testing Slice Expansion

import pytest
import torch
from torch import nn

from dace.frontend.python.module import DaceModule
from dace.testing import torch_tensors_close
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
import argparse
import dace
import numpy as np
from multiprocessing import Process, Queue


class Model(nn.Module):
    def __init__(self, start, stop):
        super(Model, self).__init__()
        self.start = start
        self.stop = stop

    def forward(self, x):
        x = x[self.start:self.stop, :]
        return x


def run(data_shape: tuple, start: int, stop: int, queue=None):
    '''
    Evaluates a specific configuration
    '''
    ptmodel = Model(start, stop)
    x = torch.rand(data_shape)

    torch_output = ptmodel(torch.clone(x))
    import dace.libraries.onnx as donnx
    with dace.library.change_default(donnx.ONNXSlice, "pure"):
        dace_model = DaceModule(
            ptmodel,
            auto_optimize=False,
            dummy_inputs=(x, ),
        )
        dace_output = dace_model(x)
    assert np.allclose(torch_output.detach().numpy(), dace_output)

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
        sdfg.apply_transformations([FPGATransformSDFG])
        sdfg.expand_library_nodes()
        sdfg.apply_transformations_repeated([InlineSDFG])

    # Reset the SDFG
    dace_model.reset_sdfg()

    # Append transformation hook
    dace_model.append_post_onnx_hook("TransformToFPGA", TransformToFPGA)

    # Execute Module with FPGA expansion
    with dace.library.change_default(donnx.ONNXSlice, "fpga"):
        dace_output_fpga = dace_model(torch.clone(x)).numpy()

    diff = np.linalg.norm(torch_output.detach().numpy() -
                          dace_output_fpga) / np.linalg.norm(
                              torch_output.detach().numpy())
    print("Difference: ", diff)
    if queue is not None:
        # we are testing
        queue.put(diff)
    else:
        assert diff < 1e-6
    del dace_model, ptmodel, x


@pytest.mark.fpga
def test():
    '''
        Evaluates multiple combination of input size/start/stop
        '''
    print("----------- Testing Slice ---------------")
    data_shapes = [(96, 32), (96, 32), (96, 32)]
    starts = [0, 32, 64]
    stops = [32, 64, -1]
    for i in range(0, len(starts)):
        print(
            "###############################################################")
        print(
            f"# Configuration: data_shape={data_shapes[i]}, start={starts[i]}, stop={stops[i]}"
        )
        print(
            "###############################################################")
        queue = Queue()
        p = Process(target=run,
                    args=(data_shapes[i], starts[i], stops[i], queue))
        p.start()
        p.join()
        assert (queue.get() < 1e-6)
    print("Success!")
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-test",
                        action="store_true",
                        default=False,
                        help="Perform tests (USE ONLY WITH EMULATION)")

    args = vars(parser.parse_args())

    t = args["test"]
    if t:
        test()
    else:
        run((96, 32), 0, 32)
