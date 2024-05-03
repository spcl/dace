import torch
import numpy as np
import pytest

from dace.frontend.python.module import DaceModule

from dace.transformation.dataflow import RedundantSecondArray
from dace.transformation.onnx import ConstantFolding

from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from dace.transformation.dataflow import PruneConnectors
from dace.transformation.dataflow import streaming_memory as sm
from dace import StorageType
from dace import SDFG
from multiprocessing import Process, Queue
import argparse
import dace
import pytest
from dace.util import utils
###################################################################
# Transformer configurations to be used for MHA
# Note:
# - base and large, refer to original Bert model
# - tiny and small are just for testing
# - lu20, refers to the test configuration from "Hardware Accelerator for Multi-Head Attention and
#       Position-Wise Feed-Forward in the Transformer" by Lu et al. They use the original transformer base model

# Key:
# H = #Heads
# P = #projections
# N = # features (sometimes referred as d_model)
# SM, SN = input/output sequence length
# numb_emb= 4N (after MHA, sometimes referred as feed forward filter size or d_ff)
# Typically, N = P*H
configurations = {
    "tiny": {
        "H": 4,
        "P": 8,
        "N": 32,
        "SM": 16,
        "SN": 16
    },
    "small": {
        "H": 12,
        "P": 32,
        "N": 384,
        "SM": 32,
        "SN": 32
    },
    "base": {
        "H": 12,
        "P": 64,
        "N": 768,
        "SM": 128,
        "SN": 128
    },
    "large": {
        "H": 16,
        "P": 64,
        "N": 1024,
        "SM": 512,
        "SN": 512
    },
    "lu20": {
        "H": 8,
        "P": 64,
        "N": 512,
        "SM": 64,
        "SN": 64
    },
}


def evaluate(batch_size=1,
             configuration_name="tiny",
             execute_cpu_dace=False,
             queue=None):

    B = batch_size
    conf = configurations[configuration_name]
    H = conf["H"]
    P = conf["P"]
    N = conf["N"]
    SM = conf["SM"]
    SN = conf["SN"]

    print("******************************************************")
    print("Executing MHA with configuration: ", configuration_name)
    print("B: ", B, " H: ", H, " P: ", P, " N: ", N, " SM: ", SM, " SN:", SN)
    print("******************************************************")

    #############

    K, Q, V = [
        torch.randn([SM, B, N]),
        torch.randn([SN, B, N]),
        torch.randn([SM, B, N])
    ]
    ptmodel = torch.nn.MultiheadAttention(N, H, bias=False)

    pt_outputs = ptmodel(Q, K, V)

    import dace.libraries.onnx as donnx
    old_default = donnx.default_implementation

    try:
        donnx.default_implementation = "pure"

        if execute_cpu_dace:
            dace_model = DaceModule(ptmodel,
                                    dummy_inputs=(Q, K, V),
                                    auto_optimize=False)
            # dace_outputs_0 = dace_model(Q, K, V)

        else:
            dace_model = DaceModule(ptmodel,
                                    dummy_inputs=(Q, K, V),
                                    auto_optimize=False)

        ################################################
        # Apply transformations
        dace_model.dace_model.sdfg.apply_transformations_repeated(
            [ConstantFolding, RedundantSecondArray],
            validate_all=True,
            print_report=True)
        if execute_cpu_dace:
            dace_outputs_1 = dace_model(Q, K, V)
            assert np.allclose(pt_outputs[0].detach().numpy(),
                               dace_outputs_1[0],
                               atol=1e-06)
            assert np.allclose(pt_outputs[1].detach().numpy(),
                               dace_outputs_1[1],
                               atol=1e-06)

        ##########################################
        # Transform to FPGA

        def TransformToFPGA(dace_module):
            '''
            Transforms the given module to run on FPGA.
            This includes (vectorization and) library node expansions.
            :param dace_module:
            :return:
            '''
            sdfg = dace_module.sdfg
            sdfg.apply_transformations([FPGATransformSDFG])

            # Vectorize container (if needed)
            # TODO:
            # vec_width = 4  # we can not go further in this because of the systolic organization
            # vec_type = dace.vector(dace.float32, vec_width)
            # #
            # # #vectorize input B matmul, output not vectorized
            # input_data_name = "ONNX_26"
            # utils.vectorize_array_and_memlet(sdfg, input_data_name, vec_type)
            # print("Applying vectorization {} to Array {}".format(
            #     vec_width, input_data_name))
            #
            # # vectorize input B matmul, output not vectorized
            # input_data_name = "ONNX_36"
            # utils.vectorize_array_and_memlet(sdfg, input_data_name, vec_type)
            # print("Applying vectorization {} to Array {}".format(
            #     vec_width, input_data_name))
            #
            # # vectorize input B matmul, output not vectorized
            # input_data_name = "ONNX_47"
            # utils.vectorize_array_and_memlet(sdfg, input_data_name, vec_type)
            # ##################################

            sdfg.expand_library_nodes()
            sdfg.apply_transformations_repeated([InlineSDFG])
            sdfg.apply_transformations_repeated(PruneConnectors)

        # Reset the SDFG
        dace_model.reset_sdfg()

        # Append transformation hook
        dace_model.append_post_onnx_hook("TransformToFPGA", TransformToFPGA)

        # Execute Module with FPGA expansion

        with dace.library.change_default(
                donnx.ONNXMatMul, "fpga"), dace.library.change_default(
                    donnx.ONNXReshape, "fpga"), dace.library.change_default(
                        donnx.ONNXSoftmax,
                        "fpga"), dace.library.change_default(
                            donnx.ONNXReduceSum,
                            "fpga"), dace.library.change_default(
                                donnx.ONNXSlice, "fpga"):
            dace_output_fpga = dace_model(Q, K, V)

    finally:
        donnx.default_implementation = old_default
    if queue is not None:
        diff0 = np.linalg.norm(pt_outputs[0].detach().numpy() -
                               dace_output_fpga[0].numpy()) / np.linalg.norm(
                                   pt_outputs[0].detach().numpy())
        diff1 = np.linalg.norm(pt_outputs[1].detach().numpy() -
                               dace_output_fpga[1].numpy()) / np.linalg.norm(
                                   pt_outputs[1].detach().numpy())
        queue.put(diff0)
        queue.put(diff1)
    else:
        assert np.allclose(pt_outputs[0].detach().numpy(),
                           dace_output_fpga[0],
                           atol=1e-06)
        assert np.allclose(pt_outputs[1].detach().numpy(),
                           dace_output_fpga[1],
                           atol=1e-06)
    del dace_model, ptmodel, Q, K, V


@pytest.mark.fpga
def test():
    # Multiprocess is needed for testing otherwise Intel Compiler mess up with threads
    queue = Queue()
    p = Process(target=evaluate, args=(1, "tiny", False, queue))
    p.start()
    p.join()
    assert (queue.get() < 1e-6)
    assert (queue.get() < 1e-6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("B", type=int, nargs="?", default=1, help="Batch size")
    parser.add_argument("conf",
                        type=str,
                        nargs="?",
                        default="tiny",
                        help="Configuration")

    args = vars(parser.parse_args())
    B = args["B"]
    conf = args["conf"]
    evaluate(B, conf, False)
