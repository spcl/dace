# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest

from dace.transformation.interstate import GPUTransformSDFG

# Define data type to use
dtype = dace.float64
np_dtype = np.float64


@dace.program
def aGPU_aCPU(a: dtype[1]):
    a += 10


@pytest.mark.skip
def test_aGPU_aCPU():
    sdfg: dace.SDFG = aGPU_aCPU.to_sdfg(strict=True)
    sdfg.name = 'aGPU_aCPU'
    sdfg.apply_transformations(GPUTransformSDFG,
                               options={'gpu_id':
                                        0})  # options={'number_of_gpus':4})
    graph = sdfg.nodes()[0]
    gpu_a = graph.nodes()[2]
    mem = graph.out_edges(gpu_a)[0].data
    mem.wcr = 'lambda old, new: new'

    np.random.seed(0)
    a = np.ndarray(shape=1, dtype=np_dtype)
    a.fill(0)

    sdfg(a=a)
    res = 10
    assert np.isclose(a, res, atol=0, rtol=1e-7)

    # program_objects = sdfg.generate_code()
    # from dace.codegen import compiler
    # out_path = '.dacecache/local/reductions/'+sdfg.name
    # program_folder = compiler.generate_program_folder(sdfg, program_objects,
    #                                                   out_path)


if __name__ == "__main__":
    test_aGPU_aCPU()