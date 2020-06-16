import dace
import numpy as np

from dace.perf.roofline import Roofline
from dace.perf.specs import *
from dace.perf.optimizer import SDFGRooflineOptimizer

from dace.transformation.heterogeneous import ReduceMap
from dace.transformation.heterogeneous import SubgraphFusion
from dace.transformation.heterogeneous import MultiExpansion

import dace.libraries.standard as stdlib

import timeit

import dace.measure.pipeline as pipeline
from dace.measure.runner import Runner


dace_dtype = dace.float32
H, B, SN, SM = (dace.symbol(s) for s in ('H', 'B', 'SN', 'SM'))


@dace.program
def softmax(X_in: dace_dtype[H, B, SN, SM]):
    tmp_max = dace.reduce(lambda a, b: max(a, b), X_in, axis=3, identity = 0)

    tmp_out = np.ndarray([H, B, SN, SM], dtype=dace_dtype)
    out = np.ndarray([H, B, SN, SM], dtype=dace_dtype)

    # No broadcasting rules
    for i, j, k, l in dace.map[0:H, 0:B, 0:SN, 0:SM]:
        with dace.tasklet:
            inp << X_in[i, j, k, l]
            mx << tmp_max[i, j, k]
            o >> tmp_out[i, j, k, l]
            o = math.exp(inp - mx)
    #tmp_out = np.exp(X_in - tmp_max)

    tmp_sum = dace.reduce(lambda a, b: a + b, tmp_out, identity=0, axis=3)
    for i, j, k, l in dace.map[0:H, 0:B, 0:SN, 0:SM]:
        with dace.tasklet:
            inp << tmp_out[i, j, k, l]
            sm << tmp_sum[i, j, k]
            o >> out[i, j, k, l]
            o = inp / sm

    return out


sdfg = softmax.to_sdfg()
roofline = Roofline(PERF_CPU_CRAPBOOK, symbols = {H:3, B:3, SN:5, SM:5})
H.set(10); B.set(10); SN.set(50); SM.set(50)


def test_graph():
    ################ first, expand the reduce node
    print(sdfg.nodes()[0])
    sdfg.view()
    #sdfg.apply_gpu_transformations()
    print(sdfg.nodes()[0])
    sdfg.view()
    pipeline.expand_reduce(sdfg, sdfg.nodes()[0])
    sdfg.view()

    ############### second, do MultiExpansion
    pipeline.expand_maps(sdfg, sdfg.nodes()[0])
    sdfg.view()

    ############ third, do MapFusion
    pipeline.fusion(sdfg, sdfg.nodes()[0])

    sdfg.apply_strict_transformations()
    sdfg.view()

    sdfg.validate()


def test_result(debug = False):

    debugger = Runner(measure_mode = ['median', 'avg', 'std'],
                      view_roofline = True)

    debugger.go(sdfg, sdfg.nodes()[0], None, H, B, SN, SM,
                performance_spec = dace.perf.specs.PERF_CPU_CRAPBOOK,
                output=[])

    #############


if __name__ == "__main__":
    #test_result()
    #sdfg.apply_gpu_transformations()
    #graph2 = sdfg.nodes()[0]
    #print(graph2)
    #sdfg.view()
    test_graph()
