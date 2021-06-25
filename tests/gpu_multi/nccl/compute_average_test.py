# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace import dtypes
from dace.memlet import Memlet
import dace.libraries.nccl as nccl
from dace.transformation.interstate import GPUTransformSDFG
from dace.transformation.dataflow import GPUMultiTransformMap
import numpy as np
from mpi4py import MPI as MPI4PY
import pytest

###############################################################################
N = dace.symbol('N')
N.set(30)

@dace.program
def nccl_reduction(X: dace.float32[N], output: dace.float32[N]):
    for i in dace.map[0:N]:
        X[i]=i
    dace.nccl.Reduce(X, output, 'ncclSum', 0)




    



###############################################################################


def test_nccl_reduce(dtype):
    sdfg = nccl_reduction.to_sdfg(strict=False)
    sdfg.apply_transformations(GPUMultiTransformMap)
    sdfg.apply_strict_transformations()
    graph = sdfg.start_state
    output = graph.sink_nodes()[0]
    reduction=graph.predecessors(output)[0]
    Xn=graph.predecessors(reduction)[0]
    mapExit=graph.predecessors(Xn)[0]
    mp = graph.memlet_path(graph.in_edges(Xn)[0])
    graph.add_memlet_path(mp[0].src, mp[0].dst, reduction, memlet=mp[0].data, dst_conn='_inbuffer')
    graph.remove_memlet_path(graph.in_edges(Xn)[0])
    graph.remove_memlet_path(graph.out_edges(Xn)[0])
    sdfg.add_array('cpu_output', shape=[N], dtype = dace.float32)
    cpu_out = graph.add_access('cpu_output')
    graph.add_memlet_path(output, cpu_out, memlet=Memlet.from_array(output, output.desc(sdfg)))
    output.desc(sdfg).location['gpu'] = 0
    output.desc(sdfg).storage=dtypes.StorageType.GPU_Global

    program_objects = sdfg.generate_code()
    from dace.codegen import compiler
    out_path = '.dacecache/local/nccl/'+sdfg.name
    program_folder = compiler.generate_program_folder(sdfg, program_objects,
                                                      out_path)


###############################################################################

if __name__ == "__main__":
    test_nccl_reduce(dace.float32)


###############################################################################
