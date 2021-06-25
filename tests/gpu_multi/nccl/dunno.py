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
dtype = dace.float32
# @dace.program
# def nccl_reduction(X: dace.float32[N], output: dace.float32[N]):
#     for i in dace.map[0:N]:
#         X[i]=i
#     dace.nccl.Reduce(X, output, 'ncclSum', 0)



def make_sdfg():
    red_sdfg = dace.SDFG('reduction')

    red_sdfg.add_array('input', [N], dtype)
    red_sdfg.add_array('output', [N], dtype)

    reduction_state = red_sdfg.add_state('reduction_state')
    input = reduction_state.add_read('input')
    output = reduction_state.add_write('output')
    reduction = nccl.Allreduce('ncclAllreduce', operation='ncclSum', location={'gpu':'gpu_id'})

    # input path (input->reduction)
    reduction_state.add_memlet_path(input,
                          reduction,
                          dst_conn='_inbuffer',
                          memlet=dace.Memlet.simple('input', '0'))
    
    # output path (reduction->output)
    reduction_state.add_memlet_path(reduction,
                          output,
                          src_conn='_outbuffer',
                          memlet=dace.Memlet.simple('output', '0'))
    sdfg = dace.SDFG('nccl_allreduce')
    sdfg.add_array('x', [4*N], dtype)
    sdfg.add_transient('gpu_x', [N], dtype, dtypes.StorageType.GPU_Global, location={'gpu':'gpu_id'})
    sdfg.add_transient('gpu_output', [N], dtype, dtypes.StorageType.GPU_Global, location={'gpu':'gpu_id'})
    sdfg.add_array('output', [N], dtype)

    state = sdfg.add_state('all')
    me, mx = state.add_map('multiGPU', dict(gpu_id='0:4'), dtypes.ScheduleType.GPU_Multidevice)
    
    nsdfg = state.add_nested_sdfg(red_sdfg, sdfg, {'input'}, {'output'})
    xin = state.add_read('x')
    gpux = state.add_access('gpu_x')
    gpuout = state.add_access('gpu_output')
    
    redout = state.add_read('output')

    # Connect dataflow nodes
    state.add_memlet_path(xin,
                        me,
                        gpux,
                        # nsdfg,
                        memlet=dace.Memlet('x[gpu_id*N:(gpu_id+1)*N]'))
    state.add_edge(gpux, None, nsdfg, 'input', dace.Memlet.from_array(gpux, gpux.desc(sdfg)))
    state.add_edge(nsdfg, 'output', gpuout, None, dace.Memlet.from_array(gpuout, gpuout.desc(sdfg)))
    # Connect dataflow nodes
    # state.add_memlet_path(xin,
    #                     me,
    #                     nsdfg,
    #                     memlet=dace.Memlet.simple('x', 'gpu'),
    #                     dst_conn='input')
    state.add_memlet_path(gpuout,
                        mx,
                        redout,
                        memlet=dace.Memlet('output[0:N]'))
    
    sdfg.remove_symbol('gpu_id')
    
    return sdfg

    





###############################################################################

def test_nccl_reduce():
    sdfg = make_sdfg()
    sdfg.apply_strict_transformations()
    # sdfg.view()

    program_objects = sdfg.generate_code()
    from dace.codegen import compiler
    out_path = '.dacecache/local/nccl/'+sdfg.name
    program_folder = compiler.generate_program_folder(sdfg, program_objects,
                                                      out_path)


###############################################################################

if __name__ == "__main__":
    test_nccl_reduce()


###############################################################################
