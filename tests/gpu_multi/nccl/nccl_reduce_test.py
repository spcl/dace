# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.memlet import Memlet
import dace.libraries.nccl as nccl
from dace.transformation.interstate import GPUTransformSDFG
import numpy as np
from mpi4py import MPI as MPI4PY
import pytest

###############################################################################


def make_sdfg(dtype):

    N = dace.symbol("N")

    sdfg = dace.SDFG("nccl_reduce")
    state = sdfg.add_state("dataflow")
    # sdfg.add_array('X', [N], dtype, transient=False)
    sdfg.add_array("inbuf", [N], dtype, transient=False, location={'gpu':0})
    sdfg.add_array("inbuf", [N], dtype, transient=False, location={'gpu':1})
    sdfg.add_array("inbuf", [N], dtype, transient=False, location={'gpu':2})
    sdfg.add_array("outbuf", [N], dtype, transient=False, location={'gpu':0})
    inbuf = state.add_access("inbuf")
    outbuf = state.add_access("outbuf")
    root = state.add_access("root")

    reduce_node = nccl.nodes.reduce.Reduce("reduce", op='ncclProd')
    state.add_memlet_path(inbuf,
                          reduce_node,
                          dst_conn="_inbuffer",
                          memlet=Memlet.simple(inbuf, "0:n", num_accesses=n))
    state.add_memlet_path(root,
                          reduce_node,
                          dst_conn="_root",
                          memlet=Memlet.simple(root, "0:1", num_accesses=1))
    state.add_memlet_path(reduce_node,
                          outbuf,
                          src_conn="_outbuffer",
                          memlet=Memlet.simple(outbuf, "0:n", num_accesses=n))

    return sdfg


###############################################################################


def test_nccl_reduce(dtype):
    sdfg = make_sdfg(dtype)
    sdfg.apply_transformations(GPUTransformSDFG)

    program_objects = sdfg.generate_code()
    from dace.codegen import compiler
    out_path = '.dacecache/local/nccl/'+sdfg.name
    program_folder = compiler.generate_program_folder(sdfg, program_objects,
                                                      out_path)


###############################################################################

if __name__ == "__main__":
    test_nccl_reduce(dace.float32)


###############################################################################
