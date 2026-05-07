# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import os
import tempfile
import numpy as np

import dace as dp
from dace.sdfg import SDFG
from dace.memlet import Memlet


def test():
    # Externals (parameters, symbols)
    N = dp.symbol('N')

    @dp.program
    def sdfg_internal(input: dp.float32, output: dp.float32[1]):

        @dp.tasklet
        def init():
            inp << input
            out >> output
            out = inp

        for k in range(4):

            @dp.tasklet
            def do():
                inp << input
                oin << output
                out >> output
                out = oin * inp

    # Construct SDFG
    mysdfg = SDFG('outer_sdfg')
    mysdfg.add_array('A', [N, N], dp.float32)
    mysdfg.add_array('B', [N, N], dp.float32)
    state = mysdfg.add_state()
    A = state.add_access('A')
    B = state.add_access('B')

    map_entry, map_exit = state.add_map('elements', [('i', '0:N'), ('j', '0:N')])
    nsdfg = state.add_nested_sdfg(sdfg_internal.to_sdfg(), {'input'}, {'output'})

    # Add edges
    state.add_memlet_path(A, map_entry, nsdfg, dst_conn='input', memlet=Memlet.simple(A, 'i,j'))
    state.add_memlet_path(nsdfg, map_exit, B, src_conn='output', memlet=Memlet.simple(B, 'i,j'))

    N = 64

    input = dp.ndarray([N, N], dp.float32)
    output = dp.ndarray([N, N], dp.float32)
    input[:] = np.random.rand(N, N).astype(dp.float32.type)
    output[:] = dp.float32(0)

    mysdfg(A=input, B=output, N=N)

    diff = np.linalg.norm(output - np.power(input, 5)) / (N * N)
    assert diff <= 1e-5


def test_external_nsdfg():
    N = dp.symbol('N')

    @dp.program
    def sdfg_internal(input: dp.float32, output: dp.float32[1]):

        @dp.tasklet
        def init():
            inp << input
            out >> output
            out = inp

        for k in range(4):

            @dp.tasklet
            def do():
                inp << input
                oin << output
                out >> output
                out = oin * inp

    # Construct SDFG
    mysdfg = SDFG('outer_sdfg')
    mysdfg.add_array('A', [N, N], dp.float32)
    mysdfg.add_array('B', [N, N], dp.float32)
    state = mysdfg.add_state()
    A = state.add_access('A')
    B = state.add_access('B')

    map_entry, map_exit = state.add_map('elements', [('i', '0:N'), ('j', '0:N')])
    internal = sdfg_internal.to_sdfg()
    fd, filename = tempfile.mkstemp(suffix='.sdfg')
    internal.save(filename)
    nsdfg = state.add_nested_sdfg(None, {'input'}, {'output'}, name='sdfg_internal', external_path=filename)

    # Add edges
    state.add_memlet_path(A, map_entry, nsdfg, dst_conn='input', memlet=Memlet.simple(A, 'i,j'))
    state.add_memlet_path(nsdfg, map_exit, B, src_conn='output', memlet=Memlet.simple(B, 'i,j'))

    N = 64

    input = dp.ndarray([N, N], dp.float32)
    output = dp.ndarray([N, N], dp.float32)
    input[:] = np.random.rand(N, N).astype(dp.float32.type)
    output[:] = dp.float32(0)

    mysdfg(A=input, B=output, N=N)

    diff = np.linalg.norm(output - np.power(input, 5)) / (N * N)
    assert diff <= 1e-5

    os.close(fd)


if __name__ == "__main__":
    test()
    test_external_nsdfg()
