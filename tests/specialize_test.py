#!/usr/bin/env python3
import math
import numpy as np

import dace as dp
from dace.sdfg import SDFG
from dace.memlet import Memlet

if __name__ == '__main__':
    print('Constant specialization test')

    N = dp.symbol('N')
    M = dp.symbol('M')
    N.set(20)
    M.set(30)
    fullrange = '1:N-1,0:M'
    irange = '1:N-1'
    jrange = '0:M'

    input = np.random.rand(N.get(), M.get()).astype(np.float32)
    output = dp.ndarray([N, M], dtype=dp.float32)
    output[:] = dp.float32(0)

    ##########################################################################
    spec_sdfg = SDFG('spectest')
    state = spec_sdfg.add_state()
    A = state.add_array('A', [N, M], dp.float32)
    Atrans = state.add_transient('At', [N - 2, M], dp.float32)
    B = state.add_array('B', [N, M], dp.float32)

    state.add_edge(A, None, Atrans, None, Memlet.simple(A, fullrange))
    _, me, mx = state.add_mapped_tasklet(
        'compute', dict(i=irange, j=jrange),
        dict(a=Memlet.simple(Atrans, 'i-1,j')), 'b = math.exp(a)',
        dict(b=Memlet.simple(B, 'i,j')))
    state.add_edge(Atrans, None, me, None, Memlet.simple(Atrans, fullrange))
    state.add_edge(mx, None, B, None, Memlet.simple(B, fullrange))
    ##########################################################################

    code_nonspec = spec_sdfg.generate_code()

    if 'Dynamic' not in code_nonspec[0].code:
        print('ERROR: Constants were needlessly specialized')
        exit(1)

    spec_sdfg.specialize(dict(N=N, M=M))
    code_spec = spec_sdfg.generate_code()

    if 'Dynamic' in code_spec[0].code:
        print('ERROR: Constants were not properly specialized')
        exit(2)

    func = dp.compile(spec_sdfg)
    func(A=input, B=output, N=N, M=M)

    diff = np.linalg.norm(
        np.exp(input[1:(N.get() - 1), 0:M.get()]) - output[1:-1, :]) / N.get()
    print("Difference:", diff)
    print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 3)
