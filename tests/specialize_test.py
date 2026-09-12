# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import re

import numpy as np

import dace as dp
from dace.sdfg import SDFG
from dace.memlet import Memlet


def test_constant_specialization():
    N = dp.symbol('N', dtype=dp.int64)
    M = dp.symbol('M', dtype=dp.int64)
    n = 20
    m = 30
    fullrange = '1:N-1,0:M'
    irange = '1:N-1'
    jrange = '0:M'

    input = np.random.rand(n, m).astype(np.float32)
    output = dp.ndarray([n, m], dtype=dp.float32)
    output[:] = dp.float32(0)

    ##########################################################################
    spec_sdfg = SDFG('spectest')
    spec_sdfg.add_array('A', [N, M], dp.float32)
    spec_sdfg.add_transient('At', [N - 2, M], dp.float32)
    spec_sdfg.add_array('B', [N, M], dp.float32)

    state = spec_sdfg.add_state()
    A = state.add_access('A')
    Atrans = state.add_access('At')
    B = state.add_access('B')

    state.add_edge(A, None, Atrans, None, Memlet.simple(A, fullrange))
    _, me, mx = state.add_mapped_tasklet('compute', dict(i=irange, j=jrange), dict(a=Memlet.simple(Atrans, 'i-1,j')),
                                         'b = math.exp(a)', dict(b=Memlet.simple(B, 'i,j')))
    state.add_edge(Atrans, None, me, None, Memlet.simple(Atrans, fullrange))
    state.add_edge(mx, None, B, None, Memlet.simple(B, fullrange))

    spec_sdfg.fill_scope_connectors()
    dp.propagate_memlets_sdfg(spec_sdfg)
    spec_sdfg.validate()
    ##########################################################################

    # Specialization is visible in the entry signature: N and M are runtime parameters before it
    # and compile-time constants after. Asserting that instead of a copy-template spelling keeps
    # the test valid for both CPU generators -- only the legacy one emits ``CopyND ... Dynamic``.
    entry = re.compile(r'void __program_spectest_internal\([^)]*\)')

    # Was ``'Dynamic' in code``: the implicit copy used to lower to the ``dace::CopyNDDynamic``
    # runtime template, whose name doubled as the "shape is still symbolic" marker. Explicit copy
    # nodes lower it to a mapped tasklet, so assert specialization itself.
    code_nonspec = spec_sdfg.generate_code()
    sig_nonspec = entry.search(code_nonspec[0].code)
    assert sig_nonspec is not None
    assert re.search(r'\bN\b', sig_nonspec.group(0)) and re.search(r'\bM\b', sig_nonspec.group(0))
    assert 'constexpr int64_t N' not in code_nonspec[0].code
    assert 'constexpr int64_t M' not in code_nonspec[0].code

    spec_sdfg.specialize(dict(N=n, M=m))
    code_spec = spec_sdfg.generate_code()

    sig_spec = entry.search(code_spec[0].code)
    assert sig_spec is not None
    assert not re.search(r'\bN\b', sig_spec.group(0)) and not re.search(r'\bM\b', sig_spec.group(0))
    assert re.search(r'constexpr\s+\w+\s+N\s*=\s*%d' % n, code_spec[0].code)
    assert re.search(r'constexpr\s+\w+\s+M\s*=\s*%d' % m, code_spec[0].code)

    func = spec_sdfg.compile()
    func(A=input, B=output, N=n, M=m)

    diff = np.linalg.norm(np.exp(input[1:(n - 1), 0:m]) - output[1:-1, :]) / n
    assert diff <= 1e-5


if __name__ == "__main__":
    test_constant_specialization()
