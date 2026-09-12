# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for an issue where copy code would be generated multiple times. """
import re

import dace
import numpy as np


def test_multicopy():
    sdfg = dace.SDFG('multicopy')
    sdfg.add_array('A', [1], dace.float64)
    sdfg.add_array('B', [1], dace.float64)
    sdfg.add_array('C', [1], dace.float64)
    state = sdfg.add_state()
    a = state.add_read('A')
    b = state.add_write('B')
    c = state.add_write('C')
    state.add_nedge(a, b, dace.Memlet('A[0]'))
    state.add_nedge(a, c, dace.Memlet('C[0]'))

    # Check generated code: one copy per outgoing edge of A, no duplicates. The regression under
    # test is DUPLICATED copy code, so what matters is the count, not which lowering produced it:
    # the legacy CPU target spells a copy as a dace::CopyND call, the readable one as a direct
    # assignment reading the source (explicit copy nodes lower a single-element copy to a
    # scalar-assignment tasklet, which the readable generator then inlines). Both count as one site.
    code = sdfg.generate_code()[0].clean_code
    assert len(re.findall(r'CopyND|= A\[', code)) == 2

    # Check outputs
    A = np.random.rand(1)
    B = np.random.rand(1)
    C = np.random.rand(1)
    sdfg(A=A, B=B, C=C)
    assert np.allclose(A, B)
    assert np.allclose(A, C)


if __name__ == '__main__':
    test_multicopy()
