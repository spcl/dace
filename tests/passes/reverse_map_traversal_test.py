# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A parallel Map must not walk memory backwards.

``normalize_negative_stride`` keeps a reversed loop's iteration ORDER, which a carried recurrence
needs. Once ``LoopToMap`` has made a Map of it the order is free, but the reversal stayed baked in
the access -- an ascending parameter over descending addresses, which defeats the prefetcher.
"""
import numpy as np

import dace
from dace.codegen import codegen
from dace.transformation.passes.canonicalize import pipeline as canon
from dace.transformation.passes.canonicalize.reverse_map_traversal import reverse_descending_maps

N = dace.symbol('N', dtype=dace.int64)


def emitted(sdfg: dace.SDFG) -> str:
    return '\n'.join(o.clean_code for o in codegen.generate_code(sdfg) if o.language == 'cpp')


def test_a_reversed_stream_is_re_expressed_forwards():
    """``for i in range(N-1, -1, -1): a[i] = b[i] + 1`` must emit ``a[p]``, not ``a[N-1-p]``."""

    @dace.program
    def reverse_stream(a: dace.float64[N], b: dace.float64[N]):
        for i in range(N - 1, -1, -1):
            a[i] = b[i] + 1.0

    sdfg = reverse_stream.to_sdfg(simplify=False)
    canon.canonicalize(sdfg)
    code = emitted(sdfg)
    assert 'a_idx(_loop_it_0)' in code or 'a[_loop_it_0]' in code, f'access is still reversed:\n{code}'
    assert 'N - _loop_it_0' not in code, f'a descending access survived:\n{code}'


def test_the_result_is_unchanged():
    """The flip is a bijection of the range: same elements, same values."""

    @dace.program
    def reverse_stream(a: dace.float64[N], b: dace.float64[N]):
        for i in range(N - 1, -1, -1):
            a[i] = b[i] + 1.0

    n = 257
    b = np.arange(n, dtype=np.float64)
    got = np.zeros(n, dtype=np.float64)
    sdfg = reverse_stream.to_sdfg(simplify=False)
    canon.canonicalize(sdfg)
    sdfg(a=got, b=b, N=n)
    assert np.allclose(got, b + 1.0)


def test_a_forward_map_is_left_alone():
    """Nothing to orient: an ascending access must not be flipped into a descending one."""

    @dace.program
    def forward_stream(a: dace.float64[N], b: dace.float64[N]):
        for i in range(N):
            a[i] = b[i] + 1.0

    sdfg = forward_stream.to_sdfg(simplify=False)
    canon.canonicalize(sdfg)
    assert reverse_descending_maps(sdfg) is None, 'an already-ascending map was flipped'


def test_a_mixed_direction_map_is_left_alone():
    """A map that reads backwards and writes forwards has no good direction; leave it."""

    @dace.program
    def mixed(a: dace.float64[N], b: dace.float64[N]):
        for i in dace.map[0:N]:
            a[i] = b[N - 1 - i]

    sdfg = mixed.to_sdfg(simplify=False)
    assert reverse_descending_maps(sdfg) is None, 'a map with both directions was flipped'


if __name__ == '__main__':
    test_a_reversed_stream_is_re_expressed_forwards()
    test_the_result_is_unchanged()
    test_a_forward_map_is_left_alone()
    test_a_mixed_direction_map_is_left_alone()
