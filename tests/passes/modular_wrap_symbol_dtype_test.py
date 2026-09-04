# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A wrapping-modulo loop must parallelize whichever width its size symbol is declared at.

A symbol's dtype is part of its identity, and the two halves of the band proof that accepts a
modular split reach it in different flavours: the modulus rides a memlet subset (stored symbolic,
declared width) while the loop bound is recovered from a string-backed property (re-parsed, so
``DEFAULT_SYMBOL_TYPE``). Same name, two symbols, and ``N - N`` does not cancel -- so
``ext_modular_wrap`` found its split point at ``x = N - 1``, failed to prove the near half in band
0, kept its sequential loop, and ran 250x slower than the un-canonicalized form.
"""
import pytest

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import pipeline as canon


def structure(sdfg: dace.SDFG) -> tuple:
    maps = sum(1 for g in sdfg.all_sdfgs_recursive() for st in g.all_states() for n in st.nodes()
               if isinstance(n, nodes.MapEntry))
    loops = sum(1 for g in sdfg.all_sdfgs_recursive() for b in g.all_control_flow_regions(recursive=True)
                if isinstance(b, LoopRegion))
    return maps, loops


@pytest.mark.parametrize('dtype', [dace.int32, dace.int64])
def test_modular_wrap_parallelizes_at_either_symbol_width(dtype):
    """``a[(i + 1) % N] = b[i]`` splits at ``i = N - 1`` and both halves become one Map."""
    N = dace.symbol('N', dtype=dtype)
    K = 1

    @dace.program
    def modular_wrap(a: dace.float64[N], b: dace.float64[N]):
        for i in range(N):
            a[(i + K) % N] = b[i]

    sdfg = modular_wrap.to_sdfg(simplify=False)
    canon.canonicalize(sdfg)
    maps, loops = structure(sdfg)
    assert maps >= 1, f'the wrap never became a Map at {dtype} (maps={maps}, loops={loops})'
    assert loops == 0, f'a sequential loop survived at {dtype} (maps={maps}, loops={loops})'


def test_the_two_widths_agree():
    """The declared width must not change WHAT canonicalization produces, only how it is typed."""
    shapes = []
    for dtype in (dace.int32, dace.int64):
        N = dace.symbol('N', dtype=dtype)

        @dace.program
        def modular_wrap(a: dace.float64[N], b: dace.float64[N]):
            for i in range(N):
                a[(i + 1) % N] = b[i]

        sdfg = modular_wrap.to_sdfg(simplify=False)
        canon.canonicalize(sdfg)
        shapes.append(structure(sdfg))
    assert shapes[0] == shapes[1], f'int32 gave {shapes[0]}, int64 gave {shapes[1]}'


if __name__ == '__main__':
    test_modular_wrap_parallelizes_at_either_symbol_width(dace.int64)
    test_the_two_widths_agree()
