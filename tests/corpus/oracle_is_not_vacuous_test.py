# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Every corpus kernel's numpy oracle must actually compute something.

A gate entry whose oracle leaves its inputs untouched certifies nothing: the comparison is then
``inputs == inputs``, which an SDFG that computes NOTHING also satisfies. That is not hypothetical
-- it is how a tile constant larger than the array extent (``range(1, LEN_2D - 1 - 64, 64)`` with
``LEN_2D = 32`` is empty) and a threshold no sample ever clears (``standard_normal > 3``) sat in
the suite looking green.

Catching it needs no compilation: run the oracle on a copy and require that it changed at least one
compared array. Cheap enough to run over the whole corpus.
"""
import numpy as np
import pytest

import tests.corpus.measure_parallelization as mp
from tests.corpus.tsvc_2_5 import tsvc_2_5 as _T25


def _changed(arrays, ref) -> bool:
    """Did the oracle move any compared array off its input value?"""
    return any(not np.array_equal(np.asarray(arrays[n]), np.asarray(ref[n])) for n in arrays)


@pytest.mark.parametrize('name', mp.CORPORA['tsvc'][0]())
def test_tsvc_oracle_writes_something(name):
    arrays, _, ref = mp.tsvc_reference(name)
    assert _changed(arrays, ref), f'{name}: the numpy oracle left every array at its input value'


@pytest.mark.parametrize('name', mp.CORPORA['tsvc25'][0]())
def test_tsvc25_oracle_writes_something(name):
    program = [p for p in _T25.collect() if p.name == name][0]
    arrays, _, ref = mp.tsvc25_reference(program)
    assert _changed(arrays, ref), f'{name}: the numpy oracle left every array at its input value'


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
