# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""An extent spelled with the name of a SCALAR ARGUMENT cannot be resolved.

A shape has to be symbolic at parse time, so a descriptor whose shape names ``M`` needs a symbol
``M`` -- and a scalar argument of that name already owns it. The refusal is correct (a data
descriptor holds a run-time value, and DaCe has no data-dependent shapes), so what the raise owes
the reader is WHICH descriptor collided and what to do instead: declare the name as a
``dace.symbol``. This pins that text, because the bare "the name is used by a data descriptor"
that ``add_symbol`` raises names neither the shape nor the remedy -- npbench's ``lenet`` reported
it for ``C_before_fc1`` with no hint that the fix was in the program's signature.
"""
import numpy as np
import pytest

import dace

N = dace.symbol('N', dtype=dace.int64)


def test_extent_naming_a_descriptor_names_both_sides():
    sdfg = dace.SDFG('extent_symbol_collision')
    sdfg.add_scalar('M', dace.int64)
    with pytest.raises(FileExistsError) as info:
        sdfg.add_array('a', (N, dace.symbol('M', dtype=dace.int64)), dace.float64)
    message = str(info.value)
    assert 'M' in message
    assert '"a"' in message, message
    assert 'dace.symbol' in message, message


def test_reshape_by_scalar_argument_is_refused_with_context():
    """The reported shape: a scalar parameter used as a ``np.reshape`` extent."""

    @dace.program
    def reshape_by_scalar(a: dace.float64[N, 4], M: dace.int64, out: dace.float64[N, 4]):
        b = np.reshape(a, (N, M))
        out[:] = b

    with pytest.raises(FileExistsError) as info:
        reshape_by_scalar.to_sdfg(simplify=False)
    message = str(info.value)
    assert 'M' in message
    assert 'dace.symbol' in message, message


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
