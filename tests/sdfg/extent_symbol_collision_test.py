# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""An array added through the SDFG API with an extent spelled like a SCALAR DESCRIPTOR is refused.

A shape has to be symbolic, so a descriptor whose shape names ``M`` needs a symbol ``M`` -- and a
scalar of that name already owns it. The raise names WHICH descriptor collided and the remedy:
declare the name as a ``dace.symbol``. The Python frontend reads such a scalar into a symbol
instead, see ``tests/size_scalar_shape_promotion_test.py``.
"""
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
