# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests that a refused memlet subscript reports the construct it was refused for. """
import pytest

import dace as dc
from dace.frontend.python.common import DaceSyntaxError

R = dc.symbol('R', dtype=dc.int64, positive=True)
C = dc.symbol('C', dtype=dc.int64, positive=True)


def test_a_per_axis_boolean_mask_names_the_mask_not_only_the_rank():
    """A per-axis mask is refused for its position, not for the array's rank, and a message that
    reports dimensionality alone sends the reader to the shapes instead of to the subscript."""

    @dc.program
    def mask_columns(tg: dc.float64[R, C], keep: dc.bool[C]):
        tg[:, ~keep] = 0.0

    with pytest.raises(DaceSyntaxError) as caught:
        mask_columns.to_sdfg(simplify=False)
    message = str(caught.value)
    assert 'Only one boolean array is allowed' in message, message
    assert 'keep' in message, message


def test_a_boolean_mask_of_the_wrong_shape_names_the_shape_mismatch():
    """The two boolean refusals have different causes and different fixes, so one message must not
    stand in for the other."""

    @dc.program
    def mask_rows(tg: dc.float64[R, C], keep: dc.bool[C]):
        tg[~keep] = 0.0

    with pytest.raises(DaceSyntaxError) as caught:
        mask_rows.to_sdfg(simplify=False)
    message = str(caught.value)
    assert 'Shape of boolean index must match' in message, message


if __name__ == '__main__':
    test_a_per_axis_boolean_mask_names_the_mask_not_only_the_rank()
    test_a_boolean_mask_of_the_wrong_shape_names_the_shape_mismatch()
