# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""An invalid return value is reported as an ``InvalidSDFGError`` naming the problem, not as a ``TypeError``."""
import pytest

import dace
from dace.sdfg.validation import InvalidSDFGError


def _sdfg_returning(name: str, add) -> dace.SDFG:
    sdfg = dace.SDFG(f'return_{name.strip("_")}')
    add(sdfg)
    sdfg.add_state()
    return sdfg


@pytest.mark.parametrize('add,message', [
    (lambda sdfg: sdfg.add_array('__return_1', [2], dace.float64), 'not consecutively named'),
    (lambda sdfg: sdfg.add_array('__return', [2], dace.float64, transient=True), 'can not be a transient'),
    (lambda sdfg: sdfg.add_scalar('__return', dace.float64), 'scalars can not be returned'),
    (lambda sdfg: sdfg.add_stream('__return', dace.float64), 'Only arrays can be returned'),
])
def test_an_invalid_return_value_raises_a_validation_error(add, message):
    sdfg = _sdfg_returning('__return', add)
    with pytest.raises(InvalidSDFGError, match=message):
        sdfg.validate()


if __name__ == '__main__':
    test_an_invalid_return_value_raises_a_validation_error(lambda sdfg: sdfg.add_array('__return_1', [2], dace.float64),
                                                           'not consecutively named')
    test_an_invalid_return_value_raises_a_validation_error(
        lambda sdfg: sdfg.add_array('__return', [2], dace.float64, transient=True), 'can not be a transient')
    test_an_invalid_return_value_raises_a_validation_error(lambda sdfg: sdfg.add_scalar('__return', dace.float64),
                                                           'scalars can not be returned')
    test_an_invalid_return_value_raises_a_validation_error(lambda sdfg: sdfg.add_stream('__return', dace.float64),
                                                           'Only arrays can be returned')
