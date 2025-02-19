# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

import dace


def test_is_start_state_deprecation():
    sdfg = dace.SDFG('deprecation_test')
    with pytest.deprecated_call():
        sdfg.add_state('state1', is_start_state=True)
    sdfg2 = dace.SDFG('deprecation_test2')
    state = dace.SDFGState('state2')
    with pytest.deprecated_call():
        sdfg2.add_node(state, is_start_state=True)


if __name__ == '__main__':
    test_is_start_state_deprecation()
