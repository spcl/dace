# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np
import dace
from dace.sdfg.state import UserRegion


def test_user_region_no_name():
    @dace.program
    def func(A: dace.float64[1]):
        with dace.user_region:
            A[0] = 20
        return A
    func.use_experimental_cfg_blocks = True
    sdfg = func.to_sdfg()
    user_region = sdfg.reset_cfg_list()[1]
    assert isinstance(user_region, UserRegion)
    A = np.zeros(shape=(1,))
    assert func(A) == 20

def test_user_region_with_name():
    @dace.program
    def func():
        with dace.user_region("my user region"):
            pass
    func.use_experimental_cfg_blocks = True
    sdfg = func.to_sdfg()
    user_region: UserRegion = sdfg.reset_cfg_list()[1]
    assert user_region.label == "my user region"

def test_nested_user_regions():
    @dace.program
    def func():
        with dace.user_region("outer region"):
            with dace.user_region("middle region"):
                with dace.user_region("inner region"):
                    pass
    func.use_experimental_cfg_blocks = True
    sdfg = func.to_sdfg()
    outer: UserRegion = sdfg.nodes()[1]
    assert outer.label == "outer region"
    middle: UserRegion = outer.nodes()[1]
    assert middle.label == "middle region"
    inner: UserRegion = middle.nodes()[1]
    assert inner.label == "inner region"

if __name__ == "__main__":
    test_user_region_no_name()
    test_user_region_with_name()
    test_nested_user_regions()