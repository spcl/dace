# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import dace
from dace.sdfg.state import UserRegion


def test_user_region_no_name():
    @dace.program
    def func(A: dace.float32[1]):
        with dace.user_region:
            A[0] = 20
    func.use_experimental_cfg_blocks = True
    sdfg = func.to_sdfg()
    user_region = sdfg.reset_cfg_list()[1]
    assert isinstance(user_region, UserRegion)

if __name__ == "__main__":
    test_user_region_no_name()