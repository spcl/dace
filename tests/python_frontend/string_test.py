# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import numpy as np

def callback_inhibitor(f):
    return f

def test_string_literal_in_callback():
    success = False
    @callback_inhibitor
    def cb(a):
        nonlocal success
        if a == 'a':
            success = True


    @dace
    def tester(a):
        cb('a')

    
    a = np.random.rand(1)
    tester(a)

    assert success is True


if __name__ == '__main__':
    test_string_literal_in_callback()
