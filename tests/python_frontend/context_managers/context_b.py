# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


def dace_blocker(f):
    return f


class MyContextManager:

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
    
    @dace_blocker
    def __enter__(self):
        a = self.rng.integers(51, 100)
        b = self.rng.integers(51, 100)
        print(f'Computing GCD of {a} and {b}')
        return np.gcd(a, b)
    
    @dace_blocker
    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        pass


ctx = MyContextManager(42)


@dace.program
def my_dace_ctxmgr_program():
    with ctx as c:
        print(c)
