# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace


def test_cache_return_values():
    @dace.program
    def test(x):
        return x * x

    a = test(5)
    b = test(6)

    assert a == 25 and b == 36


if __name__ == '__main__':
    test_cache_return_values()
