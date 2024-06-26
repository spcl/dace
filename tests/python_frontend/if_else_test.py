# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import dace

def test_if():
    @dace.program(use_experimental_cfg_blocks=True)
    def prog(i: int):
        if i < 20:
            return 11
        return 10
    assert prog(i=10) == 11
    assert prog(i=30) == 10

def test_if_else():
    @dace.program(use_experimental_cfg_blocks=True)
    def prog(i: int):
        if i > 0:
            return 30
        else:
            return 40
    assert prog(i=+1) == 30
    assert prog(i=-1) == 40

def test_nested_if_else():
    @dace.program(use_experimental_cfg_blocks=True)
    def prog(i: int):
        if i == 0:
            return 0
        elif i == 1:
            return 10
        elif i == 2:
            return 20
        elif i == 3:
            return 30
    for i in range(4):
        assert prog(i) == 10*i


if __name__ == "__main__":
    test_if()
    test_if_else()
    test_nested_if_else()