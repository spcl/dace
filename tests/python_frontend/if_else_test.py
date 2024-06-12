# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import dace

def test_if():
    @dace.program(use_experimental_cfg_blocks=True)
    def prog():
        i = 10
        if i < 20:
            i += 1
        return i
    
    assert prog() == 11

def test_if_else():
    @dace.program(use_experimental_cfg_blocks=False)
    def prog(i: int):
        if i > 0:
            return 30
        else:
            return 40
    with dace.config.set_temporary('optimizer', 'automatic_simplification', value=True):
        sdfg = prog.to_sdfg()
        sdfg.save("if_else_test.sdfg")
        assert prog(i=+1) == 30
        assert prog(i=-1) == 40

if __name__ == "__main__":
    # test_if()
    test_if_else()
