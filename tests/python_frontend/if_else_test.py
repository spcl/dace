# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import dace

def test_if():
    @dace.program(use_experimental_cfg_blocks=False)
    def prog(i: int):
        if i < 20:
            return 11
        return 10
    with dace.config.set_temporary('optimizer', 'automatic_simplification', value=False):
        prog.to_sdfg().save("test_if.sdfg")
        assert prog(i=10) == 11
        assert prog(i=30) == 10

def test_if_else():
    @dace.program(use_experimental_cfg_blocks=True)
    def prog(i: int):
        if i > 0:
            i = 30
        else:
            i = 40
        return i
    with dace.config.set_temporary('optimizer', 'automatic_simplification', value=True):
        sdfg = prog.to_sdfg()
        sdfg.save("if_else_test.sdfg")
        assert prog(i=+1) == 30
        assert prog(i=-1) == 40

if __name__ == "__main__":
    # test_if()
    test_if_else()
