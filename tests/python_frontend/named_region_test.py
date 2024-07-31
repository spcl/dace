# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np
import dace
from dace.sdfg.state import NamedRegion


def test_named_region_no_name():
    @dace.program
    def func(A: dace.float64[1]):
        with dace.named:
            A[0] = 20
        return A
    func.use_experimental_cfg_blocks = True
    sdfg = func.to_sdfg()
    named_region = sdfg.reset_cfg_list()[1]
    assert isinstance(named_region, NamedRegion)
    A = np.zeros(shape=(1,))
    assert func(A) == 20

def test_named_region_with_name():
    @dace.program
    def func():
        with dace.named("my named region"):
            pass
    func.use_experimental_cfg_blocks = True
    sdfg = func.to_sdfg()
    named_region: NamedRegion = sdfg.reset_cfg_list()[1]
    assert named_region.label == "my named region"

def test_nested_named_regions():
    @dace.program
    def func():
        with dace.named("outer region"):
            with dace.named("middle region"):
                with dace.named("inner region"):
                    pass
    func.use_experimental_cfg_blocks = True
    sdfg = func.to_sdfg()
    outer: NamedRegion = sdfg.nodes()[1]
    assert outer.label == "outer region"
    middle: NamedRegion = outer.nodes()[1]
    assert middle.label == "middle region"
    inner: NamedRegion = middle.nodes()[1]
    assert inner.label == "inner region"

if __name__ == "__main__":
    test_named_region_no_name()
    test_named_region_with_name()
    test_nested_named_regions()