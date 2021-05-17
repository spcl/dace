# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

# Tests for PE detections, both among and within connected components

import dace
import numpy as np
import pytest
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG, GPUTransformSDFG, NestSDFG


def test_PEs_inside_component():
    '''
    Tests for PEs detection inside a single Component.
    It computes z =(x+y) + (v+w)

    High-level overview:
     ┌───────────┐        ┌───────────┐
     │ Add_Map_0 │        │ Add_Map_1 │
     └──────┬────┘        └──────┬────┘
            │   ┌───────────┐    │
            └─► │ Add_Map_2 │◄───┘
                └───────────┘
    Map_0 and Map_1 should belong to two distinct PEs
    :return:
    '''
    @dace.program
    def PEs_inside_component(x: dace.float32[8], y: dace.float32[8],
                             v: dace.float32[8], w: dace.float32[8]):
        tmp1 = x + y
        tmp2 = v + w
        return tmp1 + tmp2

    x = np.random.rand(8).astype(np.float32)
    y = np.random.rand(8).astype(np.float32)
    v = np.random.rand(8).astype(np.float32)
    w = np.random.rand(8).astype(np.float32)

    sdfg = PEs_inside_component.to_sdfg()
    sdfg.apply_transformations([FPGATransformSDFG, InlineSDFG])
    sdfg.save('/tmp/out.sdfg')
    program = sdfg.compile()
    for node, state in program.sdfg.all_nodes_recursive():
        if hasattr(node, '_pe'):
            print(node, node._pe)

    z = program(x=x, y=y, v=v, w=w)
    assert np.allclose(z, x + y + v + w)


if __name__ == "__main__":
    test_PEs_inside_component()
