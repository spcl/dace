# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import pytest
import copy
import dace
from dace.transformation.passes.offset_loop_and_maps import OffsetLoopsAndMaps


# klev, kidia, kfdia : Symbols
# z1, z2. rmin: scalar
# za, zli, zliqfrac, zlicefrac : Array[nlev, kfdia]
# zqx: Array[klev, kfidia, 5]

klev = dace.symbol("klev")
kidia = dace.symbol("kidia")
kfdia = dace.symbol("kfdia")

@dace.program
def _cloudsc_snippet_one(
    za : dace.float64[klev, kfdia],
    zliqfrac : dace.float64[klev, kfdia],
    zicefrac : dace.float64[klev, kfdia],
    zqx : dace.float64[klev, kfdia, 5],
    zli : dace.float64[klev, kfdia],
    zy: dace.float64[klev, kfdia, 5],
    zx: dace.float64[klev, kfdia, 4],
    rlmin: dace.float64,
    z1: dace.int64,
    z2: dace.int64
):
    for i in range(1, klev + 1):
        for j in range(kidia + 1, kfdia + 1):
            za[i-1, j-1] = 2.0 * za[i-1, j-1] - 5
            cond1 = rlmin > (0.5 * (zqx[i-1, j-1, z1] + zqx[i, j, z2]))
            if cond1:
                zliqfrac[i-1, j-1] = zqx[i-1, j-1, z1] * zli[i-1, j-1]
                zicefrac[i-1, j-1] = 1 - zliqfrac[i-1, j-1]
            else:
                zliqfrac[i-1, j-1] = 0
                zicefrac[i-1, j-1] = 0
            for m in dace.map[1:5:1]:
                zx[i - 1, j - 1, m - 1] = zy[i - 1, z1, z2]


def test_loop_offsetting():
    sdfg = _cloudsc_snippet_one.to_sdfg()
    sdfg.validate()
    sdfg.compile()
    sdfg.simplify()
    sdfg.validate()

    copy_sdfg = copy.deepcopy(sdfg)
    OffsetLoopsAndMaps(begin_expr=None, offset_expr=-1).apply_pass(copy_sdfg, {})
    copy_sdfg.validate()
    copy_sdfg.compile()

def test_begin_expr_condition():
    pass

def test_with_conditional():
    pass

def test_with_symbolic_offset_expr():
    pass

def test_with_symbolic_begin_expr():
    pass

if __name__ == "__main__":
    test_loop_offsetting()
    test_begin_expr_condition()
    test_with_conditional()
    test_with_symbolic_begin_expr()
    test_with_symbolic_offset_expr()
