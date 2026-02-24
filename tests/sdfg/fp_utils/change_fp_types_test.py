# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import pytest

from dace.sdfg.fp_utils.change_fp_types import change_fptype


@dace.program
def _program(A: dace.float64[10, 10], B: dace.float64[10, 10], C: dace.float64[10, 10]):
    for i, j in dace.map[0:10, 0:10]:
        C[i, j] = A[i, j] + B[i, j]


@dace.program
def _program2(A: dace.float64[10, 10], B: dace.float64[10, 10], C: dace.float64[10, 10], idx: dace.int64[10, 10],
              idy: dace.int64[10, 10]):
    for i, j in dace.map[0:10, 0:10]:
        C[i, j] = A[i, j] + B[idy[i, j], idx[i, j]]


def test_simple():
    sdfg = _program.to_sdfg()
    sdfg.validate()
    change_fptype(sdfg=sdfg,
                  src_fptype=dace.float64,
                  dst_fptype=dace.float32,
                  cast_in_and_out_data=True,
                  arrays_to_replace=None)
    sdfg.validate()
    sdfg.compile()
    sdfg.save("change_fp_types_test.sdfgz", compress=True)


def test_nested():
    sdfg = _program2.to_sdfg()
    sdfg.validate()
    change_fptype(sdfg=sdfg,
                  src_fptype=dace.float64,
                  dst_fptype=dace.float32,
                  cast_in_and_out_data=True,
                  arrays_to_replace=None)
    sdfg.validate()
    sdfg.save("change_fp_types_test_nested.sdfgz", compress=True)
    sdfg.compile()


if __name__ == "__main__":
    test_simple()
    test_nested()
