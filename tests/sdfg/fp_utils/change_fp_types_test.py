# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import pytest

from dace.sdfg.fp_utils.change_fp_types import change_fptype


@dace.program
def _program(A: dace.float64[10, 10], B: dace.float64[10, 10], C: dace.float64[10, 10]):
    for i, j in dace.map[0:10, 0:10]:
        C[i, j] = A[i, j] + B[i, j]


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


if __name__ == "__main__":
    test_simple()
