# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The CUDA Tile C++ dialect spells what the Tile C++ API takes; these need no GPU."""

import pytest

import dace
from dace.codegen import cppunparse
from dace.codegen.exceptions import CodegenError
from dace.codegen.python.dialects.cuda_tile_cpp import CudaTileCpp, check_tile_toolchain, cpp


@pytest.mark.parametrize("architecture", ["80", "89", "100", "120"])
def test_an_ampere_or_newer_target_builds_tile_code(architecture: str):
    with dace.config.set_temporary("compiler", "cuda", "cuda_arch", value=architecture):
        check_tile_toolchain(True)


def test_a_pre_ampere_target_is_refused():
    with (
        dace.config.set_temporary("compiler", "cuda", "cuda_arch", value="75"),
        pytest.raises(CodegenError, match="sm_80 or newer"),
    ):
        check_tile_toolchain(True)


def test_a_full_reduction_of_a_two_dim_tile_folds_both_dims():
    assert cpp(CudaTileCpp().reduce("+", "x", None, 2)) == "ct::sum<0>(ct::sum<1>(x))"


@pytest.mark.parametrize("op", ["min", "max"])
def test_a_masked_lane_of_a_min_or_max_reads_a_cpp_infinity(op: str):
    dialect = CudaTileCpp()

    text = cpp(cppunparse.pyexpr2cpp(dialect.select("m", "x", dialect.identity(op))))

    assert "INFINITY" in text and "float(" not in text
