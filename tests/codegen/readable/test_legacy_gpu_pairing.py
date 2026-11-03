# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The readable CPU code generator driven by the legacy CUDA/HIP generator, after GPU canonicalization.

Code generation only: no GPU is needed to see the generated device unit.
"""

import collections
import re

import dace
from dace.transformation.passes.canonicalize.finalize import finalize_for_target, offload_to_gpu
from dace.transformation.passes.canonicalize.pipeline import canonicalize

LEN_1D = dace.symbol("LEN_1D", dtype=dace.int64, positive=True)


@dace.program
def s255(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    x = b[LEN_1D - 1]
    y = b[LEN_1D - 2]
    for i in range(LEN_1D):
        a[i] = (b[i] + x + y) * 0.333
        y = x + 0.0
        x = b[i]


@dace.program
def s252(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    t = 0.0
    for i in range(LEN_1D):
        s = b[i] * c[i]
        a[i] = s + t
        t = s + 0.0


def device_code(program) -> str:
    """The device unit of ``program`` after GPU canonicalization, readable CPU plus legacy HIP generators."""
    with (
        dace.config.set_temporary("compiler", "cpu", "implementation", value="experimental_readable"),
        dace.config.set_temporary("compiler", "cuda", "implementation", value="legacy"),
        dace.config.set_temporary("compiler", "cuda", "backend", value="hip"),
        dace.config.set_temporary("compiler", "emit_tree_reductions", value=False),
    ):
        sdfg = program.to_sdfg()
        canonicalize(sdfg, target="gpu", validate_all=False)
        offload_to_gpu(sdfg)
        finalize_for_target(sdfg, target="gpu")
        code_objects = sdfg.generate_code()
    return next(obj.clean_code for obj in code_objects if obj.target.target_name == "cuda")


def test_device_unit_defines_each_index_helper_once():
    """An allocation the legacy generator dispatches without delegating writes into the same device unit."""
    helpers = collections.Counter(re.findall(r"constexpr \w+ (\w+_idx)\(", device_code(s255)))
    assert helpers, "no index helper in the device unit"
    assert all(count == 1 for count in helpers.values()), helpers


def test_a_kernel_scalar_the_frame_does_not_place_is_no_kernel_argument():
    """``t`` becomes a constant the kernel declares itself, so the legacy generator finds no allocation for it."""
    assert "constexpr double t" in device_code(s252)


if __name__ == "__main__":
    test_device_unit_defines_each_index_helper_once()
    test_a_kernel_scalar_the_frame_does_not_place_is_no_kernel_argument()
