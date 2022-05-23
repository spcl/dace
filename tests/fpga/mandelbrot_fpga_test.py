# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.fpga_testing import fpga_test, import_sample
from dace.transformation.interstate import FPGATransformSDFG
from pathlib import Path


# TODO: Pipeline control flow while-loop?
@fpga_test(assert_ii_1=False)
def test_mandelbrot_fpga():
    mandelbrot = import_sample(Path("simple") / "mandelbrot.py")
    h, w, max_iterations = 64, 64, 1000
    out = dace.ndarray([h, w], dtype=dace.uint16)
    out[:] = dace.uint32(0)
    sdfg = mandelbrot.mandelbrot.to_sdfg()
    sdfg.apply_transformations(FPGATransformSDFG)
    sdfg(output=out, maxiter=max_iterations, W=w, H=h)
    return sdfg


if __name__ == "__main__":
    test_mandelbrot_fpga(None)
