# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A forced include among the host flags (``-include FILE``) must reach nvcc as ``--pre-include``.

Wrapped in ``-Xcompiler``, nvcc hands it to both host-compiler runs, the second of which reads the
already preprocessed ``.cudafe1.cpp``: the system headers then arrive twice without their include
guards and the build fails (``conflicting declaration 'typedef struct __fsid_t __fsid_t'``).
"""

import os
import tempfile

import numpy as np
import pytest

import dace
from dace.codegen.targets.cuda import nvcc_host_flags


def test_forced_include_becomes_pre_include():
    flags = nvcc_host_flags(["-O3", "-include", "/opt/h/vecmath.h", "-fopenmp"])
    assert flags == ["-Xcompiler=-O3", "--pre-include=/opt/h/vecmath.h", "-Xcompiler=-fopenmp"]


def test_other_host_flags_stay_wrapped():
    assert nvcc_host_flags(["-march=native", "-ffp-contract=fast"]) == [
        "-Xcompiler=-march=native",
        "-Xcompiler=-ffp-contract=fast",
    ]


@pytest.mark.gpu
def test_gpu_build_with_a_forced_include_header():
    with tempfile.TemporaryDirectory() as folder:
        header = os.path.join(folder, "forced.h")
        with open(header, "w") as handle:
            # Pulls in the libc and libstdc++ headers a second inclusion would redeclare.
            handle.write("#include <math.h>\n#include <stdlib.h>\n#define DACE_FORCED_INCLUDE_SEEN 1\n")

        @dace.program
        def forced_include_tester(a: dace.float64[64], b: dace.float64[64]):
            for i in dace.map[0:64]:
                b[i] = a[i] + 1.0

        args = dace.Config.get("compiler", "cpu", "args") + f" -include {header}"
        with dace.config.set_temporary("compiler", "cpu", "args", value=args):
            with dace.config.set_temporary("default_build_folder", value=os.path.join(folder, "build")):
                sdfg = forced_include_tester.to_sdfg()
                sdfg.apply_gpu_transformations()
                csdfg = sdfg.compile()

        a = np.arange(64, dtype=np.float64)
        b = np.zeros(64, dtype=np.float64)
        csdfg(a=a, b=b)
        assert np.allclose(b, a + 1.0)


if __name__ == "__main__":
    test_forced_include_becomes_pre_include()
    test_other_host_flags_stay_wrapped()
    test_gpu_build_with_a_forced_include_header()
