# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import dace
import warnings
from matrix_product_transpose_test import *
from dace.codegen.compiler import CompilerConfigurationError, CompilationError

# This tests that we can generate library nodes from the numpy frontend, and
# choose the global BLAS backend using the DaCe environment variable.


def run_test(implementation):

    print("Testing implementation {}...".format(implementation))
    dace.Config.set("library",
                    "blas",
                    "default_implementation",
                    value=implementation)
    dace.Config.set("library", "blas", "override", value=True)

    A = np.random.rand(K, M).astype(np.float32)
    B = np.random.rand(N, K).astype(np.float32)
    C = np.zeros([M, N], dtype=np.float32)

    matrix_product_transpose_test(A, B, C)

    realC = np.transpose(A) @ np.transpose(B)
    rel_error = np.linalg.norm(C - realC) / np.linalg.norm(realC)
    print('Relative_error:', rel_error)
    if rel_error >= 1e-5:
        print("Test failed.")
        exit(1)


if __name__ == '__main__':

    for implementation in ["MKL", "pure"]:
        try:
            run_test(implementation)
        except (CompilerConfigurationError, CompilationError):
            warnings.warn(
                "Configuration/compilation failed, library missing or "
                "misconfigured, skipping test for {}.".format(implementation))
