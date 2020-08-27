import os

import pytest


def parameterize_gpu(function):
    use_gpu = "ONNX_TEST_CUDA" in os.environ
    if use_gpu:
        return pytest.mark.parametrize("gpu", [True, False])(function)
    else:
        return pytest.mark.parametrize("gpu", [False])(function)
