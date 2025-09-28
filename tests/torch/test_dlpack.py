import ctypes

import pytest

import dace
import numpy as np

from dace.libraries.torch.dlpack import array_to_torch_tensor


@pytest.mark.torch
def test_desc_to_dlpack():
    mydata = np.arange(6).reshape(2, 3).astype(np.float32)

    ptr = ctypes.c_void_p(mydata.__array_interface__["data"][0])
    tensor = array_to_torch_tensor(ptr, dace.float32[2, 3])
    assert np.allclose(tensor, mydata)
    mydata += 1
    assert np.allclose(tensor, mydata)
