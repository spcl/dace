import pytest
import torch
import torch.nn as nn

import dace
from dace.autodiff import AutoDiffException


def test_fail_non_float():

    with pytest.raises(AutoDiffException) as info:

        @dace.module(backward=True,
                     dummy_inputs=(torch.ones(10, dtype=torch.long), ))
        class MyModule(nn.Module):
            def forward(self, x):
                return x + 1

        MyModule()

    assert "float edges" in str(info.value)
