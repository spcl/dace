# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Hooks for PyTorch tensors to make them compatible with dace
"""
import copy

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from dace import data

from dace.autodiff.library.library import ParameterArray

if TORCH_AVAILABLE:

    def create_descriptor_tensor(self: torch.Tensor) -> data.Data:
        """
        Creates a descriptor for a tensor.
        If the tensor requires grad, we convert to a ParameterArray
        """

        desc = data.create_datadescriptor(self, no_custom_desc=True)
        if not isinstance(desc, data.Array):
            raise ValueError("Unsupported descriptor: {}".format(desc))

        if not self.requires_grad:
            return desc

        new_desc = copy.deepcopy(desc)
        new_desc.__class__ = ParameterArray
        new_desc.gradient = None
        return new_desc

    # register with pytorch
    torch.Tensor.__descriptor__ = create_descriptor_tensor
