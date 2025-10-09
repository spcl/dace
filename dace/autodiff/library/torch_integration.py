"""
Hooks for PyTorch tensors to make them compatible with dace
"""

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
        # initialize with no gradient buffer
        new_desc = ParameterArray(desc.dtype,
                                  desc.shape,
                                  storage=desc.storage,
                                  location=desc.location,
                                  allow_conflicts=desc.allow_conflicts,
                                  transient=desc.transient,
                                  strides=desc.strides,
                                  offset=desc.offset,
                                  lifetime=desc.lifetime,
                                  alignment=desc.alignment,
                                  debuginfo=desc.debuginfo,
                                  total_size=desc.total_size)
        return new_desc

    # register with pytorch
    torch.Tensor.__descriptor__ = create_descriptor_tensor
