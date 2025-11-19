# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Python interface for DaCe PyTorch/Torch integration.

This module provides decorators and utilities for converting PyTorch modules
to DaCe-accelerated implementations.
"""

from functools import wraps
from typing import Optional, Tuple, List

from dace.dtypes import paramdec


@paramdec
def module(moduleclass,
           dummy_inputs: Optional[Tuple] = None,
           cuda: Optional[bool] = None,
           training: bool = False,
           backward=False,
           inputs_to_skip: Optional[List[str]] = None,
           simplify: bool = True,
           auto_optimize: bool = True,
           sdfg_name: Optional[str] = None,
           compile_torch_extension: bool = True,
           debug_transients: bool = False):
    """ Decorator to apply on a definition of a ``torch.nn.Module`` to
        convert it to a data-centric module upon construction.

        :Example:

            >>> import dace
            >>> import torch.nn as nn
            >>> @dace.ml.module
            ... class MyDecoratedModule(nn.Module):
            ...     def forward(self, x):
            ...        x = torch.log(x)
            ...        x = torch.sqrt(x)
            ...        return x
            >>> module_instance = MyDecoratedModule()
            >>> module_instance(torch.ones(2))
            tensor([0., 0.])

        :param moduleclass: the model to wrap.
        :param dummy_inputs: a tuple of tensors to use as input when tracing ``model``.
        :param cuda: if ``True``, the module will execute using CUDA. If ``None``, it will be detected from the
                     ``module``.
        :param training: whether to use train mode when tracing ``model``.
        :param backward: whether to enable the backward pass.
        :param inputs_to_skip: if provided, a list of inputs to skip computing gradients for.
                               (only relevant when the backward pass is enabled)
        :param simplify: whether to apply simplification transforms after conversion (this generally improves performance,
                             but can be slow).
        :param auto_optimize: whether to apply automatic optimizations.
        :param sdfg_name: the name to give to the sdfg (defaults to ``dace_model``).
        :param compile_torch_extension: if True, a torch C++ extension will be compiled and used for this module.
                                        Otherwise, a python ctypes implementation will be used.
        :param debug_transients: if True, the module will have all transients as outputs.
    """
    wraps(moduleclass)

    def _create(*args, **kwargs):
        # Lazy import DaceModule when decorator is actually used
        try:
            from dace.frontend.ml.torch import DaceModule
        except ImportError:
            raise ImportError("DaceModule requires PyTorch. Install with: pip install torch")

        return DaceModule(moduleclass(*args, **kwargs),
                          dummy_inputs=dummy_inputs,
                          cuda=cuda,
                          training=training,
                          backward=backward,
                          inputs_to_skip=inputs_to_skip,
                          simplify=simplify,
                          auto_optimize=auto_optimize,
                          sdfg_name=sdfg_name,
                          compile_torch_extension=compile_torch_extension,
                          debug_transients=debug_transients)

    return _create
