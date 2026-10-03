# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""User-facing entry point: ``dace.ml.compile``."""
from typing import Any, Callable, Optional

import torch

from .backend import DaceBackend


def compile(model: Optional[Any] = None,
            *,
            dynamic: Optional[bool] = True,
            fullgraph: bool = False,
            sdfg_name: Optional[str] = None,
            simplify: bool = True,
            auto_optimize: bool = False,
            onnx_fallback: bool = True,
            extra_decompositions=None,
            native_ops=None,
            save_sdfg: Optional[str] = None,
            verbose: bool = False,
            **torch_compile_kwargs) -> Callable:
    """
    Compiles a ``torch.nn.Module`` or function with DaCe through TorchDynamo.

    Equivalent to ``torch.compile(model, backend=DaceBackend(...), dynamic=dynamic, fullgraph=fullgraph)``. With
    ``dynamic=True`` (the default) tensor sizes and strides become DaCe symbols, so the generated SDFG is compiled once
    and reused for all shapes that satisfy Dynamo's guards.

    Can be used as a decorator::

        @dace.ml.compile
        def f(x):
            return torch.sin(x) + 1

    :param model: The module or function to compile. If ``None``, returns a decorator.
    :param dynamic: Passed to ``torch.compile``. ``True`` traces with symbolic shapes up front.
    :param fullgraph: Passed to ``torch.compile``. If ``True``, graph breaks raise errors.
    :param sdfg_name: Base name for the generated SDFGs.
    :param simplify: Whether to simplify the generated SDFGs.
    :param auto_optimize: Whether to apply DaCe auto-optimization before compiling.
    :param onnx_fallback: Whether to use ``dace.libraries.onnx`` expansions for operators without a native lowering.
    :param extra_decompositions: Additional ATen operators to decompose before lowering.
    :param native_ops: ATen operators that must not be decomposed.
    :param save_sdfg: If set, directory in which generated SDFGs are saved.
    :param torch_compile_kwargs: Additional keyword arguments for ``torch.compile``.
    :return: The compiled callable. The backend instance is available as ``compiled._dace_backend``.
    """
    options = dict(sdfg_name=sdfg_name,
                   simplify=simplify,
                   auto_optimize=auto_optimize,
                   onnx_fallback=onnx_fallback,
                   extra_decompositions=extra_decompositions,
                   native_ops=native_ops,
                   save_sdfg=save_sdfg,
                   verbose=verbose)

    def _compile(m):
        backend = DaceBackend(**options)
        compiled = torch.compile(m, backend=backend, dynamic=dynamic, fullgraph=fullgraph, **torch_compile_kwargs)
        try:
            compiled._dace_backend = backend
        except AttributeError:  # pragma: no cover
            pass
        return compiled

    if model is None:
        return _compile
    return _compile(model)
