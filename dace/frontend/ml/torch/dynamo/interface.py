# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""User-facing entry points: ``dace.ml.compile`` and ``dace.ml.to_sdfg``."""

import functools
import inspect
from collections.abc import Callable
from typing import Any

import torch

from . import shapes
from .backend import CAPTURE_CONFIG, DaceBackend
from .capture import without_conv_backend_guards
from .export import compile as export_compile


def compile(
    model: Any | None = None,
    *descriptors,
    dynamic: bool | None = True,
    dynamic_shapes: Any = None,
    fullgraph: bool = False,
    sdfg_name: str | None = None,
    name: str | None = None,
    simplify: bool = True,
    auto_optimize: bool = False,
    fuse_activations: bool = False,
    onnx_fallback: bool = True,
    extra_decompositions=None,
    native_ops=None,
    save_sdfg: str | None = None,
    random_seed: int | None = None,
    verbose: bool = False,
    **torch_compile_kwargs,
) -> Callable:
    """
    Compiles a ``torch.nn.Module`` or function with DaCe through TorchDynamo.

    Two routes, selected by whether descriptor arguments follow the model:

    * **Lazy (``torch.compile``):** ``dace.ml.compile(model)`` with no positional arguments returns a drop-in callable
      that captures and compiles the SDFG on its first call (and reuses it for every shape that satisfies Dynamo's
      guards). This is equivalent to ``torch.compile(model, backend=DaceBackend(...), dynamic=dynamic,
      fullgraph=fullgraph)``. The backend of a compiled object is exposed as ``compiled._dace_backend`` (its
      ``last_sdfg``/``compile_count``); :func:`dace.frontend.ml.torch.dynamo.sdfg_of` returns its most recent SDFG.

    * **Ahead of time:** ``dace.ml.compile(model, dace.float32[N, 3, 224, 224])`` with DaCe data descriptors (see
      :func:`to_sdfg`) compiles the inference SDFG immediately and returns a :class:`.export.CompiledModule` that runs
      on torch tensors (binding the module's current parameters at call time) and exposes ``.sdfg``,
      ``.compiled_sdfg``, and ``.save(path)``.

    With ``dynamic=True`` (the default, lazy route) tensor sizes and strides become DaCe symbols, so the generated SDFG
    is compiled once and reused for all shapes that satisfy Dynamo's guards.

    By default Dynamo decides which sizes share a symbol (sizes that are equal in the first call do) and specializes
    sizes 0 and 1. ``dynamic_shapes`` overrides this::

        # Every dimension of every argument is its own symbol (named x_dim0, x_dim1, ...); weights stay static
        dace.ml.compile(model, dynamic_shapes='all')

        # Named dimensions, in the vocabulary of torch.export (strings or torch.export.Dim objects)
        batch = torch.export.Dim('batch', min=2)
        dace.ml.compile(model, dynamic_shapes={'x': {0: batch, 1: 'seq'}, 'cache': {0: batch, 1: 'cache_len'}})

    See :mod:`dace.frontend.ml.torch.dynamo.shapes` for the full specification format.

    Can be used as a decorator (lazy route only)::

        @dace.ml.compile
        def f(x):
            return torch.sin(x) + 1

    :param model: The module or function to compile. If ``None``, returns a decorator.
    :param descriptors: Ahead-of-time: DaCe data descriptors / symbols / ints for the model's arguments.
    :param dynamic: Passed to ``torch.compile``. ``True`` traces with symbolic shapes up front.
    :param dynamic_shapes: ``'all'`` or a per-argument specification of symbolic dimensions and their names.
    :param fullgraph: Passed to ``torch.compile``. If ``True``, graph breaks raise errors.
    :param sdfg_name: Base name for the generated SDFGs (lazy route).
    :param name: Name of the ahead-of-time SDFG (ahead-of-time route).
    :param simplify: Whether to simplify the generated SDFGs.
    :param auto_optimize: Whether to apply DaCe auto-optimization before compiling (lazy route).
    :param fuse_activations: Whether to fold ReLU maps into the convolution nodes feeding them, so that cuDNN/MIOpen
                             run one fused kernel for convolution, bias and activation (off by default).
    :param onnx_fallback: Whether to use ``dace.libraries.onnx`` expansions for operators without a native lowering.
    :param extra_decompositions: Additional ATen operators to decompose before lowering.
    :param native_ops: ATen operators that must not be decomposed.
    :param save_sdfg: If set, directory in which generated SDFGs are saved (lazy route).
    :param random_seed: Fixes the seed of random operators (dropout) in the exported SDFG, which otherwise draw a fresh
                        seed from torch's generator on every call (ahead-of-time route).
    :param torch_compile_kwargs: Additional keyword arguments for ``torch.compile`` (lazy route).
    :return: The compiled callable, or a :class:`.export.CompiledModule` on the ahead-of-time route.
    """
    if descriptors:
        if model is None:
            raise TypeError("dace.ml.compile needs a model when descriptor arguments are given")
        return export_compile(
            model,
            *descriptors,
            dynamic_shapes=dynamic_shapes,
            simplify=simplify,
            name=name or sdfg_name,
            random_seed=random_seed,
            fuse_activations=fuse_activations,
        )

    options = dict(
        sdfg_name=sdfg_name,
        simplify=simplify,
        auto_optimize=auto_optimize,
        fuse_activations=fuse_activations,
        onnx_fallback=onnx_fallback,
        extra_decompositions=extra_decompositions,
        native_ops=native_ops,
        save_sdfg=save_sdfg,
        verbose=verbose,
    )

    def _compile(m):
        target = m.forward if isinstance(m, torch.nn.Module) else m
        signature = inspect.signature(target)
        spec = shapes.normalize(dynamic_shapes, signature)
        if spec is not None and dynamic is False:
            raise ValueError("dynamic_shapes requires dynamic=True (or None)")
        backend = DaceBackend(dynamic_shapes=spec, signature=signature, **options)
        compiled = torch.compile(m, backend=backend, dynamic=dynamic, fullgraph=fullgraph, **torch_compile_kwargs)
        compiled = shapes.install_marking(_with_capture_config(compiled), m, spec)
        try:
            compiled._dace_backend = backend
        except AttributeError:  # pragma: no cover
            pass
        return compiled

    if model is None:
        return _compile
    return _compile(model)


def _with_capture_config(compiled: Callable) -> Callable:
    """
    Traces every call of ``compiled`` with :data:`~.backend.CAPTURE_CONFIG` (modules stay modules) and without the
    shape guards of ATen's convolution backend selection (:func:`~.capture.without_conv_backend_guards`).
    """
    call = compiled.forward if isinstance(compiled, torch.nn.Module) else compiled

    @functools.wraps(call)
    def configured(*args, **kwargs):
        with torch._dynamo.config.patch(**CAPTURE_CONFIG), without_conv_backend_guards():
            return call(*args, **kwargs)

    if isinstance(compiled, torch.nn.Module):
        compiled.forward = configured
        return compiled
    configured.compiled = compiled
    return configured
