# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Ahead-of-time export: turn a ``torch.nn.Module`` or function into an SDFG directly from DaCe data descriptors, without
first calling it on tensors.

:func:`to_sdfg` captures the inference graph for the given descriptors (:func:`.capture.capture`) and lowers it to an
SDFG whose arguments are named after the ``forward`` parameters, the module's parameters and buffers (by their
qualified names), and the user's DaCe symbols. :func:`compile` compiles that SDFG and wraps it in a
:class:`CompiledModule` that runs on torch tensors, binding the module's current parameters at call time (the same
sources :meth:`.capture.CapturedProgram.bind` uses).
"""

import inspect
import warnings
from typing import Any

import torch

from dace.sdfg import SDFG

from .backend import select_implementations
from .capture import CapturedProgram, capture
from .context import RANDOM_SEED_SYMBOL, sanitize_name
from .convertible import example_arguments
from .runtime import CompiledGraph

_SUPPORTED_INPUTS = ("argument", "parameter", "buffer", "attribute", "global", "size", "stride", "storage_offset")


def _warn_if_training(model: Any) -> None:
    """
    Warns when a module is exported in training mode, where stochastic layers are traced for training.

    Ahead-of-time export captures a single graph for the module's current mode. In training mode dropout draws a mask
    and batch-norm uses batch statistics, which is usually not what inference wants. The lazy ``torch.compile`` route
    re-traces on mode changes, so it does not warn.
    """
    if isinstance(model, torch.nn.Module) and model.training:
        warnings.warn(
            "module is in training mode; dropout/batch-norm are traced for training (call .eval() for inference)"
        )


def _default_name(model: Any) -> str:
    """Base SDFG name for a module (its class name) or function (its ``__name__``)."""
    if isinstance(model, torch.nn.Module):
        return sanitize_name(type(model).__name__.lower())
    return sanitize_name(getattr(model, "__name__", "captured"))


def _capture_from_descriptors(
    model: Any, args: tuple, kwargs: dict, dynamic_shapes: Any, simplify: bool
) -> CapturedProgram:
    """
    Captures ``model`` for arguments that are DaCe data descriptors (``dace.float32[N, 3, 224, 224]``), DaCe symbols or
    ints (scalar arguments), or torch tensors (metadata only).

    The descriptors are turned into fake example tensors and a ``dynamic_shapes`` specification that names the resulting
    Dynamo symbols after the DaCe symbols of the descriptors (:func:`.convertible.example_arguments`), so symbolic sizes
    keep their names. An explicit ``dynamic_shapes`` argument overrides that specification.
    """
    target = model.forward if isinstance(model, torch.nn.Module) else model
    signature = inspect.signature(target)
    bound = signature.bind(*args, **kwargs)
    examples, derived_spec = example_arguments(bound.arguments)
    bound.arguments.update(examples)
    spec = dynamic_shapes if dynamic_shapes is not None else derived_spec
    return capture(model, *bound.args, dynamic_shapes=spec, specialize_float=True, simplify=simplify, **bound.kwargs)


def _check_inputs(program: CapturedProgram, model: Any) -> None:
    what = type(model).__name__ if isinstance(model, torch.nn.Module) else getattr(model, "__name__", "function")
    for ref in program.inputs:
        if ref.kind not in _SUPPORTED_INPUTS:
            raise NotImplementedError(
                f"{what}: graph input {ref.text} ({ref.kind}) cannot be provided ahead of time yet"
            )


def to_sdfg(
    model: Any,
    *args,
    dynamic_shapes: Any = None,
    simplify: bool = True,
    name: str | None = None,
    random_seed: int | None = None,
    **kwargs,
) -> SDFG:
    """
    Builds the inference SDFG of a ``torch.nn.Module`` or function from DaCe data descriptors, without example tensors
    and without compiling it.

    Positional and keyword arguments describe the ``forward`` (or function) arguments: DaCe data descriptors such as
    ``dace.float32[N, 3, 224, 224]`` (symbolic sizes allowed, and kept under their names), DaCe symbols or Python ints
    for scalar arguments, or torch tensors (only their shape, stride, dtype, and device are used). The SDFG's arguments
    are the ``forward`` parameter names for the inputs, the qualified names of the module's parameters and buffers
    (e.g. ``features_0_weight``), and the user's DaCe symbols.

    :param model: The module (in the desired mode, e.g. ``.eval()``) or function.
    :param args: Positional argument descriptors.
    :param dynamic_shapes: Overrides the symbolic-size specification derived from the descriptors (see
                           :func:`dace.frontend.ml.torch.dynamo.compile`).
    :param simplify: Whether to simplify the SDFG.
    :param name: Name of the SDFG (defaults to the module's class name or the function's name).
    :param random_seed: Fixes the seed of the random operators (e.g. training-mode dropout) in the SDFG. By default
                        the seed is the symbol ``dace_random_seed``, which the runtimes draw from torch's generator on
                        every call; individual nodes can still be pinned through their ``seed`` property.
    :param kwargs: Keyword argument descriptors.
    :return: The simplified, validated SDFG (not compiled).
    """
    _warn_if_training(model)
    program = _capture_from_descriptors(model, args, kwargs, dynamic_shapes, simplify)
    _check_inputs(program, model)
    sdfg = program.import_graph(name or _default_name(model), return_arrays=True).sdfg
    _specialize_random_seed(sdfg, random_seed)
    select_implementations(sdfg)
    return sdfg


def _specialize_random_seed(sdfg: SDFG, random_seed: int | None) -> None:
    """Replaces the per-call random seed symbol by a constant, if the SDFG has one and a seed was requested."""
    if random_seed is not None and RANDOM_SEED_SYMBOL in sdfg.symbols:
        sdfg.specialize({RANDOM_SEED_SYMBOL: int(random_seed)})


class CompiledModule:
    """
    A compiled inference SDFG of a ``torch.nn.Module`` or function, built ahead of time from DaCe data descriptors by
    :func:`compile`.

    Calling it with torch tensors runs the compiled SDFG and returns torch tensors, binding the module's current
    parameters and buffers at call time (changing them does not require recompiling). The SDFG and the compiled SDFG
    are available for inspection and :meth:`save`.
    """

    def __init__(self, program: CapturedProgram, name: str, random_seed: int | None = None):
        self._program = program
        self._result = program.import_graph(name)
        _specialize_random_seed(self._result.sdfg, random_seed)
        select_implementations(self._result.sdfg)
        self._compiled = self._result.sdfg.compile()
        self._runtime = CompiledGraph(
            self._compiled, self._result.inputs, self._result.outputs, communicators=self._result.communicators
        )

    @property
    def sdfg(self) -> SDFG:
        """The generated SDFG (before compilation)."""
        return self._result.sdfg

    @property
    def compiled_sdfg(self):
        """The compiled SDFG (:class:`~dace.codegen.compiled_sdfg.CompiledSDFG`)."""
        return self._compiled

    def save(self, path: str) -> None:
        """Saves the SDFG to ``path`` (``.sdfg`` or ``.sdfgz`` by extension)."""
        self.sdfg.save(path)

    def __call__(self, *tensors) -> Any:
        with torch.no_grad():
            outputs: list[Any] = self._runtime(*self._program.bind(*tensors))
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)


def compile(
    model: Any,
    *args,
    dynamic_shapes: Any = None,
    simplify: bool = True,
    name: str | None = None,
    random_seed: int | None = None,
    **kwargs,
) -> CompiledModule:
    """
    Compiles the inference SDFG of a ``torch.nn.Module`` or function ahead of time from DaCe data descriptors (see
    :func:`to_sdfg`) and returns a :class:`CompiledModule` that runs on torch tensors.

    This is the ahead-of-time overload of :func:`dace.frontend.ml.torch.dynamo.compile`; it is selected when
    descriptor arguments follow the model. ``random_seed`` fixes the seed of random operators (see :func:`to_sdfg`).
    """
    _warn_if_training(model)
    program = _capture_from_descriptors(model, args, kwargs, dynamic_shapes, simplify)
    _check_inputs(program, model)
    return CompiledModule(program, name or _default_name(model), random_seed)


def sdfg_of(compiled: Any) -> SDFG | None:
    """
    Returns the most recent SDFG of a compiled object.

    * A :class:`CompiledModule` returns its own SDFG.
    * A callable produced by :func:`dace.frontend.ml.torch.dynamo.compile` returns its backend's ``last_sdfg`` (the
      backend is also reachable as ``compiled._dace_backend``).
    * A callable produced by ``torch.compile(model, backend='dace')`` has no private backend handle; its graphs are
      compiled by the shared :data:`~.backend.dace_backend`, whose ``last_sdfg`` is returned.

    :param compiled: A :class:`CompiledModule` or a callable compiled with the ``dace`` backend.
    :return: The SDFG, or ``None`` if no graph has been compiled yet.
    """
    if isinstance(compiled, CompiledModule):
        return compiled.sdfg
    backend = getattr(compiled, "_dace_backend", None)
    if backend is None:
        from .backend import dace_backend  # The instance registered under the name 'dace'

        backend = dace_backend
    return backend.last_sdfg
