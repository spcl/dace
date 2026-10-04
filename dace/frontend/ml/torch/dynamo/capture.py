# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Graph capture: the functional ATen graph that TorchDynamo and AOTAutograd produce for a module or function, together
with where every graph input comes from and which assumptions (guards) Dynamo made, without compiling anything.

:func:`capture` is the entry point for consumers that build their own SDFG from the graph (e.g., a ``torch.nn.Module``
used inside a ``@dace.program``). The ``dace`` ``torch.compile`` backend describes its graphs the same way
(:func:`.sources.describe_graph`), so its SDFG arguments are named after the arguments, parameters, and buffers they
come from.
"""
import dataclasses
import inspect
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch._dynamo.exc import BackendCompilerFailed

from dace.sdfg import SDFG

from . import shapes
from .backend import DaceBackend, DaceBackendOptions
from .importer import GraphImporter, ImportResult
from .runtime import CompiledGraph
from .sources import CapturedGuard, GraphDescription, SourceRef, shape_assumptions


@dataclasses.dataclass
class CapturedProgram:
    """The result of :func:`capture`."""
    graph: torch.fx.GraphModule  #: Functional ATen graph (AOTAutograd forward graph, inference)
    example_inputs: List[Any]  #: Fake inputs of ``graph`` in placeholder order
    inputs: List[SourceRef]  #: Source of each placeholder of ``graph``
    symbol_names: Dict[str, str]  #: Dynamo symbol name -> DaCe symbol name
    guards: List[CapturedGuard]  #: Dynamo guards on arguments, module state, and global state
    shape_guards: List[Any]  #: Relations between symbols that Dynamo assumed (DaCe expressions or strings)
    symbol_ranges: Dict[str, Tuple[Any, Any]]  #: DaCe symbol name -> (lower, upper) value range
    dynamo_graph: torch.fx.GraphModule  #: The graph as Dynamo traced it (torch-level operators)
    options: DaceBackendOptions = dataclasses.field(default_factory=DaceBackendOptions)
    owner: Any = None  #: The captured module, if a module was captured
    signature: Optional[inspect.Signature] = None  #: Signature of the captured callable (``forward`` for modules)
    global_vars: Dict[str, Any] = dataclasses.field(default_factory=dict)  #: Globals of the captured callable
    import_result: Optional[ImportResult] = None  #: The most recent result of :meth:`import_graph`

    def input_names(self) -> List[Optional[str]]:
        return GraphDescription(self.inputs, self.symbol_names, self.guards).input_names()

    def compile(self, name: str = 'captured') -> Callable[..., List[Any]]:
        """
        Compiles the graph and returns a function with the signature of the captured callable that returns the flat
        list of graph outputs. No guards are checked: the caller must respect :attr:`guards`.
        """
        compiled = CompiledGraph(self.import_graph(name).sdfg.compile(), self.import_result)

        def call(*args, **kwargs) -> List[Any]:
            with torch.no_grad():
                return compiled(*self.bind(*args, **kwargs))

        return call

    def import_graph(self, name: str, return_arrays: bool = False) -> ImportResult:
        """
        Lowers the graph to an SDFG (simplified and validated according to the capture options).

        :param name: Name of the SDFG.
        :param return_arrays: Name the outputs ``__return``/``__return_<i>``, as for an SDFG called from a
                              ``@dace.program``.
        """
        importer = GraphImporter(self.options)
        self.import_result = importer.import_graph(self.graph,
                                                   self.example_inputs,
                                                   name,
                                                   symbol_names=self.symbol_names,
                                                   input_names=self.input_names(),
                                                   return_arrays=return_arrays)
        return self.import_result

    def to_sdfg(self, name: str = 'captured') -> SDFG:
        return self.import_graph(name).sdfg

    def bind(self, *args, **kwargs) -> List[Any]:
        """
        Evaluates the graph inputs for a call of the captured callable with ``args`` and ``kwargs`` from their sources
        (arguments, current parameter/buffer/attribute values of the module, globals), in placeholder order.

        :return: The positional inputs of :attr:`graph`.
        """
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = []
        for ref, example in zip(self.inputs, self.example_inputs):
            value = ref.evaluate(self.owner, bound.arguments, self.global_vars)
            if isinstance(example, torch.Tensor) and not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value, dtype=example.dtype)  # Python scalars traced as 0-d tensors
            values.append(value)
        return values


class _CaptureComplete(Exception):
    """Raised by the capturing backend once the graph is captured, so that nothing is executed."""

    def __init__(self, program: CapturedProgram):
        super().__init__('graph captured')
        self.program = program


class _CaptureBackend(DaceBackend):
    """A ``DaceBackend`` that stops after AOTAutograd produced the forward graph and returns it instead."""

    def _compile_forward(self, gm: torch.fx.GraphModule, example_inputs: List[Any]) -> Callable:
        description = self.last_description
        inputs = description.inputs
        if len(inputs) != len(example_inputs):
            # AOTAutograd changed the inputs (e.g., deduplicated aliased arguments): sources are not positional
            inputs = [SourceRef('unknown', None, (), node.name) for node in gm.graph.nodes if node.op == 'placeholder']
        relations, ranges = shape_assumptions(description.symbol_names)
        raise _CaptureComplete(
            CapturedProgram(gm, list(example_inputs), inputs, description.symbol_names, description.guards, relations,
                            ranges, self.last_dynamo_graph, self.options))


def capture(fn: Callable,
            *args,
            dynamic: Optional[bool] = True,
            dynamic_shapes: Any = None,
            extra_decompositions: Optional[Sequence] = None,
            native_ops: Optional[Sequence] = None,
            simplify: bool = True,
            specialize_float: bool = False,
            **kwargs) -> CapturedProgram:
    """
    Captures the functional ATen graph of ``fn(*args, **kwargs)`` (a ``torch.nn.Module`` or function) as TorchDynamo
    and AOTAutograd produce it with the DaCe decomposition table, without compiling or executing it.

    The whole call must be captured into one graph: graph breaks raise Dynamo's error, as with ``fullgraph=True``.
    Gradients are not recorded (inference graph).

    :param fn: The module or function.
    :param args: Example arguments. Only their metadata (shapes, strides, dtypes, devices) is used.
    :param dynamic: Passed to ``torch.compile``; ``True`` makes sizes symbolic.
    :param dynamic_shapes: ``'all'`` or a per-argument specification of symbolic dimensions and their names (see
                           :func:`dace.frontend.ml.torch.dynamo.compile`).
    :param extra_decompositions: Additional ATen operators to decompose.
    :param native_ops: ATen operators that must not be decomposed.
    :param simplify: Whether SDFGs built from the capture are simplified.
    :param specialize_float: Treat Python floats (arguments, module attributes) as constants guarded by value, instead
                             of tracing them as 0-d tensor inputs.
    :param kwargs: Example keyword arguments.
    :return: The captured program.
    """
    target = fn.forward if isinstance(fn, torch.nn.Module) else fn
    signature = inspect.signature(target)
    spec = shapes.normalize(dynamic_shapes, signature)
    if spec is not None and dynamic is False:
        raise ValueError('dynamic_shapes requires dynamic=True (or None)')
    backend = _CaptureBackend(dynamic_shapes=spec,
                              signature=signature,
                              extra_decompositions=extra_decompositions,
                              native_ops=native_ops,
                              simplify=simplify)
    compiled = torch.compile(fn, backend=backend, dynamic=dynamic, fullgraph=True)
    if spec is not None:
        shapes.mark_arguments(signature.bind(*args, **kwargs), spec)
    try:
        with torch.no_grad(), torch._dynamo.config.patch(specialize_float=specialize_float):
            compiled(*args, **kwargs)
    except BackendCompilerFailed as ex:
        if isinstance(ex.inner_exception, _CaptureComplete):
            program = ex.inner_exception.program
            program.owner = fn if isinstance(fn, torch.nn.Module) else None
            program.signature = signature
            program.global_vars = target.__globals__
            return program
        raise
    raise RuntimeError('TorchDynamo did not compile the callable (it may have been skipped); nothing was captured')
