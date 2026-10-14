# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Graph capture: the functional ATen graph that TorchDynamo and AOTAutograd produce for a module or function, together
with where every graph input comes from and which assumptions (guards) Dynamo made, without compiling anything.

:func:`capture` is the entry point for consumers that build their own SDFG from the graph (e.g., a ``torch.nn.Module``
used inside a ``@dace.program``). The ``dace`` ``torch.compile`` backend describes its graphs the same way
(:func:`.sources.describe_graph`), so its SDFG arguments are named after the arguments, parameters, and buffers they
come from.
"""

import contextlib
import dataclasses
import inspect
import re
import traceback
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import torch
from torch._dynamo.exc import BackendCompilerFailed
from torch._guards import TracingContext
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from dace.sdfg import SDFG

from . import shapes
from .backend import CAPTURE_CONFIG, DaceBackend, DaceBackendOptions, select_implementations
from .importer import GraphImporter, ImportResult
from .runtime import CompiledGraph
from .sources import (
    CapturedGuard,
    GraphDescription,
    ShapeAssumption,
    SourceRef,
    shape_assumptions,
    symbol_names_from_sources,
)


class CaptureError(RuntimeError):
    """
    TorchDynamo or AOTAutograd could not trace the callable given to :func:`capture`.

    The message states the failing condition with concrete sizes and the user's symbol names in place of Dynamo's
    symbols (``s53``), followed by the lines of the model's code that led to it. The exception Dynamo raised is
    chained as the cause; its internal stack trace is only kept with ``TORCHDYNAMO_VERBOSE=1``.
    """


@dataclasses.dataclass
class CapturedProgram:
    """The result of :func:`capture`."""

    graph: torch.fx.GraphModule  #: Functional ATen graph (AOTAutograd forward graph, inference)
    example_inputs: list[Any]  #: Fake inputs of ``graph`` in placeholder order
    inputs: list[SourceRef]  #: Source of each placeholder of ``graph``
    symbol_names: dict[str, str]  #: Dynamo symbol name -> DaCe symbol name
    guards: list[CapturedGuard]  #: Dynamo guards on arguments, module state, and global state
    shape_guards: list[Any]  #: Relations between symbols that Dynamo assumed (DaCe expressions or strings)
    symbol_ranges: dict[str, tuple[Any, Any]]  #: DaCe symbol name -> (lower, upper) value range
    dynamo_graph: torch.fx.GraphModule  #: The graph as Dynamo traced it (torch-level operators)
    options: DaceBackendOptions = dataclasses.field(default_factory=DaceBackendOptions)
    owner: Any = None  #: The captured module, if a module was captured
    signature: inspect.Signature | None = None  #: Signature of the captured callable (``forward`` for modules)
    global_vars: dict[str, Any] = dataclasses.field(default_factory=dict)  #: Globals of the captured callable
    import_result: ImportResult | None = None  #: The most recent result of :meth:`import_graph`
    #: :attr:`shape_guards` with the location (model line and torch function) that introduced each of them
    assumptions: list[ShapeAssumption] = dataclasses.field(default_factory=list)

    def input_names(self) -> list[str | None]:
        return GraphDescription(self.inputs, self.symbol_names, self.guards).input_names()

    def compile(self, name: str = "captured") -> Callable[..., list[Any]]:
        """
        Compiles the graph and returns a function with the signature of the captured callable that returns the flat
        list of graph outputs. No guards are checked: the caller must respect :attr:`guards`.
        """
        result = self.import_graph(name)
        select_implementations(result.sdfg)
        compiled = CompiledGraph(
            result.sdfg.compile(), result.inputs, result.outputs, communicators=result.communicators
        )

        def call(*args, **kwargs) -> list[Any]:
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
        self.import_result = importer.import_graph(
            self.graph,
            self.example_inputs,
            name,
            symbol_names=self.symbol_names,
            input_names=self.input_names(),
            return_arrays=return_arrays,
        )
        return self.import_result

    def to_sdfg(self, name: str = "captured") -> SDFG:
        return self.import_graph(name).sdfg

    def bind(self, *args, **kwargs) -> list[Any]:
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
        super().__init__("graph captured")
        self.program = program


class _CaptureBackend(DaceBackend):
    """A ``DaceBackend`` that stops after AOTAutograd produced the forward graph and returns it instead."""

    def __init__(self, **options):
        super().__init__(**options)
        self.shape_env: ShapeEnv | None = None  #: Shape environment of the most recent Dynamo-level graph

    def __call__(self, gm: torch.fx.GraphModule, example_inputs: list[Any]) -> Callable:
        context = TracingContext.try_get()
        if context is not None and context.fake_mode is not None:
            self.shape_env = context.fake_mode.shape_env
        return super().__call__(gm, example_inputs)

    def _compile_forward(
        self, gm: torch.fx.GraphModule, example_inputs: list[Any], description: GraphDescription | None = None
    ) -> Callable:
        description = description or self.last_description
        inputs = description.inputs
        if len(inputs) != len(example_inputs):
            # AOTAutograd changed the inputs (e.g., deduplicated aliased arguments): sources are not positional
            inputs = [SourceRef("unknown", None, (), node.name) for node in gm.graph.nodes if node.op == "placeholder"]
        assumptions, ranges = shape_assumptions(description.symbol_names)
        raise _CaptureComplete(
            CapturedProgram(
                gm,
                list(example_inputs),
                inputs,
                description.symbol_names,
                description.guards,
                [assumption.relation for assumption in assumptions],
                ranges,
                self.last_dynamo_graph,
                self.options,
                assumptions=assumptions,
            )
        )


def capture(
    fn: Callable,
    *args,
    dynamic: bool | None = True,
    dynamic_shapes: Any = None,
    extra_decompositions: Sequence | None = None,
    native_ops: Sequence | None = None,
    simplify: bool = True,
    specialize_float: bool = False,
    **kwargs,
) -> CapturedProgram:
    """
    Captures the functional ATen graph of ``fn(*args, **kwargs)`` (a ``torch.nn.Module`` or function) as TorchDynamo
    and AOTAutograd produce it with the DaCe decomposition table, without compiling or executing it.

    The whole call must be captured into one graph: graph breaks raise Dynamo's error, as with ``fullgraph=True``.
    Gradients are not recorded (inference graph).

    With a per-argument ``dynamic_shapes`` specification, only the named dimensions are symbolic: every other
    dimension of the tensor arguments is marked static, so that it is a plain integer in the graph, in the shape
    guards, and in error messages (integer arguments stay symbolic). ``'all'`` and ``None`` leave the decision to
    ``dynamic``.

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
    :raises CaptureError: If TorchDynamo or AOTAutograd cannot trace the call.
    """
    target = fn.forward if isinstance(fn, torch.nn.Module) else fn
    signature = inspect.signature(target)
    spec = shapes.normalize(dynamic_shapes, signature)
    if spec is not None and dynamic is False:
        raise ValueError("dynamic_shapes requires dynamic=True (or None)")
    backend = _CaptureBackend(
        dynamic_shapes=spec,
        signature=signature,
        extra_decompositions=extra_decompositions,
        native_ops=native_ops,
        simplify=simplify,
    )
    compiled = torch.compile(fn, backend=backend, dynamic=dynamic, fullgraph=True)
    if spec is not None:
        shapes.mark_arguments(signature.bind(*args, **kwargs), spec, static_by_default=spec != shapes.ALL)
    try:
        with (
            torch.no_grad(),
            torch._dynamo.config.patch(specialize_float=specialize_float, **CAPTURE_CONFIG),
            without_conv_backend_guards(),
        ):
            compiled(*args, **kwargs)
    except BackendCompilerFailed as ex:
        if isinstance(ex.inner_exception, _CaptureComplete):
            program = ex.inner_exception.program
            program.owner = fn if isinstance(fn, torch.nn.Module) else None
            program.signature = signature
            program.global_vars = target.__globals__
            return program
        raise _capture_error(fn, ex, backend, spec, signature) from _without_internal_frames(ex)
    except Exception as ex:
        raise _capture_error(fn, ex, backend, spec, signature) from _without_internal_frames(ex)
    raise RuntimeError("TorchDynamo did not compile the callable (it may have been skipped); nothing was captured")


# ---------------------------------------------------------------------------------------------- tracing heuristics
@contextlib.contextmanager
def without_conv_backend_guards() -> Iterator[None]:
    """
    Keeps ATen's convolution backend selection from recording shape guards while tracing.

    ``FakeTensorMode`` runs ``torch._C._select_conv_backend`` on the fake operands of every convolution, only to infer
    the memory format of the output. The CPU heuristics of the selection (``ConvParams::use_mkldnn`` in ATen's
    ``Convolution.cpp``) compare the batch size with constants (``input.size(0) >= 16``), so a symbolic batch turns the
    comparison into a shape guard that specializes the traced graph (``N < 16``) and narrows the symbol's value range,
    although every backend computes the same values and, for operands that are not channels-last, the same memory
    format. The selection therefore runs with guards suppressed. Nothing is lost: the shape checks it performs are
    repeated by the meta kernel of the convolution (``calc_conv_nd_return_shape``), which records them, and a check
    that fails for the example sizes still raises.
    """
    original = torch._C._select_conv_backend
    if getattr(original, "__wrapped__", None) is not None:  # Nested use
        yield
        return

    def select(input, weight, *args, **kwargs):
        shape_env = _shape_env_of(input)
        if shape_env is None:
            return original(input, weight, *args, **kwargs)
        with shape_env.suppress_guards():
            return original(input, weight, *args, **kwargs)

    select.__wrapped__ = original
    torch._C._select_conv_backend = select
    try:
        yield
    finally:
        torch._C._select_conv_backend = original


# ---------------------------------------------------------------------------------------------- error reporting
_DYNAMO_SYMBOL = re.compile(r"\bs\d+\b")
#: Dynamo appends the user frames and its own hints to the message; they are reported separately
_DYNAMO_MESSAGE_SUFFIXES = ("\nfrom user code:", "\nSet TORCHDYNAMO_VERBOSE=1")


def _capture_error(fn: Callable, ex: BaseException, backend: _CaptureBackend, spec: Any, signature) -> CaptureError:
    """Translates an exception of ``torch.compile`` into a :class:`CaptureError` (see its documentation)."""
    what = type(fn).__name__ if isinstance(fn, torch.nn.Module) else getattr(fn, "__qualname__", repr(fn))
    innermost = list(_exception_chain(ex))[-1]
    message = str(innermost)
    for suffix in _DYNAMO_MESSAGE_SUFFIXES:
        message = message.split(suffix)[0]
    message = message.strip()
    shape_env = backend.shape_env or _shape_env_in(ex)
    if shape_env is not None:
        names = symbol_names_from_sources(shape_env, spec, signature)
        if backend.last_description is not None:
            names.update(backend.last_description.symbol_names)
        message = _substitute_symbols(message, shape_env, names)
    lines = [f"Capturing {what} failed: {type(innermost).__name__}: {message}"]
    frames = next((e.real_stack for e in _exception_chain(ex) if getattr(e, "real_stack", None)), None)
    if frames:
        lines.append("from user code:")
        lines.append("".join(traceback.format_list(frames)).rstrip())
    lines.append("Set TORCHDYNAMO_VERBOSE=1 for the complete stack trace.")
    return CaptureError("\n".join(lines))


def _exception_chain(ex: BaseException) -> Iterator[BaseException]:
    """The exception and the exceptions it was raised from (or during), outermost first."""
    seen = set()
    while ex is not None and id(ex) not in seen:
        seen.add(id(ex))
        yield ex
        ex = ex.__cause__ or ex.__context__


def _without_internal_frames(ex: BaseException) -> BaseException:
    """
    Drops the stack traces of ``ex`` and the exceptions it was raised from unless ``TORCHDYNAMO_VERBOSE=1`` (Dynamo's
    own convention): without it, the traceback consists of Dynamo's internals.
    """
    if not torch._dynamo.config.verbose:
        for e in _exception_chain(ex):
            e.__traceback__ = None
    return ex


def _shape_env_in(ex: BaseException) -> ShapeEnv | None:
    """The shape environment of the fake tensors and symbolic scalars in the frames the exception passed through."""
    for e in _exception_chain(ex):
        tb = e.__traceback__
        while tb is not None:
            for value in list(tb.tb_frame.f_locals.values()):
                shape_env = _shape_env_of(value)
                if shape_env is not None:
                    return shape_env
            tb = tb.tb_next
    return None


def _shape_env_of(value: Any) -> ShapeEnv | None:
    if isinstance(value, ShapeEnv):
        return value
    if isinstance(value, FakeTensorMode):
        return value.shape_env
    if isinstance(value, FakeTensor):
        return value.fake_mode.shape_env
    if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        return value.node.shape_env
    return None


def _substitute_symbols(message: str, shape_env: ShapeEnv, names: dict[str, str]) -> str:
    """Replaces Dynamo's symbols in ``message`` by the user's names, or by their example values (hints) if unnamed."""
    hints = getattr(shape_env, "backed_var_to_val", None)
    if hints is None:  # pragma: no cover - older torch
        hints = shape_env.var_to_val
    spelling = {str(sym): names.get(str(sym), str(value)) for sym, value in hints.items()}
    return _DYNAMO_SYMBOL.sub(lambda match: spelling.get(match.group(0), match.group(0)), message)
