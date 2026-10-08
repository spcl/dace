# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Training ``@dace.program`` s that use ``torch.nn.Module`` s with DaCe's automatic differentiation.

* A :class:`TrainingStep` turns a program that computes a loss into one SDFG that computes the loss and its
  gradients with respect to the parameters of the modules the program uses (and to tensor arguments that require
  gradients) in a single call. The gradients are accumulated into ``.grad`` as ``loss.backward()`` would.
* A :class:`DifferentiableProgram` makes a program a differentiable PyTorch function (``torch.autograd.Function``)
  with a forward and a backward SDFG (:func:`dace.autodiff.make_backward_pass`), so that it composes with other
  PyTorch code: its outputs can feed further operations and ``backward()`` runs the backward SDFG.

Either way the usual ``torch.optim`` optimizers apply. Parameters are closure arrays of the program, read by
reference: an optimizer updating them in place is seen by the next call without recompiling.
"""

import dataclasses
import inspect
import warnings
from typing import Any, Dict, List, Set, Tuple

import torch

from dace import data, symbolic
from dace.autodiff import (
    BACKWARD_PHASE,
    FORWARD_PHASE,
    BackwardPass,
    TwoPhaseBackwardPass,
    add_backward_pass,
    make_backward_pass,
)
from dace.codegen.compiled_sdfg import CompiledSDFG
from dace.data import create_datadescriptor
from dace.frontend.python.parser import DaceProgram, infer_symbols_from_datadescriptor

from .dtypes import to_torch_dtype

#: Name of the container a program's return value is written to
_RETURN = "__return"


def _gradient_name(name: str) -> str:
    """The container ``add_backward_pass`` uses for the gradient of ``name``."""
    return f"gradient_{name}"


@dataclasses.dataclass
class _Compiled:
    csdfg: CompiledSDFG
    differentiated: List[str]  #: Containers whose gradients the SDFG computes (closure parameters and arguments)
    arguments: Set[str]  #: Names of the SDFG's arguments


class _ProgramCompiler:
    """Compiles a program once per argument types and set of tensors that require gradients."""

    def __init__(self, program: DaceProgram):
        self.program = program
        self._signature = inspect.signature(program.f)
        self._compiled: Dict[Tuple, Any] = {}
        self.compile_count = 0  #: Number of SDFGs compiled (test oracle for compile-once behavior)

    def _arguments(self, args: Tuple, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        arguments = dict(zip(self.program.argnames, args))
        arguments.update(kwargs)
        return arguments

    def _key(self, arguments: Dict[str, Any]) -> Tuple:
        """Argument types and the names of the closure parameters and arguments that require gradients."""
        values = {**self.program.__sdfg_closure__(), **arguments}
        differentiated = tuple(
            name for name, value in values.items() if isinstance(value, torch.Tensor) and value.requires_grad
        )
        return tuple(self._argument_type(name, value) for name, value in arguments.items()), differentiated

    def _argument_type(self, name: str, value: Any) -> str:
        """The type of an argument as far as the SDFG is concerned: its annotation, or its data descriptor."""
        parameter = self._signature.parameters.get(name)
        if parameter is not None and isinstance(parameter.annotation, data.Data):
            return repr(parameter.annotation)
        if isinstance(value, torch.Tensor):
            return repr((value.dtype, tuple(value.shape), tuple(value.stride())))
        return repr((type(value), value if isinstance(value, (bool, int, float, str)) else None))

    def _parse(self, args: Tuple, kwargs: Dict[str, Any], arguments: Dict[str, Any]) -> Tuple[Any, Tuple, List[str]]:
        """Parses the program; returns the SDFG, the cache key, and the containers to differentiate."""
        sdfg = self.program.to_sdfg(*args, simplify=True, **kwargs)
        # The closure (and with it the modules' parameters) is only known once the program has been parsed
        key = self._key(arguments)
        differentiated = list(key[1])
        if not differentiated:
            raise ValueError(f"{self.program.name} uses no parameters or arguments that require gradients")
        return sdfg, key, differentiated

    def _call_arguments(self, sdfg, arguments: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """The SDFG arguments for a call (tensors detached) with inferred symbols, and the objects they come from."""
        objects = {**arguments, **self.program.__sdfg_closure__()}
        call = {name: _detached(value) for name, value in objects.items()}
        call.update(
            infer_symbols_from_datadescriptor(
                sdfg, {name: create_datadescriptor(value) for name, value in call.items() if name in sdfg.arrays}
            )
        )
        return call, objects


class TrainingStep(_ProgramCompiler):
    """
    A callable that runs a loss program and accumulates the gradients of its parameters (see :mod:`.training`).

    The program must return the loss as a scalar (e.g., ``return np.sum(...)``). Gradients are computed for every
    ``torch.nn.Parameter`` in the program's closure that requires gradients and every tensor argument that requires
    gradients. The SDFG is compiled once per combination of argument types (symbolic sizes in the program's type
    annotations compile once for all sizes) and of arguments requiring gradients.
    """

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """Runs the program and accumulates the gradients; returns the loss (a 0-d tensor)."""
        arguments = self._arguments(args, kwargs)
        compiled = self._compile(args, kwargs, arguments)
        sdfg = compiled.csdfg.sdfg
        call, objects = self._call_arguments(sdfg, arguments)
        loss_desc = sdfg.arrays[_RETURN]
        call[_gradient_name(_RETURN)] = torch.ones(loss_desc.shape, dtype=to_torch_dtype(loss_desc.dtype))
        gradients = {}
        for name in compiled.differentiated:
            target = objects[name]
            gradients[name] = call[_gradient_name(name)] = torch.zeros_like(
                target, memory_format=torch.contiguous_format
            )
        # The compiled SDFG reuses its return array on the next call: copy the loss
        loss = torch.tensor(
            compiled.csdfg(**{name: value for name, value in call.items() if name in compiled.arguments})
        )

        for name, gradient in gradients.items():
            target = objects[name]
            if target.grad is None:
                target.grad = gradient
            else:
                target.grad += gradient
        return loss.reshape(())

    def _compile(self, args: Tuple, kwargs: Dict[str, Any], arguments: Dict[str, Any]) -> _Compiled:
        key = self._key(arguments)
        if key in self._compiled:
            return self._compiled[key]

        sdfg, key, differentiated = self._parse(args, kwargs, arguments)
        if _RETURN not in sdfg.arrays or sdfg.arrays[_RETURN].total_size != 1:
            raise ValueError(f"{self.program.name} must return the loss as a scalar to be trained")
        add_backward_pass(sdfg, outputs=[_RETURN], inputs=differentiated)
        missing = [name for name in differentiated + [_RETURN] if _gradient_name(name) not in sdfg.arrays]
        if missing:
            raise ValueError(f"Automatic differentiation of {self.program.name} produced no gradient for {missing}")
        compiled = _Compiled(sdfg.compile(), differentiated, set(sdfg.arglist()))
        self.compile_count += 1
        self._compiled[key] = compiled
        return compiled


def training_step(program: DaceProgram) -> TrainingStep:
    """
    Returns a callable that runs ``program`` (a ``@dace.program`` returning a scalar loss) and accumulates the
    gradients of the module parameters it uses into their ``.grad``, as ``loss.backward()`` would.

    Example::

        @dace.program
        def loss_program(x: dace.float32[N, 8], target: dace.float32[N, 2]):
            return np.sum((model(x) - target)**2)

        step = dace.ml.training_step(loss_program)
        for x, target in batches:
            optimizer.zero_grad()
            loss = step(x, target)
            optimizer.step()
    """
    return TrainingStep(program)


@dataclasses.dataclass
class SDFGPair:
    """Compiled forward and backward SDFGs (see :func:`dace.autodiff.make_backward_pass`)."""

    forward: CompiledSDFG
    backward: CompiledSDFG
    backward_pass: BackwardPass
    outputs: List[str]  #: Return containers of the program
    differentiated: List[str]  #: Containers whose gradients the backward SDFG computes
    forward_arguments: Set[str]  #: Names of the forward SDFG's arguments
    backward_arguments: Set[str]  #: Names of the backward SDFG's arguments


class DifferentiableProgram(_ProgramCompiler):
    """
    A ``@dace.program`` as a differentiable PyTorch function (see :mod:`.training`): calling it runs the forward SDFG
    and returns tensors that PyTorch's autograd backpropagates through with the backward SDFG.

    Gradients flow to every ``torch.nn.Parameter`` in the program's closure that requires gradients and to every
    tensor argument that requires gradients. The SDFGs are compiled once per combination of argument types and of
    tensors requiring gradients.
    """

    def __call__(self, *args, **kwargs):
        arguments = self._arguments(args, kwargs)
        pair = self._compile(args, kwargs, arguments)
        call, objects = self._call_arguments(pair.forward.sdfg, arguments)
        outputs = SDFGPairFunction.apply(pair, call, *[objects[name] for name in pair.differentiated])
        return outputs[0] if len(outputs) == 1 else outputs

    def _compile(self, args: Tuple, kwargs: Dict[str, Any], arguments: Dict[str, Any]) -> SDFGPair:
        key = self._key(arguments)
        if key in self._compiled:
            return self._compiled[key]
        sdfg, key, differentiated = self._parse(args, kwargs, arguments)
        outputs = sorted(name for name, desc in sdfg.arrays.items() if name.startswith(_RETURN) and not desc.transient)
        if not outputs:
            raise ValueError(f"{self.program.name} returns nothing to differentiate")
        pair = compile_pair(make_backward_pass(sdfg, outputs=outputs, inputs=differentiated), outputs, differentiated)
        self.compile_count += 2
        self._compiled[key] = pair
        return pair


def compile_pair(backward_pass: BackwardPass, outputs: List[str], differentiated: List[str]) -> SDFGPair:
    """Compiles the forward and backward SDFGs of ``backward_pass``."""
    return SDFGPair(
        backward_pass.forward.compile(),
        backward_pass.backward.compile(),
        backward_pass,
        outputs,
        differentiated,
        set(backward_pass.forward.arglist()),
        set(backward_pass.backward.arglist()),
    )


class SDFGPairFunction(torch.autograd.Function):
    """
    Runs a forward SDFG, and its backward SDFG in ``backward``. ``apply(pair, call, *differentiated)`` takes the SDFG
    arguments by name (tensors detached, and symbols) and the tensors to differentiate, in the order of
    ``pair.differentiated``; it returns the outputs in the order of ``pair.outputs``.
    """

    @staticmethod
    def forward(ctx, pair: SDFGPair, call: Dict[str, Any], *differentiated: torch.Tensor):
        backward_pass = pair.backward_pass
        sdfg = pair.forward.sdfg
        symbols = {name: value for name, value in call.items() if name in sdfg.symbols}
        # Data for the backward pass: inputs are forwarded as themselves, the rest is written by the forward SDFG
        forwarded = {
            name: call[name] if name in call else _allocate(sdfg.arrays[name], symbols)
            for name in backward_pass.forwarded
        }
        outputs = {name: _allocate(sdfg.arrays[name], symbols) for name in pair.outputs}
        arguments = {name: value for name, value in call.items() if name in pair.forward_arguments}
        with warnings.catch_warnings():  # Return arrays are passed as arguments, so that torch allocates them
            warnings.filterwarnings("ignore", message="Return value .* is passed as a regular argument")
            pair.forward(**{**arguments, **forwarded, **outputs})
        ctx.pair, ctx.call, ctx.forwarded = pair, call, forwarded
        ctx.shapes = [(t.shape, t.dtype) for t in differentiated]
        return tuple(outputs[name] for name in pair.outputs)

    @staticmethod
    def backward(ctx, *output_gradients: torch.Tensor):
        pair, call = ctx.pair, ctx.call
        backward_pass = pair.backward_pass
        sdfg = pair.backward.sdfg
        symbols = {name: value for name, value in call.items() if name in pair.forward.sdfg.symbols}
        arguments = {backward_pass.forwarded[name]: value for name, value in ctx.forwarded.items()}
        for name, gradient in zip(pair.outputs, output_gradients):
            if name in backward_pass.output_gradients:
                desc = sdfg.arrays[backward_pass.output_gradients[name]]
                arguments[backward_pass.output_gradients[name]] = (
                    gradient.contiguous()
                    if gradient is not None
                    else torch.zeros(_shape(desc, symbols), dtype=to_torch_dtype(desc.dtype))
                )
        gradients = {}
        for name, (shape, dtype) in zip(pair.differentiated, ctx.shapes):
            gradient = backward_pass.input_gradients.get(name)
            if gradient is not None:
                gradients[name] = arguments[gradient] = torch.zeros(shape, dtype=dtype)
        arguments.update(
            {name: value for name, value in call.items() if name in pair.backward_arguments and name not in arguments}
        )
        for name in pair.backward_arguments - set(arguments) - set(sdfg.symbols):
            # Containers the backward SDFG writes but nobody reads (e.g., outputs of a recomputed forward pass)
            if isinstance(sdfg.arrays.get(name), data.Array):
                arguments[name] = _allocate(sdfg.arrays[name], symbols)
        pair.backward(**arguments)
        return (None, None, *[gradients.get(name) for name in pair.differentiated])


@dataclasses.dataclass
class CompiledTwoPhase:
    """A compiled SDFG with a forward and a backward phase (see :func:`dace.autodiff.make_two_phase_backward_pass`)."""

    csdfg: CompiledSDFG
    two_phase: TwoPhaseBackwardPass
    outputs: List[str]  #: Return containers
    differentiated: List[str]  #: Containers whose gradients the backward phase computes


class TwoPhaseFunction(torch.autograd.Function):
    """
    Runs the forward phase of a two-phase SDFG, and its backward phase in ``backward``. The forward phase writes the
    tape, which is saved for the backward phase. ``apply(compiled, call, *differentiated)`` takes the SDFG arguments
    by name (tensors detached, and symbols) and the tensors to differentiate, in the order of
    ``compiled.differentiated``; it returns the outputs in the order of ``compiled.outputs``.
    """

    @staticmethod
    def forward(ctx, compiled: CompiledTwoPhase, call: Dict[str, Any], *differentiated: torch.Tensor):
        two_phase, sdfg = compiled.two_phase, compiled.csdfg.sdfg
        symbols = {name: value for name, value in call.items() if name in sdfg.symbols}
        tape = {name: _allocate(sdfg.arrays[name], symbols) for name in two_phase.tape}
        outputs = {name: _allocate(sdfg.arrays[name], symbols) for name in compiled.outputs}
        _call_phase(compiled, FORWARD_PHASE, two_phase.forward_arguments, {**call, **tape, **outputs})
        # Tensors the backward phase reads are saved, so that PyTorch detects their modification in place
        saved = {**{name: value for name, value in call.items() if isinstance(value, torch.Tensor)}, **tape, **outputs}
        saved = {name: value for name, value in saved.items() if name in two_phase.backward_arguments}
        ctx.save_for_backward(*saved.values())
        ctx.compiled, ctx.saved_names = compiled, list(saved)
        ctx.symbols = symbols
        ctx.shapes = [(t.shape, t.dtype) for t in differentiated]
        return tuple(outputs[name] for name in compiled.outputs)

    @staticmethod
    def backward(ctx, *output_gradients: torch.Tensor):
        compiled = ctx.compiled
        two_phase, sdfg = compiled.two_phase, compiled.csdfg.sdfg
        given = dict(zip(ctx.saved_names, ctx.saved_tensors))
        given.update(ctx.symbols)
        for name, gradient in zip(compiled.outputs, output_gradients):
            cotangent = two_phase.output_gradients.get(name)
            if cotangent is not None:
                desc = sdfg.arrays[cotangent]
                given[cotangent] = (
                    gradient.contiguous()
                    if gradient is not None
                    else torch.zeros(_shape(desc, ctx.symbols), dtype=to_torch_dtype(desc.dtype))
                )
        gradients = {}
        for name, (shape, dtype) in zip(compiled.differentiated, ctx.shapes):
            gradient = two_phase.input_gradients.get(name)
            if gradient is not None:
                gradients[name] = given[gradient] = torch.zeros(shape, dtype=dtype)
        _call_phase(compiled, BACKWARD_PHASE, two_phase.backward_arguments, given)
        return (None, None, *[gradients.get(name) for name in compiled.differentiated])


def _call_phase(compiled: CompiledTwoPhase, phase: int, used: Set[str], given: Dict[str, Any]):
    """Calls one phase of a two-phase SDFG: arrays of the other phase are None, its symbols are placeholders."""
    arguments = {compiled.two_phase.phase: phase}
    for name, desc in compiled.csdfg.sdfg.arglist().items():
        if name in arguments:
            continue
        if name in used:
            arguments[name] = given[name]
        elif isinstance(desc, data.Array):
            arguments[name] = None
        else:
            arguments[name] = given.get(name, 0)
    with warnings.catch_warnings():  # Return arrays are passed as arguments, so that torch allocates them
        warnings.filterwarnings("ignore", message="Return value .* is passed as a regular argument")
        compiled.csdfg(**arguments)


def differentiable(program: DaceProgram) -> DifferentiableProgram:
    """
    Returns ``program`` (a ``@dace.program``, which may use ``torch.nn.Module`` s) as a differentiable PyTorch
    function: its outputs are tensors, and ``backward()`` on anything computed from them runs DaCe's backward SDFG of
    the program, accumulating gradients into the parameters of the modules it uses and into arguments that require
    gradients.

    Example::

        @dace.program
        def block(x: dace.float32[N, 8]):
            return np.tanh(model(x)) * 2

        block = dace.ml.differentiable(block)
        loss = torch.nn.functional.mse_loss(head(block(x)), y)
        loss.backward()
    """
    return DifferentiableProgram(program)


def _shape(desc: data.Data, symbols: Dict[str, Any]) -> Tuple[int, ...]:
    return tuple(int(symbolic.evaluate(size, symbols)) for size in desc.shape)


def _allocate(desc: data.Data, symbols: Dict[str, Any]) -> torch.Tensor:
    """A torch tensor for a container, with its strides."""
    strides = tuple(int(symbolic.evaluate(stride, symbols)) for stride in desc.strides)
    return torch.empty_strided(_shape(desc, symbols), strides, dtype=to_torch_dtype(desc.dtype))


def _detached(value: Any) -> Any:
    return value.detach() if isinstance(value, torch.Tensor) and value.requires_grad else value
