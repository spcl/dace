# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Training ``@dace.program`` s that use ``torch.nn.Module`` s with DaCe's automatic differentiation.

A :class:`TrainingStep` turns a program that computes a loss into one SDFG that computes the loss and its gradients
with respect to the parameters of the modules the program uses (and to tensor arguments that require gradients) in a
single call. The gradients are accumulated into ``.grad`` as ``loss.backward()`` would, so that the usual
``torch.optim`` optimizers apply. Parameters are closure arrays of the program, read by reference: an optimizer
updating them in place is seen by the next call without recompiling.
"""
import dataclasses
import inspect
from typing import Any, Dict, List, Set, Tuple

import torch

from dace import data
from dace.autodiff import add_backward_pass
from dace.codegen.compiled_sdfg import CompiledSDFG
from dace.data import create_datadescriptor
from dace.frontend.python.parser import DaceProgram, infer_symbols_from_datadescriptor

from .dtypes import to_torch_dtype

#: Name of the container a program's return value is written to
_RETURN = '__return'


def _gradient_name(name: str) -> str:
    """The container ``add_backward_pass`` uses for the gradient of ``name``."""
    return f'gradient_{name}'


@dataclasses.dataclass
class _Compiled:
    csdfg: CompiledSDFG
    differentiated: List[str]  #: Containers whose gradients the SDFG computes (closure parameters and arguments)
    arguments: Set[str]  #: Names of the SDFG's arguments


class TrainingStep:
    """
    A callable that runs a loss program and accumulates the gradients of its parameters (see :mod:`.training`).

    The program must return the loss as a scalar (e.g., ``return np.sum(...)``). Gradients are computed for every
    ``torch.nn.Parameter`` in the program's closure that requires gradients and every tensor argument that requires
    gradients. The SDFG is compiled once per combination of argument types (symbolic sizes in the program's type
    annotations compile once for all sizes) and of arguments requiring gradients.
    """

    def __init__(self, program: DaceProgram):
        self.program = program
        self._signature = inspect.signature(program.f)
        self._compiled: Dict[Tuple, _Compiled] = {}
        self.compile_count = 0  #: Number of SDFGs compiled (test oracle for compile-once behavior)

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """Runs the program and accumulates the gradients; returns the loss (a 0-d tensor)."""
        arguments = dict(zip(self.program.argnames, args))
        arguments.update(kwargs)
        compiled = self._compile(args, kwargs, arguments)

        closure = self.program.__sdfg_closure__()
        objects = {**arguments, **closure}
        call: Dict[str, Any] = {name: _detached(value) for name, value in objects.items()}
        sdfg = compiled.csdfg.sdfg
        call.update(
            infer_symbols_from_datadescriptor(sdfg, {
                name: create_datadescriptor(value)
                for name, value in call.items() if name in sdfg.arrays
            }))
        loss_desc = sdfg.arrays[_RETURN]
        call[_gradient_name(_RETURN)] = torch.ones(loss_desc.shape, dtype=to_torch_dtype(loss_desc.dtype))
        gradients = {}
        for name in compiled.differentiated:
            target = objects[name]
            gradients[name] = call[_gradient_name(name)] = torch.zeros_like(target,
                                                                            memory_format=torch.contiguous_format)
        # The compiled SDFG reuses its return array on the next call: copy the loss
        loss = torch.tensor(
            compiled.csdfg(**{
                name: value
                for name, value in call.items() if name in compiled.arguments
            }))

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

        sdfg = self.program.to_sdfg(*args, simplify=True, **kwargs)
        # The closure (and with it the modules' parameters) is only known once the program has been parsed
        key = self._key(arguments)
        differentiated = list(key[1])
        if _RETURN not in sdfg.arrays or sdfg.arrays[_RETURN].total_size != 1:
            raise ValueError(f'{self.program.name} must return the loss as a scalar to be trained')
        if not differentiated:
            raise ValueError(f'{self.program.name} uses no parameters or arguments that require gradients')
        add_backward_pass(sdfg, outputs=[_RETURN], inputs=differentiated)
        missing = [name for name in differentiated + [_RETURN] if _gradient_name(name) not in sdfg.arrays]
        if missing:
            raise ValueError(f'Automatic differentiation of {self.program.name} produced no gradient for {missing}')
        compiled = _Compiled(sdfg.compile(), differentiated, set(sdfg.arglist()))
        self.compile_count += 1
        self._compiled[key] = compiled
        return compiled

    def _key(self, arguments: Dict[str, Any]) -> Tuple:
        """Argument types and the names of the closure parameters and arguments that require gradients."""
        values = {**self.program.__sdfg_closure__(), **arguments}
        differentiated = tuple(name for name, value in values.items()
                               if isinstance(value, torch.Tensor) and value.requires_grad)
        return tuple(self._argument_type(name, value) for name, value in arguments.items()), differentiated

    def _argument_type(self, name: str, value: Any) -> str:
        """The type of an argument as far as the SDFG is concerned: its annotation, or its data descriptor."""
        parameter = self._signature.parameters.get(name)
        if parameter is not None and isinstance(parameter.annotation, data.Data):
            return repr(parameter.annotation)
        if isinstance(value, torch.Tensor):
            return repr((value.dtype, tuple(value.shape), tuple(value.stride())))
        return repr((type(value), value if isinstance(value, (bool, int, float, str)) else None))


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


def _detached(value: Any) -> Any:
    return value.detach() if isinstance(value, torch.Tensor) and value.requires_grad else value
