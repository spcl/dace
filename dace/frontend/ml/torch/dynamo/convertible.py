# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
``torch.nn.Module`` objects used inside ``@dace.program``.

The Python frontend wraps every module it finds in a program's closure with :func:`as_sdfg_convertible`. When the
program is parsed, the module is captured with TorchDynamo (:func:`.capture.capture`) for the data descriptors of the
call site and nested as an SDFG:

* Symbolic sizes of the arguments (DaCe symbols) become the module's symbolic sizes, under the same names.
* Parameters and buffers are closure arrays of the program, passed by reference (updates are visible without parsing
  again).
* Dynamo's guards on the module (its ``training`` flag, attribute values, submodule types, ...) and on global state
  become part of the program's cache key (``SDFGConvertible.__sdfg_guards__``): a change parses the program again.
"""
import inspect
import itertools
import warnings
import weakref
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import numpy
import sympy
import torch

from dace import data, dtypes, symbolic
from dace.frontend.python.common import SDFGClosure, SDFGConvertible, structured_argument_name
from dace.sdfg import SDFG

from .capture import CapturedProgram, capture
from .context import sanitize_name
from .dtypes import to_torch_dtype
from .shapes import DimSpec
from .sources import CapturedGuard, SourceRef
from .symbols import free_symbol_names

#: One adapter per module, so that the Python frontend sees the same object at every use
_ADAPTERS: 'weakref.WeakKeyDictionary[torch.nn.Module, ModuleConvertible]' = weakref.WeakKeyDictionary()

#: Guards on a value, as the projection of the value that the guard compares
_GUARD_PROJECTIONS: Dict[str, Callable[[Any], Any]] = {
    'TYPE_MATCH': type,
    'ID_MATCH': id,
    'MODULE_MATCH': id,
    'BUILTIN_MATCH': id,
    'FUNCTION_MATCH': id,
    'CLOSURE_MATCH': id,
    'CONSTANT_MATCH': lambda value: value,
    'EQUALS_MATCH': lambda value: value,
    'EMPTY_NN_MODULE_HOOKS_DICT': len,
    'DICT_LENGTH': len,
    'SEQUENCE_LENGTH': len,
    'LIST_LENGTH': len,
}

#: Guards on global state that change the captured graph
_GLOBAL_STATE_GUARDS: Dict[str, Callable[[], Any]] = {
    'DEFAULT_DEVICE': lambda: str(torch.get_default_device()),
    'DETERMINISTIC_ALGORITHMS': torch.are_deterministic_algorithms_enabled,
}

#: Guards that need no runtime check: descriptors of arguments and closure arrays are part of the cache key already,
#: and the captured graph is an inference graph regardless of the grad mode
_IMPLIED_GUARDS = {'TENSOR_MATCH', 'GRAD_MODE', 'SHAPE_ENV'}

#: Graph inputs that a module conversion can provide
_SUPPORTED_INPUTS = ('argument', 'parameter', 'buffer', 'attribute', 'global', 'size', 'stride', 'storage_offset')


def as_sdfg_convertible(module: torch.nn.Module) -> 'ModuleConvertible':
    """Returns the (unique) SDFG-convertible adapter of ``module``."""
    adapter = _ADAPTERS.get(module)
    if adapter is None:
        adapter = ModuleConvertible(module)
        _ADAPTERS[module] = adapter
    return adapter


class ModuleConvertible(SDFGConvertible):
    """A ``torch.nn.Module`` as an SDFG-convertible object (see the module documentation)."""

    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.signature = inspect.signature(module.forward)
        self.program: Optional[CapturedProgram] = None  #: The most recent capture
        self.unchecked_guards: List[CapturedGuard] = []  #: Guards of the most recent capture that are not checked
        self._closure: Dict[str, Callable[[], Any]] = {}
        self._guards: Dict[str, Callable[[], Any]] = {}

    @property
    def name(self) -> str:
        return sanitize_name(type(self.module).__name__.lower())

    # ------------------------------------------------------------------------------------------ SDFGConvertible
    def __sdfg_signature__(self) -> Tuple[Sequence[str], Sequence[str]]:
        variadic = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        return [p.name for p in self.signature.parameters.values() if p.kind not in variadic], []

    def closure_resolver(self,
                         constant_args: Dict[str, Any],
                         given_args: Set[str],
                         parent_closure: Optional[SDFGClosure] = None) -> SDFGClosure:
        # Parameters and buffers are known before capturing: they are the closure arrays of the program
        closure = SDFGClosure()
        for qualname, tensor in self._state_tensors():
            closure.closure_arrays[sanitize_name(qualname)] = (qualname, data.create_datadescriptor(tensor),
                                                               _state_getter(self.module, qualname), False)
        return closure

    def __sdfg_closure__(self, reevaluate: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        if self.program is None:
            return {sanitize_name(qualname): tensor for qualname, tensor in self._state_tensors()}
        return {name: getter() for name, getter in self._closure.items()}

    def __sdfg_guards__(self) -> Dict[str, Callable[[], Any]]:
        return dict(self._guards)

    def __sdfg__(self, *args, **kwargs) -> SDFG:
        bound = self.signature.bind(*args, **kwargs)
        examples, spec = example_arguments(bound.arguments)
        bound.arguments.update(examples)
        program = capture(self.module, *bound.args, dynamic_shapes=spec, specialize_float=True, **bound.kwargs)

        for ref in program.inputs:
            if ref.kind not in _SUPPORTED_INPUTS:
                raise NotImplementedError(f'{type(self.module).__name__}: graph input {ref.text} ({ref.kind}) cannot '
                                          'be passed from a @dace.program yet')
        result = program.import_graph(self.name, return_arrays=True)

        # Closure arrays under their container names in the SDFG. Parameters and buffers were declared to the program
        # before parsing; tensors in plain attributes and globals are reported now and passed from then on.
        self._closure = {}
        for spec in result.inputs:
            ref = program.inputs[spec.position]
            if spec.kind != 'tensor':
                continue
            if ref.kind in ('parameter', 'buffer'):
                self._closure[spec.name] = _state_getter(self.module, ref.qualname)
            elif ref.kind in ('attribute', 'global'):
                self._closure[spec.name] = _source_evaluator(ref, _identity, self.module, program.global_vars)

        self._guards, self.unchecked_guards = guard_evaluators(program, self.module, f'__torch_{id(self.module):x}')
        _warn_on_symbolic_assumptions(program, type(self.module).__name__)
        self.program = program
        return result.sdfg

    # ------------------------------------------------------------------------------------------ helpers
    def _state_tensors(self) -> List[Tuple[str, torch.Tensor]]:
        return list(self.module.named_parameters()) + list(self.module.named_buffers())


def _state_getter(module: torch.nn.Module, qualname: str) -> Callable[[], torch.Tensor]:
    """
    Returns a function that looks up a parameter or buffer by name at call time. It returns the tensor object itself:
    the Python frontend matches closure arrays by identity.
    """
    owner_name, _, attr = qualname.rpartition('.')

    def get() -> torch.Tensor:
        owner = module.get_submodule(owner_name) if owner_name else module
        return getattr(owner, attr)

    return get


# ---------------------------------------------------------------------------------------------- example arguments
def example_arguments(arguments: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Creates example values for capturing from the arguments of a call site in a ``@dace.program``, which are data
    descriptors (possibly with symbolic sizes), constants, or lists, tuples, and dicts of those, and the
    ``dynamic_shapes`` specification that names the resulting Dynamo symbols after the DaCe symbols.

    Symbolic sizes get distinct example values (hints) of at least 2 that differ from every static size of the
    arguments, so that Dynamo neither specializes them nor unifies them with unrelated sizes ("duck shaping").
    Arguments that are integer scalars or symbols become symbolic integers named after the argument (or the element,
    e.g., ``xs_0``).

    :return: A tuple of (example values by argument name, ``dynamic_shapes`` specification by argument name).
    """
    static: Set[int] = set()
    symbols: List[str] = []
    for leaf in _leaves(arguments):
        if isinstance(leaf, data.Array):
            for size in itertools.chain(leaf.shape, leaf.strides):
                if symbolic.issymbolic(size):
                    symbols.extend(sorted(free_symbol_names(sympy.sympify(size))))
                else:
                    static.add(int(size))
        elif symbolic.issymbolic(leaf):
            symbols.extend(sorted(free_symbol_names(leaf)))
        elif isinstance(leaf, int) and not isinstance(leaf, bool):
            static.add(leaf)
    candidates = (h for h in itertools.count(2) if h not in static)
    hints: Dict[str, int] = {sym: next(candidates) for sym in dict.fromkeys(symbols)}

    examples: Dict[str, Any] = {}
    spec: Dict[str, Any] = {}
    for name, value in arguments.items():
        examples[name], spec[name] = _example(value, name, (), hints, candidates)
    return examples, spec


def _leaves(value: Any) -> Iterator[Any]:
    if isinstance(value, dict):
        for v in value.values():
            yield from _leaves(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _leaves(v)
    else:
        yield value


def _example(value: Any, name: str, path: Tuple[Any, ...], hints: Dict[str, int],
             candidates: Iterator[int]) -> Tuple[Any, Any]:
    """Example value and ``dynamic_shapes`` entry of one (possibly structured) argument; see :func:`example_arguments`."""
    if isinstance(value, dict):
        pairs = {k: _example(v, name, path + (k, ), hints, candidates) for k, v in value.items()}
        return {k: e for k, (e, _) in pairs.items()}, {k: s for k, (_, s) in pairs.items()}
    if isinstance(value, (list, tuple)):
        pairs = [_example(v, name, path + (k, ), hints, candidates) for k, v in enumerate(value)]
        return type(value)(e for e, _ in pairs), [s for _, s in pairs]

    element = structured_argument_name(name, path)
    if isinstance(value, data.Scalar):
        if numpy.issubdtype(value.dtype.type, numpy.integer):
            return next(candidates), DimSpec(name=element, strict=False)  # A runtime integer: symbolic
        # Other runtime scalars are traced as 0-d tensors (not specialized to the example value)
        return torch.zeros((), dtype=to_torch_dtype(value.dtype)), None
    if isinstance(value, data.Array):
        dims = {
            k: DimSpec(name=str(size) if isinstance(size, sympy.Symbol) else f'{element}_dim{k}', strict=False)
            for k, size in enumerate(value.shape) if symbolic.issymbolic(size)
        }
        return _example_tensor(value, hints), dims
    if symbolic.issymbolic(value):
        return int(symbolic.evaluate(value, hints)), DimSpec(name=element, strict=False)
    if isinstance(value, int) and not isinstance(value, bool):
        return value, DimSpec(name=element, strict=False)
    return value, None


def _example_tensor(desc: data.Array, hints: Dict[str, int]) -> torch.Tensor:
    shape = [int(symbolic.evaluate(s, hints)) if symbolic.issymbolic(s) else int(s) for s in desc.shape]
    strides = [int(symbolic.evaluate(s, hints)) if symbolic.issymbolic(s) else int(s) for s in desc.strides]
    device = 'cuda' if desc.storage == dtypes.StorageType.GPU_Global else 'cpu'
    return torch.empty_strided(shape, strides, dtype=to_torch_dtype(desc.dtype), device=device)


# ---------------------------------------------------------------------------------------------- guards
def guard_evaluators(program: CapturedProgram, module: torch.nn.Module,
                     prefix: str) -> Tuple[Dict[str, Callable[[], Any]], List[CapturedGuard]]:
    """
    Translates the guards of a capture into evaluators for a ``@dace.program`` cache key.

    Guards on arguments and closure arrays are implied by their descriptors. Guards on module attributes and globals
    evaluate the guarded value (or its type, identity, or length, depending on the guard). Global state guards
    evaluate the state.

    :return: A tuple of (evaluators by unique name, guards that are neither implied nor checked).
    """
    evaluators: Dict[str, Callable[[], Any]] = {}
    unchecked: List[CapturedGuard] = []
    for index, guard in enumerate(program.guards):
        if guard.kind in _IMPLIED_GUARDS or (guard.source is not None and guard.source.kind == 'argument'):
            continue
        if guard.source is None:
            evaluator = _GLOBAL_STATE_GUARDS.get(guard.kind)
        elif guard.kind in _GUARD_PROJECTIONS and guard.source.kind != 'unknown':
            evaluator = _source_evaluator(guard.source, _GUARD_PROJECTIONS[guard.kind], module, program.global_vars)
        else:
            evaluator = None
        if evaluator is None:
            unchecked.append(guard)
        else:
            evaluators[f'{prefix}_guard_{index}'] = evaluator
    return evaluators, unchecked


def _identity(value: Any) -> Any:
    return value


def _source_evaluator(ref: SourceRef, projection: Callable[[Any], Any], module: torch.nn.Module,
                      global_vars: Dict[str, Any]) -> Callable[[], Any]:

    def evaluate() -> Any:
        try:
            return projection(ref.evaluate(module, {}, global_vars))
        except (AttributeError, KeyError, IndexError, TypeError) as ex:
            return ('<unavailable>', type(ex).__name__)  # A changed structure invalidates the cache entry, too

    return evaluate


def _warn_on_symbolic_assumptions(program: CapturedProgram, what: str) -> None:
    """Warns if Dynamo specialized the graph on relations between the user's symbolic sizes."""
    user_symbols = set(program.symbol_names.values())
    for relation in program.shape_guards:
        if isinstance(relation, sympy.Basic) and free_symbol_names(relation) & user_symbols:
            warnings.warn(f'{what}: the captured graph assumes {relation}; results are only valid when it holds')
