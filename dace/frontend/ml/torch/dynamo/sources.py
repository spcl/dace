# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Where the inputs of a captured graph come from, and which assumptions (guards) TorchDynamo made while tracing it.

Dynamo identifies every graph input and every guarded value by a ``Source`` (e.g.,
``L['self']._modules['fc1']._parameters['weight']``). This module translates sources into :class:`SourceRef` objects
relative to the compiled callable (arguments, parameters, buffers, module attributes, globals, tensor sizes) and
collects the guards and symbol assumptions of a compilation.
"""
import dataclasses
import inspect
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import sympy
import torch
from torch._dynamo import source as dsource
from torch._guards import ChainedSource, TracingContext

from . import shapes
from .context import sanitize_name
from .symbols import SymbolTable, UnsupportedSymbolicExpression

#: Containers of ``torch.nn.Module`` attributes that appear in Dynamo sources (``self._modules['fc1']``)
_MODULE_CONTAINERS = {'_modules': None, '_parameters': 'parameter', '_buffers': 'buffer'}
_TENSOR_PROPERTY_KINDS = ('size', 'stride', 'storage_offset')
_TENSOR_PROPERTIES = {
    dsource.TensorProperty.SIZE: 'size',
    dsource.TensorProperty.STRIDE: 'stride',
    dsource.TensorProperty.STORAGE_OFFSET: 'storage_offset'
}


@dataclasses.dataclass(frozen=True)
class SourceRef:
    """Where a graph input or a guarded value comes from, in terms of the captured callable."""
    #: ``'argument'``, ``'parameter'``, ``'buffer'``, ``'attribute'`` (of the module), ``'global'``, ``'size'``,
    #: ``'stride'``, ``'storage_offset'`` (of a tensor given by the other fields), or ``'unknown'``
    kind: str
    #: Argument or global name; ``None`` for module-relative sources
    root: Optional[str]
    #: Accessors from the root (attribute names, item indices), e.g. ``('fc1', 'weight')`` for a parameter
    path: Tuple[Any, ...]
    #: Dynamo's spelling of the source, e.g. ``"L['self']._modules['fc1']._parameters['weight']"``
    text: str
    #: Dimension of a ``size``/``stride`` source
    dim: Optional[int] = None
    #: Accessors as Dynamo spelled them, including ``torch.nn.Module`` containers (``_modules``, ``_parameters``,
    #: ``_buffers``) that :attr:`path` omits; ``None`` if equal to :attr:`path`
    raw_path: Optional[Tuple[Any, ...]] = None

    @property
    def qualname(self) -> str:
        """Dotted name relative to the callable, e.g. ``fc1.weight`` for a parameter or ``x`` for an argument."""
        return '.'.join(([self.root] if self.root else []) + [str(p) for p in self.path])

    def evaluate(self, owner: Any, arguments: Dict[str, Any], global_vars: Dict[str, Any]) -> Any:
        """
        Returns the current value of this source.

        :param owner: The module whose parameters, buffers, and attributes module-relative sources refer to.
        :param arguments: Arguments of the call by name.
        :param global_vars: Globals of the captured function.
        """
        if self.kind == 'unknown':
            raise ValueError(f'Cannot evaluate the Dynamo source {self.text}')
        if self.root is None:
            value = owner
        elif self.kind == 'global' or (self.kind in _TENSOR_PROPERTY_KINDS and self.root not in arguments):
            value = global_vars[self.root]
        else:
            value = arguments[self.root]
        for accessor in (self.raw_path if self.raw_path is not None else self.path):
            value = value[accessor] if isinstance(value, (dict, list, tuple)) else getattr(value, accessor)
        if self.kind == 'size':
            return value.size(self.dim)
        if self.kind == 'stride':
            return value.stride(self.dim)
        if self.kind == 'storage_offset':
            return value.storage_offset()
        return value


@dataclasses.dataclass(frozen=True)
class CapturedGuard:
    """An assumption Dynamo made while tracing; the graph is only valid while it holds."""
    kind: str  #: Dynamo's guard type, e.g. ``'TENSOR_MATCH'``, ``'CONSTANT_MATCH'``, ``'TYPE_MATCH'``, ``'GRAD_MODE'``
    source: Optional[SourceRef]  #: The guarded value, or ``None`` for global state (grad mode, default device, ...)


@dataclasses.dataclass
class GraphDescription:
    """What the backend knows about a Dynamo-level graph before AOTAutograd lowers it to ATen."""
    inputs: List[SourceRef]  #: Source of each placeholder, in placeholder order
    symbol_names: Dict[str, str]  #: Dynamo symbol name -> DaCe symbol name
    guards: List[CapturedGuard]

    def input_names(self) -> List[Optional[str]]:
        """Container names for the placeholders: qualified names of arguments, parameters, buffers, and attributes."""
        return [
            sanitize_name(ref.qualname) if ref.kind in ('argument', 'parameter', 'buffer', 'attribute',
                                                        'global') else None for ref in self.inputs
        ]


class SourceResolver:
    """
    Translates Dynamo ``Source`` objects into :class:`SourceRef` relative to the captured callable.

    :param signature: Signature of the callable (``forward`` for modules, without ``self``). Dynamo may start tracing
                      in a wrapper frame (``nn.Module.__call__``) whose inputs are ``args[i]``/``kwargs[name]``; the
                      signature maps those back to argument names.
    """

    def __init__(self, signature: Optional[inspect.Signature] = None):
        parameters = list(signature.parameters.values()) if signature is not None else []
        self.parameter_names = {p.name for p in parameters}
        self.positional_names = [p.name for p in parameters if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        self.has_signature = signature is not None

    def resolve(self, source) -> SourceRef:
        text = source.name if source is not None else ''
        accessors: List[Any] = []
        prop: Optional[Tuple[str, Optional[int]]] = None
        while True:
            if isinstance(source, dsource.LocalSource):
                return self._classify(source.local_name, tuple(reversed(accessors)), prop, text, is_global=False)
            if isinstance(source, dsource.GlobalSource):
                return self._classify(source.global_name, tuple(reversed(accessors)), prop, text, is_global=True)
            if isinstance(source, dsource.TensorPropertySource) and not accessors and prop is None:
                prop = (_TENSOR_PROPERTIES[source.prop], source.idx)
                source = source.base
            elif isinstance(source, (dsource.GetItemSource, dsource.DictGetItemSource)):
                if not isinstance(source.index, (int, str)):
                    return SourceRef('unknown', None, (), text)
                accessors.append(source.index)
                source = source.base
            elif isinstance(source, dsource.AttrSource):
                accessors.append(source.member)
                source = source.base
            elif isinstance(source, ChainedSource):  # Wrappers such as NNModuleSource or FloatTensorSource
                source = source.base
            else:
                return SourceRef('unknown', None, (), text)

    def _classify(self, root: str, path: Tuple[Any, ...], prop: Optional[Tuple[str, Optional[int]]], text: str,
                  is_global: bool) -> SourceRef:
        kind = 'global'
        if not is_global:
            root, path = self._unwrap(root, path)
            module_relative = any(p in _MODULE_CONTAINERS for p in path if isinstance(p, str))
            if root is None:
                return SourceRef('unknown', None, (), text)
            if root in self.parameter_names or (not self.has_signature and not module_relative):
                kind = 'argument'
            else:  # The module itself (``self`` of ``forward``)
                kind, root, raw_path = 'attribute', None, path
                stripped = []
                for p in path:
                    if isinstance(p, str) and p in _MODULE_CONTAINERS:
                        kind = _MODULE_CONTAINERS[p] or kind
                    else:
                        stripped.append(p)
                path = tuple(stripped)
                if path != raw_path:
                    return SourceRef(prop[0] if prop else kind, root, path, text, prop[1] if prop else None, raw_path)
        if prop is not None:
            return SourceRef(prop[0], root, path, text, prop[1])
        return SourceRef(kind, root, path, text)

    def _unwrap(self, local: str, path: Tuple[Any, ...]) -> Tuple[Optional[str], Tuple[Any, ...]]:
        if local == 'args' and 'args' not in self.parameter_names and path and isinstance(path[0], int):
            return (self.positional_names[path[0]] if path[0] < len(self.positional_names) else None), path[1:]
        if local == 'kwargs' and 'kwargs' not in self.parameter_names and path and isinstance(path[0], str):
            return path[0], path[1:]
        return local, path


def describe_graph(gm: torch.fx.GraphModule, signature: Optional[inspect.Signature], spec: Any) -> GraphDescription:
    """
    Describes a graph handed to a backend by Dynamo: the source of every placeholder, the DaCe names of its shape
    symbols, and the guards Dynamo installed. Must be called during compilation (it reads the tracing context).

    :param gm: The Dynamo-level graph (its placeholders carry ``meta['grapharg']``).
    :param signature: Signature of the compiled callable, if known.
    :param spec: Normalized ``dynamic_shapes`` specification (see :func:`.shapes.normalize`), or ``None``.
    """
    resolver = SourceResolver(signature)
    inputs = []
    for node in gm.graph.nodes:
        if node.op != 'placeholder':
            continue
        grapharg = node.meta.get('grapharg')
        source = grapharg.source if grapharg is not None else None
        inputs.append(resolver.resolve(source) if source is not None else SourceRef('unknown', None, (), node.name))

    guards = []
    context = TracingContext.try_get()
    if context is not None:
        seen = set()
        for guard in context.guards_context.dynamo_guards:
            key = (guard.create_fn_name(), guard.name)
            if key in seen:
                continue
            seen.add(key)
            source = resolver.resolve(guard.originating_source) if guard.name else None
            guards.append(CapturedGuard(key[0], source))
    guards.sort(key=lambda g: (g.kind, g.source.text if g.source else ''))
    return GraphDescription(inputs, symbol_names_for_graph(gm, spec, signature), guards)


def shape_assumptions(symbol_names: Dict[str, str]) -> Tuple[List[Any], Dict[str, Tuple[Any, Any]]]:
    """
    Returns the symbol relations Dynamo and AOTAutograd assumed so far (as DaCe expressions where translatable, as
    strings otherwise) and the value ranges of all symbols, from the shape environment of the current compilation.
    """
    context = TracingContext.try_get()
    if context is None or context.fake_mode is None or context.fake_mode.shape_env is None:
        return [], {}
    shape_env = context.fake_mode.shape_env
    symtab = SymbolTable(names=symbol_names)
    relations = []
    for guard in shape_env.guards:
        try:
            relations.append(symtab.to_dace(guard.expr))
        except UnsupportedSymbolicExpression:
            relations.append(str(guard.expr))
    ranges = {}
    for sym, value_range in shape_env.var_to_range.items():
        ranges[symbol_names.get(str(sym), str(sym))] = (_bound(value_range.lower), _bound(value_range.upper))
    return relations, ranges


def _bound(value) -> Optional[int]:
    """An integer bound of a symbol's value range, or ``None`` if it is unbounded (``int_oo``) or not an integer."""
    return int(value) if isinstance(value, sympy.Integer) else None


# ---------------------------------------------------------------------------------------------- naming symbols
def _default_name(local: str, path: Sequence[Any], dim: Optional[int]) -> str:
    name = local + ''.join(f'_{p}' for p in path)
    if dim is not None:
        name = f'{name}_dim{dim}'
    return ''.join(c if c.isalnum() or c == '_' else '_' for c in name)


def _bare_symbol(size: Any):
    if isinstance(size, torch.SymInt) and size.node.constant is None:
        expr = size.node.expr
        if expr.is_Symbol:
            return expr
    return None


def symbol_names_for_graph(gm: torch.fx.GraphModule,
                           spec: Any,
                           signature: Optional[inspect.Signature] = None) -> Dict[str, str]:
    """
    Maps Dynamo shape symbols (``s77``) to user-facing names, using the placeholders' sources of a Dynamo-level graph.

    :param gm: The graph handed to the backend by Dynamo (its placeholders carry ``meta['grapharg']``).
    :param spec: A normalized specification (see :func:`.shapes.normalize`).
    :param signature: Signature of the compiled callable, used when Dynamo starts tracing in a wrapper frame whose
                      inputs are ``args[i]``/``kwargs[name]`` (e.g. ``nn.Sequential``).
    :return: Dict from Dynamo symbol name to DaCe symbol name.
    """
    if spec is None:
        return {}
    names: Dict[str, str] = {}
    used: Dict[str, str] = {}  # dace name -> dynamo symbol
    resolver = SourceResolver(signature)

    def assign(sym_name: str, name: str) -> None:
        if sym_name in names:
            return  # first name wins (the program tied two user dimensions to one symbol)
        if name in used and used[name] != sym_name:
            base, n = name, 1
            while name in used:
                n += 1
                name = f'{base}_{n}'
            warnings.warn(
                f'dynamic_shapes names several dimensions "{base}" but the program does not constrain them to '
                f'be equal; the extra one is called "{name}" (use torch._check(a == b) to tie them)')
        names[sym_name] = name
        used[name] = sym_name

    for node in gm.graph.nodes:
        if node.op != 'placeholder':
            continue
        grapharg = node.meta.get('grapharg')
        if grapharg is None or grapharg.source is None:
            continue
        ref = resolver.resolve(grapharg.source)
        if ref.kind != 'argument':
            continue
        entry = shapes.spec_for_argument(spec, ref.root)
        for key in ref.path:
            entry = shapes.spec_at(entry, key)
        if entry is None:
            continue
        value = node.meta.get('example_value')
        if isinstance(value, torch.Tensor):
            for i, ds in enumerate(shapes.dim_specs(entry, value.dim())):
                sym = _bare_symbol(value.shape[i])
                if ds is None or sym is None:
                    continue
                assign(sym.name, ds.name or _default_name(ref.root, ref.path, i))
        elif isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            ds = shapes.parse_dim(entry) if entry != shapes.ALL else shapes.DimSpec(strict=False)
            sym = _bare_symbol(value) if isinstance(value, torch.SymInt) else None
            if ds is not None and sym is not None:
                assign(sym.name, ds.name or _default_name(ref.root, ref.path, None))
    return names
