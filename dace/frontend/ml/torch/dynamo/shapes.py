# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
User control over which sizes are symbolic, and naming of the resulting DaCe symbols.

TorchDynamo decides on its own which sizes become symbols: with ``dynamic=True`` every size is symbolic, but equal
sizes in the first call share one symbol ("duck shaping") and sizes 0 and 1 are specialized. The ``dynamic_shapes``
argument of :func:`dace.frontend.ml.torch.dynamo.compile` gives this control to the user, with the same vocabulary as
``torch.export``:

* ``'all'``: every dimension of every tensor argument becomes its own symbol (parameters and buffers stay static).
  Dimensions of size 0 or 1 in the first call remain static, because PyTorch specializes them and treats size-1
  operands as broadcasting candidates. Symbols are named ``<argument>_dim<k>``.
* A specification mirroring the arguments: a dict keyed by argument name (or a tuple in positional order) whose
  entries mirror the structure of each argument. A tensor entry is a dict ``{dim: name}`` or a sequence with one entry
  per dimension; an integer argument takes a single entry. Each entry is a ``str`` (the DaCe symbol name), a
  ``torch.export.Dim`` (its name and bounds are used), ``Dim.AUTO``/``Dim.DYNAMIC`` (anonymous symbol), ``'all'``
  (everything below is symbolic) or ``None``/``Dim.STATIC`` (static).

Named dimensions are marked with ``torch._dynamo.mark_dynamic`` (an error is raised if the program specializes them),
``'all'`` uses ``maybe_mark_dynamic`` (lenient). Two dimensions with the same name are only one symbol if the program
constrains them to be equal (for instance by using both in one operation); otherwise the second one is suffixed and a
warning is emitted, because ``torch.compile`` cannot impose the equality from the outside (``torch.export`` can).
"""
import dataclasses
import inspect
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

ALL = 'all'


@dataclasses.dataclass
class DimSpec:
    """One symbolic dimension requested by the user."""
    name: Optional[str] = None  #: DaCe symbol name (``None``: keep Dynamo's name)
    min: Optional[int] = None
    max: Optional[int] = None
    strict: bool = True  #: Raise if the program specializes the dimension (``mark_dynamic`` vs ``maybe_mark_dynamic``)


def _is_export_dim(entry: Any) -> bool:
    from torch.export.dynamic_shapes import Dim
    return isinstance(entry, Dim)


def _is_derived_dim(entry: Any) -> bool:
    from torch.export.dynamic_shapes import _DerivedDim
    return isinstance(entry, _DerivedDim)


def _is_static_dim(entry: Any) -> bool:
    from torch.export.dynamic_shapes import _StaticDim
    return isinstance(entry, _StaticDim)


def _dim_hint_type(entry: Any) -> Optional[str]:
    """Returns 'AUTO', 'DYNAMIC' or 'STATIC' for ``torch.export.Dim.AUTO`` etc., ``None`` otherwise."""
    from torch.export.dynamic_shapes import _DimHint
    if isinstance(entry, _DimHint):
        return entry.type.name
    return None


def parse_dim(entry: Any) -> Optional[DimSpec]:
    """Translates one per-dimension entry of a ``dynamic_shapes`` specification. ``None`` means static."""
    if entry is None or isinstance(entry, int):
        return None
    if entry == ALL:
        return DimSpec(strict=False)
    if isinstance(entry, str):
        return DimSpec(name=entry)
    if _is_export_dim(entry):
        if _is_static_dim(entry):
            return None
        # Derived dims (e.g. 2*batch) have a root but no name of their own: keep them anonymous
        if _is_derived_dim(entry):
            return DimSpec()
        mn = getattr(entry, 'min', None)
        mx = getattr(entry, 'max', None)
        # torch uses 0/2 and sys.maxsize-like sentinels for unbounded dims; pass only explicit bounds
        if mn is not None and mn <= 0:
            mn = None
        if mx is not None and mx >= 2**62:
            mx = None
        return DimSpec(name=entry.__name__, min=mn, max=mx)
    hint = _dim_hint_type(entry)
    if hint == 'STATIC':
        return None
    if hint in ('AUTO', 'DYNAMIC'):
        return DimSpec(min=getattr(entry, 'min', None), max=getattr(entry, 'max', None), strict=(hint == 'DYNAMIC'))
    raise TypeError(f'Unsupported dynamic_shapes entry {entry!r}')


def normalize(spec: Any, signature: inspect.Signature) -> Any:
    """
    Normalizes the user-level ``dynamic_shapes`` into a dict keyed by parameter name (or :data:`ALL`).

    :param spec: ``'all'``/``True``, a dict keyed by parameter name, or a sequence in positional order.
    :param signature: Signature of the compiled function or module ``forward``.
    """
    if spec is None or spec is False:
        return None
    if spec is True or spec == ALL:
        return ALL
    params = [p for p in signature.parameters.values()]
    if isinstance(spec, dict):
        names = {p.name for p in params}
        unknown = set(spec) - names
        if unknown:
            raise ValueError(f'dynamic_shapes refers to unknown argument(s) {sorted(unknown)}; '
                             f'available: {sorted(names)}')
        return dict(spec)
    if isinstance(spec, (list, tuple)):
        positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        if len(spec) > len(positional):
            raise ValueError(f'dynamic_shapes has {len(spec)} entries but the function takes {len(positional)} '
                             'positional arguments')
        return {p.name: s for p, s in zip(positional, spec)}
    raise TypeError(f'dynamic_shapes must be "all", a dict, or a sequence, not {type(spec).__name__}')


def spec_for_argument(spec: Any, name: str) -> Any:
    if spec is None:
        return None
    if spec == ALL:
        return ALL
    return spec.get(name)


def spec_at(spec: Any, key: Any) -> Any:
    """Descends one level (list index or dict key) into a specification."""
    if spec is None or spec == ALL:
        return spec
    if isinstance(spec, dict):
        return spec.get(key)
    if isinstance(spec, (list, tuple)):
        return spec[key] if isinstance(key, int) and 0 <= key < len(spec) else None
    return None


def dim_specs(spec: Any, ndim: int) -> List[Optional[DimSpec]]:
    """Per-dimension specification for a tensor of rank ``ndim``."""
    if spec is None:
        return [None] * ndim
    if spec == ALL:
        return [DimSpec(strict=False)] * ndim
    if isinstance(spec, dict):
        return [parse_dim(spec.get(i)) for i in range(ndim)]
    if isinstance(spec, (list, tuple)):
        if len(spec) != ndim:
            raise ValueError(f'dynamic_shapes entry has {len(spec)} dimensions, tensor has {ndim}')
        return [parse_dim(s) for s in spec]
    if isinstance(spec, str) or _is_export_dim(spec) or _dim_hint_type(spec) is not None:
        raise TypeError('A tensor entry of dynamic_shapes must be a dict {dim: name} or a per-dimension sequence')
    raise TypeError(f'Unsupported dynamic_shapes entry {spec!r}')


# ---------------------------------------------------------------------------------------------- marking arguments
def mark_value(value: Any, spec: Any) -> None:
    """Marks the tensors in ``value`` (recursively) as dynamic according to ``spec``."""
    if spec is None:
        return
    if isinstance(value, torch.Tensor):
        for i, ds in enumerate(dim_specs(spec, value.dim())):
            if ds is None:
                continue
            if ds.name is None and not ds.strict and value.shape[i] in (0, 1):
                continue  # 'all': sizes 0/1 are specialized by PyTorch anyway (and may be broadcasting)
            if ds.strict:
                bounds = {}
                if ds.min is not None or ds.max is not None:
                    bounds = dict(min=ds.min if ds.min is not None else 0, max=ds.max if ds.max is not None else 2**62)
                torch._dynamo.mark_dynamic(value, i, **bounds)
            else:
                torch._dynamo.maybe_mark_dynamic(value, i)
        return
    if isinstance(value, (list, tuple)):
        for k, v in enumerate(value):
            mark_value(v, spec_at(spec, k))
    elif isinstance(value, dict):
        for k, v in value.items():
            mark_value(v, spec_at(spec, k))
    # ints/floats are made symbolic by dynamic=True; nothing to mark


def mark_arguments(bound: inspect.BoundArguments, spec: Any) -> None:
    for name, value in bound.arguments.items():
        mark_value(value, spec_for_argument(spec, name))


def install_marking(compiled: Callable, original: Callable, spec: Any) -> Callable:
    """
    Makes ``compiled`` mark its arguments before every call. Modules get a forward pre-hook (and stay modules);
    functions are wrapped.
    """
    if spec is None:
        return compiled
    target = original.forward if isinstance(original, torch.nn.Module) else original
    signature = inspect.signature(target)

    def mark(*args, **kwargs):
        try:
            bound = signature.bind(*args, **kwargs)
        except TypeError:
            return
        mark_arguments(bound, spec)

    if isinstance(compiled, torch.nn.Module):
        compiled.register_forward_pre_hook(lambda mod, args, kwargs: mark(*args, **kwargs), with_kwargs=True)
        return compiled

    import functools

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        mark(*args, **kwargs)
        return compiled(*args, **kwargs)

    wrapper.compiled = compiled
    return wrapper


# ---------------------------------------------------------------------------------------------- naming symbols
def source_path(source) -> Optional[Tuple[str, Tuple[Any, ...]]]:
    """Resolves a Dynamo ``Source`` to ``(local name, accessor path)`` or ``None`` if it is not a plain argument."""
    from torch._dynamo import source as S
    path: List[Any] = []
    while source is not None:
        if isinstance(source, S.LocalSource):
            return source.local_name, tuple(reversed(path))
        if isinstance(source, (S.GetItemSource, S.DictGetItemSource)):
            index = source.index
            if getattr(source, 'index_is_slice', False) or not isinstance(index, (int, str)):
                return None
            path.append(index)
            source = source.base
        elif isinstance(source, S.AttrSource):
            path.append(source.member)
            source = source.base
        else:
            return None
    return None


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
    :param spec: A normalized specification (see :func:`normalize`).
    :param signature: Signature of the compiled callable, used when Dynamo starts tracing in a wrapper frame whose
                      inputs are ``args[i]``/``kwargs[name]`` (e.g. ``nn.Sequential``).
    :return: Dict from Dynamo symbol name to DaCe symbol name.
    """
    if spec is None:
        return {}
    names: Dict[str, str] = {}
    used: Dict[str, str] = {}  # dace name -> dynamo symbol
    parameters = list(signature.parameters.values()) if signature is not None else []
    parameter_names = {p.name for p in parameters}
    positional_names = [p.name for p in parameters if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]

    def assign(sym_name: str, name: str) -> None:
        if sym_name in names:
            return  # first name wins (the program tied two user dimensions to one symbol)
        if name in used and used[name] != sym_name:
            base, n = name, 1
            while name in used:
                n += 1
                name = f'{base}_{n}'
            warnings.warn(f'dynamic_shapes names several dimensions "{base}" but the program does not constrain them to '
                          f'be equal; the extra one is called "{name}" (use torch._check(a == b) to tie them)')
        names[sym_name] = name
        used[name] = sym_name

    def unwrap(local: str, path: Tuple[Any, ...]) -> Tuple[Optional[str], Tuple[Any, ...]]:
        # Wrapper frames (nn.Module.__call__ and friends) expose the user's arguments as args[i] / kwargs[name]
        if local == 'args' and 'args' not in parameter_names and path and isinstance(path[0], int):
            return (positional_names[path[0]] if path[0] < len(positional_names) else None), path[1:]
        if local == 'kwargs' and 'kwargs' not in parameter_names and path and isinstance(path[0], str):
            return path[0], path[1:]
        return local, path

    for node in gm.graph.nodes:
        if node.op != 'placeholder':
            continue
        grapharg = node.meta.get('grapharg')
        if grapharg is None or grapharg.source is None:
            continue
        resolved = source_path(grapharg.source)
        if resolved is None:
            continue
        local, path = unwrap(*resolved)
        if local is None:
            continue
        entry = spec_for_argument(spec, local)
        for key in path:
            entry = spec_at(entry, key)
        if entry is None:
            continue
        value = node.meta.get('example_value')
        if isinstance(value, torch.Tensor):
            for i, ds in enumerate(dim_specs(entry, value.dim())):
                sym = _bare_symbol(value.shape[i])
                if ds is None or sym is None:
                    continue
                assign(sym.name, ds.name or _default_name(local, path, i))
        elif isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            ds = parse_dim(entry) if entry != ALL else DimSpec(strict=False)
            sym = _bare_symbol(value) if isinstance(value, torch.SymInt) else None
            if ds is not None and sym is not None:
                assign(sym.name, ds.name or _default_name(local, path, None))
    return names
