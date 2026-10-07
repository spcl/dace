# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Lowering context: values flowing through the FX walker and schedule-tree emission helpers.

The frontend emits a :class:`~dace.sdfg.analysis.schedule_tree.treenodes.ScheduleTreeRoot`; the context keeps the
descriptor repository (``containers``), the symbol table, the FX node to value environment, and a stack of scopes
(child lists) into which nodes are emitted. HOP bodies and branches are lowered into nested scopes.
"""

from __future__ import annotations

import contextlib
import dataclasses
import itertools
import keyword
import re
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import sympy
import torch
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

from dace import InterstateEdge, data, dtypes, nodes, subsets, symbolic
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg.analysis.schedule_tree import treenodes as tn

from .dtypes import to_dace_dtype
from .symbols import SymbolTable, SymExpr


class UnsupportedOpError(NotImplementedError):
    def __init__(self, target, hint: str = ""):
        self.target = target
        msg = f"The DaCe TorchDynamo frontend does not support {target}"
        if hint:
            msg += f": {hint}"
        super().__init__(msg)


@dataclasses.dataclass(eq=False)
class TensorValue:
    """A tensor materialized as a DaCe data container (array or view)."""

    name: str
    desc: data.Data
    tshape: Tuple[SymExpr, ...]  #: torch-level shape (empty for rank-0 tensors; the container is then shape (1,))
    tstrides: Tuple[SymExpr, ...]
    torch_dtype: torch.dtype
    device: Any = None
    source: Optional[str] = None  #: 'input', 'const', or None for transients

    @property
    def dtype(self) -> dtypes.typeclass:
        return self.desc.dtype

    @property
    def rank(self) -> int:
        return len(self.tshape)

    @property
    def is_view(self) -> bool:
        return isinstance(self.desc, data.View)

    def full_range(self) -> subsets.Range:
        return subsets.Range.from_array(self.desc)

    def memlet(self) -> Memlet:
        return Memlet.from_array(self.name, self.desc)


@dataclasses.dataclass(eq=False)
class SymValue:
    """A symbolic integer/boolean (``torch.SymInt``) expressed over DaCe symbols."""

    expr: SymExpr


@dataclasses.dataclass(eq=False)
class ConstValue:
    """A Python constant (number, string, dtype, device, FX submodule, ...)."""

    value: Any


@dataclasses.dataclass(eq=False)
class TupleValue:
    items: List[Any]


Value = Union[TensorValue, SymValue, ConstValue, TupleValue, None]


def as_sym(v) -> SymExpr:
    """Coerces a value usable as a size/index into a DaCe symbolic expression."""
    if isinstance(v, SymValue):
        return v.expr
    if isinstance(v, ConstValue):
        v = v.value
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, sympy.Basic)):
        return v
    raise TypeError(f"Expected a symbolic integer, got {type(v).__name__}: {v!r}")


def as_const(v) -> Any:
    if isinstance(v, ConstValue):
        return v.value
    if isinstance(v, SymValue):
        return v.expr
    if isinstance(v, TupleValue):
        return [as_const(i) for i in v.items]
    if isinstance(v, (list, tuple)):
        return type(v)(as_const(i) for i in v)
    return v


_RESERVED = set(keyword.kwlist) | {
    "int",
    "float",
    "double",
    "bool",
    "char",
    "short",
    "long",
    "unsigned",
    "signed",
    "void",
    "auto",
    "new",
    "delete",
    "this",
    "template",
    "typename",
    "class",
    "struct",
    "union",
    "enum",
    "const",
    "static",
    "inline",
    "namespace",
    "using",
    "operator",
    "public",
    "private",
    "protected",
    "virtual",
    "throw",
    "try",
    "catch",
    "default",
    "switch",
    "case",
    "goto",
    "do",
    "extern",
    "register",
    "volatile",
    "sizeof",
    "typedef",
    "friend",
    "mutable",
    "explicit",
    "export",
    "typeid",
    "wchar_t",
    "constexpr",
    "nullptr",
    "main",
}


def sanitize_name(name: str) -> str:
    name = re.sub(r"\W", "_", str(name))
    if not name or name[0].isdigit():
        name = "_" + name
    while name.startswith("__"):
        name = name[1:]
    if name in _RESERVED:
        name = name + "_"
    return name


def _container_shape(tshape: Sequence[SymExpr], tstrides: Sequence[SymExpr]) -> Tuple[Tuple, Tuple]:
    if len(tshape) == 0:
        return (1,), (1,)
    return tuple(tshape), tuple(tstrides)


class LoweringContext:
    """State shared by all lowerings of one compiled graph (and the HOP subgraphs it contains)."""

    def __init__(
        self,
        name: str,
        symtab: SymbolTable,
        options,
        importer=None,
        storage: dtypes.StorageType = dtypes.StorageType.Default,
    ):
        self.name = name
        self.symtab = symtab
        self.options = options
        self.importer = importer
        self.storage = storage
        self.containers: Dict[str, data.Data] = {}
        self.root_children: List[tn.ScheduleTreeNode] = []
        self._scopes: List[List[tn.ScheduleTreeNode]] = [self.root_children]
        self.env: Dict[Any, Value] = {}
        self._names: set = set()
        self._counter = itertools.count()
        #: Compile-time constant containers (name -> (descriptor, value)), see ``ScheduleTreeRoot.constants``
        self.constants: Dict[str, Tuple[data.Data, Any]] = {}
        #: Dynamo's unbacked (data-dependent) symbols the program assigns, by Dynamo name
        self.assigned_symbols: Dict[str, symbolic.symbol] = {}

    # ------------------------------------------------------------------ naming / containers
    def new_name(self, prefix: str) -> str:
        base = sanitize_name(prefix)
        name = base
        i = 0
        while name in self.containers or name in self.symtab.symbols or name in self._names:
            i += 1
            name = f"{base}_{i}"
        self._names.add(name)
        return name

    def add_array(
        self,
        prefix: str,
        tshape: Sequence[SymExpr],
        torch_dtype: torch.dtype,
        tstrides: Optional[Sequence[SymExpr]] = None,
        transient: bool = True,
        device=None,
        source: Optional[str] = None,
        storage: Optional[dtypes.StorageType] = None,
        exact_name: bool = False,
    ) -> TensorValue:
        """
        Registers a new array container. ``tstrides=None`` yields a C-contiguous layout. With ``exact_name``, the
        container is called ``prefix`` verbatim (e.g., ``__return``), which must not be in use.
        """
        tshape = tuple(tshape)
        if tstrides is None:
            tstrides = contiguous_strides(tshape)
        shape, strides = _container_shape(tshape, tstrides)
        if exact_name:
            if prefix in self.containers or prefix in self._names:
                raise ValueError(f"Container name {prefix} is already in use")
            self._names.add(prefix)
            name = prefix
        else:
            name = self.new_name(prefix)
        desc = data.Array(
            to_dace_dtype(torch_dtype), shape, strides=strides, transient=transient, storage=storage or self.storage
        )
        self.containers[name] = desc
        return TensorValue(name, desc, tshape, tuple(tstrides), torch_dtype, device, source)

    def add_tensor_like(
        self, prefix: str, val: torch.Tensor, transient: bool = True, contiguous: bool = False, **kwargs
    ) -> TensorValue:
        """
        Registers a container matching a FakeTensor's shape, dtype and strides.

        Containers keep torch's layout (``val.stride()``) because the strides of every downstream view in the FX graph
        are expressed relative to it. Only pass ``contiguous=True`` for containers no FX view can refer to.
        """
        tshape = self.symtab.shape(val.shape)
        tstrides = None if contiguous else dense_strides(self.symtab.shape(val.stride()), tshape)
        return self.add_array(prefix, tshape, val.dtype, tstrides, transient=transient, device=val.device, **kwargs)

    def add_view(
        self,
        prefix: str,
        base: TensorValue,
        tshape: Sequence[SymExpr],
        tstrides: Sequence[SymExpr],
        torch_dtype: torch.dtype,
    ) -> TensorValue:
        shape, strides = _container_shape(tshape, tstrides)
        name = self.new_name(prefix)
        desc = data.ArrayView(
            to_dace_dtype(torch_dtype), shape, strides=strides, transient=True, storage=base.desc.storage
        )
        self.containers[name] = desc
        return TensorValue(name, desc, tuple(tshape), tuple(tstrides), torch_dtype, base.device)

    # ------------------------------------------------------------------ scopes / emission
    def emit(self, node: tn.ScheduleTreeNode) -> tn.ScheduleTreeNode:
        self._scopes[-1].append(node)
        return node

    @contextlib.contextmanager
    def scope(self, children: List[tn.ScheduleTreeNode]) -> Iterator[List[tn.ScheduleTreeNode]]:
        self._scopes.append(children)
        try:
            yield children
        finally:
            self._scopes.pop()

    # ------------------------------------------------------------------ data-dependent symbols
    def unassigned_unbacked(self, value: Any) -> Optional[sympy.Symbol]:
        """
        If ``value`` (e.g., an output size in ``node.meta['val']``) is one of Dynamo's unbacked symbols that the
        program has not assigned yet, returns it; the lowering that produces it must assign it.
        """
        if not isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            return None
        expr = value.node.expr
        if not isinstance(expr, sympy.Symbol) or self.defines_unbacked(expr.name):
            return None
        return expr if free_unbacked_symbols(value) else None

    def defines_unbacked(self, name: str) -> bool:
        """Whether the program assigns (or substitutes) Dynamo's unbacked symbol ``name``."""
        return name in self.assigned_symbols or name in self.symtab.aliases

    def alias_unbacked(self, unbacked: sympy.Symbol, value: SymExpr) -> SymExpr:
        """Substitutes ``value`` (over other symbols) for an unbacked symbol, e.g., an extent computed from it."""
        self.symtab.aliases[unbacked.name] = value
        return value

    def assign_symbol(
        self,
        unbacked: sympy.Symbol,
        value: Union[str, SymExpr],
        dtype: dtypes.typeclass = dtypes.int64,
        nonnegative: bool = False,
    ) -> symbolic.symbol:
        """
        Defines the DaCe symbol of an unbacked symbol and assigns it ``value`` (an expression over symbols and
        containers) on an interstate edge at the current position of the program.
        """
        symbol = self.symtab.define(unbacked.name, dtype, nonnegative=nonnegative)
        code = value if isinstance(value, str) else symbolic.symstr(value, cpp_mode=False)
        self.emit(
            tn.AssignNode(name=symbol.name, value=CodeBlock(code), edge=InterstateEdge(assignments={symbol.name: code}))
        )
        self.assigned_symbols[unbacked.name] = symbol
        return symbol

    # ------------------------------------------------------------------ emission helpers
    def emit_view(
        self, prefix: str, base: TensorValue, val: torch.Tensor, subset: Optional[subsets.Range] = None
    ) -> TensorValue:
        """
        Emits a view of ``base`` with the shape/strides of FakeTensor ``val``. ``subset`` selects the viewed region of
        ``base`` (defaults to the whole container); its first element determines the view's offset.
        """
        return self.emit_view_raw(
            prefix, base, self.symtab.shape(val.shape), self.symtab.shape(val.stride()), val.dtype, subset
        )

    def emit_view_raw(
        self,
        prefix: str,
        base: TensorValue,
        tshape: Sequence[SymExpr],
        tstrides: Sequence[SymExpr],
        torch_dtype: torch.dtype,
        subset: Optional[subsets.Range] = None,
    ) -> TensorValue:
        view = self.add_view(prefix, base, tshape, tstrides, torch_dtype)
        subset = subset if subset is not None else base.full_range()
        memlet = Memlet(data=base.name, subset=subset, other_subset=view.full_range())
        self.emit(
            tn.ViewNode(target=view.name, source=base.name, memlet=memlet, src_desc=base.desc, view_desc=view.desc)
        )
        return view

    def emit_copy(
        self,
        src: TensorValue,
        dst: TensorValue,
        src_subset: Optional[subsets.Range] = None,
        dst_subset: Optional[subsets.Range] = None,
    ) -> None:
        memlet = Memlet(
            data=src.name,
            subset=src_subset if src_subset is not None else src.full_range(),
            other_subset=dst_subset if dst_subset is not None else dst.full_range(),
        )
        self.emit(tn.CopyNode(target=dst.name, memlet=memlet))

    def emit_tasklet(
        self,
        name: str,
        inputs: Dict[str, Memlet],
        code: str,
        outputs: Dict[str, Memlet],
        language: dtypes.Language = dtypes.Language.Python,
    ) -> tn.TaskletNode:
        tasklet = nodes.Tasklet(sanitize_name(name), set(inputs.keys()), set(outputs.keys()), code, language=language)
        return self.emit(tn.TaskletNode(tasklet, dict(inputs), dict(outputs)))

    def emit_mapped_tasklet(
        self,
        name: str,
        params: Sequence[str],
        ranges: Sequence[Tuple[SymExpr, SymExpr, SymExpr]],
        inputs: Dict[str, Memlet],
        code: str,
        outputs: Dict[str, Memlet],
    ) -> None:
        """Emits ``map params in ranges: tasklet``; without parameters emits a bare tasklet."""
        if len(params) == 0:
            self.emit_tasklet(name, inputs, code, outputs)
            return
        name = sanitize_name(name)
        tasklet = nodes.Tasklet(name, set(inputs.keys()), set(outputs.keys()), code)
        mapnode = nodes.Map(name + "_map", list(params), subsets.Range(list(ranges)))
        self.emit(
            tn.MapScope(node=nodes.MapEntry(mapnode), children=[tn.TaskletNode(tasklet, dict(inputs), dict(outputs))])
        )

    def emit_nested_mapped_tasklet(
        self,
        name: str,
        outer_params: Sequence[str],
        outer_ranges: Sequence[Tuple],
        inner_params: Sequence[str],
        inner_ranges: Sequence[Tuple],
        inputs: Dict[str, Memlet],
        code: str,
        outputs: Dict[str, Memlet],
    ) -> None:
        """Emits ``map outer: map inner: tasklet`` (e.g. a reduction window inside an output-element map)."""
        name = sanitize_name(name)
        tasklet = nodes.Tasklet(name, set(inputs.keys()), set(outputs.keys()), code)
        inner = nodes.Map(
            name + "_inner_map",
            list(inner_params),
            subsets.Range(list(inner_ranges)),
            schedule=dtypes.ScheduleType.Sequential,
        )
        outer = nodes.Map(name + "_map", list(outer_params), subsets.Range(list(outer_ranges)))
        inner_scope = tn.MapScope(
            node=nodes.MapEntry(inner), children=[tn.TaskletNode(tasklet, dict(inputs), dict(outputs))]
        )
        self.emit(tn.MapScope(node=nodes.MapEntry(outer), children=[inner_scope]))

    def emit_library_call(
        self, node: nodes.LibraryNode, in_memlets: Dict[str, Memlet], out_memlets: Dict[str, Memlet]
    ) -> None:
        self.emit(tn.LibraryCall(node=node, in_memlets=dict(in_memlets), out_memlets=dict(out_memlets)))

    # ------------------------------------------------------------------ misc
    def unique(self, prefix: str) -> str:
        return f"{sanitize_name(prefix)}_{next(self._counter)}"


def contiguous_strides(shape: Sequence[SymExpr]) -> Tuple[SymExpr, ...]:
    strides = []
    acc: SymExpr = 1
    for s in reversed(tuple(shape)):
        strides.append(acc)
        acc = acc * s
    return tuple(reversed(strides))


def dense_strides(tstrides: Sequence[SymExpr], tshape: Sequence[SymExpr]) -> Optional[Tuple[SymExpr, ...]]:
    """
    Returns ``tstrides`` if they describe a dense (non-overlapping) layout usable for a freshly allocated container,
    otherwise ``None`` (meaning: use a contiguous layout). Broadcast (stride 0) layouts are not dense.
    """
    for s in tstrides:
        if (isinstance(s, int) and s == 0) or (isinstance(s, sympy.Basic) and s == 0):
            return None
    return tuple(tstrides)


def index_memlet(name: str, indices: Sequence[Union[str, SymExpr]]) -> Memlet:
    """Builds a single-element memlet ``name[i0, i1, ...]``."""
    idx = ", ".join(str(i) for i in indices) if len(indices) > 0 else "0"
    return Memlet.simple(name, idx)
