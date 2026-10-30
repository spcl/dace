# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Capture of data-dependent Python control flow as a control-flow graph of traced blocks.

At the first conditional jump on tensor data in a frame (the *region entry*), the rest of the frame is split into
blocks at its leaders (:meth:`.analysis.CodeInfo.leaders`). Every block is traced once by Dynamo as a subgraph
(``speculate_subgraph``) of a continuation function that starts at the leader and takes the block's live variables.
A block ends when control reaches another leader (an unconditional edge), a conditional jump on tensor data (a
branch), or a return. Blocks are traced from a worklist: tracing a block yields its successors and the values that
flow into them, so loops, joins, ``break``, ``continue``, ``else`` clauses of loops, and early returns are all just
edges. Whatever layout CPython chose for the bytecode, the graph is the same up to the leaders.

The frame then calls the opaque operator ``dace::cfg`` (:mod:`.transport`) with the values the blocks read and
returns its result. The DaCe importer lowers the recorded graph to schedule-tree control flow.

Variables that hold tensors or symbolic integers are block inputs. Lists, tuples and dicts (with constant keys) of
such values are flattened: their leaves are block inputs and the continuation rebuilds the container before it jumps
to the leader (:func:`_flatten`). Cell variables (locals captured by nested functions or comprehensions) are passed by
value like other locals. Other live values (Python constants, modules, other objects) are bound to the continuation as
default arguments and specialize the block: a block start is traced once per distinct set of such values, container
structures and input metadata, up to :data:`MAX_SPECIALIZATIONS`, after which the capture falls back to Dynamo's graph
break.

For loops over ranges and tensors are captured as loops instead of being unrolled. Their state (the sequence and the
number of items taken) are variables of the graph, and a placeholder takes the iterator's place on the stack, so that
blocks can start inside loop bodies. ``FOR_ITER`` is a branch on whether another item exists; its taken edge enters
the same instruction again, which then pushes the item. At loop headers, integers the loop assigns are *widened* to
unbacked symbols, so that the body is traced once for all iterations; branches on such symbols (e.g., the loop
condition) are captured like branches on tensor data. A for loop is captured from its ``GET_ITER`` when its trip count
is symbolic (``range(x.shape[0])``, a tensor), when the ``capture_loops`` option asks for every loop with at least two
iterations, or, after a restart of the analysis, when its body has data-dependent control flow.
"""

import dataclasses
import dis
import functools
import operator
import sys
import threading
import types
from collections.abc import Callable, Iterator
from typing import Any, Optional

import sympy
import torch
from torch._dynamo import bytecode_transformation as bt
from torch._dynamo import exc
from torch._dynamo import symbolic_convert as sc
from torch._dynamo.exc import RestartAnalysis
from torch._dynamo.variables import (
    BuiltinVariable,
    ConstantVariable,
    ConstDictVariable,
    ListVariable,
    RangeVariable,
    SliceVariable,
    SymNodeVariable,
    TensorVariable,
    TupleVariable,
    VariableTracker,
)
from torch._dynamo.variables.builder import wrap_fx_proxy
from torch._dynamo.variables.functions import NestedUserFunctionVariable
from torch._dynamo.variables.higher_order_ops import speculate_subgraph
from torch._dynamo.variables.misc import CellVariable
from torch.fx.experimental.symbolic_shapes import free_symbols, free_unbacked_symbols
from torch.utils._sympy.numbers import int_oo
from torch.utils._sympy.value_ranges import bound_sympy

from dace.frontend.ml.torch.dynamo.backend import CAPTURE_CONFIG, DaceBackend

from . import transport
from .analysis import JUMP_ON_TRUTH, CodeInfo
from .transport import BRANCH, GOTO, RETURN, Binding, BlockRecord, CfgRecord

#: Traces of one block start (for different constants or input metadata) before giving up
MAX_SPECIALIZATIONS = 4

#: Values of the ``capture_loops`` option: capture for loops whose trip count is symbolic (or whose body needs it),
#: or every for loop over a range or tensor with at least two iterations
CAPTURE_LOOPS = ("symbolic", "all")


class CaptureFallback(Exception):
    """The control flow cannot be captured; Dynamo's stock behavior (a graph break) applies instead."""


# ---------------------------------------------------------------------------------------------- values at boundaries
#: Kinds of nodes of the structure (``spec``) of a variable's value: a leaf (a block input or a constant), a container
#: of children, a range (the sequence of a loop: its bounds are the leaves), or a function defined in the frame
#: (rebuilt from its code and the frame's cells)
LEAF, LIST, TUPLE, DICT, RANGE, FUNCTION = "leaf", "list", "tuple", "dict", "range", "function"

#: Python constants that can be rebuilt by ``LOAD_CONST`` in a continuation
_LITERAL_TYPES = (int, float, bool, str, bytes, type(None), type(Ellipsis))

#: Prefix of the hidden block inputs that carry a loop-counter symbol a tensor's shape depends on
_SYMBOL_PREFIX = "__dace_sym_"


def _flatten(vt, cells: frozenset = frozenset()) -> tuple[Any, list] | None:
    """
    The structure of a variable's value and its leaves (graph values and constants), or ``None`` for values that
    cannot be flattened (modules, other objects): lists, tuples and dicts with constant keys are flattened
    recursively. A function defined in the frame (a lambda or ``def``) whose closure consists of the frame's cells
    (``cells``) is rebuilt from its code and its default arguments, which are its leaves.
    """
    vt = vt.realize() if hasattr(vt, "realize") else vt
    if _is_graph_value(vt):
        return (LEAF,), [vt]
    if isinstance(vt, ConstantVariable) and isinstance(vt.as_python_constant(), _LITERAL_TYPES):
        return (LEAF,), [vt]
    if type(vt) in (ListVariable, TupleVariable, RangeVariable):
        specs, leaves = [], []
        for item in vt.items:
            flat = _flatten(item, cells)
            if flat is None:
                return None
            specs.append(flat[0])
            leaves.extend(flat[1])
        kind = {ListVariable: LIST, TupleVariable: TUPLE, RangeVariable: RANGE}[type(vt)]
        return (kind, tuple(specs)), leaves
    if type(vt) is ConstDictVariable and vt.user_cls is dict:
        entries, leaves = [], []
        for key, item in vt.items.items():
            key_vt = getattr(key, "vt", key)
            if not key_vt.is_python_constant():
                return None
            key_value = key_vt.as_python_constant()
            flat = _flatten(item, cells)
            if flat is None or not isinstance(key_value, _LITERAL_TYPES):
                return None
            entries.append((key_value, flat[0]))
            leaves.extend(flat[1])
        return (DICT, tuple(entries)), leaves
    if type(vt) is NestedUserFunctionVariable and vt.closure is not None:
        # The cells of a closure made in a block do not survive the block (its side effects are discarded once it
        # is traced); the successor remakes the function from the code with its own cells of the same names
        code = vt.get_code()
        if (
            getattr(vt, "kwdefaults", None) is not None
            or getattr(vt, "annotations", None) is not None
            or not set(code.co_freevars) <= cells
        ):
            return None
        flat = _flatten(vt.defaults, cells) if vt.defaults is not None else ((TUPLE, ()), [])
        if flat is None:
            return None
        return (FUNCTION, code, flat[0]), flat[1]
    return None


def _unflatten(spec, leaves: Iterator) -> VariableTracker:
    """Rebuilds a value from its structure and an iterator over its leaves (as variable trackers)."""
    kind = spec[0]
    if kind == LEAF:
        return next(leaves)
    if kind in (LIST, TUPLE, RANGE):
        items = [_unflatten(child, leaves) for child in spec[1]]
        return {LIST: ListVariable, TUPLE: TupleVariable, RANGE: RangeVariable}[kind](items)
    if kind == FUNCTION:  # A stand-in with the function's code and defaults (its cells are the successor's)
        code, defaults_spec = spec[1], spec[2]
        defaults = _unflatten(defaults_spec, leaves) if defaults_spec[1] else None
        return NestedUserFunctionVariable(
            ConstantVariable.create(code.co_qualname),
            ConstantVariable.create(code),
            {},
            defaults,
            None,
            TupleVariable([]),
        )
    return ConstDictVariable({ConstantVariable.create(k): _unflatten(child, leaves) for k, child in spec[1]}, dict)


def _build_value(tx, spec, leaves: Iterator) -> VariableTracker:
    """Like :func:`_unflatten`, for values that are handed back to the frame (built like ``BUILD_LIST`` does)."""
    kind = spec[0]
    if kind == LEAF:
        return next(leaves)
    if kind in (LIST, TUPLE):
        items = [_build_value(tx, child, leaves) for child in spec[1]]
        return VariableTracker.build(tx, items if kind == LIST else tuple(items))
    if kind == RANGE:
        return RangeVariable([_build_value(tx, child, leaves) for child in spec[1]])
    if kind == FUNCTION:
        raise CaptureFallback("the frame returns a function")
    return VariableTracker.build(
        tx, {ConstantVariable.create(k): _build_value(tx, child, leaves) for k, child in spec[1]}
    )


def _leaf_count(spec) -> int:
    if spec[0] == LEAF:
        return 1
    if spec[0] == FUNCTION:
        return _leaf_count(spec[2])
    return sum(_leaf_count(child if spec[0] != DICT else child[1]) for child in spec[1])


def _leaf_names(local: str, spec, taken: set[str]) -> list[str]:
    """
    The parameter names of a local's leaves: the local's own name for a bare leaf, ``<local>__<k>`` for the k-th
    leaf of a container (made unique against ``taken``). Deterministic, so that the block that produces a value and
    the block that consumes it agree on the names.
    """
    if spec[0] == LEAF:
        return [local]
    names = []
    for k in range(_leaf_count(spec)):
        name = f"{local}__{k}"
        while name in taken:
            name += "_"
        names.append(name)
    return names


@dataclasses.dataclass
class _Tree:
    """A live local at a block start: its structure and its leaves."""

    local: str
    spec: Any
    leaf_names: list[str]
    leaves: list[Any]  #: Variable trackers: graph values (parameters) or constants (rebuilt by ``LOAD_CONST``)

    def parameters(self) -> list[str]:
        return [name for name, leaf in zip(self.leaf_names, self.leaves) if _is_graph_value(leaf)]

    def builder(self) -> list[tuple[str, Any]]:
        """The leaves as the continuation's prefix needs them: ``('param', name)`` or ``('const', value)``."""
        return [
            ("param", name) if _is_graph_value(leaf) else ("const", leaf.as_python_constant())
            for name, leaf in zip(self.leaf_names, self.leaves)
        ]

    def needs_builder(self) -> bool:
        """Whether the continuation has to build the local (a container, a function, or a constant)."""
        return self.spec[0] != LEAF or not _is_graph_value(self.leaves[0])

    def symbols(self, unbacked: bool = True) -> set:
        """The symbols the shapes of the tree's tensor leaves depend on: unbacked ones (loop counters), or all."""
        result: set = set()
        for leaf in self.leaves:
            if isinstance(leaf, TensorVariable):
                example = _example(leaf)
                if unbacked:
                    result |= set(free_unbacked_symbols(example.size())) | set(free_unbacked_symbols(example.stride()))
                else:
                    result |= set(free_symbols(example.size())) | set(free_symbols(example.stride()))
        return result

    def provides(self) -> set:
        """The symbols the tree's symbolic-integer leaves bind (bare symbols only)."""
        return {leaf.sym_num.node.expr for leaf in self.leaves if _is_symint(leaf) and leaf.sym_num.node.expr.is_Symbol}


@dataclasses.dataclass
class _Exit:
    kind: str
    targets: dict[Any, int]  #: Leader index by edge label (``None`` for goto, truth value for branch)
    names: list[str]  #: Leaf variables passed on as graph outputs (after the predicate for a branch)
    trees: list[_Tree] = dataclasses.field(default_factory=list)  #: Flattened live locals
    side: dict[str, Any] = dataclasses.field(default_factory=dict)  #: Other live values passed on (not graph outputs)
    extra: dict[Any, dict[str, Any]] = dataclasses.field(default_factory=dict)  #: Side values for one edge label
    return_spec: Any = None  #: Structure of the frame's return value


@dataclasses.dataclass
class _Active:
    """The block being traced."""

    code: types.CodeType  #: Continuation code object
    prefix: int  #: Number of instructions prepended to the original code
    start: int
    #: Widened symbols whose value is known in this block (it is only entered with that constant, e.g., the counter
    #: of a loop at its entry), by sympy symbol
    known: dict[Any, int] = dataclasses.field(default_factory=dict)
    entered: bool = False
    exit: _Exit | None = None


class Region:
    """The capture of one frame region, from its entry jump to every return reachable from it."""

    def __init__(self, tx, info: CodeInfo, entry_targets: dict[Any, int], symbolic_branches: bool = False):
        """
        :param entry_targets: The instructions control enters the region at: by truth value of the entry predicate
                              (a branch), or under ``None`` (a goto, e.g., into a loop).
        :param symbolic_branches: Keep relations between sizes symbolic instead of guarding on them (see
                                  :class:`ControlFlowCapture`).
        """
        self.tx = tx
        self.info = info
        self.symbolic_branches = symbolic_branches
        self.entry_targets: dict[Any, int] = dict(entry_targets)
        self.leaders = info.leaders(list(entry_targets.values()))
        region = info.region(list(entry_targets.values()))
        depths = info.stack_depths()
        for leader in self.leaders:
            # Only iterators of for loops may be on the stack; blocks start with placeholders for them
            if depths.get(leader, -1) != info.iterators(leader):
                raise CaptureFallback("a block would start with values on the stack")
        if info.has_exception_handlers(region):
            raise CaptureFallback("the control flow is inside a try or with block")
        self.active: _Active | None = None
        self.blocks: list[BlockRecord] = []
        self._keys: dict[tuple, int] = {}
        self._starts: dict[int, int] = {}
        self._incoming: dict[int, dict[str, Any]] = {}  #: Block inputs (by parameter name) and side values
        self._trees: dict[int, list[_Tree]] = {}  #: Live locals of each block, flattened
        self._known: dict[int, dict[Any, int]] = {}  #: Widened symbols with known values, per block
        self._outer_tensors: list[Any] = []  #: Outer proxies of the operator's tensor inputs
        self._outer_syms: list[Any] = []  #: Outer proxies of the operator's SymInt inputs
        self.return_examples: list[Any] | None = None
        self.return_spec: Any = None
        self._widened: dict[tuple[int, str], SymNodeVariable] = {}
        self._nonnegative_assumed: set = set()
        self.entry_constants: dict[str, Any] = {}

    # ------------------------------------------------------------------------------------------ operator inputs
    def bind_outer(self, proxy) -> Binding:
        """The operator input for an outer (frame-level) proxy, added if new."""
        example = proxy.node.meta.get("example_value")
        is_sym = isinstance(example, (torch.SymInt, torch.SymBool))
        values = self._outer_syms if is_sym else self._outer_tensors
        for index, existing in enumerate(values):
            if existing.node is proxy.node:
                return Binding("sym" if is_sym else "tensor", index)
        values.append(proxy)
        return Binding("sym" if is_sym else "tensor", len(values) - 1)

    # ------------------------------------------------------------------------------------------ blocks
    def live(self, start: int, available: dict[str, Any]) -> list[str]:
        """
        The variables live at ``start``: user variables, the state of the loops ``start`` is in, and the cells that
        functions among them close over.
        """
        loops = self.info.loops_containing(start)
        names = set(self.info.livevars(start))
        for loop in loops:
            names.update((_seq_name(loop), _index_name(loop)))
        if start in loops:
            names.add(_take_name(start))
        names = {name for name in names if name in available}
        work = list(names)
        while work:  # A function defined in the frame reads the frame's cells (its closure)
            value = available[work.pop()]
            if type(value) is NestedUserFunctionVariable and value.closure is not None:
                for name in value.get_code().co_freevars:
                    if name in available and name not in names:
                        names.add(name)
                        work.append(name)
        return sorted(names)

    def widen(self, start: int, trees: list[_Tree]) -> dict[Any, int]:
        """
        At a loop header, integers the loop assigns become unbacked symbols: the loop body is traced once for all
        iterations instead of once per value. Returns the symbols that replaced constants, with the constants: a
        block entered with them (the loop's entry) knows their values, which decides the loop's first test
        (:meth:`decide`) without an edge for the case that the loop does not run.
        """
        known: dict[Any, int] = {}
        if not self.info.is_loop_header(start):
            return known
        carried = self.info.stored_names(self.info.loop_body(start)) | {_index_name(start)}
        for tree in trees:
            if tree.local not in carried:
                continue
            for k, (name, value) in enumerate(zip(tree.leaf_names, tree.leaves)):
                if not (_is_int_constant(value) or _is_symint(value)):
                    continue
                widened = self._widened_symbol(start, name, nonnegative=self._nonnegative(value))
                if (start, name) in self._nonnegative_assumed and value is not widened and not self._nonnegative(value):
                    # The assumption made when widening does not hold on this edge (checked on every edge, this
                    # proves it by induction)
                    raise CaptureFallback(f"loop variable {name} may become negative")
                if _is_int_constant(value):
                    known[widened.sym_num.node.expr] = value.as_python_constant()
                tree.leaves[k] = widened
        return known

    def decide(self, predicate) -> bool | None:
        """
        The value of a symbolic predicate in the block being traced if it is known: its data-dependent symbols are
        all known in the block (:attr:`_Active.known`), and what remains is constant or a relation between sizes that
        Dynamo may guard on (unless ``symbolic_branches`` keeps those symbolic). ``None`` otherwise.
        """
        if not isinstance(predicate, SymNodeVariable) or not isinstance(predicate.sym_num, torch.SymBool):
            return None
        expr = predicate.sym_num.node.expr
        if self.active.known:
            expr = expr.xreplace(self.active.known)
        if free_unbacked_symbols(expr):
            return None
        if expr in (sympy.true, sympy.false):
            return bool(expr)
        if self.symbolic_branches or not expr.free_symbols:
            return None
        try:
            return bool(self.tx.output.shape_env.evaluate_expr(expr))  # Guards on the relation
        except Exception:  # noqa: BLE001 - Dynamo could not decide (e.g., data-dependent after all)
            return None

    def _nonnegative(self, value) -> bool:
        if _is_int_constant(value):
            return value.as_python_constant() >= 0
        expr = value.sym_num.node.expr
        return bool(bound_sympy(expr, self.tx.output.shape_env.var_to_range).lower >= 0)

    def _widened_symbol(self, start: int, name: str, nonnegative: bool = False) -> SymNodeVariable:
        key = (start, name)
        if key not in self._widened:
            shape_env = self.tx.output.shape_env
            with shape_env.ignore_fresh_unbacked_symbols():
                symbol = shape_env.create_unbacked_symint()
            if nonnegative or name.startswith(_INDEX_PREFIX):
                # Assumed when the first value is nonnegative, and checked on every other edge into the loop. Bounds
                # make indexing with the variable possible (Dynamo needs to know the sign of an index).
                shape_env.constrain_symbol_range(symbol.node.expr, compiler_min=0, compiler_max=int_oo)
                self._nonnegative_assumed.add(key)
            # Only the example value of a block input's node is used (the CFG binds the input along its edges)
            graph = torch.fx.Graph()
            node = graph.placeholder(f"widened_{start}_{name.strip('_')}")
            node.meta["example_value"] = symbol
            self._widened[key] = SymNodeVariable(torch.fx.Proxy(node), symbol)
        return self._widened[key]

    def flatten_locals(
        self, names: list[str], values: dict[str, Any], clean: bool = False
    ) -> tuple[list[_Tree], dict[str, Any]]:
        """
        Splits locals into flattened values (:class:`_Tree`) and other objects (``name -> value``). With ``clean``,
        tensor leaves become block inputs (:func:`_block_input`).
        """
        taken = set(self.info.code.co_varnames)
        cells = frozenset(self.info.code.co_cellvars) | frozenset(self.info.code.co_freevars)
        trees, objects = [], {}
        for name in names:
            flat = _flatten(values[name], cells)
            if flat is None:
                objects[name] = values[name]
                continue
            spec, leaves = flat
            if clean:
                leaves = [_block_input(leaf) for leaf in leaves]
            trees.append(_Tree(name, spec, _leaf_names(name, spec, taken), list(leaves)))
        return trees, objects

    def bind_symbols(self, trees: list[_Tree], available: dict[str, Any], translator=None) -> None:
        """
        Adds the hidden inputs that carry the symbols the shapes of tensor leaves depend on, which a block must bind
        before such a tensor: loop-counter symbols (a ragged chunk ``z[i:i+c]`` is sized by the counter; at a block
        exit, ``translator`` given, the symbol's value in the block is passed on; at a block start it comes in from
        the predecessor under its hidden name), and, at a block start, sizes of the frame that only such tensors
        mention (Dynamo lifts the symbols of a shape through the whole expression, so every symbol in it has to be
        bound by then).
        """
        provided = set().union(*(tree.provides() for tree in trees)) if trees else set()
        needed = set().union(*(tree.symbols() for tree in trees)) if trees else set()
        for symbol in sorted(needed - provided, key=str):
            name = f"{_SYMBOL_PREFIX}{symbol}"
            value = available.get(name)
            if value is None and translator is not None:
                value = self._symbol_in_block(translator, symbol)
            if value is None or not _is_symint(value):
                raise CaptureFallback(
                    f"a tensor whose shape depends on the data-dependent size {symbol} flows to where that size is "
                    "not available (e.g., across the loop's back edge)"
                )
            trees.append(_Tree(name, (LEAF,), [name], [value]))
        if translator is not None:
            return
        widened = {vt.sym_num.node.expr for vt in self._widened.values()}
        ragged = [tree for tree in trees if tree.symbols()]
        bound = set().union(*(tree.symbols(unbacked=False) for tree in trees if tree not in ragged)) if trees else set()
        provided = set().union(*(tree.provides() for tree in trees)) if trees else set()
        for tree in ragged:
            for symbol in sorted(tree.symbols(unbacked=False) - needed - widened - bound - provided, key=str):
                name = f"{_SYMBOL_PREFIX}{symbol}"
                value = self._frame_symbol(symbol)
                if value is None:
                    raise CaptureFallback(f"the size {symbol} of a chunk is not a size of the frame")
                trees.append(_Tree(name, (LEAF,), [name], [value]))
                provided.add(symbol)

    def _symbol_in_block(self, translator, symbol) -> SymNodeVariable | None:
        """
        The block's own value of a data-dependent symbol: a widened loop counter bound as an input of the block, or
        a size the block computed (an ``item()``).
        """
        tracer = translator.output.current_tracer
        proxy = tracer.bound_symbols.get(symbol)
        if proxy is None:
            return None
        widened = next((vt for vt in self._widened.values() if vt.sym_num.node.expr == symbol), None)
        if widened is not None:
            return SymNodeVariable.create(translator, proxy, widened.sym_num)
        shape_env = translator.output.shape_env
        hint = shape_env.unbacked_var_to_val.get(symbol) if hasattr(shape_env, "unbacked_var_to_val") else None
        return SymNodeVariable.create(translator, proxy, shape_env.create_symintnode(symbol, hint=hint))

    def chunk_nonempty(self, translator, start, size) -> bool:
        """
        Whether a chunk ``[start:...]`` of a dimension of ``size`` elements has at least one element because the
        enclosing captured loop guarantees it: ``start`` is the loop's current item and ``size`` its bound (the loop
        runs while ``item < bound``).
        """
        frame = translator
        while frame is not None and not self.is_block_translator(frame):
            frame = getattr(frame, "parent", None)
        if frame is None:
            return False
        start_expr, size_expr = _sym_expr(start), _sym_expr(size)
        if start_expr is None or size_expr is None:
            return False
        for header in self.info.loops_containing(self.instruction_index(frame)):
            sequence = frame.symbolic_locals.get(_seq_name(header))
            index = self._widened.get((header, _index_name(header)))
            if not isinstance(sequence, RangeVariable) or index is None:
                continue
            first, stop, step = (_sym_expr(item) for item in sequence.items)
            if first is None or stop is None or step is None:
                continue
            item = first + index.sym_num.node.expr * step
            if sympy.simplify(start_expr - item) == 0 and sympy.simplify(size_expr - stop) == 0:
                return True
        return False

    def _frame_symbol(self, symbol) -> SymNodeVariable | None:
        """A size symbol of the frame (bound by its inputs), as a value a block can take as an input."""
        proxy = self.tx.output.root_tracer.bound_symbols.get(symbol)
        if proxy is None:
            return None
        shape_env = self.tx.output.shape_env
        sym_num = shape_env.create_symintnode(symbol, hint=shape_env.var_to_val.get(symbol))
        return SymNodeVariable.create(self.tx, proxy, sym_num)

    def block_for(self, start: int, incoming: dict[str, Any]) -> tuple[int, bool]:
        """The block id for ``start`` given the incoming variable values; ``True`` if it still has to be traced."""
        live = self.live(start, incoming)
        trees, objects = self.flatten_locals(live, incoming, clean=True)
        known = self.widen(start, trees)
        self.bind_symbols(trees, incoming)
        key = (
            start,
            tuple((t.local, t.spec, tuple(_signature(leaf) for leaf in t.leaves)) for t in trees),
            tuple((name, _signature(vt)) for name, vt in objects.items()),
            tuple(sorted((str(symbol), value) for symbol, value in known.items())),
        )
        if key in self._keys:
            return self._keys[key], False
        self._starts[start] = self._starts.get(start, 0) + 1
        if self._starts[start] > MAX_SPECIALIZATIONS:
            raise CaptureFallback(
                f"block at instruction {start} needs more than {MAX_SPECIALIZATIONS} specializations "
                "(e.g., a Python value that changes in a loop)"
            )
        block_id = len(self.blocks)
        self._keys[key] = block_id
        # Symbolic integers first, then tensors whose shapes are sizes of the frame, then tensors sized by loop
        # counters: Dynamo binds the symbols of an input's shape when it meets them, in this order
        leaves = [(name, leaf) for tree in trees for name, leaf in zip(tree.leaf_names, tree.leaves)]
        inputs: dict[str, Any] = {name: leaf for name, leaf in leaves if isinstance(leaf, SymNodeVariable)}
        tensors = [(name, leaf) for name, leaf in leaves if isinstance(leaf, TensorVariable)]
        inputs.update({name: leaf for name, leaf in tensors if not _ragged(leaf)})
        inputs.update({name: leaf for name, leaf in tensors if _ragged(leaf)})
        parameters = list(inputs)
        examples = [_example(inputs[name]) for name in parameters]
        self.blocks.append(BlockRecord(block_id, start, None, parameters, examples))
        inputs.update(objects)
        self._incoming[block_id] = inputs
        self._trees[block_id] = trees
        self._known[block_id] = known
        return block_id, True

    def trace(self, entry_values: dict[str, Any]) -> None:
        """Traces all blocks reachable from the entry and registers the control-flow graph."""
        work = []
        entry_successors = {}
        for label, target in self.entry_targets.items():
            block_id, new = self.block_for(target, entry_values)
            entry_successors[label] = block_id
            if new:
                work.append(block_id)
        while work:
            block_id = work.pop(0)
            for label, (successor, new) in self._trace_block(block_id).items():
                if new:
                    work.append(successor)

        # Entry bindings: the frame-level values of the variables the entry blocks read (sizes of the frame that
        # hidden inputs carry need none: the lowering knows them)
        entry_bindings = {}
        for block_id in set(entry_successors.values()):
            for name in self.blocks[block_id].input_names:
                if name.startswith(_SYMBOL_PREFIX):
                    continue
                value = self._entry_leaf(block_id, name, entry_values)
                if _is_graph_value(value):
                    entry_bindings[name] = self.bind_outer(value.as_proxy())
                else:  # A Python integer for a symbolic (widened) input
                    self.entry_constants[name] = value.as_python_constant()
        if self.return_examples is None:
            raise CaptureFallback("no path returns from the frame")
        self.entry_bindings = entry_bindings
        self.entry_successors = entry_successors

    def _entry_leaf(self, block_id: int, name: str, entry_values: dict[str, Any]):
        """The frame-level value of a block input (a leaf of a live local at the entry, before widening)."""
        cells = frozenset(self.info.code.co_cellvars) | frozenset(self.info.code.co_freevars)
        for tree in self._trees[block_id]:
            if name in tree.leaf_names:
                _, leaves = _flatten(entry_values[tree.local], cells)
                return leaves[tree.leaf_names.index(name)]
        raise CaptureFallback(f"block input {name} has no value at the entry")

    def _trace_block(self, block_id: int) -> dict[Any, tuple[int, bool]]:
        tx = self.tx
        block = self.blocks[block_id]
        incoming = self._incoming[block_id]
        others = [name for name in incoming if name not in block.input_names]
        markers = self.info.iterators(block.start)
        builders = [(t.local, t.spec, t.builder()) for t in self._trees[block_id] if t.needs_builder()]
        code, prefix = _continuation(self.info, block.start, block.input_names + others, markers, builders)
        function = NestedUserFunctionVariable(
            ConstantVariable.create(code.co_name),
            ConstantVariable.create(code),
            tx.f_globals,
            TupleVariable([incoming[name] for name in others]) if others else None,
            None,
            None,
            None,
        )
        args = [incoming[name] for name in block.input_names]
        previous = self.active
        self.active = _Active(code, prefix, block.start, dict(self._known[block_id]))
        try:
            (output, _), graph, lifted = speculate_subgraph(
                tx,
                function,
                args,
                {},
                f"dace_cfg_block_{block_id}",
                set_subgraph_inputs="flatten_manual",
                should_flatten_outputs=True,
            )
            block_exit = self.active.exit
        finally:
            self.active = previous
        if block_exit is None:
            raise CaptureFallback(f"block at instruction {block.start} ended without a recognized exit")

        # Placeholders after the block inputs are values of the frame (lifted free variables)
        placeholders = [node for node in graph.nodes if node.op == "placeholder"]
        inner_to_outer = {inner.node: outer for outer, inner in lifted.items()}
        block.graph = graph
        block.lifted = [self.bind_outer(inner_to_outer[node]) for node in placeholders[len(block.input_names) :]]
        block.exit_kind = block_exit.kind
        block.output_names = list(block_exit.names)

        outputs = list(output.unpack_var_sequence(tx)) if hasattr(output, "unpack_var_sequence") else list(output.items)
        if block_exit.kind == RETURN:
            examples = [_example(vt) for vt in outputs]
            if self.return_examples is None:
                self.return_examples, self.return_spec = examples, block_exit.return_spec
            elif self.return_spec != block_exit.return_spec or [_metadata(e) for e in examples] != [
                _metadata(e) for e in self.return_examples
            ]:
                raise CaptureFallback("the frame returns values of different types or shapes on different paths")
            block.successors = None
            return {}

        values = dict(zip(block_exit.names, outputs[1:] if block_exit.kind == BRANCH else outputs))
        passed = dict(block_exit.side)
        for tree in block_exit.trees:
            leaves = iter(
                values[name] if _is_graph_value(leaf) else leaf for name, leaf in zip(tree.leaf_names, tree.leaves)
            )
            passed[tree.local] = _unflatten(tree.spec, leaves)
        block.output_constants = {
            name: leaf.as_python_constant()
            for tree in block_exit.trees
            for name, leaf in zip(tree.leaf_names, tree.leaves)
            if _is_int_constant(leaf)
        }
        successors = {}
        for label, target in block_exit.targets.items():
            successors[label] = self.block_for(target, {**passed, **block_exit.extra.get(label, {})})
        block.successors = (
            successors[None][0]
            if block_exit.kind == GOTO
            else {label: successor for label, (successor, _) in successors.items()}
        )
        return successors

    # ------------------------------------------------------------------------------------------ exits
    def end_block(
        self,
        translator,
        kind: str,
        targets: dict[Any, int],
        predicate=None,
        value=None,
        extra: dict[Any, dict[str, Any]] | None = None,
    ) -> None:
        """
        Ends the block being traced by returning the values its successors need from the continuation.

        :param extra: Python values passed on one edge only, by edge label (e.g., the "take" flag of a loop).
        """
        if kind == RETURN:
            flat = _flatten(value)
            if flat is None or not all(isinstance(leaf, TensorVariable) for leaf in flat[1]):
                raise CaptureFallback(f"the frame returns a {type(value).__name__} that is not made of tensors")
            spec, leaves = flat
            self.active.exit = _Exit(RETURN, {}, [], return_spec=spec)
            _return_from(translator, TupleVariable(leaves))
            return
        available = _locals_of(translator)
        live = sorted({name for target in targets.values() for name in self.live(target, available)})
        trees, side = self.flatten_locals(live, available)
        self.bind_symbols(trees, available, translator)
        names = [name for tree in trees for name in tree.parameters()]
        graph_values = [leaf for tree in trees for leaf in tree.leaves if _is_graph_value(leaf)]
        self.active.exit = _Exit(kind, dict(targets), names, trees=trees, side=side, extra=dict(extra or {}))
        _return_from(translator, TupleVariable(([predicate] if kind == BRANCH else []) + graph_values))

    def is_block_translator(self, translator) -> bool:
        return self.active is not None and translator.f_code is self.active.code

    def instruction_index(self, translator) -> int:
        return translator.instruction_pointer - self.active.prefix

    # ------------------------------------------------------------------------------------------ result
    def emit(self) -> Any:
        """Registers the graph and emits the ``dace::cfg`` call in the frame; returns the frame's return value."""
        tx = self.tx
        predicate_binding = self._predicate_binding
        record = transport.register(
            lambda cfg_id: CfgRecord(
                cfg_id,
                self.blocks,
                predicate_binding,
                self.entry_successors,
                self.entry_bindings,
                self.return_examples,
                tx.f_code.co_name,
                self.entry_constants,
                self._predicate_expr,
            )
        )
        proxy = tx.output.create_proxy(
            "call_function",
            torch.ops.dace.cfg.default,
            (record.id, list(self._outer_tensors), list(self._outer_syms)),
            {},
        )
        result = wrap_fx_proxy(tx, proxy, example_value=list(self.return_examples))
        items = list(result.unpack_var_sequence(tx)) if hasattr(result, "unpack_var_sequence") else list(result.items)
        return _build_value(tx, self.return_spec, iter(items))

    def capture(self, predicate, entry_values: dict[str, Any]) -> Any:
        """
        Traces the region and returns the frame's return value (the result of the ``dace::cfg`` call).

        :param predicate: The entry branch's predicate, or ``None`` for a goto.
        :param entry_values: The variables at the entry.
        """
        self._predicate_binding, self._predicate_expr = None, None
        if isinstance(predicate, SymNodeVariable):  # Over symbols of the frame, which the SDFG knows
            self._predicate_expr = predicate.sym_num.node.expr
        elif predicate is not None:
            self._predicate_binding = self.bind_outer(predicate.as_proxy())
        self.trace(entry_values)
        return self.emit()


# ---------------------------------------------------------------------------------------------- helpers
#: Prefixes of the variables that hold the state of a captured for loop (suffixed with the loop's FOR_ITER index)
_SEQUENCE_PREFIX, _INDEX_PREFIX, _TAKE_PREFIX = "__dace_seq_", "__dace_index_", "__dace_take_"


def _seq_name(loop: int) -> str:
    """The iterated sequence (a tensor, or a range)."""
    return f"{_SEQUENCE_PREFIX}{loop}"


def _index_name(loop: int) -> str:
    """The number of items taken so far."""
    return f"{_INDEX_PREFIX}{loop}"


def _take_name(loop: int) -> str:
    """Set on the edge into the loop body: the FOR_ITER then takes the next item instead of testing for one."""
    return f"{_TAKE_PREFIX}{loop}"


def _is_graph_value(vt) -> bool:
    return isinstance(vt, (TensorVariable, SymNodeVariable))


def _is_int_constant(vt) -> bool:
    return isinstance(vt, ConstantVariable) and type(vt.as_python_constant()) is int


def _is_symint(vt) -> bool:
    return isinstance(vt, SymNodeVariable) and isinstance(vt.sym_num, torch.SymInt)


def _ragged(vt) -> bool:
    """Whether a tensor's shape depends on an unbacked symbol (e.g., the last chunk of a loop)."""
    example = _example(vt)
    return bool(free_unbacked_symbols(example.size()) or free_unbacked_symbols(example.stride()))


def _is_constant(vt, value) -> bool:
    return isinstance(vt, ConstantVariable) and vt.as_python_constant() is value


def _sym_expr(vt):
    """A symbolic integer or int constant as a sympy expression (``None`` for anything else)."""
    if _is_int_constant(vt):
        return sympy.Integer(vt.as_python_constant())
    if _is_symint(vt):
        return vt.sym_num.node.expr
    return None


def _example(vt) -> Any:
    if isinstance(vt, SymNodeVariable):
        return vt.sym_num
    return vt.as_proxy().node.meta["example_value"]


def _metadata(example) -> tuple:
    if isinstance(example, torch.Tensor):
        return (
            "tensor",
            example.dtype,
            str(example.device),
            tuple(str(s) for s in example.size()),
            tuple(str(s) for s in example.stride()),
        )
    return ("value", str(example))


def _signature(vt) -> tuple:
    """What a block specialization depends on for one incoming value."""
    if isinstance(vt, (TensorVariable, SymNodeVariable)):
        return _metadata(_example(vt))
    if isinstance(vt, RangeVariable):  # The sequence of a captured loop
        return ("range", tuple(_signature(item) for item in vt.items))
    if vt.is_python_constant():
        value = vt.as_python_constant()
        try:
            hash(value)
            return ("constant", type(value).__name__, value)
        except TypeError:
            return ("constant", type(value).__name__, repr(value))
    return ("object", type(vt).__name__, vt.source.name if vt.source is not None else id(vt))


def _block_input(vt):
    """
    ``vt`` as the input of a block. A tensor whose storage offset depends on a loop counter (e.g., ``row = x[i]``)
    gets an example without it: the block copies its inputs into containers of its own, and the counter is not a
    variable of the block (Dynamo could not lift it into the block's graph).
    """
    if not isinstance(vt, TensorVariable):
        return vt
    example = vt.as_proxy().node.meta["example_value"]
    if not free_unbacked_symbols(example.storage_offset()):
        return vt
    if free_unbacked_symbols(example.size()) or free_unbacked_symbols(example.stride()):
        raise CaptureFallback("a tensor whose shape depends on a loop counter flows between blocks")
    with example.fake_mode:
        clean = torch.empty_strided(example.size(), example.stride(), dtype=example.dtype, device=example.device)
    graph = torch.fx.Graph()
    node = graph.placeholder("block_input")
    node.meta["example_value"] = clean
    return vt.clone(proxy=torch.fx.Proxy(node), source=None, mutation_type=None)


def _is_marker(vt) -> bool:
    """The placeholder for a for loop's iterator on the stack."""
    return isinstance(vt, ConstantVariable) and vt.as_python_constant() is None


def _is_data_dependent_bool(vt) -> bool:
    """A symbolic boolean over data-dependent symbols (loop counters, ``.item()``), which Dynamo cannot guard on."""
    return isinstance(vt, SymNodeVariable) and bool(free_unbacked_symbols(vt.sym_num))


def _is_symbolic_bool(vt) -> bool:
    """A symbolic boolean, e.g., a comparison of sizes (Dynamo would guard on its value and specialize)."""
    return (
        isinstance(vt, SymNodeVariable)
        and isinstance(vt.sym_num, torch.SymBool)
        and bool(vt.sym_num.node.expr.free_symbols)
    )


def _loop_header(info: CodeInfo, get_iter: int) -> int | None:
    following = get_iter + 1
    if following < len(info.instructions) and info.instructions[following].opname == "FOR_ITER":
        return following
    return None


def _iterable(vt) -> bool:
    """Whether a for loop over ``vt`` can be captured: a range with a positive constant step, or a tensor."""
    if isinstance(vt, RangeVariable):
        step = vt.items[2]
        return step.is_python_constant() and step.as_python_constant() > 0
    return isinstance(vt, TensorVariable) and vt.as_proxy().node.meta["example_value"].dim() > 0


def _symbolic_iteration(vt) -> bool:
    """Whether Dynamo would have to specialize the trip count of a loop over ``vt``."""
    if isinstance(vt, RangeVariable):
        return any(not item.is_python_constant() for item in vt.items)
    if isinstance(vt, TensorVariable):
        return not isinstance(vt.as_proxy().node.meta["example_value"].shape[0], int)
    return False


def _constant_trip_count(vt) -> int | None:
    """The trip count of a loop over ``vt`` if it is a constant."""
    if isinstance(vt, RangeVariable) and all(item.is_python_constant() for item in vt.items):
        return len(range(*(item.as_python_constant() for item in vt.items)))
    if isinstance(vt, TensorVariable):
        length = vt.as_proxy().node.meta["example_value"].shape[0]
        return length if isinstance(length, int) else None
    return None


def _binary(tx, op: Callable, a, b):
    return BuiltinVariable(op).call_function(tx, [a, b], {})


def _loop_condition(tx, sequence, index):
    """Whether the loop has another item: ``start + index * step < stop``, or ``index < len(tensor)``."""
    if isinstance(sequence, RangeVariable):
        start, stop, step = sequence.items
        return _binary(tx, operator.lt, _binary(tx, operator.add, start, _binary(tx, operator.mul, index, step)), stop)
    return _binary(tx, operator.lt, index, sequence.call_method(tx, "size", [ConstantVariable.create(0)], {}))


def _loop_item(tx, sequence, index):
    if isinstance(sequence, RangeVariable):
        start, _, step = sequence.items
        return _binary(tx, operator.add, start, _binary(tx, operator.mul, index, step))
    return _binary(tx, operator.getitem, sequence, index)


def _return_from(translator, value) -> None:
    """Makes the translator return ``value`` from its frame."""
    translator.push(value)
    translator.RETURN_VALUE(bt.create_instruction("RETURN_VALUE"))


def _locals_of(translator) -> dict[str, Any]:
    """
    The frame's variables by name: locals, and the contents of its cell variables (locals captured by nested
    functions) and free variables (the frame's own closure). Uninitialized cells are left out.
    """
    cells = set(translator.cell_and_freevars())
    result = {}
    for name, vt in translator.symbolic_locals.items():
        if name in cells:
            if not isinstance(vt, CellVariable):
                continue
            try:
                vt = translator.output.side_effects.load_cell(vt)
            except Exception:  # noqa: BLE001 - "read uninitialized cell" is Dynamo's Unsupported
                continue
        result[name] = vt.realize()
    return result


_CONTINUATIONS: dict[types.CodeType, tuple[types.CodeType, int]] = {}


def _continuation(
    info: CodeInfo, start: int, argnames: list[str], markers: int = 0, builders: list[tuple] = ()
) -> tuple[types.CodeType, int]:
    """
    A code object that runs ``info.code`` from instruction ``start`` with the locals ``argnames`` as positional
    parameters. A prefix (``RESUME`` on 3.11+, the construction of the container and constant locals in ``builders``
    from their leaves, ``markers`` placeholders for the iterators of the enclosing for loops, then a jump) is
    prepended; returns the code and the prefix length.

    Cell and free variables of the frame are passed by value: the continuation has them as cell variables (its own
    ``co_freevars`` is empty), which Dynamo seeds from the parameters of the same name, as ``MAKE_CELL`` does. A
    continuation therefore cannot write to the frame's closure (``nonlocal``); the region must not contain such writes.

    :param builders: ``(local, spec, leaves)`` per local the prefix builds, with ``leaves`` as ``('param', name)`` or
                     ``('const', value)`` in the order of the structure's leaves.
    """
    prefix_len = 0
    cellvars = set(info.code.co_cellvars) | set(info.code.co_freevars)

    def transform(instructions: list[bt.Instruction], code_options: dict[str, Any]) -> None:
        nonlocal prefix_len
        prefix = [bt.create_instruction("RESUME", arg=0)] if sys.version_info >= (3, 11) else []
        for local, spec, leaves in builders:
            prefix.extend(_build_instructions(spec, iter(leaves)))
            prefix.append(bt.create_instruction("STORE_DEREF" if local in cellvars else "STORE_FAST", argval=local))
        prefix.extend(bt.create_load_const(None) for _ in range(markers))
        prefix.append(bt.create_jump_absolute(instructions[start]))
        prefix_len = len(prefix)
        instructions[:0] = prefix
        name = f"__dace_block_{code_options['co_name']}_{start}"
        code_options["co_argcount"] = len(argnames)
        code_options["co_posonlyargcount"] = 0
        code_options["co_kwonlyargcount"] = 0
        code_options["co_varnames"] = tuple(argnames) + tuple(
            v for v in code_options["co_varnames"] if v not in argnames
        )
        code_options["co_flags"] &= ~(0x04 | 0x08)  # Neither *args nor **kwargs
        # The frame's closure becomes cells of the continuation, passed by value
        code_options["co_cellvars"] = tuple(code_options["co_cellvars"]) + tuple(code_options["co_freevars"])
        code_options["co_freevars"] = ()
        # Locals the prefix builds that the frame does not know (e.g., the state of a captured loop)
        built = [local for local, _, _ in builders if local not in cellvars]
        code_options["co_varnames"] += tuple(v for v in built if v not in code_options["co_varnames"])
        code_options["co_name"] = name
        if "co_qualname" in code_options:
            code_options["co_qualname"] = name

    code, _ = bt.transform_code_object(info.code, transform)
    _CONTINUATIONS[code] = (info.code, prefix_len)
    return code, prefix_len


def _build_instructions(spec, leaves: Iterator) -> list[bt.Instruction]:
    """Instructions that build a value of structure ``spec`` from its leaves (parameters or constants)."""
    kind = spec[0]
    if kind == LEAF:
        what, value = next(leaves)
        if what == "param":
            return [bt.create_instruction("LOAD_FAST", argval=value)]
        return [bt.create_load_const(value, checked=False)]
    result = []
    if kind in (LIST, TUPLE):
        for child in spec[1]:
            result.extend(_build_instructions(child, leaves))
        result.append(bt.create_instruction("BUILD_LIST" if kind == LIST else "BUILD_TUPLE", arg=len(spec[1])))
        return result
    if kind == RANGE:  # ``range(start, stop, step)`` from its bounds (the builtin, as Dynamo traces it)
        result.extend(bt.add_push_null([bt.create_instruction("LOAD_GLOBAL", argval="range")]))
        for child in spec[1]:
            result.extend(_build_instructions(child, leaves))
        result.extend(bt.create_call_function(len(spec[1]), False))
        return result
    if kind == FUNCTION:  # Defaults (if any), the closure from the frame's cells, the code; as CPython emits it
        code, defaults_spec = spec[1], spec[2]
        flags = 0x08
        if defaults_spec[1]:
            result.extend(_build_instructions(defaults_spec, leaves))
            flags |= 0x01
        load_cell = "LOAD_FAST" if sys.version_info >= (3, 13) else "LOAD_CLOSURE"
        result.extend(bt.create_instruction(load_cell, argval=name) for name in code.co_freevars)
        result.append(bt.create_instruction("BUILD_TUPLE", arg=len(code.co_freevars)))
        result.append(bt.create_load_const(code, checked=False))
        if sys.version_info < (3, 11):
            result.append(bt.create_load_const(code.co_name))
        if sys.version_info >= (3, 13):
            result.append(bt.create_instruction("MAKE_FUNCTION"))
            result.append(bt.create_instruction("SET_FUNCTION_ATTRIBUTE", arg=0x08))
            if flags & 0x01:
                result.append(bt.create_instruction("SET_FUNCTION_ATTRIBUTE", arg=0x01))
        else:
            result.append(bt.create_instruction("MAKE_FUNCTION", arg=flags))
        return result
    for key, child in spec[1]:
        result.append(bt.create_load_const(key, checked=False))
        result.extend(_build_instructions(child, leaves))
    result.append(bt.create_instruction("BUILD_MAP", arg=len(spec[1])))
    return result


# ---------------------------------------------------------------------------------------------- Dynamo internals
# The exceptions and attributes below are Dynamo internals that differ across torch versions; they are resolved with
# ``getattr`` for that reason.

#: Exceptions Dynamo uses for control flow; they pass through the capture untouched
_PASSTHROUGH_EXCEPTIONS = tuple(
    t
    for t in (
        getattr(sc, "ReturnValueOp", None),
        getattr(sc, "YieldValueOp", None),
        getattr(exc, "ObservedException", None),
        getattr(exc, "RestartAnalysis", None),
        getattr(exc, "TensorifyScalarRestartAnalysis", None),
        getattr(exc, "SkipFrame", None),
        getattr(exc, "BackendCompilerFailed", None),
    )
    if isinstance(t, type)
)

#: Errors of speculative tracing that subclass a passthrough exception but must take the fallback path: on torch
#: 2.14+, ``get_fake_value`` wraps fake-tensor ``RuntimeError`` s in ``FakeTensorObservedException``
_SPECULATION_ERRORS = tuple(t for t in (getattr(exc, "FakeTensorObservedException", None),) if isinstance(t, type))


def _backend_of(tx) -> Any:
    """The user backend object behind a translator's OutputGraph (unwrapping Dynamo's backend wrappers)."""
    fn = tx.output.compiler_fn
    for _ in range(4):
        inner = getattr(fn, "_torchdynamo_orig_backend", None) or getattr(fn, "compiler_fn", None)
        if inner is None or inner is fn:
            break
        fn = inner
    return fn


def _position_key(tx, inst) -> tuple:
    """A source position identifying a jump across analyses of the same frame."""
    pos = inst.positions
    if pos is not None and pos.lineno is not None:
        return (tx.f_code.co_filename, pos.lineno, pos.col_offset, pos.end_col_offset, inst.opname)
    return (tx.f_code.co_filename, inst.starts_line, inst.opname)


def _translator_classes() -> list[type]:
    result, stack = [], [sc.InstructionTranslatorBase]
    while stack:
        cls = stack.pop()
        result.append(cls)
        stack.extend(cls.__subclasses__())
    return result


# ---------------------------------------------------------------------------------------------- Dynamo handlers
def _region_of(translator) -> tuple[Optional["ControlFlowCapture"], Region | None]:
    capture = getattr(_backend_of(translator), "control_flow", None)
    if not isinstance(capture, ControlFlowCapture):
        return None, None
    return capture, (capture.regions[-1] if capture.regions else None)


def _make_jump_handler(original: Callable) -> Callable:

    @functools.wraps(original)
    def handler(self, inst):
        backend, region = _region_of(self)
        if backend is None or not self.stack:
            return original(self, inst)
        value = self.stack[-1].realize()
        symbolic = _is_data_dependent_bool(value) or (backend.symbolic_branches and _is_symbolic_bool(value))
        if symbolic and region is None:
            # A branch on sizes (with ``symbolic_branches``) or on data-dependent scalars: an edge, not a guard
            return _capture_region(backend, self, inst, value, original)
        if symbolic and region is not None and region.is_block_translator(self):
            # A comparison of loop counters or data-dependent scalars (e.g., ``.item()``) in a captured block
            successors = region.info.branch_successors(self.indexof[inst] - region.active.prefix)
            self.pop()
            decision = region.decide(value)
            if decision is None:
                region.end_block(self, BRANCH, successors, predicate=value)
            else:  # Known in this block (e.g., a loop counter at the loop's entry)
                region.end_block(self, GOTO, {None: successors[decision]})
            return None
        if not isinstance(value, TensorVariable):
            return original(self, inst)
        following = self.indexof[inst] + 1
        if following < len(self.instructions) and self.instructions[following].opname in (
            "LOAD_ASSERTION_ERROR",
            "LOAD_COMMON_CONSTANT",
        ):
            return original(self, inst)  # ``assert tensor``: Dynamo rewrites it itself

        if region is not None and region.is_block_translator(self):
            successors = region.info.branch_successors(self.indexof[inst] - region.active.prefix)
            self.pop()
            region.end_block(self, BRANCH, successors, predicate=value)
            return None
        if region is not None:
            raise CaptureFallback("data-dependent control flow in a function called from a captured block")
        return _capture_region(backend, self, inst, value, original)

    return handler


def _capture_region(backend: "ControlFlowCapture", tx, inst, value, original: Callable, loop: bool = False):
    """
    Captures the rest of the frame from a conditional jump on ``value`` or, with ``loop``, from the ``GET_ITER`` of a
    for loop over ``value`` (a symbolic range or a tensor).
    """
    key = _position_key(tx, inst)
    if key in backend.blacklist or tx.block_stack:
        return original(tx, inst)
    info = CodeInfo(tx.f_code)
    index = tx.indexof[inst]
    if not loop and len(tx.stack) > 1:
        # Inside loops Dynamo unrolls (their iterators are on the stack): capture the outermost one from its GET_ITER
        loops = info.loops_containing(index)
        if len(tx.stack) - 1 == info.iterators(index) == len(loops):
            get_iter = min(loops) - 1
            request = _position_key(tx, info.instructions[get_iter])
            if request not in backend.loop_requests:
                backend.loop_requests.add(request)
                backend.log("restart", tx.f_code.co_name, f"capture the loop at instruction {get_iter}")
                # The next pass takes a different path from the GET_ITER on: do not replay this pass's speculation
                tx.speculation_log.clear()
                raise RestartAnalysis(restart_reason="dace control-flow capture of an enclosing loop")
        return original(tx, inst)
    if len(tx.stack) != 1:
        return original(tx, inst)
    try:
        if loop:
            header = _loop_header(info, index)
            if header is None:
                raise CaptureFallback("GET_ITER is not followed by a FOR_ITER")
            region = Region(tx, info, {None: header}, backend.symbolic_branches)
        else:
            region = Region(tx, info, info.branch_successors(index), backend.symbolic_branches)
    except CaptureFallback as ex:
        backend.log("fallback", tx.f_code.co_name, str(ex))
        return original(tx, inst)
    entry_values = _locals_of(tx)
    if loop:
        entry_values[_seq_name(header)] = value
        entry_values[_index_name(header)] = ConstantVariable.create(0)
    backend.regions.append(region)
    # Blocks keep .item() (float attributes, the length of a ragged chunk): the configuration and the frame's fake
    # mode (created with the configuration at the start of the frame) allow data-dependent scalars meanwhile
    fake_mode = getattr(tx, "fake_mode", None)
    allowed = getattr(fake_mode, "allow_scalar_outputs", None)
    try:
        if fake_mode is not None:
            fake_mode.allow_scalar_outputs = True
        with torch._dynamo.config.patch(**CAPTURE_CONFIG):
            result = region.capture(None if loop else value, entry_values)
    except Exception as ex:  # noqa: BLE001 - also Unsupported raised by nested speculation
        if isinstance(ex, _PASSTHROUGH_EXCEPTIONS) and not isinstance(ex, _SPECULATION_ERRORS):
            raise  # Dynamo control flow (returns, restarts, exceptions raised by the program)
        # Speculation may have changed Dynamo's state: blacklist the jump and restart the analysis of the frame; on
        # the next pass Dynamo's stock handler graph-breaks here
        backend.blacklist.add(key)
        message = (str(ex).splitlines() or [""])[0][:200]
        backend.log("error", tx.f_code.co_name, f"{type(ex).__name__}: {message}")
        tx.speculation_log.clear()  # The next pass may diverge before this point (e.g., at a captured GET_ITER)
        raise RestartAnalysis(restart_reason=f"dace control-flow capture failed: {message}") from ex
    finally:
        if fake_mode is not None and allowed is not None:
            fake_mode.allow_scalar_outputs = allowed
        backend.regions.pop()
    backend.log("cfg", tx.f_code.co_name, f"{len(region.blocks)} blocks")
    tx.pop()
    if tx.parent is None:
        # The frame returns the result: its locals are dead (Dynamo would otherwise keep those live at the jump as
        # additional graph outputs)
        cells = set(tx.cell_and_freevars())
        tx.symbolic_locals = {k: v for k, v in tx.symbolic_locals.items() if k in cells}
    _return_from(tx, result)
    return None


def _make_get_iter_handler(original: Callable) -> Callable:

    def stock(translator, inst):
        """Dynamo's own ``GET_ITER``, with a range whose bounds were kept symbolic specialized as Dynamo would have."""
        top = translator.stack[-1].realize() if translator.stack else None
        if isinstance(top, RangeVariable) and any(_is_symint(item) for item in top.items):
            translator.stack[-1] = RangeVariable(
                [
                    ConstantVariable.create(item.evaluate_expr(translator.output)) if _is_symint(item) else item
                    for item in top.items
                ]
            )
        return original(translator, inst)

    @functools.wraps(original)
    def handler(self, inst):
        backend, region = _region_of(self)
        if backend is None or not self.stack:
            return original(self, inst)
        iterable = self.stack[-1].realize()
        if not _iterable(iterable):
            return stock(self, inst)
        if region is not None and region.is_block_translator(self):
            # A for loop in a captured block: its state becomes CFG variables, a placeholder takes the iterator's place
            header = _loop_header(region.info, self.indexof[inst] - region.active.prefix)
            if header is None or not all(_is_marker(v) for v in self.stack[:-1]):
                return stock(self, inst)  # (Inside a loop Dynamo unrolls, this loop is unrolled as well)
            self.pop()
            self.symbolic_locals[_seq_name(header)] = iterable
            self.symbolic_locals[_index_name(header)] = ConstantVariable.create(0)
            self.push(ConstantVariable.create(None))
            return None
        if region is None and backend.wants_loop(iterable, _position_key(self, inst)):
            # Dynamo would specialize the trip count and unroll the loop (or its body has data-dependent control
            # flow, or every loop is wanted): capture it as a loop instead
            return _capture_region(backend, self, inst, iterable, stock, loop=True)
        return stock(self, inst)

    return handler


def _make_for_iter_handler(original: Callable) -> Callable:

    @functools.wraps(original)
    def handler(self, inst):
        backend, region = _region_of(self)
        if region is None or not region.is_block_translator(self):
            return original(self, inst)
        header = self.indexof[inst] - region.active.prefix
        sequence = self.symbolic_locals.get(_seq_name(header))
        if sequence is None:
            return original(self, inst)  # A loop Dynamo unrolls (e.g., over a list of modules)
        index = self.symbolic_locals[_index_name(header)]
        take = self.symbolic_locals.get(_take_name(header))
        if take is not None and take.as_python_constant():
            # On the edge into the body: take the next item
            del self.symbolic_locals[_take_name(header)]
            self.push(_loop_item(self, sequence, index))
            self.symbolic_locals[_index_name(header)] = _binary(self, operator.add, index, ConstantVariable.create(1))
            return None
        predicate = _loop_condition(self, sequence, index)
        if not _is_data_dependent_bool(predicate):
            raise CaptureFallback(f"the condition of the loop at instruction {header} is not symbolic ({predicate})")
        take = {_take_name(header): ConstantVariable.create(True)}
        decision = region.decide(predicate)
        if decision is True:  # Known to have another item (e.g., at the loop's entry): no exit edge from here
            region.end_block(self, GOTO, {None: header}, extra={None: take})
        elif decision is False:
            region.end_block(self, GOTO, {None: region.info.for_iter_exit(header)})
        else:
            region.end_block(
                self,
                BRANCH,
                {True: header, False: region.info.for_iter_exit(header)},
                predicate=predicate,
                extra={True: take},
            )
        return None

    return handler


def _make_call_range(original: Callable) -> Callable:

    @functools.wraps(original)
    def call_range(self, tx, *args, **kwargs):
        # Dynamo specializes symbolic bounds when it builds the range (``__index__``); keep them symbolic when the
        # range is iterated right away (``for i in range(n)``), so that the loop can be captured with a symbolic trip
        # count (``GET_ITER``), and when a block's prologue rebuilds the sequence of a loop from its inputs. Ranges
        # used otherwise (``list(range(n))``, ``len``, ``reversed``) stay Dynamo's.
        backend, region = _region_of(tx)
        if (
            backend is not None
            and not kwargs
            and 1 <= len(args) <= 3
            and (_next_opname(tx) == "GET_ITER" or _in_prologue(tx, region))
            and any(_is_symint(a.realize()) for a in args)
            and all(_is_symint(a.realize()) or _is_int_constant(a.realize()) for a in args)
        ):
            return RangeVariable([a.realize() for a in args])
        return original(self, tx, *args, **kwargs)

    return call_range


def _in_prologue(tx, region) -> bool:
    """Whether ``tx`` is tracing the prologue of a block's continuation (rebuilding its inputs, before its code)."""
    return (
        region is not None
        and region.is_block_translator(tx)
        and tx.current_instruction is not None
        and tx.indexof[tx.current_instruction] < region.active.prefix
    )


def _next_opname(tx) -> str | None:
    """The name of the instruction after the one being translated."""
    pointer = tx.instruction_pointer
    if pointer is None or pointer >= len(tx.instructions):
        return None
    return tx.instructions[pointer].opname


def _make_return_handler(original: Callable) -> Callable:

    @functools.wraps(original)
    def handler(self, inst):
        backend, region = _region_of(self)
        if region is not None and region.is_block_translator(self):
            if inst.opname != "RETURN_VALUE":
                raise CaptureFallback(f"{inst.opname} in a captured block")
            region.end_block(self, RETURN, {}, value=self.pop())
            return None
        return original(self, inst)

    return handler


def _make_step(original: Callable) -> Callable:

    @functools.wraps(original)
    def step(self):
        backend, region = _region_of(self)
        if region is not None and region.is_block_translator(self) and self.instruction_pointer is not None:
            index = region.instruction_index(self)
            if index in region.leaders:
                if index == region.active.start and not region.active.entered:
                    region.active.entered = True
                else:
                    markers = region.info.iterators(index)
                    if len(self.stack) != markers:
                        raise CaptureFallback(f"values on the stack at block boundary {index}")
                    if not all(_is_marker(v) for v in self.stack):
                        # Inside a loop Dynamo unrolls (its iterator is on the stack, e.g., over the items of a
                        # dict): no block boundary here, the unrolled loop stays in the block
                        return original(self)
                    try:
                        region.end_block(self, GOTO, {None: index})
                    except sc.ReturnValueOp:
                        pass  # The block function returned (``step`` catches this for instruction handlers)
                    return False
        return original(self)

    return step


# ---------------------------------------------------------------------------------------------- symbolic indices
def _chunk_bounds(size, start, stop, nonempty):
    """
    Traced in place of ``x[start:stop]`` along a dimension of ``size`` elements when the bounds depend on a loop
    counter (the chunks of a loop, the last of which may be ragged): the start and the length of the chunk, with the
    length as a size-like data-dependent symbol (an ``item()``), so that the shapes computed from it are sizes torch
    can reason about (``view(-1)`` needs a nonzero number of elements) instead of ``Max(0, Min(...) - Min(...))``
    expressions over the counter. ``nonempty`` states that the chunk has at least one element (the loop's condition
    guarantees it when the chunk starts at the loop's item and the dimension is the loop's bound).
    """
    start = min(start, size)
    stop = min(stop, size)
    length = torch.clamp_min(torch.scalar_tensor(stop - start, dtype=torch.int64), 0).item()
    torch._check_is_size(length)
    if nonempty:
        torch._check(length >= 1)
    return start, length


def _is_ragged_slice(item) -> bool:
    """A slice whose start or stop depends on a data-dependent symbol, with the default step."""
    if not isinstance(item, SliceVariable):
        return False
    start, stop, step = item.items
    if not (_is_constant(step, None) or (_is_int_constant(step) and step.as_python_constant() == 1)):
        return False
    return any(_is_symint(bound) and free_unbacked_symbols(bound.sym_num) for bound in (start, stop))


def _nonnegative_bound(tx, vt) -> bool:
    if _is_int_constant(vt):
        return vt.as_python_constant() >= 0
    if _is_symint(vt):
        return bool(bound_sympy(vt.sym_num.node.expr, tx.output.shape_env.var_to_range).lower >= 0)
    return False


def _chunk_of(tx, region, tensor, dim: int, item) -> list | None:
    """``(start, length)`` of a ragged slice as traced values (see :func:`_chunk_bounds`), or ``None`` to leave it."""
    start, stop, _ = item.items
    size = tensor.call_method(tx, "size", [ConstantVariable.create(dim)], {})
    if _is_constant(start, None):
        start = ConstantVariable.create(0)
    if _is_constant(stop, None):
        stop = size
    if not (_nonnegative_bound(tx, start) and _nonnegative_bound(tx, stop)):
        return None
    # The chunk has at least one element when the loop guarantees it (its item at its bound), or by the chunking
    # idiom ``x[i:i+c]`` with a data-dependent start: a loop that produced an empty chunk would have run past the
    # end of the data (torch records the assumption as a runtime assertion, which the SDFG does not check)
    nonempty = (region is not None and region.chunk_nonempty(tx, start, size)) or _chunk_idiom(start, stop)
    result = tx.inline_user_function_return(
        VariableTracker.build(tx, _chunk_bounds), [size, start, stop, ConstantVariable.create(nonempty)], {}
    )
    return list(result.unpack_var_sequence(tx))


def _chunk_idiom(start, stop) -> bool:
    """``x[i:i+c]`` for a data-dependent ``i`` and a positive constant ``c``."""
    start_expr, stop_expr = _sym_expr(start), _sym_expr(stop)
    if start_expr is None or stop_expr is None or not free_unbacked_symbols(start_expr):
        return False
    difference = sympy.simplify(stop_expr - start_expr)
    return bool(difference.is_number and difference > 0)


def _peel_symbolic_indices(tx, tensor, key, region=None) -> tuple[Any, Any] | None:
    """
    Rewrites ``tensor[key]`` for a tuple ``key`` that contains symbolic integers (e.g., ``x[..., i]`` with a loop
    counter ``i``) or slices with data-dependent bounds (``x[i:i+c]``): every symbolic integer becomes a ``select`` on
    the tensor and every such slice a ``narrow`` with a size-like length (:func:`_chunk_bounds`), so that only ints,
    plain slices, ``None`` and ``...`` remain in the key. Torch's indexing does not accept symbolic integers inside
    tuples, and Dynamo only rewrites a bare symbolic index. Returns the rewritten tensor and the remaining key
    (``None`` if nothing remains), or ``None`` if there is nothing to rewrite (or an index tensor, which is left to
    torch).
    """
    if isinstance(key, SliceVariable):  # ``x[i:i+c]``: a bare slice is the one-item key
        key = TupleVariable([key])
    elif isinstance(key, ListVariable):  # ``x[slicer]`` with a list of slices (deprecated in torch, still common)
        key = TupleVariable(list(key.items))
    if not isinstance(key, TupleVariable):
        return None
    items = [item.realize() for item in key.items]
    if not any(_is_symint(item) or _is_ragged_slice(item) for item in items):
        return None
    if any(isinstance(item, TensorVariable) for item in items):
        return None
    rank = tensor.as_proxy().node.meta["example_value"].dim()
    consumes = [not (_is_constant(item, Ellipsis) or _is_constant(item, None)) for item in items]
    dim = 0
    remaining = []
    for position, item in enumerate(items):
        if _is_constant(item, Ellipsis):
            dim = rank - sum(consumes[position + 1 :])
            remaining.append(item)
        elif _is_constant(item, None):
            remaining.append(item)
        elif _is_symint(item):
            tensor = tensor.call_method(tx, "select", [ConstantVariable.create(dim), item], {})
            rank -= 1
        elif _is_ragged_slice(item) and (chunk := _chunk_of(tx, region, tensor, dim, item)) is not None:
            tensor = tensor.call_method(tx, "narrow", [ConstantVariable.create(dim), *chunk], {})
            remaining.append(SliceVariable([ConstantVariable.create(None)] * 3, tx=tx))
            dim += 1
        else:
            remaining.append(item)
            dim += 1
    return tensor, (TupleVariable(remaining) if remaining else None)


def _make_getitem(original: Callable) -> Callable:

    @functools.wraps(original)
    def method(self, tx, *args, **kwargs):
        backend, region = _region_of(tx)
        if backend is not None and len(args) == 1 and not kwargs:
            peeled = _peel_symbolic_indices(tx, self, args[0], region)
            if peeled is not None:
                tensor, key = peeled
                return tensor if key is None else original(tensor, tx, key)
        return original(self, tx, *args, **kwargs)

    return method


_ARITHMETIC = frozenset((operator.add, operator.sub, operator.mul, operator.floordiv, operator.truediv, operator.mod,
                         operator.pow, operator.neg, operator.pos, operator.iadd, operator.isub, operator.imul))


def _bool_as_int(tx, vt):
    """``torch.sym_ite(b, 1, 0)`` for a symbolic boolean ``b``: what ``int(b)`` means, without deciding ``b``."""
    proxy = tx.output.create_proxy("call_function", torch.sym_ite, (vt.as_proxy(), 1, 0), {})
    return SymNodeVariable.create(tx, proxy, None)


def _make_builtin_call(original: Callable) -> Callable:
    """
    Arithmetic on a symbolic boolean (``n // c + (n % c != 0)``): Dynamo decides the boolean and guards on it. With
    ``symbolic_branches`` (or for data-dependent booleans) it stays symbolic as a 0/1 integer instead. Dynamo's
    handlers for symbolic operands build the graph node themselves, so this wraps the entry point of every builtin.
    """

    @functools.wraps(original)
    def method(self, tx, args, kwargs):
        if self.fn in _ARITHMETIC and not kwargs:
            backend, _ = _region_of(tx)
            if backend is not None:
                converted = [_bool_as_int(tx, a) if _symbolic_bool_operand(backend, a) else a for a in args]
                if any(c is not a for c, a in zip(converted, args)):
                    return original(self, tx, converted, kwargs)
        return original(self, tx, args, kwargs)

    return method


def _symbolic_bool_operand(backend, vt) -> bool:
    """A symbolic boolean that must not be decided: data-dependent, or over sizes with ``symbolic_branches``."""
    if not isinstance(vt, SymNodeVariable) or not isinstance(vt.sym_num, torch.SymBool):
        return False
    return _is_data_dependent_bool(vt) or (backend.symbolic_branches and _is_symbolic_bool(vt))


def _make_insert_op(original: Callable) -> Callable:
    """``BINARY_SUBSCR`` on a tensor reaches the graph through ``BuiltinVariable`` (``operator.getitem``)."""

    @functools.wraps(original)
    def method(self, tx, args, kwargs):
        if self.fn is operator.getitem and len(args) == 2 and not kwargs and isinstance(args[0], TensorVariable):
            backend, region = _region_of(tx)
            if backend is not None:
                peeled = _peel_symbolic_indices(tx, args[0], args[1], region)
                if peeled is not None:
                    tensor, key = peeled
                    return tensor if key is None else original(self, tx, [tensor, key], kwargs)
        return original(self, tx, args, kwargs)

    return method


def _make_setitem(original: Callable) -> Callable:

    @functools.wraps(original)
    def method(self, tx, key, value):
        backend, region = _region_of(tx)
        if backend is not None:
            peeled = _peel_symbolic_indices(tx, self, key, region)
            if peeled is not None:
                tensor, key = peeled
                if key is None:
                    return tensor.call_method(tx, "copy_", [value], {})
                return original(tensor, tx, key, value)
        return original(self, tx, key, value)

    return method


def _forget_range_handlers() -> None:
    """Drops Dynamo's cached handlers of ``range(...)`` calls, which hold the (un)patched ``call_range``."""
    cache = BuiltinVariable.call_function_handler_cache
    for key in [k for k in cache if k and k[0] is range]:
        del cache[key]


class _Patches:
    """Installs the handlers in every translator class (once; they are inert for other backends)."""

    _saved: list[tuple[Any, Any, Any]] = []
    _refcount = 0
    _lock = threading.Lock()

    @classmethod
    def install(cls) -> None:
        with cls._lock:
            cls._refcount += 1
            if cls._refcount > 1:
                return
            for klass in _translator_classes():
                table = klass.__dict__.get("dispatch_table")
                if table is None:
                    continue
                for name in JUMP_ON_TRUTH:
                    op = dis.opmap.get(name)
                    if op is not None:
                        cls._saved.append((table, op, table[op]))
                        table[op] = _make_jump_handler(table[op])
                for name in ("RETURN_VALUE", "RETURN_CONST"):
                    op = dis.opmap.get(name)
                    if op is not None:
                        cls._saved.append((table, op, table[op]))
                        table[op] = _make_return_handler(table[op])
                for name, make in (("GET_ITER", _make_get_iter_handler), ("FOR_ITER", _make_for_iter_handler)):
                    op = dis.opmap[name]
                    cls._saved.append((table, op, table[op]))
                    table[op] = make(table[op])
            base = sc.InstructionTranslatorBase
            cls._saved.append((base, "step", base.step))
            base.step = _make_step(base.step)
            cls._saved.append((BuiltinVariable, "call_range", BuiltinVariable.call_range))
            BuiltinVariable.call_range = _make_call_range(BuiltinVariable.call_range)
            for name, make in (("method___getitem__", _make_getitem), ("method___setitem__", _make_setitem)):
                cls._saved.append((TensorVariable, name, getattr(TensorVariable, name)))
                setattr(TensorVariable, name, make(getattr(TensorVariable, name)))
            cls._saved.append(
                (BuiltinVariable, "_handle_insert_op_in_graph", BuiltinVariable._handle_insert_op_in_graph)
            )
            BuiltinVariable._handle_insert_op_in_graph = _make_insert_op(BuiltinVariable._handle_insert_op_in_graph)
            cls._saved.append((BuiltinVariable, "call_function", BuiltinVariable.call_function))
            BuiltinVariable.call_function = _make_builtin_call(BuiltinVariable.call_function)
            _forget_range_handlers()

    @classmethod
    def uninstall(cls) -> None:
        with cls._lock:
            cls._refcount = max(0, cls._refcount - 1)
            if cls._refcount > 0:
                return
            for container, key, original in reversed(cls._saved):
                if isinstance(container, dict):
                    container[key] = original
                else:
                    setattr(container, key, original)
            cls._saved.clear()
            _forget_range_handlers()


class ControlFlowCapture:
    """
    The control-flow capture state of a backend (``backend.control_flow``), created by :class:`DaceBackend` when its
    ``control_flow`` option is set. Installs the Dynamo handlers (once per process; they only act on frames compiled
    with a backend that has this state).

    :param symbolic_branches: Also capture branches on symbolic sizes (``if x.shape[0] > 4``) as edges of the graph,
                              instead of guarding on their outcome and compiling once per outcome.
    :param capture_loops: ``'symbolic'`` captures for loops whose trip count is symbolic (and loops whose bodies
                          branch on data); ``'all'`` also captures loops over constant ranges and tensors with at
                          least two iterations, which Dynamo would otherwise unroll.
    """

    def __init__(self, symbolic_branches: bool = False, capture_loops: str = "symbolic"):
        if capture_loops not in CAPTURE_LOOPS:
            raise ValueError(f"capture_loops must be one of {CAPTURE_LOOPS}, not {capture_loops!r}")
        self.symbolic_branches = symbolic_branches
        self.capture_loops = capture_loops
        self.regions: list[Region] = []
        self.blacklist: set = set()
        self.loop_requests: set = set()  #: GET_ITERs whose loops are captured because their bodies need it
        self.events: list[tuple[str, str, str]] = []  #: (kind, function name, detail)
        _Patches.install()

    def log(self, kind: str, where: str, detail: str = "") -> None:
        self.events.append((kind, where, detail))

    def kinds(self) -> list[str]:
        return [kind for kind, _, _ in self.events]

    def wants_loop(self, iterable, key: tuple) -> bool:
        """Whether the for loop over ``iterable`` at ``key`` (its ``GET_ITER``) is captured as a loop."""
        if _symbolic_iteration(iterable) or key in self.loop_requests:
            return True
        if self.capture_loops == "all":
            count = _constant_trip_count(iterable)
            return count is not None and count >= 2
        return False


class ControlFlowBackend(DaceBackend):
    """
    ``DaceBackend`` with control-flow capture enabled (``DaceBackend(control_flow=True, ...)``): data-dependent
    ``if``/``while`` (and every other control flow in the rest of the frame) and for loops over symbolic ranges are
    captured as a control-flow graph instead of graph-breaking or unrolling (EXPERIMENTAL). The capture's events are
    exposed for tests.
    """

    def __init__(self, symbolic_branches: bool = False, capture_loops: str = "symbolic", **options):
        super().__init__(control_flow=True, symbolic_branches=symbolic_branches, capture_loops=capture_loops, **options)

    @property
    def events(self) -> list[tuple[str, str, str]]:
        return self.control_flow.events

    def log(self, kind: str, where: str, detail: str = "") -> None:
        self.control_flow.log(kind, where, detail)

    def kinds(self) -> list[str]:
        return self.control_flow.kinds()
