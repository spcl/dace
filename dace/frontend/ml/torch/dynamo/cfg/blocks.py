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

Variables that hold tensors or symbolic integers are block inputs. Other live values (Python constants, modules, other
objects) are bound to the continuation as default arguments and specialize the block: a block start is traced once
per distinct set of such values and input metadata, up to :data:`MAX_SPECIALIZATIONS`, after which the capture falls
back to Dynamo's graph break.

For loops over ranges and tensors are captured as loops instead of being unrolled. Their state (the sequence and the
number of items taken) are variables of the graph, and a placeholder takes the iterator's place on the stack, so that
blocks can start inside loop bodies. ``FOR_ITER`` is a branch on whether another item exists; its taken edge enters
the same instruction again, which then pushes the item. At loop headers, integers the loop assigns are *widened* to
unbacked symbols, so that the body is traced once for all iterations; branches on such symbols (e.g., the loop
condition) are captured like branches on tensor data. A for loop is captured from its ``GET_ITER`` when its trip count
is symbolic (``range(x.shape[0])``, a tensor) or, after a restart of the analysis, when its body has data-dependent
control flow.
"""
import dataclasses
import dis
import functools
import operator
import sys
import threading
import types
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch._dynamo import bytecode_transformation as bt
from torch._dynamo import exc
from torch._dynamo import symbolic_convert as sc
from torch._dynamo.exc import RestartAnalysis
from torch._dynamo.variables import (BuiltinVariable, ConstantVariable, RangeVariable, SymNodeVariable, TensorVariable,
                                     TupleVariable)
from torch._dynamo.variables.builder import wrap_fx_proxy
from torch._dynamo.variables.functions import NestedUserFunctionVariable
from torch._dynamo.variables.higher_order_ops import speculate_subgraph
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.utils._sympy.numbers import int_oo
from torch.utils._sympy.value_ranges import bound_sympy

from ..backend import CAPTURE_CONFIG, DaceBackend
from . import transport
from .analysis import JUMP_ON_TRUTH, CodeInfo
from .transport import BRANCH, GOTO, RETURN, Binding, BlockRecord, CfgRecord

#: Traces of one block start (for different constants or input metadata) before giving up
MAX_SPECIALIZATIONS = 4


class CaptureFallback(Exception):
    """The control flow cannot be captured; Dynamo's stock behavior (a graph break) applies instead."""


@dataclasses.dataclass
class _Exit:
    kind: str
    targets: Dict[Any, int]  #: Leader index by edge label (``None`` for goto, truth value for branch)
    names: List[str]  #: CFG variables passed on as graph outputs (after the predicate for a branch)
    return_count: Optional[int] = None  #: Number of tensors returned (``None``: a single tensor, not a tuple)
    side: Dict[str, Any] = dataclasses.field(default_factory=dict)  #: Other live values passed on (not graph outputs)
    extra: Dict[Any, Dict[str, Any]] = dataclasses.field(default_factory=dict)  #: Side values for one edge label


@dataclasses.dataclass
class _Active:
    """The block being traced."""
    code: types.CodeType  #: Continuation code object
    prefix: int  #: Number of instructions prepended to the original code
    start: int
    entered: bool = False
    exit: Optional[_Exit] = None


class Region:
    """The capture of one frame region, from its entry jump to every return reachable from it."""

    def __init__(self, tx, info: CodeInfo, entry_targets: Dict[Any, int]):
        """
        :param entry_targets: The instructions control enters the region at: by truth value of the entry predicate
                              (a branch), or under ``None`` (a goto, e.g., into a loop).
        """
        self.tx = tx
        self.info = info
        self.entry_targets: Dict[Any, int] = dict(entry_targets)
        self.leaders = info.leaders(list(entry_targets.values()))
        region = info.region(list(entry_targets.values()))
        depths = info.stack_depths()
        for leader in self.leaders:
            # Only iterators of for loops may be on the stack; blocks start with placeholders for them
            if depths.get(leader, -1) != info.iterators(leader):
                raise CaptureFallback('a block would start with values on the stack')
        if info.has_exception_handlers(region):
            raise CaptureFallback('the control flow is inside a try or with block')
        self.active: Optional[_Active] = None
        self.blocks: List[BlockRecord] = []
        self._keys: Dict[Tuple, int] = {}
        self._starts: Dict[int, int] = {}
        self._outer_tensors: List[Any] = []  #: Outer proxies of the operator's tensor inputs
        self._outer_syms: List[Any] = []  #: Outer proxies of the operator's SymInt inputs
        self.return_examples: Optional[List[Any]] = None
        self.return_count: Optional[int] = None
        self._widened: Dict[Tuple[int, str], SymNodeVariable] = {}
        self._nonnegative_assumed: set = set()
        self.entry_constants: Dict[str, Any] = {}

    # ------------------------------------------------------------------------------------------ operator inputs
    def bind_outer(self, proxy) -> Binding:
        """The operator input for an outer (frame-level) proxy, added if new."""
        example = proxy.node.meta.get('example_value')
        is_sym = isinstance(example, (torch.SymInt, torch.SymBool))
        values = self._outer_syms if is_sym else self._outer_tensors
        for index, existing in enumerate(values):
            if existing.node is proxy.node:
                return Binding('sym' if is_sym else 'tensor', index)
        values.append(proxy)
        return Binding('sym' if is_sym else 'tensor', len(values) - 1)

    # ------------------------------------------------------------------------------------------ blocks
    def live(self, start: int, available: Dict[str, Any]) -> List[str]:
        """The variables live at ``start``: user variables, and the state of the loops ``start`` is in."""
        loops = self.info.loops_containing(start)
        names = set(self.info.livevars(start))
        for loop in loops:
            names.update((_seq_name(loop), _index_name(loop)))
        if start in loops:
            names.add(_take_name(start))
        return sorted(name for name in names if name in available)

    def widen(self, start: int, incoming: Dict[str, Any]) -> Dict[str, Any]:
        """
        At a loop header, integers the loop assigns become unbacked symbols: the loop body is traced once for all
        iterations instead of once per value.
        """
        if not self.info.is_loop_header(start):
            return incoming
        carried = self.info.stored_names(self.info.loop_body(start)) | {_index_name(start)}
        result = dict(incoming)
        for name, value in incoming.items():
            if name in carried and (_is_int_constant(value) or _is_symint(value)):
                widened = self._widened_symbol(start, name, nonnegative=self._nonnegative(value))
                if (start, name) in self._nonnegative_assumed and value is not widened and not self._nonnegative(value):
                    # The assumption made when widening does not hold on this edge (checked on every edge, this
                    # proves it by induction)
                    raise CaptureFallback(f'loop variable {name} may become negative')
                result[name] = widened
        return result

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
            node = graph.placeholder(f'widened_{start}_{name.strip("_")}')
            node.meta['example_value'] = symbol
            self._widened[key] = SymNodeVariable(torch.fx.Proxy(node), symbol)
        return self._widened[key]

    def block_for(self, start: int, incoming: Dict[str, Any]) -> Tuple[int, bool]:
        """The block id for ``start`` given the incoming variable values; ``True`` if it still has to be traced."""
        live = self.live(start, incoming)
        incoming = self.widen(start, {name: _block_input(incoming[name]) for name in live})
        key = (start, tuple((name, _signature(incoming[name])) for name in live))
        if key in self._keys:
            return self._keys[key], False
        self._starts[start] = self._starts.get(start, 0) + 1
        if self._starts[start] > MAX_SPECIALIZATIONS:
            raise CaptureFallback(f'block at instruction {start} needs more than {MAX_SPECIALIZATIONS} specializations '
                                  '(e.g., a Python value that changes in a loop)')
        block_id = len(self.blocks)
        self._keys[key] = block_id
        inputs = [name for name in live if _is_graph_value(incoming[name])]
        examples = [_example(incoming[name]) for name in inputs]
        self.blocks.append(BlockRecord(block_id, start, None, inputs, examples))
        self._incoming[block_id] = {name: incoming[name] for name in live}
        return block_id, True

    def trace(self, entry_values: Dict[str, Any]) -> CfgRecord:
        """Traces all blocks reachable from the entry and registers the control-flow graph."""
        self._incoming: Dict[int, Dict[str, Any]] = {}
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

        # Entry bindings: the frame-level values of the variables the entry blocks read
        entry_bindings = {}
        for block_id in set(entry_successors.values()):
            for name in self.blocks[block_id].input_names:
                if _is_graph_value(entry_values[name]):
                    entry_bindings[name] = self.bind_outer(entry_values[name].as_proxy())
                else:  # A Python integer for a symbolic input
                    self.entry_constants[name] = entry_values[name].as_python_constant()
        if self.return_examples is None:
            raise CaptureFallback('no path returns from the frame')
        self.entry_bindings = entry_bindings
        self.entry_successors = entry_successors
        return None

    def _trace_block(self, block_id: int) -> Dict[Any, Tuple[int, bool]]:
        tx = self.tx
        block = self.blocks[block_id]
        incoming = self._incoming[block_id]
        others = [name for name in incoming if name not in block.input_names]
        markers = self.info.iterators(block.start)
        code, prefix = _continuation(self.info, block.start, block.input_names + others, markers)
        function = NestedUserFunctionVariable(ConstantVariable.create(code.co_name), ConstantVariable.create(code),
                                              tx.f_globals,
                                              TupleVariable([incoming[name] for name in others]) if others else None,
                                              None, None)
        args = [incoming[name] for name in block.input_names]
        previous = self.active
        self.active = _Active(code, prefix, block.start)
        try:
            (output, _), graph, lifted = speculate_subgraph(tx,
                                                            function,
                                                            args, {},
                                                            f'dace_cfg_block_{block_id}',
                                                            set_subgraph_inputs='flatten_manual',
                                                            should_flatten_outputs=True)
            block_exit = self.active.exit
        finally:
            self.active = previous
        if block_exit is None:
            raise CaptureFallback(f'block at instruction {block.start} ended without a recognized exit')

        # Placeholders after the block inputs are values of the frame (lifted free variables)
        placeholders = [node for node in graph.nodes if node.op == 'placeholder']
        inner_to_outer = {inner.node: outer for outer, inner in lifted.items()}
        block.graph = graph
        block.lifted = [self.bind_outer(inner_to_outer[node]) for node in placeholders[len(block.input_names):]]
        block.exit_kind = block_exit.kind
        block.output_names = list(block_exit.names)

        outputs = list(output.unpack_var_sequence(tx)) if hasattr(output, 'unpack_var_sequence') else list(output.items)
        if block_exit.kind == RETURN:
            examples = [_example(vt) for vt in outputs]
            if self.return_examples is None:
                self.return_examples, self.return_count = examples, block_exit.return_count
            elif (self.return_count != block_exit.return_count
                  or [_metadata(e) for e in examples] != [_metadata(e) for e in self.return_examples]):
                raise CaptureFallback('the frame returns values of different types or shapes on different paths')
            block.successors = None
            return {}

        values = outputs[1:] if block_exit.kind == BRANCH else outputs
        passed = dict(zip(block_exit.names, values))
        passed.update(block_exit.side)
        block.output_constants = {
            name: vt.as_python_constant()
            for name, vt in block_exit.side.items() if _is_int_constant(vt)
        }
        successors = {}
        for label, target in block_exit.targets.items():
            successors[label] = self.block_for(target, {**passed, **block_exit.extra.get(label, {})})
        block.successors = (successors[None][0] if block_exit.kind == GOTO else {
            label: successor
            for label, (successor, _) in successors.items()
        })
        return successors

    # ------------------------------------------------------------------------------------------ exits
    def end_block(self,
                  translator,
                  kind: str,
                  targets: Dict[Any, int],
                  predicate=None,
                  value=None,
                  extra: Optional[Dict[Any, Dict[str, Any]]] = None) -> None:
        """
        Ends the block being traced by returning the values its successors need from the continuation.

        :param extra: Python values passed on one edge only, by edge label (e.g., the "take" flag of a loop).
        """
        if kind == RETURN:
            if isinstance(value, TensorVariable):
                items, count = [value], None
            elif isinstance(value, TupleVariable) and all(isinstance(v, TensorVariable) for v in value.items):
                items, count = list(value.items), len(value.items)
            else:
                raise CaptureFallback(f'the frame returns a {type(value).__name__}, not tensors')
            self.active.exit = _Exit(RETURN, {}, [], count)
            _return_from(translator, TupleVariable(items))
            return
        live = sorted({name for target in targets.values() for name in self.live(target, translator.symbolic_locals)})
        values = {name: translator.symbolic_locals[name].realize() for name in live}
        names = [name for name in live if _is_graph_value(values[name])]
        side = {name: vt for name, vt in values.items() if name not in names}
        self.active.exit = _Exit(kind, dict(targets), names, side=side, extra=dict(extra or {}))
        _return_from(translator, TupleVariable(([predicate] if kind == BRANCH else []) + [values[n] for n in names]))

    def is_block_translator(self, translator) -> bool:
        return self.active is not None and translator.f_code is self.active.code

    def instruction_index(self, translator) -> int:
        return translator.instruction_pointer - self.active.prefix

    # ------------------------------------------------------------------------------------------ result
    def emit(self) -> Any:
        """Registers the graph and emits the ``dace::cfg`` call in the frame; returns the frame's return value."""
        tx = self.tx
        predicate_binding = self._predicate_binding
        record = transport.register(lambda cfg_id: CfgRecord(
            cfg_id, self.blocks, predicate_binding, self.entry_successors, self.entry_bindings, self.return_examples, tx
            .f_code.co_name, self.entry_constants, self._predicate_expr))
        proxy = tx.output.create_proxy('call_function', torch.ops.dace.cfg.default,
                                       (record.id, list(self._outer_tensors), list(self._outer_syms)), {})
        result = wrap_fx_proxy(tx, proxy, example_value=list(self.return_examples))
        items = list(result.unpack_var_sequence(tx)) if hasattr(result, 'unpack_var_sequence') else list(result.items)
        return items[0] if self.return_count is None else TupleVariable(items)

    def capture(self, predicate, entry_values: Dict[str, Any]) -> Any:
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
_SEQUENCE_PREFIX, _INDEX_PREFIX, _TAKE_PREFIX = '__dace_seq_', '__dace_index_', '__dace_take_'


def _seq_name(loop: int) -> str:
    """The iterated sequence (a tensor, or a range)."""
    return f'{_SEQUENCE_PREFIX}{loop}'


def _index_name(loop: int) -> str:
    """The number of items taken so far."""
    return f'{_INDEX_PREFIX}{loop}'


def _take_name(loop: int) -> str:
    """Set on the edge into the loop body: the FOR_ITER then takes the next item instead of testing for one."""
    return f'{_TAKE_PREFIX}{loop}'


def _is_graph_value(vt) -> bool:
    return isinstance(vt, (TensorVariable, SymNodeVariable))


def _is_int_constant(vt) -> bool:
    return isinstance(vt, ConstantVariable) and type(vt.as_python_constant()) is int


def _is_symint(vt) -> bool:
    return isinstance(vt, SymNodeVariable) and isinstance(vt.sym_num, torch.SymInt)


def _example(vt) -> Any:
    if isinstance(vt, SymNodeVariable):
        return vt.sym_num
    return vt.as_proxy().node.meta['example_value']


def _metadata(example) -> Tuple:
    if isinstance(example, torch.Tensor):
        return ('tensor', example.dtype, str(example.device), tuple(str(s) for s in example.size()),
                tuple(str(s) for s in example.stride()))
    return ('value', str(example))


def _signature(vt) -> Tuple:
    """What a block specialization depends on for one incoming variable."""
    if isinstance(vt, (TensorVariable, SymNodeVariable)):
        return _metadata(_example(vt))
    if isinstance(vt, RangeVariable):  # The sequence of a captured loop
        return ('range', tuple(_signature(item) for item in vt.items))
    if vt.is_python_constant():
        value = vt.as_python_constant()
        try:
            hash(value)
            return ('constant', type(value).__name__, value)
        except TypeError:
            return ('constant', type(value).__name__, repr(value))
    return ('object', type(vt).__name__, vt.source.name if vt.source is not None else id(vt))


def _block_input(vt):
    """
    ``vt`` as the input of a block. A tensor whose storage offset depends on a loop counter (e.g., ``row = x[i]``)
    gets an example without it: the block copies its inputs into containers of its own, and the counter is not a
    variable of the block (Dynamo could not lift it into the block's graph).
    """
    if not isinstance(vt, TensorVariable):
        return vt
    example = vt.as_proxy().node.meta['example_value']
    if not free_unbacked_symbols(example.storage_offset()):
        return vt
    if free_unbacked_symbols(example.size()) or free_unbacked_symbols(example.stride()):
        raise CaptureFallback('a tensor whose shape depends on a loop counter flows between blocks')
    with example.fake_mode:
        clean = torch.empty_strided(example.size(), example.stride(), dtype=example.dtype, device=example.device)
    graph = torch.fx.Graph()
    node = graph.placeholder('block_input')
    node.meta['example_value'] = clean
    return vt.clone(proxy=torch.fx.Proxy(node), source=None, mutation_type=None)


def _is_marker(vt) -> bool:
    """The placeholder for a for loop's iterator on the stack."""
    return isinstance(vt, ConstantVariable) and vt.as_python_constant() is None


def _is_data_dependent_bool(vt) -> bool:
    """A symbolic boolean over data-dependent symbols (loop counters, ``.item()``), which Dynamo cannot guard on."""
    return isinstance(vt, SymNodeVariable) and bool(free_unbacked_symbols(vt.sym_num))


def _is_symbolic_bool(vt) -> bool:
    """A symbolic boolean, e.g., a comparison of sizes (Dynamo would guard on its value and specialize)."""
    return (isinstance(vt, SymNodeVariable) and isinstance(vt.sym_num, torch.SymBool)
            and bool(vt.sym_num.node.expr.free_symbols))


def _loop_header(info: CodeInfo, get_iter: int) -> Optional[int]:
    following = get_iter + 1
    if following < len(info.instructions) and info.instructions[following].opname == 'FOR_ITER':
        return following
    return None


def _iterable(vt) -> bool:
    """Whether a for loop over ``vt`` can be captured: a range with a positive constant step, or a tensor."""
    if isinstance(vt, RangeVariable):
        step = vt.items[2]
        return step.is_python_constant() and step.as_python_constant() > 0
    return isinstance(vt, TensorVariable) and vt.as_proxy().node.meta['example_value'].dim() > 0


def _symbolic_iteration(vt) -> bool:
    """Whether Dynamo would have to specialize the trip count of a loop over ``vt``."""
    if isinstance(vt, RangeVariable):
        return any(not item.is_python_constant() for item in vt.items)
    if isinstance(vt, TensorVariable):
        return not isinstance(vt.as_proxy().node.meta['example_value'].shape[0], int)
    return False


def _binary(tx, op: Callable, a, b):
    return BuiltinVariable(op).call_function(tx, [a, b], {})


def _loop_condition(tx, sequence, index):
    """Whether the loop has another item: ``start + index * step < stop``, or ``index < len(tensor)``."""
    if isinstance(sequence, RangeVariable):
        start, stop, step = sequence.items
        return _binary(tx, operator.lt, _binary(tx, operator.add, start, _binary(tx, operator.mul, index, step)), stop)
    return _binary(tx, operator.lt, index, sequence.call_method(tx, 'size', [ConstantVariable.create(0)], {}))


def _loop_item(tx, sequence, index):
    if isinstance(sequence, RangeVariable):
        start, _, step = sequence.items
        return _binary(tx, operator.add, start, _binary(tx, operator.mul, index, step))
    return _binary(tx, operator.getitem, sequence, index)


def _return_from(translator, value) -> None:
    """Makes the translator return ``value`` from its frame."""
    translator.push(value)
    translator.RETURN_VALUE(bt.create_instruction('RETURN_VALUE'))


_CONTINUATIONS: Dict[types.CodeType, Tuple[types.CodeType, int]] = {}


def _continuation(info: CodeInfo, start: int, argnames: List[str], markers: int = 0) -> Tuple[types.CodeType, int]:
    """
    A code object that runs ``info.code`` from instruction ``start`` with the locals ``argnames`` as positional
    parameters. A prefix (``RESUME`` on 3.11+, ``markers`` placeholders for the iterators of the enclosing for loops,
    then a jump) is prepended; returns the code and the prefix length.
    """
    if info.code.co_freevars or info.code.co_cellvars:
        raise CaptureFallback('frames with cell or free variables are not supported yet')
    prefix_len = 0

    def transform(instructions: List[bt.Instruction], code_options: Dict[str, Any]) -> None:
        nonlocal prefix_len
        prefix = [bt.create_instruction('RESUME', arg=0)] if sys.version_info >= (3, 11) else []
        prefix.extend(bt.create_load_const(None) for _ in range(markers))
        prefix.append(bt.create_jump_absolute(instructions[start]))
        prefix_len = len(prefix)
        instructions[:0] = prefix
        name = f'__dace_block_{code_options["co_name"]}_{start}'
        code_options['co_argcount'] = len(argnames)
        code_options['co_posonlyargcount'] = 0
        code_options['co_kwonlyargcount'] = 0
        code_options['co_varnames'] = tuple(argnames) + tuple(v
                                                              for v in code_options['co_varnames'] if v not in argnames)
        code_options['co_flags'] &= ~(0x04 | 0x08)  # Neither *args nor **kwargs
        code_options['co_name'] = name
        if 'co_qualname' in code_options:
            code_options['co_qualname'] = name

    code, _ = bt.transform_code_object(info.code, transform)
    _CONTINUATIONS[code] = (info.code, prefix_len)
    return code, prefix_len


# ---------------------------------------------------------------------------------------------- Dynamo internals
# The exceptions and attributes below are Dynamo internals that differ across torch versions; they are resolved with
# ``getattr`` for that reason.

#: Exceptions Dynamo uses for control flow; they pass through the capture untouched
_PASSTHROUGH_EXCEPTIONS = tuple(t
                                for t in (getattr(sc, 'ReturnValueOp', None), getattr(sc, 'YieldValueOp', None),
                                          getattr(exc, 'ObservedException', None),
                                          getattr(exc, 'RestartAnalysis', None),
                                          getattr(exc, 'TensorifyScalarRestartAnalysis', None),
                                          getattr(exc, 'SkipFrame', None), getattr(exc, 'BackendCompilerFailed', None))
                                if isinstance(t, type))

#: Errors of speculative tracing that subclass a passthrough exception but must take the fallback path: on torch
#: 2.14+, ``get_fake_value`` wraps fake-tensor ``RuntimeError`` s in ``FakeTensorObservedException``
_SPECULATION_ERRORS = tuple(t for t in (getattr(exc, 'FakeTensorObservedException', None), ) if isinstance(t, type))


def _backend_of(tx) -> Any:
    """The user backend object behind a translator's OutputGraph (unwrapping Dynamo's backend wrappers)."""
    fn = tx.output.compiler_fn
    for _ in range(4):
        inner = getattr(fn, '_torchdynamo_orig_backend', None) or getattr(fn, 'compiler_fn', None)
        if inner is None or inner is fn:
            break
        fn = inner
    return fn


def _position_key(tx, inst) -> Tuple:
    """A source position identifying a jump across analyses of the same frame."""
    pos = inst.positions
    if pos is not None and pos.lineno is not None:
        return (tx.f_code.co_filename, pos.lineno, pos.col_offset, pos.end_col_offset, inst.opname)
    return (tx.f_code.co_filename, inst.starts_line, inst.opname)


def _translator_classes() -> List[type]:
    result, stack = [], [sc.InstructionTranslatorBase]
    while stack:
        cls = stack.pop()
        result.append(cls)
        stack.extend(cls.__subclasses__())
    return result


# ---------------------------------------------------------------------------------------------- Dynamo handlers
def _region_of(translator) -> Tuple[Optional['ControlFlowBackend'], Optional[Region]]:
    backend = _backend_of(translator)
    if not isinstance(backend, ControlFlowBackend):
        return None, None
    return backend, (backend.regions[-1] if backend.regions else None)


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
            region.end_block(self, BRANCH, successors, predicate=value)
            return None
        if not isinstance(value, TensorVariable):
            return original(self, inst)
        following = self.indexof[inst] + 1
        if following < len(self.instructions) and self.instructions[following].opname in ('LOAD_ASSERTION_ERROR',
                                                                                          'LOAD_COMMON_CONSTANT'):
            return original(self, inst)  # ``assert tensor``: Dynamo rewrites it itself

        if region is not None and region.is_block_translator(self):
            successors = region.info.branch_successors(self.indexof[inst] - region.active.prefix)
            self.pop()
            region.end_block(self, BRANCH, successors, predicate=value)
            return None
        if region is not None:
            raise CaptureFallback('data-dependent control flow in a function called from a captured block')
        return _capture_region(backend, self, inst, value, original)

    return handler


def _capture_region(backend: 'ControlFlowBackend', tx, inst, value, original: Callable, loop: bool = False):
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
                backend.log('restart', tx.f_code.co_name, f'capture the loop at instruction {get_iter}')
                # The next pass takes a different path from the GET_ITER on: do not replay this pass's speculation
                tx.speculation_log.clear()
                raise RestartAnalysis(restart_reason='dace control-flow capture of an enclosing loop')
        return original(tx, inst)
    if len(tx.stack) != 1:
        return original(tx, inst)
    try:
        if loop:
            header = _loop_header(info, index)
            if header is None:
                raise CaptureFallback('GET_ITER is not followed by a FOR_ITER')
            region = Region(tx, info, {None: header})
        else:
            region = Region(tx, info, info.branch_successors(index))
    except CaptureFallback as ex:
        backend.log('fallback', tx.f_code.co_name, str(ex))
        return original(tx, inst)
    entry_values = {name: tx.symbolic_locals[name].realize() for name in tx.symbolic_locals}
    if loop:
        entry_values[_seq_name(header)] = value
        entry_values[_index_name(header)] = ConstantVariable.create(0)
    backend.regions.append(region)
    try:
        with torch._dynamo.config.patch(**CAPTURE_CONFIG):  # Blocks keep .item() (e.g., of float attributes)
            result = region.capture(None if loop else value, entry_values)
    except Exception as ex:  # noqa: BLE001 - also Unsupported raised by nested speculation
        if isinstance(ex, _PASSTHROUGH_EXCEPTIONS) and not isinstance(ex, _SPECULATION_ERRORS):
            raise  # Dynamo control flow (returns, restarts, exceptions raised by the program)
        # Speculation may have changed Dynamo's state: blacklist the jump and restart the analysis of the frame; on
        # the next pass Dynamo's stock handler graph-breaks here
        backend.blacklist.add(key)
        message = (str(ex).splitlines() or [''])[0][:200]
        backend.log('error', tx.f_code.co_name, f'{type(ex).__name__}: {message}')
        tx.speculation_log.clear()  # The next pass may diverge before this point (e.g., at a captured GET_ITER)
        raise RestartAnalysis(restart_reason=f'dace control-flow capture failed: {message}') from ex
    finally:
        backend.regions.pop()
    backend.log('cfg', tx.f_code.co_name, f'{len(region.blocks)} blocks')
    tx.pop()
    if tx.parent is None:
        # The frame returns the result: its locals are dead (Dynamo would otherwise keep those live at the jump as
        # additional graph outputs)
        cells = set(tx.cell_and_freevars())
        tx.symbolic_locals = {k: v for k, v in tx.symbolic_locals.items() if k in cells}
    _return_from(tx, result)
    return None


def _make_get_iter_handler(original: Callable) -> Callable:

    @functools.wraps(original)
    def handler(self, inst):
        backend, region = _region_of(self)
        if backend is None or not self.stack:
            return original(self, inst)
        iterable = self.stack[-1].realize()
        if not _iterable(iterable):
            return original(self, inst)
        if region is not None and region.is_block_translator(self):
            # A for loop in a captured block: its state becomes CFG variables, a placeholder takes the iterator's place
            header = _loop_header(region.info, self.indexof[inst] - region.active.prefix)
            if header is None:
                return original(self, inst)
            self.pop()
            self.symbolic_locals[_seq_name(header)] = iterable
            self.symbolic_locals[_index_name(header)] = ConstantVariable.create(0)
            self.push(ConstantVariable.create(None))
            return None
        if region is None and (_symbolic_iteration(iterable) or _position_key(self, inst) in backend.loop_requests):
            # Dynamo would specialize the trip count and unroll the loop (or its body has data-dependent control
            # flow): capture it as a loop instead
            return _capture_region(backend, self, inst, iterable, original, loop=True)
        return original(self, inst)

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
            raise CaptureFallback(f'the condition of the loop at instruction {header} is not symbolic ({predicate})')
        region.end_block(self,
                         BRANCH, {
                             True: header,
                             False: region.info.for_iter_exit(header)
                         },
                         predicate=predicate,
                         extra={True: {
                             _take_name(header): ConstantVariable.create(True)
                         }})
        return None

    return handler


def _make_call_range(original: Callable) -> Callable:

    @functools.wraps(original)
    def call_range(self, tx, *args, **kwargs):
        # Dynamo specializes symbolic bounds when it builds the range (``__index__``); keep them symbolic, so that a
        # loop over the range can be captured with a symbolic trip count (``GET_ITER``)
        backend, _ = _region_of(tx)
        if backend is not None and not kwargs and 1 <= len(args) <= 3 and any(
                _is_symint(a.realize())
                for a in args) and all(_is_symint(a.realize()) or _is_int_constant(a.realize()) for a in args):
            return RangeVariable([a.realize() for a in args])
        return original(self, tx, *args, **kwargs)

    return call_range


def _make_return_handler(original: Callable) -> Callable:

    @functools.wraps(original)
    def handler(self, inst):
        backend, region = _region_of(self)
        if region is not None and region.is_block_translator(self):
            if inst.opname != 'RETURN_VALUE':
                raise CaptureFallback(f'{inst.opname} in a captured block')
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
                    if len(self.stack) != markers or not all(_is_marker(v) for v in self.stack):
                        raise CaptureFallback(f'values on the stack at block boundary {index}')
                    try:
                        region.end_block(self, GOTO, {None: index})
                    except sc.ReturnValueOp:
                        pass  # The block function returned (``step`` catches this for instruction handlers)
                    return False
        return original(self)

    return step


def _forget_range_handlers() -> None:
    """Drops Dynamo's cached handlers of ``range(...)`` calls, which hold the (un)patched ``call_range``."""
    cache = BuiltinVariable.call_function_handler_cache
    for key in [k for k in cache if k and k[0] is range]:
        del cache[key]


class _Patches:
    """Installs the handlers in every translator class (once; they are inert for other backends)."""
    _saved: List[Tuple[Any, Any, Any]] = []
    _refcount = 0
    _lock = threading.Lock()

    @classmethod
    def install(cls) -> None:
        with cls._lock:
            cls._refcount += 1
            if cls._refcount > 1:
                return
            for klass in _translator_classes():
                table = klass.__dict__.get('dispatch_table')
                if table is None:
                    continue
                for name in JUMP_ON_TRUTH:
                    op = dis.opmap.get(name)
                    if op is not None:
                        cls._saved.append((table, op, table[op]))
                        table[op] = _make_jump_handler(table[op])
                for name in ('RETURN_VALUE', 'RETURN_CONST'):
                    op = dis.opmap.get(name)
                    if op is not None:
                        cls._saved.append((table, op, table[op]))
                        table[op] = _make_return_handler(table[op])
                for name, make in (('GET_ITER', _make_get_iter_handler), ('FOR_ITER', _make_for_iter_handler)):
                    op = dis.opmap[name]
                    cls._saved.append((table, op, table[op]))
                    table[op] = make(table[op])
            base = sc.InstructionTranslatorBase
            cls._saved.append((base, 'step', base.step))
            base.step = _make_step(base.step)
            cls._saved.append((BuiltinVariable, 'call_range', BuiltinVariable.call_range))
            BuiltinVariable.call_range = _make_call_range(BuiltinVariable.call_range)
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


class ControlFlowBackend(DaceBackend):
    """
    ``DaceBackend`` that captures data-dependent ``if``/``while`` (and every other control flow in the rest of the
    frame) as a control-flow graph instead of graph-breaking (EXPERIMENTAL).

    :param symbolic_branches: Also capture branches on symbolic sizes (``if x.shape[0] > 4``) as edges of the graph,
                              instead of guarding on their outcome and compiling once per outcome.
    """

    def __init__(self, symbolic_branches: bool = False, **options):
        super().__init__(**options)
        self.symbolic_branches = symbolic_branches
        self.regions: List[Region] = []
        self.blacklist: set = set()
        self.loop_requests: set = set()  #: GET_ITERs whose loops are captured because their bodies need it
        self.events: List[Tuple[str, str, str]] = []  #: (kind, function name, detail)
        _Patches.install()

    def log(self, kind: str, where: str, detail: str = '') -> None:
        self.events.append((kind, where, detail))

    def kinds(self) -> List[str]:
        return [kind for kind, _, _ in self.events]
