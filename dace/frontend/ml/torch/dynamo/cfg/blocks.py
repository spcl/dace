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
"""
import dataclasses
import dis
import functools
import sys
import threading
import types
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch._dynamo import bytecode_transformation as bt
from torch._dynamo import exc
from torch._dynamo import symbolic_convert as sc
from torch._dynamo.exc import RestartAnalysis
from torch._dynamo.variables import ConstantVariable, SymNodeVariable, TensorVariable, TupleVariable
from torch._dynamo.variables.builder import wrap_fx_proxy
from torch._dynamo.variables.functions import NestedUserFunctionVariable
from torch._dynamo.variables.higher_order_ops import speculate_subgraph

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

    def __init__(self, tx, info: CodeInfo, entry_jump: int):
        self.tx = tx
        self.info = info
        branches = info.branch_successors(entry_jump)
        self.entry_targets: Dict[bool, int] = dict(branches)
        self.leaders = info.leaders(list(branches.values()))
        region = info.region(list(branches.values()))
        depths = info.stack_depths()
        if any(depths.get(leader, -1) != 0 for leader in self.leaders):
            raise CaptureFallback('a block would start with values on the stack (e.g., inside a for loop)')
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

    # ------------------------------------------------------------------------------------------ operator inputs
    def bind_outer(self, proxy) -> Binding:
        """The operator input for an outer (frame-level) proxy, added if new."""
        example = proxy.node.meta.get('example_value')
        is_sym = isinstance(example, torch.SymInt)
        values = self._outer_syms if is_sym else self._outer_tensors
        for index, existing in enumerate(values):
            if existing.node is proxy.node:
                return Binding('sym' if is_sym else 'tensor', index)
        values.append(proxy)
        return Binding('sym' if is_sym else 'tensor', len(values) - 1)

    # ------------------------------------------------------------------------------------------ blocks
    def block_for(self, start: int, incoming: Dict[str, Any]) -> Tuple[int, bool]:
        """The block id for ``start`` given the incoming variable values; ``True`` if it still has to be traced."""
        live = sorted(name for name in self.info.livevars(start) if name in incoming)
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

    def trace(self) -> CfgRecord:
        """Traces all blocks reachable from the entry and registers the control-flow graph."""
        tx = self.tx
        self._incoming: Dict[int, Dict[str, Any]] = {}
        entry_values = {name: tx.symbolic_locals[name].realize() for name in tx.symbolic_locals}
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
                entry_bindings[name] = self.bind_outer(entry_values[name].as_proxy())
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
        code, prefix = _continuation(self.info, block.start, block.input_names + others)
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
        successors = {}
        for label, target in block_exit.targets.items():
            successors[label] = self.block_for(target, passed)
        block.successors = (successors[None][0] if block_exit.kind == GOTO else {
            label: successor
            for label, (successor, _) in successors.items()
        })
        return successors

    # ------------------------------------------------------------------------------------------ exits
    def end_block(self, translator, kind: str, targets: Dict[Any, int], predicate=None, value=None) -> None:
        """Ends the block being traced by returning the values its successors need from the continuation."""
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
        live = sorted({
            name
            for target in targets.values()
            for name in self.info.livevars(target) if name in translator.symbolic_locals
        })
        values = {name: translator.symbolic_locals[name].realize() for name in live}
        names = [name for name in live if _is_graph_value(values[name])]
        side = {name: vt for name, vt in values.items() if name not in names}
        self.active.exit = _Exit(kind, dict(targets), names, side=side)
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
        record = transport.register(
            lambda cfg_id: CfgRecord(cfg_id, self.blocks, predicate_binding.index, self.entry_successors, self.
                                     entry_bindings, self.return_examples, tx.f_code.co_name))
        proxy = tx.output.create_proxy('call_function', torch.ops.dace.cfg.default,
                                       (record.id, list(self._outer_tensors), list(self._outer_syms)), {})
        result = wrap_fx_proxy(tx, proxy, example_value=list(self.return_examples))
        items = list(result.unpack_var_sequence(tx)) if hasattr(result, 'unpack_var_sequence') else list(result.items)
        return items[0] if self.return_count is None else TupleVariable(items)

    def capture(self, predicate) -> Any:
        """Traces the region and returns the frame's return value (the result of the ``dace::cfg`` call)."""
        self._predicate_binding = self.bind_outer(predicate.as_proxy())
        self.trace()
        return self.emit()


# ---------------------------------------------------------------------------------------------- helpers
def _is_graph_value(vt) -> bool:
    return isinstance(vt, (TensorVariable, SymNodeVariable))


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
    if vt.is_python_constant():
        value = vt.as_python_constant()
        try:
            hash(value)
            return ('constant', type(value).__name__, value)
        except TypeError:
            return ('constant', type(value).__name__, repr(value))
    return ('object', type(vt).__name__, vt.source.name if vt.source is not None else id(vt))


def _return_from(translator, value) -> None:
    """Makes the translator return ``value`` from its frame."""
    translator.push(value)
    translator.RETURN_VALUE(bt.create_instruction('RETURN_VALUE'))


_CONTINUATIONS: Dict[types.CodeType, Tuple[types.CodeType, int]] = {}


def _continuation(info: CodeInfo, start: int, argnames: List[str]) -> Tuple[types.CodeType, int]:
    """
    A code object that runs ``info.code`` from instruction ``start`` with the locals ``argnames`` as positional
    parameters. A prefix (``RESUME`` on 3.11+, then a jump) is prepended; returns the code and the prefix length.
    """
    if info.code.co_freevars or info.code.co_cellvars:
        raise CaptureFallback('frames with cell or free variables are not supported yet')
    prefix_len = 0

    def transform(instructions: List[bt.Instruction], code_options: Dict[str, Any]) -> None:
        nonlocal prefix_len
        prefix = [bt.create_instruction('RESUME', arg=0)] if sys.version_info >= (3, 11) else []
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


def _capture_region(backend: 'ControlFlowBackend', tx, inst, value, original: Callable):
    key = _position_key(tx, inst)
    if key in backend.blacklist or len(tx.stack) != 1 or tx.block_stack:
        return original(tx, inst)
    try:
        region = Region(tx, CodeInfo(tx.f_code), tx.indexof[inst])
    except CaptureFallback as ex:
        backend.log('fallback', tx.f_code.co_name, str(ex))
        return original(tx, inst)
    backend.regions.append(region)
    try:
        with torch._dynamo.config.patch(**CAPTURE_CONFIG):  # Blocks keep .item() (e.g., of float attributes)
            result = region.capture(value)
    except Exception as ex:  # noqa: BLE001 - also Unsupported raised by nested speculation
        if isinstance(ex, _PASSTHROUGH_EXCEPTIONS) and not isinstance(ex, _SPECULATION_ERRORS):
            raise  # Dynamo control flow (returns, restarts, exceptions raised by the program)
        # Speculation may have changed Dynamo's state: blacklist the jump and restart the analysis of the frame; on
        # the next pass Dynamo's stock handler graph-breaks here
        backend.blacklist.add(key)
        message = (str(ex).splitlines() or [''])[0][:200]
        backend.log('error', tx.f_code.co_name, f'{type(ex).__name__}: {message}')
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
                    if self.stack:
                        raise CaptureFallback(f'values on the stack at block boundary {index}')
                    try:
                        region.end_block(self, GOTO, {None: index})
                    except sc.ReturnValueOp:
                        pass  # The block function returned (``step`` catches this for instruction handlers)
                    return False
        return original(self)

    return step


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
            base = sc.InstructionTranslatorBase
            cls._saved.append((base, 'step', base.step))
            base.step = _make_step(base.step)

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


class ControlFlowBackend(DaceBackend):
    """
    ``DaceBackend`` that captures data-dependent ``if``/``while`` (and every other control flow in the rest of the
    frame) as a control-flow graph instead of graph-breaking (EXPERIMENTAL).
    """

    def __init__(self, **options):
        super().__init__(**options)
        self.regions: List[Region] = []
        self.blacklist: set = set()
        self.events: List[Tuple[str, str, str]] = []  #: (kind, function name, detail)
        _Patches.install()

    def log(self, kind: str, where: str, detail: str = '') -> None:
        self.events.append((kind, where, detail))

    def kinds(self) -> List[str]:
        return [kind for kind, _, _ in self.events]
