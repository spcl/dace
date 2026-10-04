# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
EXPERIMENTAL: capture plain Python ``if``/``while`` on tensor data inside TorchDynamo.

Dynamo graph-breaks at a data-dependent conditional jump (``generic_jump`` in ``torch/_dynamo/symbolic_convert.py``):
it compiles the prefix graph and emits glue bytecode that calls one of two lazily traced *continuation* functions.
This module patches the jump handlers (and only those) in the translators' per-class ``dispatch_table`` -- installed
process-wide when a :class:`CfgDaceBackend` is created and inert for every other backend, because Dynamo's documented
``backend_ctx_ctor`` hook is unreachable through ``torch.compile`` (see :class:`ControlFlowCapture`) -- and instead

- ``if``: tail-duplicates the rest of the frame into the two continuations and traces both **symbolically at capture
  time** (fake tensors, nothing is executed) through
  ``torch.cond`` (``speculate_subgraph``), so both sides share the root ``OutputGraph`` (ShapeEnv, fake mode,
  guards). The result of ``cond`` is the frame's return value.
- ``while`` (a jump recognised as a loop guard by a natural-loop analysis of the bytecode): traces the loop body as
  the ``body_fn`` of ``torch.while_loop`` with the predicate as an extra carried flag; the body continuation runs
  until it re-reaches the loop's back-edge jump, where it returns ``(predicate, *carried locals)``. After the HOP the
  translator binds the carried locals to the loop outputs and continues at the loop exit, so nothing after the loop
  is duplicated.

Both shapes reach the DaCe importer as the existing ``cond``/``while_loop`` HOPs, which
``dace.frontend.ml.torch.dynamo.ops.control_flow`` lowers to ``IfScope``/``WhileScope`` -- no ``GBlock`` is needed
for reducible control flow. Anything the prototype cannot express (non-empty value stack, early exits such as
``break``/``return`` inside a loop, loops with two different exits, cell/free variables, non-tensor live locals that
are not constants/SymInts) raises :class:`CaptureFallback` and falls back to Dynamo's stock behaviour at that jump.

The module is deliberately self-contained and must not be imported by the production backend.
"""
from __future__ import annotations

import contextlib
import dataclasses
import dis
import functools
import inspect
import sys
import threading
import types
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch._dynamo import bytecode_analysis, bytecode_transformation as bt
from torch._dynamo import exc
from torch._dynamo import symbolic_convert as sc

from ..backend import DaceBackend

#: Exceptions Dynamo uses for control flow; they must pass through the capture handlers untouched
_PASSTHROUGH_EXCEPTIONS = tuple(t
                                for t in (getattr(sc, 'ReturnValueOp', None), getattr(sc, 'YieldValueOp', None),
                                          getattr(exc, 'ObservedException', None),
                                          getattr(exc, 'RestartAnalysis', None),
                                          getattr(exc, 'TensorifyScalarRestartAnalysis', None),
                                          getattr(exc, 'SkipFrame', None), getattr(exc, 'BackendCompilerFailed', None))
                                if isinstance(t, type))

#: Errors raised while speculatively tracing a capture that subclass a passthrough exception but must still take the
#: fallback path. On torch >= 2.14, ``get_fake_value`` wraps fake-tensor ``RuntimeError``s (e.g. ``cond`` branches
#: with mismatched outputs) in ``FakeTensorObservedException``, a subclass of ``ObservedException``; older versions
#: lack the class (resolved with ``getattr`` because Dynamo's exception set differs across torch versions).
_SPECULATION_ERRORS = tuple(t for t in (getattr(exc, 'FakeTensorObservedException', None), ) if isinstance(t, type))

# ---------------------------------------------------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------------------------------------------------


class CaptureFallback(Exception):
    """Raised internally when a jump cannot be captured; the stock Dynamo handler runs instead."""


@dataclasses.dataclass
class CaptureEvent:
    kind: str  #: 'if' | 'while' | 'backedge' | 'fallback' | 'error'
    code_name: str
    index: int  #: instruction index in the original (un-prefixed) code object
    detail: str = ''


@dataclasses.dataclass
class LoopCapture:
    """A ``while_loop`` capture in progress: its body continuation is being traced."""
    code: types.CodeType  #: original code object
    jump_index: int  #: index of the guard jump
    body_index: int  #: index of the first body instruction (the loop entry)
    exit_index: int  #: index of the first instruction after the loop
    carried: List[str]  #: carried local names (in order, after the predicate flag)
    loop: Any = None  #: the natural loop (``_Loop``) guarded by the jump


@dataclasses.dataclass
class _Loop:
    header: int
    body: Set[int]
    stores: Set[str]


class _CodeInfo:
    """Cleaned instruction list of an original code object plus a natural-loop analysis of its CFG."""

    def __init__(self, code: types.CodeType):
        self.code = code
        self.instructions: List[bt.Instruction] = bt.cleaned_instructions(code)
        self.indexof = {inst: i for i, inst in enumerate(self.instructions)}
        n = len(self.instructions)
        self.succ: List[List[int]] = []
        for i, inst in enumerate(self.instructions):
            target = self.indexof[inst.target] if inst.target is not None else None
            if inst.opcode in bytecode_analysis.TERMINAL_OPCODES:
                s = [target] if target is not None else []
            elif inst.opcode in bytecode_analysis.JUMP_OPCODES:
                s = [i + 1, target]
            else:
                s = [i + 1] if i + 1 < n else []
            self.succ.append([x for x in s if x is not None and x < n])
        self.targets = {self.indexof[inst.target] for inst in self.instructions if inst.target is not None}
        self.loops = self._natural_loops()

    # -- live variables -----------------------------------------------------------------------------------------------
    def livevars(self, index: int) -> Set[str]:
        return bytecode_analysis.livevars_analysis(self.instructions, self.instructions[index])

    # -- loops --------------------------------------------------------------------------------------------------------
    def _natural_loops(self) -> List[_Loop]:
        n = len(self.instructions)
        preds: List[List[int]] = [[] for _ in range(n)]
        for u, ss in enumerate(self.succ):
            for v in ss:
                preds[v].append(u)
        # Iterative dominator computation (instruction-level; the frames of interest are small)
        full = set(range(n))
        dom: List[Set[int]] = [{0}] + [set(full) for _ in range(n - 1)]
        changed = True
        while changed:
            changed = False
            for v in range(1, n):
                if not preds[v]:
                    new = {v}
                else:
                    new = set(full)
                    for p in preds[v]:
                        new &= dom[p]
                    new |= {v}
                if new != dom[v]:
                    dom[v] = new
                    changed = True
        loops: Dict[int, Set[int]] = {}
        for u in range(n):
            for v in self.succ[u]:
                if v in dom[u]:  # back-edge u -> v
                    body = {v}
                    stack = [u]
                    while stack:
                        x = stack.pop()
                        if x in body:
                            continue
                        body.add(x)
                        stack.extend(preds[x])
                    loops.setdefault(v, set()).update(body)
        result = []
        for header, body in loops.items():
            stores: Set[str] = set()
            for k in body:
                inst = self.instructions[k]
                if 'STORE_FAST' in inst.opname or inst.opname == 'DELETE_FAST':
                    argval = inst.argval
                    stores.update(argval if isinstance(argval, tuple) else (argval, ))
            result.append(_Loop(header, body, stores))
        return result

    def in_header_block(self, loop: _Loop, index: int) -> bool:
        """True if ``index`` lies in the straight-line block that starts at the loop header."""
        if index < loop.header:
            return False
        for k in range(loop.header, index):
            inst = self.instructions[k]
            if inst.opcode in bytecode_analysis.JUMP_OPCODES or inst.opcode in bytecode_analysis.TERMINAL_OPCODES:
                return False
            if k + 1 in self.targets and k + 1 != loop.header:
                return False
        return True

    def guard_loop(self, index: int, succ_a: int, succ_b: int) -> Optional[Tuple[_Loop, int, int]]:
        """
        If the conditional jump at ``index`` guards a loop (one successor enters the innermost loop that does not
        contain the other), returns ``(loop, inside_successor, outside_successor)``.
        """
        candidates = [loop for loop in self.loops if (succ_a in loop.body) != (succ_b in loop.body)]
        if not candidates:
            return None
        loop = min(candidates, key=lambda l: len(l.body))
        inside, outside = (succ_a, succ_b) if succ_a in loop.body else (succ_b, succ_a)
        if index in loop.body and not self.in_header_block(loop, index):
            return None  # a conditional exit inside the loop (e.g. ``break``), not a guard
        return loop, inside, outside


class CaptureState:
    """Per-backend state: analysis caches, generated continuations, active loop captures and an event log."""

    def __init__(self):
        self.events: List[CaptureEvent] = []
        self.loop_stack: List[LoopCapture] = []
        self.blacklist: Set[Tuple] = set()  #: source positions of jumps whose capture failed
        self.continuations: Dict[types.CodeType, Tuple[types.CodeType, int]] = {}  #: cont. code -> (orig, prefix)
        self.infos: Dict[types.CodeType, _CodeInfo] = {}

    def info(self, code: types.CodeType) -> _CodeInfo:
        if code not in self.infos:
            self.infos[code] = _CodeInfo(code)
        return self.infos[code]

    def log(self, kind: str, code: types.CodeType, index: int, detail: str = '') -> None:
        self.events.append(CaptureEvent(kind, code.co_name, index, detail))

    def kinds(self) -> List[str]:
        return [e.kind for e in self.events]

    def resolve(self, tx) -> Tuple[_CodeInfo, int]:
        """Maps a translator to the original code object and the prefix length of its (continuation) code."""
        code = tx.f_code
        orig, prefix = self.continuations.get(code, (code, 0))
        info = self.info(orig)
        if len(tx.instructions) != len(info.instructions) + prefix:
            raise CaptureFallback(f'instruction count mismatch ({len(tx.instructions)} vs '
                                  f'{len(info.instructions)} + {prefix})')
        return info, prefix


# ---------------------------------------------------------------------------------------------------------------------
# Continuation code objects
# ---------------------------------------------------------------------------------------------------------------------


def make_continuation(state: CaptureState,
                      info: _CodeInfo,
                      resume_index: int,
                      argnames: Sequence[str],
                      f_globals: dict,
                      stack_prefix: Sequence[bt.Instruction] = ()) -> types.FunctionType:
    """
    Builds a function that runs ``info.code`` from instruction ``resume_index`` with the locals ``argnames`` passed
    positionally. Only a two-instruction prefix (``RESUME``; jump) is prepended, so the original instructions keep
    their indices shifted by the prefix length. ``stack_prefix`` instructions run before the jump (used to seed
    the value stack).
    """
    if info.code.co_freevars or info.code.co_cellvars:
        raise CaptureFallback('frames with cell/free variables are not supported by the prototype')
    prefix_len = 0

    def transform(instructions: List[bt.Instruction], code_options: Dict[str, Any]) -> None:
        nonlocal prefix_len
        target = instructions[resume_index]
        prefix: List[bt.Instruction] = []
        if sys.version_info >= (3, 11):
            prefix.append(bt.create_instruction('RESUME', arg=0))
        prefix.extend(stack_prefix)
        prefix.append(bt.create_jump_absolute(target))
        prefix_len = len(prefix)
        instructions[:0] = prefix
        name = f'__dace_cont_{code_options["co_name"]}_{resume_index}'
        code_options['co_argcount'] = len(argnames)
        code_options['co_posonlyargcount'] = 0
        code_options['co_kwonlyargcount'] = 0
        code_options['co_varnames'] = tuple(argnames) + tuple(v
                                                              for v in code_options['co_varnames'] if v not in argnames)
        code_options['co_flags'] &= ~(inspect.CO_VARARGS | inspect.CO_VARKEYWORDS)
        code_options['co_name'] = name
        if 'co_qualname' in code_options:
            code_options['co_qualname'] = name

    code, _ = bt.transform_code_object(info.code, transform)
    state.continuations[code] = (info.code, prefix_len)
    return types.FunctionType(code, f_globals, code.co_name)


# ---------------------------------------------------------------------------------------------------------------------
# Jump capture
# ---------------------------------------------------------------------------------------------------------------------

#: Conditional jumps on the truth value of TOS (by Python version); maps to ``truth_fn(True)`` == "jumps when true"
_JUMP_ON_TRUTH = {
    'POP_JUMP_IF_FALSE': False,
    'POP_JUMP_IF_TRUE': True,
    'POP_JUMP_FORWARD_IF_FALSE': False,
    'POP_JUMP_BACKWARD_IF_FALSE': False,
    'POP_JUMP_FORWARD_IF_TRUE': True,
    'POP_JUMP_BACKWARD_IF_TRUE': True,
}

_FLAG = '___dace_loop_flag'


def _while_cond_fn(flag, *rest):
    # ``clone``: while_loop rejects subgraph outputs that alias inputs (``supports_aliasing`` is False for the HOP)
    return flag.clone()


def _fresh(tx, vt):
    """
    Clones a tensor output that would alias a subgraph input: a placeholder of the current subgraph, or a value of an
    outer graph (``speculate_subgraph`` lifts those into placeholders lazily, on first use -- a value that is only
    returned is lifted *as* the output, which AOTAutograd's HOP functionalization rejects as aliasing).
    """
    from torch._dynamo.variables import TensorVariable
    if isinstance(vt, TensorVariable):
        node = vt.as_proxy().node
        if node.op == 'placeholder' or node.graph is not tx.output.current_tracer.graph:
            return vt.call_method(tx, 'clone', [], {})
    return vt


def _position_key(tx, inst: bt.Instruction) -> Tuple:
    pos = getattr(inst, 'positions', None)
    if pos is not None and pos.lineno is not None:
        return (tx.f_code.co_filename, pos.lineno, pos.col_offset, pos.end_col_offset, inst.opname)
    return (tx.f_code.co_filename, inst.starts_line, inst.opname)


def _local_operands(tx, names: Sequence[str]):
    from torch._dynamo.variables import ConstantVariable, SymNodeVariable, TensorVariable
    cells = set(tx.cell_and_freevars())
    operands = []
    for name in names:
        if name in cells:
            raise CaptureFallback(f'live local {name!r} is a cell variable')
        vt = tx.symbolic_locals[name].realize()
        if not isinstance(vt, (TensorVariable, ConstantVariable, SymNodeVariable)):
            raise CaptureFallback(f'live local {name!r} is a {type(vt).__name__}, not a tensor/SymInt/constant')
        operands.append(vt)
    return operands


def _logical_not(tx, value):
    from torch._dynamo.variables.torch import TorchInGraphFunctionVariable
    return TorchInGraphFunctionVariable(torch.logical_not).call_function(tx, [value], {})


def _return_from(tx, value) -> None:
    """Makes the translator return ``value`` from its frame (root: compiles the graph; inlined: symbolic result)."""
    tx.push(value)
    tx.RETURN_VALUE(bt.create_instruction('RETURN_VALUE'))


def _hop(name):
    from torch._dynamo.variables.higher_order_ops import TorchHigherOrderOperatorVariable
    return TorchHigherOrderOperatorVariable.make(getattr(torch.ops.higher_order, name))


def capture_jump(tx, inst: bt.Instruction, value, jumps_on_true: bool, state: CaptureState) -> None:
    """Handles a data-dependent conditional jump whose predicate ``value`` (a tensor) is TOS."""
    from torch._dynamo.variables import TupleVariable

    info, prefix = state.resolve(tx)
    j = tx.indexof[inst] - prefix
    if _position_key(tx, inst) in state.blacklist:
        raise CaptureFallback('a previous capture of this jump failed')
    if len(tx.stack) != 1:
        raise CaptureFallback(f'value stack is not empty below the predicate ({len(tx.stack) - 1} entries)')
    if tx.block_stack:
        raise CaptureFallback('jump inside a with/try block')
    next_j, target_j = j + 1, tx.indexof[inst.target] - prefix
    t_succ, f_succ = (target_j, next_j) if jumps_on_true else (next_j, target_j)

    # 1. Back-edge of the loop whose body is being traced: return (predicate, *carried) from the body continuation
    if state.loop_stack:
        cap = state.loop_stack[-1]
        if cap.code is info.code and cap.body_index in (t_succ, f_succ):
            if tx.parent is None:
                raise CaptureFallback('back-edge reached in the root translator')
            other = f_succ if t_succ == cap.body_index else t_succ
            if other != cap.exit_index:
                raise CaptureFallback(f'loop with two different exits ({other} vs {cap.exit_index})')
            tx.pop()
            pred_continue = value if t_succ == cap.body_index else _logical_not(tx, value)
            state.log('backedge', info.code, j, f'carried={cap.carried}')
            _return_from(tx, TupleVariable([pred_continue] + [_fresh(tx, tx.symbolic_locals[n]) for n in cap.carried]))
            return
        if any(c.code is info.code and c.jump_index == j for c in state.loop_stack):
            raise CaptureFallback('re-entered an active loop guard through another path (irreducible)')

    guard = info.guard_loop(j, t_succ, f_succ)
    if guard is not None and state.loop_stack:
        cap = state.loop_stack[-1]
        if cap.code is info.code and cap.loop is not None and guard[0].header == cap.loop.header:
            guard = None  # a conditional exit (``break``) of the loop being captured, not a new loop
    if guard is not None:
        _capture_while(tx, inst, value, info, prefix, j, guard, jumps_on_true, state)
    else:
        _capture_if(tx, inst, value, info, prefix, j, t_succ, f_succ, state)


def _capture_if(tx, inst, value, info: _CodeInfo, prefix: int, j: int, t_succ: int, f_succ: int,
                state: CaptureState) -> None:
    from torch._dynamo.variables import TupleVariable, UserFunctionVariable

    live = info.livevars(t_succ) | info.livevars(f_succ)
    for cap in state.loop_stack:  # the continuations may end at an enclosing back-edge/exit, which read the carries
        if cap.code is info.code:
            live |= set(cap.carried) | {_FLAG}
    names = sorted(n for n in live if n in tx.symbolic_locals)
    operands = _local_operands(tx, names)
    true_fn = make_continuation(state, info, t_succ, names, tx.f_globals)
    false_fn = make_continuation(state, info, f_succ, names, tx.f_globals)

    tx.pop()
    state.log('if', info.code, j, f'live={names}')
    result = _hop('cond').call_function(
        tx, [value, UserFunctionVariable(true_fn),
             UserFunctionVariable(false_fn),
             TupleVariable(operands)], {})
    if tx.parent is None:
        # The root frame returns the cond result: its locals are dead (prune_dead_locals would keep the ones live at
        # the jump and codegen them as extra graph outputs).
        cells = set(tx.cell_and_freevars())
        tx.symbolic_locals = {k: v for k, v in tx.symbolic_locals.items() if k in cells}
    _return_from(tx, result)


def _capture_while(tx, inst, value, info: _CodeInfo, prefix: int, j: int, guard, jumps_on_true: bool,
                   state: CaptureState) -> None:
    from torch._dynamo.variables import TupleVariable, UserFunctionVariable

    loop, body_j, exit_j = guard
    live = info.livevars(body_j) | info.livevars(exit_j)
    names = sorted(n for n in live if n in tx.symbolic_locals)
    carried = [n for n in names if n in loop.stores]
    additional = [n for n in names if n not in loop.stores]
    # Carried operands are cloned: AOTAutograd rejects a while_loop whose carried and additional inputs alias
    # (e.g. ``inner = acc`` right before an inner loop makes the same tensor a carry and an additional input)
    carried_vts = [vt.call_method(tx, 'clone', [], {}) if vt.is_tensor() else vt for vt in _local_operands(tx, carried)]
    additional_vts = _local_operands(tx, additional)
    body_fn = make_continuation(state, info, body_j, [_FLAG] + carried + additional, tx.f_globals)

    tx.pop()
    # The flag means "run (another) iteration": invert the predicate when the loop is entered on the false edge
    # (e.g. ``while True: ...; if t: break`` guarded by its break test)
    next_j = j + 1
    enters_on_true = (body_j == (tx.indexof[inst.target] - prefix)) == jumps_on_true if body_j != next_j else \
        (not jumps_on_true)
    init_flag = value if enters_on_true else _logical_not(tx, value)
    cap = LoopCapture(info.code, j, body_j, exit_j, carried, loop)
    state.loop_stack.append(cap)
    state.log('while', info.code, j, f'carried={carried} additional={additional}')
    try:
        result = _hop('while_loop').call_function(tx, [
            UserFunctionVariable(_while_cond_fn),
            UserFunctionVariable(body_fn),
            TupleVariable([init_flag] + carried_vts),
            TupleVariable(additional_vts),
        ], {})
    finally:
        state.loop_stack.pop()
    outputs = result.unpack_var_sequence(tx) if hasattr(result, 'unpack_var_sequence') else list(result.items)
    if len(outputs) != 1 + len(carried):
        raise CaptureFallback(f'while_loop returned {len(outputs)} values, expected {1 + len(carried)}')
    for name, vt in zip(carried, outputs[1:]):
        tx.symbolic_locals[name] = vt
    # Continue after the loop (mirrors InstructionTranslatorBase.jump)
    tx.instruction_pointer = exit_j + prefix
    tx.start_point = tx.instruction_pointer


def _make_jump_handler(orig: Callable, jumps_on_true: bool) -> Callable:

    @functools.wraps(orig)
    def handler(self, inst):
        from torch._dynamo.variables import TensorVariable
        backend = _backend_of(self)
        if not isinstance(backend, CfgDaceBackend) or not self.stack:
            return orig(self, inst)
        state = backend.capture
        value = self.stack[-1].realize()
        if not isinstance(value, TensorVariable):
            return orig(self, inst)
        # ``assert tensor_expr``: leave it to Dynamo's own rewriting (``_assert_async``); the false path only raises
        following = self.instructions[self.indexof[inst] +
                                      1] if self.indexof[inst] + 1 < len(self.instructions) else None
        if following is not None and following.opname in ('LOAD_ASSERTION_ERROR', 'LOAD_COMMON_CONSTANT'):
            return orig(self, inst)
        try:
            capture_jump(self, inst, value, jumps_on_true, state)
        except CaptureFallback as e:
            state.log('fallback', self.f_code, self.indexof[inst], str(e))
            return orig(self, inst)
        except Exception as e:  # noqa: BLE001 - Unsupported from nested speculation, or HOP validation errors
            if isinstance(e, _PASSTHROUGH_EXCEPTIONS) and not isinstance(e, _SPECULATION_ERRORS):
                raise  # Dynamo control flow (ReturnValueOp from _return_from, restarts, observed user exceptions)
            # The graph may have been mutated: blacklist the jump and restart the analysis of the frame; on the next
            # pass the stock handler graph-breaks here. (Re-raising, e.g. as ``Unsupported``, would make Dynamo skip
            # the whole frame when the jump is in the root frame.) A failure nested in an enclosing capture's
            # speculation also restarts; the enclosing jump then fails on the next pass and is blacklisted in turn.
            state.blacklist.add(_position_key(self, inst))
            msg = (str(e).splitlines() or [''])[0][:200]
            state.log('error', self.f_code, self.indexof[inst], f'{type(e).__name__}: {msg}')
            if isinstance(e, exc.UserError):
                raise
            raise exc.RestartAnalysis(restart_reason=f'dace control-flow capture failed: {type(e).__name__}: {msg}') \
                from e

    return handler


_UNCONDITIONAL_JUMPS = ('JUMP_FORWARD', 'JUMP_BACKWARD', 'JUMP_ABSOLUTE', 'JUMP_BACKWARD_NO_INTERRUPT')


def _make_exit_jump_handler(orig: Callable) -> Callable:
    """
    Unconditional jumps: inside a loop-body continuation, a jump to the loop exit (``break``) ends the body with the
    flag cleared, so the ``while_loop`` terminates and the translator continues at the exit as for a normal exit.
    """

    @functools.wraps(orig)
    def handler(self, inst):
        from torch._dynamo.variables import TupleVariable
        from torch._dynamo.variables.torch import TorchInGraphFunctionVariable
        backend = _backend_of(self)
        if isinstance(backend, CfgDaceBackend) and backend.capture.loop_stack and self.parent is not None:
            state = backend.capture
            cap = state.loop_stack[-1]
            try:
                info, prefix = state.resolve(self)
            except CaptureFallback:
                return orig(self, inst)
            if (cap.code is info.code and self.indexof[inst.target] - prefix == cap.exit_index and not self.stack
                    and not self.block_stack and _FLAG in self.symbolic_locals):
                flag = self.symbolic_locals[_FLAG]
                # ``flag != flag`` is a False scalar of the flag's dtype/device. (``torch.zeros_like(flag)`` would be
                # the obvious choice, but the importer's constant-fill lowering emits numpy-2's ``np.False_`` repr into
                # a Python tasklet for bool fills, which does not compile.)
                cleared = TorchInGraphFunctionVariable(torch.ne).call_function(self, [flag, flag], {})
                state.log('exit', info.code, self.indexof[inst] - prefix, f'carried={cap.carried}')
                _return_from(self,
                             TupleVariable([cleared] + [_fresh(self, self.symbolic_locals[n]) for n in cap.carried]))
                return
        return orig(self, inst)

    return handler


def _make_return_handler(orig: Callable) -> Callable:
    """
    ``RETURN_VALUE`` inside a continuation: outputs that alias the continuation's inputs (e.g. ``return z`` after an
    ``if`` that did not modify ``z``) are cloned. Dynamo tolerates the alias under ``no_grad`` but AOTAutograd's
    ``cond`` functionalization rejects it ("cond_true might be aliasing the input or the output").
    """

    @functools.wraps(orig)
    def handler(self, inst):
        from torch._dynamo.variables import ListVariable, TupleVariable
        backend = _backend_of(self)
        if (isinstance(backend, CfgDaceBackend) and self.parent is not None and self.stack
                and self.f_code in backend.capture.continuations):
            top = self.stack[-1]
            if isinstance(top, (TupleVariable, ListVariable)):
                self.stack[-1] = type(top)([_fresh(self, item) for item in top.items])
            else:
                self.stack[-1] = _fresh(self, top)
        return orig(self, inst)

    return handler


def _translator_classes() -> List[type]:
    result, stack = [], [sc.InstructionTranslatorBase]
    while stack:
        cls = stack.pop()
        result.append(cls)
        stack.extend(cls.__subclasses__())
    return result


class ControlFlowCapture(contextlib.AbstractContextManager):
    """
    Installs the capture handlers: swaps the conditional-jump entries of every translator class's ``dispatch_table``
    (one list per class, built by ``BytecodeDispatchTableMeta``). The handlers are inert for translators whose
    OutputGraph does not belong to a :class:`CfgDaceBackend`, so the patch can stay installed process-wide.

    Finding: Dynamo's documented ``backend_ctx_ctor`` hook cannot be used from the ``torch.compile(backend=...)``
    API in torch 2.13 -- ``torch.compile`` wraps the backend in ``torch._TorchCompileWrapper`` and ``_optimize`` in
    ``WrapBackendDebug`` before reading the attribute, and neither wrapper forwards it (``WrapBackendDebug`` copies the
    instance ``__dict__`` through ``functools.wraps``, ``_TorchCompileWrapper`` copies nothing). Hence the
    reference-counted global installation below; the context-manager interface is kept for explicit scoping.
    """
    _saved: List[Tuple[list, int, Callable]] = []
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
                for opname, jumps_on_true in _JUMP_ON_TRUTH.items():
                    op = dis.opmap.get(opname)
                    if op is None:
                        continue
                    cls._saved.append((table, op, table[op]))
                    table[op] = _make_jump_handler(table[op], jumps_on_true)
                for opname in _UNCONDITIONAL_JUMPS:
                    op = dis.opmap.get(opname)
                    if op is None:
                        continue
                    cls._saved.append((table, op, table[op]))
                    table[op] = _make_exit_jump_handler(table[op])
                for opname in ('RETURN_VALUE', ):
                    op = dis.opmap.get(opname)
                    if op is None:
                        continue
                    cls._saved.append((table, op, table[op]))
                    table[op] = _make_return_handler(table[op])

    @classmethod
    def uninstall(cls) -> None:
        with cls._lock:
            cls._refcount = max(0, cls._refcount - 1)
            if cls._refcount > 0:
                return
            for table, op, orig in reversed(cls._saved):
                table[op] = orig
            cls._saved.clear()

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, *exc):
        self.uninstall()
        return None


def _backend_of(tx) -> Any:
    """The user backend object behind a translator's OutputGraph (unwrapping Dynamo's backend wrappers)."""
    fn = tx.output.compiler_fn
    for _ in range(4):
        inner = getattr(fn, '_torchdynamo_orig_backend', None) or getattr(fn, 'compiler_fn', None)
        if inner is None or inner is fn:
            break
        fn = inner
    return fn


class CfgDaceBackend(DaceBackend):
    """``DaceBackend`` plus the experimental bytecode-level control-flow capture (``backend='dace'`` API unchanged)."""

    def __init__(self, **options):
        super().__init__(**options)
        self.capture = CaptureState()
        ControlFlowCapture.install()  # process-wide, gated per translator on the backend (see ControlFlowCapture)
