# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Bytecode analysis for control-flow capture: the control-flow graph of a code object as Dynamo sees it, the leaders
that split it into blocks, the value-stack depth at every instruction, and live variables.

Successors are normalized (:meth:`CodeInfo.resolve`) by following unconditional jumps and no-ops, so that the
different layouts CPython 3.10-3.14 emit for the same Python control flow yield the same graph.
"""
import dis
import sys
import types
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from torch._dynamo import bytecode_analysis
from torch._dynamo import bytecode_transformation as bt

#: Unconditional jumps (all versions)
UNCONDITIONAL_JUMPS = frozenset(('JUMP_FORWARD', 'JUMP_BACKWARD', 'JUMP_ABSOLUTE', 'JUMP_BACKWARD_NO_INTERRUPT'))
#: Instructions without effect that may sit between a jump and its target
NO_OPS = frozenset(('NOP', 'NOT_TAKEN'))
#: Conditional jumps on the truth value of the top of the stack, mapped to "jumps when true"
JUMP_ON_TRUTH = {
    'POP_JUMP_IF_FALSE': False,
    'POP_JUMP_IF_TRUE': True,
    'POP_JUMP_FORWARD_IF_FALSE': False,
    'POP_JUMP_BACKWARD_IF_FALSE': False,
    'POP_JUMP_FORWARD_IF_TRUE': True,
    'POP_JUMP_BACKWARD_IF_TRUE': True,
}


class CodeInfo:
    """Cleaned instructions (as Dynamo executes them) of a code object and their control-flow graph."""

    def __init__(self, code: types.CodeType):
        self.code = code
        self.instructions: List[bt.Instruction] = bt.cleaned_instructions(code)
        self.indexof = {inst: i for i, inst in enumerate(self.instructions)}
        n = len(self.instructions)
        self.successors: List[List[int]] = []
        for i, inst in enumerate(self.instructions):
            target = self.indexof[inst.target] if inst.target is not None else None
            if inst.opcode in bytecode_analysis.TERMINAL_OPCODES:
                successors = [target] if target is not None else []
            elif inst.opcode in bytecode_analysis.JUMP_OPCODES:
                successors = [i + 1, target]
            else:
                successors = [i + 1]
            self.successors.append([s for s in successors if s is not None and s < n])
        self._depths: Optional[Dict[int, int]] = None
        self._iterators: Optional[Dict[int, int]] = None
        self._dominators: Optional[Dict[int, Set[int]]] = None
        self._live: Optional[Dict[int, Set[str]]] = None
        self._loops: Dict[int, Set[int]] = {}

    def resolve(self, index: int) -> int:
        """
        Returns the instruction at which control actually continues when it reaches ``index``, skipping
        unconditional jumps and no-ops.
        """
        seen = set()
        while index < len(self.instructions) and index not in seen:
            seen.add(index)
            inst = self.instructions[index]
            if inst.opname in UNCONDITIONAL_JUMPS:
                index = self.indexof[inst.target]
            elif inst.opname in NO_OPS:
                index += 1
            else:
                break
        return index

    def branch_successors(self, index: int) -> Optional[Dict[bool, int]]:
        """For a conditional jump on a truth value, the resolved successors by truth value; otherwise ``None``."""
        inst = self.instructions[index]
        if inst.opname not in JUMP_ON_TRUTH:
            return None
        target, fallthrough = self.resolve(self.indexof[inst.target]), self.resolve(index + 1)
        jumps_on = JUMP_ON_TRUTH[inst.opname]
        return {jumps_on: target, not jumps_on: fallthrough}

    def region(self, entries: List[int]) -> Set[int]:
        """Instructions reachable from ``entries``."""
        seen: Set[int] = set()
        work = list(entries)
        while work:
            i = work.pop()
            if i in seen:
                continue
            seen.add(i)
            work.extend(self.successors[i])
        return seen

    def leaders(self, entries: List[int]) -> FrozenSet[int]:
        """
        The instructions that start a block in the region reachable from ``entries``: the entries, every resolved
        jump target, and every resolved successor of a conditional jump. Blocks end where control reaches another
        leader, so join points are never traced twice. A ``FOR_ITER`` is a branch between its loop body (which
        continues in the same block, with the next item on the stack) and its exit after the iterator is cleaned up
        (:meth:`for_iter_exit`).
        """
        result = {self.resolve(e) for e in entries}
        for i in self.region(entries):
            inst = self.instructions[i]
            if inst.opname == 'FOR_ITER':
                result.add(i)
                result.add(self.for_iter_exit(i))
            elif inst.target is not None and inst.opcode in bytecode_analysis.JUMP_OPCODES:
                result.add(self.resolve(self.indexof[inst.target]))
                if inst.opname not in UNCONDITIONAL_JUMPS:
                    result.add(self.resolve(i + 1))
        return frozenset(result)

    def for_iter_exit(self, index: int) -> int:
        """
        Where a loop continues once its ``FOR_ITER`` (at ``index``) is exhausted and the iterator is off the stack.
        The jump target is the cleanup on 3.12+ (``END_FOR``, then ``POP_TOP`` on 3.13 or ``POP_ITER`` on 3.14),
        which CPython skips on exhaustion.
        """
        i = self.indexof[self.instructions[index].target]
        if self.instructions[i].opname == 'END_FOR':
            i += 1
            if sys.version_info[:2] == (3, 13) and self.instructions[i].opname == 'POP_TOP':
                i += 1
        if self.instructions[i].opname == 'POP_ITER':
            i += 1
        return self.resolve(i)

    def dominators(self) -> Dict[int, Set[int]]:
        """The instructions that dominate each reachable instruction (every path from the start passes them)."""
        if self._dominators is None:
            nodes = sorted(self.region([0]))
            predecessors: Dict[int, List[int]] = {n: [] for n in nodes}
            for n in nodes:
                for s in self.successors[n]:
                    predecessors[s].append(n)
            everything = set(nodes)
            dominators = {n: set(everything) for n in nodes}
            dominators[0] = {0}
            changed = True
            while changed:
                changed = False
                for n in nodes[1:]:
                    preds = [dominators[p] for p in predecessors[n]]
                    new = (set.intersection(*preds) if preds else set()) | {n}
                    if new != dominators[n]:
                        dominators[n] = new
                        changed = True
            self._dominators = dominators
        return self._dominators

    def loop_body(self, header: int) -> Set[int]:
        """
        The natural loop of ``header``: the instructions that reach one of its back edges (an edge to ``header`` from
        an instruction it dominates) without passing through ``header``. Empty if ``header`` heads no loop.
        """
        if header not in self._loops:
            dominators = self.dominators()
            sources = [n for n, doms in dominators.items() if header in self.successors[n] and header in doms]
            predecessors: Dict[int, List[int]] = {}
            for n in dominators:
                for s in self.successors[n]:
                    predecessors.setdefault(s, []).append(n)
            body = {header} if sources else set()
            work = list(sources)
            while work:
                n = work.pop()
                if n in body:
                    continue
                body.add(n)
                work.extend(predecessors.get(n, []))
            self._loops[header] = body
        return self._loops[header]

    def is_loop_header(self, index: int) -> bool:
        return bool(self.loop_body(index))

    def loops_containing(self, index: int) -> List[int]:
        """The ``FOR_ITER`` instructions of the for loops whose body contains ``index``."""
        return [
            i for i, inst in enumerate(self.instructions) if inst.opname == 'FOR_ITER' and index in self.loop_body(i)
        ]

    def stored_names(self, indices: Set[int]) -> Set[str]:
        """Local variables assigned by the instructions."""
        return {
            self.instructions[i].argval
            for i in indices if self.instructions[i].opname in (
                'STORE_FAST', 'STORE_FAST_STORE_FAST') and isinstance(self.instructions[i].argval, str)
        }

    def stack_depths(self) -> Dict[int, int]:
        """The value-stack depth before every reachable instruction (from the start of the code object)."""
        if self._depths is None:
            self._analyze_stack()
        return self._depths

    def iterators(self, index: int) -> int:
        """
        How many values at the bottom of the stack before ``index`` are iterators of for loops (pushed by a
        ``GET_ITER`` with only iterators below it). A block can start where this equals the stack depth.
        """
        if self._iterators is None:
            self._analyze_stack()
        return self._iterators.get(index, 0)

    def _analyze_stack(self) -> None:
        depths: Dict[int, int] = {0: 0}
        iterators: Dict[int, int] = {0: 0}
        work = [0]
        while work:
            i = work.pop()
            inst = self.instructions[i]
            for s in self.successors[i]:
                jump = inst.target is not None and s == self.indexof[inst.target] and s != i + 1
                depth = depths[i] + bytecode_analysis.stack_effect(inst.opcode, inst.arg, jump=jump)
                if inst.opname == 'GET_ITER' and iterators[i] == depths[i] - 1:
                    count = iterators[i] + 1
                else:
                    count = min(iterators[i], depth)
                if s not in depths:
                    depths[s] = depth
                    iterators[s] = count
                    work.append(s)
        self._depths, self._iterators = depths, iterators

    def has_exception_handlers(self, indices: Set[int]) -> bool:
        """Whether any of the instructions is covered by an exception handler (``try``, ``with``) on 3.11+."""
        if sys.version_info < (3, 11):
            return any(self.instructions[i].opname in ('SETUP_FINALLY', 'SETUP_WITH') for i in indices)
        return any(self.instructions[i].exn_tab_entry is not None for i in indices)

    def livevars(self, index: int) -> Set[str]:
        """
        Local variables that may be read at or after ``index`` before being written (exact on the control-flow
        graph; Dynamo's own analysis keeps every variable of a loop body live).
        """
        if self._live is None:
            self._live = self._liveness()
        return self._live.get(index, set())

    def _liveness(self) -> Dict[int, Set[str]]:
        uses: Dict[int, Tuple[Set[str], Set[str]]] = {}  # Instruction -> (reads before writes, writes)
        for i, inst in enumerate(self.instructions):
            reads, writes = set(), set()
            if inst.opcode in dis.haslocal:
                names = inst.argval if isinstance(inst.argval, tuple) else (inst.argval, )
                if inst.opname.startswith('STORE_FAST_LOAD_FAST'):  # Store the first, then load the second
                    writes.add(names[0])
                    reads.update(n for n in names[1:] if n != names[0])
                elif 'STORE' in inst.opname:
                    writes.update(names)
                else:  # LOAD_FAST*, DELETE_FAST
                    reads.update(names)
            uses[i] = (reads, writes)
        live: Dict[int, Set[str]] = {i: set() for i in range(len(self.instructions))}
        changed = True
        while changed:
            changed = False
            for i in reversed(range(len(self.instructions))):
                reads, writes = uses[i]
                out = set().union(*(live[s] for s in self.successors[i])) if self.successors[i] else set()
                new = (out - writes) | reads
                if new != live[i]:
                    live[i] = new
                    changed = True
        return live


def opname(inst: bt.Instruction) -> str:
    return dis.opname[inst.opcode]
