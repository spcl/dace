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
from typing import Dict, FrozenSet, List, Optional, Set

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
        leader, so join points are never traced twice.
        """
        result = {self.resolve(e) for e in entries}
        for i in self.region(entries):
            inst = self.instructions[i]
            if inst.target is not None and inst.opcode in bytecode_analysis.JUMP_OPCODES:
                result.add(self.resolve(self.indexof[inst.target]))
                if inst.opname not in UNCONDITIONAL_JUMPS:
                    result.add(self.resolve(i + 1))
        return frozenset(result)

    def stack_depths(self) -> Dict[int, int]:
        """The value-stack depth before every reachable instruction (from the start of the code object)."""
        if self._depths is None:
            depths: Dict[int, int] = {0: 0}
            work = [0]
            while work:
                i = work.pop()
                inst = self.instructions[i]
                for s in self.successors[i]:
                    jump = inst.target is not None and s == self.indexof[inst.target] and s != i + 1
                    effect = bytecode_analysis.stack_effect(inst.opcode, inst.arg, jump=jump)
                    depth = depths[i] + effect
                    if s not in depths:
                        depths[s] = depth
                        work.append(s)
            self._depths = depths
        return self._depths

    def has_exception_handlers(self, indices: Set[int]) -> bool:
        """Whether any of the instructions is covered by an exception handler (``try``, ``with``) on 3.11+."""
        if sys.version_info < (3, 11):
            return any(self.instructions[i].opname in ('SETUP_FINALLY', 'SETUP_WITH') for i in indices)
        return any(self.instructions[i].exn_tab_entry is not None for i in indices)

    def livevars(self, index: int) -> Set[str]:
        """Local variables that may be read at or after ``index`` before being written."""
        return bytecode_analysis.livevars_analysis(self.instructions, self.instructions[index])


def opname(inst: bt.Instruction) -> str:
    return dis.opname[inst.opcode]
