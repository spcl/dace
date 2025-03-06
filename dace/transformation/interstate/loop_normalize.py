# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
""" Loop normalization transformation """

import copy
import sympy as sp
from typing import Set, Optional

from dace import sdfg as sd, symbolic, properties
from dace.sdfg import SDFG, InterstateEdge
from dace.sdfg import utils as sdutil
from dace.sdfg.state import ControlFlowRegion, LoopRegion
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis
from dace.sdfg.nodes import CodeBlock
from dace.symbolic import pystr_to_symbolic


# Returns a list of all defined symbols in the loop body
def _defined_symbols(loop: LoopRegion) -> Set[str]:
    defined = set()
    for edge, _ in loop.all_edges_recursive():
        if isinstance(edge, InterstateEdge):
            defined.update(edge.data.assignments.keys())
    return defined


# Check if we can normalize loop init
def _can_normalize_init(loop: LoopRegion, sdfg: SDFG) -> bool:
    # Iteration variable not altered in the loop body and Init is not zero
    itervar = loop.loop_variable
    start = loop_analysis.get_init_assignment(loop)
    defined_syms = _defined_symbols(loop)
    return (
        itervar not in defined_syms
        and symbolic.resolve_symbol_to_constant(start, sdfg) != 0
    )


# Check if we can normalize loop step
def _can_normalize_step(loop: LoopRegion, sdfg: SDFG) -> bool:
    # Iteration variable not altered in the loop body, increment not altered in body, step does not contain iteration variable, and Step is not one
    itervar = loop.loop_variable
    step = loop_analysis.get_loop_stride(loop)
    defined_syms = _defined_symbols(loop)
    return (
        itervar not in defined_syms
        and step.free_symbols.isdisjoint(defined_syms)
        and step.free_symbols.isdisjoint({itervar})
        and symbolic.resolve_symbol_to_constant(step, sdfg) != 1
    )


# Modifies the condition of a loop by a shift and scale
def _modify_cond(condition, var, step, start, norm_start, norm_step):
    condition = pystr_to_symbolic(condition.as_string)
    itersym = pystr_to_symbolic(var)
    # Find condition by matching expressions
    end: Optional[sp.Expr] = None
    a = sp.Wild('a')
    op = ''
    match = condition.match(itersym < a)
    if match:
        op = '<'
        end = match[a]
    if end is None:
        match = condition.match(itersym <= a)
        if match:
            op = '<='
            end = match[a]
    if end is None:
        match = condition.match(itersym > a)
        if match:
            op = '>'
            end = match[a]
    if end is None:
        match = condition.match(itersym >= a)
        if match:
            op = '>='
            end = match[a]
    if len(op) == 0:
        raise ValueError('Cannot match loop condition for loop normalization')
    
    # Invert the operator for reverse loops
    is_reverse = step < 0
    if is_reverse:
        if op == '<':
            op = '>='
        elif op == '<=':
            op = '>'
        elif op == '>':
            op = '<='
        elif op == '>=':
            op = '<'
        
        # swap start and end
        start, end = end, start

        # negate step
        step = -step

    if norm_start and norm_step:
        cond = f"{itersym} {op} (({end}) - ({start})) / {step}"
    elif norm_start:
        cond = f"{itersym} {op} ({end}) - ({start})"
    elif norm_step:
      cond = f"{itersym} {op} ({end}) / {step}"
    else:
      raise ValueError("At least one of norm_start or norm_step must be True")
    
    if is_reverse:
        cond = f"{cond} + 1"

    return cond


@properties.make_properties
@xf.explicit_cf_compatible
class LoopNormalize(xf.MultiStateTransformation):
    """
    Normalizes a control flow loop to start from 0 and increment by 1. Partially normalizes the loop, if it is not possible to fully normalize it (e.g., if the loop has a non-constant increment).
    """

    loop = xf.PatternNode(LoopRegion)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.loop)]

    def annotates_memlets(self) -> bool:
        return True

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        # If loop information cannot be determined, fail.
        start = loop_analysis.get_init_assignment(self.loop)
        step = loop_analysis.get_loop_stride(self.loop)
        itervar = self.loop.loop_variable
        if start is None or step is None or itervar is None:
            return False

        # If we can normalize any part of the loop, return True
        if _can_normalize_init(self.loop, sdfg) or _can_normalize_step(self.loop, sdfg):
            return True

        # Otherwise, don't normalize
        return False

    def apply(self, graph: ControlFlowRegion, sdfg: sd.SDFG):
        norm_init = _can_normalize_init(self.loop, sdfg)
        norm_step = _can_normalize_step(self.loop, sdfg)

        start = loop_analysis.get_init_assignment(self.loop)
        step = loop_analysis.get_loop_stride(self.loop)
        itervar = self.loop.loop_variable

        # Create the conversion expression
        if norm_init and norm_step:
            val = f"{itervar} * {step} + {start}"
        elif norm_init:
            val = f"{itervar} + {start}"
        elif norm_step:
            val = f"{itervar} * {step}"
        else:
            raise ValueError("Loop cannot be normalized")

        # Replace each occurrence of the old iteration variable with the new one in the loop body, but not in the loop header
        new_iter = sdfg.find_new_symbol(f"{itervar}_norm")
        old_loop_init = copy.deepcopy(self.loop.init_statement)
        old_loop_cond = copy.deepcopy(self.loop.loop_condition)
        old_loop_step = copy.deepcopy(self.loop.update_statement)

        self.loop.replace_dict({itervar: new_iter}, replace_keys=False)
        self.loop.init_statement = old_loop_init
        self.loop.loop_condition = old_loop_cond
        self.loop.update_statement = old_loop_step

        # Add new state before the loop to compute the new iteration symbol
        start_state = self.loop.start_block
        self.loop.add_state_before(
            start_state, is_start_block=True, assignments={new_iter: val}
        )

        # Adjust loop header
        if norm_init:
            self.loop.init_statement = CodeBlock(f"{itervar} = 0")
        if norm_step:
            self.loop.update_statement = CodeBlock(f"{itervar} = {itervar} + 1")
        self.loop.loop_condition = CodeBlock(_modify_cond(self.loop.loop_condition, itervar, step, start, norm_init, norm_step))
