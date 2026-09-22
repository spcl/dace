# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pass that shrinks a loop's iteration range from a conditional that guards its whole body."""
import ast
import copy
import itertools
from functools import cmp_to_key
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Set, Tuple

import numpy as np
import sympy

import dace
from dace import SDFG, dtypes, symbolic
from dace.properties import CodeBlock, Property, make_properties
from dace.sdfg import InterstateEdge
from dace.sdfg import nodes as nd
from dace.sdfg.state import (BreakBlock, ConditionalBlock, ContinueBlock, ControlFlowRegion, LoopRegion, ReturnBlock,
                             SDFGState)
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.helpers import move_branch_cfg_up_discard_conditions
from dace.transformation.passes.analysis import loop_analysis

# Comparison operators the range analysis understands, with their negation and their mirror image (for when the
# iteration variable sits on the right-hand side).
_NEGATED_OP = {
    ast.Lt: ast.GtE,
    ast.LtE: ast.Gt,
    ast.Gt: ast.LtE,
    ast.GtE: ast.Lt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
}
_MIRRORED_OP = {
    ast.Lt: ast.Gt,
    ast.LtE: ast.GtE,
    ast.Gt: ast.Lt,
    ast.GtE: ast.LtE,
    ast.Eq: ast.Eq,
    ast.NotEq: ast.NotEq,
}

# Functions that may appear in a symbolic loop bound (they are understood by ``pystr_to_symbolic`` and by code
# generation of loop headers).
_SYMBOLIC_FUNCTIONS = {'min', 'max', 'Min', 'Max', 'int_ceil', 'int_floor', 'abs', 'Abs', 'ceiling', 'floor'}

# Upper bound on the number of conjunctive clauses a guard is allowed to expand into.
_MAX_CLAUSES = 64


class _Interval(NamedTuple):
    """A closed integer interval ``[lo, hi]``; ``None`` on either side means unbounded."""
    lo: Optional[sympy.Expr]
    hi: Optional[sympy.Expr]


_UNBOUNDED = _Interval(None, None)


class _LoopInfo(NamedTuple):
    loop: LoopRegion
    itervar: str
    start: sympy.Expr
    end: sympy.Expr  # Inclusive last iterate under normal termination
    stride: int
    op: str  # Comparison operator of the original loop condition ('<', '<=', '>', '>=')
    body_defined: Set[str]  # Symbols (re-)assigned anywhere inside the loop body

    @property
    def ascending(self) -> bool:
        return self.stride > 0

    @property
    def iteration_range(self) -> _Interval:
        return _Interval(self.start, self.end) if self.ascending else _Interval(self.end, self.start)


class _Range(NamedTuple):
    """One reduced loop: its iterate interval plus the guard atoms that could not be folded into the range."""
    interval: _Interval
    residual: List[ast.expr]
    clause: int
    position: int


class _Guard(NamedTuple):
    """The conditional guarding a loop body, as found by ``_find_guard``."""
    block: ConditionalBlock
    branch: ControlFlowRegion  # The single branch that has any effect
    condition: ast.expr  # Effective condition of that branch, with the prologue assignments substituted in
    prologue: List[Any]  # Inter-state edges on the path from the loop start to the conditional, in order
    prologue_symbols: Set[str]  # Symbols assigned on those edges


class _Substitute(ast.NodeTransformer):
    """Replace reads of symbols by expressions."""

    def __init__(self, replacements: Dict[str, ast.expr]):
        self.replacements = replacements

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load) and node.id in self.replacements:
            return copy.deepcopy(self.replacements[node.id])
        return node


def _num(expr) -> Optional[sympy.Number]:
    """``expr`` simplified, if it reduces to a concrete number, else ``None``."""
    try:
        simplified = sympy.simplify(expr)
    except Exception:
        return None
    return simplified if getattr(simplified, 'is_number', False) else None


def _intersect(a: _Interval, b: _Interval) -> _Interval:
    if a.lo is None:
        lo = b.lo
    elif b.lo is None:
        lo = a.lo
    else:
        lo = sympy.Max(a.lo, b.lo)
    if a.hi is None:
        hi = b.hi
    elif b.hi is None:
        hi = a.hi
    else:
        hi = sympy.Min(a.hi, b.hi)
    return _Interval(lo, hi)


def _provably_empty(iv: _Interval) -> bool:
    if iv.lo is None or iv.hi is None:
        return False
    diff = _num(iv.hi - iv.lo)
    return diff is not None and diff < 0


def _provably_before(a: _Interval, b: _Interval) -> bool:
    """Whether every point of ``a`` lies strictly below every point of ``b``."""
    if a.hi is None or b.lo is None:
        return False
    gap = _num(b.lo - a.hi)
    return gap is not None and gap > 0


def _mentions(node: ast.AST, name: str) -> bool:
    return any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(node))


def _expr_of(code: CodeBlock) -> ast.expr:
    node = code.code[0]
    if isinstance(node, ast.Expr):
        node = node.value
    return copy.deepcopy(node)


def _conjunction(atoms: Sequence[ast.expr]) -> ast.expr:
    if len(atoms) == 1:
        return atoms[0]
    return ast.BoolOp(op=ast.And(), values=list(atoms))


def _unparse(node: ast.AST) -> str:
    return ast.unparse(ast.fix_missing_locations(copy.deepcopy(node)))


# ----------------------------------------------------------------------------------------------------------------------
# Boolean normalization
# ----------------------------------------------------------------------------------------------------------------------


def _dnf(node: ast.AST, negate: bool = False) -> Optional[List[List[ast.expr]]]:
    """Disjunctive normal form of a Python boolean expression.

    :return: A list of clauses, each a list of atoms that must all hold; negation is pushed into the atoms (flipping
             comparison operators where possible). An empty clause is a tautology; an empty clause list is a
             contradiction. ``None`` if the expression blows up beyond ``_MAX_CLAUSES`` clauses.
    """
    if isinstance(node, ast.Expr):
        node = node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return _dnf(node.operand, not negate)
    if isinstance(node, ast.BoolOp):
        parts = [_dnf(v, negate) for v in node.values]
        if any(p is None for p in parts):
            return None
        # ``and`` is a conjunction, and so is a negated ``or`` (De Morgan).
        if isinstance(node.op, ast.And) != negate:
            clauses: List[List[ast.expr]] = [[]]
            for part in parts:
                clauses = [c1 + c2 for c1 in clauses for c2 in part]
                if len(clauses) > _MAX_CLAUSES:
                    return None
            return clauses
        return [clause for part in parts for clause in part]
    if isinstance(node, ast.Compare) and len(node.ops) > 1:
        # ``a < i < b`` is ``a < i and i < b``.
        operands = [node.left] + list(node.comparators)
        atoms = [
            ast.Compare(left=operands[k], ops=[node.ops[k]], comparators=[operands[k + 1]])
            for k in range(len(node.ops))
        ]
        return _dnf(ast.BoolOp(op=ast.And(), values=atoms), negate)
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return [[]] if node.value != negate else []
    if negate:
        if isinstance(node, ast.Compare) and type(node.ops[0]) in _NEGATED_OP:
            return [[ast.Compare(left=node.left, ops=[_NEGATED_OP[type(node.ops[0])]()], comparators=node.comparators)]]
        node = ast.UnaryOp(op=ast.Not(), operand=node)
    return [[node]]


# ----------------------------------------------------------------------------------------------------------------------
# Atom analysis
# ----------------------------------------------------------------------------------------------------------------------


def _symbolic_atom(atom: ast.expr, info: _LoopInfo) -> Optional[List[_Interval]]:
    """Intervals of the iteration variable on which ``atom`` holds, for atoms of the shape ``i <cmp> C`` or
    ``C <cmp> i`` with a loop-invariant, integer, symbolic ``C``. ``None`` if the atom is not of that shape."""
    if not isinstance(atom, ast.Compare) or len(atom.ops) != 1:
        return None
    opcls = type(atom.ops[0])
    if opcls not in _MIRRORED_OP:
        return None
    left, right = atom.left, atom.comparators[0]
    itervar = info.itervar
    if isinstance(left, ast.Name) and left.id == itervar and not _mentions(right, itervar):
        other = right
    elif isinstance(right, ast.Name) and right.id == itervar and not _mentions(left, itervar):
        other = left
        opcls = _MIRRORED_OP[opcls]
    else:
        return None

    sdfg = info.loop.sdfg
    for n in ast.walk(other):
        if isinstance(n, ast.Name):
            if n.id in sdfg.arrays or n.id in info.body_defined:
                return None  # Reads data or a symbol that changes inside the loop.
        elif isinstance(n, ast.Constant):
            if not isinstance(n.value, int) or isinstance(n.value, bool):
                return None
        elif isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name) or n.func.id not in _SYMBOLIC_FUNCTIONS:
                return None
        elif not isinstance(n, (ast.BinOp, ast.UnaryOp, ast.operator, ast.unaryop, ast.expr_context)):
            return None
    try:
        bound = symbolic.pystr_to_symbolic(_unparse(other))
    except Exception:
        return None
    # Fold in scalar compile-time constants.
    constants = sdfg.constants
    substitutions = {
        s: constants[str(s)]
        for s in bound.free_symbols if str(s) in constants and not isinstance(constants[str(s)], np.ndarray)
    }
    if substitutions:
        bound = bound.subs(substitutions)
    if bound.is_number and not bound.is_integer:
        return None
    for s in bound.free_symbols:
        if str(s) in sdfg.symbols and not np.issubdtype(sdfg.symbols[str(s)].type, np.integer):
            return None

    if opcls is ast.Lt:
        return [_Interval(None, bound - 1)]
    if opcls is ast.LtE:
        return [_Interval(None, bound)]
    if opcls is ast.Gt:
        return [_Interval(bound + 1, None)]
    if opcls is ast.GtE:
        return [_Interval(bound, None)]
    if opcls is ast.Eq:
        return [_Interval(bound, bound)]
    return [_Interval(None, bound - 1), _Interval(bound + 1, None)]  # NotEq


_CONSTANT_ATOM_NODES = (ast.Compare, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Name, ast.Constant, ast.Subscript,
                        ast.Tuple, ast.operator, ast.unaryop, ast.cmpop, ast.boolop, ast.expr_context)


def _constant_atom(atom: ast.expr, info: _LoopInfo, max_enumeration: int) -> Optional[List[_Interval]]:
    """Intervals of the iteration variable on which ``atom`` holds, for atoms whose only inputs are the iteration
    variable and compile-time constants (``sdfg.constants``), e.g. ``cst[i] > 0``. The atom is evaluated for every
    iterate in its finite domain and consecutive true iterates form the intervals. ``None`` if the atom is not
    of that kind or its domain cannot be bounded."""
    sdfg = info.loop.sdfg
    constants = sdfg.constants
    itervar = info.itervar
    for n in ast.walk(atom):
        if isinstance(n, ast.Name):
            if n.id != itervar and n.id not in constants:
                return None
        elif not isinstance(n, _CONSTANT_ATOM_NODES):
            return None

    try:
        code = compile(ast.fix_missing_locations(ast.Expression(body=copy.deepcopy(atom))), '<guard>', 'eval')
    except Exception:
        return None
    namespace: Dict[str, Any] = dict(constants)

    def evaluate(iterate: Optional[int]) -> Optional[bool]:
        if iterate is not None:
            namespace[itervar] = iterate
        try:
            return bool(eval(code, {'__builtins__': {}}, namespace))
        except Exception:
            return None

    if not _mentions(atom, itervar):
        # Loop-invariant constant expression: a tautology or a contradiction.
        value = evaluate(None)
        if value is None:
            return None
        return [_UNBOUNDED] if value else []

    # Bound the domain of the iteration variable from the constant arrays it indexes (an out-of-bounds index is
    # undefined behavior, so those iterates may be assumed never to be visited), and from a constant loop range.
    itersym = symbolic.pystr_to_symbolic(itervar)
    scalar_constants = {symbolic.pystr_to_symbolic(k): v for k, v in constants.items() if not isinstance(v, np.ndarray)}
    domain = _UNBOUNDED
    for n in ast.walk(atom):
        if not isinstance(n, ast.Subscript):
            continue
        if not isinstance(n.value, ast.Name) or not isinstance(constants.get(n.value.id, None), np.ndarray):
            return None
        array = constants[n.value.id]
        indices = list(n.slice.elts) if isinstance(n.slice, ast.Tuple) else [n.slice]
        if len(indices) != array.ndim:
            return None
        for dim, index in enumerate(indices):
            if isinstance(index, ast.Slice):
                return None
            if not _mentions(index, itervar):
                # Constant index: it must be in bounds (a negative index would silently wrap in Python).
                try:
                    value = eval(
                        compile(ast.fix_missing_locations(ast.Expression(body=copy.deepcopy(index))), '<index>',
                                'eval'), {'__builtins__': {}}, dict(constants))
                except Exception:
                    return None
                if not (0 <= int(value) < array.shape[dim]):
                    return None
                continue
            try:
                index_expr = symbolic.pystr_to_symbolic(_unparse(index)).subs(scalar_constants)
                slope = sympy.simplify(index_expr.diff(itersym))
                offset = sympy.simplify(index_expr - slope * itersym)
            except Exception:
                return None
            if not (slope.is_number and slope.is_integer and slope != 0 and offset.is_number and offset.is_integer):
                return None  # Only affine indices ``a * i + b`` with integer ``a``, ``b`` are supported.
            # In-bounds iterates: 0 <= a * i + b <= shape - 1.
            first = sympy.Rational(-offset, slope)
            last = sympy.Rational(array.shape[dim] - 1 - offset, slope)
            if slope < 0:
                first, last = last, first
            domain = _intersect(domain, _Interval(sympy.ceiling(first), sympy.floor(last)))
    start_num, end_num = _num(info.start), _num(info.end)
    if start_num is not None and end_num is not None:
        domain = _intersect(domain, _Interval(sympy.Min(start_num, end_num), sympy.Max(start_num, end_num)))
    lo, hi = _num(domain.lo), _num(domain.hi)
    if lo is None or hi is None:
        return None
    lo, hi = int(lo), int(hi)
    if hi < lo:
        return []
    if hi - lo + 1 > max_enumeration:
        return None

    # Only iterates the loop actually visits when the alignment (i.e., the start) is known.
    step = abs(info.stride)
    if step > 1 and start_num is not None:
        candidates = [k for k in range(lo, hi + 1) if (k - int(start_num)) % step == 0]
    else:
        candidates = list(range(lo, hi + 1))

    intervals: List[_Interval] = []
    run_start: Optional[int] = None
    previous: Optional[int] = None
    for k in candidates:
        value = evaluate(k)
        if value is None:
            return None
        if value:
            if run_start is None:
                run_start = k
            previous = k
        elif run_start is not None:
            intervals.append(_Interval(sympy.Integer(run_start), sympy.Integer(previous)))
            run_start = None
    if run_start is not None:
        intervals.append(_Interval(sympy.Integer(run_start), sympy.Integer(previous)))
    return intervals


# ----------------------------------------------------------------------------------------------------------------------
# Loop / structure analysis
# ----------------------------------------------------------------------------------------------------------------------


def _is_plain_empty_state(block) -> bool:
    return (isinstance(block, SDFGState) and not isinstance(block, (BreakBlock, ContinueBlock, ReturnBlock))
            and block.number_of_nodes() == 0)


def _is_empty_region(region: ControlFlowRegion) -> bool:
    """Whether executing ``region`` has no effect: only empty states and unconditional, assignment-free edges."""
    for block in region.nodes():
        if _is_plain_empty_state(block):
            continue
        if type(block) is ControlFlowRegion and _is_empty_region(block):
            continue
        return False
    return all(e.data.is_unconditional() and not e.data.assignments for e in region.edges())


def _has_direct_break(region) -> bool:
    """Whether ``region`` contains a ``break`` that targets it (breaks of nested loops do not count)."""
    for block in region.nodes():
        if isinstance(block, BreakBlock):
            return True
        if isinstance(block, LoopRegion):
            continue
        if isinstance(block, ConditionalBlock):
            if any(_has_direct_break(branch) for _, branch in block.branches):
                return True
        elif isinstance(block, ControlFlowRegion) and _has_direct_break(block):
            return True
    return False


def _symbols_defined_in(loop: LoopRegion) -> Set[str]:
    """Symbols assigned anywhere inside the loop body (inter-state edges and nested loop headers)."""
    defined: Set[str] = set()
    for edge, _ in loop.all_edges_recursive():
        if isinstance(edge.data, InterstateEdge):
            defined.update(edge.data.assignments.keys())
    for region in loop.all_control_flow_regions():
        if region is loop or not isinstance(region, LoopRegion):
            continue
        if region.loop_variable:
            defined.add(region.loop_variable)
        for stmt_block in (region.init_statement, region.update_statement):
            if stmt_block is None or stmt_block.language != dtypes.Language.Python:
                continue
            for stmt in stmt_block.code:
                for target in getattr(stmt, 'targets', []):
                    if isinstance(target, ast.Name):
                        defined.add(target.id)
    return defined


def _loop_info(loop: LoopRegion) -> Optional[_LoopInfo]:
    """Iteration variable, bounds and (constant, non-zero) stride of a canonical for-loop, or ``None``."""
    if loop.inverted or not loop.loop_variable or loop.sdfg is None:
        return None
    if any(code.language != dtypes.Language.Python for code in loop.get_meta_codeblocks()):
        return None
    start = loop_analysis.get_init_assignment(loop)
    stride_expr = loop_analysis.get_loop_stride(loop)
    if start is None or stride_expr is None:
        return None
    stride_value = symbolic.resolve_symbol_to_constant(stride_expr, loop.sdfg)
    if stride_value is None or float(stride_value) != int(stride_value) or int(stride_value) == 0:
        return None
    stride = int(stride_value)

    try:
        condition = symbolic.pystr_to_symbolic(loop.loop_condition.as_string)
    except Exception:
        return None
    itersym = symbolic.pystr_to_symbolic(loop.loop_variable)
    wild = sympy.Wild('a')
    op = None
    for candidate_op, pattern in (('<', itersym < wild), ('<=', itersym <= wild), ('>', itersym > wild), ('>=', itersym
                                                                                                          >= wild)):
        match = condition.match(pattern)
        if match:
            op = candidate_op
            bound = match[wild]
            break
    if op is None:
        return None
    if itersym in bound.free_symbols or itersym in start.free_symbols:
        return None
    if (stride > 0) != (op in ('<', '<=')):
        return None
    end = {'<': bound - 1, '<=': bound, '>': bound + 1, '>=': bound}[op]

    body_defined = _symbols_defined_in(loop)
    if loop.loop_variable in body_defined:
        return None
    # The reduced loops evaluate the original init expression again (and the guard bounds only once), so the
    # symbols they depend on must not change inside the loop.
    if any(str(s) in body_defined for s in itertools.chain(start.free_symbols, end.free_symbols)):
        return None
    return _LoopInfo(loop, loop.loop_variable, start, end, stride, op, body_defined)


def _find_guard(loop: LoopRegion) -> Optional[_Guard]:
    """The conditional that guards the whole loop body, its single live branch and that branch's effective condition.

    The loop body must consist of exactly one ``ConditionalBlock`` plus, possibly, empty states connected by
    unconditional edges. Edges on the path from the loop start to the conditional may carry symbol assignments (a
    guard prologue, e.g. ``cstarr_index = cstarr[i]`` as produced by the Python frontend); those are substituted into
    the condition. Exactly one branch of the conditional may have any effect.
    """
    conditionals = [n for n in loop.nodes() if isinstance(n, ConditionalBlock)]
    if len(conditionals) != 1:
        return None
    guard = conditionals[0]
    if any(n is not guard and not _is_plain_empty_state(n) for n in loop.nodes()):
        return None
    if any(not e.data.is_unconditional() for e in loop.edges()):
        return None
    if any(cond is not None and cond.language != dtypes.Language.Python for cond, _ in guard.branches):
        return None

    # The linear path from the loop start to the conditional.
    try:
        node = loop.start_block
    except ValueError:
        return None
    prologue = []
    seen = set()
    while node is not guard:
        if id(node) in seen:
            return None
        seen.add(id(node))
        out_edges = loop.out_edges(node)
        if len(out_edges) != 1:
            return None
        prologue.append(out_edges[0])
        node = out_edges[0].dst
    prologue_ids = {id(e) for e in prologue}
    if any(e.data.assignments and id(e) not in prologue_ids for e in loop.edges()):
        return None

    live = [(k, cond, branch) for k, (cond, branch) in enumerate(guard.branches) if not _is_empty_region(branch)]
    if len(live) != 1:
        return None
    index, cond, branch = live[0]
    # An if/elif/else chain: the live branch runs iff its condition holds and no earlier condition does.
    atoms: List[ast.expr] = [
        ast.UnaryOp(op=ast.Not(), operand=_expr_of(c)) for c, _ in guard.branches[:index] if c is not None
    ]
    if cond is not None:
        atoms.append(_expr_of(cond))
    elif not atoms:
        return None
    condition = _conjunction(atoms)

    # Fold the prologue assignments into the condition, last edge first, so that the condition is expressed in
    # terms of the values at the start of the iteration.
    prologue_symbols: Set[str] = set()
    for edge in reversed(prologue):
        assignments = edge.data.assignments
        if not assignments:
            continue
        try:
            replacements = {name: ast.parse(rhs, mode='eval').body for name, rhs in assignments.items()}
        except SyntaxError:
            return None
        # Assignments of one edge are evaluated together; a right-hand side reading a symbol assigned on the same
        # edge would make the substitution order-dependent.
        if any(_mentions(rhs, name) for rhs in replacements.values() for name in replacements):
            return None
        condition = _Substitute(replacements).visit(condition)
        prologue_symbols.update(assignments.keys())
    return _Guard(guard, branch, condition, prologue, prologue_symbols)


def _drop_dead_prologue_assignments(guard: _Guard) -> None:
    """Remove prologue assignments that only fed the guard condition, which has been folded into the loop range."""
    needed = set(guard.branch.used_symbols(all_symbols=True))
    remaining = [(edge, name, rhs) for edge in guard.prologue for name, rhs in edge.data.assignments.items()]
    while True:
        referenced = set(needed)
        for _, _, rhs in remaining:
            referenced |= set(symbolic.free_symbols_and_functions(rhs))
        dead = [entry for entry in remaining if entry[1] not in referenced]
        if not dead:
            return
        for edge, name, _ in dead:
            edge.data.assignments = {k: v for k, v in edge.data.assignments.items() if k != name}
        remaining = [entry for entry in remaining if entry[1] in referenced]


def _symbol_used_outside(loop: LoopRegion, symbol: str) -> bool:
    """Whether ``symbol`` (the loop variable or a symbol assigned in the loop) is read anywhere outside the loop,
    where the reduced loops would leave a different final value."""
    sdfg = loop.sdfg
    for desc in sdfg.arrays.values():
        if symbol in map(str, desc.free_symbols):
            return True
    child = loop
    graph = loop.parent_graph
    while graph is not None:
        if not isinstance(graph, SDFG):
            # Headers of enclosing loops / conditions of enclosing conditionals.
            if symbol in graph.used_symbols(all_symbols=True, with_contents=False):
                return True
        for node in graph.nodes():
            if node is child:
                continue
            used_symbols = getattr(node, 'used_symbols', None)
            if used_symbols is not None and symbol in used_symbols(all_symbols=True):
                return True
        for edge in graph.edges():
            if symbol in edge.data.free_symbols:
                return True
        if isinstance(graph, SDFG):
            break
        child, graph = graph, graph.parent_graph
    return False


# ----------------------------------------------------------------------------------------------------------------------
# Rewriting
# ----------------------------------------------------------------------------------------------------------------------


def _reparent_copied_region(region, sdfg: SDFG) -> None:
    """Restore the pointers ``copy.deepcopy`` drops on a control-flow subtree (see ``LoopUnroll``)."""
    blocks = [region] + list(region.all_control_flow_blocks())
    for block in blocks:
        block.sdfg = sdfg
    for state in region.all_states():
        for node in state.nodes():
            if isinstance(node, nd.NestedSDFG) and node.sdfg is not None:
                node.sdfg.parent = state
                node.sdfg.parent_sdfg = sdfg
                node.sdfg.parent_nsdfg_node = node


def _repair_moved_block(block, sdfg: SDFG) -> None:
    if isinstance(block, SDFGState):
        states = [] if isinstance(block, (BreakBlock, ContinueBlock, ReturnBlock)) else [block]
    elif isinstance(block, (BreakBlock, ContinueBlock, ReturnBlock)):
        states = []
    else:
        states = list(block.all_states())
    for state in states:
        for node in state.nodes():
            if isinstance(node, nd.NestedSDFG) and node.sdfg is not None:
                node.sdfg.parent = state
                node.sdfg.parent_sdfg = sdfg
                node.sdfg.parent_nsdfg_node = node


def _new_header(info: _LoopInfo, interval: _Interval) -> Tuple[Optional[str], Optional[str]]:
    """Init statement and condition of the loop restricted to ``interval`` (``None`` where unchanged)."""
    itervar, start, end, stride = info.itervar, info.start, info.end, info.stride
    init = condition = None
    if info.ascending:
        if interval.lo is not None:
            if stride == 1:
                first = sympy.Max(start, interval.lo)
            else:
                # First visited iterate at or above the lower bound. ``Max`` keeps the numerator non-negative,
                # which the C++ ``int_ceil`` requires.
                first = start + stride * symbolic.int_ceil(sympy.Max(0, interval.lo - start), stride)
            init = f'{itervar} = {symbolic.symstr(first)}'
        if interval.hi is not None:
            if info.op == '<':
                condition = f'{itervar} < {symbolic.symstr(sympy.Min(end + 1, interval.hi + 1))}'
            else:
                condition = f'{itervar} <= {symbolic.symstr(sympy.Min(end, interval.hi))}'
    else:
        step = -stride
        if interval.hi is not None:
            if step == 1:
                first = sympy.Min(start, interval.hi)
            else:
                first = start - step * symbolic.int_ceil(sympy.Max(0, start - interval.hi), step)
            init = f'{itervar} = {symbolic.symstr(first)}'
        if interval.lo is not None:
            if info.op == '>':
                condition = f'{itervar} > {symbolic.symstr(sympy.Max(end - 1, interval.lo - 1))}'
            else:
                condition = f'{itervar} >= {symbolic.symstr(sympy.Max(end, interval.lo))}'
    return init, condition


def _specialize_loop(new_loop: LoopRegion, info: _LoopInfo, rng: _Range) -> None:
    """Restrict ``new_loop`` (the original or a deep copy of it) to one range and strip the folded guard."""
    init, condition = _new_header(info, rng.interval)
    if init is not None:
        new_loop.init_statement = CodeBlock(init)
    if condition is not None:
        new_loop.loop_condition = CodeBlock(condition)

    guard = next(n for n in new_loop.nodes() if isinstance(n, ConditionalBlock))
    live_branch = next(b for _, b in guard.branches if not _is_empty_region(b))
    sdfg = new_loop.sdfg
    if rng.residual:
        # Keep the conditional, guarded only by what could not be folded into the range.
        guard._branches = [(CodeBlock(_unparse(_conjunction(rng.residual))), live_branch)]
        return
    existing = {n.label for n in new_loop.nodes()}
    before = {id(n) for n in new_loop.nodes()}
    move_branch_cfg_up_discard_conditions(guard, live_branch)
    for moved in [n for n in new_loop.nodes() if id(n) not in before]:
        if moved.label in existing:
            moved.label = dace.utils.find_new_name(moved.label, existing)
        existing.add(moved.label)
        _repair_moved_block(moved, sdfg)


def _rewrite(info: _LoopInfo, guard: _Guard, ranges: List[_Range]) -> None:
    loop = info.loop
    parent = loop.parent_graph
    sdfg = loop.sdfg
    try:
        was_start = parent.start_block is loop
    except ValueError:
        was_start = False

    if not ranges:
        # The body never runs: the loop degenerates to an empty state.
        replacement = parent.add_state(loop.label + '_empty', is_start_block=was_start)
        for ie in list(parent.in_edges(loop)):
            parent.add_edge(ie.src, replacement, ie.data)
        for oe in list(parent.out_edges(loop)):
            parent.add_edge(replacement, oe.dst, oe.data)
        parent.remove_node(loop)
        return

    # The residual guards are expressed in terms of the iteration start (prologue assignments substituted), so a
    # prologue assignment is only still needed if the branch body reads it.
    _drop_dead_prologue_assignments(guard)

    # The original loop object becomes the first loop of the chain (keeping its incoming edges and start-block
    # status); the remaining ranges get deep copies, taken before the original is modified.
    loops: List[LoopRegion] = [loop]
    for k in range(1, len(ranges)):
        new_loop = copy.deepcopy(loop)
        _reparent_copied_region(new_loop, sdfg)
        taken = {n.label for n in parent.nodes()} | {l.label for l in loops}
        new_loop.label = dace.utils.find_new_name(f'{loop.label}_{k}', taken)
        parent.add_node(new_loop)
        loops.append(new_loop)

    for new_loop, rng in zip(loops, ranges):
        _specialize_loop(new_loop, info, rng)

    if len(loops) > 1:
        for oe in list(parent.out_edges(loop)):
            parent.add_edge(loops[-1], oe.dst, oe.data)
            parent.remove_edge(oe)
        for pred, succ in zip(loops, loops[1:]):
            parent.add_edge(pred, succ, InterstateEdge())


# ----------------------------------------------------------------------------------------------------------------------
# The pass
# ----------------------------------------------------------------------------------------------------------------------


@make_properties
@transformation.explicit_cf_compatible
class LoopRangeReduction(ppl.Pass):
    """Turn a conditional that guards an entire loop body into a reduced iteration range.

    For a loop whose body is a single ``ConditionalBlock``, the pass computes the set of iterates on which the
    guarding condition holds and replaces the loop by one loop per contiguous run of such iterates, without the
    conditional. The condition may restrict the iteration variable

    * symbolically (``i >= 1 and i < M`` turns ``for i in range(N)`` into ``for i in range(1, min(N, M))``), or
    * through compile-time constant data in ``sdfg.constants`` (``cstarr[i] > 0`` with ``cstarr = [0,0,0,1,1,0,0,2]``
      turns ``for i in range(8)`` into ``for i in range(3, 5)`` followed by ``for i in range(7, 8)``).

    Atoms of the condition that cannot be analyzed (e.g., reads of runtime data) are kept as a residual guard inside
    the reduced loop. Symbol assignments on the path from the loop start to the conditional (such as the
    ``cstarr_index = cstarr[i]`` prologue the Python frontend emits for ``if cstarr[i] > 0``) are folded into the
    condition first, and dropped once nothing else reads them. An array that is also registered in
    ``sdfg.constants`` is analyzed through its constant value. The transformation is only applied when it is provably
    value-preserving: the loop must be a canonical for-loop with a constant stride, its iteration variable must not
    be modified inside the body nor be read after the loop, and a body containing a ``break`` is only split when a
    single range results.
    """

    CATEGORY: str = 'Optimization Preparation'

    max_ranges = Property(dtype=int,
                          default=32,
                          desc='Do not apply if the guard splits the loop into more than this many loops.')
    max_enumeration = Property(dtype=int,
                               default=1 << 20,
                               desc='Upper bound on the number of iterates evaluated when analyzing a guard that '
                               'reads compile-time constant data.')

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.CFG)

    def depends_on(self):
        return []

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, int]]:
        """
        Reduce the range of every guarded loop in ``sdfg`` (and its nested SDFGs), innermost loops first.

        :param sdfg: The SDFG to modify in place.
        :param pipeline_results: Results of prior passes in the pipeline (unused).
        :return: ``{'reduced_loops': <count>}`` with the number of loops rewritten, or ``None`` if nothing changed.
        """
        loops = [
            region for region in sdfg.all_control_flow_regions(recursive=True, parent_first=False)
            if isinstance(region, LoopRegion)
        ]
        count = 0
        for loop in loops:
            if self._reduce(loop):
                count += 1
        if count == 0:
            return None
        sdfg.reset_cfg_list()
        return {'reduced_loops': count}

    def report(self, pass_retval: Optional[Dict[str, int]]) -> str:
        if not pass_retval:
            return 'No loop ranges reduced.'
        return f'Reduced the range of {pass_retval["reduced_loops"]} loops.'

    def _reduce(self, loop: LoopRegion) -> bool:
        if loop.parent_graph is None:
            return False
        guard = _find_guard(loop)
        if guard is None:
            return False
        info = _loop_info(loop)
        if info is None:
            return False
        ranges = self._analyze(guard.condition, info)
        if ranges is None:
            return False
        if len(ranges) != 1 and _has_direct_break(loop):
            return False  # A break would also have to skip the remaining loops.
        if any(not r.residual for r in ranges):
            # Splicing the branch body into the loop needs a unique exit block.
            if len([n for n in guard.branch.nodes() if guard.branch.out_degree(n) == 0]) != 1:
                return False
        if any(_symbol_used_outside(loop, symbol) for symbol in {info.itervar} | guard.prologue_symbols):
            return False
        _rewrite(info, guard, ranges)
        return True

    def _analyze(self, condition: ast.expr, info: _LoopInfo) -> Optional[List[_Range]]:
        """The reduced ranges implied by ``condition``, in iteration order, or ``None`` if the loop must be left
        alone (nothing to gain, or ranges that cannot be proven disjoint)."""
        clauses = _dnf(condition)
        if clauses is None:
            return None
        ranges: List[_Range] = []
        changed = False
        for clause_index, clause in enumerate(clauses):
            intervals = [_UNBOUNDED]
            residual: List[ast.expr] = []
            if not clause:
                changed = True  # A literal tautology: the guard can go.
            for atom in clause:
                atom_intervals = _symbolic_atom(atom, info)
                if atom_intervals is None:
                    atom_intervals = _constant_atom(atom, info, self.max_enumeration)
                if atom_intervals is None:
                    residual.append(atom)
                    continue
                changed = True
                intervals = [
                    iv for a in intervals for b in atom_intervals for iv in (_intersect(a, b), )
                    if not _provably_empty(iv)
                ]
                if not intervals:
                    break
            if not intervals:
                changed = True  # A clause that never holds is dropped.
                continue
            for position, interval in enumerate(intervals):
                if _provably_empty(_intersect(interval, info.iteration_range)):
                    changed = True
                    continue
                ranges.append(_Range(interval, residual, clause_index, position))
            if len(ranges) > self.max_ranges:
                return None
        if not changed:
            return None

        # Ranges of different clauses may only be emitted as separate loops if they are provably disjoint (else the
        # body could run twice for one iterate); within a clause they are disjoint and ordered by construction.
        for a, b in itertools.combinations(ranges, 2):
            if a.clause != b.clause and not (_provably_before(a.interval, b.interval)
                                             or _provably_before(b.interval, a.interval)):
                return None

        def compare(a: _Range, b: _Range) -> int:
            if a.clause == b.clause:
                return a.position - b.position
            return -1 if _provably_before(a.interval, b.interval) else 1

        ranges.sort(key=cmp_to_key(compare))
        if not info.ascending:
            ranges.reverse()
        return ranges
