# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Materialize loop-defined IV-symbol exit values with a unique name.

A loop whose body increments / multiplies an SDFG symbol per iteration
(``k = k + step`` on an interstate edge) and whose post-loop code reads that
symbol's *final* value blocks ``LoopToMap``: parallelisation has no canonical
"last iteration", so the symbol's exit value is undefined under a map. The IV
update itself is otherwise well-behaved -- the body is affine in ``k``, the
trip count is known, and the closed form ``k_init + step * N`` is available --
but it is the *post-loop reader* that pins the loop sequential.

This pass detects the pattern, computes the closed-form exit value, allocates
a *fresh unique* symbol ``_<sym>_post_<N>`` whose value is the closed form, and
rewrites every reader reachable from the loop's exit edges to read the new
symbol. The original ``sym`` keeps its pre-loop value untouched, the in-loop
update is unaffected, and ``LoopToMap`` is no longer blocked by the
"loop-defined symbol used after the loop" check.

Inspired by the post-value epilogue path in
:class:`~dace.transformation.passes.unique_loop_iterators.UniqueLoopIterators`,
which materialises the loop iterator's exit value (under its original name) for
Fortran-frontend compatibility. The key difference: that pass *renames in place*
because the iterator is owned by the loop; this pass materialises under a
*fresh* name because the underlying symbol may carry a value the pre-loop code
also needs (so renaming everywhere would clobber the seed).

Scope:

* Loops with a single ``LoopRegion`` body, ``loop_variable`` set, and a known
  ``init`` / ``end`` / ``stride`` for the trip-count formula.
* Interstate-edge assignments of the affine form ``sym = sym + c`` /
  ``sym = sym * c`` on the loop's body edges, where ``c`` is a numeric literal or
  a loop-invariant SDFG symbol.
* The symbol must have a known pre-loop seed (it appears in ``sdfg.symbols`` /
  ``sdfg.free_symbols`` / ``sdfg.constants``).

Out of scope:

* Symbols whose update isn't ``sym = sym OP c`` (compound updates, conditional
  updates, data-array reads in the increment expression).
* Loops where the symbol is *also* read on the SAME interstate edge that
  writes it (true loop-carried dependence -- closed form is still valid but the
  in-loop reader must see the per-iteration value, which a single materialisation
  does not produce).
"""
import ast
from typing import Dict, List, Optional, Set, Tuple

import dace
from dace import SDFG, properties, symbolic
from dace.sdfg.state import ControlFlowBlock, ControlFlowRegion, LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis

#: Prefix for the materialised post-loop symbol; self-identifying in dumps and
#: collision-free against frontend or user-chosen names.
_POST_PREFIX = "_loop_exit_"


def _parse_affine_update(rhs_str: str, lhs: str) -> Optional[Tuple[type, str]]:
    """Parse ``rhs_str`` as ``lhs OP c`` (or ``c OP lhs``); return ``(ast op class, c_str)``.

    Accepts ``Add`` and ``Mult`` operators. ``c`` may be any expression that
    does NOT reference ``lhs`` itself (which would make the update non-affine).
    """
    try:
        expr = ast.parse(str(rhs_str), mode='eval').body
    except (SyntaxError, ValueError, TypeError):
        return None
    if not isinstance(expr, ast.BinOp) or type(expr.op) not in (ast.Add, ast.Mult):
        return None

    def _is_lhs(node):
        return isinstance(node, ast.Name) and node.id == lhs

    def _refs_lhs(node):
        return any(_is_lhs(n) for n in ast.walk(node))

    if _is_lhs(expr.left) and not _refs_lhs(expr.right):
        c_node = expr.right
    elif _is_lhs(expr.right) and not _refs_lhs(expr.left):
        c_node = expr.left
    else:
        return None
    return type(expr.op), ast.unparse(c_node)


def _is_loop_invariant_symbol(name: str, loop: LoopRegion, sdfg: SDFG, sdfg_free_symbols: Set[str]) -> bool:
    """Symbol exists in the SDFG and isn't reassigned inside the loop's body.

    ``sdfg_free_symbols`` is ``sdfg.free_symbols`` precomputed by the caller: that
    property walks the whole SDFG on every access, so it is passed in rather than
    recomputed per candidate symbol.
    """
    if name == loop.loop_variable:
        return False
    if name not in sdfg.symbols and name not in sdfg.constants and name not in sdfg_free_symbols:
        return False
    for e in loop.edges():
        if e.data.assignments and name in e.data.assignments:
            return False
    return True


def _expr_is_loop_invariant(expr_str: str, loop: LoopRegion, sdfg: SDFG, ignore: Set[str],
                            sdfg_free_symbols: Set[str]) -> bool:
    """Every ``ast.Name`` in ``expr_str`` is loop-invariant in ``loop`` (or in ``ignore``)."""
    try:
        expr = ast.parse(expr_str, mode='eval').body
    except (SyntaxError, ValueError, TypeError):
        return False
    for n in ast.walk(expr):
        if isinstance(n, ast.Name):
            if n.id in ignore:
                continue
            if n.id == loop.loop_variable:
                return False
            if not _is_loop_invariant_symbol(n.id, loop, sdfg, sdfg_free_symbols):
                # Allow numeric literal names like ``True`` / ``False`` to pass.
                if not hasattr(__import__('builtins'), n.id):
                    return False
    return True


def _detect_iv_symbols(loop: LoopRegion, sdfg: SDFG, sdfg_free_symbols: Set[str]) -> Dict[str, Tuple[type, str]]:
    """Find symbols updated by ``sym = sym OP c`` on the loop body's interstate
    edges, where the update appears at most once per traversal and ``c`` is
    loop-invariant.

    :returns: ``{sym_name: (ast_op_class, c_expr_string)}``.
    """
    found: Dict[str, Tuple[type, str]] = {}
    seen_lhs: Set[str] = set()
    for e in loop.edges():
        if not e.data.assignments:
            continue
        for lhs, rhs in e.data.assignments.items():
            if lhs in seen_lhs:
                # Multiple update sites for the same symbol -- not a clean affine IV.
                found.pop(lhs, None)
                continue
            seen_lhs.add(lhs)
            parsed = _parse_affine_update(str(rhs), lhs)
            if parsed is None:
                continue
            op_type, c_expr = parsed
            # The seed value (pre-loop) must already be in scope.
            if lhs not in sdfg.symbols and lhs not in sdfg_free_symbols and lhs not in sdfg.constants:
                continue
            if not _expr_is_loop_invariant(c_expr, loop, sdfg, set(), sdfg_free_symbols):
                continue
            found[lhs] = (op_type, c_expr)
    return found


def _trip_count(loop: LoopRegion) -> Optional[str]:
    """Compute ``(end - init) // stride + 1`` as a sympy expression string."""
    start = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if start is None or end is None or stride is None:
        return None
    try:
        n = symbolic.simplify((end - start) // stride + 1)
    except Exception:
        return None
    return symbolic.symstr(n)


def _closed_form(op_type: type, init: str, c: str, n: str) -> Optional[str]:
    """Closed-form value of an affine recurrence after ``n`` steps."""
    if op_type is ast.Add:
        return f"(({init}) + ({c}) * ({n}))"
    if op_type is ast.Mult:
        return f"(({init}) * (({c}) ** ({n})))"
    return None


def _next_post_id(sdfg: SDFG, sdfg_free_symbols: Set[str]) -> int:
    """Lowest ``<N>`` not in use among existing ``_loop_exit_*_<N>`` symbols."""
    used: Set[int] = set()
    for s in list(sdfg.symbols.keys()) + list(sdfg_free_symbols):
        if s.startswith(_POST_PREFIX):
            tail = s.rsplit('_', 1)[-1]
            if tail.isdigit():
                used.add(int(tail))
    n = 0
    while n in used:
        n += 1
    return n


def _post_loop_blocks(parent: ControlFlowRegion, loop: LoopRegion) -> Set[ControlFlowBlock]:
    """BFS from each out-edge destination of ``loop`` collecting every block in
    the post-loop region (within ``parent``)."""
    visited: Set[ControlFlowBlock] = set()
    frontier: List[ControlFlowBlock] = [e.dst for e in parent.out_edges(loop)]
    while frontier:
        b = frontier.pop()
        if b in visited or b is loop:
            continue
        visited.add(b)
        for e in parent.out_edges(b):
            if e.dst not in visited:
                frontier.append(e.dst)
    return visited


def _rewrite_post_loop_readers(parent: ControlFlowRegion, post_blocks: Set[ControlFlowBlock], old_name: str,
                               new_name: str, sdfg: SDFG):
    """Replace every reference to ``old_name`` with ``new_name`` in the
    post-loop blocks: interstate edges into / between them (when both endpoints
    are post-loop), and the blocks' own contents.
    """
    repl = {old_name: new_name}
    for block in post_blocks:
        block.replace_dict(repl)
    # Interstate edges OUT of post-loop blocks (and between them) also need
    # their assignments/conditions rewritten -- ``replace_dict`` on the block
    # alone won't touch the edge data.
    for block in post_blocks:
        for e in parent.out_edges(block):
            if e.dst in post_blocks:
                if e.data.condition is not None:
                    e.data.condition.code = [
                        ast.parse(ast.unparse(_RenameNames(repl).visit(c)).strip(), mode='exec').body[0]
                        for c in (e.data.condition.code or [])
                    ]
                if e.data.assignments:
                    e.data.assignments = {k: _replace_in_expr(v, repl) for k, v in e.data.assignments.items()}


class _RenameNames(ast.NodeTransformer):
    """Substitute identifiers in an ast tree."""

    def __init__(self, mapping: Dict[str, str]):
        self.mapping = mapping

    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.copy_location(ast.Name(id=self.mapping.get(node.id, node.id), ctx=node.ctx), node)


def _replace_in_expr(s: str, mapping: Dict[str, str]) -> str:
    try:
        tree = ast.parse(str(s), mode='eval').body
    except (SyntaxError, ValueError, TypeError):
        return s
    return ast.unparse(_RenameNames(mapping).visit(tree))


@properties.make_properties
@xf.explicit_cf_compatible
class MaterializeLoopExitSymbols(ppl.Pass):
    """Materialise per-loop exit values of affine-IV symbols under fresh names.

    Run BEFORE ``LoopToMap`` so the "loop-defined symbol used after the loop"
    refusal disappears for any IV-shaped symbol whose only post-loop role is
    the closed-form exit value.
    """

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Symbols

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _) -> Optional[int]:
        """Materialise every eligible IV symbol's exit value. Returns the count
        or ``None`` if nothing matched."""
        materialised = 0
        for sd in sdfg.all_sdfgs_recursive():
            # ``sd.free_symbols`` walks the entire SDFG on every access. It is
            # invariant across the per-loop scan below -- only our own
            # materialisations mutate ``sd`` -- so compute it once here and refresh
            # it only after a loop is actually materialised (which is rare).
            sd_free_symbols = sd.free_symbols
            for cfg in list(sd.all_control_flow_regions()):
                if not (isinstance(cfg, LoopRegion) and cfg.loop_variable):
                    continue
                count = self._try_materialise(cfg, sd, sd_free_symbols)
                if count:
                    materialised += count
                    sd_free_symbols = sd.free_symbols
        return materialised or None

    def _try_materialise(self, loop: LoopRegion, sdfg: SDFG, sdfg_free_symbols: Set[str]) -> int:
        parent = loop.parent_graph
        if parent is None:
            return 0

        # IV symbols updated by body interstate-edge assignments (``k = k + step``).
        iv_symbols = dict(_detect_iv_symbols(loop, sdfg, sdfg_free_symbols))

        # The loop iterator itself: ``_loop_it_<N> = _loop_it_<N> + stride`` is the
        # same affine recurrence, just emitted by ``loop.update_statement`` rather
        # than a body interstate edge. Treat it identically.
        loop_var = loop.loop_variable
        if (loop_var and (loop_var in sdfg.symbols or loop_var in sdfg_free_symbols) and loop_var not in iv_symbols):
            stride = loop_analysis.get_loop_stride(loop)
            if stride is not None:
                iv_symbols[loop_var] = (ast.Add, str(symbolic.symstr(stride)))

        if not iv_symbols:
            return 0
        trip = _trip_count(loop)
        if trip is None:
            return 0

        post_blocks = _post_loop_blocks(parent, loop)
        if not post_blocks:
            return 0

        count = 0
        next_id = _next_post_id(sdfg, sdfg_free_symbols)
        for sym_name, (op_type, c_expr) in iv_symbols.items():
            # Check this symbol is actually READ in the post-loop region.
            if not self._is_read_in(post_blocks, sym_name, parent):
                continue
            # The closed form's seed is the symbol's *pre-loop* value, which is
            # always the symbol's own name here (``k`` for body-symbol pattern,
            # the iterator's init expression is folded into the trip count for
            # the iterator pattern -- but both express it as ``sym + step * N``
            # when ``sym`` carries the seed, which it does for body symbols and
            # for the iterator immediately before each iteration starts).
            seed = sym_name
            if sym_name == loop.loop_variable:
                init = loop_analysis.get_init_assignment(loop)
                if init is not None:
                    seed = str(symbolic.symstr(init))
            closed = _closed_form(op_type, seed, c_expr, trip)
            if closed is None:
                continue
            new_name = f"{_POST_PREFIX}{sym_name}_{next_id}"
            next_id += 1
            sdfg.add_symbol(new_name, sdfg.symbols.get(sym_name, dace.int64))
            # Splice a post-loop state right after ``loop`` that assigns
            # ``new_name = closed_form`` on its in-edge. Existing out-edges
            # from ``loop`` cascade through the new state unchanged.
            anchor = parent.add_state_after(loop, f"_loop_exit_{sym_name}_{next_id - 1}")
            for e in parent.in_edges(anchor):
                if e.src is loop:
                    e.data.assignments[new_name] = closed
                    break
            _rewrite_post_loop_readers(parent, post_blocks - {anchor}, sym_name, new_name, sdfg)
            count += 1
        return count

    def _is_read_in(self, blocks: Set[ControlFlowBlock], sym_name: str, parent: ControlFlowRegion) -> bool:
        """Whether any block in ``blocks`` (or their downstream interstate
        edges that stay within ``blocks``) references ``sym_name`` in its data
        flow / assignments / conditions."""
        for block in blocks:
            for s in (block.all_states() if hasattr(block, 'all_states') else [block]):
                if not hasattr(s, 'used_symbols'):
                    continue
                if sym_name in s.used_symbols(all_symbols=True):
                    return True
            for e in parent.out_edges(block):
                if e.dst not in blocks:
                    continue
                if e.data.assignments and any(sym_name in str(v) for v in e.data.assignments.values()):
                    return True
                if e.data.condition is not None and sym_name in ' '.join(
                        ast.unparse(c) if isinstance(c, ast.AST) else str(c) for c in (e.data.condition.code or [])):
                    return True
        return False


__all__ = ['MaterializeLoopExitSymbols']
