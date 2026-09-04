# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import ast
import itertools
from dataclasses import dataclass
from dace.sdfg.state import (
    AbstractControlFlowRegion,
    ControlFlowBlock,
    ControlFlowRegion,
    ConditionalBlock,
    LoopRegion,
)
from dace.transformation import pass_pipeline as ppl, transformation
from dace import SDFG, properties, SDFGState
from typing import Any, Dict, Set, Optional
from dace import data as dt
from dace.frontend.python import astutils
from dace.sdfg.analysis import cfg as cfg_analysis
from dace.symbolic import SymbolicType, equalize_symbols_across, pystr_to_symbolic, scalars


def free_symbol_names(value) -> Set[str]:
    """Free symbol names of an interstate-edge assignment RHS; empty for ``None``."""
    if value is None:
        return set()
    try:
        return {str(s) for s in pystr_to_symbolic(value).free_symbols}
    except Exception:
        return set()


def opaque_scalar_names(sdfg: SDFG) -> Set[str]:
    """``Scalar`` descriptors of ``sdfg`` whose readers must not propagate. A scalar written here
    changes under the reader. A non-transient scalar of a nested SDFG is a connector the parent
    rewrites per invocation, and folding its value back into control flow undoes the
    ``ScalarToSymbolPromotion`` that put the symbol there; only a never-written argument of the
    top-level SDFG is a read-only parameter."""
    opaque: Set[str] = set()
    for state in sdfg.all_states():
        for n in state.data_nodes():
            if state.in_degree(n) == 0:
                continue
            desc = sdfg.arrays.get(n.data)
            if isinstance(desc, dt.Scalar):
                opaque.add(n.data)
    if sdfg.parent_sdfg is not None:
        opaque |= {name for name, desc in sdfg.arrays.items() if isinstance(desc, dt.Scalar) and not desc.transient}
    return opaque


def meta_read_symbols(blk: ControlFlowBlock) -> Set[str]:
    """Symbols read by a region's meta code, which ``free_symbols`` hides once the body rebinds
    them -- the meta code runs first, so the read is live."""
    if not isinstance(blk, AbstractControlFlowRegion):
        return set()
    names: Set[str] = set()
    for code in blk.get_meta_codeblocks():
        if code is None:
            continue
        try:
            names |= {str(s) for s in code.get_free_symbols()}
        except Exception:
            pass
    return names


def is_array_access(value: Optional[str]) -> bool:
    """``value`` reads a data container (``tbl[i]``, ``b_slice.idx``); never propagated, since a
    descriptor shape must be a function of symbols."""
    if value is None:
        return False
    if "[" in value or "]" in value:
        return True
    return reads_struct_member(value)


def reads_struct_member(value: str) -> bool:
    """``value`` reads an attribute off a plain name; a callee (``math.floor``) does not count."""
    try:
        tree = ast.parse(value.strip(), mode='eval')
    except (SyntaxError, ValueError):
        return False
    callees = {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    return any(isinstance(node, ast.Attribute) and id(node) not in callees for node in ast.walk(tree))


def resolve_value(value, table: Dict[str, Any]):
    """Substitute known symbol values from ``table`` into an assignment RHS, against the pre-edge
    table since one edge's assignments are simultaneous. Textual over the AST like
    ``InterstateEdge.replace_dict``: a sympy round trip renames every call it models, so
    ``abs(z)`` comes back as ``Abs(z)`` and codegen emits a name C++ does not have."""
    if value is None:
        return None
    if is_array_access(value):
        return value
    try:
        tree = ast.parse(value.strip(), mode='eval')
    except (SyntaxError, ValueError):
        return value
    repl = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name) or node.id in repl:
            continue
        known = table.get(node.id)
        if known is not None and not is_array_access(known):
            repl[node.id] = known
    if not repl:
        return value
    try:
        return astutils.unparse(astutils.ASTFindReplace(repl).visit(tree))
    except Exception:
        return value


def assign_targets(code) -> Set[str]:
    """Names bound by the statements of a ``CodeBlock``."""
    names: Set[str] = set()
    if code is None:
        return names
    stmts = code.code if isinstance(code.code, list) else []
    for stmt in stmts:
        if not isinstance(stmt, ast.AST):
            continue
        for node in ast.walk(stmt):
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            else:
                continue
            names |= {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}
    return names


def loop_bound_symbols(loop: LoopRegion) -> Set[str]:
    """Symbols a loop rebinds per iteration: its interstate-edge assignments plus what the meta
    code of it and of any loop nested in it binds. ``replace_meta_accesses`` rewrites the init and
    update statements whole, LHS included, so a value known for an iteration variable would spell
    ``(- 1) = ((- 1) + 1)``."""
    bound = {s for e in loop.all_interstate_edges() for s in e.data.assignments.keys()}
    for blk in itertools.chain((loop, ), loop.all_control_flow_blocks()):
        if not isinstance(blk, LoopRegion):
            continue
        if blk.loop_variable:
            bound.add(blk.loop_variable)
        bound |= assign_targets(blk.init_statement)
        bound |= assign_targets(blk.update_statement)
    return bound


def reads_data(value, owner: SDFG) -> bool:
    """``value`` names a data container of ``owner``."""
    return bool(free_symbol_names(value) & set(owner.arrays))


def consistent_bindings(sd: SDFG) -> Dict[str, Optional[str]]:
    """Every symbol ``sd``'s interstate edges bind, mapped to the RHS all its binding edges agree
    on -- or ``None`` where they disagree or the binding is self-referential (``i = i + 1``).

    Collection only; it says what a symbol IS, not whether substituting it is safe. The
    elimination round below adds its own scope guard on top, and
    :func:`resolve_bindings` reads the same table without mutating anything.
    """
    bindings: Dict[str, Optional[str]] = {}
    for e in sd.all_interstate_edges():
        for lhs, rhs in e.data.assignments.items():
            if rhs is None or lhs in free_symbol_names(rhs):
                bindings[lhs] = None
                continue
            if lhs not in bindings:
                bindings[lhs] = rhs
            elif bindings[lhs] is not None and bindings[lhs] != rhs:
                bindings[lhs] = None
    return bindings


def resolve_bindings(expr: SymbolicType, sd: SDFG, rounds: int = 8, expand_data_reads: bool = False) -> SymbolicType:
    """``expr`` with every consistently-bound interstate symbol expanded into its RHS, to a fixed
    point (bounded by ``rounds``).

    A QUERY, not a rewrite: it returns a new expression and leaves the SDFG bit-identical.
    :class:`SymbolPropagation` deliberately refuses to substitute a binding whose RHS names a loop
    variable, because ``replace_dict`` would then size a descriptor by it -- so a promoted index
    such as ``__sym_i_times_inc = i * inc`` stays opaque in the graph, and a structural matcher
    asking ``coeff(i)`` about it reads 0, i.e. "loop-invariant". This recovers the relation for the
    matcher without touching the descriptors.

    A binding that is a BARE data read (``bsym = b_scal``, ``bsym = b[i]``) is left alone by
    default: its value is a runtime datum, so for a structural matcher expanding it only renames
    the symbol to a container and answers nothing. A container reached inside a larger expression
    (``i * inc``, ``inc`` a scalar argument) is kept -- that is the spelling the index carried
    before promotion, and the relation to the loop variable is the whole point of asking.

    ``expand_data_reads`` expands those too, for the one caller that gains from it: a solver that
    models the container itself. The frontend materializes the SAME read under a fresh name at
    every use (a branch condition and the subscript it guards each get their own ``idx[i]``
    symbol), so leaving them opaque hands the solver two unrelated variables where the program has
    one value. Expanding restores the identity, and re-indexes it by the loop variable so distinct
    iterations no longer share one opaque symbol.

    :param expr: A symbolic expression (a memlet-subset bound, typically).
    :param sd: The SDFG whose interstate edges carry the bindings.
    :param rounds: Substitution rounds before giving up on a chain.
    :param expand_data_reads: Also expand bindings whose RHS reads a data container.
    :returns: The expanded expression; unresolved symbols simply stay put.
    """
    bindings = consistent_bindings(sd)
    if not bindings:
        return expr
    resolved = expr
    for _ in range(rounds):
        repl = {}
        for sym in resolved.free_symbols:
            rhs = bindings.get(str(sym))
            if rhs is None:
                continue
            if not expand_data_reads and (is_array_access(rhs) or rhs.strip() in sd.arrays):
                continue
            try:
                repl[sym] = pystr_to_symbolic(rhs)
            except Exception:  # noqa: BLE001 - an unparseable RHS just stays opaque
                continue
        if not repl:
            break
        resolved = resolved.xreplace(repl)
    # Each RHS is parsed on its own, so one name can come back as several instances (a plain one
    # beside a stamped one). Merge them: ``coeff`` / ``in free_symbols`` on the result go through
    # identity, and duplicates make both answer wrong without raising.
    return equalize_symbols_across(resolved)[0]


@dataclass(unsafe_hash=True)
@properties.make_properties
@transformation.explicit_cf_compatible
class SymbolPropagation(ppl.Pass):
    """
    Propagates symbols that were assigned to one value forward through the SDFG, reducing the number of overall symbols.
    """

    CATEGORY: str = "Simplification"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Symbols | ppl.Modifies.Edges | ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified != ppl.Modifies.Nothing

    def apply_pass(self, sdfg: SDFG, _) -> Optional[Set[str]]:
        # Assumption: Symbols can only change in InterStateEdges

        before_free: Set[str] = {str(s) for s in sdfg.free_symbols}

        all_cfg_blks = dict()
        for node, parent in sdfg.all_nodes_recursive():
            if isinstance(node, ControlFlowBlock):
                all_cfg_blks[node] = parent

        # An unwritten Scalar of the top-level SDFG is read-only and propagates like a symbol.
        self._opaque_scalars: Dict[SDFG, Set[str]] = {}
        for sd in sdfg.all_sdfgs_recursive():
            self._opaque_scalars[sd] = opaque_scalar_names(sd)

        in_syms = {cfg_blk: {} for cfg_blk in all_cfg_blks.keys()}
        out_syms = {cfg_blk: {} for cfg_blk in all_cfg_blks.keys()}

        # Perform a forward fixed-point iteration to propagate symbols, in execution order: a value
        # then travels the length of a chain per sweep instead of one block per sweep, which is what
        # made the sweep count scale with the depth of the CFG. ConstantPropagation orders its own
        # fixed point the same way.
        ordered_blks = self._execution_order(sdfg, all_cfg_blks)
        readers = self._readers(ordered_blks)

        # Worklist over that order: a block is revisited only when something it reads moved, so the
        # sweeps after the first cost the blocks that actually changed rather than all of them.
        dirty = {id(blk) for blk, _ in ordered_blks}
        while dirty:
            for cfg_blk, parent in ordered_blks:
                if id(cfg_blk) not in dirty:
                    continue
                dirty.discard(id(cfg_blk))
                moved = False

                new_in_syms = self._get_in_syms(sdfg, cfg_blk, parent, in_syms, out_syms)
                if new_in_syms != in_syms[cfg_blk]:
                    moved = True
                    in_syms[cfg_blk] = new_in_syms

                new_out_syms = self._get_out_syms(cfg_blk, parent, in_syms, out_syms)
                if new_out_syms != out_syms[cfg_blk]:
                    moved = True
                    out_syms[cfg_blk] = new_out_syms

                if moved:
                    dirty |= readers[id(cfg_blk)]

        # An honest return set is what lets a FixedPointPipeline converge.
        propagated: Set[str] = set()
        for cfg_blk, parent in all_cfg_blks.items():
            propagated |= self._update_syms(cfg_blk, parent, in_syms, out_syms)
        # Substitution leaves the defining assignment behind; sweep to a fixed point.
        eliminated = self._eliminate_dead_iedge_assignments(sdfg)
        if eliminated:
            propagated |= eliminated

        # A new free symbol means a value rendered into a name that does not resolve.
        new_free: Set[str] = {str(s) for s in sdfg.free_symbols} - before_free
        if new_free:
            raise ValueError(f"SymbolPropagation introduced free symbol(s) {sorted(new_free)}: a propagated "
                             f"value rendered to an unresolvable name. Symbol propagation must only eliminate "
                             f"symbols, never introduce them.")

        return propagated if propagated else None

    def _eliminate_dead_iedge_assignments(self, sdfg: SDFG) -> Set[str]:
        """Drop interstate-edge assignments whose LHS is no longer referenced anywhere."""
        removed: Set[str] = set()
        while True:
            this_round = self._eliminate_round(sdfg)
            if not this_round:
                break
            removed |= this_round
        return removed

    def _eliminate_round(self, sdfg: SDFG) -> Set[str]:
        """One sweep across ``sdfg`` and its nested SDFGs; descriptors go first, since
        ``free_symbols`` pulls shape symbols through the access nodes."""
        eliminated: Set[str] = set()
        for sd in sdfg.all_sdfgs_recursive():
            # Propagatable: every binding edge agrees, and the RHS is not self-referential.
            bindings = consistent_bindings(sd)
            # ``replace_dict`` also rewrites descriptor shapes, which live at SDFG scope, so
            # ``K = i + 1`` would size a transient by a loop variable and allocate it outside.
            invariant = {str(s) for s in sd.free_symbols} | set(sd.constants_prop.keys())
            safe_subs = {
                sym: rhs
                for sym, rhs in bindings.items()
                if rhs is not None and not is_array_access(rhs) and free_symbol_names(rhs) <= invariant
            }

            if safe_subs:
                sd.replace_dict(safe_subs, replace_keys=False, replace_in_graph=False)

            used_in_ir: Set[str] = set()
            for blk in sd.all_control_flow_blocks():
                used_in_ir |= {str(s) for s in blk.free_symbols}
                used_in_ir |= meta_read_symbols(blk)
            for e in sd.all_interstate_edges():
                for rhs in e.data.assignments.values():
                    used_in_ir |= free_symbol_names(rhs)
                if e.data.condition is not None:
                    try:
                        used_in_ir |= {str(s) for s in e.data.condition.get_free_symbols()}
                    except Exception:
                        pass

            sd_eliminated: Set[str] = set()
            for e in sd.all_interstate_edges():
                for lhs in list(e.data.assignments.keys()):
                    if lhs not in used_in_ir:
                        del e.data.assignments[lhs]
                        sd_eliminated.add(lhs)
            # Drop orphaned declarations, else nested-SDFG validation demands the symbol.
            if sd_eliminated:
                still_bound = {k for ie in sd.all_interstate_edges() for k in ie.data.assignments.keys()}
                for name in sd_eliminated:
                    if (name in sd.symbols and name not in still_bound and name not in used_in_ir):
                        del sd.symbols[name]
            eliminated |= sd_eliminated
        return eliminated

    # Given a cfg_blk, builds the incoming set of symbols
    def _execution_order(self, sdfg: SDFG, all_cfg_blks: Dict[ControlFlowBlock, ControlFlowRegion]):
        """``(block, parent)`` pairs in execution order, one entry per block of ``all_cfg_blks``.

        The sort runs per SDFG because it does not cross nested SDFGs, and it only reaches blocks
        with a path from the start block -- an unreachable one still carries symbols, so it is
        appended rather than dropped."""
        ordered = []
        seen = set()
        for sd in sdfg.all_sdfgs_recursive():
            try:
                blocks = list(cfg_analysis.blockorder_topological_sort(sd, recursive=True))
            except KeyError:
                # The sort walks a dominator tree rooted at the start block, so a CFG with a second
                # source -- which this pass supports -- has blocks the tree never names. Fall back
                # to insertion order for that SDFG rather than ordering part of it.
                continue
            for blk in blocks:
                if id(blk) not in seen and blk in all_cfg_blks:
                    seen.add(id(blk))
                    ordered.append((blk, all_cfg_blks[blk]))
        for blk, parent in all_cfg_blks.items():
            if id(blk) not in seen:
                seen.add(id(blk))
                ordered.append((blk, parent))
        return ordered

    def _readers(self, ordered_blks) -> Dict[int, Set[int]]:
        """For each block, the blocks whose own tables read it -- who to revisit when it moves.

        Deliberately a superset: missing an edge here would stop the iteration early, at a table
        that is not yet a fixed point, so every reader named by :meth:`_get_in_syms` and
        :meth:`_get_out_syms` is listed, and anything uncertain is listed too.
        """
        readers: Dict[int, Set[int]] = {id(blk): set() for blk, _ in ordered_blks}
        present = set(readers)

        def add(src: ControlFlowBlock, dst) -> None:
            if dst is not None and id(dst) in present:
                readers[id(src)].add(id(dst))

        for blk, parent in ordered_blks:
            # A successor's in_syms reads this block's out_syms. A ConditionalBlock holds its
            # branches as sub-regions rather than graph nodes, so it has no edges to walk.
            if not isinstance(parent, ConditionalBlock):
                for edge in parent.out_edges(blk):
                    add(blk, edge.dst)
            # The parent's out_syms combines its sinks, or its branches when it is conditional.
            add(blk, parent)
            # A region hands its in_syms to whatever runs first inside it.
            if isinstance(blk, ConditionalBlock):
                for branch in blk.sub_regions():
                    add(blk, branch)
            elif isinstance(blk, AbstractControlFlowRegion):
                add(blk, blk.start_block)

        return readers

    def _get_in_syms(
        self,
        sdfg: SDFG,
        cfg_blk: ControlFlowBlock,
        parent: ControlFlowRegion,
        in_syms: Dict[ControlFlowBlock, Dict[str, Any]],
        out_syms: Dict[ControlFlowBlock, Dict[str, Any]],
    ) -> Dict[str, Any]:
        # The filters below need the SDFG that OWNS this block: ``sdfg`` is the top-level one.
        owner = cfg_blk.sdfg
        # Combine the outgoing symbols of all incoming edges with their assignments to the cfg_blk
        new_in_syms = {}
        for i, edge in enumerate(parent.in_edges(cfg_blk)):
            sym_table = dict(out_syms[edge.src])
            # One edge's assignments fire simultaneously: a value naming a symbol this edge
            # rebinds stays live, else the rebinding is counted twice.
            edge_keys = set(edge.data.assignments.keys())
            resolved = {}
            for k, v in edge.data.assignments.items():
                rv = resolve_value(v, sym_table)
                if rv is not None and (free_symbol_names(rv) & edge_keys):
                    rv = None
                resolved[k] = rv
            # A carried-in value naming a symbol this edge reassigns is stale downstream.
            for sym in list(sym_table.keys()):
                val = sym_table[sym]
                if sym not in edge_keys and val is not None and (free_symbol_names(val) & edge_keys):
                    sym_table[sym] = None
            sym_table.update(resolved)

            # Nested array accesses are not supported.
            sym_table = {k: v for k, v in sym_table.items() if not is_array_access(v)}

            # Views are seen as pointers.
            sym_table = {
                k: v
                for k, v in sym_table.items() if v is None or not any([
                    str(s) in owner.arrays and isinstance(owner.arrays[str(s)], dt.View)
                    for s in pystr_to_symbolic(v).free_symbols
                ])
            }

            opaque = self._opaque_scalars.get(owner, set())
            sym_table = {k: v for k, v in sym_table.items() if v is None or not (scalars(v, owner.arrays) & opaque)}

            if i == 0:
                new_in_syms = sym_table
            else:
                self._combine_syms(new_in_syms, sym_table)

        # A nested start block inherits its parent's symbols; a nested SDFG has a symbol mapping.
        if (parent.start_block == cfg_blk and not isinstance(parent, SDFG)) or (isinstance(parent, ConditionalBlock)
                                                                                and cfg_blk in parent.sub_regions()):
            # Some shapes carry their own, so combine rather than assert.
            if new_in_syms:
                self._combine_syms(new_in_syms, in_syms[parent])
            else:
                new_in_syms = in_syms[parent]

            if isinstance(parent, LoopRegion):
                new_in_syms = dict(new_in_syms)
                for sym in loop_bound_symbols(parent):
                    if sym in new_in_syms:
                        new_in_syms[sym] = None

        return new_in_syms

    # Given a cfg_blk, builds the outgoing set of symbols
    def _get_out_syms(
        self,
        cfg_blk: ControlFlowBlock,
        parent: ControlFlowRegion,
        in_syms: Dict[ControlFlowBlock, Dict[str, Any]],
        out_syms: Dict[ControlFlowBlock, Dict[str, Any]],
    ) -> Dict[str, Any]:
        if isinstance(cfg_blk, LoopRegion):
            new_out_syms = dict(in_syms[cfg_blk])
            for sym in loop_bound_symbols(cfg_blk):
                if sym in new_out_syms:
                    new_out_syms[sym] = None
            return new_out_syms

        elif isinstance(cfg_blk, ConditionalBlock):
            new_out_syms = dict(out_syms[cfg_blk.sub_regions()[0]])
            for b in cfg_blk.sub_regions():
                self._combine_syms(new_out_syms, out_syms[b])

            # Without an else branch, the incoming table is the implicit else.
            has_non_conds = any([c is None for c, _ in cfg_blk.branches])
            if not has_non_conds:
                self._combine_syms(new_out_syms, in_syms[cfg_blk])

            return new_out_syms

        elif isinstance(cfg_blk, SDFGState):
            # Cannot change symbols in SDFGStates
            return in_syms[cfg_blk]

        else:
            # Use sink symbols as outgoing symbols
            sink_nodes = [n for n in cfg_blk.nodes() if cfg_blk.out_degree(n) == 0 and isinstance(n, ControlFlowBlock)]
            if len(sink_nodes) == 0:
                return in_syms[cfg_blk]

            new_out_syms = dict(out_syms[sink_nodes[0]])
            for n in sink_nodes:
                self._combine_syms(new_out_syms, out_syms[n])
            return new_out_syms

    def _block_free_symbols(self, cfg_blk: ControlFlowBlock, parent: ControlFlowRegion) -> Set[str]:
        """Names of symbols read by ``cfg_blk`` and by its outgoing edges."""
        free = {str(s) for s in cfg_blk.free_symbols}
        free |= {str(s) for edge in parent.out_edges(cfg_blk) for s in edge.data.free_symbols}
        return free

    # Given a cfg_blk, updates the symbols in the cfg_blk
    def _update_syms(
        self,
        cfg_blk: ControlFlowBlock,
        parent: ControlFlowRegion,
        in_syms: Dict[ControlFlowBlock, Dict[str, Any]],
        out_syms: Dict[ControlFlowBlock, Dict[str, Any]],
    ) -> Set[str]:
        new_in_syms = dict(in_syms[cfg_blk])
        new_out_syms = dict(out_syms[cfg_blk])

        new_in_syms = {sym: val for sym, val in new_in_syms.items() if val is not None}
        new_out_syms = {sym: val for sym, val in new_out_syms.items() if val is not None}

        candidates = set(new_in_syms) | set(new_out_syms)
        if not candidates:
            return set()
        free_before = self._block_free_symbols(cfg_blk, parent)

        # An acyclic chain converges within ``#symbols`` rounds; the cap stops a cyclic one.
        max_iters = len(new_in_syms) + len(new_out_syms) + 2

        # Folding a loop-carried value into ``while udiff > 0.001`` spins forever.
        loop_carried: Set[str] = set()
        if isinstance(cfg_blk, LoopRegion):
            loop_carried = loop_bound_symbols(cfg_blk)

        # A state reaches data through connectors only, so a container name written into tasklet
        # code or a subset stops being a read: the connector is pruned, ``used_symbols`` reports a
        # free symbol and ``arglist`` raises ``KeyError``, or inlining renames the container away.
        # Interstate edges and region meta code carry read memlets, so they stay allowed.
        state_subs = new_in_syms
        if isinstance(cfg_blk, SDFGState):
            state_subs = {s: v for s, v in new_in_syms.items() if not reads_data(v, cfg_blk.sdfg)}

        changed = True
        iters = 0
        while changed and iters < max_iters:
            iters += 1
            changed = False
            free_sym = {str(s) for s in cfg_blk.free_symbols}
            free_edge_sym = {str(s) for edge in parent.out_edges(cfg_blk) for s in edge.data.free_symbols}

            # Only what the block still reads: ``replace_in_codeblock`` shadows a name in a C++
            # tasklet by prepending ``auto i = ...;``, and a second round prepends it again.
            if isinstance(cfg_blk, LoopRegion):
                meta_read = meta_read_symbols(cfg_blk)
                cfg_blk.replace_meta_accesses({
                    s: v
                    for s, v in new_in_syms.items() if s in meta_read and s not in loop_carried
                })
            elif isinstance(cfg_blk, ConditionalBlock):
                meta_read = meta_read_symbols(cfg_blk)
                cfg_blk.replace_meta_accesses({s: v for s, v in new_in_syms.items() if s in meta_read})
            elif isinstance(cfg_blk, SDFGState):
                cfg_blk.replace_dict({s: v for s, v in state_subs.items() if s in free_sym})
            else:
                pass  # Nested CFGs inherit their parent's symbols

            # Same rule out: a substitution naming a key of this edge reads its own output.
            for edge in parent.out_edges(cfg_blk):
                edge_free = {str(s) for s in edge.data.free_symbols}
                edge_keys = set(edge.data.assignments.keys())
                edge_subs = {
                    s: v
                    for s, v in new_out_syms.items() if s in edge_free and not (free_symbol_names(v) & edge_keys)
                }
                edge.data.replace_dict(edge_subs, replace_keys=False)

            new_free_edge_sym = {str(s) for edge in parent.out_edges(cfg_blk) for s in edge.data.free_symbols}
            if free_sym != {str(s) for s in cfg_blk.free_symbols} or free_edge_sym != new_free_edge_sym:
                changed = True

        # The candidate symbols that are no longer read here were propagated.
        return candidates & (free_before - self._block_free_symbols(cfg_blk, parent))

    def _combine_syms(self, sym1: Dict[str, Any], sym2: Dict[str, Any]) -> None:
        """Meet of two symbol tables at a join, in place: a symbol survives only when both sides
        agree, since a key absent on one side means no value is known there."""
        for sym, val in sym2.items():
            if sym not in sym1 or sym1[sym] != val:
                sym1[sym] = None
        for sym in sym1:
            if sym not in sym2:
                sym1[sym] = None
