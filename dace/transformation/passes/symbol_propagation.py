# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import ast
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
from dace.symbolic import pystr_to_symbolic, scalars, symstr


def _free_symbols(value) -> Set[str]:
    """Free symbol names of an interstate-edge assignment RHS; empty for ``None``."""
    if value is None:
        return set()
    try:
        return {str(s) for s in pystr_to_symbolic(value).free_symbols}
    except Exception:
        return set()


def _mutated_scalar_names(sdfg: SDFG) -> Set[str]:
    """Names of ``Scalar`` descriptors written somewhere in ``sdfg``; an unwritten one is a
    read-only parameter and propagates as safely as a symbol."""
    mutated: Set[str] = set()
    for state in sdfg.all_states():
        for n in state.data_nodes():
            if state.in_degree(n) == 0:
                continue
            desc = sdfg.arrays.get(n.data)
            if isinstance(desc, dt.Scalar):
                mutated.add(n.data)
    return mutated


def _meta_read_symbols(blk: ControlFlowBlock) -> Set[str]:
    """Symbols a region's meta code reads: branch conditions, loop init / condition / update.
    ``free_symbols`` subtracts what the body rebinds, but the meta code runs first, so that read
    is live and must keep its assignment alive."""
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


def _is_array_access(value: Optional[str]) -> bool:
    """An assignment RHS reading a data container (``tbl[i]``, ``b_slice.idx``) is never
    propagated: sympy renders it as ``tbl(i)``, losing the bracket the filters key on, and a
    descriptor shape must be a function of symbols."""
    if value is None:
        return False
    if "[" in value or "]" in value:
        return True
    return _reads_struct_member(value)


def _reads_struct_member(value: str) -> bool:
    """``value`` reads an attribute off a plain name; a qualified call (``math.floor``) spells
    the same way, so a callee attribute does not count."""
    try:
        tree = ast.parse(value.strip(), mode='eval')
    except (SyntaxError, ValueError):
        return False  # not parseable as an expression -> leave it to the other filters
    callees = {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    return any(isinstance(node, ast.Attribute) and id(node) not in callees for node in ast.walk(tree))


def _resolve(value, table: Dict[str, Any]):
    """Substitute known symbol values from ``table`` into an assignment RHS, against the
    pre-edge table since one edge's assignments are simultaneous. Resolving now rather than
    chaining raw strings keeps a cyclic dependency from forming a substitution cycle."""
    if value is None:
        return None
    # Leave array-access values (``tbl[i]``) untouched (see :func:`_is_array_access`).
    if _is_array_access(value):
        return value
    try:
        expr = pystr_to_symbolic(value)
        repl = {}
        for s in expr.free_symbols:
            name = str(s)
            known = table.get(name)
            if known is not None and not _is_array_access(known):
                repl[s] = pystr_to_symbolic(known)
        if repl:
            expr = expr.subs(repl)
        # ``symstr``, not ``str``: operator-backed functions print as ``__right_shift(a, 1)``.
        return symstr(expr)
    except Exception:
        return value


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
        # If anything was modified, reapply
        return modified != ppl.Modifies.Nothing

    def apply_pass(self, sdfg: SDFG, _) -> Optional[Set[str]]:
        # Assumption: Symbols can only change in InterStateEdges

        before_free: Set[str] = {str(s) for s in sdfg.free_symbols}

        # Get all CFG blocks present in the SDFG
        all_cfg_blks = dict()
        for node, parent in sdfg.all_nodes_recursive():
            if isinstance(node, ControlFlowBlock):
                all_cfg_blks[node] = parent

        # Cached per SDFG: an unwritten Scalar is read-only and propagates like a symbol.
        self._mutated_scalars: Dict[SDFG, Set[str]] = {}
        for sd in sdfg.all_sdfgs_recursive():
            self._mutated_scalars[sd] = _mutated_scalar_names(sd)

        # For each CFG Block maintain a dict of incoming and outgoing symbols
        in_syms = {cfg_blk: {} for cfg_blk in all_cfg_blks.keys()}
        out_syms = {cfg_blk: {} for cfg_blk in all_cfg_blks.keys()}

        # Perform a forward fixed-point iteration to propagate symbols
        changed = True
        while changed:
            changed = False

            # Update incoming symbols
            for cfg_blk, parent in all_cfg_blks.items():
                new_in_syms = self._get_in_syms(sdfg, cfg_blk, parent, in_syms, out_syms)
                # Check if the incoming symbols have changed
                if new_in_syms != in_syms[cfg_blk]:
                    changed = True
                    in_syms[cfg_blk] = new_in_syms

            # Update outgoing symbols
            for cfg_blk, parent in all_cfg_blks.items():
                new_out_syms = self._get_out_syms(cfg_blk, parent, in_syms, out_syms)
                # Check if the outgoing symbols have changed
                if new_out_syms != out_syms[cfg_blk]:
                    changed = True
                    out_syms[cfg_blk] = new_out_syms

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
        """Drop interstate-edge assignments whose LHS is no longer referenced anywhere.
        ``replace_dict`` reaches into descriptors as well, so substitute there first (a no-op,
        since the symbol is that expression), then sweep to a fixed point."""
        removed: Set[str] = set()
        while True:
            this_round = self._eliminate_round(sdfg)
            if not this_round:
                break
            removed |= this_round
        return removed

    def _eliminate_round(self, sdfg: SDFG) -> Set[str]:
        """One sweep of dead-edge elimination across ``sdfg`` and its nested SDFGs; descriptors
        go first, since ``free_symbols`` pulls shape symbols through the access nodes."""
        eliminated: Set[str] = set()
        for sd in sdfg.all_sdfgs_recursive():
            # Propagatable: every binding edge agrees, and the RHS is not self-referential.
            bindings: Dict[str, Optional[str]] = {}
            for e in sd.all_interstate_edges():
                for lhs, rhs in e.data.assignments.items():
                    if rhs is None or lhs in _free_symbols(rhs):
                        bindings[lhs] = None
                        continue
                    if lhs not in bindings:
                        bindings[lhs] = rhs
                    elif bindings[lhs] is not None and bindings[lhs] != rhs:
                        bindings[lhs] = None
            # ``replace_dict`` also rewrites descriptor shapes, which live at SDFG scope: a
            # right-hand side reading data, or built from a symbol an inner block assigns (a loop
            # iteration variable), is not legal there. ``K = i + 1`` would size a transient by the
            # loop variable and allocate it outside the loop.
            invariant = {str(s) for s in sd.free_symbols} | set(sd.constants_prop.keys())
            safe_subs = {
                sym: rhs
                for sym, rhs in bindings.items()
                if rhs is not None and not _is_array_access(rhs) and _free_symbols(rhs) <= invariant
            }

            if safe_subs:
                sd.replace_dict(safe_subs, replace_keys=False, replace_in_graph=False)

            used_in_ir: Set[str] = set()
            for blk in sd.all_control_flow_blocks():
                used_in_ir |= {str(s) for s in blk.free_symbols}
                used_in_ir |= _meta_read_symbols(blk)
            for e in sd.all_interstate_edges():
                for rhs in e.data.assignments.values():
                    used_in_ir |= _free_symbols(rhs)
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
                rv = _resolve(v, sym_table)
                if rv is not None and (_free_symbols(rv) & edge_keys):
                    rv = None
                resolved[k] = rv
            # A carried-in value naming a symbol this edge reassigns is stale downstream.
            for sym in list(sym_table.keys()):
                val = sym_table[sym]
                if sym not in edge_keys and val is not None and (_free_symbols(val) & edge_keys):
                    sym_table[sym] = None
            sym_table.update(resolved)

            # Filter out symbols containing arrays accesses as they cannot be safely propagated (nested array accesses are not supported)
            sym_table = {k: v for k, v in sym_table.items() if not _is_array_access(v)}

            # Also filter out symbols containing views as they cannot be safely propagated (they are seen as pointers)
            sym_table = {
                k: v
                for k, v in sym_table.items() if v is None or not any([
                    str(s) in owner.arrays and isinstance(owner.arrays[str(s)], dt.View)
                    for s in pystr_to_symbolic(v).free_symbols
                ])
            }

            # A mutated scalar can change across the SDFG, so its readers cannot propagate.
            mutated = self._mutated_scalars.get(owner, set())
            sym_table = {k: v for k, v in sym_table.items() if v is None or not (scalars(v, owner.arrays) & mutated)}

            # Combine the symbols
            if i == 0:
                new_in_syms = sym_table
            else:
                self._combine_syms(new_in_syms, sym_table)

        # Nested starting CFBGs should inherit the symbols from their parent
        # Ignore SDFGs as nested SDFGs have symbol mappings
        if (parent.start_block == cfg_blk and not isinstance(parent, SDFG)) or (isinstance(parent, ConditionalBlock)
                                                                                and cfg_blk in parent.sub_regions()):
            # A start or branch region inherits the parent's symbols; some shapes carry their
            # own, so combine rather than assert.
            if new_in_syms:
                self._combine_syms(new_in_syms, in_syms[parent])
            else:
                new_in_syms = in_syms[parent]

            # For LoopRegions, remove loop carried variables from the incoming symbols
            if isinstance(parent, LoopRegion):
                new_in_syms = dict(new_in_syms)
                all_syms = set([s for e in parent.all_interstate_edges() for s in e.data.assignments.keys()])
                for sym in all_syms:
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
            # Any symbol that is assigned in the loop region is not propagated out
            new_out_syms = dict(in_syms[cfg_blk])
            for edge in cfg_blk.all_interstate_edges():
                for sym in edge.data.assignments.keys():
                    if sym in new_out_syms:
                        new_out_syms[sym] = None
            return new_out_syms

        elif isinstance(cfg_blk, ConditionalBlock):
            # Combine all outgoing symbols of the branches
            new_out_syms = dict(out_syms[cfg_blk.sub_regions()[0]])
            for b in cfg_blk.sub_regions():
                self._combine_syms(new_out_syms, out_syms[b])

            # If no else branch is present, also combine the incoming table (implicit else branch)
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

        # Remove all symbols that are None
        new_in_syms = {sym: val for sym, val in new_in_syms.items() if val is not None}
        new_out_syms = {sym: val for sym, val in new_out_syms.items() if val is not None}

        candidates = set(new_in_syms) | set(new_out_syms)
        if not candidates:
            return set()
        free_before = self._block_free_symbols(cfg_blk, parent)

        # An acyclic chain converges within ``#symbols`` rounds; the cap stops a cyclic one.
        max_iters = len(new_in_syms) + len(new_out_syms) + 2

        # A symbol the body reassigns is loop-carried: folding the incoming value into
        # ``while udiff > 0.001`` spins forever.
        loop_carried: Set[str] = set()
        if isinstance(cfg_blk, LoopRegion):
            loop_carried = {s for e in cfg_blk.all_interstate_edges() for s in e.data.assignments.keys()}

        changed = True
        iters = 0
        while changed and iters < max_iters:
            iters += 1
            changed = False
            free_sym = cfg_blk.free_symbols
            free_edge_sym = set([sym for edge in parent.out_edges(cfg_blk) for sym in edge.data.free_symbols])

            # Replace all symbols in the cfg_blk with their values
            if isinstance(cfg_blk, LoopRegion):
                meta_syms = {s: v for s, v in new_in_syms.items() if s not in loop_carried}
                cfg_blk.replace_meta_accesses(meta_syms)
            elif isinstance(cfg_blk, ConditionalBlock):
                cfg_blk.replace_meta_accesses(new_in_syms)
            elif isinstance(cfg_blk, SDFGState):
                cfg_blk.replace_dict(new_in_syms)
            else:
                # Don't replace, as the nested CFBGs should inherit the symbols from their parent
                pass

            # Same rule out: a substitution naming a key of this edge reads its own output.
            for edge in parent.out_edges(cfg_blk):
                edge_keys = set(edge.data.assignments.keys())
                if edge_keys:
                    edge_subs = {s: v for s, v in new_out_syms.items() if not (_free_symbols(v) & edge_keys)}
                else:
                    edge_subs = new_out_syms
                edge.data.replace_dict(edge_subs, replace_keys=False)

            # Check if the symbols have changed
            new_free_edge_sym = set([sym for edge in parent.out_edges(cfg_blk) for sym in edge.data.free_symbols])
            if free_sym != cfg_blk.free_symbols or free_edge_sym != new_free_edge_sym:
                changed = True

        # The candidate symbols that are no longer read here were propagated.
        return candidates & (free_before - self._block_free_symbols(cfg_blk, parent))

    def _combine_syms(self, sym1: Dict[str, Any], sym2: Dict[str, Any]) -> None:
        """Meet of two symbol tables at a control-flow join, in place: a symbol survives only
        when both sides agree, since a key absent on one side means no value known there."""
        for sym, val in sym2.items():
            if sym not in sym1 or sym1[sym] != val:
                sym1[sym] = None
        for sym in sym1:
            if sym not in sym2:
                sym1[sym] = None
