# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import copy
from dataclasses import dataclass
from dace.sdfg.state import (
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
    """Free symbol names of an interstate-edge assignment value (RHS).

    :param value: The assignment RHS (a string), or ``None``.
    :returns: The set of free symbol names; empty for ``None`` / unparseable.
    """
    if value is None:
        return set()
    try:
        return {str(s) for s in pystr_to_symbolic(value).free_symbols}
    except Exception:
        return set()


def _mutated_scalar_names(sdfg: SDFG) -> Set[str]:
    """Names of ``Scalar`` descriptors in ``sdfg`` that are written somewhere -- an
    AccessNode of that scalar has at least one in-edge.

    A ``Scalar`` whose value is fixed for the whole SDFG run (no in-edges into any
    of its AccessNodes) is semantically a read-only parameter, indistinguishable
    from a symbol for propagation. The Fortran frontend registers ``intent(in)``
    arguments such as ``kidia`` / ``kfdia`` / ``klev`` as ``Scalar`` descriptors;
    refusing to propagate ``kfdia_plus_1 = (kfdia + 1)`` because the RHS reads a
    ``Scalar`` would strand every bound-symbol alias forever (cloudsc has hundreds).
    The stricter "is this scalar actually mutated?" check is sound: if the value
    is fixed for the SDFG run, the symbol behaves like any other free symbol.

    :param sdfg: The SDFG to inspect.
    :returns: Names of ``Scalar`` descriptors with at least one write site.
    """
    mutated: Set[str] = set()
    for state in sdfg.all_states():
        for n in state.data_nodes():
            if state.in_degree(n) == 0:
                continue
            desc = sdfg.arrays.get(n.data)
            if isinstance(desc, dt.Scalar):
                mutated.add(n.data)
    return mutated


def _resolve(value, table: Dict[str, Any]):
    """Substitute known symbol values from ``table`` into an assignment RHS.

    Interstate-edge assignments are simultaneous, so the RHSes of one edge are
    resolved against the PRE-edge (incoming) symbol table -- a swap
    ``{tx: y, ty: x}`` resolves ``tx`` to the old value of ``y`` and ``ty`` to
    the old value of ``x``. Resolving here (rather than leaving raw
    ``tx: 'y'`` strings to be chained later) collapses symbol-to-symbol chains
    to constants/expressions up front, so a cyclic dependency (``x: tx,
    tx: y, y: ty, ty: x``) never forms a substitution cycle.

    :param value: The assignment RHS (a string), or ``None``.
    :param table: Known ``{symbol: value-string-or-None}`` mapping.
    :returns: The resolved RHS string (or ``None`` for ``None`` input).
    """
    if value is None:
        return None
    # Leave array-access values (``tbl[i]``) untouched: parsing them through
    # sympy would turn ``tbl[i]`` into ``tbl(i)`` and lose the ``[`` that the
    # downstream filter uses to drop non-propagatable nested-array accesses.
    if "[" in value or "]" in value:
        return value
    try:
        expr = pystr_to_symbolic(value)
        repl = {}
        for s in expr.free_symbols:
            name = str(s)
            known = table.get(name)
            # Skip substituting array-access values for the same reason.
            if known is not None and "[" not in known and "]" not in known:
                repl[s] = pystr_to_symbolic(known)
        if repl:
            expr = expr.subs(repl)
        # Render with the DaCe printer, not sympy's ``str``: the operator-backed
        # functions (``__bitwise_and``, ``__right_shift``, ...) print as their
        # class names under ``str`` (``__right_shift(a, 1)``), which is neither
        # valid Python (the codeblock language) nor valid C++. ``symstr`` lowers
        # them to the corresponding operator, so the propagated value round-trips
        # through the Python codeblock and the C++ codegen alike.
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

        # Postcondition (checked at the end): propagation only ever *eliminates*
        # symbols (substitutes known values forward, drops dead assignments), so
        # it must never INTRODUCE a free (externally-required / undefined) symbol.
        # Recording the entry set lets a value rendered into a bad spelling -- e.g.
        # an operator-function printed as its sympy class name ``__right_shift`` --
        # be caught here instead of leaking into a condition / codegen.
        before_free: Set[str] = {str(s) for s in sdfg.free_symbols}

        # Get all CFG blocks present in the SDFG
        all_cfg_blks = dict()
        for node, parent in sdfg.all_nodes_recursive():
            if isinstance(node, ControlFlowBlock):
                all_cfg_blks[node] = parent

        # Per-SDFG: which ``Scalar`` descriptors are actually mutated (have an
        # in-edge into one of their AccessNodes). A scalar that is NEVER written
        # is a read-only parameter and behaves like a symbol for propagation
        # purposes (Fortran ``intent(in)`` args like ``kfdia`` show up as
        # ``Scalar`` descriptors but their value is fixed for the whole SDFG
        # run, so they propagate safely -- the cloudsc ``kfdia_plus_1_N``
        # bound-symbol aliases need this to clean up, and any shape symbol
        # referencing a read-only Scalar is similarly safe). Cached per-SDFG so
        # the per-block filter is O(1) per call.
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

        # Update symbols in the cfg_blk, accumulating the symbols actually
        # propagated (eliminated from a block / edge). The pipeline treats a
        # non-None return as "this pass modified the SDFG" and a None return as
        # "no change" (see ``Pipeline.apply_pass``); returning an honest set
        # lets a FixedPointPipeline such as ``SimplifyPass`` converge instead of
        # re-running this pass forever on a no-op.
        propagated: Set[str] = set()
        for cfg_blk, parent in all_cfg_blks.items():
            propagated |= self._update_syms(cfg_blk, parent, in_syms, out_syms)
        # Substitution leaves the *defining* iedge assignment (``k_plus_1 = klev + 1``)
        # in place even after every consumer has been rewritten to use the resolved
        # value. Sweep those dead assignments to a fixed point so the pass output
        # is canonical (e.g. cloudsc's 346 bound-symbol ``+1`` assignments disappear
        # once their downstream uses are gone).
        eliminated = self._eliminate_dead_iedge_assignments(sdfg)
        if eliminated:
            propagated |= eliminated

        # Postcondition: propagation must not introduce a new free symbol. A
        # violation means a propagated value was rendered into a name that does
        # not resolve (e.g. ``__right_shift(a, 1)`` instead of ``(a >> 1)``);
        # fail here rather than emit an SDFG with an unbound symbol.
        new_free: Set[str] = {str(s) for s in sdfg.free_symbols} - before_free
        if new_free:
            raise ValueError(f"SymbolPropagation introduced free symbol(s) {sorted(new_free)}: a propagated "
                             f"value rendered to an unresolvable name. Symbol propagation must only eliminate "
                             f"symbols, never introduce them.")

        return propagated if propagated else None

    def _eliminate_dead_iedge_assignments(self, sdfg: SDFG) -> Set[str]:
        """Drop interstate-edge assignments whose LHS is no longer referenced anywhere.

        After :meth:`_update_syms` rewrites every use site in the dataflow graph, the
        defining iedge assignment ``X = expr`` becomes dead -- *unless* ``X`` is still
        referenced by an array descriptor's shape / strides / offset. The IR-level
        ``replace_dict`` does not reach into descriptors, so those references survive
        propagation and pin the iedge alive. Before deciding an iedge is dead, we
        substitute ``X -> expr`` into the owning SDFG's descriptors (a semantic no-op
        since the symbol IS that expression by construction), then sweep.

        Iterates to a fixed point so chained assignments (``a = klev + 1; b = a + 1``)
        unravel from the leaves inward.

        :param sdfg: The SDFG to clean up.
        :returns: The set of LHS names that were removed (empty if no change).
        """
        removed: Set[str] = set()
        while True:
            this_round = self._eliminate_round(sdfg)
            if not this_round:
                break
            removed |= this_round
        return removed

    def _eliminate_round(self, sdfg: SDFG) -> Set[str]:
        """One sweep of dead-iedge elimination across ``sdfg`` and its nested SDFGs.

        Substitutes propagatable iedge LHSes into the SDFG's descriptors first, since
        ``SDFGState.free_symbols`` pulls array-shape symbols via the access nodes that
        read those arrays (state.py:709). Without the descriptor substitution, the
        symbol still reads as "used in IR" and the iedge never gets eliminated.
        """
        eliminated: Set[str] = set()
        for sd in sdfg.all_sdfgs_recursive():
            # Gather candidate substitutions: a symbol is propagatable if every iedge
            # binding it agrees on the same RHS (no per-edge disagreement -> ambiguous)
            # and the RHS does not self-reference (a self-reference like ``i = i + 1``
            # marks a loop-carried iter, which cannot be substituted out).
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
            safe_subs = {sym: rhs for sym, rhs in bindings.items() if rhs is not None}

            # Substitute every propagatable LHS into the SDFG's descriptors. Symbols
            # whose value was already substituted everywhere will have no live shape
            # reference after this; the dead-iedge sweep below then drops them.
            if safe_subs:
                sd.replace_dict(safe_subs, replace_keys=False, replace_in_graph=False)

            # Now compute the IR-level used set; this no longer includes the symbols
            # that have been folded into descriptors.
            used_in_ir: Set[str] = set()
            for blk in sd.all_control_flow_blocks():
                used_in_ir |= {str(s) for s in blk.free_symbols}
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
            # Drop the now-orphaned declarations so nested-SDFG validation does not
            # demand the symbol from outside.
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
        # Container-aware filters below must consult the SDFG that OWNS this block:
        # ``all_cfg_blks`` spans every nested SDFG, but ``sdfg`` is always the
        # top-level one, so looking up ``sdfg.arrays`` for a nested block silently
        # misses its containers (e.g. cloudsc's ``zqe = zqe_5`` inside the
        # LoopToMap-nested ``loop_body``: ``zqe_5`` is a mutated Scalar of the
        # NESTED SDFG, invisible in the root's arrays, so the mutated-scalar guard
        # no-opped and the fold produced a connector-less container read in
        # tasklet code).
        owner = cfg_blk.sdfg
        # Combine the outgoing symbols of all incoming edges with their assignments to the cfg_blk
        new_in_syms = {}
        for i, edge in enumerate(parent.in_edges(cfg_blk)):
            sym_table = copy.deepcopy(out_syms[edge.src])
            # Resolve this edge's RHSes against the PRE-edge table (simultaneous
            # assignment semantics), then apply -- collapsing symbol chains and
            # breaking cyclic dependencies instead of storing raw chained strings.
            # A resolved value that references ANY symbol assigned on this SAME
            # edge must stay LIVE (None) rather than be propagated: the edge's
            # assignments fire simultaneously and rebind those symbols at
            # runtime, so the resolved value (computed from the OLD values)
            # differs from what a downstream use -- which sees the NEW values --
            # would compute. Propagating it would double-count the rebinding
            # (e.g. on ``{m: t, n: t + 1}`` with ``t = m + 2``: ``m`` resolves to
            # ``m + 2`` (self-ref) and ``n`` to ``m + 3`` (reads reassigned
            # ``m``); both must stay live so ``B[m]`` / ``B[n]`` read the edge's
            # outputs, not a re-applied expression).
            edge_keys = set(edge.data.assignments.keys())
            # Resolve this edge's RHSes against the (un-invalidated) PRE-edge
            # table: the assignments read their incoming values, and a carried
            # value such as ``k = j + 1`` still denotes the pre-edge ``j`` here.
            resolved = {}
            for k, v in edge.data.assignments.items():
                rv = _resolve(v, sym_table)
                if rv is not None and (_free_symbols(rv) & edge_keys):
                    rv = None
                resolved[k] = rv
            # A value CARRIED IN from the predecessor (already in ``sym_table``,
            # not (re)assigned on this edge) that references a symbol this edge
            # reassigns is now STALE for the downstream block: it was computed
            # from that symbol's pre-edge value, but the edge rebinds the
            # symbol, so the block past this edge sees the NEW value.
            # Propagating the carried value would read the wrong (old) value
            # (e.g. carrying ``k = j + 1`` past an edge ``j = j + 2`` would make
            # ``c[k]`` read ``c[j + 1]`` against the reassigned ``j``, an
            # off-by-two). Invalidate such entries (-> None / live) -- the same
            # simultaneity rule the per-edge guard above applies to the edge's
            # own assignments. Done AFTER resolving so the edge's own RHSes still
            # see the carried pre-edge values.
            for sym in list(sym_table.keys()):
                val = sym_table[sym]
                if sym not in edge_keys and val is not None and (_free_symbols(val) & edge_keys):
                    sym_table[sym] = None
            sym_table.update(resolved)

            # Filter out symbols containing arrays accesses as they cannot be safely propagated (nested array accesses are not supported)
            sym_table = {k: v for k, v in sym_table.items() if v is None or ("[" not in v and "]" not in v)}

            # Also filter out symbols containing views as they cannot be safely propagated (they are seen as pointers)
            sym_table = {
                k: v
                for k, v in sym_table.items() if v is None or not any([
                    str(s) in owner.arrays and isinstance(owner.arrays[str(s)], dt.View)
                    for s in pystr_to_symbolic(v).free_symbols
                ])
            }

            # Skip assignments whose RHS reads a MUTATED scalar -- one whose
            # value can change across the SDFG (any AccessNode of that scalar
            # has an in-edge). Read-only scalars (e.g. Fortran ``intent(in)``
            # args like ``kfdia`` registered as ``Scalar`` descriptors but
            # never written, or shape symbols backed by such scalars) are
            # constant for the run and behave like symbols, so propagating
            # ``kfdia_plus_1 = (kfdia + 1)`` -- and any shape-symbol expression
            # that resolves through them -- is safe and required for cloudsc's
            # bound-symbol aliases to clean up.
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
            # A start / branch region normally has no in-edges, so the
            # edge-accumulated table is empty and it inherits the parent's
            # incoming symbols. On some cross-CFG shapes the block can already
            # carry edge-accumulated symbols; combine conservatively
            # (disagreements -> None) rather than assert emptiness, which
            # crashed on those shapes.
            if new_in_syms:
                self._combine_syms(new_in_syms, in_syms[parent])
            else:
                new_in_syms = in_syms[parent]

            # For LoopRegions, remove loop carried variables from the incoming symbols
            if isinstance(parent, LoopRegion):
                new_in_syms = copy.deepcopy(new_in_syms)
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
            new_out_syms = copy.deepcopy(in_syms[cfg_blk])
            for edge in cfg_blk.all_interstate_edges():
                for sym in edge.data.assignments.keys():
                    if sym in new_out_syms:
                        new_out_syms[sym] = None
            return new_out_syms

        elif isinstance(cfg_blk, ConditionalBlock):
            # Combine all outgoing symbols of the branches
            new_out_syms = copy.deepcopy(out_syms[cfg_blk.sub_regions()[0]])
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

            new_out_syms = copy.deepcopy(out_syms[sink_nodes[0]])
            for n in sink_nodes:
                self._combine_syms(new_out_syms, out_syms[n])
            return new_out_syms

    def _block_free_symbols(self, cfg_blk: ControlFlowBlock, parent: ControlFlowRegion) -> Set[str]:
        """Names of symbols read by ``cfg_blk`` and by its outgoing edges.

        :param cfg_blk: The block to inspect.
        :param parent: The block's parent region (for its out-edges).
        :returns: The set of free-symbol names.
        """
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
        new_in_syms = copy.deepcopy(in_syms[cfg_blk])
        new_out_syms = copy.deepcopy(out_syms[cfg_blk])

        # Remove all symbols that are None
        new_in_syms = {sym: val for sym, val in new_in_syms.items() if val is not None}
        new_out_syms = {sym: val for sym, val in new_out_syms.items() if val is not None}

        # Symbols this block could propagate, and the symbols it reads before
        # substitution -- their set difference after substitution is what was
        # actually eliminated (returned so the pipeline knows what changed).
        candidates = set(new_in_syms) | set(new_out_syms)
        if not candidates:
            return set()
        free_before = self._block_free_symbols(cfg_blk, parent)

        # Iteration cap: each pass resolves at least one more substitution
        # level, so a legitimate (acyclic) substitution chain converges within
        # ``#symbols`` passes. A CYCLIC value dependency (e.g. a swap
        # ``x: tx, tx: y, y: ty, ty: x``) would otherwise oscillate the free-
        # symbol set forever; the cap guarantees termination (leaving the
        # cyclic symbols un-substituted, which is conservative and correct).
        max_iters = len(new_in_syms) + len(new_out_syms) + 2

        # Symbols reassigned inside a loop body are loop-carried: the loop's
        # condition and update statement observe their body-updated value on
        # every iteration past the first, not the value flowing in from before
        # the loop. Substituting the incoming value into these meta-accesses
        # would fold a stale first-iteration value into the condition -- e.g.
        # ``while udiff > 0.001`` with ``udiff = 1.0`` ahead of the loop and
        # ``udiff = <reduction>`` inside collapses to ``1.0 > 0.001``, an
        # infinite loop -- so a read by the condition keeps the symbol live.
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

            # Also replace all symbols in the outgoing edges with their values.
            # Interstate-edge assignments are SIMULTANEOUS: a symbol read in an
            # assignment RHS denotes its INCOMING value, not the value being
            # assigned on the same edge. Substituting a propagated value that
            # references a symbol which is itself a KEY on this edge would make
            # the RHS read the edge's outgoing value -- a same-edge read-write
            # race that validation rejects (e.g. substituting ``anext -> a + b``
            # into ``{b: a, a: anext}`` yields ``{b: a, a: a + b}``). Drop such
            # colliding substitutions for that edge.
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
        """Meet of two symbol tables at a control-flow join; modifies ``sym1`` in place.

        A symbol keeps its value only when BOTH sides agree. A key present on one side and absent
        from the other carries no shared guarantee -- absence means "no value known on that path",
        not "unchanged" -- so it becomes live (``None``). Without the second loop a branch whose
        binding the array-access / View / mutated-scalar filters in :meth:`_get_in_syms` dropped
        lets the other branch's value escape the join: cloudsc's ``then: fac = 1.0`` vs
        ``else: fac = zfokoop[jl]`` folded every post-join use of ``fac`` to the literal ``1.0``.
        """
        for sym, val in sym2.items():
            if sym not in sym1 or sym1[sym] != val:
                sym1[sym] = None
        for sym in sym1:
            if sym not in sym2:
                sym1[sym] = None
