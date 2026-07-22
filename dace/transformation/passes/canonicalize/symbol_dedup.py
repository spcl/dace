# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Common-subexpression elimination for duplicate interstate-edge symbols.

Canonicalization can leave an interstate edge carrying two (or more) symbol
assignments with an *identical* right-hand side::

    ISEDGE assign: {'idx_index': 'idx[i]', 'idx_index_0': 'idx[i]'}

This shows up after a statement-split (``SplitStatements``) fissions a shared
gather into two single-statement maps that ``MapFusion`` then re-fuses over the
same range: each statement carried its own copy of the gather index, so the
fused body computes ``idx[i]`` twice. ``SymbolPropagation`` and
``ConstantPropagation`` do not merge these because the RHS reads a data array
(not a pure symbolic value), so neither is safe to *forward-substitute*; but the
two symbols provably hold the same value everywhere, so one can be *eliminated*.

:class:`SymbolDedup` performs that elimination as a sound CSE on interstate
symbols: two symbols are merged only if they are assigned on exactly the same
set of interstate edges and with an equal RHS on each of those edges (so at
every point one is read, the other has an identical reaching definition). All
uses of the dropped symbol are rewritten to the surviving keeper via DaCe's
symbol-replacement machinery, keeping the transform value-preserving (bit-exact).
"""
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from dace import SDFG, properties
from dace.sdfg.state import LoopRegion
from dace.symbolic import pystr_to_symbolic
from dace.transformation import pass_pipeline as ppl, transformation


@dataclass(unsafe_hash=True)
@properties.make_properties
@transformation.explicit_cf_compatible
class SymbolDedup(ppl.Pass):
    """Merge interstate-edge symbols that provably hold the same value everywhere.

    Sound CSE for interstate symbols: two symbols ``s1`` and ``s2`` are merged
    iff they are defined on exactly the same set of interstate edges and, on each
    such edge, with an equal RHS (compared after
    :func:`~dace.symbolic.pystr_to_symbolic` normalization, to be robust to
    formatting). Under that condition every reaching definition of ``s2`` is
    accompanied by an identical reaching definition of ``s1``, so replacing every
    use of ``s2`` with ``s1`` -- and dropping ``s2``'s now-redundant assignment
    and the symbol itself -- preserves values exactly.
    """

    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        # Renames symbols (Symbols), rewrites interstate edges + memlets (Edges),
        # and tasklet code / map ranges (Nodes).
        return ppl.Modifies.Symbols | ppl.Modifies.Edges | ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # Deduplication is idempotent on its own output but a preceding pass that
        # changes symbols or edges may expose fresh duplicates, so reapply then.
        return modified & (ppl.Modifies.Symbols | ppl.Modifies.Edges) != ppl.Modifies.Nothing

    def apply_pass(self, sdfg: SDFG, _: Dict) -> Optional[int]:
        """Deduplicate provably-equal interstate symbols across the whole SDFG.

        :param sdfg: The SDFG to modify.
        :param _: Pipeline results (unused).
        :return: The number of symbols eliminated, or ``None`` if none were.
        """
        merged = 0
        # Each SDFG is its own symbol namespace (nested SDFGs remap via
        # symbol_mapping), so dedup one SDFG scope at a time.
        for nested in sdfg.all_sdfgs_recursive():
            merged += self._dedup_sdfg(nested)
        return merged if merged else None

    def report(self, pass_retval: Optional[int]) -> str:
        return f'Deduplicated {pass_retval or 0} interstate-edge symbol(s).'

    def _normalize_rhs(self, rhs) -> str:
        """Normalize an assignment RHS so equal-but-differently-formatted
        expressions compare equal; fall back to the raw text if it will not parse."""
        text = str(rhs)
        try:
            return str(pystr_to_symbolic(text))
        except Exception:
            return text.strip()

    def _protected_symbols(self, sdfg: SDFG) -> Set[str]:
        """Symbols that must never be dropped: externally-required free symbols,
        parameters bound by the parent nested-SDFG mapping, and loop variables
        (whose updates live in loop metadata, not interstate edges)."""
        protected: Set[str] = set(map(str, sdfg.free_symbols))
        nsdfg = sdfg.parent_nsdfg_node
        if nsdfg is not None:
            protected |= set(nsdfg.symbol_mapping.keys())
        for cfg in sdfg.all_control_flow_regions(recursive=False):
            if isinstance(cfg, LoopRegion) and cfg.loop_variable:
                protected.add(str(cfg.loop_variable))
        return protected

    def _dedup_sdfg(self, sdfg: SDFG) -> int:
        """Deduplicate symbols within a single SDFG's interstate edges."""
        # 1. Collect, per symbol, the (edge -> normalized RHS) map of its
        #    assignments. Edge identity keys the "same set of edges" comparison.
        defs: Dict[str, Dict[int, str]] = {}
        for edge in sdfg.all_interstate_edges():
            for sym, rhs in edge.data.assignments.items():
                defs.setdefault(sym, {})[id(edge)] = self._normalize_rhs(rhs)

        if len(defs) < 2:
            return 0

        # 2. Group symbols by their def-signature. Two symbols share a signature
        #    iff they are assigned on exactly the same edges with an equal RHS on
        #    each -- i.e. they are provably equal everywhere and mergeable.
        by_signature: Dict[FrozenSet[Tuple[int, str]], List[str]] = {}
        for sym, edge_map in defs.items():
            by_signature.setdefault(frozenset(edge_map.items()), []).append(sym)

        protected = self._protected_symbols(sdfg)

        # 3. Per equivalence class, keep one canonical symbol (shortest name, ties
        #    broken lexicographically -- prefers 'idx_index' over 'idx_index_0')
        #    and map every other droppable member onto it.
        repl: Dict[str, str] = {}
        for syms in by_signature.values():
            if len(syms) < 2:
                continue
            keeper = min(syms, key=lambda name: (len(name), name))
            for sym in syms:
                if sym != keeper and sym not in protected:
                    repl[sym] = keeper

        if not repl:
            return 0

        # 4. Rewrite every USE of a dropped symbol to its keeper -- interstate-edge
        #    conditions and assignment RHSes, memlet subsets, tasklet code, map
        #    ranges, and nested-SDFG symbol mappings -- via DaCe's replacement
        #    machinery. replace_keys=False leaves assignment/symbol dict keys for
        #    the explicit removal below (so no key-collapse warnings are emitted).
        sdfg.replace_dict(repl, replace_in_graph=True, replace_keys=False)

        # 5. Drop each redundant assignment (the keeper is assigned identically on
        #    the same edge) and remove the now-dead symbols from the SDFG.
        dropped = set(repl.keys())
        for edge in sdfg.all_interstate_edges():
            for sym in [s for s in edge.data.assignments if s in dropped]:
                del edge.data.assignments[sym]
        for sym in dropped:
            if sym in sdfg.symbols:
                sdfg.remove_symbol(sym)

        return len(dropped)
