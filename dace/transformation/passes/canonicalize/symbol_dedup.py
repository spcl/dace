# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Common-subexpression elimination for duplicate interstate-edge symbols."""

from dataclasses import dataclass
from typing import Any

from dace import SDFG, properties
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.symbolic import pystr_to_symbolic
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation


@dataclass(unsafe_hash=True)
@properties.make_properties
@transformation.explicit_cf_compatible
class SymbolDedup(ppl.Pass):
    """Merge interstate-edge symbols that provably hold the same value everywhere."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        # Renames symbols (Symbols), rewrites interstate edges + memlets (Edges), and tasklet code / map ranges (Nodes).
        return ppl.Modifies.Symbols | ppl.Modifies.Edges | ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # Deduplication is idempotent on its own output but a preceding pass that
        # changes symbols or edges may expose fresh duplicates, so reapply then.
        return modified & (ppl.Modifies.Symbols | ppl.Modifies.Edges) != ppl.Modifies.Nothing

    def apply_pass(self, sdfg: SDFG, _: dict[str, Any]) -> int | None:
        """Deduplicate provably-equal interstate symbols across the whole SDFG."""
        merged = 0
        # Each SDFG is its own symbol namespace (nested SDFGs remap via
        # symbol_mapping), so dedup one SDFG scope at a time.
        for nested in sdfg.all_sdfgs_recursive():
            merged += self._dedup_sdfg(nested)
        return merged or None

    def report(self, pass_retval: int | None) -> str:
        return f"Deduplicated {pass_retval or 0} interstate-edge symbol(s)."

    def _normalize_rhs(self, rhs: object) -> str:
        """Normalize an assignment RHS so equal-but-differently-formatted
        expressions compare equal; fall back to the raw text if it will not parse."""
        text = str(rhs)
        try:
            return str(pystr_to_symbolic(text))
        except Exception:
            return text.strip()

    def _protected_symbols(self, sdfg: SDFG) -> dict[str, None]:
        """Symbols that must never be dropped: externally-required free symbols and
        parameters bound by the parent nested-SDFG mapping."""
        protected: dict[str, None] = dict.fromkeys(map(str, sdfg.free_symbols))
        nsdfg = sdfg.parent_nsdfg_node
        if nsdfg is not None:
            protected.update(dict.fromkeys(nsdfg.symbol_mapping.keys()))
        return protected

    def _dedup_sdfg(self, sdfg: SDFG) -> int:
        """Deduplicate symbols within a single SDFG's interstate edges."""
        # 1. Collect, per symbol, the (edge -> normalized RHS) map of its assignments.
        defs: dict[str, dict[int, str]] = {}
        for edge in sdfg.all_interstate_edges():
            for sym, rhs in edge.data.assignments.items():
                defs.setdefault(sym, {})[id(edge)] = self._normalize_rhs(rhs)

        if len(defs) < 2:
            return 0

        # 2. Group symbols by their def-signature.
        by_signature: dict[tuple[Any, frozenset[tuple[int, str]]], list[str]] = {}
        for sym, edge_map in defs.items():
            by_signature.setdefault((sdfg.symbols.get(sym), frozenset(edge_map.items())), []).append(sym)

        # A class of one merges nothing, so the whole-SDFG ``free_symbols`` walk behind
        # ``_protected_symbols`` is only paid once a merge is actually on the table.
        classes = [syms for syms in by_signature.values() if len(syms) >= 2]
        if not classes:
            return 0
        # A loop header assigns its variable as well: it equals the others only until that loop runs.
        scoped = {
            str(cfg.loop_variable)
            for cfg in sdfg.all_control_flow_regions(recursive=False)
            if isinstance(cfg, LoopRegion) and cfg.loop_variable
        }
        for state in sdfg.states():
            for node in state.nodes():
                if isinstance(node, nodes.MapEntry):
                    scoped.update(node.map.params)
        classes = [[sym for sym in syms if sym not in scoped] for syms in classes]
        classes = [syms for syms in classes if len(syms) >= 2]
        protected = self._protected_symbols(sdfg)

        # 3. Per equivalence class, keep one canonical symbol (shortest name, ties broken lexicographically -- prefers
        # 'idx_index' over 'idx_index_0') and map every other droppable member onto it.
        repl: dict[str, str] = {}
        for syms in classes:
            keeper = min(syms, key=lambda name: (len(name), name))
            for sym in syms:
                if sym != keeper and sym not in protected:
                    repl[sym] = keeper

        if not repl:
            return 0

        # 4. Rewrite every USE of a dropped symbol to its keeper
        sdfg.replace_dict(repl, replace_in_graph=True, replace_keys=False)

        # 5. Drop each redundant assignment (the keeper is assigned identically on
        #    the same edge) and remove the now-dead symbols from the SDFG.
        dropped = dict.fromkeys(repl.keys())
        for edge in sdfg.all_interstate_edges():
            for sym in [s for s in edge.data.assignments if s in dropped]:
                del edge.data.assignments[sym]
        for sym in dropped:
            if sym in sdfg.symbols:
                sdfg.remove_symbol(sym)

        return len(dropped)
