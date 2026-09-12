# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Turn a symbol an interstate edge assigns FROM ARRAY DATA back into ordinary data flow.

The frontend routes an array element into a body through an interstate assignment
(``b_index = b[i]``), so the state that consumes it reads a bare symbol that names no map
parameter yet holds a different value in every lane. Two passes then disagree about that symbol:
``ConvertTaskletsToTileOps`` treats a symbol operand as tile-invariant and would splat lane 0
across the tile, and ``lane_dependent_through_interstate_assignment`` catches exactly that and
refuses the whole kernel. On CloudSC one such assignment --
``za_slice_plus_zsolac_slice = zsolac[jl - 1] + za[jk - 1, jl - 1]`` -- refused the entire SDFG,
which left 974 already-classified maps un-tiled.

The read is data flow spelled as a symbol, so give it back its data flow: demote the symbol to a
Register scalar and re-emit the assignment as a tasklet reading the arrays through memlets, which
is a shape the tile pipeline already widens. Nothing here is vectorization-specific -- it removes a
read that ``AccessNode`` analysis cannot see -- but it runs in the vectorizer because that is where
the invisible read turns into a refusal.
"""
from typing import Any

from dace import SDFG
from dace import properties
from dace import symbolic
from dace.ordered import OrderedSet
from dace.sdfg import utils as sdutil
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation


def data_reading_assigned_symbols(sd: SDFG) -> OrderedSet:
    """Symbols ``sd``'s own interstate edges assign an expression that READS one of its arrays.

    Insertion-ordered, because the demotion it drives adds arrays to the SDFG.
    """
    found: OrderedSet = OrderedSet()
    for edge in sd.all_interstate_edges():
        for name, rhs in edge.data.assignments.items():
            try:
                expr = symbolic.pystr_to_symbolic(str(rhs))
            except Exception:  # noqa: BLE001 -- an unparsable RHS is not this pass's business
                continue
            # A subscript ``arr[i]`` carries the array as its head; a whole-array read spells the
            # name directly. Both are reads that no AccessNode records.
            heads = {str(sub.args[0]) for sub in expr.atoms(symbolic.Subscript)}
            heads |= {str(s) for s in expr.free_symbols}
            if heads & set(sd.arrays.keys()):
                found.add(name.strip())
    return found


@properties.make_properties
@transformation.explicit_cf_compatible
class DemoteDataReadingInterstateSymbols(ppl.Pass):
    """Demote every symbol an interstate assignment defines from array data to a scalar transient."""

    CATEGORY: str = 'Vectorization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> dict[type[ppl.Pass] | ppl.Pass, None]:
        return {}

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        """:returns: how many symbols were demoted, or ``None`` when none was."""
        demoted = 0
        for sd in sdfg.all_sdfgs_recursive():
            # Both gates answer from a whole-SDFG scan that depends on ``sd`` alone, so ask each
            # once per unmutated span rather than once per candidate -- the same accounting
            # ``LowerInterstateConditionalAssignmentsToTasklets`` makes, and for the same reason.
            structural: set[str] | None = None
            free_syms: set[str] | None = None
            for name in data_reading_assigned_symbols(sd):
                if name not in sd.symbols:
                    continue
                if structural is None:
                    structural = sdutil.structural_symbols(sd)
                    # Consulted only for a top-level SDFG; a nested one reads its parent's mapping.
                    free_syms = sd.free_symbols if sd.parent_nsdfg_node is None else set()
                # A symbol the graph's own structure is built from (a loop bound, a nested SDFG's
                # mapping) is not data and must stay a symbol; demoting it would rewrite control
                # flow rather than a read.
                if (not sdutil.symbol_demotes_to_transient_scalar(sd, name, free_symbols=free_syms)
                        or sdutil.symbol_carries_graph_structure(sd, name, structural=structural)):
                    continue
                # The symbol's OWN dtype: the arrays it reads carry it, and every consumer already
                # reads the symbol at the type it was declared with.
                sdutil.demote_symbol_to_scalar(sd, name, sd.symbols[name], None)
                demoted += 1
                structural = free_syms = None  # the demotion rewrote ``sd``; rebuild before the next ask
        return demoted or None

    def report(self, pass_retval: int) -> str:
        return f'Demoted {pass_retval} data-reading interstate symbols to scalars.'
