# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pass declaring the symbols that a map's range makes nonnegative as such."""
from typing import Any, Dict, Optional

from dace import nodes, properties, symbolic
from dace.sdfg import SDFG
from dace.transformation import pass_pipeline as ppl, transformation


@properties.make_properties
@transformation.explicit_cf_compatible
class NonnegativeMapSymbols(ppl.Pass):
    """Replaces, within the scope of each map, the symbols its range makes nonnegative by symbols that carry that
    assumption: a map parameter whose range starts at a nonnegative value with a positive step, and the extent ``N`` of
    a range ``0:N``. Code generation prints ``%`` and ``//`` as C's operators where the assumptions of the symbols show
    that they agree with the floored ones."""

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """
        :param sdfg: The SDFG to transform, recursively including nested SDFGs.
        :param pipeline_results: Results of previously applied passes (unused).
        :returns: The number of maps whose scope was rewritten, or ``None`` if none.
        """
        count = 0
        for nested in sdfg.all_sdfgs_recursive():
            for state in nested.all_states():
                for entry in state.nodes():
                    if isinstance(entry, nodes.MapEntry) and self.declare(nested, state, entry):
                        count += 1
        return count if count > 0 else None

    @staticmethod
    def declare(sdfg: SDFG, state, entry: nodes.MapEntry) -> bool:
        nonnegative = []
        for param, (begin, end, step) in zip(entry.map.params, entry.map.range.ranges):
            if begin.is_nonnegative and step.is_positive:
                nonnegative.append(symbolic.pystr_to_symbolic(param))
            if begin == 0 and isinstance(end + 1, symbolic.symbol):
                nonnegative.append(end + 1)
        symrepl = {
            symbol: symbolic.symbol(symbol.name, sdfg.symbols.get(symbol.name, symbol.dtype), nonnegative=True)
            for symbol in nonnegative if not symbol.is_nonnegative
        }
        if not symrepl:
            return False
        state.scope_subgraph(entry).replace_dict({symbol.name: symbol.name for symbol in symrepl}, symrepl)
        return True
