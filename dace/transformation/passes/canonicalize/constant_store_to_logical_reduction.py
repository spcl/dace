# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A store of ``True`` (or ``False``) the loop counter does not index, under any guard, is a logical OR (AND) reduction
into the element it names.

``for c: if g[c]: flag[k] = True`` sets ``flag[k]`` when some iteration's guard holds and leaves it otherwise; each
store writes the same value, so the order of the iterations does not matter, but the loop is not last-write-wins
either: an iteration whose guard fails writes nothing. As the write-conflict resolution ``flag[k] = flag[k] or True``
the store means the same in a sequential loop, and the loop lift reads it as a reduction (``||``) instead of a conflict
on ``flag[k]``. ``False`` reduces with ``and`` alike. Only the boolean literals qualify (Fortran ``.TRUE.``, also into a
``LOGICAL`` held as an integer): a store of the integer ``0`` or ``1`` is a value (``enabled[s] = 0`` clears a slot of a
worklist), and reading it as a flag reduction hides the store from the lifts that prove such loops.
"""

import ast
from typing import Any

from dace import SDFG, dtypes, properties
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.analysis import loop_analysis

#: The write-conflict resolution of a store of each truth value.
LOGICAL_WCR = {True: "lambda a, b: a or b", False: "lambda a, b: a and b"}


def stored_truth(tasklet: nodes.Tasklet) -> bool | None:
    """The truth value a tasklet ``out = True`` (``False``) with no inputs stores, else ``None``."""
    if tasklet.code.language != dtypes.Language.Python or tasklet.in_connectors:
        return None
    body = tasklet.code.code
    if len(body) != 1 or not isinstance(body[0], ast.Assign) or not isinstance(body[0].value, ast.Constant):
        return None
    value = body[0].value.value
    return value if type(value) is bool else None


def constant_stores(loop: LoopRegion) -> dict[str, list[tuple[Any, bool | None]]]:
    """Per container, every value store into it in ``loop``: ``(edge, truth)``, ``truth`` ``None`` for a store that is
    not a lone truth constant."""
    found: dict[str, list[tuple[Any, bool | None]]] = {}
    for state in loop.states():
        for node in state.data_nodes():
            for edge in state.in_edges(node):
                if edge.data.is_empty():
                    continue
                truth = stored_truth(edge.src) if isinstance(edge.src, nodes.Tasklet) else None
                found.setdefault(node.data, []).append((edge, None if edge.data.wcr else truth))
    return found


@properties.make_properties
@transformation.explicit_cf_compatible
class ConstantStoreToLogicalReduction(ppl.Pass):
    """Turn the truth-constant stores a loop's counter does not index into logical reductions (module docstring)."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        converted = 0
        for region in sdfg.all_control_flow_regions(recursive=True):
            if isinstance(region, LoopRegion) and region.loop_variable:
                converted += self.convert(region)
        return converted or None

    def convert(self, loop: LoopRegion) -> int:
        read = {
            n.data
            for st in loop.states()
            for n in st.data_nodes()
            if any(not e.data.is_empty() for e in st.out_edges(n))
        }
        read |= loop_analysis.control_reads(loop)
        converted = 0
        for name, stores in constant_stores(loop).items():
            truths = {truth for edge, truth in stores}
            if name in read or len(truths) != 1 or None in truths:
                continue
            # A store the counter indexes hits its own element per iteration: no conflict to resolve
            if any(loop.loop_variable in {str(s) for s in edge.data.subset.free_symbols} for edge, truth in stores):
                continue
            for edge, truth in stores:
                edge.data.wcr = LOGICAL_WCR[truth]
            converted += 1
        return converted
