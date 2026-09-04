# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Drop transient descriptors that nothing in their own SDFG names any more."""
import re
from typing import Any, Dict, Optional, Set

from dace.ordered import OrderedSet

from dace import SDFG, data, properties
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation


def code_text(block: Any) -> str:
    """Every code string a CodeBlock-like property can carry, concatenated.

    ``CodeBlock.get_free_symbols`` walks a PYTHON ast and returns an empty set for every other
    language, so a name used only inside a C++ tasklet is invisible to symbol analysis. This
    pass reads the raw text instead, which sees it whatever the language.
    """
    if block is None:
        return ''
    code = block.code if isinstance(block, properties.CodeBlock) else block
    if isinstance(code, str):
        return code
    if isinstance(code, (list, tuple)):
        return '\n'.join(str(c) for c in code)
    return str(code)


def referenced_names(sdfg: SDFG) -> Set[str]:
    """Names that anything in ``sdfg`` itself (not its children) still refers to.

    Structural references (access nodes, memlets) are collected exactly. Everything that can only
    mention a name as TEXT -- tasklet bodies, interstate conditions and assignments, loop headers --
    is collected as word tokens, which over-approximates. Over-approximating is the safe direction:
    a name kept alive by accident costs a descriptor, a name dropped by accident is a miscompile.
    """
    used: Set[str] = set()
    text: list = [sdfg.init_code, sdfg.exit_code]
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nodes.AccessNode):
                used.add(node.data)
            elif isinstance(node, nodes.Tasklet):
                text.append(node.code)
                text.append(node.code_global)
                text.append(node.code_init)
                text.append(node.code_exit)
            elif isinstance(node, nodes.NestedSDFG):
                text.extend(str(v) for v in node.symbol_mapping.values())
            elif isinstance(node, nodes.MapEntry):
                text.append(str(node.map.range))
        for edge in state.edges():
            if edge.data is not None and edge.data.data is not None:
                used.add(edge.data.data)
    for edge in sdfg.all_interstate_edges():
        text.append(edge.data.condition)
        text.extend(str(v) for v in edge.data.assignments.values())
    for cfr in sdfg.all_control_flow_regions():
        if isinstance(cfr, LoopRegion):
            text.extend((cfr.loop_condition, cfr.init_statement, cfr.update_statement))
        elif isinstance(cfr, ConditionalBlock):
            text.extend(cond for cond, _branch in cfr.branches)
    used |= set(re.findall(r'[A-Za-z_]\w*', '\n'.join(code_text(t) for t in text if t is not None)))
    return used


@properties.make_properties
@transformation.explicit_cf_compatible
class PruneUnreferencedTransients(ppl.Pass):
    """Remove transient descriptors no node, memlet, symbol or code string names any more.

    Stages late in the recipe delete the last reader of a temporary without deleting its
    descriptor, and the reclaimers that would collect it (``ArrayElimination``) run before them
    and skip ``Scalar`` outright. The residue is invisible to codegen but not to a re-run: it is
    descriptor churn between one canonicalization and the next, and it is dead weight in every
    serialized SDFG in between.
    """

    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & (ppl.Modifies.AccessNodes | ppl.Modifies.Descriptors))

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Set[str]]:
        removed: OrderedSet = OrderedSet()
        for sd in sdfg.all_sdfgs_recursive():
            used = referenced_names(sd)
            for name, desc in list(sd.arrays.items()):
                if not desc.transient or name in used or name in sd.symbols or name in sd.constants_prop:
                    continue
                if isinstance(desc, data.DistributedDescriptor):
                    continue
                if isinstance(desc, data.Structure) and len(desc.members) > 0:
                    continue
                sd.remove_data(name, validate=False)
                removed.add(f'{sd.label}.{name}')
        return set(removed) or None

    def report(self, pass_retval: Set[str]) -> str:
        return f'Pruned {len(pass_retval)} unreferenced transients: {sorted(pass_retval)}.'
