# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Replace an interstate read ``A[e]`` of a transient that one map fills with ``A[i] = f(i)`` by ``f(e)``.

``np.arange(n)`` lands as such a fill. A scatter keyed on ``rows[i]`` then shows the loop variable itself as its
slot, which the dependence tests read as injective; through the array it is a data-dependent index that only a
runtime conflict check can clear.
"""

import ast
import copy
from typing import Any

from dace import SDFG, data, dtypes
from dace.frontend.python import astutils
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation


class IotaFill(ast.NodeTransformer):
    """Strips the dtype casts around a fill expression (``int64_t(0) + i * int64_t(1)``)."""

    CASTS = frozenset(tc.to_string() for tc in dtypes.TYPECLASS_TO_STRING) | frozenset(
        tc.ctype for tc in dtypes.TYPECLASS_TO_STRING
    )

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        name = astutils.rname(node.func)
        if name is not None and name.split(".")[-1] in self.CASTS and len(node.args) == 1 and not node.keywords:
            return node.args[0]
        return node


def assigned_symbols(sdfg: SDFG) -> set[str]:
    """Symbols an interstate edge or a map of ``sdfg`` (nested SDFGs included) binds."""
    names = {lhs for sd in sdfg.all_sdfgs_recursive() for e in sd.all_interstate_edges() for lhs in e.data.assignments}
    names |= {p for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry) for p in n.map.params}
    return names


def iota_fill(sdfg: SDFG, name: str) -> tuple[str, ast.AST] | None:
    """``(param, f)`` when the only write of the 1-D integer transient ``name`` is one map ``name[param] = f(param)``
    over its whole extent, ``f`` reading nothing but ``param``, constants and symbols nothing rebinds."""
    desc = sdfg.arrays[name]
    if not isinstance(desc, data.Array) or isinstance(desc, data.View) or not desc.transient:
        return None
    if len(desc.shape) != 1 or desc.dtype not in dtypes.INTEGER_TYPES:
        return None
    writes = [(st, e) for st in sdfg.states() for n in st.data_nodes() if n.data == name for e in st.in_edges(n)]
    if len(writes) != 1:
        return None
    state, outer = writes[0]
    exit_node = outer.src
    if not isinstance(exit_node, nodes.MapExit) or outer.data.subset is None:
        return None
    if outer.data.subset.num_elements() != desc.total_size:
        return None
    entry = state.entry_node(exit_node)
    if len(entry.map.params) != 1 or entry.map.range.ranges[0] != (0, desc.shape[0] - 1, 1):
        return None
    param = entry.map.params[0]
    inner = [e for e in state.in_edges(exit_node) if e.dst_conn == "IN_" + outer.src_conn[len("OUT_"):]]
    if len(inner) != 1 or not isinstance(inner[0].src, nodes.Tasklet):
        return None
    tasklet, memlet = inner[0].src, inner[0].data
    if str(memlet.subset) != param or memlet.wcr is not None or tasklet.language != dtypes.Language.Python:
        return None
    if any(not e.data.is_empty() for e in state.in_edges(tasklet)) or len(state.out_edges(tasklet)) != 1:
        return None
    body = tasklet.code.code
    if len(body) != 1 or not isinstance(body[0], ast.Assign) or len(body[0].targets) != 1:
        return None
    if astutils.rname(body[0].targets[0]) != inner[0].src_conn:
        return None
    fill = IotaFill().visit(copy.deepcopy(body[0].value))
    rebound = assigned_symbols(sdfg) - {param}
    for node in ast.walk(fill):
        if isinstance(node, (ast.Subscript, ast.Call, ast.Attribute)):
            return None
        if isinstance(node, ast.Name) and (node.id in rebound or node.id in sdfg.arrays):
            return None
    return param, fill


class SubstituteIota(ast.NodeTransformer):
    """Rewrites ``name[e]`` into the fill expression at ``e``."""

    def __init__(self, name: str, param: str, fill: ast.AST):
        self.name, self.param, self.fill = name, param, fill
        self.count = 0

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        if not isinstance(node.value, ast.Name) or node.value.id != self.name:
            return node
        index = node.slice
        if isinstance(index, ast.Tuple):
            if len(index.elts) != 1:
                return node
            index = index.elts[0]
        param, at = self.param, index

        class At(ast.NodeTransformer):
            def visit_Name(self, n: ast.Name) -> ast.AST:
                return copy.deepcopy(at) if n.id == param else n

        self.count += 1
        return At().visit(copy.deepcopy(self.fill))


def rewrite_reads(sdfg: SDFG, name: str, param: str, fill: ast.AST) -> int:
    """Substitute every interstate read of ``name`` in ``sdfg`` and in the nested SDFGs it reaches whole."""
    count = 0
    for e in sdfg.all_interstate_edges():
        sub = SubstituteIota(name, param, fill)
        for lhs, rhs in list(e.data.assignments.items()):
            if name in rhs:
                e.data.assignments[lhs] = astutils.unparse(sub.visit(ast.parse(rhs, mode="eval").body))
        if name in e.data.condition.as_string:
            e.data.condition = CodeBlock(astutils.unparse(sub.visit(ast.parse(e.data.condition.as_string).body[0])))
        count += sub.count
    fill_symbols = {n.id for n in ast.walk(fill) if isinstance(n, ast.Name)} - {param}
    for state in sdfg.states():
        for node in state.nodes():
            if not isinstance(node, nodes.NestedSDFG):
                continue
            for e in state.in_edges(node):
                if not isinstance(e.src, nodes.AccessNode) or e.src.data != name or e.dst_conn is None:
                    continue
                inner_desc = node.sdfg.arrays.get(e.dst_conn)
                if inner_desc is None or len(inner_desc.shape) != 1 or e.data.subset.ranges[0][0] != 0:
                    continue
                if e.data.subset.num_elements() != inner_desc.total_size:
                    continue
                if any(str(node.symbol_mapping.get(s, s)) != s for s in fill_symbols):
                    continue
                if any(n.data == e.dst_conn and st.in_degree(n) for st in node.sdfg.states() for n in st.data_nodes()):
                    continue
                count += rewrite_reads(node.sdfg, e.dst_conn, param, fill)
    return count


@transformation.explicit_cf_compatible
class ForwardIotaReads(ppl.Pass):
    """Forwards the fill expression of every map-filled index transient into the interstate reads of it."""

    CATEGORY: str = "Simplification"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.InterstateEdges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & (ppl.Modifies.AccessNodes | ppl.Modifies.Memlets))

    def apply_pass(self, sdfg: SDFG, _: dict[str, Any]) -> int | None:
        count = 0
        for sd in sdfg.all_sdfgs_recursive():
            for name in sorted(sd.arrays):
                found = iota_fill(sd, name)
                if found is not None:
                    count += rewrite_reads(sd, name, *found)
        return count or None
