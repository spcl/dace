# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Rewrites Python tasklet bodies to access arrays directly instead of through copy-in / copy-out connector temporaries,
for the readable CPU generator. A connector (``__in1``) becomes a subscript on the array (``A[i, j]``) at the lower
bound of its memlet subset. Connectors and edges stay in place, so dataflow, scheduling and allocation are unchanged;
the generator tells which connectors were inlined from the rewritten body.
"""
import ast
from typing import Any, TypeVar

from dace import data as dt
from dace import dtypes
from dace.properties import CodeBlock
from dace.sdfg import SDFG, SDFGState, nodes
from dace.sdfg.graph import MultiConnectorEdge
from dace.transformation import pass_pipeline as ppl

Access = tuple[str, list[str]]
T = TypeVar('T')


def inlinable_connector(sdfg: SDFG, state: SDFGState, edge: MultiConnectorEdge, is_output: bool) -> dt.Data | None:
    """
    Returns the descriptor a connector edge accesses if the connector may be replaced by a direct access, else None.
    Only a plain array or scalar in memory qualifies: not a code-to-code register, stream, reference, structure,
    container array, SDFG constant, or a WCR or dynamic access, whose lowering needs the connector.
    """
    memlet = edge.data
    if not (edge.src_conn if is_output else edge.dst_conn) or memlet.data not in sdfg.arrays:
        return None
    path = state.memlet_path(edge)
    if not isinstance(path[-1].dst if is_output else path[0].src, nodes.AccessNode):
        return None
    desc = sdfg.arrays[memlet.data]
    if not isinstance(desc, (dt.Array, dt.Scalar)) or isinstance(desc, (dt.Reference, dt.ContainerArray)):
        return None
    if memlet.wcr is not None or memlet.dynamic or memlet.subset is None or memlet.data in sdfg.constants:
        return None
    return desc


def resolve_inout(in_access: dict[str, T], out_access: dict[str, T], tasklet: nodes.Tasklet) -> dict[str, T]:
    """Merges input and output accesses by connector name. A name on both sides stands for one identifier in the
    body, so it is kept only if both sides agree."""
    inout = set(tasklet.in_connectors) & set(tasklet.out_connectors)
    merged: dict[str, T] = {}
    for name in in_access.keys() | out_access.keys():
        if name not in inout:
            merged[name] = in_access.get(name, out_access.get(name))
        elif name in in_access and name in out_access and in_access[name] == out_access[name]:
            merged[name] = in_access[name]
    return merged


class InlineTaskletConnectors(ppl.Pass):
    """ Rewrites eligible tasklet connectors into direct array accesses. """

    CATEGORY = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> set[str] | None:
        """:return: The labels of the rewritten tasklets, or None if there are none."""
        inlined: set[str] = set()
        for node, state in sdfg.all_nodes_recursive():
            if not isinstance(node, nodes.Tasklet) or not isinstance(state, SDFGState):
                continue
            if node.language == dtypes.Language.Python and self.inline_tasklet(state.sdfg, state, node):
                inlined.add(node.label)
        return inlined or None

    def element_access(self, sdfg: SDFG, state: SDFGState, edge: MultiConnectorEdge, is_output: bool) -> Access | None:
        """The array and per-dimension indices (subset starts) of a single-element connector access."""
        desc = inlinable_connector(sdfg, state, edge, is_output)
        if desc is None or edge.data.subset.num_elements() != 1:
            return None
        return edge.data.data, [str(begin) for begin, _, _ in edge.data.subset.ranges]

    def inline_tasklet(self, sdfg: SDFG, state: SDFGState, node: nodes.Tasklet) -> bool:
        accesses: dict[str, dict[str, Access]] = {'in': {}, 'out': {}}
        for edge in state.in_edges(node):
            access = self.element_access(sdfg, state, edge, False)
            if access is not None:
                accesses['in'][edge.dst_conn] = access
        for edge in state.out_edges(node):
            access = self.element_access(sdfg, state, edge, True)
            if access is not None:
                accesses['out'][edge.src_conn] = access
        merged = resolve_inout(accesses['in'], accesses['out'], node)
        if not merged:
            return False

        inliner = ConnectorInliner(merged)
        tree = inliner.visit(ast.parse(node.code.as_string))
        if not inliner.inlined:
            return False
        node.code = CodeBlock(ast.unparse(ast.fix_missing_locations(tree)), node.language)
        node.ignored_symbols = set(node.ignored_symbols) | {merged[conn][0] for conn in inliner.inlined}
        return True


class ConnectorInliner(ast.NodeTransformer):
    """ Replaces connector names with direct ``data[indices]`` subscripts. """

    def __init__(self, accesses: dict[str, Access]):
        self.accesses = accesses
        self.inlined: set[str] = set()

    def replace(self, name: str, node: ast.AST) -> ast.AST:
        data, indices = self.accesses[name]
        self.inlined.add(name)
        elements = [ast.parse(index, mode='eval').body for index in indices]
        subscript = elements[0] if len(elements) == 1 else ast.Tuple(elts=elements, ctx=ast.Load())
        access = ast.Subscript(value=ast.Name(id=data, ctx=ast.Load()), slice=subscript, ctx=ast.Load())
        return ast.copy_location(access, node)

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        # A pointer connector to a scalar is subscripted (``conn[0]``); the whole subscript is the access
        if isinstance(node.value, ast.Name) and node.value.id in self.accesses:
            return self.replace(node.value.id, node)
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        return self.replace(node.id, node) if node.id in self.accesses else node
