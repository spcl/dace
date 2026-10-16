# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``UniqueNames``: no name of an SDFG tree shadows another where one Python function would hold both.

The Python backend emits an SDFG as one function, and Python has one namespace per function. A map parameter that an
enclosing map already binds, a tasklet connector or local that spells a container, symbol, iterator or module, and a
container or symbol of a nested SDFG that its parent already names would each clobber the outer binding. This pass
renames the inner one to ``<name>_<N>``. Loop iterators are made unique first by
:class:`~dace.transformation.passes.unique_loop_iterators.UniqueLoopIterators`.
"""

import ast
import builtins
import keyword
import re

from dace import SDFG, dtypes
from dace.libraries.tileops.nodes.tile_op import TileOp
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.replace import replace_dict
from dace.sdfg.sdfg import interstate_bound_names
from dace.sdfg.state import LoopRegion, SDFGState, StateSubgraphView
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.unique_loop_iterators import UniqueLoopIterators
from dace.transformation.transformation import explicit_cf_compatible

#: Names the generated module binds or imports, which no SDFG name may shadow.
RESERVED_NAMES = frozenset(
    {*dir(builtins), *keyword.kwlist, "np", "numpy", "cupy", "math", "dace", "tl", "triton", "ct", "T", "tilelang"}
)


def fresh_name(name: str, taken: set[str]) -> str:
    """The first ``<name>_<N>`` not in ``taken``, which it joins."""
    n = 1
    while f"{name}_{n}" in taken:
        n += 1
    new = f"{name}_{n}"
    taken.add(new)
    return new


def sdfg_names(sdfg: SDFG) -> set[str]:
    """The names ``sdfg`` binds for its whole body: containers, constants, symbols and loop iterators."""
    loops = {r.loop_variable for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable}
    return {*sdfg.arrays, *sdfg.constants_prop, *sdfg.symbols, *interstate_bound_names(sdfg), *loops}


def scope_bindings(entry: nodes.EntryNode) -> list[str]:
    """The names a scope entry binds: map parameters and dynamic range connectors."""
    params = entry.map.params if isinstance(entry, nodes.MapEntry) else [entry.consume.pe_index]
    return [*params, *(conn for conn in entry.in_connectors if not conn.startswith("IN_"))]


def enclosing_bindings(scope: dict, node: nodes.Node) -> set[str]:
    """The names the scopes around ``node`` bind."""
    names: set[str] = set()
    parent = scope[node]
    while parent is not None:
        names.update(scope_bindings(parent))
        parent = scope[parent]
    return names


def scope_depth(scope: dict, node: nodes.Node) -> int:
    """The number of scopes around ``node``."""
    depth, parent = 0, scope[node]
    while parent is not None:
        depth, parent = depth + 1, scope[parent]
    return depth


def tasklet_locals(code: str) -> set[str]:
    """The names a Python tasklet body assigns."""
    return {n.id for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}


class NameRenamer(ast.NodeTransformer):
    """Renames ``ast.Name`` ids by a mapping."""

    def __init__(self, repl: dict[str, str]):
        self.repl = repl

    def visit_Name(self, node: ast.Name) -> ast.Name:
        node.id = self.repl.get(node.id, node.id)
        return node


def rename_connector(state: SDFGState, node: nodes.CodeNode, old: str, new: str) -> None:
    """Rename the connector ``old`` of ``node`` and of the edges that attach to it."""
    for connectors, edges, attr in (
        (node.in_connectors, state.in_edges(node), "dst_conn"),
        (node.out_connectors, state.out_edges(node), "src_conn"),
    ):
        if old not in connectors:
            continue
        connectors[new] = connectors.pop(old)
        for edge in edges:
            if getattr(edge, attr) == old:
                setattr(edge, attr, new)


def rename_scope_params(state: SDFGState, taken: set[str]) -> int:
    """Rename every scope binding an enclosing scope or ``taken`` already holds, outermost scope first."""
    renamed = 0
    scope = state.scope_dict()
    entries = [n for n in state.nodes() if isinstance(n, (nodes.MapEntry, nodes.ConsumeEntry))]
    entries.sort(key=lambda entry: scope_depth(scope, entry))
    for entry in entries:
        shadowed = taken | enclosing_bindings(scope, entry)
        repl = {name: fresh_name(name, taken) for name in scope_bindings(entry) if name in shadowed}
        if not repl:
            continue
        for old, new in repl.items():
            if old in entry.in_connectors:
                rename_connector(state, entry, old, new)
        subgraph = state.scope_subgraph(entry, include_entry=True, include_exit=True)
        replace_dict(subgraph, repl)
        rename_strings(subgraph, repl)
        for nested in (n for n in subgraph.nodes() if isinstance(n, nodes.NestedSDFG)):
            rename_strings(nested.sdfg, {old: new for old, new in repl.items() if old in nested.sdfg.free_symbols})
        renamed += len(repl)
    return renamed


def rename_tasklet(state: SDFGState, scope: dict, tasklet: nodes.Tasklet, taken: set[str]) -> int:
    """Rename the connectors and locals of a Python tasklet that shadow a name bound around it."""
    if tasklet.language is not dtypes.Language.Python:
        return 0
    shadowed = taken | enclosing_bindings(scope, tasklet)
    names = {*tasklet.in_connectors, *tasklet.out_connectors, *tasklet_locals(tasklet.code.as_string)}
    repl = {name: fresh_name(name, taken | names) for name in sorted(names & shadowed)}
    if not repl:
        return 0
    for old, new in repl.items():
        rename_connector(state, tasklet, old, new)
    tree = NameRenamer(repl).visit(ast.parse(tasklet.code.as_string))
    tasklet.code = CodeBlock(ast.unparse(tree), dtypes.Language.Python)
    return len(repl)


def same_binding(state: SDFGState, node: nodes.NestedSDFG, name: str) -> bool:
    """Whether the nested name ``name`` binds the parent's ``name`` itself: a symbol mapped to it, or a connector of
    an edge that moves it. Such a name holds the parent's value, so it shadows nothing."""
    if name in node.symbol_mapping:
        return str(node.symbol_mapping[name]) == name
    edges = [*state.in_edges_by_connector(node, name), *state.out_edges_by_connector(node, name)]
    return bool(edges) and all(edge.data.data == name for edge in edges)


def rename_strings(graph: SDFG | StateSubgraphView, repl: dict[str, str]) -> None:
    """Rename inside the string properties of the tile nodes of ``graph``: ``replace`` renames a string only when it
    is the name itself, and a tile node spells expressions (``lane_bounds``, ``expr_a``) as strings."""
    names = re.compile(rf"\b({'|'.join(map(re.escape, repl))})\b")
    found = graph.all_nodes_recursive() if isinstance(graph, SDFG) else ((n, None) for n in graph.nodes())
    for node, _ in found:
        if not isinstance(node, TileOp):
            continue
        for prop in type(node).__properties__:
            value = getattr(node, prop)
            if isinstance(value, str):
                setattr(node, prop, names.sub(lambda m: repl[m.group(1)], value))
            elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                setattr(node, prop, [names.sub(lambda m: repl[m.group(1)], item) for item in value])


def rename_nested(state: SDFGState, node: nodes.NestedSDFG, taken: set[str]) -> int:
    """Rename the containers and symbols of a nested SDFG its parent already names, then its own body."""
    inner = node.sdfg
    inner_names = sdfg_names(inner)
    clashes = sorted(name for name in inner_names & taken if not same_binding(state, node, name))
    repl = {name: fresh_name(name, taken | inner_names) for name in clashes}
    if repl:
        inner.replace_dict(repl)
        rename_strings(inner, repl)
        for old, new in repl.items():
            rename_connector(state, node, old, new)
            if old in node.symbol_mapping:
                node.symbol_mapping[new] = node.symbol_mapping.pop(old)
    return len(repl) + rename_sdfg(inner, taken | sdfg_names(inner))


def rename_sdfg(sdfg: SDFG, taken: set[str]) -> int:
    """Make every name of ``sdfg`` and its nested SDFGs distinct from ``taken`` and from the scopes around it."""
    renamed = 0
    for state in sdfg.states():
        renamed += rename_scope_params(state, taken)
        scope = state.scope_dict()
        for node in state.nodes():
            if isinstance(node, nodes.Tasklet):
                renamed += rename_tasklet(state, scope, node, taken)
            elif isinstance(node, nodes.NestedSDFG):
                renamed += rename_nested(state, node, taken | enclosing_bindings(scope, node))
    return renamed


@explicit_cf_compatible
class UniqueNames(ppl.Pass):
    """Rename every name that would shadow another in the one namespace of the generated Python function."""

    CATEGORY: str = "Cleanup"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Symbols | ppl.Modifies.Descriptors | ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _) -> int | None:
        """:returns: The number of renamed names, or ``None`` if nothing shadowed."""
        loops = UniqueLoopIterators().apply_pass(sdfg, {})
        renamed = rename_sdfg(sdfg, set(RESERVED_NAMES) | sdfg_names(sdfg)) + len(loops or ())
        return renamed or None
