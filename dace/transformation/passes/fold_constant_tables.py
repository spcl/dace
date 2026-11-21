# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Fold literal-filled tables into SDFG constants."""

import ast
from typing import Any

import numpy as np

from dace import data, properties, symbolic
from dace.ordered import OrderedSet
from dace.sdfg import SDFG, nodes
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation

#: Per table: the state that fills it, and the fill tasklets with the index and value each writes.
Fill = tuple[SDFGState, dict[nodes.Tasklet, tuple[int, object]]]


@properties.make_properties
@transformation.explicit_cf_compatible
class FoldConstantTables(ppl.Pass):
    """Replace each table filled once with literals (CloudSC's ``iphase[0:5] = 1, 2, 1, 2, 0``) by an SDFG constant,
    removing the fill. Host code and every kernel then read the same constant array, which needs no placement.

    Only a transient 1-D array qualifies whose every element is written exactly once, by an input-less tasklet
    assigning a literal, in one state, and that nothing else writes. A loop around that state rewrites the same values.
    """

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.Nodes)

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> OrderedSet[str] | None:
        """:return: the names of the folded tables, or None if there is none."""
        folded = literal_fills(sdfg)
        for name, (state, writes) in folded.items():
            desc = sdfg.arrays[name]
            values = np.zeros(int(desc.total_size), dtype=desc.dtype.type)
            for index, value in writes.values():
                values[index] = value
            sdfg.add_constant(name, values, desc)
            for tasklet in writes:
                targets = [edge.dst for edge in state.out_edges(tasklet)]
                state.remove_node(tasklet)
                state.remove_nodes_from([target for target in targets if state.degree(target) == 0])
        return OrderedSet(folded) or None


def read_constant_tables(sdfg: SDFG) -> int:
    """Write each literal read of a 1-D constant on an interstate edge as its value, so that a symbol taken from a table
    is a number: lulesh's face corners index the arrays they select, and a symbolic index proves nothing disjoint.
    :return: the number of assignments rewritten."""
    rewritten = 0
    for nsdfg in sdfg.all_sdfgs_recursive():
        tables = {name for name, value in nsdfg.constants.items() if np.ndim(value) == 1}
        if not tables:
            continue
        for edge in nsdfg.all_interstate_edges():
            for name, value in edge.data.assignments.items():
                if any(table in value for table in tables):
                    parsed = ast.parse(value, mode="eval")
                    written = ast.unparse(parsed)
                    read = ast.unparse(ConstantReads(nsdfg, tables).visit(parsed))
                    if read != written:
                        edge.data.assignments[name] = read
                        rewritten += 1
    return rewritten


class ConstantReads(ast.NodeTransformer):
    """Replace ``table[literal]`` by the element, for ``tables`` among ``sdfg``'s constants."""

    def __init__(self, sdfg: SDFG, tables: set[str]) -> None:
        self.sdfg = sdfg
        self.tables = tables

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        if not isinstance(node.value, ast.Name) or node.value.id not in self.tables:
            return node
        if not isinstance(node.slice, ast.Constant) or not isinstance(node.slice.value, int):
            return node
        return ast.Constant(self.sdfg.constants[node.value.id][node.slice.value].item())


def literal_fills(sdfg: SDFG) -> dict[str, Fill]:
    """Transient 1-D tables whose every element is written once, by a tasklet assigning a literal, in one state, and
    that nothing else writes."""
    fills: dict[str, Fill] = {}
    refused: OrderedSet[str] = OrderedSet(sdfg.constants)
    for state in sdfg.states():
        for node in state.data_nodes():
            desc = sdfg.arrays[node.data]
            if type(desc) is not data.Array or not desc.transient or len(desc.shape) != 1:
                continue
            for edge in state.in_edges(node):
                write = literal_write(state, edge)
                owner = fills.setdefault(node.data, (state, {}))[0]
                if write is None or owner is not state:
                    refused.add(node.data)
                else:
                    fills[node.data][1][edge.src] = write
    return {
        name: fill
        for name, fill in fills.items()
        if name not in refused
        and not symbolic.issymbolic(sdfg.arrays[name].total_size)
        and sorted(index for index, _ in fill[1].values()) == list(range(int(sdfg.arrays[name].total_size)))
    }


def literal_write(state: SDFGState, edge) -> tuple[int, object] | None:
    """``(index, value)`` when ``edge`` comes from an input-less tasklet assigning a literal to one element."""
    tasklet = edge.src
    if not isinstance(tasklet, nodes.Tasklet) or state.in_degree(tasklet) or state.out_degree(tasklet) != 1:
        return None
    if edge.data.subset is None or edge.data.subset.num_elements() != 1:
        return None
    index = edge.data.subset.min_element()[0]
    if symbolic.issymbolic(index):
        return None
    try:
        (statement,) = ast.parse(tasklet.code.as_string.strip()).body
        if not isinstance(statement, ast.Assign):
            return None
        value = ast.literal_eval(statement.value)
    except (SyntaxError, ValueError, TypeError):
        return None
    return int(index), value
