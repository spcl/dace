# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Stop a sequential map over padded slots at the count that masks them.

A vectorized spec walks every padded slot and masks the dead ones (examinimd's neighbour rows)::

    valid = slot < neigh_counts[i]
    j = neigh_list[i, slot] if valid else 0
    fpair = expr(j) if valid and rsq < cut else 0.0
    f[i, :] += fpair * (x[i, :] - x[j, :])          # WCR sum over slot

The bound is read from data, so this is a CPU specialization: canonical form keeps the rectangular map a GPU tile or a
vectorizer specializes from. The body is interpreted under ``slot < count == False``. It must write nothing that leaves
the map but WCR sums, and each must add ``z``: a literal zero or ``0.0 * v``, with ``v`` one value across masked slots.
``z`` is ``+-0`` or NaN, as ``v`` may be non-finite, so ``a + z + z == a + z``: the first masked slot stays and speaks
for the rest. What runs is a subset of the original iterations in their order, plus one read of the count at an index the
mask reads. A dropped iteration repeats the kept one's flags; its slot-dependent work is dead, and the compile flags
keep no such flags (``-fno-trapping-math``). Anything the interpretation cannot speak for refuses.
"""

import ast
import collections
import copy
import functools
from typing import Any

import sympy

from dace import SDFG, data, dtypes, symbolic
from dace.frontend.operations import detect_reduction_type
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.cpu_specialization.range_privatize_scatter_reductions import is_pure_call

#: Body values on masked slots, besides known numbers. VARIES may differ between masked slots, SAME is one unknown value
#: across them, and ABSORBED is ``0.0 * v`` of a SAME ``v``.
VARIES, SAME, ABSORBED = "varies", "same", "absorbed"


class Refusal(Exception):
    """The body holds something the interpretation cannot speak for."""


def cast_operand(node: ast.expr) -> ast.expr:
    """``x`` of a ``dace.int64(x)`` or ``int64_t(x)``, or ``node``."""
    if not isinstance(node, ast.Call) or len(node.args) != 1:
        return node
    typeclass = ast.unparse(node.func).removeprefix("dace.").removesuffix("_t")
    return node.args[0] if typeclass in dtypes.TYPECLASS_BY_NAME else node


def statement(node: nodes.Node) -> tuple[str, ast.expr] | None:
    """Target and value of a Python tasklet holding one assignment to a name."""
    if not isinstance(node, nodes.Tasklet) or node.code.language != dtypes.Language.Python:
        return None
    code = node.code.code
    if not isinstance(code, list) or len(code) != 1 or not isinstance(code[0], ast.Assign):
        return None
    targets = code[0].targets
    return (targets[0].id, code[0].value) if len(targets) == 1 and isinstance(targets[0], ast.Name) else None


def masked_value(node: ast.expr, slot: str, inputs: dict[str, Any], mask: ast.expr | None) -> Any:
    """What ``node`` evaluates to on masked slots, where ``mask`` is False."""

    def value(operand: ast.expr) -> Any:
        return masked_value(operand, slot, inputs, mask)

    node = cast_operand(node)
    if node is mask:
        return False
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, (bool, int, float)) else SAME
    if isinstance(node, ast.Name):
        return VARIES if node.id == slot else inputs.get(node.id, SAME)
    if isinstance(node, ast.IfExp) and isinstance(test := value(node.test), (bool, int, float)):
        return value(node.body if test else node.orelse)
    if isinstance(node, ast.Call) and (not is_pure_call(node.func) or node.keywords):
        raise Refusal(ast.unparse(node.func))
    if not isinstance(
        node, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp, ast.IfExp, ast.Call, ast.Subscript, ast.Tuple)
    ):
        raise Refusal(type(node).__name__)
    operands = node.args if isinstance(node, ast.Call) else ast.iter_child_nodes(node)
    values = [value(operand) for operand in operands if isinstance(operand, ast.expr)]
    op = getattr(node, "op", None)
    if isinstance(op, (ast.And, ast.BitAnd)) and False in values:
        return False
    if VARIES in values:
        return VARIES
    return ABSORBED if isinstance(op, ast.Mult) and (ABSORBED in values or 0 in values) else SAME


def index_polynomial(node: ast.expr, param: str) -> Any:
    """``node``, adding and multiplying ``param`` and integer literals, as a polynomial; None otherwise."""
    node = cast_operand(node)
    if isinstance(node, ast.Name) and node.id == param:
        return sympy.Symbol(param)
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mult)):
        left, right = index_polynomial(node.left, param), index_polynomial(node.right, param)
        if left is not None and right is not None:
            return left + right if isinstance(node.op, ast.Add) else left * right
    return None


class SDFGFacts:
    """What analyzing the masks of one SDFG reads, each computed once. Valid until the first apply."""

    def __init__(self, sdfg: SDFG) -> None:
        self.sdfg = sdfg
        self.sums: dict[str, bool] = {}

    @functools.cached_property
    def accesses(self) -> dict[str, list[tuple[SDFGState, nodes.AccessNode]]]:
        found = collections.defaultdict(list)
        for state in self.sdfg.states():
            for node in state.data_nodes():
                found[node.data].append((state, node))
        return found

    @functools.cached_property
    def shared(self) -> set[str]:
        """Transients of several states or read by control flow."""
        shared = set(self.sdfg.shared_transients(check_toplevel=False))
        for region in self.sdfg.all_control_flow_regions():
            shared |= set(region.used_symbols(all_symbols=True, with_contents=False))
        return shared

    def is_sum(self, wcr: str) -> bool:
        if wcr not in self.sums:
            self.sums[wcr] = detect_reduction_type(wcr) == dtypes.ReductionType.Sum
        return self.sums[wcr]

    def is_index_table(self, name: str) -> bool:
        """Whether transient ``name`` is written only by a map assigning ``name[i] = i``, as ``np.arange`` is."""
        writes = [(state, edge) for state, node in self.accesses[name] for edge in state.in_edges(node)]
        if len(writes) != 1 or not self.sdfg.arrays[name].transient:
            return False
        state, edge = writes[0]
        if not isinstance(edge.src, nodes.MapExit) or len(edge.src.map.params) != 1 or state.in_degree(edge.src) != 1:
            return False
        param, write = edge.src.map.params[0], state.in_edges(edge.src)[0]
        parsed = statement(write.src)
        if parsed is None or write.src.in_connectors or str(write.data.subset) != param:
            return False
        polynomial = index_polynomial(parsed[1], param)
        return polynomial is not None and sympy.expand(polynomial) == sympy.Symbol(param)


def compared_names(tasklet: nodes.Tasklet, code: ast.expr) -> tuple[str, str] | None:
    """``(s, c)`` when ``code`` is ``s < c`` or ``c > s`` for a connector ``c``."""
    if not isinstance(code, ast.Compare) or len(code.ops) != 1 or not isinstance(code.ops[0], (ast.Lt, ast.Gt)):
        return None
    left, right = cast_operand(code.left), cast_operand(code.comparators[0])
    if isinstance(code.ops[0], ast.Gt):
        left, right = right, left
    if not isinstance(left, ast.Name) or not isinstance(right, ast.Name) or right.id not in tasklet.in_connectors:
        return None
    return left.id, right.id


def reads_slot(state: SDFGState, facts: SDFGFacts, tasklet: nodes.Tasklet, name: str, slot: str) -> bool:
    """Whether ``name`` is ``slot``, directly or read from an index table."""
    if name == slot:
        return True
    read = [e.data for e in state.in_edges(tasklet) if e.dst_conn == name and not e.data.is_empty()]
    return bool(read) and str(read[0].subset) == slot and facts.is_index_table(read[0].data)


class MaskedBody:
    """Interprets the body of ``entry``'s map on masked slots, collecting what it adds to writes leaving the map."""

    def __init__(self, facts: SDFGFacts, state: SDFGState, entry: nodes.MapEntry, mask: nodes.Tasklet) -> None:
        self.facts, self.sdfg, self.state, self.mask = facts, facts.sdfg, state, mask
        self.slot = entry.map.params[0]
        self.scope = state.scope_subgraph(entry, include_entry=False, include_exit=False)
        self.members = set(self.scope.nodes())
        self.locals = {n.data for n in self.members if isinstance(n, nodes.AccessNode)}
        self.written = self.locals | {e.data.data for e in state.in_edges(state.exit_node(entry))}
        self.values: dict[str, Any] = {}
        self.leaving: list[Any] = []

    def read(self, edge) -> Any:
        root = self.state.memlet_path(edge)[0].src
        if root in self.members:
            if not isinstance(root, nodes.AccessNode) or root.data not in self.values:
                raise Refusal(f"read before written: {root}")
            return self.values[root.data]
        if edge.data.data in self.written:
            raise Refusal(f"reads what it writes: {edge.data.data}")
        return VARIES if self.slot in edge.data.subset.free_symbols else SAME

    def write(self, edge, value: Any) -> None:
        path = self.state.memlet_path(edge)
        sink = path[-1].dst
        if sink not in self.members:
            if any(e.data.wcr is None or not self.facts.is_sum(e.data.wcr) for e in path):
                raise Refusal(f"not a sum: {edge.data}")
            self.leaving.append(value)
        elif edge.data.wcr is not None or sink.data in self.values:
            # Accumulated, or assigned more than once: known to be one value at most
            self.values[sink.data] = VARIES if VARIES in (value, self.values.get(sink.data)) else SAME
        else:
            self.values[sink.data] = value

    def tasklet(self, node: nodes.Tasklet) -> None:
        parsed = statement(node)
        if parsed is None or node.has_side_effects(self.sdfg) or node.code_global.as_string:
            raise Refusal(node.label)
        target, code = parsed
        inputs = {e.dst_conn: self.read(e) for e in self.state.in_edges(node) if not e.data.is_empty()}
        value = masked_value(code, self.slot, inputs, code if node is self.mask else None)
        for edge in self.state.out_edges(node):
            if not edge.data.is_empty():
                if edge.src_conn != target:
                    raise Refusal(node.label)
                self.write(edge, value)

    def interpret(self) -> None:
        for node in sdutil.dfs_topological_sort(self.scope):
            if isinstance(node, nodes.Tasklet):
                self.tasklet(node)
            elif isinstance(node, nodes.AccessNode):
                desc = self.sdfg.arrays[node.data]
                if not desc.transient or isinstance(desc, (data.Stream, data.View)):
                    raise Refusal(node.data)
                for edge in self.state.out_edges(node):
                    if not edge.data.is_empty() and isinstance(edge.dst, (nodes.AccessNode, nodes.MapExit)):
                        self.write(edge, self.values.get(node.data, VARIES))
            elif isinstance(node, nodes.MapEntry):
                if self.slot in node.map.range.free_symbols or sdutil.has_dynamic_map_inputs(self.state, node):
                    raise Refusal(node.label)
            elif not isinstance(node, nodes.MapExit):
                raise Refusal(type(node).__name__)

    def neutral(self) -> bool:
        return bool(self.leaving) and all(value is ABSORBED or value == 0 for value in self.leaving)


def count_read(state: SDFGState, entry: nodes.MapEntry, mask: nodes.Tasklet, count: str) -> tuple[Any, Memlet] | None:
    """The edge that brings the count into ``entry``'s map and its read, when the read can move up to the map."""
    leaf = next(e for e in state.in_edges(mask) if e.dst_conn == count)
    if leaf.data.is_empty() or leaf.data.subset.num_elements() != 1:
        return None
    if state.sdfg.arrays[leaf.data.data].dtype not in dtypes.INTEGER_TYPES:
        return None
    path = state.memlet_path(leaf)
    entering = next((i for i, edge in enumerate(path) if edge.dst is entry), None)
    used = leaf.data.subset.free_symbols
    if entering is None or used & {entry.map.params[0]}:
        return None
    # A map between ``entry`` and the mask: its iterator is not defined at ``entry``
    if any(isinstance(e.dst, nodes.MapEntry) and used & set(e.dst.map.params) for e in path[entering + 1 :]):
        return None
    return path[entering], leaf.data


def analyze(
    facts: SDFGFacts, state: SDFGState, mask: nodes.Tasklet, entry: nodes.MapEntry, count: str
) -> tuple[SDFGState, nodes.MapEntry, Any, Memlet] | None:
    """What :func:`apply` needs to stop ``entry``'s map after the first slot ``mask`` kills, if that is sound."""
    step = entry.map.range[0][2]
    read = None if step != 1 or sdutil.has_dynamic_map_inputs(state, entry) else count_read(state, entry, mask, count)
    if read is None:
        return None
    outer, memlet = read
    body = MaskedBody(facts, state, entry, mask)
    try:
        body.interpret()
    except Refusal:
        return None
    if memlet.data in body.written or not body.neutral() or body.locals & facts.shared:
        return None
    if any(node not in body.members for name in body.locals for _, node in facts.accesses[name]):
        return None
    return state, entry, outer, memlet


def apply(sdfg: SDFG, state: SDFGState, entry: nodes.MapEntry, outer: Any, memlet: Memlet) -> None:
    begin, end, step = entry.map.range[0]
    dtype = sdfg.arrays[memlet.data].dtype
    name = sdfg.find_new_symbol(f"__count_{memlet.data}")
    entry.map.range[0] = (begin, sympy.Min(end, sympy.Max(symbolic.symbol(name, dtype), begin)), step)
    entry.add_in_connector(name, dtype)
    state.add_edge(
        outer.src, outer.src_conn, entry, name, Memlet(data=memlet.data, subset=copy.deepcopy(memlet.subset))
    )


def masks_in(state: SDFGState, facts: SDFGFacts) -> list[tuple[nodes.Tasklet, nodes.MapEntry, str]]:
    """The ``slot < count`` tasklets of ``state``, with the map that owns ``slot`` and the count's connector."""
    compares = [
        (node, names)
        for node in state.nodes()
        if (parsed := statement(node)) is not None and (names := compared_names(node, parsed[1])) is not None
    ]
    scopes = state.scope_dict() if compares else {}
    found = []
    for node, (slot_name, count) in compares:
        entry = scopes[node]
        while entry is not None:
            if len(entry.map.params) == 1 and reads_slot(state, facts, node, slot_name, entry.map.params[0]):
                found.append((node, entry, count))
                break
            entry = scopes[entry]
    return found


@transformation.explicit_cf_compatible
class BoundMaskedMaps(ppl.Pass):
    """Stop a map over padded slots at the count that masks them; see the module docstring."""

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Memlets | ppl.Modifies.Scopes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.Scopes)

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        """:return: the number of maps bounded, or None if there is none."""
        bounded = 0
        for nested in sdfg.all_sdfgs_recursive():
            facts = SDFGFacts(nested)
            plans = {}
            for state in nested.states():
                for mask, entry, count in masks_in(state, facts):
                    if entry not in plans and (plan := analyze(facts, state, mask, entry, count)) is not None:
                        plans[entry] = plan
            for plan in plans.values():
                apply(nested, *plan)
            bounded += len(plans)
        return bounded or None
