# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Fusion of a point-wise activation into the convolution node that produces its input.

A convolution followed by an element-wise ReLU map over its whole result costs a second pass over the activations
(and, with cuDNN/MIOpen, a third one for the bias). :class:`FuseConvolutionActivation` removes the map and sets the
:attr:`~dace.libraries.ml.Convolution.activation` of the node instead, which every implementation honours (the pure
expansion applies it when it writes the result; cuDNN and MIOpen use their fused convolution+bias+activation kernels
where they support the configuration). The map may write into a slice of a larger container (a concatenation): the
convolution then writes there directly.
"""

import ast
import copy

from dace import SDFG, SDFGState, nodes, symbolic
from dace.sdfg import utils as sdutil
from dace.transformation import transformation as xf


class FuseConvolutionActivation(xf.SingleStateTransformation):
    """
    Matches ``convolution -> transient -> map{tasklet}`` where the map applies a ReLU to every element of the
    transient (which nothing else reads) and folds the map into the convolution node. Applies to the node classes in
    :attr:`fusable` (:class:`dace.libraries.ml.Convolution` and the ATen wrapper of the TorchDynamo frontend).
    """

    #: Library node classes with an ``activation`` property that this transformation may set
    fusable: set[type] = set()

    convolution = xf.PatternNode(nodes.LibraryNode)
    access = xf.PatternNode(nodes.AccessNode)
    map_entry = xf.PatternNode(nodes.MapEntry)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.convolution, cls.access, cls.map_entry)]

    def can_be_applied(self, graph: SDFGState, expr_index: int, sdfg: SDFG, permissive: bool = False) -> bool:
        conv, access, entry = self.convolution, self.access, self.map_entry
        if type(conv) not in self.fusable or conv.activation != "none":
            return False
        desc = sdfg.arrays[access.data]
        # The pre-activation result is a transient that only the map reads
        if not desc.transient or graph.in_degree(access) != 1 or graph.out_degree(access) != 1:
            return False
        if (
            sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.AccessNode) and n.data == access.data)
            > 1
        ):
            return False
        conv_edge = graph.in_edges(access)[0]
        if conv_edge.data.wcr is not None or not _covers(conv_edge.data.subset, desc.shape):
            return False
        # A map over every element with a single ReLU tasklet, writing one container
        if graph.in_degree(entry) != 1:
            return False
        exit_node = graph.exit_node(entry)
        inner = list(graph.scope_subgraph(entry, include_entry=False, include_exit=False).nodes())
        if len(inner) != 1 or not isinstance(inner[0], nodes.Tasklet) or graph.out_degree(exit_node) != 1:
            return False
        tasklet = inner[0]
        if graph.in_degree(tasklet) != 1 or graph.out_degree(tasklet) != 1:
            return False
        read, write = graph.in_edges(tasklet)[0], graph.out_edges(tasklet)[0]
        outer = graph.out_edges(exit_node)[0]
        out = outer.dst
        if not isinstance(out, nodes.AccessNode) or sdfg.arrays[out.data].dtype != desc.dtype:
            return False
        if write.data.wcr is not None or read.data.data != access.data or write.data.data != out.data:
            return False
        params = [symbolic.pystr_to_symbolic(p) for p in entry.map.params]
        if len(params) != len(desc.shape) or not _is_full_range(entry.map.range, desc.shape):
            return False
        if not _is_index(read.data.subset, params, [0] * len(params)):
            return False
        # The map's output: the same elements, possibly offset into a larger container (concatenation)
        if len(outer.data.subset) != len(params) or not _covers(outer.data.subset, desc.shape):
            return False
        if not _is_index(write.data.subset, params, [rng[0] for rng in outer.data.subset]):
            return False
        return _is_relu(tasklet.code.as_string, read.dst_conn, write.src_conn)

    def apply(self, graph: SDFGState, sdfg: SDFG) -> None:
        conv, access, entry = self.convolution, self.access, self.map_entry
        exit_node = graph.exit_node(entry)
        conv_edge = graph.in_edges(access)[0]
        outer = graph.out_edges(exit_node)[0]
        memlet = copy.deepcopy(outer.data)
        removed = list(graph.scope_subgraph(entry).nodes()) + [access]
        graph.remove_nodes_from(removed)
        graph.add_edge(conv, conv_edge.src_conn, outer.dst, None, memlet)
        conv.activation = "relu"
        sdfg.remove_data(access.data, validate=False)


def fuse_activations(sdfg: SDFG) -> int:
    """Applies :class:`FuseConvolutionActivation` everywhere in ``sdfg``; returns the number of fused activations."""
    return sdfg.apply_transformations_repeated(FuseConvolutionActivation, validate=False)


def _covers(subset, shape) -> bool:
    """Whether ``subset`` has the extent of ``shape`` in every dimension."""
    if len(subset) != len(shape):
        return False
    return all(symbolic.equal(rng[1] - rng[0] + 1, size) is not False for rng, size in zip(subset, shape))


def _is_full_range(map_range, shape) -> bool:
    return all(
        symbolic.equal(rng[0], 0) is not False and symbolic.equal(rng[1] + 1, size) is not False and rng[2] == 1
        for rng, size in zip(map_range, shape)
    )


def _is_index(subset, params, offsets) -> bool:
    """Whether ``subset`` is the single element ``[params[i] + offsets[i]]``."""
    if len(subset) != len(params):
        return False
    for rng, param, offset in zip(subset, params, offsets):
        if symbolic.equal(rng[0], rng[1]) is False:
            return False
        if symbolic.equal(rng[0] - param, offset) is not True:
            return False
    return True


def _is_relu(code: str, in_conn: str, out_conn: str) -> bool:
    """Whether the tasklet computes ``out = relu(in)``: ``max(in, 0)`` or the NaN-propagating conditional form."""
    try:
        body = ast.parse(code).body
    except SyntaxError:
        return False
    if len(body) != 1 or not isinstance(body[0], ast.Assign) or len(body[0].targets) != 1:
        return False
    target = body[0].targets[0]
    if not isinstance(target, ast.Name) or target.id != out_conn:
        return False

    def strip_casts(node):  # ``dace.float32(x)`` and the like
        while isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords and _callee(node) not in _MAX:
            node = node.args[0]
        return node

    def is_input(node):
        node = strip_casts(node)
        return isinstance(node, ast.Name) and node.id == in_conn

    def is_zero(node):
        node = strip_casts(node)
        return isinstance(node, ast.Constant) and not isinstance(node.value, bool) and node.value == 0

    def is_cmp(node, op, rhs):
        return (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], op)
            and is_input(node.left)
            and rhs(node.comparators[0])
        )

    expr = strip_casts(body[0].value)
    if isinstance(expr, ast.Call) and _callee(expr) in _MAX and len(expr.args) == 2:
        a, b = expr.args
        return (is_input(a) and is_zero(b)) or (is_zero(a) and is_input(b))
    if isinstance(expr, ast.IfExp) and is_input(expr.body) and is_zero(expr.orelse):
        test = expr.test
        conditions = test.values if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or) else [test]
        positive = [c for c in conditions if is_cmp(c, ast.Gt, is_zero)]
        nan = [c for c in conditions if is_cmp(c, ast.NotEq, is_input)]
        return len(positive) == 1 and len(positive) + len(nan) == len(conditions)
    return False


_MAX = {"max", "maximum", "fmax"}


def _callee(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""
