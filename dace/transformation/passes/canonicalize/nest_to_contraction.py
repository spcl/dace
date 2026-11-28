# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift a perfect loop/map nest around one contraction library node to a single ``Einsum``."""

import string
from dataclasses import dataclass
from typing import Any

from dace import SDFG, data, dtypes, subsets, symbolic
from dace.frontend.operations import detect_reduction_type
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.canonicalize.loop_iteration_value import perfect_nest, replace_loop_with_state
from dace.transformation.passes.canonicalize.loop_to_einsum import _is_sum_tasklet
from dace.transformation.passes.canonicalize.loop_to_transpose import _is_copy_tasklet
from dace.transformation.passes.canonicalize.zero_prior import ZeroStore, zero_before_block
from dace.transformation.transformation import explicit_cf_compatible


@dataclass(frozen=True, slots=True)
class Axis:
    """One nest variable running ``begin .. end`` (inclusive), step 1."""

    param: str
    begin: Any
    end: Any
    parallel: bool


@dataclass(frozen=True, slots=True)
class Point:
    """A dimension indexed by one nest-invariant expression."""

    index: Any


@dataclass(slots=True)
class Operand:
    """An access of ``array``: per dimension the nest variable indexing it, a ``Point``, or ``None`` (full range)."""

    array: str
    dims: list[str | Point | None]
    root: nodes.AccessNode


@dataclass(slots=True)
class Contraction:
    """What the lifted ``Einsum`` needs."""

    einsum_str: str
    alpha: Any
    beta: Any
    operands: list[Operand]
    output: Operand
    axes: dict[str, Axis]
    #: The zeros stored before the nest that the overwriting einsum makes dead.
    dead_zeros: ZeroStore | None = None


def same(a: Any, b: Any) -> bool:
    return not symbolic.inequal_symbols(symbolic.pystr_to_symbolic(a), symbolic.pystr_to_symbolic(b))


def loop_axis(loop: LoopRegion) -> Axis | None:
    if loop.inverted or not loop.loop_variable:
        return None
    begin = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if begin is None or end is None or stride is None or not symbolic.equal_valued(1, stride):
        return None
    return Axis(loop.loop_variable, begin, end, False)


def map_axes(entry: nodes.MapEntry) -> list[Axis] | None:
    axes = []
    for param, (begin, end, step) in zip(entry.map.params, entry.map.range):
        if not symbolic.equal_valued(1, step):
            return None
        axes.append(Axis(param, begin, end, True))
    return axes


def contraction_types() -> tuple[type, ...]:
    from dace.libraries.blas import nodes as blas  # Avoid import loop
    from dace.libraries.linalg.nodes.tensordot import TensorDot  # Avoid import loop

    return (blas.Einsum, blas.MatMul, blas.Gemm, blas.BatchedMatMul, blas.Gemv, blas.Dot, TensorDot)


def matmul_terms(ra: int, rb: int, trans_a: bool, trans_b: bool) -> tuple[str, str, str] | None:
    """numpy ``matmul`` index strings for operand ranks ``ra``, ``rb``; ``None`` for a broadcast it cannot name."""
    a = {1: "k", 2: "ik", 3: "bik"}.get(ra)
    b = {1: "k", 2: "kj", 3: "bkj"}.get(rb)
    if a is None or b is None or (ra == 3 and rb == 1) or (ra == 1 and rb == 3):
        return None
    if trans_a and ra >= 2:
        a = a[:-2] + a[-1] + a[-2]
    if trans_b and rb >= 2:
        b = b[:-2] + b[-1] + b[-2]
    out = ("b" if 3 in (ra, rb) else "") + ("i" if ra >= 2 else "") + ("j" if rb >= 2 else "")
    return a, b, out


def body_contraction(node: nodes.LibraryNode, ranks: dict[str, int]) -> tuple[str, list[str], str, Any, Any] | None:
    """``(einsum string, operand connectors in term order, output connector, alpha, beta)`` of ``node``."""
    from dace.libraries.blas import nodes as blas  # Avoid import loop
    from dace.libraries.linalg.nodes.tensordot import TensorDot  # Avoid import loop

    if isinstance(node, blas.Einsum):
        if "->" not in node.einsum_str or "." in node.einsum_str or len(node.out_connectors) != 1:
            return None
        conns = sorted(node.in_connectors)
        if any(c in ("_alpha", "_beta") for c in conns):
            return None
        return node.einsum_str.replace(" ", ""), conns, next(iter(node.out_connectors)), node.alpha, node.beta
    if isinstance(node, (blas.MatMul, blas.BatchedMatMul, blas.Gemm)):
        if isinstance(node, blas.Gemm) and (node.alpha_input or node.beta_input):
            return None
        terms = matmul_terms(ranks.get("_a"), ranks.get("_b"), node.transA, node.transB)
        if terms is None or (isinstance(node, blas.Gemm) and (ranks["_a"], ranks["_b"]) != (2, 2)):
            return None
        return f"{terms[0]},{terms[1]}->{terms[2]}", ["_a", "_b"], "_c", node.alpha, node.beta
    if isinstance(node, blas.Gemv):
        if (ranks.get("_A"), ranks.get("_x")) != (2, 1):
            return None
        return ("ji,j->i" if node.transA else "ij,j->i"), ["_A", "_x"], "_y", node.alpha, node.beta
    if isinstance(node, blas.Dot):
        if node.conjugate or (ranks.get("_x"), ranks.get("_y")) != (1, 1):
            return None
        return "k,k->", ["_x", "_y"], "_result", 1, 0
    if isinstance(node, TensorDot):
        if "_left_tensor" not in ranks or "_right_tensor" not in ranks:
            return None
        letters = iter(string.ascii_letters)
        left = [next(letters) for _ in range(ranks["_left_tensor"])]
        right = [
            left[node.left_axes[node.right_axes.index(i)]] if i in node.right_axes else next(letters)
            for i in range(ranks["_right_tensor"])
        ]
        out = [c for i, c in enumerate(left) if i not in node.left_axes]
        out += [c for i, c in enumerate(right) if i not in node.right_axes]
        if node.permutation:
            out = [out[i] for i in node.permutation]
        terms = f"{''.join(left)},{''.join(right)}->{''.join(out)}"
        return terms, ["_left_tensor", "_right_tensor"], "_out_tensor", 1, 0
    return None


@dataclass(slots=True)
class BodyMatcher:
    """Reads one nest body."""

    sdfg: SDFG
    state: SDFGState
    axes: dict[str, Axis]
    state_counts: dict[str, int]
    claimed: dict[nodes.Node, None]
    wcr_sum: bool = False

    def outside_node(self, edge: MultiConnectorEdge[Memlet], incoming: bool) -> nodes.AccessNode | None:
        path = self.state.memlet_path(edge)
        end = path[0].src if incoming else path[-1].dst
        return end if isinstance(end, nodes.AccessNode) else None

    def is_local(self, node: nodes.Node) -> bool:
        """A view or body-private transient, which the lift deletes."""
        if not isinstance(node, nodes.AccessNode):
            return False
        desc = self.sdfg.arrays[node.data]
        return (
            (isinstance(desc, data.View) or desc.transient)
            and self.state_counts.get(node.data, 0) == 1
            and sum(1 for n in self.state.data_nodes() if n.data == node.data) == 1
        )

    def classify(self, array: str, subset: subsets.Subset) -> list[str | Point | None] | None:
        desc = self.sdfg.arrays[array]
        if not isinstance(subset, subsets.Range) or len(subset) != len(desc.shape):
            return None
        dims: list[str | Point | None] = []
        for (begin, end, step), extent in zip(subset, desc.shape):
            names = {str(s) for s in symbolic.pystr_to_symbolic(begin).free_symbols} | {
                str(s) for s in symbolic.pystr_to_symbolic(end).free_symbols
            }
            if str(begin) in self.axes and str(end) == str(begin):
                if str(begin) in dims:
                    return None
                dims.append(str(begin))
            elif not names & self.axes.keys() and str(begin) == str(end):
                dims.append(Point(begin))
            elif (
                not names & self.axes.keys()
                and symbolic.equal_valued(1, step)
                and symbolic.equal_valued(0, begin)
                and same(end, extent - 1)
                and not symbolic.equal_valued(1, extent)
            ):
                dims.append(None)
            else:
                return None
        return dims

    def local_extents(self, operand: Operand) -> list[Any]:
        shape = self.sdfg.arrays[operand.array].shape
        return [s for s, d in zip(shape, operand.dims) if d is None]

    def resolve(self, edge: MultiConnectorEdge[Memlet], incoming: bool) -> Operand | None:
        """The global access behind a libnode or accumulation-map connector, through at most one view or copy."""
        if edge.data.is_empty() or edge.data.wcr is not None:
            return None
        far = edge.src if incoming else edge.dst
        hop_shape = None
        if self.is_local(far):
            desc = self.sdfg.arrays[far.data]
            if not isinstance(edge.data.subset, subsets.Range) or edge.data.data != far.data:
                return None
            if any(
                not symbolic.equal_valued(0, b) or not same(e, s - 1)
                for (b, e, _), s in zip(edge.data.subset, desc.shape)
            ):
                return None
            hops = self.state.in_edges(far) if incoming else self.state.out_edges(far)
            if len(hops) != 1 or (self.state.out_degree(far) if incoming else self.state.in_degree(far)) != 1:
                return None
            self.claimed[far] = None
            hop_shape = [s for s in desc.shape if not symbolic.equal_valued(1, s)]
            edge = hops[0]
            if edge.data.is_empty() or edge.data.wcr is not None:
                return None
        root = self.outside_node(edge, incoming)
        if root is None or self.is_local(root):
            return None
        memlet = edge.data
        subset = memlet.subset if memlet.data == root.data else memlet.other_subset
        if memlet.data != root.data and hop_shape is None:
            return None
        dims = self.classify(root.data, subset) if subset is not None else None
        if dims is None:
            return None
        operand = Operand(root.data, dims, root)
        if hop_shape is not None:
            extents = self.local_extents(operand)
            if len(extents) != len(hop_shape) or not all(same(a, b) for a, b in zip(extents, hop_shape)):
                return None
        self.claimed[root] = None
        return operand

    def accumulation(self, edge: MultiConnectorEdge[Memlet]) -> Operand | None:
        """``X[idx] = X[idx] + T[...]`` or ``X[idx] (+)= T[...]`` over the whole of the local ``T`` ``edge`` writes."""
        temp = edge.dst
        if not self.is_local(temp) or isinstance(self.sdfg.arrays[temp.data], data.View):
            return None
        outs = self.state.out_edges(temp)
        if len(outs) != 1 or self.state.in_degree(temp) != 1:
            return None
        if isinstance(outs[0].dst, nodes.Tasklet):
            return self.element_accumulation(temp, outs[0].dst)
        if not isinstance(outs[0].dst, nodes.MapEntry):
            return None
        return self.map_accumulation(temp, outs[0].dst)

    def map_accumulation(self, temp: nodes.AccessNode, entry: nodes.MapEntry) -> Operand | None:
        """``X[idx] = X[idx] + T[...]`` or ``X[idx] (+)= T[...]`` by a map over the whole of the local ``T``."""
        exit_node = self.state.exit_node(entry)
        temp_shape = self.sdfg.arrays[temp.data].shape
        summed = self.summing_tasklet(temp, entry, exit_node, temp_shape)
        if summed is None:
            return None
        tasklet, write, other_reads = summed
        params = entry.map.params
        if [str(b) for b, _, _ in write.subset if str(b) in params] != params or any(
            str(b) != str(e) or (str(b) not in params and {str(f) for f in b.free_symbols} & set(params))
            for b, e, _ in write.subset
        ):
            return None
        exit_edges = [e for e in self.state.out_edges(exit_node) if not e.data.is_empty()]
        if len(exit_edges) != 1:
            return None
        root = self.outside_node(exit_edges[0], incoming=False)
        if root is None or root.data != write.data or self.is_local(root):
            return None
        shape = self.sdfg.arrays[root.data].shape
        region = subsets.Range(
            [(0, n - 1, 1) if str(b) in params else (b, b, 1) for (b, _, _), n in zip(write.subset, shape)]
        )
        dims = self.classify(root.data, region)
        if dims is None:
            return None
        extents = [n for n, d in zip(shape, dims) if d is None]
        if len(extents) != len(temp_shape) or not all(same(a, b) for a, b in zip(extents, temp_shape)):
            return None
        for read in other_reads:
            source = self.outside_node(read, incoming=True)
            if source is None or source.data != root.data:
                return None
            self.claimed[source] = None
        for n in (temp, entry, exit_node, tasklet, root):
            self.claimed[n] = None
        return Operand(root.data, dims, root)

    def summing_tasklet(
        self, temp: nodes.AccessNode, entry: nodes.MapEntry, exit_node: nodes.MapExit, temp_shape: list
    ) -> tuple[nodes.Tasklet, Memlet, list[MultiConnectorEdge[Memlet]]] | None:
        """The map's one tasklet, adding ``T`` at the map's point to one other read or by a WCR sum, with its write
        and its reads of anything but ``T``."""
        scope = self.state.scope_children()[entry]
        tasklets = [n for n in scope if n is not exit_node]
        if len(tasklets) != 1 or not isinstance(tasklets[0], nodes.Tasklet):
            return None
        tasklet = tasklets[0]
        ranges = dict(zip(entry.map.params, entry.map.range))
        if len(ranges) != len(temp_shape) or any(
            not symbolic.equal_valued(0, b) or not symbolic.equal_valued(1, s) or not same(e, n - 1)
            for (b, e, s), n in zip(ranges.values(), temp_shape)
        ):
            return None
        out_edges = self.state.out_edges(tasklet)
        if len(out_edges) != 1:
            return None
        write = out_edges[0].data
        temp_reads = [e for e in self.state.in_edges(tasklet) if e.data.data == temp.data]
        other_reads = [e for e in self.state.in_edges(tasklet) if e.data.data != temp.data]
        if len(temp_reads) != 1 or [str(b) for b, _, _ in temp_reads[0].data.subset] != entry.map.params:
            return None
        if write.wcr is not None:
            if (
                other_reads
                or not _is_copy_tasklet(tasklet)
                or detect_reduction_type(write.wcr) != dtypes.ReductionType.Sum
            ):
                return None
            self.wcr_sum = True
        elif len(other_reads) != 1 or not _is_sum_tasklet(tasklet) or other_reads[0].data.subset != write.subset:
            return None
        if write.data == temp.data:
            return None
        return tasklet, write, other_reads

    def element_accumulation(self, temp: nodes.AccessNode, tasklet: nodes.Tasklet) -> Operand | None:
        """``X[idx] = X[idx] + T`` or ``X[idx] (+)= T`` for a one-element local ``T``, written to ``X`` directly or
        through one one-element local copy."""
        if any(not symbolic.equal_valued(1, n) for n in self.sdfg.arrays[temp.data].shape):
            return None
        writes = self.state.out_edges(tasklet)
        reads = [e for e in self.state.in_edges(tasklet) if e.src is not temp]
        if len(writes) != 1 or self.state.in_degree(tasklet) != len(reads) + 1:
            return None
        write = writes[0]
        if write.data.wcr is not None:
            if (
                reads
                or not _is_copy_tasklet(tasklet)
                or detect_reduction_type(write.data.wcr) != dtypes.ReductionType.Sum
            ):
                return None
            self.wcr_sum = True
        elif len(reads) != 1 or not _is_sum_tasklet(tasklet):
            return None
        hop = write.dst
        if write.data.wcr is None and self.is_local(hop):
            hops = self.state.out_edges(hop)
            if (
                self.state.in_degree(hop) != 1
                or len(hops) != 1
                or hops[0].data.wcr is not None
                or any(not symbolic.equal_valued(1, n) for n in self.sdfg.arrays[hop.data].shape)
            ):
                return None
            self.claimed[hop] = None
            write = hops[0]
        root = self.outside_node(write, incoming=False)
        if root is None or self.is_local(root):
            return None
        subset = write.data.subset if write.data.data == root.data else write.data.other_subset
        dims = self.classify(root.data, subset) if subset is not None else None
        if dims is None or None in dims:
            return None
        for read in reads:
            source = self.outside_node(read, incoming=True)
            if source is None or source.data != root.data or read.data.data != root.data or read.data.subset != subset:
                return None
            self.claimed[source] = None
        for n in (temp, tasklet, root):
            self.claimed[n] = None
        return Operand(root.data, dims, root)


def descend(state: SDFGState, entry: nodes.MapEntry | None) -> list[nodes.MapEntry] | None:
    """The map chain from ``entry`` (or the top level) down to the scope holding the library node."""
    children = state.scope_children()
    chain = [] if entry is None else [entry]
    while True:
        level = [n for n in children[chain[-1] if chain else None] if not isinstance(n, nodes.MapExit)]
        if any(isinstance(n, nodes.LibraryNode) for n in level):
            return chain
        inner = [n for n in level if isinstance(n, nodes.MapEntry)]
        others = [n for n in level if not isinstance(n, (nodes.MapEntry, nodes.AccessNode))]
        if len(inner) != 1 or others or (chain and len(level) != 1):
            return None
        chain.append(inner[0])


def widened(sdfg: SDFG, operand: Operand, axes: dict[str, Axis]) -> subsets.Range:
    shape = sdfg.arrays[operand.array].shape
    return subsets.Range(
        [
            (0, n - 1, 1)
            if d is None
            else (d.index, d.index, 1)
            if isinstance(d, Point)
            else (axes[d].begin, axes[d].end, 1)
            for d, n in zip(operand.dims, shape)
        ]
    )


def checked_axes(loops: list[LoopRegion], chain: list[nodes.MapEntry]) -> dict[str, Axis] | None:
    """The nest's axes by parameter: rectangular, unit step, each parameter once."""
    axes_list = [loop_axis(loop) for loop in loops]
    for entry in chain:
        more = map_axes(entry)
        if more is None:
            return None
        axes_list.extend(more)
    if not axes_list or any(a is None for a in axes_list):
        return None
    axes = {a.param: a for a in axes_list}
    if len(axes) != len(axes_list):
        return None
    for axis in axes_list:
        bounds = symbolic.pystr_to_symbolic(axis.begin).free_symbols | symbolic.pystr_to_symbolic(axis.end).free_symbols
        if {str(s) for s in bounds} & axes.keys():
            return None
    return axes


def read_operands(
    matcher: BodyMatcher, node: nodes.LibraryNode
) -> tuple[dict[str, Operand], Operand, bool, MultiConnectorEdge[Memlet]] | None:
    """The library node's input operands by connector, its output operand, whether the body accumulates into the
    output, and the node's output edge."""
    state = matcher.state
    inputs: dict[str, Operand] = {}
    for edge in state.in_edges(node):
        operand = matcher.resolve(edge, incoming=True)
        if operand is None:
            return None
        inputs[edge.dst_conn] = operand
    out_edges = state.out_edges(node)
    if len(out_edges) != 1:
        return None
    output = matcher.resolve(out_edges[0], incoming=False)
    accumulates = output is None
    if accumulates:
        output = matcher.accumulation(out_edges[0])
        if output is None:
            return None
    return inputs, output, accumulates, out_edges[0]


def body_is_claimed(
    state: SDFGState, loops: list[LoopRegion], chain: list[nodes.MapEntry], level: list[nodes.Node], claimed: dict
) -> bool:
    """Every node of the body is part of the contraction, and no ordering edge touches the nest."""
    if any(n not in claimed for n in level if not isinstance(n, nodes.MapExit)):
        return False
    if (
        loops
        and chain
        and any(
            n not in claimed and n is not chain[0] and n is not state.exit_node(chain[0])
            for n in state.scope_children()[None]
        )
    ):
        return False
    scope_edges = state.scope_subgraph(chain[0]).edges() if chain else state.edges()
    if any(e.data.is_empty() for e in scope_edges):
        return False
    return not (
        chain
        and any(e.data.is_empty() for e in [*state.in_edges(chain[0]), *state.out_edges(state.exit_node(chain[0]))])
    )


def local_einsum(
    node: nodes.LibraryNode,
    inputs: dict[str, Operand],
    output: Operand,
    accumulates: bool,
    out_edge: MultiConnectorEdge[Memlet],
    axes: dict[str, Axis],
) -> tuple[str, list[str], str, list[Operand], Any, Any] | None:
    """The body node's einsum, its input terms and output term, its operands in term order, alpha and beta (1 when
    the body accumulates), or ``None`` when the operands do not fit the node."""
    ranks = {conn: sum(1 for d in op.dims if d is None) for conn, op in inputs.items()}
    body = body_contraction(node, ranks)
    if body is None:
        return None
    einsum_str, conns, out_conn, alpha, beta = body
    if out_conn != out_edge.src_conn:
        return None
    for conn, operand in inputs.items():
        if conn in conns:
            continue
        # The prior output a nonzero-beta Gemm (``_c``), MatMul (``_cin``) or Gemv (``_y``) reads: the output region.
        if conn not in ("_c", "_cin", "_y") or operand.array != output.array or operand.dims != output.dims:
            return None
    if accumulates:
        if not symbolic.equal_valued(0, beta):
            return None
        beta = 1
    for coefficient in (alpha, beta):
        if {str(s) for s in symbolic.pystr_to_symbolic(coefficient).free_symbols} & axes.keys():
            return None
    terms, out_term = einsum_str.split("->")
    terms = terms.split(",")
    if len(terms) != len(conns) or any(len(t) != ranks[c] for t, c in zip(terms, conns)):
        return None
    if len(out_term) != sum(1 for d in output.dims if d is None):
        return None
    operands = [inputs[c] for c in conns]
    if any(op.array == output.array for op in operands):
        return None
    return einsum_str, terms, out_term, operands, alpha, beta


def match_nest(
    sdfg: SDFG, state: SDFGState, loops: list[LoopRegion], chain: list[nodes.MapEntry], state_counts: dict[str, int]
) -> Contraction | None:
    """The contraction a nest computes, or ``None`` when it breaks the module contract."""
    axes = checked_axes(loops, chain)
    if axes is None:
        return None
    innermost = chain[-1] if chain else None
    level = [
        n for n in state.scope_children()[innermost] if not (innermost is not None and n is state.exit_node(innermost))
    ]
    libs = [n for n in level if isinstance(n, nodes.LibraryNode)]
    if len(libs) != 1 or not isinstance(libs[0], contraction_types()):
        return None
    node = libs[0]
    matcher = BodyMatcher(sdfg, state, axes, state_counts, {node: None})
    found = read_operands(matcher, node)
    if found is None:
        return None
    inputs, output, accumulates, out_edge = found
    local = local_einsum(node, inputs, output, accumulates, out_edge, axes)
    if local is None:
        return None
    einsum_str, terms, out_term, operands, alpha, beta = local
    if not body_is_claimed(state, loops, chain, level, matcher.claimed):
        return None
    lifted = nest_einsum(einsum_str, terms, out_term, operands, output, axes, matcher.wcr_sum)
    if lifted is None:
        return None
    einsum, contracted = lifted
    if contracted and not symbolic.equal_valued(1, beta):
        return None
    # Summing onto zeros stored just before the nest (``C[:] = 0``) is overwriting; covered, they are dead.
    zeros = zero_before_block(loops[0], output.array) if loops and symbolic.equal_valued(1, beta) else None
    region = None if zeros is None else widened(sdfg, output, axes)
    if zeros is None or not zeros.subset.covers_precise(region, sdfg.facts()):
        return Contraction(einsum, alpha, beta, operands, output, axes)
    dead = None if zeros.observed or not region.covers_precise(zeros.subset, sdfg.facts()) else zeros
    return Contraction(einsum, alpha, 0, operands, output, axes, dead)


def nest_einsum(
    einsum_str: str,
    terms: list[str],
    out_term: str,
    operands: list[Operand],
    output: Operand,
    axes: dict[str, Axis],
    wcr_sum: bool,
) -> tuple[str, bool] | None:
    """The body's einsum with a letter per nest axis, and whether an axis is contracted."""
    unused = iter(c for c in string.ascii_letters if c not in einsum_str)
    letters = {param: next(unused, None) for param in axes}
    if None in letters.values():
        return None

    def extend(operand: Operand, local: str) -> str:
        chars = iter(local)
        return "".join(next(chars) if d is None else letters[d] for d in operand.dims if not isinstance(d, Point))

    full_terms = [extend(op, t) for op, t in zip(operands, terms)]
    full_out = extend(output, out_term)
    contracted = False
    for param, axis in axes.items():
        letter = letters[param]
        if not any(letter in t for t in full_terms):
            return None
        if letter not in full_out:
            if axis.parallel and not wcr_sum:
                return None
            contracted = True
    return f"{','.join(full_terms)}->{full_out}", contracted


def add_einsum(sdfg: SDFG, state: SDFGState, label: str, contraction: Contraction, reuse: bool) -> nodes.LibraryNode:
    from dace.libraries.blas.nodes.einsum import Einsum  # Avoid import loop

    axes = contraction.axes
    node = Einsum(label)
    node.einsum_str = contraction.einsum_str
    node.alpha = contraction.alpha
    node.beta = contraction.beta
    state.add_node(node)
    reads: dict[str, nodes.AccessNode] = {}
    for i, operand in enumerate(contraction.operands):
        conn = f"_ein{i:02d}"
        node.add_in_connector(conn)
        if not reuse and operand.array not in reads:
            reads[operand.array] = state.add_read(operand.array)
        src = operand.root if reuse else reads[operand.array]
        state.add_edge(src, None, node, conn, Memlet(data=operand.array, subset=widened(sdfg, operand, axes)))
    output = contraction.output
    node.add_out_connector("_out")
    dst = output.root if reuse else state.add_write(output.array)
    state.add_edge(node, "_out", dst, None, Memlet(data=output.array, subset=widened(sdfg, output, axes)))
    return node


def state_counts_of(sdfg: SDFG) -> dict[str, int]:
    counts: dict[str, int] = {}
    for state in sdfg.states():
        for name in dict.fromkeys(n.data for n in state.data_nodes()):
            counts[name] = counts.get(name, 0) + 1
    return counts


@explicit_cf_compatible
class NestToContraction(ppl.Pass):
    """Replace a perfect loop/map nest around one contraction library node by one ``Einsum`` (module docstring)."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.CFG)

    def apply_pass(self, sdfg: SDFG, _: dict[str, Any]) -> int | None:
        kinds = contraction_types()
        count = 0
        for sd in sdfg.all_sdfgs_recursive():
            if not any(isinstance(n, kinds) for state in sd.states() for n in state.nodes()):
                continue
            count += self.lift_loops(sd) + self.lift_maps(sd)
        return count or None

    def lift_loops(self, sdfg: SDFG) -> int:
        candidates = [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion)]
        gone: dict[LoopRegion, None] = {}
        counts = state_counts_of(sdfg)
        count = 0
        for loop in candidates:
            if loop in gone:
                continue
            found = perfect_nest(loop)
            if found is None:
                continue
            loops, state = found
            chain = descend(state, None)
            if chain is None:
                continue
            contraction = match_nest(sdfg, state, loops, chain, counts)
            if contraction is None:
                continue
            for region in loop.all_control_flow_regions():
                gone[region] = None
            new_state = replace_loop_with_state(loop.parent_graph, loop, loop.label + "_contraction")
            add_einsum(sdfg, new_state, loop.label + "_contraction", contraction, reuse=False)
            if contraction.dead_zeros is not None:
                contraction.dead_zeros.remove()
            counts = state_counts_of(sdfg)
            count += 1
        return count

    def lift_maps(self, sdfg: SDFG) -> int:
        kinds = contraction_types()
        count = 0
        counts = state_counts_of(sdfg)
        for state in list(sdfg.states()):
            if not any(isinstance(n, kinds) for n in state.nodes()):
                continue
            for entry in [n for n in state.scope_children()[None] if isinstance(n, nodes.MapEntry)]:
                chain = descend(state, entry)
                if chain is None:
                    continue
                contraction = match_nest(sdfg, state, [], chain, counts)
                if contraction is None:
                    continue
                roots = [op.root for op in contraction.operands] + [contraction.output.root]
                sources = dict.fromkeys(e.src for e in state.in_edges(entry))
                outside = [*sources, *(e.dst for e in state.out_edges(state.exit_node(entry)))]
                state.remove_nodes_from(state.scope_subgraph(entry, include_entry=True, include_exit=True).nodes())
                einsum = add_einsum(sdfg, state, entry.map.label + "_contraction", contraction, reuse=True)
                state.remove_nodes_from(
                    [
                        n
                        for n in dict.fromkeys(outside)
                        if n not in roots and n in state.nodes() and state.degree(n) == 0
                    ]
                )
                # A source the einsum does not read (the prior value of an accumulated ``X``) still orders the
                # nest after its producer: beta=1 reads ``X`` in place through the output edge.
                for n in sources:
                    if n not in roots and n in state.nodes():
                        state.add_nedge(n, einsum, Memlet())
                counts = state_counts_of(sdfg)
                count += 1
        return count


__all__ = ["NestToContraction"]
