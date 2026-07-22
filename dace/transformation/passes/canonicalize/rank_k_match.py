# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Recognise a hand-written symmetric rank-k / rank-2k update loop nest.

Shared machinery for :class:`~dace.transformation.passes.canonicalize.loop_to_syrk.LoopToSyrk`
and :class:`~dace.transformation.passes.canonicalize.loop_to_syr2k.LoopToSyr2k`. Both
recognise the same skeleton -- the polybench / npbench slice-vectorized form::

    for i in range(N):                                  # outer LoopRegion
        C[i, :i + 1] *= beta[0]                         # beta-scale state
        for k in range(M):                              # inner LoopRegion
            C[i, :i + 1] += <rank-k body>               # accumulate state

and differ only in the body expression (``alpha*A[i,k]*A[:i+1,k]`` for syrk;
``A[:i+1,k]*alpha*B[i,k] + B[:i+1,k]*alpha*A[i,k]`` for syr2k).

WHY A SYMBOLIC MATCHER, NOT A STRUCTURAL ONE: the Python frontend lowers those two
slice statements into a *chain* of staged temporaries -- ``alpha[0]`` through a scalar
AccessNode, ``A[i,k]`` through another, their product through a third, then a map that
broadcasts it against the ``A[:i+1,k]`` slice, then a second map that adds the result
onto ``C``. The exact chain depends on operand order, on how many binary operators the
source spells, and on how much of it simplification folds away. Matching that shape
node-by-node would be long and brittle.

Instead this module *evaluates* the dataflow: :class:`StateValueResolver` walks back
from the state's ``C`` sink and returns a sympy expression for the value written to
``C[i, j]``, with every array read left as an opaque leaf symbol. The polybench syrk
body resolves to exactly ``C[i,j] + alpha[0]*A[i,k]*A[j,k]`` regardless of how the
frontend staged it, so the matcher just compares that expression against the one the
BLAS primitive is defined to compute. Any deviation -- a different coefficient, a
transposed operand, an extra term -- makes the comparison fail and the lift a clean
no-op.
"""
from typing import Dict, List, NamedTuple, Optional, Tuple

import sympy

from dace import SDFG, data as dt, subsets, symbolic
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation.passes.analysis import loop_analysis

# Guard against a cyclic / pathological dataflow walk (the resolver recurses through
# producer edges; a well-formed state bottoms out in a handful of steps).
MAX_RESOLVE_DEPTH = 64


class ArrayRead(NamedTuple):
    """One opaque array read appearing in a resolved expression."""
    array: str
    index: Tuple[object, ...]


class RankKMatch(NamedTuple):
    """Operands and orientation of a recognised rank-k / rank-2k nest."""
    c: str
    a: str
    b: Optional[str]  # None for syrk
    alpha: str
    beta: str
    uplo: str
    trans: str
    n: object
    k: object


class StateValueResolver:
    """Symbolically evaluate what one state writes to an array element.

    ``value_at(sink, index)`` returns a sympy expression for the value the state
    stores into ``sink`` at ``index``, with each array read that the state does not
    itself produce represented by a fresh leaf symbol (recorded in ``leaves``). Reads
    of the array being written resolve to leaves too -- a read AccessNode has no
    producer -- so an in-place ``C[...] += ...`` naturally yields ``C_prior + ...``.

    Raises ``ValueError`` on anything it cannot evaluate exactly; callers treat that
    as "no match".
    """

    def __init__(self, state: SDFGState):
        self.state = state
        self.leaves: Dict[sympy.Symbol, ArrayRead] = {}
        self.by_key: Dict[Tuple[str, Tuple[str, ...]], sympy.Symbol] = {}

    def leaf(self, array: str, index) -> sympy.Symbol:
        """A stable leaf symbol for the read ``array[index]`` (same read -> same symbol)."""
        key = (array, tuple(str(symbolic.simplify(i)) for i in index))
        existing = self.by_key.get(key)
        if existing is not None:
            return existing
        sym = sympy.Symbol(f"__rk_read{len(self.leaves)}")
        self.leaves[sym] = ArrayRead(array, tuple(index))
        self.by_key[key] = sym
        return sym

    def value_at(self, node: nodes.AccessNode, index, depth: int = 0) -> sympy.Basic:
        """Value stored into AccessNode ``node`` at ``index``."""
        if depth > MAX_RESOLVE_DEPTH:
            raise ValueError("rank-k resolve: dataflow too deep")
        producers = [e for e in self.state.in_edges(node) if e.data is not None and not e.data.is_empty()]
        if not producers:
            return self.leaf(node.data, index)  # a read node: the value on state entry
        if len(producers) != 1:
            raise ValueError(f"rank-k resolve: {node.data} has {len(producers)} producers")
        edge = producers[0]
        if isinstance(edge.src, nodes.MapExit):
            return self.through_map_exit(edge.src, edge.src_conn, index, depth)
        if isinstance(edge.src, nodes.Tasklet):
            return self.eval_tasklet(edge.src, {}, depth)
        if isinstance(edge.src, nodes.AccessNode):
            return self.value_at(edge.src, subset_indices(edge.data.subset), depth + 1)
        raise ValueError(f"rank-k resolve: unsupported producer {type(edge.src).__name__}")

    def through_map_exit(self, map_exit: nodes.MapExit, out_conn: str, index, depth: int) -> sympy.Basic:
        """Evaluate the tasklet inside ``map_exit``'s scope that produces ``index``,
        binding the map parameters from its write subset."""
        in_conn = "IN_" + out_conn[len("OUT_"):] if out_conn.startswith("OUT_") else out_conn
        for edge in self.state.in_edges(map_exit):
            if edge.dst_conn != in_conn:
                continue
            if not isinstance(edge.src, nodes.Tasklet):
                raise ValueError("rank-k resolve: map exit not fed by a tasklet")
            if edge.data.wcr is not None:
                raise ValueError("rank-k resolve: WCR write inside the body")
            entry = self.state.entry_node(edge.src)
            if entry is None:
                raise ValueError("rank-k resolve: tasklet outside a map scope")
            binding = unify(subset_indices(edge.data.subset), index, entry.map.params)
            return self.eval_tasklet(edge.src, binding, depth)
        raise ValueError("rank-k resolve: no producer into map exit")

    def eval_tasklet(self, tasklet: nodes.Tasklet, binding: Dict[str, object], depth: int) -> sympy.Basic:
        """Evaluate a single-assignment tasklet, resolving each input connector."""
        code = (tasklet.code.as_string or "").strip()
        if code.count("=") != 1:
            raise ValueError(f"rank-k resolve: not a single assignment: {code!r}")
        lhs, rhs = (s.strip() for s in code.split("=", 1))
        if lhs not in tasklet.out_connectors:
            raise ValueError("rank-k resolve: assignment target is not an out connector")
        expr = symbolic.pystr_to_symbolic(rhs)
        bind_syms = {sympy.Symbol(name): value for name, value in binding.items()}

        substitutions = {}
        for edge in self.state.in_edges(tasklet):
            if edge.dst_conn is None or edge.data is None or edge.data.is_empty():
                continue
            index = [symbolic.pystr_to_symbolic(str(i)).subs(bind_syms) for i in subset_indices(edge.data.subset)]
            if isinstance(edge.src, nodes.MapEntry):
                source = source_access(self.state, edge.src, edge.src_conn)
                if source is None:
                    raise ValueError("rank-k resolve: map entry connector has no source")
                value = self.value_at(source, index, depth + 1)
            elif isinstance(edge.src, nodes.AccessNode):
                value = self.value_at(edge.src, index, depth + 1)
            elif isinstance(edge.src, nodes.Tasklet):
                value = self.eval_tasklet(edge.src, binding, depth + 1)
            else:
                raise ValueError(f"rank-k resolve: unsupported input {type(edge.src).__name__}")
            substitutions[sympy.Symbol(edge.dst_conn)] = value
        return expr.subs(substitutions).subs(bind_syms)


def subset_indices(subset: subsets.Subset) -> List[object]:
    """The per-axis begin expression of ``subset`` (its element index when every axis
    is a single point; the slice base otherwise)."""
    return [symbolic.pystr_to_symbolic(str(begin)) for begin, _, _ in subset.ndrange()]


def unify(pattern: List[object], target: List[object], params: List[str]) -> Dict[str, object]:
    """Bind map ``params`` so that ``pattern == target`` elementwise. Non-parameter
    axes must already agree symbolically."""
    if len(pattern) != len(target):
        raise ValueError("rank-k resolve: index rank mismatch")
    binding: Dict[str, object] = {}
    for p, t in zip(pattern, target):
        name = str(p)
        if name in params:
            binding[name] = t
        elif symbolic.simplify(p - t) != 0:
            raise ValueError(f"rank-k resolve: index mismatch {p} vs {t}")
    return binding


def source_access(state: SDFGState, entry: nodes.MapEntry, out_conn: str) -> Optional[nodes.AccessNode]:
    """The AccessNode feeding ``entry``'s ``OUT_x`` connector from outside the scope."""
    in_conn = "IN_" + out_conn[len("OUT_"):] if out_conn.startswith("OUT_") else out_conn
    for edge in state.in_edges(entry):
        if edge.dst_conn == in_conn and isinstance(edge.src, nodes.AccessNode):
            return edge.src
    return None


def equals(a, b) -> bool:
    """Symbolic equality of two scalar expressions."""
    try:
        return bool(symbolic.simplify(symbolic.pystr_to_symbolic(str(a)) - symbolic.pystr_to_symbolic(str(b))) == 0)
    except Exception:
        return False


def expressions_equal(actual: sympy.Basic, expected: sympy.Basic) -> bool:
    """Whether two resolved value expressions are symbolically identical."""
    try:
        return bool(sympy.simplify(sympy.expand(actual - expected)) == 0)
    except Exception:
        return False


def unit_stride(loop: LoopRegion) -> bool:
    stride = loop_analysis.get_loop_stride(loop)
    try:
        return stride is not None and symbolic.simplify(stride) == 1
    except Exception:
        return False


def loop_extent(loop: LoopRegion) -> Optional[object]:
    """``end + 1`` of a ``0``-based unit-stride loop (its trip count), else ``None``."""
    if not loop.loop_variable or not unit_stride(loop):
        return None
    init = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    if init is None or end is None or not equals(init, 0):
        return None
    return symbolic.simplify(symbolic.pystr_to_symbolic(str(end)) + 1)


def single_body_state(region: ControlFlowRegion) -> Optional[SDFGState]:
    """The region's one non-empty state (empty connective states tolerated)."""
    blocks = list(region.nodes())
    if not all(isinstance(b, SDFGState) for b in blocks):
        return None
    non_empty = [b for b in blocks if b.nodes()]
    return non_empty[0] if len(non_empty) == 1 else None


def beta_and_inner_loop(outer: LoopRegion) -> Optional[Tuple[SDFGState, LoopRegion]]:
    """Split the outer loop body into its ``beta``-scale state and its inner
    ``k`` LoopRegion, requiring exactly one of each and that the scale runs FIRST
    (the accumulation must land on the already-scaled ``C``)."""
    scale: Optional[SDFGState] = None
    inner: Optional[LoopRegion] = None
    for block in outer.nodes():
        if isinstance(block, LoopRegion):
            if inner is not None:
                return None
            inner = block
        elif isinstance(block, SDFGState):
            if not block.nodes():
                continue  # empty connective state
            if scale is not None:
                return None
            scale = block
        else:
            return None
    if scale is None or inner is None:
        return None
    if not reaches(outer, scale, inner):
        return None
    return scale, inner


def reaches(region: ControlFlowRegion, src, dst) -> bool:
    """Whether ``dst`` is reachable from ``src`` along ``region``'s control flow."""
    seen, stack = {id(src)}, [src]
    while stack:
        node = stack.pop()
        for edge in region.out_edges(node):
            if edge.dst is dst:
                return True
            if id(edge.dst) not in seen:
                seen.add(id(edge.dst))
                stack.append(edge.dst)
    return False


def sink_node(state: SDFGState) -> Optional[nodes.AccessNode]:
    """The state's single sink AccessNode (out-degree 0, in-degree > 0), or ``None``
    if the state has zero or several. A frontend staging temporary is never a sink --
    it always feeds a consumer -- so this picks out the state's real output."""
    sinks = [
        n for n in state.nodes()
        if isinstance(n, nodes.AccessNode) and state.in_degree(n) > 0 and state.out_degree(n) == 0
    ]
    if len(sinks) != 1:
        return None
    return sinks[0]


def written_arrays(state: SDFGState) -> set:
    """Names of arrays the state writes."""
    return {n.data for n in state.nodes() if isinstance(n, nodes.AccessNode) and state.in_degree(n) > 0}


def nontransient_written(state: SDFGState, sdfg: SDFG) -> set:
    """Names of NON-transient arrays the state writes.

    The frontend stages every indexed read through a transient scalar
    (``beta[0]`` -> ``beta_index``, ``A[i,k]`` -> ``A_index``, ...), so a state that
    computes one output still writes several transients. Only the non-transient writes
    describe what the state means; the transients are checked separately, at the loop
    level, by :func:`internal_writes_contained`.
    """
    return {name for name in written_arrays(state) if not is_transient(sdfg, name)}


def is_transient(sdfg: SDFG, name: str) -> bool:
    desc = sdfg.arrays.get(name)
    return desc is not None and desc.transient


def internal_writes_contained(loop: LoopRegion, root: SDFG, c_array: str) -> bool:
    """Whether lifting ``loop`` away can only lose values nothing outside it can read.

    Every array the nest writes other than ``c_array`` must be a transient that no
    state OUTSIDE the nest touches -- i.e. the frontend's per-statement staging
    scratch, dead the moment the nest ends. A write to anything else (a second output,
    or a transient read later) means the loop does more than the rank-k update and the
    lift would silently drop it. Mirrors ``LoopToEinsum._live_outside``.
    """
    inside = {id(state) for state in loop.all_states()}
    written = {name for state in loop.all_states() for name in written_arrays(state)}
    written.discard(c_array)
    if not written:
        return True
    if any(not is_transient(root, name) for name in written):
        return False
    for state in root.all_states():
        if id(state) in inside:
            continue
        if any(dn.data in written for dn in state.data_nodes()):
            return False
    return True


def triangle_of(subset: subsets.Subset, row: str, n) -> Optional[str]:
    """``'L'`` if ``subset`` is the lower-triangle row slice ``[row, 0:row+1]``,
    ``'U'`` if it is the upper-triangle row slice ``[row, row:n]``, else ``None``."""
    if subset is None or len(subset) != 2:
        return None
    axes = list(subset.ndrange())
    (rb, re_, rs), (cb, ce, cs) = axes
    if not (equals(rb, row) and equals(re_, row) and equals(rs, 1)):
        return None
    if not equals(cs, 1):
        return None
    row_sym = symbolic.pystr_to_symbolic(row)
    if equals(cb, 0) and equals(ce, row_sym):
        return "L"
    if equals(cb, row_sym) and equals(ce, symbolic.pystr_to_symbolic(str(n)) - 1):
        return "U"
    return None


def sink_write_subset(state: SDFGState, sink: nodes.AccessNode) -> Optional[subsets.Subset]:
    """The subset of the single memlet writing ``sink``."""
    edges = [e for e in state.in_edges(sink) if e.data is not None and not e.data.is_empty()]
    if len(edges) != 1:
        return None
    return edges[0].data.subset


def classify_leaves(resolver: StateValueResolver, sdfg: SDFG, c_array: str, i: str, j: sympy.Symbol,
                    k: Optional[str]) -> Optional[dict]:
    """Bucket a resolved expression's leaves by role.

    :returns: a dict with ``'c'`` (the prior ``C[i,j]`` leaf), ``'coeffs'`` (leaves that
              read a single-element array, keyed by array name), ``'row'`` / ``'col'``
              (leaves reading a 2-D operand at ``[i,k]`` / ``[j,k]``, keyed by array
              name) and ``'trans'`` (``'N'`` when operands are read row-major
              ``[i,k]``, ``'T'`` when read ``[k,i]``); ``None`` if any leaf does not
              fit one of those roles.
    """
    i_sym = symbolic.pystr_to_symbolic(i)
    k_sym = symbolic.pystr_to_symbolic(k) if k is not None else None
    out = {"c": None, "coeffs": {}, "row": {}, "col": {}, "trans": None}
    for sym, read in resolver.leaves.items():
        desc = sdfg.arrays.get(read.array)
        if desc is None:
            return None
        if is_single_element(desc):
            if len(read.index) != 1 or not equals(read.index[0], 0):
                return None
            if read.array in out["coeffs"]:
                return None
            out["coeffs"][read.array] = sym
            continue
        if len(read.index) != 2:
            return None
        first, second = read.index
        if read.array == c_array:
            if out["c"] is not None or not (equals(first, i_sym) and equals(second, j)):
                return None
            out["c"] = sym
            continue
        if k_sym is None:
            return None
        # A 2-D operand read: [i,k]/[j,k] (trans 'N') or [k,i]/[k,j] (trans 'T').
        if equals(second, k_sym):
            orientation, outer = "N", first
        elif equals(first, k_sym):
            orientation, outer = "T", second
        else:
            return None
        if out["trans"] is not None and out["trans"] != orientation:
            return None
        out["trans"] = orientation
        bucket = "row" if equals(outer, i_sym) else ("col" if equals(outer, j) else None)
        if bucket is None or read.array in out[bucket]:
            return None
        out[bucket][read.array] = sym
    return out


def is_single_element(desc: dt.Data) -> bool:
    """A ``Scalar`` or a length-1 ``Array`` -- the shape a runtime coefficient has."""
    return isinstance(desc, dt.Scalar) or (isinstance(desc, dt.Array) and all(str(s) == "1" for s in desc.shape))


def match_beta_state(state: SDFGState, sdfg: SDFG, c_array: str, i: str, j: sympy.Symbol,
                     n) -> Optional[Tuple[str, str]]:
    """Match ``C[i, <triangle>] *= beta[0]``.

    :returns: ``(beta_array, uplo)``, or ``None``.
    """
    if nontransient_written(state, sdfg) != {c_array}:
        return None
    sink = sink_node(state)
    if sink is None or sink.data != c_array:
        return None
    subset = sink_write_subset(state, sink)
    uplo = triangle_of(subset, i, n)
    if uplo is None:
        return None
    resolver = StateValueResolver(state)
    try:
        value = resolver.value_at(sink, [symbolic.pystr_to_symbolic(i), j])
    except ValueError:
        return None
    roles = classify_leaves(resolver, sdfg, c_array, i, j, None)
    if roles is None or roles["c"] is None or len(roles["coeffs"]) != 1 or roles["row"] or roles["col"]:
        return None
    beta_array, beta_sym = next(iter(roles["coeffs"].items()))
    if not expressions_equal(value, roles["c"] * beta_sym):
        return None
    return beta_array, uplo


def resolve_accumulate(state: SDFGState, sdfg: SDFG, c_array: str, i: str, j: sympy.Symbol, k: str,
                       n) -> Optional[Tuple[sympy.Basic, dict, str]]:
    """Resolve the inner ``k``-loop body's write to ``C[i, <triangle>]``.

    :returns: ``(value_expression, leaf_roles, uplo)``, or ``None`` if the state is not
              a triangular in-place update of ``C`` alone.
    """
    if nontransient_written(state, sdfg) != {c_array}:
        return None
    sink = sink_node(state)
    if sink is None or sink.data != c_array:
        return None
    subset = sink_write_subset(state, sink)
    uplo = triangle_of(subset, i, n)
    if uplo is None:
        return None
    resolver = StateValueResolver(state)
    try:
        value = resolver.value_at(sink, [symbolic.pystr_to_symbolic(i), j])
    except ValueError:
        return None
    roles = classify_leaves(resolver, sdfg, c_array, i, j, k)
    if roles is None or roles["c"] is None or len(roles["coeffs"]) != 1:
        return None
    return value, roles, uplo


def loop_invariant(loop: LoopRegion, names) -> bool:
    """Whether none of ``names`` is written anywhere inside ``loop``.

    The lift hoists the operands and coefficients out of the nest and hands them to the
    BLAS node as whole-array inputs, so each must be a value the nest only READS. A name
    the nest also produces (a precomputed temporary staged into an ``N x K`` transient,
    say) would be classified as an operand while its producer -- the nest -- is spliced
    away, leaving the node reading uninitialised memory.
    """
    written = {name for state in loop.all_states() for name in written_arrays(state)}
    return not (set(names) & written)


def operand_shape_ok(sdfg: SDFG, array: str, trans: str, n, k) -> bool:
    """Whether ``array`` is the ``N x K`` (``trans='N'``) / ``K x N`` (``'T'``) operand."""
    desc = sdfg.arrays.get(array)
    if desc is None or len(desc.shape) != 2:
        return False
    want = (n, k) if trans == "N" else (k, n)
    return equals(desc.shape[0], want[0]) and equals(desc.shape[1], want[1])


def square_output_ok(sdfg: SDFG, array: str, n) -> bool:
    desc = sdfg.arrays.get(array)
    if desc is None or len(desc.shape) != 2:
        return False
    return equals(desc.shape[0], n) and equals(desc.shape[1], n)


def replace_loop_with_state(parent: ControlFlowRegion, loop: LoopRegion, label: str) -> SDFGState:
    """Splice ``loop`` out of ``parent``, replacing it with a fresh (returned) state
    that inherits the loop's in/out interstate edges. Mirrors ``LoopToEinsum``'s CFG
    surgery."""
    import dace
    was_start = parent.start_block is loop
    in_edges = list(parent.in_edges(loop))
    out_edges = list(parent.out_edges(loop))
    state = parent.add_state(label, is_start_block=was_start)
    for edge in in_edges:
        parent.add_edge(edge.src, state, edge.data)
    for edge in out_edges:
        condition = edge.data.condition.as_string if edge.data.condition is not None else "1"
        parent.add_edge(state, edge.dst,
                        dace.InterstateEdge(condition=condition, assignments=dict(edge.data.assignments or {})))
    parent.remove_node(loop)
    return state


def outer_loop_candidates(sdfg: SDFG) -> List[Tuple[ControlFlowRegion, LoopRegion]]:
    """Every ``(parent, loop)`` whose ``loop`` could be a rank-k nest's outer loop:
    a unit-stride 0-based loop that directly contains another LoopRegion."""
    found: List[Tuple[ControlFlowRegion, LoopRegion]] = []
    for sd in sdfg.all_sdfgs_recursive():
        for region in sd.all_control_flow_regions(recursive=True):
            for block in region.nodes():
                if not isinstance(block, LoopRegion) or not block.loop_variable:
                    continue
                if any(isinstance(b, LoopRegion) for b in block.nodes()):
                    found.append((region, block))
    return found


def root_sdfg_of(region: ControlFlowRegion) -> SDFG:
    root = region
    while not isinstance(root, SDFG):
        root = root.parent_graph
    return root
