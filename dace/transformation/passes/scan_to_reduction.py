# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Rewrite a forward prefix-sum loop as an elementwise step plus a sequential scan.

A loop whose body is a single-array forward prefix scan, e.g.::

    for i in range(start, end):
        arr[i + 1] = arr[i] + g(other_inputs, i)

carries a true read-after-write dependence between iterations (``arr[i+1]``
needs ``arr[i]``), so it cannot be turned into a ``Map`` by ``LoopToMap``. The
elementwise per-iteration delta ``g(other_inputs, i)`` is, however, perfectly
parallel: only the running sum has to stay sequential. This pass splits the
loop into two so the parallel half is exposed::

    for i: delta[i] = g(other_inputs, i)          # MAP-able
    for i: arr[i + 1] = arr[i] + delta[i]         # still sequential

The decomposition is value-preserving for any associative combiner; for v1
only ``+`` is supported. Because it adds a transient (the ``delta`` buffer)
and the second stage still runs sequentially, the rewrite is not always a
win, so the pass is **opt-in**: a caller invokes it explicitly, and it is
deliberately not wired into the default ``ParallelizePipeline`` or
canonicalize pipeline.

Why this is useful even though stage 2 stays sequential: the cloudsc
``pfcqlng[jk+1, jl] = pfcqlng[jk, jl] + flux[jk, jl]`` pattern -- the
motivating real case -- has a non-trivial ``g`` (here ``flux``) computed
upstream, so the cumulative half is only the addition. After this rewrite,
downstream ``LoopToMap`` can parallelize the elementwise loop while the tiny
prefix-sum tail keeps the semantics.

.. todo::

   * Stage 2 (the prefix sum) is left as a sequential loop. For long scans,
     a parallel-prefix algorithm (Blelloch / Hillis-Steele) would let stage 2
     also run as a ``Map``; not implemented for v1.
   * Only ``+`` is recognised as the associative combiner. ``*``, ``min``,
     ``max`` are equally valid prefix-scan operators and would slot into the
     same shape.
"""
import ast
import copy
from typing import Any, Dict, List, Optional, Tuple

from dace import SDFG, data, dtypes, properties, subsets, symbolic
from dace import memlet as mm
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis

#: Per-iteration delta buffer name prefix; ``find_new_name`` disambiguates. The
#: pass also uses this prefix to recognise its own already-emitted stage-2
#: prefix-sum loops on re-application (cheap, serialization-safe idempotence).
_DELTA_PREFIX = '_scan_delta_'


@properties.make_properties
@xf.explicit_cf_compatible
class ScanToReduction(ppl.Pass):
    """Split a forward prefix-scan loop into an elementwise stage and a sequential prefix sum.

    See the module docstring for the pattern and the rationale. The pass is
    opt-in and does nothing on a loop that does not fit the canonical shape.
    """

    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Memlets | ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # Single-shot: the matcher recognises (and skips) its own stage-2 loops
        # via the ``_scan_delta_`` transient, so re-application is a no-op.
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Rewrite every matching prefix-scan loop in ``sdfg`` (and nested SDFGs).

        :param sdfg: The SDFG to transform in place.
        :param _pipeline_results: Unused; kept for the Pass interface.
        :returns: ``{loop_label: scan_array_name}`` for each loop rewritten, or
                  ``None`` if no loop matched.
        """
        # Eliminate the frontend's trivial ``__out = __inp`` copy tasklets first: they
        # sit between the Add and the scan write and hide the real producer from the
        # matcher. ``TrivialTaskletElimination`` is value-preserving, so the SDFG stays
        # semantically identical whether the pass matches or not.
        from dace.transformation.dataflow.trivial_tasklet_elimination import TrivialTaskletElimination
        from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
        PatternMatchAndApplyRepeated([TrivialTaskletElimination()]).apply_pass(sdfg, {})

        rewritten: Dict[str, str] = {}
        # Snapshot the loop set up-front: rewriting one loop adds new LoopRegions to
        # the parent CFG that we do not want to re-traverse in the same pass.
        loops: List[Tuple[LoopRegion, ControlFlowRegion]] = []
        for sd in sdfg.all_sdfgs_recursive():
            for region in sd.all_control_flow_regions():
                if isinstance(region, LoopRegion) and region.loop_variable:
                    loops.append((region, region.parent_graph))
        for loop, parent in loops:
            match = _match(loop)
            if match is None:
                continue
            _rewrite(parent, loop, match)
            rewritten[loop.label] = match.array
        return rewritten or None

    def report(self, pass_retval: Any) -> Optional[str]:
        if not pass_retval:
            return None
        return f'ScanToReduction: rewrote {len(pass_retval)} prefix-scan loop(s): {pass_retval}'


class _Match:
    """A successfully matched prefix-scan loop, with everything ``_rewrite`` needs.

    :param array: The scan array's data-descriptor name.
    :param scan_axis: Index of the scanned dimension in ``array``'s shape.
    :param write_subset: The write memlet's subset (``arr[i+1, ...]`` form, on
                         the edge entering the write AccessNode).
    :param read_node_state: The state holding the unique read AccessNode of ``arr``.
    :param read_node: The unique read AccessNode of ``arr``.
    :param write_node_state: The state holding the unique write AccessNode of ``arr``.
    :param other_name: The non-scan source array's name.
    :param tasklet: The tasklet that combines the two inputs (the ``+`` node).
    :param tasklet_state: The state containing the tasklet.
    :param scan_in_conn: The tasklet input connector reading the arr value.
    :param out_conn: The tasklet output connector that writes the arr value.
    :param delta_rhs_src: The delta-stage RHS source (the non-scan operand of
                          the tasklet's BinOp, unparsed back to Python).
    :param iter_start: The loop's start expression.
    :param iter_end: The loop's inclusive end expression.
    """

    def __init__(self, array: str, scan_axis: int, write_subset: subsets.Range, read_node_state: SDFGState,
                 read_node: nodes.AccessNode, write_node_state: SDFGState, other_name: str, tasklet: nodes.Tasklet,
                 tasklet_state: SDFGState, scan_in_conn: str, out_conn: str, delta_rhs_src: str, iter_start, iter_end):
        self.array = array
        self.scan_axis = scan_axis
        self.write_subset = write_subset
        self.read_node_state = read_node_state
        self.read_node = read_node
        self.write_node_state = write_node_state
        self.other_name = other_name
        self.tasklet = tasklet
        self.tasklet_state = tasklet_state
        self.scan_in_conn = scan_in_conn
        self.out_conn = out_conn
        self.delta_rhs_src = delta_rhs_src
        self.iter_start = iter_start
        self.iter_end = iter_end


def _match(loop: LoopRegion) -> Optional[_Match]:
    """Decide whether ``loop`` is a canonical forward prefix scan, and return the
    info needed to rewrite it. Conservative: anything off-pattern returns ``None``.

    The matcher works at the AccessNode level so the body may be a single state
    with a tasklet (1-D shape) or a single state wrapping a ``Map`` over the
    non-scan dimensions (the 2-D ``out[i+1, j] = out[i, j] + src[i, j]`` shape).

    :param loop: The candidate :class:`~dace.sdfg.state.LoopRegion`.
    :returns: A populated :class:`_Match` on success, ``None`` otherwise.
    """
    if not loop.loop_variable:
        return None
    start = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if start is None or end is None or stride is None or stride != 1:
        return None

    sdfg = _root_sdfg(loop)
    loop_var = symbolic.pystr_to_symbolic(loop.loop_variable)

    # Locate the unique pair of AccessNode read/write of a single non-transient
    # array whose write subset offsets the read by +1 on exactly one axis.
    write_hit: Optional[Tuple[SDFGState, nodes.AccessNode, subsets.Range]] = None
    read_hit: Optional[Tuple[SDFGState, nodes.AccessNode, subsets.Range]] = None
    scan_array: Optional[str] = None

    for state in loop.all_states():
        for node in state.data_nodes():
            name = node.data
            desc = sdfg.arrays.get(name)
            if desc is None or not isinstance(desc, data.Array) or desc.transient:
                continue
            # Only consider arrays whose accesses involve ``loop_var`` (the rest
            # are passive inputs / outputs and not candidate scan arrays).
            for e in state.in_edges(node):
                if e.data is None or e.data.subset is None or e.data.data != name:
                    continue
                sub = e.data.subset
                if not _uses_loop_var(sub, loop_var):
                    continue
                if scan_array is None:
                    scan_array = name
                if name != scan_array:
                    return None  # writes to two different candidate arrays
                if write_hit is not None:
                    return None  # more than one carried write
                write_hit = (state, node, sub)
            for e in state.out_edges(node):
                if e.data is None or e.data.subset is None or e.data.data != name:
                    continue
                sub = e.data.subset
                if not _uses_loop_var(sub, loop_var):
                    continue
                if scan_array is None:
                    scan_array = name
                if name != scan_array:
                    # A read of a different non-transient array is fine -- only
                    # disqualifying when paired with a write conflict above.
                    continue
                if read_hit is not None and read_hit[1] is not node:
                    return None  # more than one distinct read AccessNode of arr
                if read_hit is None:
                    read_hit = (state, node, sub)
    if scan_array is None or write_hit is None or read_hit is None:
        return None

    # Refuse a second carried RMW on a different non-transient array (e.g. a
    # scalar accumulator ``acc[0] += src[i]`` whose constant-index write the
    # loop-var filter above misses). Aggregate per data name -- DaCe splits the
    # RMW into a read AccessNode and a write AccessNode of the same name.
    other_writes: Set[str] = set()
    other_reads: Set[str] = set()
    for state in loop.all_states():
        for node in state.data_nodes():
            name = node.data
            if name == scan_array:
                continue
            desc = sdfg.arrays.get(name)
            if desc is None or not isinstance(desc, data.Array) or desc.transient:
                continue
            if state.in_degree(node) > 0:
                other_writes.add(name)
            if state.out_degree(node) > 0:
                other_reads.add(name)
    if other_writes & other_reads:
        return None  # second carried RMW on a non-transient array

    read_state, read_node, read_subset = read_hit
    write_state, write_node, write_subset = write_hit
    if not (isinstance(read_subset, subsets.Range) and isinstance(write_subset, subsets.Range)):
        return None
    if len(read_subset.ndrange()) != len(write_subset.ndrange()):
        return None
    # Carry must be the *only* use of the arr-read AccessNode (single out-edge,
    # zero in-edges) and the arr-write must be the unique sink (single in-edge,
    # zero out-edges). Anything else means the array is read or written
    # elsewhere inside the loop and the rewrite would silently change those.
    if read_state.in_degree(read_node) != 0 or read_state.out_degree(read_node) != 1:
        return None
    if write_state.in_degree(write_node) != 1 or write_state.out_degree(write_node) != 0:
        return None

    scan_axis = _find_scan_axis(read_subset, write_subset, loop_var)
    if scan_axis is None:
        return None

    # Locate the tasklet that does the ``arr_in + other_in`` combine. It is the
    # unique tasklet writing to the write AccessNode (possibly through a MapExit).
    tasklet, tasklet_state, scan_in_conn, _other_in_conn, out_conn = _trace_tasklet(write_state, write_node, scan_array)
    if tasklet is None or tasklet.code.language != dtypes.Language.Python:
        return None

    try:
        tree = ast.parse((tasklet.code.as_string or '').strip())
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        return None
    rhs = tree.body[0].value
    if not (isinstance(rhs, ast.BinOp) and isinstance(rhs.op, ast.Add)):
        return None  # v1: only ``+``

    delta_rhs_node = _other_operand(rhs, scan_in_conn)
    if delta_rhs_node is None:
        return None
    try:
        delta_rhs_src = ast.unparse(delta_rhs_node)
    except (AttributeError, ValueError):
        return None

    # Identify the non-arr source AccessNode (e.g. ``src``) -- the unique read
    # whose subset uses ``loop_var`` and which is NOT the scan array. For v1 we
    # require exactly one such source so the delta-buffer redirect is unambiguous.
    other_name = _find_other_source(loop, sdfg, scan_array, loop_var)
    if other_name is None:
        return None

    # Idempotence guard: the non-scan input is already a ``_scan_delta_``
    # transient, i.e. this is the stage-2 prefix loop we ourselves emitted.
    if other_name.startswith(_DELTA_PREFIX):
        return None

    return _Match(
        array=scan_array,
        scan_axis=scan_axis,
        write_subset=write_subset,
        read_node_state=read_state,
        read_node=read_node,
        write_node_state=write_state,
        other_name=other_name,
        tasklet=tasklet,
        tasklet_state=tasklet_state,
        scan_in_conn=scan_in_conn,
        out_conn=out_conn,
        delta_rhs_src=delta_rhs_src,
        iter_start=start,
        iter_end=end,
    )


def _uses_loop_var(subset: subsets.Subset, loop_var) -> bool:
    """``True`` if any subset bound mentions ``loop_var``."""
    for fs in subset.free_symbols:
        if symbolic.pystr_to_symbolic(str(fs)) == loop_var:
            return True
    return False


def _find_scan_axis(read: subsets.Range, write: subsets.Range, loop_var) -> Optional[int]:
    """Find the axis where ``write`` offsets ``read`` by exactly ``+1`` along ``loop_var``.

    All non-scan axes must agree (their bounds may or may not involve
    ``loop_var``, but the read and write must align). Returns the axis index, or
    ``None`` if no clean +1 scan axis exists.
    """
    scan_axis: Optional[int] = None
    for axis, ((rb, re_, rs), (wb, we_, ws)) in enumerate(zip(read.ndrange(), write.ndrange())):
        if rb != re_ or wb != we_ or rs != 1 or ws != 1:
            return None  # only single-point subsets per dimension
        rb_s = symbolic.pystr_to_symbolic(str(rb))
        wb_s = symbolic.pystr_to_symbolic(str(wb))
        delta = symbolic.simplify(wb_s - rb_s)
        if delta == 0:
            continue
        if scan_axis is not None:
            return None  # more than one differing dim
        if not (getattr(delta, 'is_Integer', False) and int(delta) == 1):
            return None
        # Each side must be ``loop_var + const`` (constants only after subtraction).
        if loop_var not in rb_s.free_symbols or loop_var not in wb_s.free_symbols:
            return None
        if loop_var in symbolic.simplify(rb_s - loop_var).free_symbols:
            return None
        if loop_var in symbolic.simplify(wb_s - loop_var).free_symbols:
            return None
        scan_axis = axis
    return scan_axis


def _trace_tasklet(
    state: SDFGState, write_node: nodes.AccessNode, scan_array: str
) -> Tuple[Optional[nodes.Tasklet], Optional[SDFGState], Optional[str], Optional[str], Optional[str]]:
    """Walk back from ``write_node`` to find the tasklet that produces its value.

    The producer is either directly connected (shape A: single tasklet) or sits
    inside a Map scope (shape B: tasklet under a MapExit). Returns
    ``(tasklet, tasklet_state, scan_in_conn, other_in_conn, out_conn)``; all
    ``None`` on any deviation from these two shapes.
    """
    incoming = list(state.in_edges(write_node))
    if len(incoming) != 1:
        return None, None, None, None, None
    e = incoming[0]
    producer = e.src

    # Walk through a pass-through intermediate transient (in=1, out=1) -- the
    # frontend slice-copy form ``Add -> intermediate AN -> write AN``.
    if isinstance(producer, nodes.AccessNode):
        desc = state.sdfg.arrays.get(producer.data)
        if (desc is not None and getattr(desc, 'transient', False) and state.in_degree(producer) == 1
                and state.out_degree(producer) == 1):
            inner = state.in_edges(producer)[0]
            producer = inner.src

    # Shape A: tasklet feeds the write AccessNode (directly or via the intermediate).
    if isinstance(producer, nodes.Tasklet):
        return _resolve_tasklet_conns(state, producer, scan_array)

    # Shape B: producer is a MapExit; walk inside the map scope to find the tasklet.
    if isinstance(producer, nodes.MapExit):
        out_conn = e.src_conn  # the MapExit output connector that fed the write
        # The matching MapExit input connector follows the ``IN_<n>/OUT_<n>`` convention.
        in_conn = _map_inner_conn(out_conn)
        if in_conn is None:
            return None, None, None, None, None
        inner_edges = [ie for ie in state.in_edges(producer) if ie.dst_conn == in_conn]
        if len(inner_edges) != 1:
            return None, None, None, None, None
        producer2 = inner_edges[0].src
        if not isinstance(producer2, nodes.Tasklet):
            return None, None, None, None, None
        return _resolve_tasklet_conns(state, producer2, scan_array)

    return None, None, None, None, None


def _resolve_tasklet_conns(
    state: SDFGState, tasklet: nodes.Tasklet, scan_array: str
) -> Tuple[Optional[nodes.Tasklet], Optional[SDFGState], Optional[str], Optional[str], Optional[str]]:
    """Classify a tasklet's two input connectors as ``scan_in_conn`` (carries
    the scan array's value) and ``other_in_conn`` (the rest). Walks back across
    a single intervening MapEntry if present, so shape B works too.
    """

    def _has_data(e):
        return e.data is not None and not e.data.is_empty()

    in_edges = [e for e in state.in_edges(tasklet) if _has_data(e)]
    out_edges = [e for e in state.out_edges(tasklet) if _has_data(e)]
    if len(in_edges) != 2 or len(out_edges) != 1:
        return None, None, None, None, None

    scan_conn = other_conn = None
    for e in in_edges:
        src_array = _source_array(state, e)
        if src_array == scan_array:
            if scan_conn is not None:
                return None, None, None, None, None
            scan_conn = e.dst_conn
        else:
            if other_conn is not None:
                return None, None, None, None, None
            other_conn = e.dst_conn
    if scan_conn is None or other_conn is None:
        return None, None, None, None, None
    return tasklet, state, scan_conn, other_conn, out_edges[0].src_conn


def _source_array(state: SDFGState, edge) -> Optional[str]:
    """Walk back across a passthrough intermediate AccessNode (transient ``in=1, out=1``,
    typically the slice-copy the frontend inserts) and/or a single MapEntry, to identify
    the originating array. Falls back to ``None`` for anything else.
    """
    src = edge.src
    # Frontend slice-copy ``AN(src) -> AN(src_index, transient) -> tasklet`` --
    # walk past the intermediate scalar to the real source ``src``.
    if isinstance(src, nodes.AccessNode):
        desc = state.sdfg.arrays.get(src.data)
        if (desc is not None and getattr(desc, 'transient', False) and state.in_degree(src) == 1
                and state.out_degree(src) == 1):
            upstream = state.in_edges(src)[0].src
            if isinstance(upstream, nodes.AccessNode):
                return upstream.data
        return src.data
    if isinstance(src, nodes.MapEntry):
        # The matching MapEntry input connector follows ``IN_<n>/OUT_<n>``.
        outer_conn = _map_outer_conn(edge.src_conn)
        if outer_conn is None:
            return None
        outer_edges = [oe for oe in state.in_edges(src) if oe.dst_conn == outer_conn]
        if len(outer_edges) != 1:
            return None
        outer_src = outer_edges[0].src
        if isinstance(outer_src, nodes.AccessNode):
            # Same passthrough handling on the outer side of the map.
            desc = state.sdfg.arrays.get(outer_src.data)
            if (desc is not None and getattr(desc, 'transient', False) and state.in_degree(outer_src) == 1
                    and state.out_degree(outer_src) == 1):
                upstream = state.in_edges(outer_src)[0].src
                if isinstance(upstream, nodes.AccessNode):
                    return upstream.data
            return outer_src.data
    return None


def _map_inner_conn(out_conn: Optional[str]) -> Optional[str]:
    """Convert a MapExit ``OUT_<n>`` connector name to its inner ``IN_<n>``."""
    if out_conn and out_conn.startswith('OUT_'):
        return 'IN_' + out_conn[4:]
    return None


def _map_outer_conn(in_conn: Optional[str]) -> Optional[str]:
    """Convert a MapEntry ``OUT_<n>`` connector name to its outer ``IN_<n>``."""
    if in_conn and in_conn.startswith('OUT_'):
        return 'IN_' + in_conn[4:]
    return None


def _find_other_source(loop: LoopRegion, sdfg: SDFG, scan_array: str, loop_var) -> Optional[str]:
    """Return the name of the unique non-arr AccessNode read inside ``loop``
    whose memlet mentions ``loop_var``, or ``None`` if there is no such unique
    source.
    """
    seen_names = set()
    for state in loop.all_states():
        for node in state.data_nodes():
            if node.data == scan_array or node.data in seen_names:
                continue
            desc = sdfg.arrays.get(node.data)
            if desc is None:
                continue
            for e in state.out_edges(node):
                if e.data is None or e.data.subset is None or e.data.data != node.data:
                    continue
                if _uses_loop_var(e.data.subset, loop_var):
                    seen_names.add(node.data)
                    break
    if len(seen_names) != 1:
        return None
    return next(iter(seen_names))


def _other_operand(binop: ast.BinOp, scan_in_conn: str) -> Optional[ast.AST]:
    """Return the operand subtree of ``binop`` that is NOT ``Name(scan_in_conn)``.

    The pass only matches a clean ``scan_in + g(...)`` (or its commuted form);
    anything that mixes the scan input into a more complex expression -- e.g.
    ``scan_in * 2`` -- is not a separable prefix scan and we refuse it.

    :param binop: The tasklet's RHS BinOp.
    :param scan_in_conn: The tasklet input connector that reads the scan array.
    :returns: The non-scan operand AST node, or ``None`` if neither operand is
              a bare reference to ``scan_in_conn``.
    """

    def _is_scan_ref(node) -> bool:
        return isinstance(node, ast.Name) and node.id == scan_in_conn

    def _mentions_scan(node) -> bool:
        return any(_is_scan_ref(sub) for sub in ast.walk(node))

    left, right = binop.left, binop.right
    if _is_scan_ref(left) and not _mentions_scan(right):
        return right
    if _is_scan_ref(right) and not _mentions_scan(left):
        return left
    return None


def _rewrite(parent: ControlFlowRegion, loop: LoopRegion, match: _Match):
    """Replace ``loop`` with two sibling :class:`LoopRegion` s: an elementwise
    delta computation followed by a sequential prefix sum into the scan array.

    The rewrite is uniform across the 1-D ``out[i+1] = out[i] + src[i]`` shape
    and the 2-D ``out[i+1, j] = out[i, j] + src[i, j]`` shape (where the body
    is a Map over ``j``). Strategy:

    1. Deep-copy the original loop -- this gives stage 2's skeleton.
    2. Surgically mutate the original loop into stage 1 (write retargeted to
       ``delta``; scan-array read disconnected from the tasklet; tasklet code
       collapsed to ``_out = <delta_rhs>``).
    3. Surgically mutate the deep-copy into stage 2 (the other input's
       AccessNode -- the formerly-``src`` source -- is renamed to ``delta``,
       and its memlets are shifted onto delta's loop-relative coordinate).

    :param parent: The control-flow region that owns ``loop``.
    :param loop: The matched scan loop.
    :param match: The :class:`_Match` produced by :func:`_match`.
    """
    import dace  # avoid an import cycle at module scope
    sdfg = _root_sdfg(parent)

    # Allocate the per-iteration delta buffer. Shape mirrors the scan array,
    # except the scan dim shrinks to the iteration count ``end - start + 1``.
    arr_desc: data.Array = sdfg.arrays[match.array]
    trip = symbolic.simplify(match.iter_end - match.iter_start + 1)
    delta_shape = list(arr_desc.shape)
    delta_shape[match.scan_axis] = trip
    delta_name, _ = sdfg.add_transient(_DELTA_PREFIX + match.array,
                                       delta_shape,
                                       arr_desc.dtype,
                                       storage=arr_desc.storage,
                                       find_new_name=True)

    # Deep-copy first so the original's node identities (used by ``match``) stay
    # valid through stage-1 surgery. The copy reuses the loop variable name;
    # we rename it below (after it's attached to the SDFG so ``replace_dict``
    # can resolve ``state.sdfg``) to avoid LoopToMap's "used after the loop"
    # false positive on the sibling scope's redefinition.
    stage2 = copy.deepcopy(loop)
    stage2.label = loop.label + '_prefix'
    stage2_var = loop.loop_variable + '_prefix'

    # Stage 1: mutate the original loop in place.
    _mutate_to_stage1(loop, match, delta_name)
    loop.label = loop.label + '_delta'

    # Stage 2: walk the deep-copy and redirect the non-arr source onto ``delta``.
    _mutate_to_stage2(stage2, match, delta_name)

    # Splice stage2 in after the original (now-stage1) loop. ``ensure_unique_name``
    # keeps the parent's block-label set consistent even if our suffixed label
    # happens to clash with something already there.
    out_edges = list(parent.out_edges(loop))
    parent.add_node(stage2, ensure_unique_name=True)
    parent.add_edge(loop, stage2, dace.InterstateEdge())
    for e in out_edges:
        parent.remove_edge(e)
        parent.add_edge(stage2, e.dst, e.data)

    # Refresh the SDFG's CFG bookkeeping so downstream analyses (e.g. ``LoopToMap``)
    # see the new ``stage2`` LoopRegion in ``sdfg.all_control_flow_regions()`` and
    # ``state.sdfg`` resolves on stage2's states.
    sdfg.reset_cfg_list()

    # Rename stage2's loop variable now that it's attached to the parent SDFG: the
    # deep-copy reused the original loop variable name, which would make
    # ``LoopToMap`` on stage 1 refuse with "loop-defined symbol used after the
    # loop" (it sees stage 2 also referencing the name in a sibling scope).
    # ``replace_dict`` updates the LoopRegion's init/condition/update statements
    # as well as all in-body memlets and tasklets.
    stage2.replace_dict({loop.loop_variable: stage2_var})


def _mutate_to_stage1(loop: LoopRegion, match: _Match, delta_name: str):
    """Turn ``loop`` into the stage-1 elementwise delta computation in place.

    Drops the carried arr-read input, retargets the arr-write to ``delta``, and
    rewrites the tasklet's body to the standalone delta RHS.
    """
    state = match.tasklet_state
    tasklet = match.tasklet

    # Drop the carried scan-array input from the tasklet, including any edges
    # routed through a surrounding MapEntry (shape B).
    _disconnect_input(state, tasklet, match.scan_in_conn, match.array)
    tasklet.code.as_string = f'{match.out_conn} = {match.delta_rhs_src}'

    # Retarget the write chain onto ``delta``. The scan write is at
    # ``loop_var + write_offset`` (1 for ``arr[i+1] = …``, 0 for ``arr[i] =
    # arr[i-1] + …``); the delta slot for this iteration is ``loop_var -
    # iter_start``, so shift by ``iter_start + write_offset``.
    write_rb = match.write_subset.ndrange()[match.scan_axis][0]
    loop_var_sym = symbolic.pystr_to_symbolic(loop.loop_variable)
    write_offset = symbolic.simplify(symbolic.pystr_to_symbolic(str(write_rb)) - loop_var_sym)
    _retarget_write_chain(match.write_node_state, match.tasklet, match.out_conn, match.array, delta_name,
                          match.scan_axis, match.iter_start + write_offset)

    # Tidy up the now-unused arr-read AccessNode: an AccessNode with no edges
    # is removed.  Validation tolerates a leftover bare read of an unmapped
    # array, but we'd rather not leave dead nodes.
    rn = match.read_node
    if match.read_node_state.degree(rn) == 0:
        match.read_node_state.remove_node(rn)

    # Fold pass-through intermediate scalar transients in the stage-1 body so
    # LoopToMap can parallelize it (its analysis refuses on the unindexed ``[0]``
    # reads/writes the slice-copy intermediates introduce).
    for st in loop.all_states():
        _inline_passthrough_intermediates(st)


def _inline_passthrough_intermediates(state: SDFGState):
    """Inline pass-through intermediate transient AccessNodes (``in=1, out=1`` in
    *this* state) by redirecting the producer edge directly to the consumer's
    connector. Only this state's AccessNode is removed; the descriptor stays so
    other states/sub-regions referencing the same data (e.g. the scan pass's
    stage-2 deep-copy) are unaffected. The ``in=1, out=1`` guard is the
    single-consumer-in-this-state check the rewrite needs to keep semantics.
    """
    for n in list(state.data_nodes()):
        if state.in_degree(n) != 1 or state.out_degree(n) != 1:
            continue
        desc = state.sdfg.arrays.get(n.data)
        if desc is None or not getattr(desc, 'transient', False):
            continue
        in_e = state.in_edges(n)[0]
        out_e = state.out_edges(n)[0]
        # The memlet that describes the real data slice: a slice-copy read
        # (producer is an AccessNode) carries the source subset on its in-edge;
        # any other producer (Tasklet/MapExit) carries the destination subset on
        # the consumer-side out-edge.
        new_memlet = copy.deepcopy(in_e.data if isinstance(in_e.src, nodes.AccessNode) else out_e.data)
        # The intermediate's side of the original ``other_subset`` (the scalar
        # ``[0]`` slot) is gone after inlining and would mismatch the surviving
        # endpoint's dimensionality, so drop it.
        new_memlet.other_subset = None
        state.add_edge(in_e.src, in_e.src_conn, out_e.dst, out_e.dst_conn, new_memlet)
        state.remove_edge(in_e)
        state.remove_edge(out_e)
        state.remove_node(n)


def _mutate_to_stage2(stage2: LoopRegion, match: _Match, delta_name: str):
    """Turn the deep-copied loop into the stage-2 prefix sum.

    The deep-copy still does ``arr[i+1] = arr[i] + src[i]``; renaming every
    ``src``-AccessNode-and-memlet to ``delta`` (with the scan-axis index
    shifted by ``-iter_start``) yields ``arr[i+1] = arr[i] + delta[i - start]``.
    """
    other_name = match.other_name
    for state in stage2.all_states():
        # Rename AccessNodes for ``other_name`` -> ``delta`` (data field only).
        for node in state.data_nodes():
            if node.data == other_name:
                node.data = delta_name
        # Rename and shift every memlet that referenced ``other_name``. This covers
        # both the AccessNode-adjacent memlets and the inner-map-scope memlets
        # (the propagated MapEntry edge plus the per-element edge to the tasklet).
        for e in state.edges():
            if e.data is not None and e.data.data == other_name:
                _shift_memlet(e.data, other_name, delta_name, match.scan_axis, match.iter_start)


def _disconnect_input(state: SDFGState, tasklet: nodes.Tasklet, conn: str, arr_name: str):
    """Remove the tasklet input ``conn`` and any edges routed through an outer
    MapEntry or an intermediate transient AccessNode (the frontend slice-copy)
    that fed it; clean up the orphaned read chain so stage 1 no longer reads
    ``arr``.
    """
    if conn not in tasklet.in_connectors:
        return
    for e in list(state.in_edges(tasklet)):
        if e.dst_conn != conn:
            continue
        outer = e.src
        state.remove_edge(e)
        # If the source is a MapEntry, also drop its matching outer input edge.
        if isinstance(outer, nodes.MapEntry):
            outer_in = _map_outer_conn(e.src_conn)
            if outer_in is not None:
                for oe in list(state.in_edges(outer)):
                    if oe.dst_conn == outer_in and isinstance(oe.src, nodes.AccessNode) and oe.src.data == arr_name:
                        state.remove_edge(oe)
                        outer.remove_in_connector(outer_in)
                        if e.src_conn in outer.out_connectors:
                            outer.remove_out_connector(e.src_conn)
        # If the source is an intermediate transient AccessNode (the slice-copy the
        # frontend inserts: ``arr -> intermediate -> tasklet``), it is now an orphan
        # (in=1, out=0). LoopToMap sees the orphan's unindexed write and refuses,
        # so peel back the orphan + the matching upstream edge from the arr-read AN.
        elif isinstance(outer, nodes.AccessNode) and state.out_degree(outer) == 0:
            desc = state.sdfg.arrays.get(outer.data)
            if desc is not None and getattr(desc, 'transient', False):
                for oe in list(state.in_edges(outer)):
                    state.remove_edge(oe)
                state.remove_node(outer)
    tasklet.remove_in_connector(conn)


def _retarget_write_chain(state: SDFGState, tasklet: nodes.Tasklet, out_conn: str, old_name: str, new_name: str,
                          scan_axis: int, iter_start):
    """Re-label the chain of edges from ``tasklet[out_conn]`` to the final
    AccessNode of ``old_name`` so they target ``new_name`` instead, shifting the
    scan-axis index by ``-iter_start`` so it indexes ``delta``'s loop-relative
    coordinate.
    """
    cur = tasklet
    cur_conn = out_conn
    while True:
        out_edges = [e for e in state.out_edges(cur) if e.src_conn == cur_conn]
        if len(out_edges) != 1:
            return
        e = out_edges[0]
        _shift_memlet(e.data, old_name, new_name, scan_axis, iter_start)
        nxt = e.dst
        if isinstance(nxt, nodes.AccessNode):
            if nxt.data == old_name:
                nxt.data = new_name
                return
            # Pass-through intermediate transient (the frontend's slice-copy holding the
            # tasklet's result before writing to the scan array). Walk through and keep
            # renaming the chain until we reach the scan AccessNode itself.
            desc = state.sdfg.arrays.get(nxt.data)
            if (desc is not None and getattr(desc, 'transient', False) and state.in_degree(nxt) == 1
                    and state.out_degree(nxt) == 1):
                cur = nxt
                cur_conn = None  # AccessNode out-edges have src_conn = None
                continue
            return
        if isinstance(nxt, nodes.MapExit):
            # ``IN_<n>`` (inside the map) pairs with ``OUT_<n>`` (outside); the
            # write keeps flowing through the matching ``OUT_<n>`` edge.
            if not e.dst_conn or not e.dst_conn.startswith('IN_'):
                return
            cur = nxt
            cur_conn = 'OUT_' + e.dst_conn[3:]
            continue
        return


def _shift_memlet(memlet: mm.Memlet, old_name: str, new_name: str, scan_axis: int, iter_start):
    """Rename ``memlet`` to ``new_name`` and shift its ``scan_axis`` index by
    ``-iter_start`` so it indexes the delta buffer's loop-relative coordinate.
    """
    if memlet is None:
        return
    if memlet.data == old_name:
        memlet.data = new_name
        if memlet.subset is not None:
            memlet.subset = _shift_subset(memlet.subset, scan_axis, iter_start)
    if memlet.other_subset is not None and getattr(memlet, '_other_data', None) == old_name:
        # ``other_data`` is rare on raw memlets; covered for completeness.
        memlet.other_subset = _shift_subset(memlet.other_subset, scan_axis, iter_start)


def _shift_subset(subset: subsets.Subset, scan_axis: int, iter_start) -> subsets.Subset:
    """Shift ``subset``'s ``scan_axis`` bounds by ``-iter_start``."""
    if not isinstance(subset, subsets.Range):
        return subset
    ranges = list(subset.ndrange())
    if scan_axis >= len(ranges):
        return subset
    rb, re_, rs = ranges[scan_axis]
    rb_s = symbolic.simplify(symbolic.pystr_to_symbolic(str(rb)) - iter_start)
    re_s = symbolic.simplify(symbolic.pystr_to_symbolic(str(re_)) - iter_start)
    ranges[scan_axis] = (rb_s, re_s, rs)
    return subsets.Range(ranges)


def _root_sdfg(region: ControlFlowRegion) -> SDFG:
    cur = region
    while not isinstance(cur, SDFG):
        cur = cur.parent_graph
    return cur
