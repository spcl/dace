# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Name every loop in the canonical form, so the standalone rendering says what each one is.

CPF output is read by a specializing pass or by a person, and both arrive at a ``for`` loop with
the same question: is this order required, or is it the order canonicalization happened to leave?
The canonical form answers it -- a parallel axis is a Map, a required order is a LoopRegion -- but
the rendering flattens both to ``for``, and the answer is gone.

So it is written down, as a ``specialization_hint`` comment, in four kinds:

* **parallel** -- a Map. Its iterations are independent; the schedule is a separate decision.
* **sequential** -- a loop whose carried dependence ``LoopToMap`` PROVED. The order is required.
* **potentially sequential** -- a loop ``LoopToMap`` declined without reaching a verdict. Nothing
  was proven either way, and the difference from the line above is the whole point of this pass:
  a proof closes the question, a decline does not.
* **wavefront** -- the diagonal, front, tile and tile-interior axes a skew produces.
  :mod:`~dace.transformation.passes.canonicalize.wavefront_skew` labels those itself, because it is
  the only thing that knows a loop is one; the wording lives here so all four kinds read alike.

One pass, run last, rather than a label at each site that could have set one: a loop that a pass
forgot would then carry no comment and read as "nobody looked", which is a claim in its own right.
Here every loop is visited, and the classifier is ``LoopToMap.can_be_applied`` -- the same oracle
:class:`~dace.transformation.passes.loop_to_reduce.PinCarriedTopLevelLoops` uses, so the comment
and the pipeline's own decision cannot disagree.

A hint is a NOTE. Nothing in the pipeline dispatches on these strings, ``hint_comment`` drops them
outside a standalone rendering, and this pass changes no graph.
"""

import re
from collections.abc import Iterator
from typing import Any, NamedTuple

import sympy

from dace import SDFG, data, symbolic
from dace.memlet import Memlet
from dace.optionals import required
from dace.ordered import OrderedSet
from dace.sdfg import nodes
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.narrowing import as_basic, as_range, free_symbols
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.subsets import Subset
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis, smt_dependence

# Three classes, named by the first word of every loop hint: ``parallel`` and ``sequential`` are
# settled and need no further dependence reasoning, ``unsure`` is the only open one. A role (a
# wavefront axis, an inner tile) follows the class word as a qualifier, never replaces it.

#: The first words of every loop hint: its class (see the three below).
CLASS_PREFIXES = ("parallel -- ", "sequential -- ", "unsure -- ")

#: A Map. Data-parallel by construction, whatever schedule it ends up carrying.
PARALLEL = (
    "parallel -- the iterations are independent\n"
    "settled: proven, no need to re-check; focus: schedule, tiling, vectorization, fusion, data layout"
)

#: Added under a multi-dimensional Map's class line: the rendering writes one nested ``for`` per dimension and
#: the comment only above the outermost, so the inner ones are named here.
MAP_DIMENSIONS = "covers the {count} nested loops over {params}: every one of them is parallel"

#: A loop whose carried dependence was proven. ``{reason}`` is the carrying ACCESS, since that is
#: the part a reader acts on -- the prose around it repeated on every such loop and said the same
#: thing each time.
SEQUENTIAL_PROVEN = (
    "sequential -- carried: {reason}\n"
    "settled: proven, {why}; no schedule or annotation makes it parallel -- removing the dependence takes an "
    "algorithmic change (break an anti dependence, privatize, rewrite the recurrence as a scan or reduction), "
    "otherwise optimize the work inside"
)

#: A carried dependence on a loop whose body is a Do-All nest (a time loop over a stencil): proven too, but a skew
#: of the loop with the nest inside it would parallelize it -- canonicalization does not skew a time loop.
SEQUENTIAL_PROVEN_OVER_DOALL = (
    "sequential -- carried: {reason}\n"
    "settled: proven, {why}; a skew of this loop with the parallel nest inside it (a time-tiled wavefront) would "
    "make it parallel, which canonicalization does not apply to a loop over a Do-All nest"
)

#: A loop the dependence test declined to answer, or never examined. NOT the same fact as the line
#: above: not proven, and the wording has to keep saying so in one word.
UNSURE = (
    "unsure -- {reason}\n"
    "open: not proven either way; kept sequential to be safe, and may well be parallel -- a fact about the "
    "indices (a bound, a permutation) settles it"
)

#: The ``UNSURE`` reason of a loop the dependence test never answered for; the specific cause follows a colon.
NEVER_EXAMINED = "never examined for dependences"
#: ... because the loop has no single iteration counter to test (a while loop).
NO_COUNTER = f"{NEVER_EXAMINED}: no single iteration counter (a while loop)"

#: A loop the dependence test finds independent that an earlier pass pinned sequential anyway (the
#: fallback arm of a specialization). Rendered without a pragma, so it must not read as parallel.
SEQUENTIAL_PINNED = (
    "sequential -- pinned by an earlier pass; the dependence test finds the iterations independent\n"
    "settled: the pass keeps the order on purpose (such as the fallback arm of a specialization)"
)

#: The untiled skew: a sequential diagonal over a parallel front. The second line is the only
#: advice a reader can act on -- which of the two correct shapes to run -- so it survives the cut
#: while the restatement of what a skew is does not.
WAVEFRONT_DIAGONAL = (
    "sequential -- wavefront diagonal {skew}: the skew put every dependence on this axis\n"
    "alternative: the unskewed nest, sequential in both axes -- worth timing on CPU, "
    "rarely on GPU"
)
WAVEFRONT_FRONT = "parallel -- wavefront front {skew}: at a fixed diagonal the points are independent"

#: The tiled skew: a sequential tile diagonal over a parallel tile column over a sequential interior.
WAVEFRONT_TILE_DIAGONAL = (
    "sequential -- wavefront tile diagonal {skew} {tile}: the tile diagonal carries "
    "every dependence\n"
    "alternatives: the element diagonal, or the unskewed nest -- all three bit-identical; "
    "a bigger tile trades kernel launches for block-local barriers"
)
WAVEFRONT_TILE_COLUMN = "parallel -- wavefront tile column {tile}: the tiles on one diagonal are independent"
WAVEFRONT_TILE_INTERIOR = (
    "sequential -- inner tile of a wavefront {tile}: the original order is kept verbatim, "
    "so the tiled result is bit-identical"
)


def skew_label(a: int, b: int, u: str, v: str) -> str:
    """The skew that produced this wavefront, as ``(t = i + j)``.

    Which skew was taken is the one fact about a wavefront a reader cannot recover from the emitted
    loop: after the rewrite the axes are named t and p and the original iterators are gone. The
    coefficient is written only where it is not 1, so the common unit skew stays short.
    """

    def term(coeff: int, name: str) -> str:
        return name if abs(coeff) == 1 else f"{abs(coeff)}*{name}"

    lead = f"-{term(a, u)}" if a < 0 else term(a, u)
    return f"(t = {lead} {'-' if b < 0 else '+'} {term(b, v)})"


def tile_label(bi: int, bj: int) -> str:
    """The tile extent the diagonal walks -- the number that trades kernel launches for barriers."""
    return f"[{bi}x{bj}]"


def refusal_reason(loop: LoopRegion) -> str | None:
    """Why ``LoopToMap`` would refuse ``loop``, ``None`` if it would accept, ``''`` if it refused without a reason.

    Asked with ``pinned_sequential`` set aside. The pin is a schedule decision an earlier pass made,
    and "loop is pinned sequential" is not an answer to the dependence question a reader is asking --
    it only names the pass that got there first. Restored either way; the probe reads the graph and
    does not touch it.
    """
    if not loop.loop_variable:
        return NO_COUNTER
    from dace.transformation.interstate.loop_to_map import LoopToMap
    from dace.transformation.passes.parallelize_loops import ParallelizeLoops

    probe = LoopToMap()
    pinned = loop.pinned_sequential
    loop.pinned_sequential = False
    try:
        applicable = ParallelizeLoops().accepts(loop, probe)
    except Exception as error:  # pylint: disable=broad-exception-caught
        # A comment must not be able to fail the compilation that asked for it. Nothing downstream
        # reads a hint, so a probe with no answer leaves the loop unsure and says so.
        return f"{NEVER_EXAMINED}: the dependence test failed ({type(error).__name__})"
    finally:
        loop.pinned_sequential = pinned
    return None if applicable else (probe.last_refusal_reason or "")


#: ``LoopToMap``'s refusal, as ``<kind> conflict on <array> within the loop body - src_subset=<subset>``.
REFUSAL_SHAPE = re.compile(r"(\w+)-after-(\w+) conflict on (\w+).*?src_subset=(.*)$", re.DOTALL)

#: The three conflict kinds, as the two-letter names a reader of dependence analysis already has.
CONFLICT_ABBREV = {("read", "write"): "RAW", ("write", "read"): "WAR", ("write", "write"): "WAW"}

#: What each proven conflict means for the order, in the words a reader of the loop needs.
CONFLICT_MEANING = {
    "RAW": "an iteration reads what an earlier one wrote",
    "WAR": "an iteration overwrites what an earlier one still has to read",
    "WAW": "two iterations write the same element and the last write must win",
}


def proven_why(access: str) -> str:
    """The order a proven conflict requires, from the kind ``access`` starts with (``RAW on ...``)."""
    return CONFLICT_MEANING.get(access.split(" ", 1)[0], "one iteration depends on another one")


def carrying_access(reason: str) -> str:
    """``reason`` as the ACCESS that carries it: ``RAW on aa[_loop_it_1 - 1, 8:LEN_2D]``.

    The refusal reads as a sentence because it is also shown to a human debugging a refusal. In a
    rendered form it is one comment on one loop, repeated for every carried loop in the unit, and
    the sentence around the access says the same thing every time. Anything that does not match the
    known shape is passed through whole rather than truncated -- a reason nobody anticipated is
    still worth reading.
    """
    hit = REFUSAL_SHAPE.search(reason)
    if hit is None:
        return reason
    first, second, array, subset = hit.groups()
    kind = CONFLICT_ABBREV.get((first, second), f"{first}-after-{second}")
    return f"{kind} on {array}[{subset.strip()}]"


#: Array name -> (outer array, element offset in the loop's names) for every array the body can see.
ArrayBindings = dict[str, tuple[str, tuple[sympy.Expr, ...]]]


class Access(NamedTuple):
    """One element access in a loop body, in the loop SDFG's names: ``array[index]``."""

    array: str
    write: bool
    index: tuple[sympy.Expr, ...]


class Body(NamedTuple):
    """What :func:`walk_body` collects: element accesses, names that vary within one iteration, the
    params of maps, and whether some map range moves with the loop."""

    accesses: list[Access]
    varying: OrderedSet[str]
    params: OrderedSet[str]
    moving: list[bool]


def as_loop_names(expr: symbolic.SymbolicType, mapping: dict[str, sympy.Expr]) -> sympy.Expr:
    """``expr`` rewritten through ``mapping`` (inner name -> outer expression) and reparsed, so one
    name is one symbol instance whichever scope minted it. A ``SymExpr`` bound (a tiled map's ``Min(N - 1, g + 63)``,
    over-approximated as ``g + 63``) is read exactly: its text names both and does not parse back."""
    expr = symbolic.pystr_to_symbolic(str(expr.expr if isinstance(expr, symbolic.SymExpr) else expr))
    repl = {sym: mapping[sym.name] for sym in free_symbols(as_basic(expr)) if sym.name in mapping}
    expr = expr.subs(repl, simultaneous=True) if repl else expr
    return symbolic.pystr_to_symbolic(str(expr))


def point_index(
    subset: Subset | None, mapping: dict[str, sympy.Expr], offset: tuple[sympy.Expr, ...]
) -> tuple[sympy.Expr, ...] | None:
    """The single element ``subset`` names, in the loop's names, or ``None`` for a range."""
    if subset is None or (offset and len(offset) != as_range(subset).dims()):
        return None
    index = []
    for dim, (begin, end, _) in enumerate(subset.ndrange()):
        begin = as_loop_names(begin, mapping)
        if begin != as_loop_names(end, mapping):
            return None
        index.append(begin + offset[dim] if offset else begin)
    return tuple(index)


def edge_accesses(
    edge: MultiConnectorEdge[Memlet], state: SDFGState, arrays: ArrayBindings, mapping: dict[str, sympy.Expr]
) -> list[Access]:
    """The element reads and writes ``edge`` makes: into or out of a tasklet, or an array copy."""
    memlet = edge.data
    if memlet.is_empty() or memlet.wcr is not None:
        return []
    if isinstance(edge.src, nodes.AccessNode) and isinstance(edge.dst, nodes.AccessNode):
        sides = [
            (edge.src.data, memlet.get_src_subset(edge, state), False),
            (edge.dst.data, memlet.get_dst_subset(edge, state), True),
        ]
    elif isinstance(edge.dst, nodes.Tasklet):
        sides = [(memlet.data, memlet.subset, False)]
    elif isinstance(edge.src, nodes.Tasklet):
        sides = [(memlet.data, memlet.subset, True)]
    else:
        return []
    found = []
    for name, subset, write in sides:
        if name not in arrays:
            continue
        outer, offset = arrays[name]
        index = point_index(subset, mapping, offset)
        if index is not None:
            found.append(Access(outer, write, index))
    return found


def nested_bindings(
    state: SDFGState, node: nodes.NestedSDFG, arrays: ArrayBindings, mapping: dict[str, sympy.Expr]
) -> tuple[ArrayBindings, dict[str, sympy.Expr]]:
    """The arrays and symbols ``node``'s body sees, rebound to the loop's names.

    An inner array binds only when its connector's outer subset has the inner rank; its element
    ``k`` is then outer element ``begin + k``, or ``k`` itself when the connector is the whole container
    (No-View). Anything else stays unbound and is ignored.
    """
    inner_arrays: ArrayBindings = {}
    for edge in list(state.in_edges(node)) + list(state.out_edges(node)):
        conn = edge.dst_conn if edge.dst is node else edge.src_conn
        if conn is None or edge.data.is_empty() or edge.data.data not in arrays:
            continue
        outer, outer_offset = arrays[edge.data.data]
        desc = node.sdfg.arrays.get(conn)
        begins = [as_loop_names(begin, mapping) for begin, _, _ in required(edge.data.subset).ndrange()]
        # A connector that is the whole container (No-View) is indexed in the container's own coordinates.
        if desc is not None and desc.is_equivalent(state.sdfg.arrays[edge.data.data], node.symbol_mapping):
            begins = [sympy.Integer(0)] * len(begins)
        if desc is None or len(desc.shape) != len(begins) or (outer_offset and len(outer_offset) != len(begins)):
            continue
        inner_arrays[conn] = (outer, tuple(b + outer_offset[d] if outer_offset else b for d, b in enumerate(begins)))
    inner_mapping = {name: as_loop_names(value, mapping) for name, value in node.symbol_mapping.items()}
    return inner_arrays, inner_mapping


def loop_header_names(loop: LoopRegion, mapping: dict[str, sympy.Expr]) -> set[str]:
    """The names ``loop``'s initialization, condition and update read, in the outer loop's names."""
    names: set[str] = set()
    for code in (loop.init_statement, loop.loop_condition, loop.update_statement):
        for name in code.get_free_symbols():
            names.update(s.name for s in mapping[name].free_symbols) if name in mapping else names.add(name)
    return names


def walk_body(
    sdfg_or_region: ControlFlowRegion,
    arrays: ArrayBindings,
    mapping: dict[str, sympy.Expr],
    loop_variable: str,
    body: Body,
) -> None:
    """Fill ``body`` from everything under ``sdfg_or_region``, through nested SDFGs."""
    for region in sdfg_or_region.all_control_flow_regions():
        if isinstance(region, LoopRegion) and region.loop_variable:
            # An inner loop whose bounds do not move with the outer one ranges over the same values in every outer
            # iteration, like a map's parameter; one whose bounds do move is a varying name.
            fixed = loop_variable not in loop_header_names(region, mapping)
            (body.params if fixed else body.varying).add(region.loop_variable)
    for edge in sdfg_or_region.all_interstate_edges():
        body.varying.update(edge.data.assignments.keys())
    for state in unconditional_states(sdfg_or_region):
        for edge in state.edges():
            body.accesses.extend(edge_accesses(edge, state, arrays, mapping))
        for node in state.nodes():
            if isinstance(node, nodes.MapEntry):
                body.params.update(node.map.params)
                ranges = [as_loop_names(bound, mapping) for dim in node.map.range.ndrange() for bound in dim]
                body.moving.append(any(loop_variable in {s.name for s in r.free_symbols} for r in ranges))
            elif isinstance(node, nodes.NestedSDFG):
                inner_arrays, inner_mapping = nested_bindings(state, node, arrays, mapping)
                walk_body(node.sdfg, inner_arrays, inner_mapping, loop_variable, body)


def unconditional_states(region: ControlFlowRegion) -> Iterator[SDFGState]:
    """The states under ``region`` outside every branch: a branch body may never run, so an access in it proves no
    dependence (the rule :func:`region_sites` applies to a solver witness)."""
    for block in region.nodes():
        if isinstance(block, SDFGState):
            yield block
        elif isinstance(block, ControlFlowRegion) and not isinstance(block, ConditionalBlock):
            yield from unconditional_states(block)


def carried_distance(write: Access, read: Access, loop_var: sympy.Symbol) -> sympy.Expr | None:
    """``d`` such that ``read`` at iteration ``i`` touches what ``write`` touched at ``i - d``."""
    if len(write.index) != len(read.index):
        return None
    distance = None
    for w, r in zip(write.index, read.index):
        slope = sympy.diff(w, loop_var)
        if slope != sympy.diff(r, loop_var) or not slope.is_number:
            return None
        gap = sympy.expand(w - r)
        if slope == 0:
            if gap != 0:
                return None
            continue
        step = gap / slope
        if not (step.is_number and step.is_integer) or (distance is not None and step != distance):
            return None
        distance = step
    return distance


def proven_carrying_access(loop: LoopRegion) -> str | None:
    """``RAW on aa[i - 1, j]`` when two element accesses in ``loop``'s body PROVE a carried dependence.

    ``LoopToMap`` judges a body by its propagated memlets, and a nested SDFG's are often the whole
    array -- so it can only decline. The element accesses inside still say it: a write ``A[f(i)]``
    and a read ``A[f(i - d)]`` with the same fixed indices elsewhere are ``d`` iterations apart.
    Only a constant step, map ranges fixed across iterations and indices that are otherwise
    loop-invariant or map params are admitted.
    """
    stride = loop_analysis.get_loop_stride(loop)
    if stride is None or not as_basic(stride).is_Integer or stride == 0:
        return None
    arrays = {name: (name, ()) for name, desc in loop.sdfg.arrays.items() if not isinstance(desc, data.View)}
    body = Body([], OrderedSet(), OrderedSet(), [])
    walk_body(loop, arrays, {}, loop.loop_variable, body)
    if any(body.moving):
        return None
    unknown = body.varying - body.params - OrderedSet([loop.loop_variable])
    stable = [a for a in body.accesses if not ({s.name for e in a.index for s in e.free_symbols} & unknown)]
    loop_var = symbolic.pystr_to_symbolic(loop.loop_variable)
    reads: dict[str, list[Access]] = {}
    for access in stable:
        if not access.write:
            reads.setdefault(access.array, []).append(access)
    for write in (a for a in stable if a.write):
        for read in reads.get(write.array, ()):
            distance = carried_distance(write, read, loop_var)
            if distance is None or distance == 0 or distance % stride != 0:
                continue
            kind = "RAW" if distance / stride > 0 else "WAR"
            # The pattern is its own witness: the first iteration and the one ``|distance|`` later
            witness = pattern_witness(loop, abs(distance / stride))
            return f"{kind} on {read.array}[{', '.join(str(e) for e in read.index)}]{witness}"
    return None


def pattern_witness(loop: LoopRegion, steps: Any) -> str:
    """`` (witness: i=0 and i=1)``: the first iteration of ``loop`` and the one ``steps`` iterations later, the pair a
    pattern-matched dependence (a constant distance, a carried symbol) conflicts on whenever the loop runs that far;
    empty when the loop has no constant-stride counter to name them by."""
    start, stride = loop_analysis.get_init_assignment(loop), loop_analysis.get_loop_stride(loop)
    if not loop.loop_variable or start is None or stride is None or not as_basic(stride).is_Integer:
        return ""
    first = symbolic.simplify(as_basic(start))
    second = symbolic.simplify(first + steps * as_basic(stride))
    return f" (witness: {loop.loop_variable}={first} and {loop.loop_variable}={second})"


class Site(NamedTuple):
    """An element access every execution of the loop body makes, with the iterators it sits under, outermost first."""

    access: Access
    iters: tuple[smt_dependence.IterRange, ...]


def loop_range(loop: LoopRegion, mapping: dict[str, sympy.Expr]) -> smt_dependence.IterRange | None:
    """``loop``'s counter with constant-stride inclusive bounds in the outer loop's names, or ``None``."""
    start, end, step = (
        loop_analysis.get_init_assignment(loop),
        loop_analysis.get_loop_end(loop),
        loop_analysis.get_loop_stride(loop),
    )
    if start is None or end is None or step is None or not as_basic(step).is_Integer or step == 0:
        return None
    if loop_analysis.loop_jumps(loop):
        return None  # an early exit may skip the later iteration a witness names
    return loop.loop_variable, as_loop_names(start, mapping), as_loop_names(end, mapping), step


def map_ranges(state: SDFGState, node: nodes.Node, mapping: dict[str, sympy.Expr]) -> tuple:
    """The parameters of every map around ``node`` in ``state``, outermost first."""
    ranges = []
    entry = state.entry_node(node)
    while entry is not None:
        dims = [
            (param, as_loop_names(begin, mapping), as_loop_names(end, mapping), step)
            for param, (begin, end, step) in zip(entry.map.params, entry.map.range.ndrange())
        ]
        ranges[:0] = dims
        entry = state.entry_node(entry)
    return tuple(ranges)


def copy_sites(
    edge: MultiConnectorEdge[Memlet],
    state: SDFGState,
    arrays: ArrayBindings,
    mapping: dict[str, sympy.Expr],
    iters: tuple,
) -> list[Site]:
    """Both ends of an array copy as sites, each range dimension a fresh iterator over that range.

    A copy moves every element of both subsets, so unlike a tasklet's or a library node's range, which the code
    behind it may touch only in part, it names elements that are certainly accessed.
    """
    if not (isinstance(edge.src, nodes.AccessNode) and isinstance(edge.dst, nodes.AccessNode)):
        return []
    if edge.data.is_empty() or edge.data.wcr is not None:
        return []
    sites = []
    for side, (name, subset, write) in enumerate(
        (
            (edge.src.data, edge.data.get_src_subset(edge, state), False),
            (edge.dst.data, edge.data.get_dst_subset(edge, state), True),
        )
    ):
        if name not in arrays or subset is None:
            continue
        outer, offset = arrays[name]
        if offset and len(offset) != as_range(subset).dims():
            continue
        index, extra = [], []
        for dim, (begin, end, step) in enumerate(subset.ndrange()):
            begin, end = as_loop_names(begin, mapping), as_loop_names(end, mapping)
            if begin == end:
                element = begin
            else:
                var = f"__copy{id(edge)}_{side}_{dim}"
                extra.append((var, begin, end, step))
                element = sympy.Symbol(var)
            index.append(element + offset[dim] if offset else element)
        sites.append(Site(Access(outer, write, tuple(index)), iters + tuple(extra)))
    return sites


def state_sites(
    state: SDFGState, arrays: ArrayBindings, mapping: dict[str, sympy.Expr], iters: tuple, out: list
) -> None:
    """The element accesses of ``state`` and of the nested SDFGs in it, under ``iters`` and their maps."""
    for edge in state.edges():
        scoped = edge.dst if isinstance(edge.dst, nodes.Tasklet) else edge.src
        inner = iters + map_ranges(state, scoped, mapping)
        copies = copy_sites(edge, state, arrays, mapping, inner)
        out.extend(copies or (Site(access, inner) for access in edge_accesses(edge, state, arrays, mapping)))
    for node in state.nodes():
        if isinstance(node, nodes.NestedSDFG):
            inner_arrays, inner_mapping = nested_bindings(state, node, arrays, mapping)
            region_sites(node.sdfg, inner_arrays, inner_mapping, iters + map_ranges(state, node, mapping), out)


def region_sites(
    region: ControlFlowRegion, arrays: ArrayBindings, mapping: dict[str, sympy.Expr], iters: tuple, out: list
) -> None:
    """Fill ``out`` with the unconditional accesses under ``region``.

    A branch body may never run, and a loop without a constant-stride counter has no range to name, so neither
    contributes: a witness has to be an execution that happens.
    """
    for block in region.nodes():
        if isinstance(block, SDFGState):
            state_sites(block, arrays, mapping, iters, out)
        elif isinstance(block, LoopRegion):
            counter = loop_range(block, mapping) if block.loop_variable else None
            if counter is not None:
                region_sites(block, arrays, mapping, iters + (counter,), out)
        elif isinstance(block, ControlFlowRegion) and not isinstance(block, ConditionalBlock):
            region_sites(block, arrays, mapping, iters, out)


def size_symbols(sdfg: SDFG) -> set[str]:
    """The symbols a conflict witness may choose a value for: those the argument arrays' shapes are written in, and
    those that only bound a loop (a trip count such as ``TSTEPS``), less any the program assigns.

    Either only says how much work there is: a value for it is an execution of the program at that size."""
    shapes = {
        s.name
        for desc in sdfg.arrays.values()
        if not desc.transient
        for dim in desc.shape
        for s in as_basic(dim).free_symbols
    }
    loops = [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]
    bounds = {name for loop in loops for name in loop_header_names(loop, {})}
    elsewhere = {
        str(s)
        for state in sdfg.states()
        for edge in state.edges()
        if edge.data is not None
        for s in edge.data.free_symbols
    }
    elsewhere |= {str(s) for edge in sdfg.all_interstate_edges(recursive=True) for s in edge.data.free_symbols}
    elsewhere |= {
        str(s)
        for block in sdfg.all_control_flow_blocks(recursive=True)
        if isinstance(block, ConditionalBlock)
        for condition, _ in block.branches
        if condition is not None
        for s in condition.get_free_symbols()
    }
    assigned = {name for edge in sdfg.all_interstate_edges(recursive=True) for name in edge.data.assignments}
    counters = {loop.loop_variable for loop in loops}
    return (shapes | (bounds - elsewhere)) - assigned - counters - set(sdfg.constants)


def witnessed_carrying_access(loop: LoopRegion) -> str | None:
    """``RAW on A[k, j] (witness: N=3, k=0 and k=1)`` when the solver finds two iterations of ``loop`` that touch
    one element of an argument array, choosing values for array sizes only.

    Every access involved runs on every iteration (no branch, no early exit, counted loops only), and its index
    and bounds name only iterators and sizes, so the witness is an execution of the program for that size. A
    transient is scratch the loop may privatize, so a conflict on one proves no order. A read meeting a write is
    tried before two writes, since it names the dependence a reader acts on.
    """
    counter = loop_range(loop, {}) if loop.loop_variable else None
    if counter is None:
        return None
    # The loops around ``loop`` in its SDFG, up to the first branch: below a branch the loop need not run at all.
    outer = []
    block = loop.parent_graph
    while isinstance(block, LoopRegion) and block.loop_variable:
        around = loop_range(block, {})
        if around is None:
            break
        outer.insert(0, around)
        block = block.parent_graph
    arrays = {name: (name, ()) for name, desc in loop.sdfg.arrays.items() if not isinstance(desc, data.View)}
    mapping = {name: sympy.sympify(value) for name, value in loop.sdfg.constants.items()}
    sites: list[Site] = []
    region_sites(loop, arrays, mapping, (counter,), sites)
    sites = [site for site in sites if not loop.sdfg.arrays[site.access.array].transient]
    sizes = size_symbols(loop.sdfg)
    pairs = [(w, o) for w in sites if w.access.write for o in sites if o.access.array == w.access.array]
    pairs.sort(key=lambda pair: pair[1].access.write)
    for write, other in pairs:
        for first, second in ((write, other), (other, write)):
            found = smt_dependence.find_conflict_witness(
                first.access.index, second.access.index, first.iters, second.iters, sizes, outer=tuple(outer)
            )
            if found is not None:
                return conflict_label(first, second, found, counter[0])
    return None


def site_text(site: Site) -> str:
    """``A[k, 0:k]``: the site's index, a copied range written as the range rather than its iterator."""
    ranges = {
        name: f"{begin}:{end + 1}" for name, begin, end, step in site.iters if name.startswith("__copy") and step == 1
    }
    return f"{site.access.array}[{', '.join(ranges.get(str(e), str(e)) for e in site.access.index)}]"


def conflict_label(first: Site, second: Site, witness: dict[str, int], carrier: str) -> str:
    """The carrying access with its witness: the access the dependence names, then the values."""
    kind = "WAW" if first.access.write and second.access.write else ("RAW" if first.access.write else "WAR")
    named = second if kind == "RAW" else first
    values = ", ".join(f"{name}={value}" for name, value in witness.items() if not name.startswith(carrier + "__"))
    order = f"{carrier}={witness[carrier + '__a']} and {carrier}={witness[carrier + '__b']}"
    where = f"{values}, {order}" if values else order
    return f"{kind} on {site_text(named)} (witness: {where})"


def loop_hint(loop: LoopRegion) -> str:
    """The hint text naming what kind of loop ``loop`` is."""
    reason = refusal_reason(loop)
    if reason is None:
        return SEQUENTIAL_PINNED if loop.pinned_sequential else PARALLEL
    if not reason:
        return UNSURE.format(reason=NEVER_EXAMINED)
    access = proof_of_carry(loop, reason)
    if access is not None:
        text = SEQUENTIAL_PROVEN_OVER_DOALL if over_doall_nest(loop) else SEQUENTIAL_PROVEN
        return text.format(reason=access, why=proven_why(access))
    return UNSURE.format(reason=carrying_access(reason))


def over_doall_nest(loop: LoopRegion) -> bool:
    """Whether ``loop``'s body is a Do-All nest: maps, no reduction and no loop of its own (a time loop over a
    stencil). A reduction inside gives a skew nothing to run in parallel."""
    blocks = list(loop.all_control_flow_blocks())
    states = [block for block in blocks if isinstance(block, SDFGState)]
    return (
        not any(isinstance(block, LoopRegion) for block in blocks)
        and any(isinstance(node, nodes.MapEntry) for state in states for node in state.nodes())
        and not any(edge.data.wcr is not None for state in states for edge in state.edges())
    )


def proof_of_carry(loop: LoopRegion, reason: str) -> str | None:
    """The access that proves ``loop`` carries a dependence, or ``None``: a solver witness, else a constant distance
    or a carried symbol ``LoopToMap`` named.

    The witness comes first because it names an execution as well as the access: two iterations and the sizes under
    which they touch one element. The other two prove accesses the solver cannot encode.
    """
    witnessed = witnessed_carrying_access(loop)
    if witnessed is not None:
        return witnessed
    # The element-level proof also names the direction: LoopToMap's "read-after-write conflict" is
    # any read meeting a write, so its access alone reads as RAW for a read-ahead WAR too.
    access = proven_carrying_access(loop)
    if access is None and refusal_matches_a_carried_symbol(reason):
        # A symbol one iteration hands to the next: the witness is two consecutive iterations
        access = carrying_access(reason) + pattern_witness(loop, 1)
    return access


#: Refusals whose pattern IS a carried dependence: a symbol the body updates from itself, or reads before it assigns.
#: "writes ... may overlap", "read-after-write conflict" and an access off the ``a*i+b`` write pattern only say the
#: test could not separate the iterations, so they stay unsure.
CARRIED_SYMBOL_PATTERNS = ("self-recurrent carried symbol", "carried symbol dependency")


def refusal_matches_a_carried_symbol(reason: str) -> bool:
    """Whether a ``LoopToMap`` refusal is a carried-symbol pattern, a proof rather than an unseparated pair."""
    return any(pattern in reason.lower() for pattern in CARRIED_SYMBOL_PATTERNS)


#: Appended to a parallel map's class line when its independence rests on entry traps.
ASSUMES = "; assumes {predicates}, trapped at entry"

#: ``if <violation>:`` opening every abort guard's code (see ``tasklet_utils.add_abort_guard``).
GUARD_HEAD = re.compile(r"^if (.*):\s*$", re.MULTILINE)


def entry_traps(sdfg: SDFG) -> list[tuple[str, set[str]]]:
    """``(predicate, names)`` for every abort guard in ``sdfg``: what the program assumes, and the symbols it names."""
    from dace.sdfg import tasklet_utils

    traps = []
    for node, _ in sdfg.all_nodes_recursive():
        if not tasklet_utils.is_abort_guard(node):
            continue
        head = GUARD_HEAD.search(node.code.as_string)
        if head is None:
            continue
        violation = head.group(1).strip()
        held = violation[len("not ") :].strip() if violation.startswith("not ") else f"not ({violation})"
        names = {s.name for s in as_basic(symbolic.pystr_to_symbolic(held)).free_symbols}
        traps.append((str(symbolic.pystr_to_symbolic(held)), names))
    return traps


def map_names(state: SDFGState, entry: nodes.MapEntry) -> set[str]:
    """The symbols the element accesses in ``entry``'s scope name: what independence can rest on. A bound only sets
    how many iterations there are, so the range is left out."""
    names: set[str] = set()
    for edge in state.scope_subgraph(entry).edges():
        if isinstance(edge.src, nodes.Tasklet) or isinstance(edge.dst, nodes.Tasklet):
            names |= {str(s) for s in edge.data.free_symbols}
    return names


def parallel_hint(state: SDFGState, entry: nodes.MapEntry, traps: list[tuple[str, set[str]]]) -> str:
    """``PARALLEL``, naming the entry traps whose symbols the map's element accesses use."""
    relied = sorted({held for held, names in traps if names & map_names(state, entry)})
    if not relied:
        return PARALLEL
    first, rest = PARALLEL.split("\n", 1)
    return f"{first}{ASSUMES.format(predicates=' and '.join(relied))}\n{rest}"


@xf.explicit_cf_compatible
class AnnotateLoopKinds(ppl.Pass):
    """Open every Map's and LoopRegion's ``specialization_hint`` with its class.

    Runs last in the canonicalize recipe, on the graph the rendering will see. A hint that already
    opens with a class is left alone: the pass that set it knew something this one cannot re-derive
    (that a loop is a wavefront axis, ``WavefrontSkew``). Any other hint is a note on what a pass did
    (``BreakAntiDependence``) and keeps its place below the class line this pass puts first.
    """

    CATEGORY: str = "Analysis"

    def modifies(self) -> ppl.Modifies:
        # Comments only. No pass reads a hint, so nothing needs rerunning because one appeared.
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _: dict[str, Any]) -> int | None:
        """:returns: The number of loops newly labelled, or ``None`` if none was."""
        labelled = 0
        traps = entry_traps(sdfg)
        for node, parent_graph in list(sdfg.all_nodes_recursive()):
            if not isinstance(node, (nodes.MapEntry, LoopRegion)):
                continue
            hint = (node.specialization_hint or "").strip()
            if hint.startswith(CLASS_PREFIXES):
                continue
            # A note another pass left (an anti-dependence it broke) says what it did, not what the loop is: the
            # verdict goes first, so every loop still opens with its class.
            kind = parallel_hint(parent_graph, node, traps) if isinstance(node, nodes.MapEntry) else loop_hint(node)
            node.specialization_hint = f"{kind}\n{hint}" if hint else kind
            labelled += 1
        return labelled or None
