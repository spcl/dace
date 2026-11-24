# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Name every loop in the canonical form, so the standalone rendering says what each one is."""

import copy
import re
from collections.abc import Iterator
from typing import Any, NamedTuple

import sympy

from dace import SDFG, data, dtypes, symbolic, symbolic_facts
from dace.cpf_lowering import LoopClass, Reason
from dace.frontend.operations import detect_reduction_type
from dace.memlet import Memlet
from dace.optionals import required
from dace.ordered import OrderedSet
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.narrowing import as_basic, as_range, free_symbols
from dace.sdfg.state import (
    AbstractControlFlowRegion,
    ConditionalBlock,
    ControlFlowBlock,
    ControlFlowRegion,
    LoopRegion,
    SDFGState,
)
from dace.subsets import Range, Subset
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.interstate.loop_to_map import LoopRefusal, Refusal
from dace.transformation.passes.analysis import canon_stats, loop_analysis, smt_dependence

# Three classes, named by the first word of every loop hint: ``parallel`` and ``sequential`` are settled and need no
# further dependence reasoning, ``unknown`` is the only open one.

#: The first words of every loop hint: its class (see the three below).
CLASS_PREFIXES = tuple(cls.prefix for cls in LoopClass)

#: A Map.
PARALLEL = (
    LoopClass.PARALLEL.prefix
    + Reason.INDEPENDENT.value
    + ": no iteration writes an element another iteration reads or writes"
)

#: The class line of a Map that accumulates into a container through a WCR edge: its iterations are not independent,
#: they reduce, and the operator and the accumulator are the evidence.
REDUCTION = LoopClass.PARALLEL.prefix + Reason.REDUCTION.value + ": {reductions}; the other accesses are independent"

#: Appended to a parallel loop's class line: loop-local containers the rendering declares once, outside the loop,
#: which every iteration writes before it reads them, so a parallel version needs a copy per iteration.
PRIVATE = "; private per iteration: {names}"

#: A ``CPU_Persistent`` map: one ``#pragma omp parallel`` region, not a loop over independent iterations.
TEAM = (
    LoopClass.PARALLEL.prefix
    + "thread team: every thread runs the code inside, in order; only the omp for loops in it split "
    "their iterations"
)

#: The operator a reduction line names, by ``dtypes.ReductionType``; any other type is named by its member name.
REDUCTION_SYMBOLS = {
    dtypes.ReductionType.Sum: "+",
    dtypes.ReductionType.Product: "*",
    dtypes.ReductionType.Min: "min",
    dtypes.ReductionType.Max: "max",
    dtypes.ReductionType.Logical_And: "&&",
    dtypes.ReductionType.Logical_Or: "||",
    dtypes.ReductionType.Bitwise_And: "&",
    dtypes.ReductionType.Bitwise_Or: "|",
    dtypes.ReductionType.Bitwise_Xor: "^",
}

#: Added under a multi-dimensional Map's class line: the rendering writes one nested ``for`` per dimension and
#: the comment only above the outermost, so the inner ones are named here.
MAP_DIMENSIONS = "the {count} nested loops over {params} are all parallel"

#: A loop whose carried dependence was proven.
PROVEN_PREFIX = LoopClass.SEQUENTIAL.prefix + "proven dependence"
SEQUENTIAL_PROVEN = PROVEN_PREFIX + " {reason}"

#: A loop the dependence test declined to answer, or never examined.
UNKNOWN = LoopClass.UNKNOWN.prefix + "{reason}"

#: An unproven loop over a literal trip count ``symbolic.is_small_extent`` calls small: too few iterations to share
#: out, so a parallel version is not worth a runtime check. It stays ``unknown``: the dependence test did not settle
#: it, whatever its length; ``{reason}`` is what the test left open.
SMALL_EXTENT = (
    LoopClass.UNKNOWN.prefix
    + Reason.SMALL_EXTENT.value
    + " ({trips} iterations), not proven: {reason}; the loop is too short to share out among threads, so no "
    "parallel version is worth a runtime check"
)

#: The ``UNKNOWN`` reason of a loop the dependence test never answered for; the specific cause follows a colon.
NEVER_EXAMINED = "not analyzed for dependences"
#: ... because the loop has no single iteration counter to test (a while loop).
NO_COUNTER = f"{NEVER_EXAMINED}: a while loop has no single counter to test"

#: A while loop: its condition reads what the previous trip computed, so the trips run in order.
WHILE_LOOP = (
    LoopClass.SEQUENTIAL.prefix + "while loop: each test of the condition reads what the previous iteration computed"
)
#: A loop the dependence test finds independent that an earlier pass pinned sequential anyway (the
#: fallback arm of a specialization). Rendered without a pragma, so it must not read as parallel.
SEQUENTIAL_PINNED = (
    LoopClass.SEQUENTIAL.prefix
    + "kept in order by an earlier pass; the dependence test finds its iterations independent"
)

#: The untiled skew: a sequential diagonal over a parallel front; ``alt`` names the other correct shape. ``{skew}`` is
#: :func:`skew_label` on the diagonal and :func:`front_label` on the front, both in the loops' own names: the skew
#: replaced the original counters, so the rendering has no name for them.
WAVEFRONT_DIAGONAL = (
    LoopClass.SEQUENTIAL.prefix
    + "wavefront step: each step reads what the previous step wrote; {skew}; alt: the unskewed nest"
)
WAVEFRONT_FRONT = LoopClass.PARALLEL.prefix + "one wavefront step: its points do not depend on each other; {skew}"

#: The tiled skew: a sequential tile diagonal over a parallel tile column over a sequential interior.
WAVEFRONT_TILE_DIAGONAL = (
    LoopClass.SEQUENTIAL.prefix + "wavefront of tiles {tile}: each step reads what the previous step wrote; {skew}; "
    "alt: the element wavefront or the unskewed nest"
)
WAVEFRONT_TILE_COLUMN = (
    LoopClass.PARALLEL.prefix + "the tiles {tile} of one wavefront step do not depend on each other; {skew}"
)
WAVEFRONT_TILE_INTERIOR = (
    LoopClass.SEQUENTIAL.prefix + "inside one wavefront tile {tile}: the original order carries the dependence"
)


def skew_label(a: int, b: int, step: str, unit: str = "") -> str:
    """What step ``step`` of the diagonal runs: ``step t runs the points with row + column = t of the original nest``;
    ``unit`` is ``"tile "`` for a diagonal of tiles."""

    def term(coeff: int, name: str) -> str:
        return name if abs(coeff) == 1 else f"{abs(coeff)}*{name}"

    row, column = f"{unit}row", f"{unit}column"
    lead = f"-{term(a, row)}" if a < 0 else term(a, row)
    what = "tiles" if unit else "points"
    sign = "-" if b < 0 else "+"
    return f"step {step} runs the {what} with {lead} {sign} {term(b, column)} = {step} of the original nest"


def front_label(step: str, front: str, along: str) -> str:
    """Which coordinate the front ``front`` of step ``step`` walks: ``p walks the column of step t``."""
    return f"{front} walks the {along} of step {step}"


def tile_label(bi: int, bj: int) -> str:
    """The tile extent the diagonal walks -- the number that trades kernel launches for barriers."""
    return f"[{bi}x{bj}]"


def refusal_reason(loop: LoopRegion, session: Any = None) -> LoopRefusal | str | None:
    """Why ``LoopToMap`` would refuse ``loop`` (its :class:`LoopRefusal`), ``None`` if it would accept, or the text of
    why it was never examined; ``session`` the :class:`ProbeSession` the caller's probes share."""
    if not loop.loop_variable:
        return NO_COUNTER
    from dace.transformation.interstate.loop_to_map import LoopToMap
    from dace.transformation.passes.parallelization.parallelize_loops import ParallelizeLoops

    probe = LoopToMap()
    # A short loop is analyzed too: its trip count decides whether to fork, never its class.
    probe.keep_small_loops_sequential = False
    pinned = loop.pinned_sequential
    loop.pinned_sequential = False
    try:
        applicable = ParallelizeLoops().accepts(loop, probe, session=session)
    except Exception as error:  # pylint: disable=broad-exception-caught
        # A comment must not be able to fail the compilation that asked for it.
        return f"{NEVER_EXAMINED}: the dependence test failed ({type(error).__name__}), a compiler bug to report"
    finally:
        loop.pinned_sequential = pinned
    if applicable:
        return None
    if probe.last_refusal is None:
        return f"{NEVER_EXAMINED}: the dependence test refused without a reason, a compiler bug to report"
    return probe.last_refusal


#: What each proven conflict means for the order, in the words a reader of the loop needs.
CONFLICT_MEANING = {
    "RAW": "an iteration reads what an earlier one wrote",
    "WAR": "an iteration overwrites what an earlier one still has to read",
    "WAW": "two iterations write the same element and the last write must win",
}


#: A carrying access: its conflict kind, the element, and the example pair of iterations if one was found.
CARRYING_ACCESS = re.compile(r"^(RAW|WAR|WAW) on (.+?)( \(example.*\))?$")


def proven_reason(access: str) -> str:
    """``on a[j]: <what the order protects> (example ...)`` from ``WAW on a[j] (example ...)``; another access
    (a carried scalar) as ``: <access>``."""
    found = CARRYING_ACCESS.match(access)
    if found is None:
        return f": {access}"
    kind, element, example = found.groups()
    return f"on {element}: {CONFLICT_MEANING[kind]}{example or ''}"


def carrying_access(refusal: LoopRefusal | str) -> str:
    """``refusal`` as the ACCESS that carries it: ``RAW on aa[_loop_it_1 - 1, 8:LEN_2D]``; any other refusal as its
    message, whole."""
    if isinstance(refusal, str):
        return refusal
    if refusal.kind is Refusal.CARRIED_DEPENDENCE and refusal.container is not None and refusal.reads:
        return f"RAW on {refusal.container}[{refusal.reads[0]}]"
    if refusal.kind is Refusal.CARRIED_SYMBOL:
        return f"{carried_symbol_text(refusal)} carries a value from one iteration to the next"
    return refusal.message


#: ``LoopToMap``'s message for a symbol an iteration assigns from its own previous value.
SELF_RECURRENT = re.compile(r"\(assignment (.+)\)$")


def carried_symbol_text(refusal: LoopRefusal) -> str:
    """The carried symbol of a ``CARRIED_SYMBOL`` refusal in the loop's words: its assignment (``pos = pos + 1``),
    else the symbols an iteration reads before it assigns them."""
    found = SELF_RECURRENT.search(refusal.message)
    if found is not None:
        return found.group(1)
    return ", ".join(sorted(refusal.symbols)) or "a scalar"


#: A loop whose order the program itself requires, by the body's structure rather than an access pair: a call with
#: an ordered side effect, or an exit after which no iteration may run.
ORDERED_CALL = LoopClass.SEQUENTIAL.prefix + "ordered call: {name} must run in iteration order"
EARLY_EXIT = LoopClass.SEQUENTIAL.prefix + "early exit ({statement}): iterations after the exit must not run"

#: ``LoopToMap``'s ordered-side-effect message starts with the node's name.
ORDERED_EFFECT = re.compile(r"^(.+?) has an ordered side effect")
#: ``LoopToMap``'s unsupported-body message names the block type last.
BODY_BLOCK = re.compile(r"contains a (\w+)")

#: The longest index a reason quotes; a longer one (a tiled bound) is left out, the array name alone stays.
SHORT_INDEX = 40


def order_requirement(refusal: LoopRefusal) -> str | None:
    """The class line of a loop whose order the program itself requires, or ``None``: an ordered call, or an exit that
    ends the loop early."""
    if refusal.kind is Refusal.ORDERED_SIDE_EFFECT:
        found = ORDERED_EFFECT.search(refusal.message)
        name = found.group(1) if found is not None else "a call in the body"
        return ORDERED_CALL.format(name=name)
    if refusal.kind is Refusal.UNSUPPORTED_BODY:
        found = BODY_BLOCK.search(refusal.message)
        block = found.group(1) if found is not None else ""
        if block in ("BreakBlock", "ReturnBlock"):
            return EARLY_EXIT.format(statement="break" if block == "BreakBlock" else "return")
    return None


def quoted(container: str | None, subset: Any) -> str:
    """``A[i + 1]``, or ``A`` alone when the index is longer than :data:`SHORT_INDEX`."""
    if container is None:
        return "an array"
    index = str(subset) if subset is not None else ""
    return f"{container}[{index}]" if index and len(index) <= SHORT_INDEX else container


#: The tail of an ``unknown`` reason without a condition of its own: what settles it, and what goes wrong otherwise.
UNSHOWN_TAIL = (
    "parallel if no iteration touches an element another one writes; otherwise an iteration may read or overwrite "
    "another one's element, so keep the order"
)


def unproven_reason(refusal: LoopRefusal, loop: LoopRegion | None = None) -> str:
    """The reason of an ``unknown`` loop, one template for every refusal: what the test could not show, naming the
    array or symbol; ``parallel if`` the condition that would settle it; ``otherwise`` what goes wrong.
    ``loop`` tells a write that stays on the same elements in every iteration from one the test could not show
    distinct: only a subset made of values fixed before the loop (a constant, a size) is the same in every iteration,
    an index loaded from an array is not."""
    kind = refusal.kind
    written = quoted(refusal.container, refusal.writes[0] if refusal.writes else None)
    read = quoted(refusal.container, refusal.reads[0] if refusal.reads else None)
    later_wins = "otherwise a later iteration's write must win, so keep the order"
    container = refusal.container or "an array"
    if (
        kind is Refusal.NON_UNIQUE_WRITE
        and loop is not None
        and refusal.writes
        and all(free_names(w) <= fixed_symbols(loop) for w in refusal.writes)
    ):
        return (
            f"every iteration writes the same elements of {container} (the write to {written} does not depend "
            f"on the loop counter {loop.loop_variable}); parallel only if {container} is private per iteration or "
            f"the update is a reduction; {later_wins}"
        )
    if kind is Refusal.NON_UNIQUE_WRITE:
        return (
            f"the solver could not show that each iteration writes a different element of {written}; parallel if no "
            f"two iterations write the same element (an index array without repeated values); {later_wins}"
        )
    if kind is Refusal.DYNAMIC_WRITE:
        return (
            f"a conditional write to {written} at an index the loop counter does not fix may hit one element in two "
            f"iterations; parallel if no two iterations write the same element; {later_wins}"
        )
    if kind is Refusal.OVERLAPPING_WRITES:
        return (
            f"the dependence test could not show that writes to {refusal.container or 'an array'} from different "
            f"iterations never overlap; parallel if they are disjoint; {later_wins}"
        )
    if kind is Refusal.CARRIED_DEPENDENCE:
        return (
            f"the loop reads {read} and also writes {container}: the dependence test could not show that an "
            "earlier iteration never writes an element this read touches; parallel if none does; otherwise that read "
            "must see the earlier iteration's value, so keep the order"
        )
    if kind is Refusal.CARRIED_SYMBOL:
        return (
            f"{carried_symbol_text(refusal)} may carry a value from one iteration to the next (a branch may skip "
            "it); parallel if every iteration assigns it before reading it; otherwise parallel only as a scan"
        )
    if kind is Refusal.LIVE_OUT_SYMBOL:
        names = ", ".join(sorted(refusal.symbols)) or "a scalar"
        return (
            f"{names} is read after the loop; parallel if the last iteration's value is kept (lastprivate); "
            "otherwise the read after the loop sees whichever iteration finished last"
        )
    return unexamined_reason(refusal, written)


def unexamined_reason(refusal: LoopRefusal, written: str) -> str:
    """:func:`unproven_reason` for the refusals that stop the test before any dependence question."""
    kind = refusal.kind
    if kind is Refusal.UNANALYZABLE_RANGE:
        return (
            f"{NEVER_EXAMINED}: the loop bounds or step could not be analyzed; parallel if the test passes once the "
            "bounds are integers fixed before the loop; otherwise keep the order, since nothing is known about the "
            "iterations"
        )
    if kind is Refusal.UNSUPPORTED_BODY:
        found = BODY_BLOCK.search(refusal.message)
        block = found.group(1) if found is not None else "block"
        if block == "ContinueBlock":
            return (
                f"{NEVER_EXAMINED}: the body skips the rest of an iteration (continue); parallel if, rewritten as an "
                "if, the test passes; otherwise keep the order"
            )
        if block == "BreakBlock":
            return (
                f"{NEVER_EXAMINED}: the body leaves the loop early (break); parallel only as a first-hit search; "
                "otherwise iterations after the exit would run"
            )
        return f"{NEVER_EXAMINED}: the body holds a {block}; {UNSHOWN_TAIL}"
    if kind is Refusal.ORDERED_SIDE_EFFECT:
        return (
            "the body has a side effect whose order counts (a call, a message); parallel if that order does not "
            "matter; otherwise the effects happen out of order"
        )
    # PINNED and any later kind: name the array, never the refusal's own text (it may hold a whole subset)
    where = f" on {written}" if refusal.container else ""
    return f"the dependence test refused the loop{where}; {UNSHOWN_TAIL}"


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
    """``expr`` rewritten through ``mapping`` (inner name -> outer expression) and reparsed, so one name is one symbol
    instance whichever scope minted it."""
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
    # A dynamic memlet may not happen, like an access under a branch: it proves no dependence.
    if memlet.is_empty() or memlet.wcr is not None or memlet.dynamic:
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
    """The arrays and symbols ``node``'s body sees, rebound to the loop's names."""
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
    for code in loop.get_meta_codeblocks():
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
    dependence by pattern (a solver witness takes the branch condition along instead, see :func:`region_sites`)."""
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
    """``RAW on aa[i - 1, j]`` when two element accesses in ``loop``'s body PROVE a carried dependence."""
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
    return f" (example: iterations {loop.loop_variable}={first} and {loop.loop_variable}={second})"


class Site(NamedTuple):
    """An element access of the loop body, with the iterators it sits under, outermost first."""

    access: Access
    iters: tuple[smt_dependence.IterRange, ...]
    #: The node that reads or writes the element (a copy's destination) and its state: a write the read flows into
    #: happens after it.
    node: nodes.Node | None = None
    state: SDFGState | None = None
    #: The branch conditions under which the access happens, in the loop's names; ``None`` when one of them cannot be
    #: written there, so the access proves no dependence.
    guard: smt_dependence.Guard | None = ()


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
    seed: int,
) -> list[Site]:
    """Both ends of an array copy as sites, each range dimension a fresh iterator over that range."""
    product = type(edge.dst).__name__ in ("Dot", "MatMul") and isinstance(edge.dst, nodes.LibraryNode)
    if not (isinstance(edge.src, nodes.AccessNode) and (product or isinstance(edge.dst, nodes.AccessNode))):
        return []
    if edge.data.is_empty() or edge.data.wcr is not None or edge.data.dynamic:
        return []
    sites = []
    sides = [(edge.src.data, edge.data.get_src_subset(edge, state), False)]
    if not product:
        sides.append((edge.dst.data, edge.data.get_dst_subset(edge, state), True))
    for side, (name, subset, write) in enumerate(sides):
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
                var = f"{COPY_ITERATOR}{seed}_{side}_{dim}"
                extra.append((var, begin, end, step))
                element = sympy.Symbol(var)
            index.append(element + offset[dim] if offset else element)
        sites.append(Site(Access(outer, write, tuple(index)), iters + tuple(extra), edge.dst, state))
    return sites


def state_sites(
    state: SDFGState,
    arrays: ArrayBindings,
    mapping: dict[str, sympy.Expr],
    iters: tuple,
    out: list,
    guard: smt_dependence.Guard | None = (),
    terms: dict[str, sympy.Basic] | None = None,
) -> None:
    """The element accesses of ``state`` and of the nested SDFGs in it, under ``iters`` and their maps; ``guard`` and
    ``terms`` as :func:`region_sites`."""
    terms = mapping if terms is None else terms
    for edge in state.edges():
        scoped = edge.dst if isinstance(edge.dst, nodes.Tasklet) else edge.src
        inner = iters + map_ranges(state, scoped, mapping)
        copies = copy_sites(edge, state, arrays, mapping, inner, len(out))
        node = scoped if isinstance(scoped, nodes.Tasklet) else edge.dst
        found = copies or [Site(a, inner, node, state) for a in edge_accesses(edge, state, arrays, mapping)]
        out.extend(site._replace(guard=guard) for site in found)
    for node in state.nodes():
        if isinstance(node, nodes.NestedSDFG):
            inner_arrays, inner_mapping = nested_bindings(state, node, arrays, mapping)
            inner_terms = {
                name: in_loop_terms(str(value), arrays, terms) for name, value in node.symbol_mapping.items()
            }
            inner_terms = {name: value for name, value in inner_terms.items() if value is not None}
            inner_iters = iters + map_ranges(state, node, mapping)
            region_sites(node.sdfg, inner_arrays, inner_mapping, inner_iters, out, guard, inner_terms)


def region_sites(
    region: ControlFlowRegion,
    arrays: ArrayBindings,
    mapping: dict[str, sympy.Expr],
    iters: tuple,
    out: list,
    guard: smt_dependence.Guard | None = (),
    terms: dict[str, sympy.Basic] | None = None,
) -> None:
    """Fill ``out`` with the accesses under ``region``, each with the branch conditions it runs under on top of
    ``guard``. ``terms`` is ``mapping`` for branch conditions only (``None``: ``mapping``): it also holds the values
    interstate edges assign, which may be conditions themselves (``c = flag[k] != 0``)."""
    terms = mapping if terms is None else terms
    known = assigned_values(region, arrays, terms)
    for block in region.nodes():
        scope = {**terms, **known.get(block, {})}
        scope = {name: value for name, value in scope.items() if value is not None}
        if isinstance(block, SDFGState):
            state_sites(block, arrays, mapping, iters, out, guard, scope)
        elif isinstance(block, LoopRegion):
            counter = loop_range(block, mapping) if block.loop_variable else None
            # A symbol the inner loop's body assigns holds the previous inner iteration's value at its top
            body = {name: value for name, value in scope.items() if name not in assigned_names(block)}
            if counter is not None:
                region_sites(block, arrays, mapping, iters + (counter,), out, guard, body)
        elif isinstance(block, ConditionalBlock):
            # A branch runs when its condition holds and every earlier branch's fails
            failed: smt_dependence.Guard | None = ()
            for condition, branch in block.branches:
                term = None if condition is None else in_loop_terms(condition.as_string, arrays, scope)
                taken = failed if condition is None else (None if term is None else failed + ((term, True),))
                inner = None if guard is None or taken is None else guard + taken
                region_sites(branch, arrays, mapping, iters, out, inner, scope)
                failed = None if failed is None or term is None else failed + ((term, False),)
        elif isinstance(block, ControlFlowRegion):
            region_sites(block, arrays, mapping, iters, out, guard, scope)


def assigned_names(block: ControlFlowBlock) -> set[str]:
    """The symbols an interstate edge under ``block`` assigns, nested SDFGs excluded: their names are their own."""
    if not isinstance(block, AbstractControlFlowRegion):
        return set()
    return {name for edge in block.all_interstate_edges() for name in edge.data.assignments}


def assigned_values(
    region: ControlFlowRegion, arrays: ArrayBindings, mapping: dict[str, sympy.Expr]
) -> dict[ControlFlowBlock, dict[str, sympy.Basic | None]]:
    """For each block of ``region``, the symbols an interstate edge of ``region`` assigns before it, in the loop's
    names: a value every path to the block sets, or ``None`` when the paths disagree or the value reads something
    the loop cannot name. A symbol missing keeps the value it had when ``region`` began."""
    if not region.edges():
        return {}
    order = list(sdutil.dfs_topological_sort(region, [region.start_block]))
    if len(order) != region.number_of_nodes():  # a cycle: nothing is known along it
        return {block: dict.fromkeys(assigned_names(region)) for block in region.nodes()}
    known: dict[ControlFlowBlock, dict[str, sympy.Basic | None]] = {region.start_block: {}}
    for block in order:
        incoming = []
        for edge in region.in_edges(block):
            if edge.src not in known:
                return {node: dict.fromkeys(assigned_names(region)) for node in region.nodes()}
            values = {name: value for name, value in known[edge.src].items() if name not in assigned_names(edge.src)}
            values.update(dict.fromkeys(assigned_names(edge.src)))
            scope = {name: value for name, value in {**mapping, **known[edge.src]}.items() if value is not None}
            for name, text in edge.data.assignments.items():
                values[name] = in_loop_terms(text, arrays, scope)
            incoming.append(values)
        if incoming:
            names = {name for values in incoming for name in values}
            known[block] = {
                name: incoming[0][name] if all(v.get(name, ...) == incoming[0].get(name) for v in incoming) else None
                for name in names
            }
    return known


def in_loop_terms(text: str, arrays: ArrayBindings, mapping: dict[str, sympy.Expr]) -> sympy.Basic | None:
    """``text``, a branch condition or an assigned value, in the loop's names: each array element it reads named in the
    loop SDFG's container; ``None`` when it reads an array the loop has no name for."""
    expr = symbolic.pystr_to_symbolic(text)
    if not isinstance(expr, sympy.Basic):
        return sympy.sympify(expr)
    names = {s: mapping[s.name] for s in expr.free_symbols if s.name in mapping}
    elements = {}
    for element in expr.atoms(symbolic.Subscript):
        name = str(element.args[0])
        if name not in arrays or (arrays[name][1] and len(arrays[name][1]) != len(element.args) - 1):
            return None
        outer, offset = arrays[name]
        index = [as_basic(i).xreplace(names) + (offset[d] if offset else 0) for d, i in enumerate(element.args[1:])]
        elements[element] = element.func(sympy.Symbol(outer), *index)
    return expr.xreplace({**names, **elements})


def size_symbols(sdfg: SDFG) -> set[str]:
    """The symbols a conflict witness may choose a value for: those the argument arrays' shapes are written in, and
    those that only bound a loop (a trip count such as ``TSTEPS``), less any the program assigns."""
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


def fixed_symbols(loop: LoopRegion) -> set[str]:
    """The symbols of ``loop``'s SDFG that nothing inside ``loop`` assigns: a value set before the loop runs or passed
    by the caller (a column index), the same in every iteration."""
    assigned = {name for edge in loop.all_interstate_edges(recursive=True) for name in edge.data.assignments}
    loops = [r for r in loop.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]
    return set(loop.sdfg.symbols) - assigned - {r.loop_variable for r in loops} - set(loop.sdfg.constants)


def is_scratch(sdfg: SDFG, name: str) -> bool:
    """Whether container ``name`` of ``sdfg`` is scratch: a transient here, or an argument of a nested SDFG that every
    caller binds to scratch."""
    desc = sdfg.arrays.get(name)
    if desc is None:
        return False
    if desc.transient:
        return True
    node, state = sdfg.parent_nsdfg_node, sdfg.parent
    if node is None or not isinstance(state, SDFGState) or node not in state.nodes():
        return False
    outer = [state.memlet_path(e)[0].src for e in state.in_edges(node) if e.dst_conn == name]
    outer += [state.memlet_path(e)[-1].dst for e in state.out_edges(node) if e.src_conn == name]
    if not outer or not all(isinstance(n, nodes.AccessNode) for n in outer):
        return False
    return all(is_scratch(state.sdfg, n.data) for n in outer)


def declared_conditions(loop: LoopRegion, traps: list[tuple[str, set[str]]]) -> smt_dependence.Guard:
    """What ``loop``'s program declares of its symbols, as conditions a witness must satisfy: the entry ``abort()``
    guards that reach ``loop``'s function (``traps``), and the relations assumed of its symbols."""
    relations = {
        symbolic_facts.RelationKind.LT: sympy.Lt,
        symbolic_facts.RelationKind.LE: sympy.Le,
        symbolic_facts.RelationKind.EQ: sympy.Eq,
        symbolic_facts.RelationKind.NE: sympy.Ne,
    }
    declared = [(symbolic.pystr_to_symbolic(held), True) for held, names in traps]
    declared += [(relations[r.kind](r.lhs, r.rhs), True) for r in loop.sdfg.facts().known() if r.kind in relations]
    return tuple(declared)


def witnessed_carrying_access(loop: LoopRegion, traps: list[tuple[str, set[str]]] | None = None) -> str | None:
    """``RAW on A[k, j] (witness: N=3, k=0 and k=1)`` when the solver finds two iterations of ``loop`` that touch one
    element of an argument array, choosing values for array sizes only; a parameter the caller fixes may appear
    when the conflict holds for every in-bounds value of it."""
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
    sites = [site._replace(guard=input_guard(loop, site.guard)) for site in sites]
    scratch = {site.access.array for site in sites if is_scratch(loop.sdfg, site.access.array)}
    # A read of scratch that no write of its own iteration reaches first sees an earlier iteration's value, which no
    # private copy has: the one conflict on scratch that proves an order. Every write must be a site for that.
    unseen = unseen_writes(loop)
    sites = [site for site in sites if site.access.array not in scratch & unseen]
    sizes = size_symbols(loop.sdfg)
    fixed = fixed_symbols(loop) - sizes
    declared = declared_conditions(loop, traps or [])
    pairs = [(w, o) for w in sites if w.access.write for o in sites if o.access.array == w.access.array]
    pairs.sort(key=lambda pair: pair[1].access.write)
    for write, other in pairs:
        for first, second in ((write, other), (other, write)):
            if first.guard is None or second.guard is None:
                continue
            flow = first.access.write and not second.access.write
            if write.access.array in scratch and not flow:
                continue
            if flow and write.access.array not in scratch and rewritten_in_iteration(second, sites):
                continue
            # A write whose condition cannot be written is taken to happen: that only refuses more witnesses
            kills = tuple(
                (site.access.index, site.iters, flows_into(second, site), site.guard or ())
                for site in sites
                if write.access.array in scratch and site.access.write and site.access.array == write.access.array
            )
            with canon_stats.timed("z3", loop):
                found = smt_dependence.find_conflict_witness(
                    first.access.index,
                    second.access.index,
                    first.iters,
                    second.iters,
                    sizes,
                    outer=tuple(outer),
                    shape=array_extent(loop.sdfg, first.access.array, len(first.access.index)),
                    fixed=fixed,
                    kills=kills,
                    guards=(first.guard, second.guard),
                    assumptions=declared,
                )
            if found is not None:
                return conflict_label(first, second, found, counter[0])
    return None


def input_guard(loop: LoopRegion, guard: smt_dependence.Guard | None) -> smt_dependence.Guard | None:
    """``guard`` when every array it reads is integer input nothing under ``loop`` writes, so a witness may choose its
    elements; else ``None``."""
    if not guard:
        return guard
    read = {str(e.args[0]) for term in guard for e in as_basic(term[0]).atoms(symbolic.Subscript)}
    written = {
        node.data
        for block in loop.all_control_flow_blocks()
        if isinstance(block, SDFGState)
        for node in block.data_nodes()
        if block.in_degree(node) > 0
    }
    for name in read:
        desc = loop.sdfg.arrays.get(name)
        # A solver integer is a value of an integer or boolean element, not always of a float one
        if name in written or desc is None or desc.dtype not in dtypes.INTEGER_TYPES:
            return None
    return guard


def flows_into(read: Site, write: Site) -> bool:
    """Whether ``write`` is downstream of ``read`` in one state under the same iterators, so in every instance of the
    two the read happens first."""
    return read.state is write.state and read.iters == write.iters and write.node in read.state.bfs_nodes(read.node)


def unseen_writes(region: ControlFlowRegion) -> set[str]:
    """The containers of ``region``'s SDFG that some write under ``region`` stores to without a site
    :func:`region_sites` sees: a write that may not happen, a library node writing, a tasklet writing more than one
    element, a write in a loop without a constant-stride counter, or such a write inside a nested SDFG (named by the
    container the nested SDFG writes out to). A write under a branch is a site with its condition."""
    found: set[str] = set()
    for block in region.all_control_flow_blocks():
        if isinstance(block, LoopRegion) and block.loop_variable and loop_range(block, {}) is None:
            found |= {n.data for s in block.all_states() for n in s.data_nodes() if s.in_degree(n) > 0}
        if not isinstance(block, SDFGState):
            continue
        for edge in block.edges():
            if edge.data.is_empty() or isinstance(
                edge.dst, (nodes.Tasklet, nodes.LibraryNode, nodes.NestedSDFG, nodes.EntryNode)
            ):
                continue  # a read
            many = isinstance(edge.src, nodes.Tasklet) and edge.data.subset.num_elements() != 1
            if isinstance(edge.src, nodes.LibraryNode) or edge.data.dynamic or many:
                found.add(edge.dst.data if isinstance(edge.dst, nodes.AccessNode) else edge.data.data)
        for node in block.nodes():
            if isinstance(node, nodes.NestedSDFG):
                inner = unseen_writes(node.sdfg)
                found |= {e.data.data for e in block.out_edges(node) if e.src_conn in inner and not e.data.is_empty()}
    return found


def array_extent(sdfg: SDFG, name: str, dims: int) -> tuple[sympy.Expr | None, ...]:
    """The shape of ``name`` per dimension, ``()`` when its rank is not ``dims``."""
    desc = sdfg.arrays.get(name)
    if desc is None or len(desc.shape) != dims:
        return ()
    return tuple(as_basic(extent) for extent in desc.shape)


def rewritten_in_iteration(read: Site, sites: list[Site]) -> bool:
    """Whether some write of the same iteration stores the very element ``read`` loads."""
    return any(
        site.access.write
        and site.access.array == read.access.array
        and site.iters == read.iters
        and len(site.access.index) == len(read.access.index)
        and all(
            sympy.simplify(sympy.sympify(a) - sympy.sympify(b)) == 0
            for a, b in zip(site.access.index, read.access.index)
        )
        for site in sites
    )


def site_text(site: Site) -> str:
    """``A[k, 0:k]``: the site's index, each iterator inside the carrying loop written as its range, since the
    rendering may rename it."""
    return f"{site.access.array}[{', '.join(index_text(e, site.iters[1:]) for e in site.access.index)}]"


def index_text(expr: sympy.Expr, inner: tuple[smt_dependence.IterRange, ...]) -> str:
    """``expr`` with the one unit-step iterator of ``inner`` it adds, as a range: ``j + 1`` -> ``1:N + 1``; a copy's
    iterator, which the rendering has no name for, always as its range."""
    found = [(name, begin, end) for name, begin, end, step in inner if step == 1 and name in free_names(expr)]
    copies = [it for it in inner if it[0].startswith(COPY_ITERATOR) and it[0] in free_names(expr)]
    if len(found) != 1 or copies and found[0][0] != copies[0][0]:
        return copy_range_text(expr, copies) if copies else str(expr)
    name, begin, end = found[0]
    var = next(s for s in expr.free_symbols if str(s) == name)
    if sympy.diff(expr, var) != 1:
        return copy_range_text(expr, copies) if copies else str(expr)
    return f"{expr.subs(var, begin)}:{expr.subs(var, end) + 1}"


#: The prefix of the iterator :func:`copy_sites` gives each range dimension of an array copy.
COPY_ITERATOR = "__copy"


def copy_range_text(expr: sympy.Expr, copies: list[smt_dependence.IterRange]) -> str:
    """``expr`` with its first copy iterator written as the range it covers, ``lo:hi + 1:stride``, and any other one
    as ``lo..hi``: the rendering has no name for either."""
    name, begin, end, step = copies[0]
    var = next(s for s in expr.free_symbols if str(s) == name)
    slope = sympy.diff(expr, var) * step
    first, last = expr.subs(var, begin), expr.subs(var, end)
    text = f"{first}:{last + 1}" if slope == 1 else f"{first}:{last + 1}:{slope}"
    if not slope.is_number:
        text = f"{first}..{last}"
    for other in copies[1:]:
        text = text.replace(other[0], f"({other[1]}..{other[2]})")
    return text


def free_names(obj: sympy.Basic | Subset) -> OrderedSet[str]:
    return OrderedSet(str(s) for s in obj.free_symbols)


def conflict_label(first: Site, second: Site, witness: dict[str, int], carrier: str) -> str:
    """The carrying access with its witness: the access the dependence names, then the values."""
    kind = "WAW" if first.access.write and second.access.write else ("RAW" if first.access.write else "WAR")
    named = second if kind == "RAW" else first
    values = ", ".join(f"{name}={value}" for name, value in witness.items() if not name.startswith(carrier + "__"))
    order = f"iterations {carrier}={witness[carrier + '__a']} and {carrier}={witness[carrier + '__b']}"
    where = f"{values}, {order}" if values else order
    return f"{kind} on {site_text(named)} (example the solver found: {where})"


def loop_hint(loop: LoopRegion, session: Any = None, traps: list[tuple[str, set[str]]] | None = None) -> str:
    """The hint text naming what kind of loop ``loop`` is; ``session`` as :func:`refusal_reason`; ``traps`` the entry
    ``abort()`` guards its function rests on, which a witness must satisfy."""
    reason = refusal_reason(loop, session)
    if reason is None:
        return SEQUENTIAL_PINNED if loop.pinned_sequential else parallel_loop_hint(loop)
    if reason == NO_COUNTER:
        return WHILE_LOOP
    if isinstance(reason, str):
        return unproven_hint(loop, f"{reason}; {UNSHOWN_TAIL}")
    required = order_requirement(reason)
    if required is not None:
        return required
    access = proof_of_carry(loop, reason, traps)
    if access is not None:
        return SEQUENTIAL_PROVEN.format(reason=proven_reason(access)) + carrier_note(loop, access)
    return unproven_hint(loop, unproven_reason(reason, loop))


def unproven_hint(loop: LoopRegion, reason: str) -> str:
    """``SMALL_EXTENT`` for a loop over a few literal iterations, else ``UNKNOWN``."""
    counter = loop_range(loop, {}) if loop.loop_variable else None
    if counter is not None:
        trips = sympy.floor((as_basic(counter[2]) - as_basic(counter[1])) / as_basic(counter[3])) + 1
        if trips.is_number and trips > 0 and symbolic.is_small_extent(trips, loop.sdfg.facts()):
            return SMALL_EXTENT.format(trips=trips, reason=reason)
    return UNKNOWN.format(reason=reason)


def recurrence_before_rewrite(loop: LoopRegion) -> str | None:
    """The ``sequential`` label of ``loop`` when its body as it stands proves a carried dependence, naming only the
    array: a pass about to rewrite the body (a wavefront skew) records it, because the rewritten body's accesses are
    guarded and prove nothing, and its counters are gone."""
    reason = refusal_reason(loop)
    if reason is None or isinstance(reason, str):
        return None
    found = re.match(r"(RAW|WAR|WAW) on (\w+)\[", proof_of_carry(loop, reason) or "")
    if found is None:
        return None
    meaning = CONFLICT_MEANING[found.group(1)]
    return f"{PROVEN_PREFIX} on {found.group(2)}: {meaning}, shown before the loops inside were skewed"


def parallel_loop_hint(loop: LoopRegion) -> str:
    """``PARALLEL`` for a loop ``LoopToMap`` accepts, or ``REDUCTION`` when its body accumulates through a WCR write
    two iterations may share; either names the loop-local containers each iteration writes."""
    states = list(loop.states())
    private = loop_private(loop, states)
    reductions = OrderedSet(
        reduction_text(wcr, name)
        for name, wcr, subset in wcr_writes(states)
        if name not in private and not distinct_per_iteration(subset, [loop.loop_variable], moving_names(loop))
    )
    first = REDUCTION.format(reductions=" and ".join(sorted(reductions))) if reductions else PARALLEL
    return first + PRIVATE.format(names=", ".join(private)) if private else first


def moving_names(loop: LoopRegion) -> list[str]:
    """The names that change within one iteration of ``loop``: inner loop counters, map parameters and interstate
    assignments at any depth. A write index that adds one of them to the counter is not shown injective."""
    names = OrderedSet(name for edge in loop.all_interstate_edges(recursive=True) for name in edge.data.assignments)
    for region in loop.all_control_flow_regions(recursive=True):
        if isinstance(region, LoopRegion) and region is not loop and region.loop_variable:
            names.add(region.loop_variable)
    for state in loop.states():
        for node in state.nodes():
            if isinstance(node, nodes.MapEntry):
                names.update(node.map.params)
    return list(names)


def distinct_per_iteration(subset: Subset | None, params: list[str], varying: list[str] | None = None) -> bool:
    """Whether two iterations over ``params`` always write different elements of ``subset``: each parameter indexes
    a dimension of its own, ``c * p + offset`` with a constant ``c != 0`` and no other parameter or ``varying`` name."""
    if subset is None:
        return False
    points = [as_basic(begin) for begin, end, step in subset.ndrange() if begin == end]
    moving = [*params, *(varying or [])]
    return all(any(own_axis(point, param, moving) for point in points) for param in params)


def own_axis(point: sympy.Expr, param: str, moving: list[str]) -> bool:
    """Whether ``point`` is ``c * param + offset``, ``c`` a nonzero constant and ``offset`` free of ``moving``."""
    named = [s for s in point.free_symbols if str(s) in moving]
    if len(named) != 1 or str(named[0]) != param:
        return False
    slope = sympy.diff(point, named[0])
    return slope.is_number and slope != 0


def reduction_text(wcr: str, name: str) -> str:
    """``+ into s``."""
    kind = detect_reduction_type(wcr)
    return f"{REDUCTION_SYMBOLS.get(kind, kind.name.lower())} into {name}"


def wcr_writes(states: list[SDFGState]) -> list[tuple[str, str, Subset]]:
    """``(container, wcr, subset)`` of every accumulating write in ``states``; a nested SDFG's as the memlet that
    leaves it."""
    found = []
    for state in states:
        found.extend(
            (edge.dst.data, edge.data.wcr, edge.data.get_dst_subset(edge, state))
            for edge in state.edges()
            if isinstance(edge.dst, nodes.AccessNode) and not edge.data.is_empty() and edge.data.wcr is not None
        )
        for node in state.nodes():
            if not isinstance(node, nodes.NestedSDFG):
                continue
            inner = {name: wcr for name, wcr, subset in wcr_writes(list(node.sdfg.states()))}
            for edge in state.out_edges(node):
                if edge.src_conn in inner and not edge.data.is_empty():
                    outer = state.memlet_path(edge)[-1]
                    found.append((outer.data.data, inner[edge.src_conn], outer.data.get_dst_subset(outer, state)))
    return found


def loop_private(loop: LoopRegion, states: list[SDFGState]) -> list[str]:
    """The transients only ``loop`` accesses that an iteration writes outside every map, at an element the counter does
    not move: one buffer the rendering declares before the loop, which ``LoopToMap`` would privatize."""
    sdfg = loop.sdfg
    inside = OrderedSet(states)
    outside = OrderedSet(n.data for state in sdfg.states() if state not in inside for n in state.data_nodes())
    found: OrderedSet[str] = OrderedSet()
    for state in states:
        for node in state.data_nodes():
            desc = sdfg.arrays[node.data]
            if not desc.transient or isinstance(desc, data.View) or node.data in outside:
                continue
            if state.entry_node(node) is not None:
                continue
            writes = [e for e in state.in_edges(node) if not e.data.is_empty()]
            subsets = [e.data.get_dst_subset(e, state) for e in writes]
            if writes and not all(s is not None and loop.loop_variable in free_names(s) for s in subsets):
                found.add(node.data)
    return sorted(found)


def carriers(loop: LoopRegion) -> dict[str, str | None]:
    """The containers that carry values from one iteration of ``loop`` to the next, each with the container a whole
    copy writes it from (``x`` carried as ``x = y``) or ``None``: written in the body, and read as input by the first
    state of the body that touches them, so the read sees the previous iteration's write."""
    seen: OrderedSet[str] = OrderedSet()
    exposed: OrderedSet[str] = OrderedSet()
    written: dict[str, str | None] = {}
    for block in loop.bfs_nodes(loop.start_block):
        inside = [block] if isinstance(block, SDFGState) else list(block.all_control_flow_blocks())
        for state in (b for b in inside if isinstance(b, SDFGState)):
            first = {n.data for n in state.data_nodes()} - seen
            exposed |= {
                n.data
                for n in state.data_nodes()
                if n.data in first and state.in_degree(n) == 0 and state.out_degree(n) > 0
            }
            seen |= first
            for node in state.data_nodes():
                for edge in state.in_edges(node):
                    if not edge.data.is_empty():
                        copied = edge.src.data if isinstance(edge.src, nodes.AccessNode) else None
                        written[node.data] = copied if written.get(node.data) is None else written[node.data]
    return {
        name: written[name] for name in exposed if name in written and not isinstance(loop.sdfg.arrays[name], data.View)
    }


def carrier_note(loop: LoopRegion, access: str) -> str:
    """``; x carries the values to the next iteration (x = y)`` for the carriers of ``loop`` other than the container
    ``access`` names, or ``""``."""
    named = re.search(r"on (\w+)\[", access) or re.search(r"on (\w+)", access)
    found = [(name, copy) for name, copy in carriers(loop).items() if named is None or name != named.group(1)]
    if not found:
        return ""
    text = ", ".join(f"{name} ({name} = {copy})" if copy else name for name, copy in found)
    return f"; {text} carr{'ies' if len(found) == 1 else 'y'} the values from one iteration to the next"


def proof_of_carry(
    loop: LoopRegion, reason: LoopRefusal, traps: list[tuple[str, set[str]]] | None = None
) -> str | None:
    """The access that proves ``loop`` carries a dependence, or ``None``: a solver witness, else a constant distance or
    a carried symbol ``LoopToMap`` named."""
    witnessed = witnessed_carrying_access(loop, traps)
    if witnessed is not None:
        return witnessed
    # The element-level proof also names the direction: LoopToMap's "read-after-write conflict" is
    # any read meeting a write, so its access alone reads as RAW for a read-ahead WAR too.
    access = proven_carrying_access(loop)
    if access is None and reason.kind is Refusal.CARRIED_SYMBOL and not has_branch(loop):
        # A symbol one iteration hands to the next: the witness is two consecutive iterations.
        access = carrying_access(reason) + pattern_witness(loop, 1)
    return access


def has_branch(loop: LoopRegion) -> bool:
    """Whether a branch anywhere under ``loop``, nested SDFGs included, may skip part of an iteration."""
    return any(isinstance(block, ConditionalBlock) for block in loop.all_control_flow_blocks(recursive=True))


#: Appended to a parallel map's class line when its independence rests on entry traps.
ASSUMES = "; independent only if {predicates} (the program aborts at entry otherwise)"

#: ``if <violation>:`` opening every abort guard's code (see ``tasklet_utils.add_abort_guard``).
GUARD_HEAD = re.compile(r"^if (.*):\s*$", re.MULTILINE)


def entry_traps(sdfg: SDFG) -> list[tuple[str, set[str]]]:
    """``(predicate, names)`` for every abort guard in ``sdfg``'s own states: what the program assumes, and the symbols
    it names."""
    from dace.sdfg import tasklet_utils

    guards = [node for state in sdfg.states() for node in state.nodes() if tasklet_utils.is_abort_guard(node)]
    return [trap for trap in map(guard_trap, guards) if trap is not None]


def guard_trap(node: nodes.Tasklet) -> tuple[str, set[str]] | None:
    """The ``(predicate, names)`` an abort guard tasklet assumes, or ``None`` if its code is not ``if <violation>:``."""
    head = GUARD_HEAD.search(node.code.as_string)
    if head is None:
        return None
    violation = head.group(1).strip()
    held = violation[len("not ") :].strip() if violation.startswith("not ") else f"not ({violation})"
    names = {s.name for s in as_basic(symbolic.pystr_to_symbolic(held)).free_symbols}
    return str(symbolic.pystr_to_symbolic(held)), names


def function_traps(sdfg: SDFG, inherited: list[tuple[str, set[str]]], out: dict[int, list]) -> None:
    """Fill ``out`` (by ``id``) with the traps every function under ``sdfg`` may rest on, in its own names: a caller's
    trap passes into a nested SDFG through its symbol mapping, and is dropped there if a symbol has no inner name."""
    # A symbol the function's code never uses is not one of its parameters in the rendering either.
    used = OrderedSet(str(name) for name in sdfg.used_symbols(all_symbols=False)) if inherited else OrderedSet()
    traps = [(held, names) for held, names in inherited if names <= used] + entry_traps(sdfg)
    out[id(sdfg)] = traps
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nodes.NestedSDFG):
                function_traps(node.sdfg, translated_traps(traps, node.symbol_mapping), out)


def translated_traps(traps: list[tuple[str, set[str]]], mapping: dict[str, Any]) -> list[tuple[str, set[str]]]:
    """``traps`` in the names of the nested SDFG ``mapping`` binds: a symbol the caller passes bare is renamed."""
    inner = {str(value): name for name, value in mapping.items() if str(value).isidentifier()}
    found = []
    for held, names in traps:
        if not names <= inner.keys():
            continue
        renamed = re.sub(r"\b\w+\b", lambda m: inner[m.group(0)] if m.group(0) in names else m.group(0), held)
        found.append((renamed, {inner[name] for name in names}))
    return found


def map_names(state: SDFGState, entry: nodes.MapEntry) -> set[str]:
    """The symbols the element accesses in ``entry``'s scope name: what independence can rest on."""
    names: set[str] = set()
    for edge in state.scope_subgraph(entry).edges():
        if isinstance(edge.src, nodes.Tasklet) or isinstance(edge.dst, nodes.Tasklet):
            names |= {str(s) for s in edge.data.free_symbols}
    return names


def map_reductions(state: SDFGState, entry: nodes.MapEntry) -> list[str]:
    """``+ into s`` for every container the map accumulates into, through a WCR edge leaving its exit or a WCR write
    inside a nested SDFG in its scope, in name order; not where every write provably lands on a different element per
    iteration."""
    found: OrderedSet[str] = OrderedSet()
    params = list(entry.map.params)
    for edge in state.out_edges(state.exit_node(entry)):
        if edge.data.is_empty() or edge.data.wcr is None:
            continue
        leaves = state.memlet_tree(edge).leaves()
        subsets = [leaf.data.get_dst_subset(leaf, state) or leaf.data.subset for leaf in leaves]
        if not all(distinct_per_iteration(subset, params) for subset in subsets):
            found.add(reduction_text(edge.data.wcr, edge.data.data))
    for call in nested_outputs(state, entry):
        if call.wcr is not None and not all(
            subset is not None and distinct_per_iteration(subset, params) for subset in call.subsets
        ):
            found.add(reduction_text(call.wcr, call.container))
    return sorted(found)


class NestedOutput(NamedTuple):
    """A container outside a map that a nested SDFG in the map's scope writes: the accumulation operator if it
    accumulates, and the subsets of its writes inside, in the map's names."""

    container: str
    wcr: str | None
    subsets: list[Subset | None]


def nested_outputs(state: SDFGState, entry: nodes.MapEntry) -> list[NestedOutput]:
    """Every write a nested SDFG in ``entry``'s scope makes to a container that leaves the map through its exit."""
    exit_node = state.exit_node(entry)
    found = []
    for node in state.scope_subgraph(entry).nodes():
        if not isinstance(node, nodes.NestedSDFG):
            continue
        accumulated = {write[0]: write[1] for write in wcr_writes(list(node.sdfg.states()))}
        for edge in state.out_edges(node):
            if edge.src_conn is None or edge.data.is_empty():
                continue
            path = state.memlet_path(edge)
            if not any(e.dst is exit_node for e in path):
                continue
            found.append(NestedOutput(path[-1].data.data, accumulated.get(edge.src_conn), writes_through(state, edge)))
    return found


def writes_through(state: SDFGState, edge: MultiConnectorEdge[Memlet]) -> list[Subset | None]:
    """The subsets the nested SDFG at the source of ``edge`` writes through it, in the caller's indices and through
    any deeper nested SDFG; ``None`` for a dynamic write whose elements are not known there. Through a view of part of
    the container the inner indices start elsewhere, so the memlet of ``edge`` bounds the writes instead."""
    node = edge.src
    outer, inner = state.sdfg.arrays[edge.data.data], node.sdfg.arrays[edge.src_conn]
    whole = str(edge.data.subset) == str(Range.from_array(outer))
    if not whole or len(inner.shape) != len(outer.shape) or list(inner.offset) != list(outer.offset):
        return [required(edge.data.subset)]
    found: list[Subset | None] = []
    for inner_state in node.sdfg.states():
        for access in inner_state.data_nodes():
            if access.data != edge.src_conn:
                continue
            for write in inner_state.in_edges(access):
                if write.data.is_empty():
                    continue
                if isinstance(write.src, nodes.NestedSDFG):
                    found += writes_through(inner_state, write)
                elif write.data.dynamic:
                    found.append(None)
                else:
                    found.append(required(write.data.get_dst_subset(write, inner_state) or write.data.subset))
    return [None if subset is None else as_outer(subset, node.symbol_mapping) for subset in found]


def as_outer(subset: Subset, mapping: dict[str, Any]) -> Subset:
    """``subset`` with a nested SDFG's symbols replaced by what its caller passes for them."""
    outer = copy.deepcopy(subset)
    outer.replace(
        {symbolic.pystr_to_symbolic(name): symbolic.pystr_to_symbolic(str(value)) for name, value in mapping.items()}
    )
    return outer


def shared_buffers(state: SDFGState, entry: nodes.MapEntry) -> list[str]:
    """Transients outside the map that a nested SDFG in its scope writes, without accumulating, at elements no map
    parameter moves: every iteration writes the same elements, so the iterations share one buffer (a scratch lifted
    out of the nested SDFG), and a parallel run needs a copy per iteration."""
    params = OrderedSet(entry.map.params)
    found: OrderedSet[str] = OrderedSet()
    for call in nested_outputs(state, entry):
        if call.wcr is not None or not state.sdfg.arrays[call.container].transient or not call.subsets:
            continue
        known = [subset for subset in call.subsets if subset is not None]
        if len(known) == len(call.subsets) and not any(params & free_names(subset) for subset in known):
            found.add(call.container)
    return sorted(found)


def parallel_hint(state: SDFGState, entry: nodes.MapEntry, traps: list[tuple[str, set[str]]]) -> str:
    """``PARALLEL``, or ``REDUCTION`` for a map that accumulates, naming the entry traps whose symbols the map's
    element accesses use."""
    if entry.map.schedule == dtypes.ScheduleType.CPU_Persistent:
        return TEAM
    first = PARALLEL
    reductions = map_reductions(state, entry)
    if reductions:
        first = REDUCTION.format(reductions=" and ".join(reductions))
    shared = shared_buffers(state, entry)
    if shared:
        first += PRIVATE.format(names=", ".join(shared))
    relied = sorted({held for held, names in traps if names & map_names(state, entry)})
    if relied:
        first += ASSUMES.format(predicates=" and ".join(relied))
    return first


def expanded_map_hint(state: SDFGState, entry: nodes.MapEntry, original: str | None) -> str | None:
    """The hint of one map of the nest ``MapExpansion`` made of a labelled map (``original``): its class line
    re-derived for its own parameters, as :func:`parallel_hint` would, keeping the entry traps the original line rests
    on; the outermost map keeps the original's notes. ``None`` when the original had no class line."""
    lines = (original or "").split("\n")
    if not lines[0].startswith(
        (LoopClass.PARALLEL.prefix + Reason.INDEPENDENT.value, LoopClass.PARALLEL.prefix + Reason.REDUCTION.value)
    ):
        return None
    first = parallel_hint(state, entry, [])
    assumed = lines[0].find(ASSUMES.split("{")[0])
    if assumed >= 0:
        first += lines[0][assumed:]
    return first


#: A contraction whose output is read only in part downstream. ``{read}`` and ``{size}`` are element counts.
PARTIALLY_READ = "contraction: only {read} of its {size} output elements are read later"


def partially_read_hint(state: SDFGState, node: nodes.LibraryNode) -> str | None:
    """:data:`PARTIALLY_READ` for a contraction library node whose one output, a transient, is read only in
    ``state`` and by fewer elements than it holds (not provably all of them), else ``None``."""
    from dace.transformation.passes.canonicalize.nest_to_contraction import contraction_types

    if not isinstance(node, contraction_types()):
        return None
    written = [e for e in state.out_edges(node) if not e.data.is_empty()]
    if len(written) != 1 or not isinstance(written[0].dst, nodes.AccessNode):
        return None
    name, sdfg = written[0].dst.data, state.sdfg
    desc = sdfg.arrays[name]
    if not desc.transient:
        return None
    holders = [(s, n) for s in sdfg.states() for n in s.data_nodes() if n.data == name]
    # Read in another state (perhaps once per trip of a loop), the reads cannot be summed here
    if any(s is not state for s, _ in holders):
        return None
    reads = [e for _, n in holders for e in state.out_edges(n) if not e.data.is_empty()]
    if not reads:
        return None
    read = sum((e.data.volume for e in reads), symbolic.pystr_to_symbolic(0))
    if (read >= desc.total_size) == True:
        return None
    return PARTIALLY_READ.format(read=read, size=desc.total_size)


def require_classes(sdfg: SDFG) -> None:
    """Raise unless every Map and LoopRegion in ``sdfg`` opens its hint with its class: the rendering's header
    promises one comment per loop, and a loop without one reads as nothing anybody checked.

    :raises RuntimeError: naming the loops and maps without a class.
    """
    missing = [
        node.label
        for node, parent in sdfg.all_nodes_recursive()
        if isinstance(node, (nodes.MapEntry, LoopRegion))
        and not (node.specialization_hint or "").strip().startswith(CLASS_PREFIXES)
    ]
    if missing:
        raise RuntimeError(f"CPF: no class (parallel, sequential or unknown) on {', '.join(missing)}")


@xf.explicit_cf_compatible
class AnnotateLoopKinds(ppl.Pass):
    """Open every Map's and LoopRegion's ``specialization_hint`` with its class."""

    CATEGORY: str = "Analysis"

    def modifies(self) -> ppl.Modifies:
        # Comments only. No pass reads a hint, so nothing needs rerunning because one appeared.
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _: dict[str, Any]) -> int | None:
        """:returns: The number of loops and contractions newly labelled, or ``None`` if none was."""
        from dace.transformation.passes.parallelization.parallelize_loops import ProbeSession

        labelled = 0
        traps: dict[int, list] = {}
        function_traps(sdfg, [], traps)
        # The header lists the entry's own abort() preconditions: a loop comment repeats only one a function renamed
        entry_held = OrderedSet(held for held, names in entry_traps(sdfg))
        # Labelling changes only the hints, which no probe reads
        session = ProbeSession()
        for node, parent_graph in list(sdfg.all_nodes_recursive()):
            if isinstance(node, nodes.LibraryNode) and not node.specialization_hint:
                node.specialization_hint = partially_read_hint(parent_graph, node)
                labelled += node.specialization_hint is not None
                continue
            if not isinstance(node, (nodes.MapEntry, LoopRegion)):
                continue
            hint = (node.specialization_hint or "").strip()
            if hint.startswith(CLASS_PREFIXES):
                continue
            # A note another pass left (an anti-dependence it broke) says what it did, not what the loop is: the
            # verdict goes first, so every loop still opens with its class.
            kind = (
                parallel_hint(
                    parent_graph,
                    node,
                    [trap for trap in traps.get(id(parent_graph.sdfg), []) if trap[0] not in entry_held],
                )
                if isinstance(node, nodes.MapEntry)
                else loop_hint(node, session, traps.get(id(parent_graph.sdfg), []))
            )
            node.specialization_hint = f"{kind}\n{hint}" if hint else kind
            labelled += 1
        return labelled or None
