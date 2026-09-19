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
from typing import Dict, List, NamedTuple, Optional, Tuple

import re

import sympy

from dace import SDFG, data, symbolic
from dace.ordered import OrderedSet
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.loop_to_reduce import loop_to_map_refusal_is_carried

#: A Map. Data-parallel by construction, whatever schedule it ends up carrying.
PARALLEL = 'parallel -- the iterations are independent'

#: A loop whose carried dependence was proven. ``{reason}`` is the carrying ACCESS, since that is
#: the part a reader acts on -- the prose around it repeated on every such loop and said the same
#: thing each time.
SEQUENTIAL_PROVEN = 'sequential -- carried: {reason}'

#: A loop the dependence test declined to answer. NOT the same fact as the line above: undecided is
#: not proven, and the wording has to keep saying so in one word.
SEQUENTIAL_UNDECIDED = 'undecided -- not proven either way: {reason}'

#: A loop the dependence test could not be asked about at all.
SEQUENTIAL_UNEXAMINED = 'unclassified -- never examined for dependences'

#: A loop the dependence test finds independent that an earlier pass pinned sequential anyway (the
#: fallback arm of a specialization). Rendered without a pragma, so it must not read as parallel.
SEQUENTIAL_PINNED = 'sequential -- pinned by an earlier pass; the dependence test finds the iterations independent'

#: The untiled skew: a sequential diagonal over a parallel front. The second line is the only
#: advice a reader can act on -- which of the two correct shapes to run -- so it survives the cut
#: while the restatement of what a skew is does not.
WAVEFRONT_DIAGONAL = ('wavefront diagonal {skew} -- sequential: the skew put every dependence on this axis\n'
                      'alternative: the unskewed nest, sequential in both axes -- worth timing on CPU, '
                      'rarely on GPU')
WAVEFRONT_FRONT = 'wavefront front {skew} -- parallel: at a fixed diagonal the points are independent'

#: The tiled skew: a sequential tile diagonal over a parallel tile column over a sequential interior.
WAVEFRONT_TILE_DIAGONAL = ('wavefront tile diagonal {skew} {tile} -- sequential: the tile diagonal carries '
                           'every dependence\n'
                           'alternatives: the element diagonal, or the unskewed nest -- all three bit-identical; '
                           'a bigger tile trades kernel launches for block-local barriers')
WAVEFRONT_TILE_COLUMN = 'wavefront tile column {tile} -- parallel: the tiles on one diagonal are independent'
WAVEFRONT_TILE_INTERIOR = ('wavefront inner tile {tile} -- sequential: the original order is kept verbatim, '
                           'so the tiled result is bit-identical')


def skew_label(a: int, b: int, u: str, v: str) -> str:
    """The skew that produced this wavefront, as ``(t = i + j)``.

    Which skew was taken is the one fact about a wavefront a reader cannot recover from the emitted
    loop: after the rewrite the axes are named t and p and the original iterators are gone. The
    coefficient is written only where it is not 1, so the common unit skew stays short.
    """

    def term(coeff: int, name: str) -> str:
        return name if abs(coeff) == 1 else f'{abs(coeff)}*{name}'

    lead = f'-{term(a, u)}' if a < 0 else term(a, u)
    return f'(t = {lead} {"-" if b < 0 else "+"} {term(b, v)})'


def tile_label(bi: int, bj: int) -> str:
    """The tile extent the diagonal walks -- the number that trades kernel launches for barriers."""
    return f'[{bi}x{bj}]'


def refusal_reason(loop: LoopRegion) -> Optional[str]:
    """Why ``LoopToMap`` would refuse ``loop``, ``None`` if it would accept, ``''`` if it cannot say.

    Asked with ``pinned_sequential`` set aside. The pin is a schedule decision an earlier pass made,
    and "loop is pinned sequential" is not an answer to the dependence question a reader is asking --
    it only names the pass that got there first. Restored either way; the probe reads the graph and
    does not touch it.
    """
    if not loop.loop_variable:
        return ''
    from dace.transformation.interstate.loop_to_map import LoopToMap
    probe = LoopToMap()
    probe.loop = loop
    pinned = loop.pinned_sequential
    loop.pinned_sequential = False
    try:
        applicable = probe.can_be_applied(loop.parent_graph, 0, loop.sdfg, permissive=False)
    except Exception:
        # A comment must not be able to fail the compilation that asked for it. Nothing downstream
        # reads a hint, so a probe with no answer leaves the loop unclassified and says so.
        return ''
    finally:
        loop.pinned_sequential = pinned
    return None if applicable else (probe.last_refusal_reason or '')


#: ``LoopToMap``'s refusal, as ``<kind> conflict on <array> within the loop body - src_subset=<subset>``.
REFUSAL_SHAPE = re.compile(r'(\w+)-after-(\w+) conflict on (\w+).*?src_subset=(.*)$', re.S)

#: The three conflict kinds, as the two-letter names a reader of dependence analysis already has.
CONFLICT_ABBREV = {('read', 'write'): 'RAW', ('write', 'read'): 'WAR', ('write', 'write'): 'WAW'}


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
    kind = CONFLICT_ABBREV.get((first, second), f'{first}-after-{second}')
    return f'{kind} on {array}[{subset.strip()}]'


class Access(NamedTuple):
    """One element access in a loop body, in the loop SDFG's names: ``array[index]``."""
    array: str
    write: bool
    index: Tuple[sympy.Expr, ...]


class Body(NamedTuple):
    """What :func:`walk_body` collects: element accesses, names that vary within one iteration, the
    params of maps, and whether some map range moves with the loop."""
    accesses: List[Access]
    varying: OrderedSet[str]
    params: OrderedSet[str]
    moving: List[bool]


def as_loop_names(expr, mapping: Dict[str, sympy.Expr]) -> sympy.Expr:
    """``expr`` rewritten through ``mapping`` (inner name -> outer expression) and reparsed, so one
    name is one symbol instance whichever scope minted it."""
    expr = symbolic.pystr_to_symbolic(str(expr))
    repl = {sym: mapping[sym.name] for sym in expr.free_symbols if sym.name in mapping}
    expr = expr.subs(repl, simultaneous=True) if repl else expr
    return symbolic.pystr_to_symbolic(str(expr))


def point_index(subset, mapping: Dict[str, sympy.Expr], offset: Tuple[sympy.Expr, ...]) -> Optional[Tuple]:
    """The single element ``subset`` names, in the loop's names, or ``None`` for a range."""
    if subset is None or (offset and len(offset) != subset.dims()):
        return None
    index = []
    for dim, (begin, end, _) in enumerate(subset.ndrange()):
        begin = as_loop_names(begin, mapping)
        if begin != as_loop_names(end, mapping):
            return None
        index.append(begin + offset[dim] if offset else begin)
    return tuple(index)


def edge_accesses(edge, state: SDFGState, arrays: Dict[str, Tuple[str, Tuple]],
                  mapping: Dict[str, sympy.Expr]) -> List[Access]:
    """The element reads and writes ``edge`` makes: into or out of a tasklet, or an array copy."""
    memlet = edge.data
    if memlet.is_empty() or memlet.wcr is not None:
        return []
    if isinstance(edge.src, nodes.AccessNode) and isinstance(edge.dst, nodes.AccessNode):
        sides = [(edge.src.data, memlet.get_src_subset(edge, state), False),
                 (edge.dst.data, memlet.get_dst_subset(edge, state), True)]
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


def nested_bindings(state: SDFGState, node: nodes.NestedSDFG, arrays: Dict[str, Tuple[str, Tuple]],
                    mapping: Dict[str, sympy.Expr]) -> Tuple[Dict, Dict]:
    """The arrays and symbols ``node``'s body sees, rebound to the loop's names.

    An inner array binds only when its connector's outer subset has the inner rank; its element
    ``k`` is then outer element ``begin + k``. Anything else stays unbound and is ignored.
    """
    inner_arrays: Dict[str, Tuple[str, Tuple]] = {}
    for edge in list(state.in_edges(node)) + list(state.out_edges(node)):
        conn = edge.dst_conn if edge.dst is node else edge.src_conn
        if conn is None or edge.data.is_empty() or edge.data.data not in arrays:
            continue
        outer, outer_offset = arrays[edge.data.data]
        desc = node.sdfg.arrays.get(conn)
        begins = [as_loop_names(begin, mapping) for begin, _, _ in edge.data.subset.ndrange()]
        if desc is None or len(desc.shape) != len(begins) or (outer_offset and len(outer_offset) != len(begins)):
            continue
        inner_arrays[conn] = (outer, tuple(b + outer_offset[d] if outer_offset else b for d, b in enumerate(begins)))
    inner_mapping = {name: as_loop_names(value, mapping) for name, value in node.symbol_mapping.items()}
    return inner_arrays, inner_mapping


def walk_body(sdfg_or_region: ControlFlowRegion, arrays: Dict[str, Tuple[str, Tuple]], mapping: Dict[str, sympy.Expr],
              loop_variable: str, body: Body) -> None:
    """Fill ``body`` from everything under ``sdfg_or_region``, through nested SDFGs."""
    for region in sdfg_or_region.all_control_flow_regions():
        if isinstance(region, LoopRegion) and region.loop_variable:
            body.varying.add(region.loop_variable)
    for edge in sdfg_or_region.all_interstate_edges():
        body.varying.update(edge.data.assignments.keys())
    for state in sdfg_or_region.all_states():
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


def carried_distance(write: Access, read: Access, loop_var: sympy.Symbol) -> Optional[sympy.Expr]:
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


def proven_carrying_access(loop: LoopRegion) -> Optional[str]:
    """``RAW on aa[i - 1, j]`` when two element accesses in ``loop``'s body PROVE a carried dependence.

    ``LoopToMap`` judges a body by its propagated memlets, and a nested SDFG's are often the whole
    array -- so it can only decline. The element accesses inside still say it: a write ``A[f(i)]``
    and a read ``A[f(i - d)]`` with the same fixed indices elsewhere are ``d`` iterations apart.
    Only a constant step, map ranges fixed across iterations and indices that are otherwise
    loop-invariant or map params are admitted.
    """
    stride = loop_analysis.get_loop_stride(loop)
    if stride is None or not stride.is_Integer or stride == 0:
        return None
    arrays = {name: (name, ()) for name, desc in loop.sdfg.arrays.items() if not isinstance(desc, data.View)}
    body = Body([], OrderedSet(), OrderedSet(), [])
    walk_body(loop, arrays, {}, loop.loop_variable, body)
    if any(body.moving):
        return None
    unknown = body.varying - body.params - OrderedSet([loop.loop_variable])
    stable = [a for a in body.accesses if not ({s.name for e in a.index for s in e.free_symbols} & unknown)]
    loop_var = symbolic.pystr_to_symbolic(loop.loop_variable)
    reads: Dict[str, List[Access]] = {}
    for access in stable:
        if not access.write:
            reads.setdefault(access.array, []).append(access)
    for write in (a for a in stable if a.write):
        for read in reads.get(write.array, ()):
            distance = carried_distance(write, read, loop_var)
            if distance is None or distance == 0 or distance % stride != 0:
                continue
            kind = 'RAW' if distance / stride > 0 else 'WAR'
            return f"{kind} on {read.array}[{', '.join(str(e) for e in read.index)}]"
    return None


def loop_hint(loop: LoopRegion) -> str:
    """The hint text naming what kind of loop ``loop`` is."""
    reason = refusal_reason(loop)
    if reason is None:
        return SEQUENTIAL_PINNED if loop.pinned_sequential else PARALLEL
    if not reason:
        return SEQUENTIAL_UNEXAMINED
    # The element-level proof also names the direction: LoopToMap's "read-after-write conflict" is
    # any read meeting a write, so its access alone reads as RAW for a read-ahead WAR too.
    access = proven_carrying_access(loop)
    if access is not None:
        return SEQUENTIAL_PROVEN.format(reason=access)
    if loop_to_map_refusal_is_carried(reason):
        return SEQUENTIAL_PROVEN.format(reason=carrying_access(reason))
    return SEQUENTIAL_UNDECIDED.format(reason=carrying_access(reason))


@xf.explicit_cf_compatible
class AnnotateLoopKinds(ppl.Pass):
    """Give every unlabelled Map and LoopRegion a ``specialization_hint`` naming its kind.

    Runs last in the canonicalize recipe, on the graph the rendering will see. A hint already set
    is left alone: the pass that set it knew something this one cannot re-derive -- which
    alternative it declined (``BreakAntiDependence``, ``Scan``) or that a loop is a wavefront axis
    (``WavefrontSkew``).
    """

    CATEGORY: str = 'Analysis'

    def modifies(self) -> ppl.Modifies:
        # Comments only. No pass reads a hint, so nothing needs rerunning because one appeared.
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _) -> Optional[int]:
        """:returns: The number of loops newly labelled, or ``None`` if none was."""
        labelled = 0
        for node, parent_graph in list(sdfg.all_nodes_recursive()):
            # A blank hint renders as nothing, so it is no label either.
            if not isinstance(node, (nodes.MapEntry, LoopRegion)) or (node.specialization_hint or '').strip():
                continue
            node.specialization_hint = PARALLEL if isinstance(node, nodes.MapEntry) else loop_hint(node)
            labelled += 1
        return labelled or None
