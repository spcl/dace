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
from typing import Optional

import re

from dace import SDFG
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
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


def loop_hint(loop: LoopRegion) -> str:
    """The hint text naming what kind of loop ``loop`` is."""
    reason = refusal_reason(loop)
    if reason is None:
        return PARALLEL
    if not reason:
        return SEQUENTIAL_UNEXAMINED
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
            if isinstance(node, nodes.MapEntry):
                if node.specialization_hint:
                    continue
                node.specialization_hint = PARALLEL
            elif isinstance(node, LoopRegion):
                if node.specialization_hint:
                    continue
                node.specialization_hint = loop_hint(node)
            else:
                continue
            labelled += 1
        return labelled or None
