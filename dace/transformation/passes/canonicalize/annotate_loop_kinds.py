# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Name every loop in the canonical form, so the standalone rendering says what each one is.

MPR output is read by a specializing pass or by a person, and both arrive at a ``for`` loop with
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

from dace import SDFG
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.loop_to_reduce import loop_to_map_refusal_is_carried

#: A Map. Data-parallel by construction, whatever schedule it ends up carrying.
PARALLEL = ('parallel -- the iterations are independent, so any order and any thread count '
            'computes the same result.\n'
            'Whether this one gets threads is a schedule decision, and a separate one.')

#: A loop whose carried dependence was proven. ``{reason}`` is ``LoopToMap``'s own words.
SEQUENTIAL_PROVEN = ('sequential -- a loop-carried dependence was PROVEN, so this iteration order is '
                     'required.\n'
                     'Proof: {reason}\n'
                     'No schedule reorders it. Parallelizing it needs a rewrite that removes the '
                     'dependence -- a scan, a skew, a privatized accumulator.')

#: A loop the dependence test declined to answer. NOT the same fact as the line above.
SEQUENTIAL_UNDECIDED = ('potentially sequential -- nothing was proven either way. The dependence test '
                        'declined before it reached a verdict.\n'
                        'Declined at: {reason}\n'
                        'This is not a proof of a dependence. A stronger analysis, or a fact only the '
                        'caller knows, may parallelize it.')

#: A loop the dependence test could not be asked about at all.
SEQUENTIAL_UNEXAMINED = ('potentially sequential -- this loop was never examined for dependences, so '
                         'nothing at all is known about it.\n'
                         'That is not a proof of a dependence. Treat it as unclassified, not as '
                         'inherently serial.')

#: The untiled skew: a sequential diagonal over a parallel front.
WAVEFRONT_DIAGONAL = ('wavefront diagonal -- sequential: the skew moved every dependence onto this axis, '
                      'which is exactly what frees the axis inside it.\n'
                      'Alternative: the original unskewed nest, fully sequential in both axes.\n'
                      'CPU: the unskewed nest is worth trying -- the diagonal walks a strided anti-diagonal '
                      'and forks a parallel region per step.\n'
                      'GPU: the diagonal is usually the better of the two; there are threads to fill.\n'
                      'Both are correct. Measure before choosing.')
WAVEFRONT_FRONT = ('wavefront front -- parallel: at a fixed diagonal the points are independent, which is '
                   'what the skew legality test established.')

#: The tiled skew: a sequential tile diagonal over a parallel tile column over a sequential interior.
WAVEFRONT_TILE_DIAGONAL = ('wavefront tile diagonal -- sequential: the tile diagonal carries every dependence, '
                           'for the reason the untiled diagonal does.\n'
                           'Alternative: the element-granularity diagonal, or the original unskewed nest.\n'
                           'CPU: the tiling is usually the better of the three -- it gives the innermost loop '
                           'unit stride and cuts the number of parallel regions by the tile area.\n'
                           'GPU: the tile extent is the trade -- a bigger tile turns kernel launches into '
                           'block-local barriers.\n'
                           'All are correct, and bit-identical. Measure before choosing.')
WAVEFRONT_TILE_COLUMN = ('wavefront tile column -- parallel: the tiles on one diagonal are independent, which is '
                         'what the tiling legality test established.')
WAVEFRONT_TILE_INTERIOR = ('wavefront tile interior -- sequential: inside a tile the original iteration order is '
                           'kept verbatim, which is why the tiled result is bit-identical to the unskewed nest.')


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


def loop_hint(loop: LoopRegion) -> str:
    """The hint text naming what kind of loop ``loop`` is."""
    reason = refusal_reason(loop)
    if reason is None:
        return PARALLEL
    if not reason:
        return SEQUENTIAL_UNEXAMINED
    if loop_to_map_refusal_is_carried(reason):
        return SEQUENTIAL_PROVEN.format(reason=reason)
    return SEQUENTIAL_UNDECIDED.format(reason=reason)


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

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _) -> Optional[int]:
        """:returns: The number of loops newly labelled, or ``None`` if none was."""
        labelled = 0
        for node, _parent in list(sdfg.all_nodes_recursive()):
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
