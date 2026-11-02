# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift a loop-carried in-place array reduction to a write-conflict resolution.

A sequential loop that accumulates into a whole array element-wise -- the
contour_integral shape ``for idx: P[i, j] = P[i, j] + X_idx[i, j]`` (an inner
element map over ``i, j`` nested in the outer ``idx`` loop) -- is a REDUCTION of
``P`` over ``idx``. But because ``P`` is read back and written every iteration,
``LoopToMap`` sees a loop-carried dependency on ``P`` and refuses to parallelize
the ``idx`` loop.

This pass recognises that pattern and drops the read-back, turning the in-place
accumulation into a WCR write ``P[i, j] (wcr: +)= X_idx[i, j]``. With ``P`` no
longer read across ``idx`` iterations, ``LoopToMap`` (a later pipeline stage)
parallelizes ``idx``; the WCR resolves the cross-iteration accumulation (an
OpenMP reduction / atomic at code generation).

``AugAssignToWCR`` performs the edge surgery, but on its own it refuses a write injective over the element map
(``P[i, j]`` -- one element per ``i, j``): the reduction axis is the ENCLOSING loop, invisible to that per-map check.
This pass supplies the enclosing-loop view and takes the rewrite permissively.

Soundness -- the lift is value-preserving (up to reduction reassociation, as for
any parallel reduction) when, for the accumulator ``A`` and the loop ``L``:

* the combine is an order-independent update (``+``, ``-`` with the accumulator on the left, ``*``, ``^``, ``min``,
  ``max``) -- WCR-safe;
* ``A`` is read at the accumulated subset ``S`` ONLY as the accumulator operand
  of its own in-place write (a PURE accumulator -- never used to compute the
  increment or anything else in ``L``); otherwise it is a recurrence, not a
  reduction;
* ``S`` is invariant over ``L``'s loop variable (every iteration hits the same
  elements -- the cross-iteration conflict the WCR resolves);
* the read-back subset equals the write subset (the tasklet reads exactly the
  element it writes).

The pre-loop value of ``A`` is the reduction seed in BOTH the sequential and the
WCR form, so ``A`` need not be initialised for the lift to be sound.

Dataflow surgery (``S`` = the accumulated element, invariant over ``idx``; ``OP`` = + * min max)::

    BEFORE  --  A is read back every iteration => LoopToMap sees a loop-carried dep, refuses:

        for idx in 0:KK                         (sequential LoopRegion)
        |
        +-- state:
        |         A --A[S]--> [map_entry] --A[S]---.
        |                                          v
        |                                     ( tasklet )  __out = A[S] OP incr
        |         incr ----> [map_entry] --incr--> ^
        |                                          |
        |                                     [map_exit] --A[S]--> A      (plain write, no WCR)

    AFTER  --  read-back dropped, WCR carries the accumulation => LoopToMap parallelizes idx:

        for idx in 0:KK                         (no loop-carried read of A)
        |
        +-- state:
        |         incr ----> [map_entry] --incr--> ( tasklet )  __out = incr
        |                                               |
        |                                          [map_exit] ==A[S] (wcr: OP)==> A

Algorithm: for each loop with a concrete trip count, each update of a map over a concrete range that writes an element
invariant over the loop variable and passes ``AugAssignToWCR``, whose accumulator the loop only read-modify-writes
(:func:`~dace.transformation.passes.loop_to_reduce.only_read_modify_written`): apply ``AugAssignToWCR``.

When to apply -- COST (measured, contour reduction, 4 threads, ~16.7M total elements)::

    inner map |  pass ON  |  pass OFF |
    ----------+-----------+-----------+---------------------------
       8x8    |    73 ms  |   468 ms  |  ON  6.4x FASTER
      16x16   |    62 ms  |   131 ms  |  ON  2.1x faster
      32x32   |    57 ms  |    41 ms  |  ON  1.4x slower   <-- crossover
      64x64   |    65 ms  |    23 ms  |  ON  2.8x slower
     128x128  |    46 ms  |    18 ms  |  ON  2.6x slower

The lift wins only when the inner parallel map is too small to fill the cores (a few hundred
elements) while the reduction axis is large; past the crossover it just adds a WCR / atomic
reduction over the loop axis on top of already-saturated parallelism and LOSES. That comparison
needs concrete extents: with a SYMBOLIC size (the common case) the decision is undecidable, so
:func:`bounds_are_concrete` and the map range check refuse the lift rather than gamble. A specialized program (constants
substituted) with a small inner map is the case that still lifts.
"""

from typing import Any

from dace import SDFG, nodes, properties, symbolic
from dace.sdfg import SDFGState
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow.wcr_conversion import AugAssignToWCR
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.loop_to_reduce import map_tasklet_augassign_candidates, only_read_modify_written
from dace.transformation.transformation import explicit_cf_compatible


def bounds_are_concrete(loop: LoopRegion) -> bool:
    """Whether the trip count of ``loop`` is a compile-time integer: see the module docstring for why a symbolic extent
    is refused."""
    bounds = (
        loop_analysis.get_init_assignment(loop),
        loop_analysis.get_loop_end(loop),
        loop_analysis.get_loop_stride(loop),
    )
    return all(bound is not None and not symbolic.pystr_to_symbolic(bound).free_symbols for bound in bounds)


def carried_update(loop: LoopRegion) -> tuple[SDFGState, AugAssignToWCR] | None:
    """The first update in a map of ``loop``, over a concrete range, of an element no iteration of ``loop`` moves, whose
    accumulator nothing else reads: the read-back it drops is what a recurrence needs."""
    itervar = loop.loop_variable
    for state in loop.states():
        for tasklet in [n for n in state.nodes() if isinstance(n, nodes.Tasklet)]:
            entry = state.entry_node(tasklet)
            if not isinstance(entry, nodes.MapEntry) or entry.map.range.free_symbols:
                continue
            for expr, binding in map_tasklet_augassign_candidates(state, entry, tasklet):
                source, target = binding[AugAssignToWCR.input], binding[AugAssignToWCR.output]
                written = state.edges_between(binding[AugAssignToWCR.map_exit], target)
                if source.data != target.data or any(
                    e.data.wcr is not None or itervar in {str(s) for s in e.data.subset.free_symbols} for e in written
                ):
                    continue
                xform = AugAssignToWCR()
                xform.setup_match(state.sdfg, state.parent_graph.cfg_id, state.block_id, binding, expr, override=True)
                if xform.can_be_applied(state, expr, state.sdfg, permissive=True) and only_read_modify_written(
                    loop, target.data
                ):
                    return state, xform
    return None


@properties.make_properties
@explicit_cf_compatible
class LiftLoopCarriedReduction(ppl.Pass):
    """Lift loop-carried in-place array reductions to WCR writes so the enclosing
    loop can be parallelized by ``LoopToMap``."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def __init__(self, only: list | None = None):
        super().__init__()
        # When given, only these loops are candidates (the segments a split left behind).
        self._only = only

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        lifted = 0
        only = None if self._only is None else {id(loop) for loop in self._only}
        if only is not None and not only:
            return None
        for sd in sdfg.all_sdfgs_recursive():
            for loop in [n for n in sd.all_control_flow_regions() if isinstance(n, LoopRegion)]:
                if loop.pinned_sequential or not loop.loop_variable or (only is not None and id(loop) not in only):
                    continue
                if not bounds_are_concrete(loop):
                    continue
                # A rewrite may fission its state, so the updates are found afresh after each
                while (update := carried_update(loop)) is not None:
                    state, xform = update
                    xform.apply(state, state.sdfg)
                    lifted += 1
        return lifted or None


__all__ = ["LiftLoopCarriedReduction"]
