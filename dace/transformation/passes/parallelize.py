# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Single-shot pipeline that turns sequential loops into parallel maps.

The recipe is the set of pre-passes that lead up to ``LoopToMap`` (and
``LoopToMap`` itself), applied once in order:

0a. ``LoopUnroll`` (short loops only) -- fully unroll every constant-trip loop
    whose iteration count is ``<= unroll_limit``, so small recurrence / reduction
    loops (e.g. the length-5 cloudsc species loop, once specialized) become inline
    straight-line code rather than being parallelized with atomics.
0b. ``LoopPeeling`` (best effort) -- for ``k`` in ``1..peel_limit`` and each of
    front / back / front+back, peel ``k`` boundary iterations off the loops, run
    the parallelization tail on a copy, and keep the single peel that yields the
    most maps; revert entirely if none beats the no-peel baseline. The boundary
    iterations often carry the special first/last-iteration case, so splitting
    them out can leave a clean, mappable loop body.
1. :class:`~dace.transformation.passes.scalar_fission.PrivatizeScalars` --
   privatize loop-local (thread-local) scalars so a loop body no longer carries
   a false scalar dependency across iterations. Runs *after* unroll and peeling
   so the freshly-introduced straight-line bodies' scalars get renamed too.
2. ``TrivialTaskletElimination`` -- drop the ``__out = __inp`` copy tasklets the
   frontend emits, exposing the bare read-modify-write spine of accumulators.
3. ``AugAssignToWCR`` -- rewrite ``arr[S] += x`` (including the frontend's
   copy-wrapped form) into a write-conflict-resolution write, so the reduction
   loop's write is no longer iteration-indexed and ``LoopToMap`` may map it.
4. :class:`~dace.transformation.passes.loop_to_reduce.LoopToReduce` -- lift the
   pure accumulator loops that remain to ``Reduce`` library nodes.
5. ``LoopToMap`` -- parallelize every loop that is now free of loop-carried
   dependencies.

The pipeline is meant to run **after** ``simplify`` and **once**: every stage is
either idempotent or internally exhaustive (``PatternMatchAndApplyRepeated``),
so there is nothing to re-apply. It is modelled on the canonicalization pipeline
rather than :class:`~dace.transformation.pass_pipeline.Pipeline` because the
latter forbids re-using a pass type (``PatternMatchAndApplyRepeated`` appears
several times) and would re-run stages on its fixed-point logic.

All transformation classes are imported lazily inside the methods: importing them
at module load would form a cycle (this module is imported by
``dace.transformation.passes`` whose subpackages those transformations import).
"""
import copy
from typing import Any, Dict, Optional

from dace import properties, symbolic
from dace.sdfg import SDFG, nodes
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl

#: Default trip-count threshold below which a constant-trip loop is unrolled.
_DEFAULT_UNROLL_LIMIT = 8
#: Default maximum number of iterations peeled (per side) when searching for a
#: peel that unblocks parallelization.
_DEFAULT_PEEL_LIMIT = 8


@properties.make_properties
class ParallelizePipeline(ppl.Pass):
    """Parallelize an SDFG's loops, lifting reductions on the way.

    Applies the parallelization recipe once, imperatively. The pipeline does not
    re-run: each stage is internally exhaustive or idempotent.

    :param validate: Validate the SDFG once at the end.
    :param validate_all: Validate the SDFG after each stage.
    :param unroll_limit: Constant-trip loops with at most this many iterations are
                         fully unrolled before parallelization; ``0`` disables it.
    :param peel_limit: Up to this many iterations are peeled (per side) in the
                       best-effort peeling search; ``0`` disables it.
    """

    CATEGORY: str = 'Optimization Preparation'

    validate = properties.Property(dtype=bool, default=False, desc='Validate the SDFG at the end.')
    validate_all = properties.Property(dtype=bool, default=False, desc='Validate the SDFG after each stage.')
    unroll_limit = properties.Property(
        dtype=int,
        default=_DEFAULT_UNROLL_LIMIT,
        desc='Fully unroll constant-trip loops with at most this many iterations before '
        'parallelization (0 disables). Keeps small recurrence/reduction loops as inline '
        'straight-line code instead of atomically-parallelized maps.')
    peel_limit = properties.Property(
        dtype=int,
        default=_DEFAULT_PEEL_LIMIT,
        desc='Best-effort loop peeling: try peeling 1..peel_limit iterations from the front, '
        'the back, and both, keep the peel that produces the most maps, and revert if none '
        'beats the no-peel baseline (0 disables).')

    def __init__(self,
                 validate: bool = False,
                 validate_all: bool = False,
                 unroll_limit: int = _DEFAULT_UNROLL_LIMIT,
                 peel_limit: int = _DEFAULT_PEEL_LIMIT):
        self.validate = validate
        self.validate_all = validate_all
        self.unroll_limit = unroll_limit
        self.peel_limit = peel_limit

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    @staticmethod
    def _loops(sdfg: SDFG):
        return [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]

    @staticmethod
    def _num_maps(sdfg: SDFG) -> int:
        return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))

    def _constant_trip_count(self, loop: LoopRegion, sdfg: SDFG) -> Optional[int]:
        """The exact iteration count of ``loop`` if it is constant, else ``None``.

        Matches the iteration count ``LoopUnroll`` would produce
        (``len(range(0, end - start + 1, stride))``).
        """
        from dace.transformation.passes.analysis import loop_analysis
        start = loop_analysis.get_init_assignment(loop)
        end = loop_analysis.get_loop_end(loop)
        stride = loop_analysis.get_loop_stride(loop)
        if start is None or end is None or stride is None or not loop.loop_variable:
            return None
        if symbolic.issymbolic(stride, sdfg.constants) or symbolic.issymbolic(end - start, sdfg.constants):
            return None
        try:
            stride_val = int(symbolic.evaluate(stride, sdfg.constants))
            diff = int(symbolic.evaluate(end - start + 1, sdfg.constants))
        except (TypeError, ValueError):
            return None
        if stride_val <= 0 or diff <= 0:
            return None
        return len(range(0, diff, stride_val))

    def _unroll_short_loops(self, sdfg: SDFG):
        """Fully unroll every constant-trip loop with ``<= unroll_limit`` iterations.

        Re-collects after each unroll since unrolling rewrites the control-flow
        structure (and may expose newly-constant inner loops).
        """
        if self.unroll_limit <= 0:
            return
        from dace.transformation.interstate.loop_unroll import LoopUnroll
        changed = True
        while changed:
            changed = False
            for loop in self._loops(sdfg):
                trip = self._constant_trip_count(loop, sdfg)
                if trip is None or trip > self.unroll_limit:
                    continue
                try:
                    LoopUnroll().apply_to(sdfg=loop.sdfg, loop=loop)
                except Exception:
                    continue  # not unrollable in this context; leave it for LoopToMap
                changed = True
                break

    def _peel_loops(self, sdfg: SDFG, count: int, direction: str) -> int:
        """Peel ``count`` iterations off every peelable loop and return how many
        loops were peeled. ``direction`` is ``'front'``, ``'back'`` or ``'both'``.

        Uses ``verify=False`` so loops with symbolic bounds (where peeling a
        boundary is most useful) are not rejected by ``LoopUnroll``'s
        constant-size gate; infeasible loops simply raise and are skipped.
        """
        from dace.transformation.interstate.loop_peeling import LoopPeeling
        sides = {'front': [True], 'back': [False], 'both': [True, False]}[direction]
        peeled = 0
        for loop in self._loops(sdfg):
            # A loop short enough to be fully consumed by the peel is the unroll
            # stage's job, not peeling's.
            trip = self._constant_trip_count(loop, sdfg)
            if trip is not None and trip <= count * len(sides):
                continue
            did = False
            for begin in sides:
                try:
                    LoopPeeling().apply_to(sdfg=loop.sdfg, loop=loop, verify=False, count=count, begin=begin)
                    did = True
                except Exception:
                    continue
            peeled += int(did)
        return peeled

    def _run_tail(self, sdfg: SDFG):
        """The reduction-aware parallelization tail: privatize -> propagate
        symbols/constants -> expose RMW -> WCR -> reduce -> map. Symbol and
        constant propagation run here because peeling/unrolling fold concrete
        iteration indices into the bodies, and propagating them can simplify
        bounds and conditions enough to expose more maps."""
        from dace.transformation.dataflow.wcr_conversion import AugAssignToWCR
        from dace.transformation.dataflow.trivial_tasklet_elimination import TrivialTaskletElimination
        from dace.transformation.interstate.loop_to_map import LoopToMap
        from dace.transformation.passes.constant_propagation import ConstantPropagation
        from dace.transformation.passes.loop_to_reduce import LoopToReduce
        from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
        from dace.transformation.passes.scalar_fission import PrivatizeScalars
        from dace.transformation.passes.symbol_propagation import SymbolPropagation
        PrivatizeScalars().apply_pass(sdfg, {})
        SymbolPropagation().apply_pass(sdfg, {})
        ConstantPropagation().apply_pass(sdfg, {})
        PatternMatchAndApplyRepeated([TrivialTaskletElimination()]).apply_pass(sdfg, {})
        PatternMatchAndApplyRepeated([AugAssignToWCR()]).apply_pass(sdfg, {})
        LoopToReduce().apply_pass(sdfg, {})
        PatternMatchAndApplyRepeated([LoopToMap()]).apply_pass(sdfg, {})

    def _l2m_candidate_maps(self, candidate: SDFG) -> int:
        """Cheap proxy used by the peeling search: does peeling unblock more
        ``LoopToMap``? Runs scalar fission -> symbol propagation -> constant
        propagation -> LoopToMap (no reduction passes) and returns the map count.
        Mutates ``candidate`` (callers pass a disposable copy)."""
        from dace.transformation.interstate.loop_to_map import LoopToMap
        from dace.transformation.passes.constant_propagation import ConstantPropagation
        from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
        from dace.transformation.passes.scalar_fission import PrivatizeScalars
        from dace.transformation.passes.symbol_propagation import SymbolPropagation
        PrivatizeScalars().apply_pass(candidate, {})
        SymbolPropagation().apply_pass(candidate, {})
        ConstantPropagation().apply_pass(candidate, {})
        PatternMatchAndApplyRepeated([LoopToMap()]).apply_pass(candidate, {})
        return self._num_maps(candidate)

    def _peel_best_effort(self, sdfg: SDFG):
        """Search peel configurations and apply the one that unblocks the most
        maps; if none beats the no-peel baseline, leave ``sdfg`` unpeeled.

        The search runs the cheap ``_l2m_candidate_maps`` proxy on throwaway
        copies; the winning peel is then applied to ``sdfg`` in place and the
        real reduction-aware tail runs afterwards in :meth:`apply_pass`.
        """
        if self.peel_limit <= 0 or not self._loops(sdfg):
            return

        baseline = self._l2m_candidate_maps(copy.deepcopy(sdfg))
        best_count, best = baseline, None
        for direction in ('front', 'back', 'both'):
            for count in range(1, self.peel_limit + 1):
                candidate = copy.deepcopy(sdfg)
                if self._peel_loops(candidate, count, direction) == 0:
                    continue
                try:
                    n_maps = self._l2m_candidate_maps(candidate)
                except Exception:
                    continue
                if n_maps > best_count:
                    best_count, best = n_maps, (count, direction)

        if best is not None:
            self._peel_loops(sdfg, *best)

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Parallelize ``sdfg`` in place.

        :param sdfg: The SDFG to parallelize.
        :returns: The number of stages applied.
        """
        self._unroll_short_loops(sdfg)
        if self.validate_all:
            sdfg.validate()
        self._peel_best_effort(sdfg)
        if self.validate_all:
            sdfg.validate()
        self._run_tail(sdfg)
        if self.validate_all or self.validate:
            sdfg.validate()
        return 3  # unroll + peel + tail


def parallelize(sdfg: SDFG,
                validate: bool = True,
                validate_all: bool = False,
                unroll_limit: int = _DEFAULT_UNROLL_LIMIT,
                peel_limit: int = _DEFAULT_PEEL_LIMIT) -> SDFG:
    """Parallelize ``sdfg``'s loops in place and return it.

    One-call recipe meant to run after ``simplify``.

    :param sdfg: The SDFG to parallelize.
    :param validate: Validate the SDFG after parallelization.
    :param validate_all: Validate the SDFG after each stage.
    :param unroll_limit: See :class:`ParallelizePipeline`.
    :param peel_limit: See :class:`ParallelizePipeline`.
    :returns: The same ``sdfg`` instance, parallelized.
    """
    ParallelizePipeline(validate=validate,
                        validate_all=validate_all,
                        unroll_limit=unroll_limit,
                        peel_limit=peel_limit).apply_pass(sdfg, {})
    return sdfg
