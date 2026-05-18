# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A ``"vectorized"`` implementation for the standard ``Reduce`` library node.

The vectorization pipeline lifts a recognised accumulator pattern into a
``dace.libraries.standard.nodes.Reduce`` node and sets its
``implementation`` to ``"vectorized"``; this module registers that
implementation. :class:`ExpandReduceVectorized` is a thin, **schedule-aware
dispatcher** (no OOP abstraction — a single ``ExpandTransformation`` as the
DaCe library API requires):

- ``Sequential`` / ``Default`` schedule — the vectorized sequential
  reduction (a per-lane vector accumulator folded by a single
  ``horizontal_reduce_<op>`` intrinsic). RV-1b-alpha delegates this to
  the proven :class:`ExpandReducePure` expansion to land the
  registration + dispatcher contract with zero risk; RV-1b-beta swaps
  in the real vectorized horizontal-reduce kernel.
- ``CPU_Multicore`` schedule — delegate to :class:`ExpandReduceOpenMP`,
  whose ``#pragma omp parallel for reduction(...)`` owns the per-thread
  private accumulators and the cross-thread combine (thread count is an
  OpenMP runtime concern, ``omp_get_max_threads()`` — never an SDFG
  symbol, no manual ``M/NUM_CORES`` tiling).
- Any other schedule (GPU, MPI, Snitch, …) or an unsupported reduction
  operator — raise ``NotImplementedError``. Loud failure over a silent
  fallback: in the CPU vectorization pipeline an unexpected schedule /
  a non-associative or custom reduction must surface, not be silently
  mis-lowered.
"""
import dace
import dace.library
from dace import dtypes
from dace.frontend.operations import detect_reduction_type
from dace.libraries.standard.nodes.reduce import (
    ExpandReduceOpenMP,
    ExpandReducePure,
    Reduce,
)
from dace.transformation import transformation as pm

#: Reduction operators with an associative identity and a matching
#: ``horizontal_reduce_<op>`` runtime primitive. Maps to the op-token
#: suffix used by ``cpu_vectorizable_math_*.h``. Sub / Div / Logical_* /
#: *_Location / Exchange / Custom are intentionally absent — they have
#: no associative-fold identity and must raise rather than mis-reduce.
REDTYPE_TO_OP = {
    dtypes.ReductionType.Sum: "add",
    dtypes.ReductionType.Product: "mul",
    dtypes.ReductionType.Max: "max",
    dtypes.ReductionType.Min: "min",
    dtypes.ReductionType.Bitwise_And: "band",
    dtypes.ReductionType.Bitwise_Or: "bor",
    dtypes.ReductionType.Bitwise_Xor: "bxor",
}

_VECTORIZED_SEQUENTIAL_SCHEDULES = (
    dtypes.ScheduleType.Default,
    dtypes.ScheduleType.Sequential,
)


@dace.library.expansion
class ExpandReduceVectorized(pm.ExpandTransformation):
    """Schedule-aware ``"vectorized"`` expansion of a ``Reduce`` node."""

    environments = []

    @staticmethod
    def expansion(node: Reduce, state, sdfg):
        """Dispatch the reduction by operator and schedule.

        :param node: The ``Reduce`` library node.
        :param state: The state containing ``node``.
        :param sdfg: The SDFG containing ``state``.
        :returns: The expanded nested SDFG (delegated).
        :raises NotImplementedError: If the reduction operator is not an
            associative one with a horizontal-reduce primitive, or the
            node's schedule is neither sequential/default nor CPU
            multicore.
        """
        redtype = detect_reduction_type(node.wcr)
        if redtype not in REDTYPE_TO_OP:
            raise NotImplementedError(
                f"ExpandReduceVectorized: reduction operator {redtype} has no associative "
                f"horizontal-reduce lowering; supported: {sorted(t.name for t in REDTYPE_TO_OP)}. "
                f"wcr={node.wcr!r}")

        schedule = node.schedule
        if schedule in _VECTORIZED_SEQUENTIAL_SCHEDULES:
            # RV-1b-alpha: correct MVP via the proven pure expansion.
            # RV-1b-beta replaces this with the per-lane vector
            # accumulator + horizontal_reduce_<op> CPP kernel.
            return ExpandReducePure.expansion(node, state, sdfg)
        if schedule == dtypes.ScheduleType.CPU_Multicore:
            # OpenMP owns the thread split + cross-thread combine via its
            # reduction clause; the per-thread chunk is the inner kernel.
            return ExpandReduceOpenMP.expansion(node, state, sdfg)

        raise NotImplementedError(
            f"ExpandReduceVectorized: schedule {schedule} is not supported in the CPU "
            f"vectorization pipeline (expected Sequential/Default or CPU_Multicore). "
            f"A GPU/MPI/other-scheduled reduction must use its own implementation, "
            f"not 'vectorized'.")


Reduce.register_implementation("vectorized", ExpandReduceVectorized)
