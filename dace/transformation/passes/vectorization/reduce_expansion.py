# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``"vectorized"`` implementation for the standard ``Reduce`` library node.

The vectorization pipeline lifts a recognised accumulator pattern into a
``dace.libraries.standard.nodes.Reduce`` with ``implementation="vectorized"``;
this module registers that implementation. :class:`ExpandReduceVectorized` is a
schedule-aware dispatcher (single ``ExpandTransformation``, as the DaCe library
API requires):

- ``Sequential`` / ``Default`` -> vectorized sequential reduction (per-lane
  vector accumulator folded by one ``horizontal_reduce_<op>`` intrinsic).
- ``CPU_Multicore`` -> :class:`ExpandReduceOpenMP`; its
  ``#pragma omp parallel for reduction(...)`` owns per-thread accumulators +
  cross-thread combine (thread count = OpenMP runtime concern,
  ``omp_get_max_threads()``, never an SDFG symbol; no manual ``M/NUM_CORES`` tiling).
- Any other schedule (GPU, MPI, Snitch, …) or unsupported reduction op -> raise
  ``NotImplementedError``. An unexpected schedule or a non-associative / custom
  reduction must surface, not be silently mis-lowered.
"""
from copy import deepcopy as dcpy

import dace
import dace.library
from dace import dtypes
from dace.frontend.operations import detect_reduction_type
from dace.libraries.standard.nodes.reduce import (
    ExpandReduceOpenMP,
    ExpandReducePure,
    Reduce,
)
from dace.symbolic import symstr
from dace.transformation import transformation as pm

#: Reduction ops with an associative identity + matching ``horizontal_reduce_<op>``
#: primitive; value = op-token suffix used by ``cpu_vectorizable_math_*.h``. Sub /
#: Div / Logical_* / *_Location / Exchange / Custom absent — no associative-fold
#: identity, must raise rather than mis-reduce.
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

#: Per-op C++ binary fold ``OP(x, y)`` for the W-wide partials + scalar tail;
#: paired with the identity element when ``Reduce`` carries no ``identity``.
_OP_CXX = {
    "add": lambda x, y: f"(({x}) + ({y}))",
    "mul": lambda x, y: f"(({x}) * ({y}))",
    "max": lambda x, y: f"std::max(({x}), ({y}))",
    "min": lambda x, y: f"std::min(({x}), ({y}))",
    "band": lambda x, y: f"(({x}) & ({y}))",
    "bor": lambda x, y: f"(({x}) | ({y}))",
    "bxor": lambda x, y: f"(({x}) ^ ({y}))",
}
_OP_IDENTITY_CXX = {
    "add": "({T})0",
    "mul": "({T})1",
    "max": "(-INFINITY)",
    "min": "(INFINITY)",
    "band": "(~(({T})0))",
    "bor": "({T})0",
    "bxor": "({T})0",
}

#: horizontal-reduce primitive is templated on a compile-time lane count; use the
#: pipeline default width (runtime header handles any width — single instruction
#: where the ISA has it, portable log-depth tree otherwise).
_VEC_W = 8


def _build_vectorized_full_reduction(node: Reduce, state, sdfg, opname: str):
    """Vectorized 1-D full-reduction nested SDFG, or ``None`` if out of scope.

    Scope: full reduction (every axis reduced) of a contiguous 1-D input to a
    single output element — the shape the vectorizer's lifted accumulators
    produce. Anything else (partial / multi-axis / multi-element output) ->
    ``None``, caller delegates to :class:`ExpandReducePure` (still correct).

    Kernel keeps ``_VEC_W`` partial accumulators, folds with one
    ``horizontal_reduce_<op>`` intrinsic, then a scalar tail handles the
    non-``W``-multiple remainder.

    :param node: the ``Reduce`` node.
    :param state: state containing ``node``.
    :param sdfg: containing SDFG.
    :param opname: the ``horizontal_reduce_<opname>`` suffix.
    :returns: expanded nested SDFG, or ``None`` if unsupported here.
    """
    node.validate(sdfg, state)
    inedge = state.in_edges(node)[0]
    outedge = state.out_edges(node)[0]
    insubset = dcpy(inedge.data.subset)
    isqdim = insubset.squeeze()
    outsubset = dcpy(outedge.data.subset)
    outsubset.squeeze()
    input_data = sdfg.arrays[inedge.data.data]
    output_data = sdfg.arrays[outedge.data.data]

    axes = node.axes if node.axes is not None else list(range(len(inedge.data.subset)))
    in_sizes = insubset.size()
    out_elems = 1
    for s in outsubset.size():
        out_elems *= s

    # Only contiguous 1-D full-reduction-to-scalar. Non-unit step (strided input
    # ``a[0:2N:2]``) can't use the contiguous SIMD load ``__inp[_i + _l]`` below ->
    # fall back to pure/OpenMP (strided SIMD gather not worth it).
    if (len(axes) != len(inedge.data.subset) or len(in_sizes) != 1 or out_elems != 1
            or any(str(step) != "1" for (_, _, step) in insubset.ranges)):
        return None

    ctype = input_data.dtype.ctype
    m_expr = symstr(in_sizes[0])

    nsdfg = dace.SDFG("reduce_vectorized")
    nsdfg.add_array("_in",
                    insubset.size(),
                    input_data.dtype,
                    strides=[s for i, s in enumerate(input_data.strides) if i in isqdim],
                    storage=input_data.storage)
    nsdfg.add_array("_out", outsubset.size(), output_data.dtype, storage=output_data.storage)
    nsdfg.append_global_code('#include "dace/cpu_vectorizable_math.h"')

    inedge._dst_conn = "_in"
    outedge._src_conn = "_out"
    node.add_in_connector("_in")
    node.add_out_connector("_out")

    if node.identity is not None:
        ident = f"({ctype})({node.identity})"
    else:
        ident = _OP_IDENTITY_CXX[opname].replace("{T}", ctype)
    fold = _OP_CXX[opname]
    W = _VEC_W
    # Per-lane loops have compile-time-constant trip (vector width): ``constexpr``
    # + full-unroll hint -> straight-line SIMD. The ``_i`` loop over runtime ``_M``
    # is data-dependent, stays a plain loop.
    code = f"""{{
constexpr int _W = {W};
const long long _M = (long long)({m_expr});
{ctype} _vacc[_W];
DACE_UNROLL
for (int _l = 0; _l < _W; ++_l) _vacc[_l] = {ident};
long long _i = 0;
for (; _i + _W <= _M; _i += _W) {{
  DACE_UNROLL
  for (int _l = 0; _l < _W; ++_l) _vacc[_l] = {fold("_vacc[_l]", "__inp[_i + _l]")};
}}
{ctype} _s = horizontal_reduce_{opname}<{ctype}, _W>(_vacc);
for (; _i < _M; ++_i) _s = {fold("_s", "__inp[_i]")};
__out = _s;
}}"""

    nstate = nsdfg.add_state()
    r = nstate.add_read("_in")
    w = nstate.add_write("_out")
    t = nstate.add_tasklet("reduce_vectorized", {"__inp"}, {"__out"}, code, dace.dtypes.Language.CPP)
    nstate.add_edge(r, None, t, "__inp", dace.Memlet(data="_in", subset=f"0:{m_expr}"))
    nstate.add_edge(t, "__out", w, None, dace.Memlet(data="_out", subset="0"))
    return nsdfg


@dace.library.expansion
class ExpandReduceVectorized(pm.ExpandTransformation):
    """Schedule-aware ``"vectorized"`` expansion of a ``Reduce`` node."""

    environments = []

    @staticmethod
    def expansion(node: Reduce, state, sdfg):
        """Dispatch the reduction by operator and schedule.

        :param node: the ``Reduce`` library node.
        :param state: state containing ``node``.
        :param sdfg: SDFG containing ``state``.
        :returns: expanded nested SDFG (delegated).
        :raises NotImplementedError: reduction op is not associative with a
            horizontal-reduce primitive, or schedule is neither sequential/default
            nor CPU multicore.
        """
        redtype = detect_reduction_type(node.wcr)
        if redtype not in REDTYPE_TO_OP:
            raise NotImplementedError(
                f"ExpandReduceVectorized: reduction operator {redtype} has no associative "
                f"horizontal-reduce lowering; supported: {sorted(t.name for t in REDTYPE_TO_OP)}. "
                f"wcr={node.wcr!r}")

        schedule = node.schedule
        if schedule in _VECTORIZED_SEQUENTIAL_SCHEDULES:
            # Vectorized kernel for the contiguous 1-D full-reduction-to-scalar
            # shape; other shapes fall back to pure expansion (still correct).
            # Loud-no-fallback policy applies to schedule/redtype, not to shape
            # coverage within the sequential path.
            opname = REDTYPE_TO_OP[redtype]
            vec = _build_vectorized_full_reduction(node, state, sdfg, opname)
            if vec is not None:
                return vec
            return ExpandReducePure.expansion(node, state, sdfg)
        if schedule == dtypes.ScheduleType.CPU_Multicore:
            # OpenMP owns thread split + cross-thread combine via its reduction
            # clause; per-thread chunk is the inner kernel.
            return ExpandReduceOpenMP.expansion(node, state, sdfg)

        raise NotImplementedError(f"ExpandReduceVectorized: schedule {schedule} is not supported in the CPU "
                                  f"vectorization pipeline (expected Sequential/Default or CPU_Multicore). "
                                  f"A GPU/MPI/other-scheduled reduction must use its own implementation, "
                                  f"not 'vectorized'.")


Reduce.register_implementation("vectorized", ExpandReduceVectorized)
