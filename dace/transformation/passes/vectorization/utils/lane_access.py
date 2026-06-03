# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Canonical per-lane access classification for vectorization.

One primitive — :func:`classify_lane_access` — answers the single
question every vectorization analysis was hand-rolling its own variant
of: *given a memlet subset, the accessed array's strides, and the
vectorized map's lane parameter, how does the access move as the lane
parameter advances by one?*

Before this module the same analysis was duplicated (and drifting) in
four places:

- ``utils.subsets.expand_memlet_expression`` — which dim to widen to W.
- ``utils.queries.collect_vectorizable_arrays`` /
  ``collect_non_unit_stride_accesses_in_map`` — strided/transposed/
  diagonal ⇒ not contiguously vectorizable.
- ``generate_iteration_mask.GenerateIterationMask._subset_fans_out`` —
  does the access fan out per lane (unsafe masked remainder)?
- ``vectorize.Vectorize._setup_strided_nsdfg_edges_inline`` — the
  inter-lane stride for the strided-load handler.

Each derives its answer from the same facts; they now all delegate
here. ``LaneAccessKind`` is a value enum and ``LaneAccess`` is a frozen
data record (no behaviour) — the only sanctioned classes per the
no-OOP-abstractions rule, same category as the name-scheme dataclasses.
"""
import enum
from dataclasses import dataclass
from typing import List, Optional, Sequence

import dace
from dace.transformation.passes.vectorization.utils.symbolic_polymorphism import free_symbol_names


class LaneAccessKind(enum.Enum):
    """How an access moves as the lane parameter advances by one."""

    #: The lane parameter does not appear in the subset (loop-invariant
    #: along the lane axis — a broadcast / constant-index read).
    CONSTANT = "constant"
    #: Unit-stride contiguous: the lane parameter appears, with
    #: coefficient 1, in the array's stride-1 dimension. A plain W-load.
    CONTIGUOUS = "contiguous"
    #: The lane parameter appears in the stride-1 dimension but with a
    #: coefficient > 1 (``A[2*i]``). W elements at a constant stride.
    STRIDED = "strided"
    #: The lane parameter appears in a single dimension whose array
    #: stride is not 1 (``zqx[z1, i, j]`` — transposed). W elements at
    #: the array stride of that dimension.
    TRANSPOSED = "transposed"
    #: The lane parameter appears in more than one dimension
    #: (``A[i, i]`` diagonal / ``A[2*i, i]`` linear combo). The
    #: inter-lane memory stride is the linearised sum over those dims.
    DIAGONAL = "diagonal"


@dataclass(frozen=True)
class LaneAccess:
    """Result of :func:`classify_lane_access` (data only, no behaviour).

    :param kind: The :class:`LaneAccessKind`.
    :param lane_dim: The single subset dimension carrying the lane
        parameter, or ``None`` for ``CONSTANT`` / ``DIAGONAL``.
    :param inter_lane_stride: Index-space step of the lane dimension's
        ``begin`` per lane (1 for ``A[i]``, ``c`` for ``A[c*i]``), or
        ``None`` if it could not be reduced to a concrete int.
    :param memory_stride: Element distance between what lane ``l`` and
        lane ``l+1`` touch. ``1`` for ``CONTIGUOUS``;
        ``array_stride[lane_dim] * inter_lane_stride`` for ``STRIDED`` /
        ``TRANSPOSED``; the linearised sum
        ``Σ_d coeff(d) * array_stride[d]`` for ``DIAGONAL``; ``None`` if
        not a concrete int.
    """

    kind: LaneAccessKind
    lane_dim: Optional[int] = None
    inter_lane_stride: Optional[int] = None
    memory_stride = None  # set in __post_init__ via object.__setattr__

    def __init__(self,
                 kind: LaneAccessKind,
                 lane_dim: Optional[int] = None,
                 inter_lane_stride: Optional[int] = None,
                 memory_stride=None):
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "lane_dim", lane_dim)
        object.__setattr__(self, "inter_lane_stride", inter_lane_stride)
        object.__setattr__(self, "memory_stride", memory_stride)

    @property
    def is_contiguous(self) -> bool:
        """A plain unit-stride W-load (or lane-invariant)."""
        return self.kind in (LaneAccessKind.CONTIGUOUS, LaneAccessKind.CONSTANT)

    @property
    def fans_out_per_lane(self) -> bool:
        """The access lowers to a per-lane gather/scatter fan."""
        return self.kind in (LaneAccessKind.STRIDED, LaneAccessKind.TRANSPOSED, LaneAccessKind.DIAGONAL)


def _has_param(expr, lane_param: str) -> bool:
    return lane_param in free_symbol_names(expr)


def _index_step(begin, lane_param: str) -> Optional[int]:
    """``begin(lane+1) - begin(lane)`` on the dace symbolic begin.

    Uses the *dace* symbol named ``lane_param`` (not a raw
    ``sympy.Symbol``: a raw symbol does not unify with the dace symbol
    carried in ``begin``, so ``.subs`` no-ops and ``int(...)`` raises —
    the K-elements-per-iter bug). Returns ``None`` if it does not reduce
    to a concrete int.
    """
    sym = dace.symbolic.symbol(lane_param)
    try:
        return int(dace.symbolic.simplify(begin.subs(sym, sym + 1) - begin))
    except (TypeError, ValueError, AttributeError):
        return None


def classify_lane_access(subset, array_strides: Sequence, lane_param: str) -> LaneAccess:
    """Classify how ``subset`` moves as ``lane_param`` advances by one.

    :param subset: The memlet subset (an iterable of ``(begin, end,
        step)`` per dimension).
    :param array_strides: The accessed array's per-dimension strides.
    :param lane_param: The vectorized map's innermost loop parameter.
    :returns: A :class:`LaneAccess` record.
    """
    dims = list(subset)
    strides = list(array_strides)
    if len(strides) != len(dims):
        # Shape/subset rank mismatch (collapsed dims etc.) — treat as
        # lane-invariant; callers fall back to their legacy path.
        return LaneAccess(LaneAccessKind.CONSTANT)

    param_dims: List[int] = [d for d, (b, _e, _s) in enumerate(dims) if _has_param(b, lane_param)]
    if not param_dims:
        return LaneAccess(LaneAccessKind.CONSTANT)

    if len(param_dims) > 1:
        # Diagonal / linear-combo: linearised inter-lane memory stride is
        # Σ_d step_d * array_stride[d] over the param-bearing dims.
        mem = 0
        ok = True
        for d in param_dims:
            step_d = _index_step(dims[d][0], lane_param)
            if step_d is None:
                ok = False
                break
            mem = mem + step_d * strides[d]
        return LaneAccess(LaneAccessKind.DIAGONAL, lane_dim=None, inter_lane_stride=None,
                          memory_stride=(mem if ok else None))

    d = param_dims[0]
    begin = dims[d][0]
    step = _index_step(begin, lane_param)
    is_stride1_dim = str(strides[d]) == "1"
    if not is_stride1_dim:
        mem = (strides[d] * step) if step is not None else None
        return LaneAccess(LaneAccessKind.TRANSPOSED, lane_dim=d, inter_lane_stride=step, memory_stride=mem)
    # Stride-1 dim: contiguous iff the lane parameter is not a >1
    # multiplicative factor (step == 1); otherwise strided.
    if step == 1:
        return LaneAccess(LaneAccessKind.CONTIGUOUS, lane_dim=d, inter_lane_stride=1, memory_stride=1)
    return LaneAccess(LaneAccessKind.STRIDED, lane_dim=d, inter_lane_stride=step,
                      memory_stride=(step if step is not None else None))
