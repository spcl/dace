# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The loop-cursor memlet access policy: addresses tracked by a loop-carried integer cursor."""

from dataclasses import dataclass, field

import dace.serialize
from dace import dtypes
from dace.sdfg.memlet_access_policy import MemletAccessPolicy, SymbolicLike, to_symbolic


@dace.serialize.serializable
@dataclass
class LoopCursor(MemletAccessPolicy):
    """The memlet's element offset is affine in the induction variable of an enclosing loop::

        offset(iteration) = base_invariant + lane_part + iteration_variable * (step / loop_stride)

    Policies of this kind are attached by hand, by a tuner, or by the
    :class:`~dace.transformation.passes.memlet_access_policies.assign_loop_cursors.AssignLoopCursors` analysis pass.
    Lowering (see
    :func:`~dace.transformation.passes.memlet_access_policies.loop_cursor_lowering.lower_loop_cursors`) materializes
    one loop-carried integer symbol per *cursor class* (same array, loop, step and lane part; memlets of a class differ
    only by a loop-invariant *immediate*), initialized in the loop's init statement and advanced in its update
    statement, plus a flat :class:`~dace.data.Reference` to the array's base set once at SDFG entry. Each memlet is
    rewritten to ``flat[cursor + immediate]`` (contiguous memlets) or to a *window* reference set per iteration to
    ``flat[cursor + immediate]`` and accessed with its original shape (non-contiguous reads). Loop nests chain: an
    inner cursor is initialized from the enclosing loop's cursor, one addition per level and no multiplication.

    :param loop: Label of the enclosing :class:`~dace.sdfg.state.LoopRegion` the policy is relative to, in the
                 memlet's SDFG or in an SDFG enclosing it along the memlet's path (the innermost loop with this
                 label and variable). The cursor lives in that loop's SDFG and is passed to the nested SDFGs
                 between the loop and the memlet as a symbol.
    :param variable: The loop's induction variable.
    :param step: Elements the address advances per loop iteration (the loop stride is already folded in).
                 ``step``, ``base_invariant`` and ``lane_part`` are in the symbols of the loop's SDFG.
    :param base_invariant: The part of the base element offset that does not depend on the loop variable or on
                           lane symbols (it may contain enclosing-loop variables, SDFG symbols, constants and
                           inner map parameters).
    :param lane_part: The part of the base element offset that depends on thread-block / lane map parameters.
    :param cursor_type: Integer type of the cursor symbol, or ``None`` for automatic (int32 when the array extent
                        is provably below 2**31, else int64).
    :param share_key: Optional explicit cursor-class override (planner/user): memlets of the same array and loop
                      with equal keys share one cursor if their addresses allow it, memlets with distinct keys
                      never do. ``None`` (the default) means "share whenever the addresses differ only by a
                      loop-invariant immediate".
    :param cursor: Name of the materialized cursor symbol (set by the lowering).
    :param reference: Name of the flat reference the rewritten memlet addresses (set by the lowering).
    :param window: Name of the per-iteration window reference, for non-contiguous memlets (set by the lowering).
    :param immediate: The loop-invariant element offset of this memlet relative to the cursor, in the symbols of
                      the memlet's SDFG (set by the lowering).
    """

    loop: str
    variable: str
    step: SymbolicLike = field(metadata={"kind": "symbolic"})
    base_invariant: SymbolicLike = field(default=0, metadata={"kind": "symbolic"})
    lane_part: SymbolicLike = field(default=0, metadata={"kind": "symbolic"})
    cursor_type: dtypes.typeclass | None = field(default=None, metadata={"kind": "dtype"})
    share_key: str | None = None
    cursor: str | None = None
    reference: str | None = None
    window: str | None = None
    immediate: SymbolicLike = field(default=None, metadata={"kind": "symbolic"})

    def __post_init__(self):
        for name in ("step", "base_invariant", "lane_part", "immediate"):
            setattr(self, name, to_symbolic(getattr(self, name)))
        if self.cursor_type is not None and not isinstance(self.cursor_type, dtypes.typeclass):
            raise TypeError(f"cursor_type must be a dace typeclass or None, got {self.cursor_type!r}")

    @property
    def is_lowered(self) -> bool:
        return self.cursor is not None and self.reference is not None

    @classmethod
    def lower(cls, sdfg, entries, **options) -> dict[str, int]:
        # dace.sdfg must not import dace.transformation at module load (import cycle)
        from dace.transformation.passes.memlet_access_policies.loop_cursor_lowering import lower_loop_cursors

        return lower_loop_cursors(sdfg, entries, **options)
