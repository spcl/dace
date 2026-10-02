# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""How a tile node reads its operands."""
from collections.abc import Sequence

import dace
from dace.libraries.tileops.validation import is_tile_shape


def scalar_operand_ref(desc: dace.data.Data, conn: str, widths: Sequence[int], off: str) -> tuple[str, bool]:
    """Per-lane C++ reference for a ``Scalar``-kind tile-op operand.

    A ``Scalar``-kind operand (one classified as a broadcast because its source
    is read through a single-element ``"0"`` memlet) may be bound to one of two
    connector ABIs:

    * a **tile-shape** :class:`dace.data.Array` (``shape == widths``) -> a
      transient an upstream tile op widened to a register tile, then read here
      through a single-element memlet. The connector is a pointer (``T* conn``)
      carrying PER-LANE data, so it must be read ``conn[off]`` exactly like a
      Tile operand. Reading it as a broadcast would emit ``(T)conn`` -- an
      invalid pointer-to-value cast.
    * anything else (a true :class:`dace.data.Scalar`, a length-1 Array, or any
      single-element access) -> DaCe passes a volume-1 connector by value
      (``T conn = ...``), so the tasklet references the bare ``conn`` and
      broadcasts it. ``[0]`` is a *memlet* concern, never a tasklet-body one --
      a by-value ``conn`` is not a pointer.

    :param desc: The data descriptor bound to ``conn``.
    :param conn: The tasklet input connector name (e.g. ``"_a"``).
    :param widths: Per-dim tile widths (innermost-last).
    :param off: The flattened per-lane offset expression (from ``tile_offset``).
    :returns: ``(ref, broadcast)`` -- the C++ reference and whether it is a
        loop-invariant broadcast. The caller casts a broadcast to the operand
        dtype; a per-lane tile read (``broadcast == False``) keeps the tile
        dtype uncast, exactly like a Tile operand.
    """
    if isinstance(desc, dace.data.Array) and is_tile_shape(desc, tuple(widths)):
        return f"{conn}[{off}]", False
    return conn, True
