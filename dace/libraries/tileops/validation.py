# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""What a tile node requires of the descriptors and memlets wired to it, and which dtype promotions it allows."""
from collections.abc import Sequence

import numpy as np
import sympy

import dace
from dace.sdfg import graph


def is_tile_shape(desc: dace.data.Data, widths: Sequence[int]) -> bool:
    """True iff ``desc`` is an :class:`dace.data.Array` whose shape equals ``widths``."""
    if not isinstance(desc, dace.data.Array):
        return False
    shape = tuple(desc.shape)
    if len(shape) != len(widths):
        return False
    return all(bool(dace.symbolic.simplify(s - w) == 0) for s, w in zip(shape, widths))


def edge_moves_a_tile(edge: graph.MultiConnectorEdge[dace.Memlet], widths: Sequence[int]) -> bool:
    """True iff ``edge``'s MEMLET moves a tile-shaped box, whatever its descriptor's shape.

    :class:`WidenAccesses` widens the memlet of a lane-indexed array in place rather than swapping
    the descriptor -- CloudSC's ``zsolqa[jm, jn, jl]`` keeps its ``(nclv, nclv, klon)`` shape -- so a
    tile write lands in a WINDOW of a larger array: ``buf[0, i:i+W]`` on a ``(3, N)`` array is a tile
    even though ``(3, N) != (W,)``. Judging that by the descriptor alone reports a rule violation on
    a perfectly good tile.

    The leading dims must each be a single element and the trailing dims must be the tile itself,
    which is exactly what makes the expansions' ``_c[off]`` walk the widened window and nothing else.

    :param edge: the ``_c`` / ``_o`` output edge.
    :param widths: the node's per-dim tile widths.
    """
    if edge.data is None or edge.data.subset is None:
        return False
    size = tuple(edge.data.subset.size())
    if len(size) < len(widths):
        return False
    split = len(size) - len(widths)
    if any(not bool(dace.symbolic.simplify(s - 1) == 0) for s in size[:split]):
        return False
    return dace.symbolic.shapes_equal(size[split:], tuple(widths))


def edge_moves_one_element(edge: graph.MultiConnectorEdge[dace.Memlet]) -> bool:
    """True iff ``edge``'s MEMLET moves exactly one element, whatever its descriptor's shape.

    Codegen binds a one-element connector BY VALUE (``T _c;``), so a lane-invariant op writing it
    assigns ``_c`` once and never walks ``_c[off]``. The descriptor cannot tell: CloudSC's
    ``imelt[4] = -99`` writes one element of an ``int[5]``.

    :param edge: the ``_c`` / ``_o`` output edge.
    """
    return edge.data is not None and edge.data.subset is not None and edge.data.subset.num_elements() == 1


def is_floating_dtype(dtype: dace.dtypes.typeclass) -> bool:
    """Whether ``dtype`` is a floating type, the ml_dtypes-backed narrow ones included.

    ``np.issubdtype(ml_dtypes.bfloat16, np.floating)`` is False -- ml_dtypes registers its scalars
    outside numpy's float hierarchy -- so a bare numpy test reads ``bfloat16`` and the two fp8 types
    as neither integer nor float, and :func:`promotion_ok` then refuses EVERY promotion off them,
    a comparison's ``-> bool`` included.

    :param dtype: The dtype to classify.
    :returns: ``True`` iff ``dtype`` holds floating-point values.
    """
    return dtype in dace.dtypes.FLOAT_TYPES or np.issubdtype(dtype.type, np.floating)


def promotion_ok(src: dace.dtypes.typeclass, dst: dace.dtypes.typeclass) -> bool:
    """Whether a Tile operand of dtype ``src`` may be promoted to the output
    dtype ``dst`` before the op (a widening conversion).

    Allowed (widening): same dtype; integer -> wider-or-equal integer; integer
    -> float / double; float -> double; integer -> bool (truthiness cast,
    ``int != 0`` — well-defined in C++ for any integer operand, used by the
    merge-cond compound combine where a comparison result stored as int64
    flows into a bool-output combine tasklet). Disallowed (narrowing -> the
    caller must crash): float / double -> integer; double -> float; integer
    narrowing (e.g. int64 -> int32).

    :param src: The Tile operand's element dtype.
    :param dst: The output (``_c``) element dtype.
    :returns: ``True`` iff promoting ``src`` to ``dst`` is non-narrowing.
    """
    if src == dst:
        return True
    s_int = np.issubdtype(src.type, np.integer)
    d_int = np.issubdtype(dst.type, np.integer)
    s_flt = is_floating_dtype(src)
    d_flt = is_floating_dtype(dst)
    s_bool = (src.type is np.bool_)
    d_bool = (dst.type is np.bool_)
    if s_int and d_flt:  # int -> float / double
        return True
    if s_int and d_int and dst.bytes >= src.bytes:  # integer widening
        return True
    # Strictly wider, not wider-or-equal: two DISTINCT floats of the same width -- float16
    # against bfloat16, or the two fp8 encodings -- trade mantissa for exponent, so neither
    # direction round-trips. The equal case that is safe is the same dtype, returned above.
    if s_flt and d_flt and dst.bytes > src.bytes:  # float -> double
        return True
    if (s_int or s_bool or s_flt) and d_bool:  # numeric -> bool (truthiness)
        return True
    return False


def strides_match_packed(shape: Sequence[int | sympy.Basic], strides: Sequence[int | sympy.Basic], order: str) -> bool:
    """True when ``strides`` is the packed contiguous form for ``shape`` in
    ``order`` ("C" -- innermost-last, stride 1 on the last dim; or "F" --
    innermost-first, stride 1 on the first dim) with NO padding between dims.

    Symbolic shapes / strides are compared via sympy ``simplify == 0``.

    :param shape: Tuple of dim sizes (may be symbolic).
    :param strides: Tuple of per-dim strides (may be symbolic).
    :param order: "C" or "F".
    :returns: ``True`` iff the layout is exactly packed in the requested order.
    """
    if len(shape) != len(strides):
        return False
    if order == "C":
        order_range = range(len(shape) - 1, -1, -1)
    elif order == "F":
        order_range = range(len(shape))
    else:
        raise ValueError(f"order must be 'C' or 'F'; got {order!r}")
    expected = 1
    for d in order_range:
        try:
            # relax_ipow so the canonicalized packed-C stride ``ipow(N, 2)`` compares equal to
            # ``N*N``; the opaque ``ipow`` never simplifies against ``expected`` (heat3d).
            # Equalize before simplifying: a stride and a shape dim can carry two same-named symbol
            # INSTANCES (different dtype/assumptions) whose subtraction never cancels (channel_flow).
            diff = strides[d] - expected
            if isinstance(diff, sympy.Basic):
                diff = dace.symbolic.simplify(dace.symbolic.relax_ipow(dace.symbolic.equalize_symbol(diff)))
            if diff != 0:
                return False
        except Exception:  # noqa: BLE001 -- conservative refusal on un-comparable expressions.
            return False
        expected = expected * shape[d]
    return True


def validate_packed_layout(node_label: str, conn_name: str, desc: dace.data.Data) -> None:
    """Refuse any source / dest array whose stride pattern is neither packed C
    nor packed Fortran (design section 2.3).

    Padded layouts -- where strides exceed the product of inner dims -- raise
    :class:`NotImplementedError` until per-arch codegen support lands. 1-D
    arrays trivially satisfy both packings and are accepted iff their single
    stride is 1.

    :param node_label: Label of the calling lib node (for error messages).
    :param conn_name: Connector name carrying the array (typically ``_src``
        or ``_dst``).
    :param desc: The array descriptor (``dace.data.Data`` subclass) wired to
        the connector.
    :raises NotImplementedError: On non-packed-C non-packed-Fortran layout.
    """
    if not isinstance(desc, dace.data.Array):
        return  # Scalars / Streams have no per-dim stride pattern to check.
    shape = tuple(desc.shape)
    strides = tuple(desc.strides)
    if len(shape) == 0:
        return
    if len(shape) == 1:
        try:
            if dace.symbolic.simplify(strides[0] - 1) != 0:
                raise NotImplementedError(f"{node_label}: {conn_name!r} has non-unit stride "
                                          f"{strides[0]} on its single dim; only packed layouts are "
                                          f"supported (section 2.3).")
        except NotImplementedError:
            raise
        except Exception:  # noqa: BLE001
            raise NotImplementedError(f"{node_label}: {conn_name!r} stride {strides[0]} could not be "
                                      f"verified against the packed-layout invariant (section 2.3).")
        return
    if not (strides_match_packed(shape, strides, "C") or strides_match_packed(shape, strides, "F")):
        raise NotImplementedError(f"{node_label}: {conn_name!r} has non-packed stride pattern "
                                  f"(shape={shape}, strides={strides}). Only packed-C and packed-"
                                  f"Fortran layouts are supported (section 2.3); padded layouts raise "
                                  f"NotImplementedError until codegen lands.")


def validate_mask_descriptor_lock(node_label: str, conn_name: str, desc: dace.data.Data, widths: Sequence[int]) -> None:
    """Refuse any mask descriptor that breaks the design section 10.2 lock.

    The locked shape: ``Array(shape=widths, dtype=bool_, storage=Register,
    transient=True)``. Anything else -- scalar masks, per-dim masks, non-bool
    predicates, non-Register storage, non-transient -- is rejected with a
    named error so the codegen never silently mis-emits.

    :param node_label: Label of the calling lib node (for error messages).
    :param conn_name: Connector name carrying the mask (typically ``_mask``
        or ``_o``).
    :param desc: The descriptor (``dace.data.Data`` subclass) of the array
        wired to the connector.
    :param widths: Tile widths ``(W_0, ..., W_{K-1})``.
    :raises ValueError: On any descriptor lock violation.
    """
    if not isinstance(desc, dace.data.Array):
        raise ValueError(f"{node_label}: {conn_name!r} mask must be a dace.data.Array, "
                         f"got {type(desc).__name__}")
    if tuple(desc.shape) != tuple(widths):
        raise ValueError(f"{node_label}: {conn_name!r} mask shape {tuple(desc.shape)} must "
                         f"match widths {tuple(widths)} (section 10.2)")
    if desc.dtype != dace.bool_:
        raise ValueError(f"{node_label}: {conn_name!r} mask dtype {desc.dtype} must be bool_ "
                         f"(section 10.2)")
    if desc.storage != dace.dtypes.StorageType.Register:
        raise ValueError(f"{node_label}: {conn_name!r} mask storage {desc.storage} must be "
                         f"Register (section 10.2)")
    if not desc.transient:
        raise ValueError(f"{node_label}: {conn_name!r} mask must be transient (section 10.2)")
