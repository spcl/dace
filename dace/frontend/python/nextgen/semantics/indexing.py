# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
The NumPy subscript model: one plan per subscript, built once.

A NumPy subscript decides several things at the same time — which source axes
survive, which collapse, where the advanced-indexing chunk lands, and where
``newaxis`` inserts — and every one of them is a function of the *index forms*
alone. Deriving them separately, as this frontend used to, means several
parallel bookkeeping structures that have to be kept in lockstep by hand;
that is what produced silently wrong result shapes when ``newaxis`` met an
advanced index.

:class:`IndexPlan` is that single derivation. It classifies each index slot
once (:class:`AxisKind`), then answers every downstream question from the same
:meth:`IndexPlan.result_layout`, so shapes, subsets and map ranges cannot
disagree with each other by construction.

NumPy's own rules, as implemented here (each verified against NumPy):

- a *slice*-formed axis survives at its size, **including size 1**
  (``A[0:1, :]`` is ``(1, 10)``, not ``(10,)``);
- an *integer*-formed axis is dropped;
- ``newaxis``/``None`` consumes no source axis and inserts a size-1 result axis;
- ``Ellipsis`` expands in place to as many whole axes as are left over;
- the advanced (array-valued) indices collapse into a single chunk whose shape
  is the broadcast of the index arrays. The chunk stays where the indices are
  when they form one run, and moves to the front when they are separated —
  and only a **slice, Ellipsis or newaxis separates**, never an integer index
  (``A[:, ind, 0, jnd]`` is ``(8, 3)``, whereas ``A[:, ind, :, jnd]`` is
  ``(3, 8, 4)``).
"""
import ast
import copy
import enum
from dataclasses import dataclass
from typing import Any, Callable, FrozenSet, List, Optional, Sequence, Tuple


class AxisKind(enum.Enum):
    """How one index slot relates to the source axis it consumes."""
    KEEP = 'keep'  #: Slice-formed: the axis survives in the result
    DROP = 'drop'  #: Integer-formed: the axis is removed from the result
    ADVANCED = 'advanced'  #: Array-valued: the axis folds into the broadcast chunk
    NEWAXIS = 'newaxis'  #: ``None``: consumes no source axis, inserts a size-1 axis


#: Result-entry kinds produced by :meth:`IndexPlan.result_layout`.
SOURCE = 'source'  #: One surviving source axis, named by :attr:`ResultAxis.axis`
NEWAXIS = 'newaxis'  #: An inserted size-1 axis
CHUNK = 'chunk'  #: The advanced-index chunk, standing for its full broadcast rank


@dataclass(frozen=True)
class IndexSlot:
    """One element of a subscript, and the source axis it consumes."""
    kind: AxisKind
    axis: Optional[int]  #: Source axis, or None for :attr:`AxisKind.NEWAXIS`


@dataclass(frozen=True)
class ResultAxis:
    """One axis of the NumPy result, in result order."""
    kind: str  #: :data:`SOURCE`, :data:`NEWAXIS`, or :data:`CHUNK`
    axis: Optional[int] = None  #: Source axis, for :data:`SOURCE` entries


def _is_newaxis(element: ast.expr) -> bool:
    return isinstance(element, ast.Constant) and element.value is None


def _is_ellipsis(element: ast.expr) -> bool:
    return isinstance(element, ast.Constant) and element.value is Ellipsis


@dataclass(frozen=True)
class IndexPlan:
    """
    The resolved per-axis meaning of one subscript.

    Build with :func:`build_plan`, which returns ``None`` for an index that
    cannot be modeled against the base rank (a malformed or not-yet-canonical
    subscript); callers must then fall back rather than guess.
    """
    slots: List[IndexSlot]
    ndim: int  #: Rank of the base container

    @property
    def advanced_axes(self) -> List[int]:
        """Source axes consumed by an array-valued index, in source order."""
        return [slot.axis for slot in self.slots if slot.kind is AxisKind.ADVANCED]

    @property
    def kept_dims(self) -> List[bool]:
        """
        Per source axis, whether it survives into the NumPy result.

        An advanced axis is *not* kept in this sense: it is consumed by the
        chunk, which is placed separately.
        """
        kept = [False] * self.ndim
        for slot in self.slots:
            if slot.kind is AxisKind.KEEP and slot.axis is not None:
                kept[slot.axis] = True
        return kept

    @property
    def separated(self) -> bool:
        """
        Whether the advanced indices are split apart, which moves the chunk to
        the front of the result.

        Only a slice, an ``Ellipsis`` (which expands to slices) or a
        ``newaxis`` separates. An integer index between two index arrays does
        not: NumPy treats it as part of the same advanced group.
        """
        advanced = [position for position, slot in enumerate(self.slots) if slot.kind is AxisKind.ADVANCED]
        if len(advanced) < 2:
            return False
        return any(self.slots[position].kind in (AxisKind.KEEP, AxisKind.NEWAXIS)
                   for position in range(advanced[0], advanced[-1]))

    def result_layout(self) -> List[ResultAxis]:
        """
        The result axes in order, with the advanced chunk as a single
        :data:`CHUNK` entry standing for its whole broadcast rank.

        This is the one derivation every consumer reads: result shape, result
        subset, and map-range placement are all this list with a different
        value substituted per entry.
        """
        advanced = [position for position, slot in enumerate(self.slots) if slot.kind is AxisKind.ADVANCED]
        separated = self.separated
        first_advanced = advanced[0] if advanced else None

        entries: List[ResultAxis] = []
        for position, slot in enumerate(self.slots):
            if slot.kind is AxisKind.ADVANCED:
                # The chunk is emitted once, at the first advanced slot, and
                # only when the group is contiguous; otherwise it is prepended
                # below.
                if not separated and position == first_advanced:
                    entries.append(ResultAxis(CHUNK))
                continue
            if slot.kind is AxisKind.DROP:
                continue
            if slot.kind is AxisKind.NEWAXIS:
                entries.append(ResultAxis(NEWAXIS))
            else:
                entries.append(ResultAxis(SOURCE, slot.axis))
        if separated:
            entries.insert(0, ResultAxis(CHUNK))
        return entries

    def place(self, per_axis: Sequence[Any], chunk: Sequence[Any] = (), newaxis: Any = 1) -> List[Any]:
        """
        Lay a per-source-axis quantity out in result order.

        Every derived quantity goes through here, which is the point of this
        class: a result shape, the subset written into it, and the strides of
        the view binding it are the same layout with a different value
        substituted, so they cannot disagree.

        :param per_axis: One value per SOURCE axis.
        :param chunk: The values contributed by the advanced-index chunk
                      (its broadcast shape, or the matching map ranges).
        :param newaxis: The value an inserted size-1 axis contributes.
        """
        result: List[Any] = []
        for entry in self.result_layout():
            if entry.kind == CHUNK:
                result.extend(chunk)
            elif entry.kind == NEWAXIS:
                result.append(newaxis)
            else:
                result.append(per_axis[entry.axis])
        return result

    def result_shape(self, sizes: Sequence[Any], chunk_shape: Sequence[Any] = ()) -> List[Any]:
        """
        The NumPy result shape.

        :param sizes: Per source axis, the extent the subset selects.
        :param chunk_shape: The broadcast shape of the index arrays; unused
                            when the subscript has no advanced index.
        """
        return self.place(sizes, chunk_shape, newaxis=1)

    def new_axis_positions(self, chunk_rank: int = 0) -> List[int]:
        """
        Where the inserted size-1 axes land in the FINAL result numbering.

        Reported against the post-collapse layout, so a chunk of rank > 1
        shifts them by its whole rank -- the renumbering that was previously
        done against the pre-collapse axis list and got this wrong.
        """
        positions: List[int] = []
        cursor = 0
        for entry in self.result_layout():
            if entry.kind == CHUNK:
                cursor += chunk_rank
            elif entry.kind == NEWAXIS:
                positions.append(cursor)
                cursor += 1
            else:
                cursor += 1
        return positions


def reverse_normalized(
        node: ast.Subscript, shape: Sequence[Any],
        constant_of: Callable[[Optional[ast.expr]], Optional[int]]) -> Tuple[ast.Subscript, FrozenSet[int]]:
    """
    Rewrite each negative-step slice into the equivalent FORWARD slice,
    reporting which axes are then traversed in reverse.

    The shared memlet parser applies forward-slice conventions unconditionally
    -- ``A[::-1]`` comes back as ``(0, 19, -1)`` and ``A[10:2:-1]`` as
    ``(10, 1, -1)`` -- so its ranges are unusable for a negative step: one has
    a negative extent, the other the wrong element count. Rewriting first hands
    it a range it computes correctly, and the *direction* travels separately
    (``DataAccess.reversed_axes``) to the one place it matters: forming the
    index, which then walks the axis from the top down.

    A memlet subset cannot express a reversal, which is why this is not a
    subset property. Only slices with a compile-time negative step (and
    compile-time bounds, since the element count depends on them) are
    rewritten; everything else is returned unchanged, so an unsupported
    spelling degrades exactly as it did before.

    :param constant_of: Resolves an index expression to a compile-time integer.
    :return: The rewritten subscript and the source axes now read in reverse.
    """
    ndim = len(shape)
    elements = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
    explicit = len([element for element in elements if not _is_newaxis(element) and not _is_ellipsis(element)])

    replacements: dict = {}
    reversed_axes = set()
    axis = 0
    for position, element in enumerate(elements):
        if _is_newaxis(element):
            continue  # Consumes no source axis
        if _is_ellipsis(element):
            axis += ndim - explicit  # Stands for however many whole axes are left
            continue
        if isinstance(element, ast.Slice) and axis < ndim:
            step = constant_of(element.step) if element.step is not None else None
            if step is not None and step < 0:
                forward = _forward_slice(element, step, shape[axis], constant_of)
                if forward is not None:
                    replacements[position] = forward
                    reversed_axes.add(axis)
        axis += 1

    if not replacements:
        return node, frozenset()

    rewritten = copy.deepcopy(node)
    target = rewritten.slice.elts if isinstance(rewritten.slice, ast.Tuple) else [rewritten.slice]
    for position, forward in replacements.items():
        target[position] = forward
    if not isinstance(rewritten.slice, ast.Tuple):
        rewritten.slice = target[0]
    return ast.fix_missing_locations(ast.copy_location(rewritten, node)), frozenset(reversed_axes)


def _forward_slice(element: ast.Slice, step: int, extent: Any,
                   constant_of: Callable[[Optional[ast.expr]], Optional[int]]) -> Optional[ast.Slice]:
    """
    The forward slice selecting the same elements as a negative-step slice.

    Follows Python's own defaults for a negative step: an omitted start is the
    last element, an omitted stop is one before the first. Returns None when a
    needed bound or the axis extent is not a compile-time integer, since the
    element count -- and hence the forward start -- cannot be derived without
    them.
    """
    if element.lower is None:
        try:
            extent = int(extent)
        except (TypeError, ValueError):
            return None  # A symbolic extent cannot supply the omitted default
        start = extent - 1
    else:
        start = constant_of(element.lower)
    stop = constant_of(element.upper) if element.upper is not None else -1
    if start is None or stop is None:
        return None
    stride = -step
    count = (start - stop + stride - 1) // stride
    if count <= 0:
        return None
    first = start - (count - 1) * stride
    return ast.Slice(lower=ast.Constant(value=first),
                     upper=ast.Constant(value=start + 1),
                     step=ast.Constant(value=stride))


def whole_container_plan(ndim: int) -> IndexPlan:
    """
    The plan of an access with no subscript at all: every axis survives.

    Distinct from ``plan=None``, which means "could not be modeled" and makes
    consumers fall back to squeezing. A whole-container read of a
    ``(1, 10)`` array is ``(1, 10)`` in NumPy, not ``(10,)``.
    """
    return IndexPlan(slots=[IndexSlot(AxisKind.KEEP, axis) for axis in range(ndim)], ndim=ndim)


def build_plan(slice_node: ast.expr, ndim: int, advanced_axes: FrozenSet[int] = frozenset()) -> Optional[IndexPlan]:
    """
    Classify a subscript's index against the base rank.

    :param slice_node: The canonical ``ast.Subscript.slice``.
    :param ndim: Rank of the base container.
    :param advanced_axes: Source axes indexed by an array, as the shared memlet
                          parser reports them (``MemletExpr.arrdims`` keys).
    :return: The plan, or None when the index cannot be modeled against
             ``ndim`` (more elements than axes, or several ``Ellipsis``) --
             callers must fall back rather than guess a shape.
    """
    raw = list(slice_node.elts) if isinstance(slice_node, ast.Tuple) else [slice_node]

    # ``newaxis`` consumes no source axis, so it plays no part in matching the
    # element count against the rank.
    consuming = [element for element in raw if not _is_newaxis(element)]
    ellipses = [element for element in consuming if _is_ellipsis(element)]
    if len(ellipses) > 1:
        return None
    explicit = len(consuming) - len(ellipses)
    if explicit > ndim:
        return None
    ellipsis_width = ndim - explicit if ellipses else 0

    slots: List[IndexSlot] = []
    axis = 0

    def consume(is_slice: bool) -> None:
        nonlocal axis
        if axis in advanced_axes:
            kind = AxisKind.ADVANCED
        else:
            kind = AxisKind.KEEP if is_slice else AxisKind.DROP
        slots.append(IndexSlot(kind, axis))
        axis += 1

    for element in raw:
        if _is_newaxis(element):
            slots.append(IndexSlot(AxisKind.NEWAXIS, None))
        elif _is_ellipsis(element):
            for _ in range(ellipsis_width):
                consume(is_slice=True)
        else:
            consume(is_slice=isinstance(element, ast.Slice))

    # Dimensions the subscript never mentions stay whole.
    while axis < ndim:
        consume(is_slice=True)

    return IndexPlan(slots=slots, ndim=ndim)
