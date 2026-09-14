# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Memlet schedules: how a leaf memlet's addressing is realized in generated code.

A *leaf* memlet is one that produces an address in generated code: a memlet bound to a tasklet, library node or
nested SDFG connector, or a copy between two access nodes. Its :attr:`~dace.memlet.Memlet.schedule` is a purely
*descriptive* record of how that address (and, for some kinds, the data movement it drives) is meant to be
realized. Schedules never change the semantics of an SDFG and contain no code-level state; they are attached by
analysis passes, tuners or by hand, and are *lowered* to ordinary SDFG constructs (symbols, loop statements,
reference containers, tasklets) by :class:`~dace.transformation.passes.memlet_schedules.LowerMemletSchedules` in
the code-generation window. Code generation itself knows nothing about schedules.

Kinds provided here:

* :class:`CopyOnAccess` -- the default and the behavior every SDFG had before schedules existed: the full offset
  expression is evaluated at every access, and data moves at the access (or through the copy library node that
  :class:`~dace.transformation.passes.insert_explicit_copies.InsertExplicitCopies` materializes for
  access-node-to-access-node copies). Lowering is a no-op.
* :class:`LoopCursor` -- the memlet's element offset is affine in the induction variable of an enclosing
  :class:`~dace.sdfg.state.LoopRegion`; lowering materializes a loop-carried integer *cursor* symbol (initialized
  in the loop's init statement, advanced in its update statement), a flat :class:`~dace.data.Reference` to the
  array's base, and rewrites the memlet to ``flat[cursor + immediate]``.

Further kinds (a buffer-resource-descriptor schedule whose cursor is the buffer instruction's ``voffset``, a
tensor-memory-accelerator schedule whose cursors are tile coordinates) follow the same protocol: a serializable
record plus a class-level :meth:`MemletSchedule.lower` that emits SDFG constructs.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import dace.serialize
from dace import symbolic

CURSOR_TYPES = ('auto', 'int32', 'int64')

SymbolicLike = Union[str, int, symbolic.SymbolicType, None]


def _to_sym(value: SymbolicLike) -> Optional[symbolic.SymbolicType]:
    if value is None:
        return None
    if isinstance(value, (str, int)):
        return symbolic.pystr_to_symbolic(value)
    return value


def _sym_json(value: Optional[symbolic.SymbolicType]):
    return None if value is None else symbolic.serialize_symbolic(value)


def _sym_from_json(value):
    return None if value is None else symbolic.deserialize_symbolic(value)


@dace.serialize.serializable
class MemletSchedule:
    """Base class of memlet schedules (see module docstring). Subclasses are serializable records; the
    ``type`` field of their JSON form names the subclass."""

    #: True for the schedule kind that reproduces the default behavior (not serialized).
    is_default = False

    # ------------------------------------------------------------------ protocol
    @property
    def is_lowered(self) -> bool:
        """True once the lowering pass materialized this schedule in the SDFG."""
        return False

    @classmethod
    def lower(cls, sdfg, entries: List[Tuple[Any, Any]], **options) -> Dict[str, int]:
        """Materialize all memlets of this schedule kind in one SDFG as ordinary SDFG constructs.

        Called by :class:`~dace.transformation.passes.memlet_schedules.LowerMemletSchedules` once per SDFG and
        schedule kind, with every ``(state, edge)`` whose memlet carries a schedule of this kind. The default does
        nothing (copy-on-access needs no lowering).

        :param sdfg: The SDFG (possibly nested) owning the states.
        :param entries: ``(state, edge)`` pairs of the leaf memlets to lower.
        :param options: Lowering options forwarded from the pass.
        :return: A dictionary of counters for the pass report (e.g. ``{'memlets': n, 'dropped': d}``).
        """
        return {}

    # ------------------------------------------------------------------ value semantics
    def copy(self) -> 'MemletSchedule':
        return type(self)()

    def __deepcopy__(self, memo):
        return self.copy()

    def __eq__(self, other) -> bool:
        if not isinstance(other, MemletSchedule):
            return NotImplemented
        return self.to_json() == other.to_json()

    def __hash__(self) -> int:
        return hash(type(self).__name__)

    def __repr__(self) -> str:
        return f'{type(self).__name__}()'

    # ------------------------------------------------------------------ (de)serialization
    def to_json(self):
        return {'type': type(self).__name__}

    @staticmethod
    def from_json(json_obj, context=None):
        typename = json_obj['type']
        if typename == 'MemletSchedule':
            raise TypeError('MemletSchedule is abstract; the JSON "type" must name a schedule kind')
        return dace.serialize.get_serializer(typename).from_json(json_obj, context)


@dace.serialize.serializable
class CopyOnAccess(MemletSchedule):
    """The default schedule: the full offset is evaluated at every access and data moves at the access (or through
    the copy library node inserted for access-node-to-access-node copies). Lowering is a no-op."""

    is_default = True

    @staticmethod
    def from_json(json_obj, context=None):
        return CopyOnAccess()


@dace.serialize.serializable
class LoopCursor(MemletSchedule):
    """The memlet's element offset is affine in the induction variable of an enclosing loop::

        offset(iteration) = base_invariant + lane_part + iteration_variable * (step / loop_stride)

    Lowering (see :func:`~dace.transformation.passes.memlet_schedules.lower_loop_cursors`) materializes one
    loop-carried integer symbol per *cursor class* (same array, loop, step and lane part; memlets of a class differ
    only by a loop-invariant *immediate*), initialized in the loop's init statement and advanced in its update
    statement, plus a flat :class:`~dace.data.Reference` to the array's base set once at SDFG entry. Each memlet is
    rewritten to ``flat[cursor + immediate]`` (contiguous memlets) or to a *window* reference set per iteration to
    ``flat[cursor + immediate]`` and accessed with its original shape (non-contiguous reads). Loop nests chain: an
    inner cursor is initialized from the enclosing loop's cursor, one addition per level and no multiplication.

    :param loop: Label of the enclosing :class:`~dace.sdfg.state.LoopRegion` the schedule is relative to.
    :param variable: The loop's induction variable.
    :param step: Elements the address advances per loop iteration (the loop stride is already folded in).
    :param base_invariant: The part of the base element offset that does not depend on the loop variable or on
                           lane symbols (it may contain enclosing-loop variables, SDFG symbols, constants and
                           inner map parameters).
    :param lane_part: The part of the base element offset that depends on thread-block / lane map parameters.
    :param cursor_type: ``'auto'`` (int32 when the array extent is provably below 2**31, else int64),
                        ``'int32'`` or ``'int64'``.
    :param share_key: Optional explicit cursor-class override (planner/user): memlets of the same array and loop
                      with equal keys share one cursor if their addresses allow it, memlets with distinct keys
                      never do. ``None`` (the default) means "share whenever the addresses differ only by a
                      loop-invariant immediate".
    :param cursor: Name of the materialized cursor symbol (set by the lowering).
    :param reference: Name of the flat reference the rewritten memlet addresses (set by the lowering).
    :param window: Name of the per-iteration window reference, for non-contiguous memlets (set by the lowering).
    :param immediate: The loop-invariant element offset of this memlet relative to the cursor (set by the lowering).
    """

    def __init__(self,
                 loop: str,
                 variable: str,
                 step: SymbolicLike,
                 base_invariant: SymbolicLike = 0,
                 lane_part: SymbolicLike = 0,
                 cursor_type: str = 'auto',
                 share_key: Optional[str] = None,
                 cursor: Optional[str] = None,
                 reference: Optional[str] = None,
                 window: Optional[str] = None,
                 immediate: SymbolicLike = None):
        if cursor_type not in CURSOR_TYPES:
            raise ValueError(f'Unknown cursor type {cursor_type!r}; expected one of {CURSOR_TYPES}')
        self.loop = loop
        self.variable = variable
        self.step = _to_sym(step)
        self.base_invariant = _to_sym(base_invariant)
        self.lane_part = _to_sym(lane_part)
        self.cursor_type = cursor_type
        self.share_key = share_key
        self.cursor = cursor
        self.reference = reference
        self.window = window
        self.immediate = _to_sym(immediate)

    @property
    def is_lowered(self) -> bool:
        return self.cursor is not None and self.reference is not None

    @classmethod
    def lower(cls, sdfg, entries, **options) -> Dict[str, int]:
        from dace.transformation.passes.memlet_schedules import lower_loop_cursors  # avoid import cycle
        return lower_loop_cursors(sdfg, entries, **options)

    def copy(self) -> 'LoopCursor':
        return LoopCursor(self.loop, self.variable, self.step, self.base_invariant, self.lane_part, self.cursor_type,
                          self.share_key, self.cursor, self.reference, self.window, self.immediate)

    def __hash__(self) -> int:
        return hash((self.loop, self.variable, str(self.step), self.cursor))

    def __repr__(self) -> str:
        lowered = f', cursor={self.cursor}, immediate={self.immediate}' if self.is_lowered else ''
        return (f'LoopCursor(loop={self.loop!r}, var={self.variable}, step={self.step}, '
                f'base={self.base_invariant}, lane={self.lane_part}, type={self.cursor_type}{lowered})')

    def to_json(self):
        return {
            'type': 'LoopCursor',
            'loop': self.loop,
            'variable': self.variable,
            'step': _sym_json(self.step),
            'base_invariant': _sym_json(self.base_invariant),
            'lane_part': _sym_json(self.lane_part),
            'cursor_type': self.cursor_type,
            'share_key': self.share_key,
            'cursor': self.cursor,
            'reference': self.reference,
            'window': self.window,
            'immediate': _sym_json(self.immediate),
        }

    @staticmethod
    def from_json(json_obj, context=None):
        return LoopCursor(loop=json_obj['loop'],
                          variable=json_obj['variable'],
                          step=_sym_from_json(json_obj.get('step')),
                          base_invariant=_sym_from_json(json_obj.get('base_invariant')),
                          lane_part=_sym_from_json(json_obj.get('lane_part')),
                          cursor_type=json_obj.get('cursor_type', 'auto'),
                          share_key=json_obj.get('share_key'),
                          cursor=json_obj.get('cursor'),
                          reference=json_obj.get('reference'),
                          window=json_obj.get('window'),
                          immediate=_sym_from_json(json_obj.get('immediate')))
