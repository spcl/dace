# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Memlet access policies: how a leaf memlet's addressing is realized in generated code.

A *leaf* memlet is one that produces an address in generated code: the innermost memlet of a memlet path, i.e.
the one bound to a tasklet, library node or to an access node inside a scope, or a copy between two access nodes.
Memlet paths are global: a nested SDFG connector receives the whole container (an equivalent descriptor, no
offset), so a path continues through the chain of nested SDFGs to the leaf memlets inside them, and the policy of a
leaf memlet may place its address computation anywhere along that path (e.g., relative to a loop of an enclosing
SDFG). The leaf memlet's :attr:`~dace.memlet.Memlet.access_policy` is a purely *descriptive* record of how that
address (and, for some kinds, the data movement it drives) is meant to be realized. Policies never change the
semantics of an SDFG and contain no code-level state; they are attached by analysis passes, tuners or by hand, and
are *lowered* to ordinary SDFG constructs (symbols, loop statements, reference containers, tasklets) by
:class:`~dace.transformation.passes.memlet_access_policies.LowerMemletAccessPolicies` in the code-generation window.
Code generation itself knows nothing about policies.

Kinds provided here:

* :class:`CopyOnAccess` -- the default and the behavior every SDFG had before access policies existed: the full offset
  expression is evaluated at every access, and data moves at the access (or through the copy library node that
  :class:`~dace.transformation.passes.insert_explicit_copies.InsertExplicitCopies` materializes for
  access-node-to-access-node copies). Lowering is a no-op.
* :class:`LoopCursor` -- the memlet's element offset is affine in the induction variable of an enclosing
  :class:`~dace.sdfg.state.LoopRegion`; lowering materializes a loop-carried integer *cursor* symbol (initialized
  in the loop's init statement, advanced in its update statement), a flat :class:`~dace.data.Reference` to the
  array's base, and rewrites the memlet to ``flat[cursor + immediate]``.

Further kinds (a buffer-resource-descriptor policy whose cursor is the buffer instruction's ``voffset``, a
tensor-memory-accelerator policy whose cursors are tile coordinates) follow the same protocol: a dataclass
record plus a class-level :meth:`MemletAccessPolicy.lower` that emits SDFG constructs.

Policies are plain dataclasses. Fields holding symbolic expressions or data types are marked with the field
metadata ``kind='symbolic'`` / ``kind='dtype'`` so that the generic JSON (de)serialization in the base class can
convert them.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import sympy

import dace.serialize
from dace import dtypes, symbolic

SymbolicLike = Union[str, int, symbolic.SymbolicType, None]


def _to_sym(value: SymbolicLike) -> Optional[symbolic.SymbolicType]:
    if value is None or isinstance(value, sympy.Basic):
        return value
    return symbolic.pystr_to_symbolic(value)


def _field_to_json(f: dataclasses.Field, value):
    kind = f.metadata.get("kind")
    if value is None:
        return None
    if kind == "symbolic":
        return symbolic.serialize_symbolic(value)
    if kind == "dtype":
        return value.to_json()
    return value


def _field_from_json(f: dataclasses.Field, value, context):
    kind = f.metadata.get("kind")
    if value is None:
        return None
    if kind == "symbolic":
        return symbolic.deserialize_symbolic(value)
    if kind == "dtype":
        return dtypes.json_to_typeclass(value, context)
    return value


@dace.serialize.serializable
@dataclass
class MemletAccessPolicy:
    """Base class of memlet access policies (see module docstring). Subclasses are dataclasses; the ``type`` field of
    their JSON form names the subclass."""

    #: True for the policy kind that reproduces the default behavior (not serialized).
    is_default: ClassVar[bool] = False

    @property
    def is_lowered(self) -> bool:
        """True once the lowering pass materialized this policy in the SDFG."""
        return False

    @classmethod
    def lower(cls, sdfg, entries: List[Tuple[Any, Any]], **options) -> Dict[str, int]:
        """Materialize all memlets of this policy kind in one SDFG as ordinary SDFG constructs.

        Called by :class:`~dace.transformation.passes.memlet_access_policies.LowerMemletAccessPolicies` once per
        policy kind, with every ``(state, edge)`` of the SDFG and its nested SDFGs whose memlet carries a policy of
        this kind (the memlet paths cross nested SDFGs, so one lowering may touch several of them). The default
        does nothing (copy-on-access needs no lowering).

        :param sdfg: The root SDFG.
        :param entries: ``(state, edge)`` pairs of the leaf memlets to lower.
        :param options: Lowering options forwarded from the pass.
        :return: A dictionary of counters for the pass report (e.g. ``{'memlets': n, 'dropped': d}``).
        """
        return {}

    def copy(self) -> "MemletAccessPolicy":
        return dataclasses.replace(self)

    def to_json(self):
        result = {"type": type(self).__name__}
        result.update({f.name: _field_to_json(f, getattr(self, f.name)) for f in dataclasses.fields(self)})
        return result

    @staticmethod
    def from_json(json_obj, context=None):
        typename = json_obj["type"]
        if typename == "MemletAccessPolicy":
            raise TypeError('MemletAccessPolicy is abstract; the JSON "type" must name a policy kind')
        cls = dace.serialize.get_serializer(typename)
        kwargs = {
            f.name: _field_from_json(f, json_obj.get(f.name), context)
            for f in dataclasses.fields(cls)
            if f.name in json_obj
        }
        return cls(**kwargs)


@dace.serialize.serializable
@dataclass
class CopyOnAccess(MemletAccessPolicy):
    """The default policy: the full offset is evaluated at every access and data moves at the access (or through
    the copy library node inserted for access-node-to-access-node copies). Lowering is a no-op."""

    is_default: ClassVar[bool] = True


@dace.serialize.serializable
@dataclass
class LoopCursor(MemletAccessPolicy):
    """The memlet's element offset is affine in the induction variable of an enclosing loop::

        offset(iteration) = base_invariant + lane_part + iteration_variable * (step / loop_stride)

    Lowering (see :func:`~dace.transformation.passes.memlet_access_policies.lower_loop_cursors`) materializes one
    loop-carried integer symbol per *cursor class* (same array, loop, step and lane part; memlets of a class differ
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
    cursor_type: Optional[dtypes.typeclass] = field(default=None, metadata={"kind": "dtype"})
    share_key: Optional[str] = None
    cursor: Optional[str] = None
    reference: Optional[str] = None
    window: Optional[str] = None
    immediate: SymbolicLike = field(default=None, metadata={"kind": "symbolic"})

    def __post_init__(self):
        for name in ("step", "base_invariant", "lane_part", "immediate"):
            setattr(self, name, _to_sym(getattr(self, name)))
        if self.cursor_type is not None and not isinstance(self.cursor_type, dtypes.typeclass):
            raise TypeError(f"cursor_type must be a dace typeclass or None, got {self.cursor_type!r}")

    @property
    def is_lowered(self) -> bool:
        return self.cursor is not None and self.reference is not None

    @classmethod
    def lower(cls, sdfg, entries, **options) -> Dict[str, int]:
        from dace.transformation.passes.memlet_access_policies import lower_loop_cursors  # avoid import cycle

        return lower_loop_cursors(sdfg, entries, **options)
