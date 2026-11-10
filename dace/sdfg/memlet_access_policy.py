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
:class:`~dace.transformation.passes.memlet_access_policy_lowering.LowerMemletAccessPolicies` in the code-generation
window. Code generation itself knows nothing about policies.

This module defines the :class:`MemletAccessPolicy` base class. The policy kinds live in
:mod:`dace.sdfg.memlet_access_policies`:

* :class:`~dace.sdfg.memlet_access_policies.copy_on_access.CopyOnAccess` -- the default and the behavior every SDFG
  had before access policies existed: the full offset expression is evaluated at every access. Lowering is a no-op.
* :class:`~dace.sdfg.memlet_access_policies.loop_cursor.LoopCursor` -- the memlet's element offset is affine in the
  induction variable of an enclosing :class:`~dace.sdfg.state.LoopRegion` and is tracked by a loop-carried cursor.

Further kinds (a buffer-resource-descriptor policy whose cursor is the buffer instruction's ``voffset``, a
tensor-memory-accelerator policy whose cursors are tile coordinates) follow the same protocol: a dataclass
record plus a class-level :meth:`MemletAccessPolicy.lower` that emits SDFG constructs.

Policies are plain dataclasses. Fields holding symbolic expressions or data types are marked with the field
metadata ``kind='symbolic'`` / ``kind='dtype'`` so that the generic JSON (de)serialization in the base class can
convert them.
"""

import dataclasses
from dataclasses import dataclass
from typing import Any, ClassVar

import sympy

import dace.serialize
from dace import dtypes, symbolic

SymbolicLike = str | int | symbolic.SymbolicType | None


def to_symbolic(value: SymbolicLike) -> symbolic.SymbolicType | None:
    """Convert a policy field value to a symbolic expression (``None`` stays ``None``)."""
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
    def lower(cls, sdfg, entries: list[tuple[Any, Any]], **options) -> dict[str, int]:
        """Materialize all memlets of this policy kind in one SDFG as ordinary SDFG constructs.

        Called by :class:`~dace.transformation.passes.memlet_access_policy_lowering.LowerMemletAccessPolicies` once
        per policy kind, with every ``(state, edge)`` of the SDFG and its nested SDFGs whose memlet carries a policy
        of this kind (the memlet paths cross nested SDFGs, so one lowering may touch several of them). The default
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
