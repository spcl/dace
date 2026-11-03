# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``CShift``: a circular shift by ``shift`` positions along ``dim``, in the Fortran or the numpy direction.

Fortran ``CSHIFT(ARRAY, SHIFT [, DIM])`` reads element ``MODULO(i + SHIFT, n)`` of the input at position ``i``;
``numpy.roll`` runs the same rotation the other way. Every section perpendicular to ``dim`` rotates on its own.
"""

import enum

import dace
from dace import SDFG, library, properties
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation.transformation import ExpandTransformation


class ShiftDirection(enum.Enum):
    """Which way a :class:`CShift` rotates."""

    #: ``out(i) = x(mod(i + shift, n))``, Fortran ``CSHIFT``.
    FORTRAN = "fortran"
    #: ``out[i] = x[mod(i - shift, n)]``, ``numpy.roll``.
    NUMPY = "numpy"


def connector_array(sdfg: SDFG, name: str, memlet: dace.Memlet, outer: SDFG) -> list:
    """Adds to ``sdfg`` the array ``name`` that the expansion reads or writes through ``memlet`` of ``outer``: its
    shape is the subset's and its strides the container's, scaled by the subset's steps. Returns the shape."""
    desc = outer.arrays[memlet.data]
    shape = memlet.subset.size()
    strides = [stride * dim_range[2] for stride, dim_range in zip(desc.strides, memlet.subset)]
    sdfg.add_array(name, shape, desc.dtype, desc.storage, strides=strides)
    return shape


@library.expansion
class ExpandCShiftPure(ExpandTransformation):
    """One map reading the input at the rotated index."""

    environments: list[type] = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        if node.shift is None:
            raise ValueError(f"{node.label}: the shift must be set before expansion")
        sdfg = SDFG(f"{node.label}_sdfg")
        shape = connector_array(sdfg, "_x", sdutil.in_edge_with_name(node, parent_state, "_x").data, parent_sdfg)
        connector_array(sdfg, "_out", sdutil.out_edge_with_name(node, parent_state, "_out").data, parent_sdfg)
        axis = node.dim - 1
        # The direction lives on the node, so this is the one place the sign is chosen
        shift = node.shift if node.direction is ShiftDirection.FORTRAN else -node.shift
        element = [f"__i{d}" for d in range(len(shape))]
        # ``FtnModulo`` is Fortran's MODULO: a negative shift wraps into range, where ``%`` would not
        rotated = [f"FtnModulo(__i{d} + ({shift}), {shape[d]})" if d == axis else i for d, i in enumerate(element)]
        sdfg.add_state().add_mapped_tasklet(
            "cshift",
            {f"__i{d}": f"0:{size}" for d, size in enumerate(shape)},
            {"__in": dace.Memlet(f"_x[{', '.join(rotated)}]")},
            "__out = __in",
            {"__out": dace.Memlet(f"_out[{', '.join(element)}]")},
            external_edges=True,
        )
        return sdfg


@library.node
class CShift(nodes.LibraryNode):
    """Rotates ``_x`` by ``shift`` positions along ``dim`` into ``_out``.

    ``direction`` says which way: :attr:`ShiftDirection.FORTRAN` is ``CSHIFT``, :attr:`ShiftDirection.NUMPY` is
    ``numpy.roll``. A node rotated the wrong way still has the right shape, dtype and multiset of values, so the
    direction is named on the node instead of negated by each caller.
    """

    implementations = {"pure": ExpandCShiftPure}
    default_implementation = "pure"

    dim = properties.Property(dtype=int, default=1, desc="Fortran 1-based axis to rotate along.")
    shift = properties.SymbolicProperty(allow_none=True, default=None, desc="Positions to shift by, set by expansion.")
    direction = properties.EnumProperty(
        dtype=ShiftDirection, default=ShiftDirection.FORTRAN, desc="Fortran CSHIFT or numpy.roll rotation."
    )

    def __init__(self, name, *, dim=1, shift=None, direction=ShiftDirection.FORTRAN, **kwargs):
        super().__init__(name, inputs={"_x"}, outputs={"_out"}, **kwargs)
        self.dim = dim
        self.shift = shift
        self.direction = direction

    def validate(self, sdfg, state):
        x = sdutil.in_edge_with_name(self, state, "_x").data.subset
        out = sdutil.out_edge_with_name(self, state, "_out").data.subset
        if not 1 <= self.dim <= x.dims():
            raise ValueError(f"{self.label}: dim={self.dim} is out of range for a rank-{x.dims()} input")
        if x.size() != out.size():
            raise ValueError(f"{self.label}: _x has the shape {x.size()} and _out {out.size()}")
