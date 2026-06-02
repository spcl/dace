# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``EOShift`` library node -- Fortran ``EOSHIFT`` (end-off shift).

Fortran ``EOSHIFT(ARRAY, SHIFT [, BOUNDARY] [, DIM])`` shifts elements
along ``DIM`` by ``SHIFT``: positive shifts move toward the start,
elements falling off the end disappear, vacated slots get the
``BOUNDARY`` value (default 0).  Higher-rank inputs shift each
``DIM``-slice independently.

Pure expansion: a single mapped tasklet decides per-element whether
to read from the source (in-range) or write the boundary.  The shift
expression is symbolic so a single expansion handles both
compile-time-constant and runtime-variable shifts; the boundary is a
SymbolicProperty (default 0) the bridge pins at lowering time.
"""
import dace
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace import SDFG, SDFGState, memlet as mm
from dace.frontend.common import op_repository as oprepo
from dace.transformation.transformation import ExpandTransformation


@dace.library.expansion
class ExpandEOShiftPure(ExpandTransformation):
    """Pure expansion of :class:`EOShift` -- not yet implemented.

    :raises NotImplementedError: always.  The lib node validates its
        connector wiring but does not produce a body until a target
        workload exercises ``EOSHIFT`` (none do today).
    """
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        raise NotImplementedError("EOShift pure expansion is not yet implemented.  EOSHIFT does not "
                                  "appear in any current target Fortran workload (ICON / ECRAD / "
                                  "cloudsc / QE / Graupel); the lib node is reserved as a typed "
                                  "bridge target.  When a workload starts using EOSHIFT, the boundary "
                                  "fill rules out a pure memlet-subset expression -- implement the "
                                  "expansion as a single Map whose tasklet body picks between the "
                                  "clamped read and the boundary value, e.g. via a Python conditional "
                                  "expression ``__out = __in if (0 <= i + shift) and (i + shift < n) "
                                  "else boundary`` over a memlet ``_x[max(0, min(n-1, i + shift))]``.")


@dace.library.node
class EOShift(dace.sdfg.nodes.LibraryNode):
    """Fortran ``EOSHIFT(arr, shift [, boundary] [, dim])`` -- end-off shift along ``dim``.

    Configurable:

    * ``dim`` (default ``1``, Fortran 1-based)  --  axis to shift along.
    * ``shift``  --  SymbolicProperty.  ``None`` -> the symbol ``__shift``.
    * ``boundary``  --  SymbolicProperty for the fill value of vacated
      slots; ``None`` defaults to ``0``.
    """

    implementations = {"pure": ExpandEOShiftPure}
    default_implementation = "pure"

    dim = dace.properties.Property(dtype=int,
                                   default=1,
                                   desc="Fortran 1-based axis to shift along (EOSHIFT default 1).")
    shift = dace.properties.SymbolicProperty(allow_none=True,
                                             default=None,
                                             desc="Shift amount; ``None`` means use the symbol ``__shift``.")
    boundary = dace.properties.SymbolicProperty(allow_none=True,
                                                default=None,
                                                desc="Fill value for vacated slots (``None`` defaults to 0).")

    def __init__(self, name, *, dim=1, shift=None, boundary=None, **kwargs):
        super().__init__(name, inputs={"_x"}, outputs={"_out"}, **kwargs)
        self.dim = dim
        self.shift = shift
        self.boundary = boundary

    def validate(self, sdfg, state):
        """:returns: ``(desc_x, desc_out, dim_zero)``.

        :raises ValueError: if shapes don't match or ``dim`` is out of range.
        """
        in_edges = state.in_edges(self)
        out_edges = state.out_edges(self)
        if len(in_edges) != 1 or in_edges[0].dst_conn != "_x":
            raise ValueError("EOShift requires an `_x` input")
        if len(out_edges) != 1 or out_edges[0].src_conn != "_out":
            raise ValueError("EOShift requires an `_out` output")
        desc_x = sdfg.arrays[in_edges[0].data.data]
        desc_out = sdfg.arrays[out_edges[0].data.data]
        rank = len(desc_x.shape)
        if not (1 <= self.dim <= rank):
            raise ValueError(f"EOShift: dim={self.dim} out of range for rank-{rank} input")
        if list(desc_out.shape) != list(desc_x.shape):
            raise ValueError(f"EOShift: input shape {list(desc_x.shape)} != output shape {list(desc_out.shape)}")
        return desc_x, desc_out, self.dim - 1


@oprepo.replaces('dace.libraries.standard.eoshift')
@oprepo.replaces('dace.libraries.standard.EOShift')
def eoshift_libnode(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, x, out, *, dim=1, shift=None, boundary=None):
    x_in = state.add_read(x)
    w = state.add_write(out)
    node = EOShift("eoshift", dim=dim, shift=shift, boundary=boundary)
    state.add_node(node)
    state.add_edge(x_in, None, node, '_x', mm.Memlet(x))
    state.add_edge(node, '_out', w, None, mm.Memlet(out))
    return []
