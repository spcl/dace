# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``CShift`` library node -- Fortran ``CSHIFT`` (circular shift).

Fortran ``CSHIFT(arr, shift [, dim])`` produces an array of the same
shape as ``arr`` whose element along ``dim`` (default 1) at position
``i`` is ``arr(MODULO(i - 1 + shift, n) + 1)``.  Operates on the
whole array: a rank-R input rotates each (R-1)-cross-section
perpendicular to ``dim`` independently.

CSHIFT does not currently appear in any kernel in our target
workloads (ICON, ECRAD, cloudsc, QE, Graupel) -- the lib node is
kept as a typed bridge target so the HLFIR frontend can route
``hlfir.cshift`` here without falling through to an
``unrecognised intrinsic`` error, but the pure expansion is a TODO
stub.  Implement the expansion (and any backend variants -- OpenMP
5.0 user-defined reduction, CUB / stdpar) when a workload requires
it; the recommended shape is a single Map whose memlet subset is
``Mod(Mod(i + shift, n) + n, n)`` so the tasklet body collapses to
``__out = __in`` (no helper required).
"""
import dace
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace import SDFG, SDFGState, memlet as mm
from dace.frontend.common import op_repository as oprepo
from dace.transformation.transformation import ExpandTransformation


@dace.library.expansion
class ExpandCShiftPure(ExpandTransformation):
    """Pure expansion of :class:`CShift` -- not yet implemented.

    :raises NotImplementedError: always.  The lib node validates its
        connector wiring but does not produce a body until a target
        workload exercises ``CSHIFT`` (none do today).
    """
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        raise NotImplementedError("CShift pure expansion is not yet implemented.  CSHIFT does not "
                                  "appear in any current target Fortran workload (ICON / ECRAD / "
                                  "cloudsc / QE / Graupel); the lib node is reserved as a typed "
                                  "bridge target.  When a workload starts using CSHIFT, implement "
                                  "the expansion as a single Map whose source memlet's subset is "
                                  "``Mod(Mod(i + shift, n) + n, n)`` -- the tasklet body collapses "
                                  "to ``__out = __in`` (no runtime helper required).")


@dace.library.node
class CShift(dace.sdfg.nodes.LibraryNode):
    """Fortran ``CSHIFT(arr, shift [, dim])`` -- circular shift along ``dim``.

    Whole-array operation: a rank-R input rotates every
    (R-1)-cross-section perpendicular to ``dim`` by ``shift``
    positions, with elements falling off one end wrapping around to
    the other.

    ``shift`` is symbolic so a single library-node instance handles
    any combination of compile-time-constant shifts and runtime
    symbols.  ``None`` (the default) leaves the lib node ready for a
    bridge-supplied shift expression at lowering time.
    """

    implementations = {"pure": ExpandCShiftPure}
    default_implementation = "pure"

    dim = dace.properties.Property(dtype=int,
                                   default=1,
                                   desc="Fortran 1-based axis to rotate along (CSHIFT default 1).")
    shift = dace.properties.SymbolicProperty(allow_none=True,
                                             default=None,
                                             desc="Shift amount; ``None`` means use the symbol ``__shift``.")

    def __init__(self, name, *, dim=1, shift=None, **kwargs):
        super().__init__(name, inputs={"_x"}, outputs={"_out"}, **kwargs)
        self.dim = dim
        self.shift = shift

    def validate(self, sdfg, state):
        """:returns: ``(desc_x, desc_out, dim_zero)``.

        :raises ValueError: if shapes don't match or dim is out of range.
        """
        in_edges = state.in_edges(self)
        out_edges = state.out_edges(self)
        if len(in_edges) != 1 or in_edges[0].dst_conn != "_x":
            raise ValueError("CShift requires an `_x` input")
        if len(out_edges) != 1 or out_edges[0].src_conn != "_out":
            raise ValueError("CShift requires an `_out` output")
        desc_x = sdfg.arrays[in_edges[0].data.data]
        desc_out = sdfg.arrays[out_edges[0].data.data]
        rank = len(desc_x.shape)
        if not (1 <= self.dim <= rank):
            raise ValueError(f"CShift: dim={self.dim} out of range for rank-{rank} input")
        if list(desc_out.shape) != list(desc_x.shape):
            raise ValueError(f"CShift: input shape {list(desc_x.shape)} != output shape {list(desc_out.shape)}")
        return desc_x, desc_out, self.dim - 1


@oprepo.replaces('dace.libraries.standard.cshift')
@oprepo.replaces('dace.libraries.standard.CShift')
def cshift_libnode(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, x, out, *, dim=1):
    x_in = state.add_read(x)
    w = state.add_write(out)
    node = CShift('cshift', dim=dim)
    state.add_node(node)
    state.add_edge(x_in, None, node, '_x', mm.Memlet(x))
    state.add_edge(node, '_out', w, None, mm.Memlet(out))
    return []
