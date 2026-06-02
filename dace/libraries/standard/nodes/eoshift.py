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
    """Pure expansion -- per-element branch on shifted-index range."""

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        desc_x, desc_out, dim_zero = node.validate(parent_sdfg, parent_state)
        dtype = desc_x.dtype.base_type
        shape = list(desc_x.shape)
        rank = len(shape)
        shift_expr = node.shift if node.shift is not None else dace.symbolic.symbol("__shift")
        boundary_expr = node.boundary if node.boundary is not None else 0

        sdfg = dace.SDFG(node.label + "_sdfg")
        sdfg.add_array("_x", shape, dtype)
        sdfg.add_array("_out", shape, dtype)
        if node.shift is None:
            sdfg.add_symbol("__shift", dace.int64)
        else:
            for sym in node.shift.free_symbols:
                name = str(sym)
                if name not in sdfg.symbols:
                    sdfg.add_symbol(name, dace.int64)
        if node.boundary is not None:
            for sym in node.boundary.free_symbols:
                name = str(sym)
                if name not in sdfg.symbols:
                    sdfg.add_symbol(name, dtype)

        state = sdfg.add_state()
        map_rng = {f"__i{d}": f"0:{shape[d]}" for d in range(rank)}
        n_axis = shape[dim_zero]
        src_parts = []
        for d in range(rank):
            if d == dim_zero:
                src_parts.append(f"0:{shape[d]}")
            else:
                src_parts.append(f"__i{d}")
        src_sub = ", ".join(src_parts)
        out_sub = ", ".join([f"__i{d}" for d in range(rank)])
        # Single-statement Python tasklet -- conditional expression
        # selects ``__in[i+shift]`` when in range, else the boundary
        # value.  The read index is clamped via ``max(0, min(n-1, i+s))``
        # so the array access is always inside bounds before the
        # conditional picks which value to commit.
        code = (f"__out = __in[max(0, min({n_axis} - 1, __i{dim_zero} + ({shift_expr})))] "
                f"if (0 <= __i{dim_zero} + ({shift_expr})) and (__i{dim_zero} + ({shift_expr}) < {n_axis}) "
                f"else ({boundary_expr})")
        state.add_mapped_tasklet(
            name="_eoshift",
            map_ranges=map_rng,
            inputs={"__in": dace.Memlet(f"_x[{src_sub}]")},
            code=code,
            outputs={"__out": dace.Memlet(f"_out[{out_sub}]")},
            external_edges=True,
        )
        return sdfg


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
