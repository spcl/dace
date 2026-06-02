# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``CShift`` library node -- Fortran ``CSHIFT`` (circular shift).

Fortran ``CSHIFT(arr, shift [, dim])`` produces an array of the same
shape as ``arr`` whose element along ``dim`` (default 1) at position
``i`` is ``arr(MODULO(i - 1 + shift, n) + 1)``.  The shift can be
positive (rotate elements toward the start) or negative (rotate
toward the end); higher-rank inputs rotate each slice along ``dim``
independently.

Pure expansion: a single Map writes the rotated layout in parallel.
The shift is a runtime symbol so the same expansion handles compile-
time-constant shifts and Fortran-dummy-scaled shifts alike.  DaCe's
scheduler picks the right OpenMP / GPU lowering for the Map based on
the storage of the data.
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
    """Pure expansion of :class:`CShift` -- a parallel Map of mod-indexed reads."""

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        desc_x, desc_out, dim_zero = node.validate(parent_sdfg, parent_state)
        shape = list(desc_x.shape)
        rank = len(shape)
        dtype = desc_x.dtype.base_type
        # ``shift`` is symbolic so a single expansion handles every
        # call site without re-baking; ``node.shift`` may be a
        # constant, an SDFG symbol, or any sympy expression resolved
        # in the parent's scope at codegen time.
        shift_expr = node.shift if node.shift is not None else dace.symbolic.symbol("__shift")

        sdfg = dace.SDFG(node.label + "_sdfg")
        sdfg.add_array("_x", shape, dtype)
        sdfg.add_array("_out", shape, dtype)
        if node.shift is None:
            sdfg.add_symbol("__shift", dace.int64)
        else:
            # Register every free symbol the shift expression depends
            # on, so the expanded nested SDFG advertises them as
            # parent-supplied symbols.  The Fortran-bridge runs scalar
            # INTENT(IN) args through ``_promote_scalar_args`` and the
            # outer SDFG carries the matching symbol; without this
            # ``add_symbol`` the inner reference dead-ends with an
            # ``unresolved free symbol`` error.
            for sym in node.shift.free_symbols:
                name = str(sym)
                if name not in sdfg.symbols:
                    sdfg.add_symbol(name, dace.int64)

        state = sdfg.add_state()
        map_rng = {f"__i{d}": f"0:{shape[d]}" for d in range(rank)}
        n_axis = shape[dim_zero]
        # Source memlet: stream the whole ``dim_zero`` axis into the
        # tasklet (the cshift permutation can land on any index for
        # negative shifts -- a single-element memlet with a computed
        # index would let DaCe's symbolic bounds-analysis miscount the
        # read range).  Off-dim entries stay element-wise.
        src_parts = []
        for d in range(rank):
            if d == dim_zero:
                src_parts.append(f"0:{shape[d]}")
            else:
                src_parts.append(f"__i{d}")
        src_sub = ", ".join(src_parts)
        out_sub = ", ".join([f"__i{d}" for d in range(rank)])
        # Tasklet does the mod-indexed read.  ``__in`` is a 1-D view
        # along ``dim_zero``; the inner index is ``(i + shift) mod n``
        # with the ``+ n) % n`` correction for negative shifts under
        # the C truncating-modulo convention.
        code = (f"__src = ((__i{dim_zero} + ({shift_expr})) % {n_axis} + {n_axis}) % {n_axis}\n"
                f"__out = __in[__src]")
        state.add_mapped_tasklet(
            name="_cshift",
            map_ranges=map_rng,
            inputs={"__in": dace.Memlet(f"_x[{src_sub}]")},
            code=code,
            outputs={"__out": dace.Memlet(f"_out[{out_sub}]")},
            external_edges=True,
        )
        return sdfg


@dace.library.node
class CShift(dace.sdfg.nodes.LibraryNode):
    """Fortran ``CSHIFT(arr, shift [, dim])`` -- circular shift along ``dim``.

    ``shift`` is symbolic so a single library-node instance handles
    any combination of compile-time-constant shifts and runtime
    symbols.  ``None`` (the default) makes the expansion fall back
    on a freshly-symbol-named ``__shift`` so the test suite can drive
    the node without a wrapper.
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
