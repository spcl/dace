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
    """Pure expansion of :class:`CShift` -- a single Map that reads
    ``_x`` at the circularly-rotated index and writes ``_out``.

    Fortran ``CSHIFT(arr, shift)`` along ``dim`` (1-based) yields
    ``out(i) = arr(MODULO(i - 1 + shift, n) + 1)``.  In 0-based Map
    iterators that is ``_out[i] = _x[mod(i + shift, n)]`` along ``dim``,
    with every other axis passed through unchanged.

    ``fortran_mod`` is the FLOORED modulus (``dace.symbolic.mod`` ->
    ``dace::math::mod``), NOT sympy's built-in ``Mod``: ``Mod`` lowers
    to the C ``%`` operator, which TRUNCATES on signed integers
    (``(-1) % 5 == -1``), so a negative ``shift`` would index out of
    bounds.  The floored ``mod`` returns ``[0, n)`` for any sign
    (``mod(-1, 5) == 4``), matching Fortran ``MODULO`` / ``CSHIFT``
    wrap semantics.  (A doubled ``Mod(Mod(x,n)+n,n)`` does NOT work --
    sympy simplifies it straight back to ``Mod(x,n)``.)

    The tasklet body is just ``__out = __in`` -- the rotation lives
    entirely in the source memlet's subset, so no runtime helper is
    needed.
    """
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        desc_x, desc_out, dim_zero = node.validate(parent_sdfg, parent_state)
        dtype = desc_x.dtype.base_type
        shape = list(desc_x.shape)
        rank = len(shape)
        n = shape[dim_zero]
        # ``shift`` must be a concrete amount -- either a compile-time
        # constant or a symbol the surrounding SDFG already declares +
        # binds.  We do NOT invent a fallback symbol here: a fabricated
        # ``__shift`` would appear in the source memlet's subset but
        # never be assigned, leaking as an unbound free symbol (the
        # SDFG would then demand it as a call argument and ``arglist``
        # would ``KeyError``).  Fail loud instead so a caller that
        # forgot to set ``shift`` gets a clear message, not a cryptic
        # free-symbol error downstream.
        if node.shift is None:
            raise ValueError(f"CShift '{node.label}': shift is None.  The shift amount must "
                             "be set on the node (a constant or an SDFG-bound symbol) before "
                             "expansion -- the bridge supplies it from the Fortran "
                             "CSHIFT(arr, shift) argument.")
        shift = node.shift

        sdfg = dace.SDFG(node.label + "_sdfg")
        sdfg.add_array("_x", shape, dtype)
        sdfg.add_array("_out", shape, dtype)

        state = sdfg.add_state()
        map_rng = {f"__i{d}": f"0:{shape[d]}" for d in range(rank)}
        # Source subset: rotate along ``dim_zero``, pass the rest through.
        in_subs = []
        for d in range(rank):
            if d == dim_zero:
                # ``fortran_mod`` is the FLOORED modulus (NOT sympy ``Mod``,
                # which lowers to C ``%`` and truncates -- breaking negative
                # shifts).  It codegens to the self-contained sign-correct
                # form ``(((a) % (b)) + (b)) % (b)``.
                in_subs.append(f"fortran_mod(__i{d} + ({shift}), {n})")
            else:
                in_subs.append(f"__i{d}")
        in_sub = ", ".join(in_subs)
        out_sub = ", ".join(f"__i{d}" for d in range(rank))
        state.add_mapped_tasklet(
            name="_cshift",
            map_ranges=map_rng,
            inputs={"__in": dace.Memlet(f"_x[{in_sub}]")},
            code="__out = __in",
            outputs={"__out": dace.Memlet(f"_out[{out_sub}]")},
            external_edges=True,
        )
        return sdfg


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
