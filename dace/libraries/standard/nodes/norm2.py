# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``Norm2`` library node -- Fortran ``NORM2`` Euclidean norm.

Fortran ``NORM2(X [, DIM])`` returns ``sqrt(sum(X**2))``.  Without
``DIM`` the result is a scalar over the whole array; with ``DIM``
the reduction is along one axis and the result is rank-(R-1).

Pure expansion: a single reduction map accumulates the sum of squares
via WCR (parallel-safe), and a trailing tasklet writes ``sqrt`` of the
accumulator into the user-facing scalar / per-slice output.  DaCe's
scheduler picks OpenMP / GPU on the Map based on storage.
"""
import dace
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace import SDFG, SDFGState, memlet as mm
from dace.frontend.common import op_repository as oprepo
from dace.transformation.transformation import ExpandTransformation


@dace.library.expansion
class ExpandNorm2Pure(ExpandTransformation):
    """Pure expansion: WCR-summed squares + sqrt finalize."""

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        desc_x, desc_out, dim_zero = node.validate(parent_sdfg, parent_state)
        dtype = desc_x.dtype.base_type
        shape = list(desc_x.shape)
        rank = len(shape)

        sdfg = dace.SDFG(node.label + "_sdfg")
        sdfg.add_array("_x", shape, dtype)

        if dim_zero is None:
            sdfg.add_array("_out", [1], dtype)
            out_shape = [1]
        else:
            out_shape = [s for d, s in enumerate(shape) if d != dim_zero]
            if not out_shape:
                out_shape = [1]
            sdfg.add_array("_out", out_shape, dtype)
        sdfg.add_transient("_sumsq", out_shape, dtype)

        # Seed the accumulator with zero.
        init_state = sdfg.add_state(node.label + "_init")
        if dim_zero is None:
            init_state.add_mapped_tasklet(
                name="_norm2_init",
                map_ranges={"__i0": "0:1"},
                inputs={},
                code="__init = 0",
                outputs={"__init": dace.Memlet("_sumsq[0]")},
                external_edges=True,
            )
        else:
            init_rng = {f"__o{d}": f"0:{shape[d]}" for d in range(rank) if d != dim_zero}
            init_subs = ", ".join([f"__o{d}" for d in range(rank) if d != dim_zero])
            init_state.add_mapped_tasklet(
                name="_norm2_init",
                map_ranges=init_rng,
                inputs={},
                code="__init = 0",
                outputs={"__init": dace.Memlet(f"_sumsq[{init_subs}]")},
                external_edges=True,
            )

        # Reduce: ``sumsq += x[i]**2``.
        reduce_state = sdfg.add_state_after(init_state, node.label + "_sumsq")
        map_rng = {f"__i{d}": f"0:{shape[d]}" for d in range(rank)}
        x_subs = ", ".join([f"__i{d}" for d in range(rank)])
        if dim_zero is None:
            out_pair_subs = "0"
        else:
            out_pair_subs = ", ".join([f"__i{d}" for d in range(rank) if d != dim_zero])
        reduce_state.add_mapped_tasklet(
            name="_norm2_sumsq",
            map_ranges=map_rng,
            inputs={"__in": dace.Memlet(f"_x[{x_subs}]")},
            code="__out = __in * __in",
            outputs={"__out": dace.Memlet(f"_sumsq[{out_pair_subs}]", wcr="lambda a, b: a + b")},
            external_edges=True,
        )

        # Finalize: ``out = sqrt(sumsq)``.
        final_state = sdfg.add_state_after(reduce_state, node.label + "_sqrt")
        if dim_zero is None:
            final_state.add_mapped_tasklet(
                name="_norm2_sqrt",
                map_ranges={"__i0": "0:1"},
                inputs={"__in": dace.Memlet("_sumsq[0]")},
                code="__out = sqrt(__in)",
                outputs={"__out": dace.Memlet("_out[0]")},
                external_edges=True,
            )
        else:
            out_rng = {f"__o{d}": f"0:{shape[d]}" for d in range(rank) if d != dim_zero}
            out_subs = ", ".join([f"__o{d}" for d in range(rank) if d != dim_zero])
            final_state.add_mapped_tasklet(
                name="_norm2_sqrt",
                map_ranges=out_rng,
                inputs={"__in": dace.Memlet(f"_sumsq[{out_subs}]")},
                code="__out = sqrt(__in)",
                outputs={"__out": dace.Memlet(f"_out[{out_subs}]")},
                external_edges=True,
            )
        return sdfg


@dace.library.node
class Norm2(dace.sdfg.nodes.LibraryNode):
    """Fortran ``NORM2(X [, DIM])`` -- L2 (Euclidean) norm.

    Configurable:

    * ``dim``  --  Fortran 1-based reduction axis.  ``None`` reduces
      the whole array to a scalar.

    No ``DIM``: output is a length-1 scalar holding ``sqrt(sum(X**2))``.
    With ``DIM``: output is rank-(R-1) with each slice's L2 norm.
    """

    implementations = {"pure": ExpandNorm2Pure}
    default_implementation = "pure"

    dim = dace.properties.Property(dtype=int,
                                   default=None,
                                   allow_none=True,
                                   desc="Fortran 1-based reduction axis.  ``None`` reduces the whole array.")

    def __init__(self, name, *, dim=None, **kwargs):
        super().__init__(name, inputs={"_x"}, outputs={"_out"}, **kwargs)
        self.dim = dim

    def validate(self, sdfg, state):
        """:returns: ``(desc_x, desc_out, dim_zero_or_None)``.

        :raises ValueError: if connector wiring is wrong or ``dim`` is
            out of range for the input rank.
        """
        in_edges = state.in_edges(self)
        out_edges = state.out_edges(self)
        if len(in_edges) != 1 or in_edges[0].dst_conn != "_x":
            raise ValueError("Norm2 requires an `_x` input")
        if len(out_edges) != 1 or out_edges[0].src_conn != "_out":
            raise ValueError("Norm2 requires an `_out` output")
        desc_x = sdfg.arrays[in_edges[0].data.data]
        desc_out = sdfg.arrays[out_edges[0].data.data]
        rank = len(desc_x.shape)
        dim_zero = None
        if self.dim is not None:
            if not (1 <= self.dim <= rank):
                raise ValueError(f"Norm2: dim={self.dim} out of range for rank-{rank} input")
            dim_zero = self.dim - 1
        return desc_x, desc_out, dim_zero


@oprepo.replaces('dace.libraries.standard.norm2')
@oprepo.replaces('dace.libraries.standard.Norm2')
def norm2_libnode(pv: 'ProgramVisitor', sdfg: SDFG, state: SDFGState, x, out, *, dim=None):
    x_in = state.add_read(x)
    w = state.add_write(out)
    node = Norm2("norm2", dim=dim)
    state.add_node(node)
    state.add_edge(x_in, None, node, '_x', mm.Memlet(x))
    state.add_edge(node, '_out', w, None, mm.Memlet(out))
    return []
