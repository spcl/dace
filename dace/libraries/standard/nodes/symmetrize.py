# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``Symmetrize`` library node: mirror one triangle of a square matrix into the
other across the main diagonal.

A hand-written symmetrization -- ``X[j, i] = X[i, j]`` over the strict upper
triangle (``i < j``) after the upper triangle has been computed -- is
*embarrassingly parallel*: every lower-triangle write reads a distinct,
already-final upper-triangle element, and the read set (upper) is disjoint from
the write set (lower). But when it is expressed in-place over a triangular loop
nest, ``LoopToMap`` sees the same array both read (``X[i, j]``) and written
(``X[j, i]``) with symmetric data-dependent indices and conservatively refuses,
leaving it sequential. Lifting the nest to this node makes the semantics
explicit and its ``pure`` expansion emits the parallel triangular copy directly.

The node is in-place: its ``_in`` and ``_out`` connectors both wire to the same
square array; the expansion reads the source triangle and writes the mirror.
"""
import dace
from dace import library, nodes, properties
from dace.transformation.transformation import ExpandTransformation


@library.expansion
class ExpandSymmetrizePure(ExpandTransformation):
    """Parallel triangular copy: ``map[i] { map[j in i+off:hi] { X[mirror] = X[src] } }``."""

    environments = []

    @staticmethod
    def expansion(node: "Symmetrize", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> dace.SDFG:
        array_name, desc = node.validate(parent_sdfg, parent_state)

        nsdfg = dace.SDFG(f"{node.label}_sdfg")
        for conn in ("_in", "_out"):
            d = desc.clone()
            d.transient = False
            nsdfg.add_datadesc(conn, d)
        nstate = nsdfg.add_state(f"{node.label}_state")

        # Both axes of the triangle are independent -> parallel maps.
        parallel = dace.dtypes.ScheduleType.Default
        ome, omx = nstate.add_map(f"{node.label}_row", {"__i": f"{node.row_lo}:{node.row_hi}"}, schedule=parallel)
        ime, imx = nstate.add_map(f"{node.label}_col", {"__j": f"__i + {node.col_offset}:{node.col_hi}"},
                                  schedule=parallel)
        # source_upper: read the upper element X[i, j] (i < j), write the lower
        # mirror X[j, i]. Otherwise read the lower and write the upper.
        read_idx = "__i, __j" if node.source_upper else "__j, __i"
        write_idx = "__j, __i" if node.source_upper else "__i, __j"

        r = nstate.add_read("_in")
        w = nstate.add_write("_out")
        t = nstate.add_tasklet(f"{node.label}_copy", {"__in"}, {"__out"}, "__out = __in")
        nstate.add_memlet_path(r, ome, ime, t, dst_conn="__in", memlet=dace.Memlet(f"_in[{read_idx}]"))
        nstate.add_memlet_path(t, imx, omx, w, src_conn="__out", memlet=dace.Memlet(f"_out[{write_idx}]"))
        return nsdfg


@library.node
class Symmetrize(nodes.LibraryNode):
    """Symmetrize a square matrix by mirroring one triangle into the other.

    In-place: ``_in`` and ``_out`` both connect to the same 2-D array. The node
    fills the triangle opposite ``source_upper`` from the source triangle; the
    diagonal and the source triangle are left untouched. The triangular
    iteration space is ``__i in [row_lo, row_hi)``, ``__j in [__i + col_offset,
    col_hi)`` (``col_offset >= 1`` excludes the diagonal).
    """

    implementations = {"pure": ExpandSymmetrizePure}
    default_implementation = "pure"

    row_lo = properties.Property(dtype=str, default="0", desc="Outer (row) index start.")
    row_hi = properties.Property(dtype=str, default="0", desc="Outer (row) index exclusive end.")
    col_offset = properties.Property(dtype=int, default=1, desc="Inner (col) start offset past the row index.")
    col_hi = properties.Property(dtype=str, default="0", desc="Inner (col) index exclusive end.")
    source_upper = properties.Property(dtype=bool,
                                       default=True,
                                       desc="Source triangle is the upper (read X[i,j], write X[j,i]); "
                                       "else the lower.")

    def __init__(self,
                 name: str,
                 row_lo: str = "0",
                 row_hi: str = "0",
                 col_offset: int = 1,
                 col_hi: str = "0",
                 source_upper: bool = True,
                 **kwargs):
        super().__init__(name, inputs={"_in"}, outputs={"_out"}, **kwargs)
        self.row_lo = row_lo
        self.row_hi = row_hi
        self.col_offset = col_offset
        self.col_hi = col_hi
        self.source_upper = source_upper

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState):
        """Resolve the single in-place array and check it is 2-D.

        :returns: ``(array_name, descriptor)``.
        :raises ValueError: on missing/extra edges or a non-2-D or non-square target.
        """
        in_edges = [e for e in state.in_edges(self) if e.dst_conn == "_in" and not e.data.is_empty()]
        out_edges = [e for e in state.out_edges(self) if e.src_conn == "_out" and not e.data.is_empty()]
        if len(in_edges) != 1 or len(out_edges) != 1:
            raise ValueError(f"{type(self).__name__} expects exactly one '_in' and one '_out' edge.")
        in_name = in_edges[0].data.data
        out_name = out_edges[0].data.data
        if in_name != out_name:
            raise ValueError(f"{type(self).__name__} is in-place: '_in' ({in_name}) and '_out' ({out_name}) "
                             f"must be the same array.")
        desc = sdfg.arrays[in_name]
        if len(desc.shape) != 2:
            raise ValueError(f"{type(self).__name__} target '{in_name}' must be 2-D; got shape {tuple(desc.shape)}.")
        return in_name, desc
