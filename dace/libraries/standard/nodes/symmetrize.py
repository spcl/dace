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
from dace import library, nodes, properties, symbolic
from dace.libraries.standard.environments.tiled_transpose import TiledTranspose
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

        # The ROW map carries the parallelism and is left at ``Default`` for each target to
        # schedule. The column map is pinned Sequential, and that is a launch-model constraint
        # rather than a dependence one: its extent depends on ``__i``, and ``Default`` under a GPU
        # kernel resolves to ``GPU_ThreadBlock``, whose extent IS the block dimension of the launch
        # -- host code, and required to be uniform across the grid, which a triangular row length is
        # not. It costs nothing on the CPU, where ``Default`` under a ``CPU_Multicore`` row map
        # already resolves to Sequential.
        #
        # This expansion is therefore the CPU lowering, and the one to use for a node already
        # inside a kernel. A host-level node on a GPU target picks ``ExpandSymmetrizeCUDA`` instead,
        # which keeps both axes parallel.
        ome, omx = nstate.add_map(f"{node.label}_row", {"__i": f"{node.row_lo}:{node.row_hi}"},
                                  schedule=dace.dtypes.ScheduleType.Default)
        ime, imx = nstate.add_map(f"{node.label}_col", {"__j": f"__i + {node.col_offset}:{node.col_hi}"},
                                  schedule=dace.dtypes.ScheduleType.Sequential)
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


@library.expansion
class ExpandSymmetrizeBoundingBox(ExpandTransformation):
    """Both axes parallel over the triangle's BOUNDING BOX, with the column index clamped.

    The triangular nest cannot go on a GPU as-is: the inner extent depends on the row, and a
    thread-block extent is the launch's block dimension, which has to be uniform across the grid.
    Sequentializing the columns is correct but leaves one thread per ROW -- 1199 threads for
    polybench correlation's paper shape, against the 1.4M element writes it has to do.

    So iterate the bounding box instead. ``__jj`` has the constant extent of the LONGEST row, and
    the column it addresses is clamped to the last one in range::

        __j = min(__i + col_offset + __jj, col_hi - 1)

    Threads past the end of a short row therefore redo that row's last element rather than running
    off it, which needs no guard and stays race-free: the write is ``X[mirror] = X[source]``, the
    source triangle is never written, and every duplicate writes the same value to the same
    address. The cost is the bounding box rather than the triangle -- close to 2x the element
    writes -- which is why the CPU keeps the triangular expansion.
    """

    environments = []

    @staticmethod
    def expansion(node: "Symmetrize", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> dace.SDFG:
        array_name, desc = node.validate(parent_sdfg, parent_state)

        nsdfg = dace.SDFG(f"{node.label}_gpu_sdfg")
        for conn in ("_in", "_out"):
            d = desc.clone()
            d.transient = False
            nsdfg.add_datadesc(conn, d)
        nstate = nsdfg.add_state(f"{node.label}_gpu_state")

        # The longest row is the first one, so its length bounds every other row.
        width = f"({node.col_hi}) - ({node.row_lo}) - {node.col_offset}"
        column = f"min(__i + {node.col_offset} + __jj, ({node.col_hi}) - 1)"
        # ONE 2-D map, not a nest. Both extents are now independent of each other, which is what
        # lets the GPU tiler split the space into grid and block itself with constant tile sizes.
        # A nest would hand the inner map's FULL width to the launch as the block dimension --
        # ``dim3((M - 1), 1, 1)``, which is 1199 threads at correlation's paper shape and past the
        # 1024 a block may hold.
        entry, exit_ = nstate.add_map(f"{node.label}_row", {
            "__i": f"{node.row_lo}:{node.row_hi}",
            "__jj": f"0:{width}",
        },
                                      schedule=dace.dtypes.ScheduleType.Default)
        read_idx = f"__i, {column}" if node.source_upper else f"{column}, __i"
        write_idx = f"{column}, __i" if node.source_upper else f"__i, {column}"

        r = nstate.add_read("_in")
        w = nstate.add_write("_out")
        t = nstate.add_tasklet(f"{node.label}_copy", {"__in"}, {"__out"}, "__out = __in")
        nstate.add_memlet_path(r, entry, t, dst_conn="__in", memlet=dace.Memlet(f"_in[{read_idx}]"))
        nstate.add_memlet_path(t, exit_, w, src_conn="__out", memlet=dace.Memlet(f"_out[{write_idx}]"))
        return nsdfg


@library.expansion
class ExpandSymmetrizeCUDA(ExpandTransformation):
    """Our own tiled kernel: ``dace::cuda_transpose::symmetrize``.

    A symmetrization is a transpose, so one of its two accesses is strided however it is written --
    reading ``X[i, j]`` along a row makes the write to ``X[j, i]`` walk a column. The kernel stages a
    32x32 tile through shared memory so both sides run along rows, and pads the tile to ``[32][33]``
    so the transposed read of it is bank-conflict free.

    Falls back to :class:`ExpandSymmetrizeBoundingBox` when the node's window is not the strict
    triangle of the whole array, which is the only shape the kernel indexes.
    """

    environments = [TiledTranspose]

    @staticmethod
    def canonical_window(node: "Symmetrize", desc) -> bool:
        """Is the node's window the strict triangle of the whole square array?

        The kernel takes one extent and derives both triangles from it, so anything else -- a
        sub-block, a non-square target -- has to go the general way instead of being indexed wrong.
        """
        rows, cols = desc.shape
        # The window bounds are STRING properties; ``pystr_to_symbolic`` is what turns them into
        # comparable expressions (sympy refuses a bare str).
        # Reparsing loses the assumptions and dtype the shape carries, so the window bound and the
        # extent become two sympy instances of one name that ``equal`` then calls undecidable.
        row_lo, row_hi, col_hi, rows, cols = symbolic.equalize_symbols_across(symbolic.pystr_to_symbolic(node.row_lo),
                                                                              symbolic.pystr_to_symbolic(node.row_hi),
                                                                              symbolic.pystr_to_symbolic(node.col_hi),
                                                                              rows, cols)
        return (symbolic.equal(rows, cols) is True and symbolic.equal(row_lo, 0) is True
                and symbolic.equal(col_hi, cols) is True and symbolic.equal(row_hi, cols - node.col_offset) is True)

    @staticmethod
    def expansion(node: "Symmetrize", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG):
        from dace.codegen.targets.cpp import sym2cpp
        array_name, desc = node.validate(parent_sdfg, parent_state)
        if not ExpandSymmetrizeCUDA.canonical_window(node, desc):
            # ``ExpandTransformation.apply`` attaches THIS class's ``environments`` to whatever is
            # returned; the SDFG fallback calls no kernel, so it must not carry the CUDA header.
            ExpandSymmetrizeCUDA.environments = []
            return ExpandSymmetrizeBoundingBox.expansion(node, parent_state, parent_sdfg)
        ExpandSymmetrizeCUDA.environments = [TiledTranspose]

        state_id = parent_state.parent_graph.node_id(parent_state)
        idstr = f'{parent_sdfg.name}_{state_id}_{parent_state.node_id(node)}'
        ctype = desc.dtype.base_type.ctype
        prototype = (f'DACE_EXPORTED gpuError_t __dace_symmetrize_{idstr}({ctype} *__sym_x, int __sym_n, '
                     f'int __sym_ld, int __sym_off, int __sym_upper, gpuStream_t __sym_stream);')
        parent_sdfg.append_global_code(prototype + '\n')
        # No ``DACE_GPU_CHECK`` in this body: the macro reports through ``__state``, which a free
        # function in the CUDA unit does not have. The status is returned and checked at the call.
        parent_sdfg.append_global_code(
            f'{prototype}\n'
            f'gpuError_t __dace_symmetrize_{idstr}({ctype} *__sym_x, int __sym_n, int __sym_ld, int __sym_off, '
            f'int __sym_upper, gpuStream_t __sym_stream) {{\n'
            f'    return ::dace::cuda_transpose::symmetrize<{ctype}>(__sym_x, __sym_n, __sym_ld, __sym_off, '
            f'__sym_upper != 0, __sym_stream);\n'
            f'}}\n', 'cuda')

        code = (f'(void)_in;  // the node is in place: ``_in`` and ``_out`` name the same array\n'
                f'DACE_GPU_CHECK(__dace_symmetrize_{idstr}(_out, (int)({sym2cpp(desc.shape[0])}), '
                f'(int)({sym2cpp(desc.strides[0])}), {int(node.col_offset)}, {int(bool(node.source_upper))}, '
                f'__dace_current_stream));')
        return nodes.Tasklet(node.name, {'_in': dace.dtypes.pointer(desc.dtype.base_type)},
                             {'_out': dace.dtypes.pointer(desc.dtype.base_type)},
                             code,
                             language=dace.dtypes.Language.CPP)


@library.node
class Symmetrize(nodes.LibraryNode):
    """Symmetrize a square matrix by mirroring one triangle into the other.

    In-place: ``_in`` and ``_out`` both connect to the same 2-D array. The node
    fills the triangle opposite ``source_upper`` from the source triangle; the
    diagonal and the source triangle are left untouched. The triangular
    iteration space is ``__i in [row_lo, row_hi)``, ``__j in [__i + col_offset,
    col_hi)`` (``col_offset >= 1`` excludes the diagonal).
    """

    implementations = {
        "pure": ExpandSymmetrizePure,
        "CUDA": ExpandSymmetrizeCUDA,
        "bounding_box": ExpandSymmetrizeBoundingBox,
    }
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
