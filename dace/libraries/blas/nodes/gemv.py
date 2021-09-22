# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import copy
from dace import properties, symbolic
import dace.library
import dace.sdfg.nodes
from dace.sdfg import SDFG, SDFGState
from dace import memlet as mm, data as dt
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.blas.nodes.matmul import _get_matmul_operands
from dace.libraries.blas import blas_helpers
from dace.frontend.common import op_repository as oprepo
from dace.libraries.blas import environments
import numpy as np
import warnings


@dace.library.expansion
class ExpandGemvPure(ExpandTransformation):

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        node.validate(parent_sdfg, parent_state)
        sdfg = dace.SDFG(node.label + "_sdfg")
        ((edge_a, outer_array_a, shape_a, strides_a), (edge_x, outer_array_x,
                                                       shape_x, strides_x),
         (edge_y, outer_array_y, shape_y,
          strides_y)) = _get_matmul_operands(node,
                                             parent_state,
                                             parent_sdfg,
                                             name_lhs="_A",
                                             name_rhs="_x",
                                             name_out="_y")
        dtype_a = outer_array_a.dtype.type
        dtype_x = outer_array_x.dtype.type
        dtype_y = outer_array_y.dtype.type

        if outer_array_a.dtype.veclen > 1 or outer_array_x.dtype.veclen > 1:
            raise NotImplementedError("Vectorization for pure GEMV NYI.")

        if node.transA:
            trans_shape_a = list(reversed(shape_a))
        else:
            trans_shape_a = shape_a

        if trans_shape_a[1] != shape_x[0]:
            raise SyntaxError(
                "Matrix-vector product size mismatch: {} vs. {}".format(
                    trans_shape_a[1], shape_x[0]))

        N, M = trans_shape_a[0], trans_shape_a[1]

        if outer_array_a.storage != outer_array_x.storage:
            raise ValueError("Input matrices must have same storage")
        storage = outer_array_a.storage

        _, array_a = sdfg.add_array("_A",
                                    shape_a,
                                    dtype_a,
                                    strides=strides_a,
                                    storage=storage)
        _, array_x = sdfg.add_array("_x",
                                    shape_x,
                                    dtype_x,
                                    strides=strides_x,
                                    storage=storage)
        _, array_y = sdfg.add_array("_y",
                                    shape_y,
                                    dtype_y,
                                    strides=strides_y,
                                    storage=storage)

        mul_program = "__out = {} * __A * __x".format(node.alpha)

        init_state = sdfg.add_state(node.label + "_initstate")
        state = sdfg.add_state_after(init_state, node.label + "_state")

        if node.beta == 0:
            mul_out, mul_out_array = "_y", array_y
            output_nodes = None
        else:
            mul_out, mul_out_array = tmp, array_tmp = sdfg.add_temp_transient(
                shape_y, dtype_y, storage=storage)

            access_tmp = state.add_read(tmp)
            output_nodes = {mul_out: access_tmp}

        # Initialization map
        init_state.add_mapped_tasklet(
            "gemv_init", {
                "_o%d" % i: "0:%s" % symbolic.symstr(d)
                for i, d in enumerate(shape_y)
            }, {},
            "out = 0", {
                "out":
                dace.Memlet("{}[{}]".format(
                    mul_out, ",".join(["_o%d" % i
                                       for i in range(len(shape_y))])))
            },
            external_edges=True)

        # Multiplication map
        state.add_mapped_tasklet("_GEMV_", {
            "__i%d" % i: "0:%s" % s
            for i, s in enumerate([N, M])
        }, {
            "__A":
            dace.Memlet(
                "_A[{}]".format("__i1, __i0" if node.transA else "__i0, __i1")),
            "__x":
            dace.Memlet("_x[__i1]")
        },
                                 mul_program, {
                                     "__out":
                                     dace.Memlet(f"{mul_out}[__i0]",
                                                 wcr="lambda x, y: x + y")
                                 },
                                 external_edges=True,
                                 output_nodes=output_nodes)

        add_program = "__y_out = ({} * __y_in) + __tmp".format(node.beta)

        memlet_idx = "__i"

        # addition map
        if node.beta != 0:
            state.add_mapped_tasklet("_Add_", {"__i": "0:{}".format(N)}, {
                "__y_in": dace.Memlet(f"_y[{memlet_idx}]"),
                "__tmp": dace.Memlet(f"{mul_out}[__i]"),
            },
                                     add_program,
                                     {"__y_out": dace.Memlet("_y[__i]")},
                                     external_edges=True,
                                     input_nodes={mul_out: access_tmp})

        return sdfg


@dace.library.expansion
class ExpandGemvFpgaAccumulate(ExpandTransformation):
    """
    This FPGA-oriented expansion iterates over the input matrix A in simple
    row-major order, with optional tiling in both dimensions, where the tiles
    are also traversed in simple row-major order. This means that y is only
    written once, but x is read for every tile in the y-dimension.

    The implementation requires accumulation on the output, and does NOT assume
    native accumulation for the given data type. Instead it uses multiple
    partial sums to ensure that II=1, and only writes the final accumulated
    value once it has been combined from the partial sums.

    This works for both transposed and non-transposed A, but vectorization is
    only implemented for non-transposed A.
    """
    # The above corresponds to gemv_v1 in FBLAS

    environments = []

    @staticmethod
    def expansion(node,
                  parent_state,
                  parent_sdfg,
                  tile_size_x=None,
                  tile_size_y=None,
                  num_partial_sums=16):
        """
        :param node: Node to expand.
        :param parent_state: State that the node is in.
        :param parent_sdfg: SDFG that the node is in.
        :param tile_size_x: Tile size along the dimension of the vector x. If
                            set to None, no tiling is used, corresponding to
                            setting the tile size equal to the full size of x.
        :param tile_size_y: Tile size along the dimension of the vector y. If
                            set to None, no tiling is used, corresponding to
                            setting the tile size equal to the full size of y.
        :param num_partial_sums: The number of distinct registers to accumulate
                                 contributions to the final sum into. Should be
                                 a power of two, and should be higher than the
                                 latency of adding two numbers of the given
                                 data type.
        """

        node.validate(parent_sdfg, parent_state)

        for e in parent_state.in_edges(node):
            if e.dst_conn == "_A":
                desc_a = parent_sdfg.arrays[e.data.data]
            elif e.dst_conn == "_x":
                desc_x = parent_sdfg.arrays[e.data.data]
        for e in parent_state.out_edges(node):
            if e.src_conn == "_y":
                desc_y = parent_sdfg.arrays[e.data.data]

        sdfg = dace.SDFG("gemv")
        state = sdfg.add_state("gemv")

        alpha = node.alpha
        beta = node.beta

        # Create local versions of input data nodes
        desc_a = desc_a.clone()
        desc_a.transient = False
        sdfg.add_datadesc("_A", desc_a)
        desc_x = desc_x.clone()
        desc_x.transient = False
        sdfg.add_datadesc("_x", desc_x)
        desc_y = desc_y.clone()
        desc_y.transient = False
        sdfg.add_datadesc("_y", desc_y)

        if node.transA and desc_a.dtype.veclen > 1:
            raise NotImplementedError(
                "Vectorization not implemented for transposed A.")

        # Create accesses
        read_a = state.add_read("_A")
        read_x = state.add_read("_x")
        if beta != 0:
            read_y = state.add_read("_y")
        write_y = state.add_write("_y")

        size_x = desc_x.shape[0]
        size_y = desc_y.shape[0]
        if tile_size_x is None:
            tile_size_x = size_x
        if tile_size_y is None:
            tile_size_y = size_y
        num_tiles_y = f"{size_y}/{tile_size_y}"
        num_tiles_x = f"{size_x}/{tile_size_x}"

        veclen = desc_a.dtype.veclen

        # Create tile map
        y_tile_entry, y_tile_exit = state.add_map(
            "y_tiles", {"ty": f"0:{num_tiles_y}"},
            schedule=dace.ScheduleType.FPGA_Device)
        x_tile_entry, x_tile_exit = state.add_map(
            "x_tiles", {"tx": f"0:{num_tiles_x}"},
            schedule=dace.ScheduleType.FPGA_Device)

        # Create y map
        y_entry, y_exit = state.add_map("y", {"iy": f"0:{tile_size_y}"},
                                        schedule=dace.ScheduleType.FPGA_Device)

        # Create x map
        x_entry, x_exit = state.add_map("x", {"ix": f"0:{tile_size_x}"},
                                        schedule=dace.ScheduleType.FPGA_Device)

        # Local buffer of x
        sdfg.add_array("x_local", (tile_size_x, ),
                       desc_x.dtype,
                       storage=dace.StorageType.FPGA_Local,
                       transient=True)
        x_local_access = state.add_read("x_local")

        if beta != 0:
            raise NotImplementedError("Not yet implemented.")

        multiply_tasklet = state.add_tasklet("multiply", {"A_in", "x_in"},
                                             {f"product": desc_a.dtype},
                                             "product = A_in * x_in")

        if isinstance(desc_a, dt.Stream):
            subset = "0"
        elif node.transA:
            subset = f"tx * {tile_size_x} + ix, ty * {tile_size_y} + iy"
        else:
            subset = f"ty * {tile_size_y} + iy, tx * {tile_size_x} + ix"
        state.add_memlet_path(read_a,
                              y_tile_entry,
                              x_tile_entry,
                              y_entry,
                              x_entry,
                              multiply_tasklet,
                              dst_conn="A_in",
                              memlet=dace.Memlet(f"_A[{subset}]"))
        read_x_entry, read_x_exit = state.add_map(
            "read_x", {"ix": f"0:{tile_size_x}"},
            schedule=dace.ScheduleType.FPGA_Device)
        subset = ("0" if isinstance(desc_x, dt.Stream) else
                  f"tx*{tile_size_x} + ix")
        read_x_tasklet = state.add_tasklet("read_x", {"x_memory"}, {"x_buffer"},
                                           "x_buffer = x_memory")
        state.add_memlet_path(read_x,
                              y_tile_entry,
                              x_tile_entry,
                              read_x_entry,
                              read_x_tasklet,
                              dst_conn="x_memory",
                              memlet=dace.Memlet(f"_x[{subset}]"))
        state.add_memlet_path(read_x_tasklet,
                              read_x_exit,
                              x_local_access,
                              src_conn="x_buffer",
                              memlet=dace.Memlet(f"x_local[ix]"))
        state.add_memlet_path(x_local_access,
                              y_entry,
                              x_entry,
                              multiply_tasklet,
                              dst_conn="x_in",
                              memlet=dace.Memlet(f"x_local[ix]"))

        # Write to buffer
        sdfg.add_array("product_vector", (1, ),
                       desc_a.dtype,
                       transient=True,
                       storage=dace.StorageType.FPGA_Local)
        product_vector = state.add_access("product_vector")
        state.add_memlet_path(multiply_tasklet,
                              product_vector,
                              src_conn="product",
                              memlet=dace.Memlet(f"product_vector[0]"))

        # Vector length conversion
        sdfg.add_array("product_scalar", (veclen, ),
                       desc_a.dtype.base_type,
                       transient=True,
                       storage=dace.StorageType.FPGA_Local)
        product_scalar = state.add_access("product_scalar")
        state.add_memlet_path(product_vector,
                              product_scalar,
                              memlet=dace.Memlet(f"product_vector[0]",
                                                 other_subset=f"0:{veclen}"))

        # Now we need to collapse this
        reduce_vector_entry, reduce_vector_exit = state.add_map(
            "reduce_vector", {"u": f"0:{veclen}"},
            schedule=dace.ScheduleType.FPGA_Device,
            unroll=True)

        reduce_vector_tasklet = state.add_tasklet(
            "reduce_vector", {"product_in", "acc_in"}, {"acc_out"},
            "acc_out = product_in + acc_in")
        state.add_memlet_path(product_scalar,
                              reduce_vector_entry,
                              reduce_vector_tasklet,
                              dst_conn="product_in",
                              memlet=dace.Memlet(f"{product_scalar}[u]"))

        # Add accumulation register
        sdfg.add_array("accumulate_product", (1, ),
                       desc_a.dtype.base_type,
                       transient=True,
                       storage=dace.StorageType.FPGA_Local)
        accumulate_product_read = state.add_access("accumulate_product")
        accumulate_product_write = state.add_access("accumulate_product")

        # Initialize it to zero
        init_reduce_vector_tasklet = state.add_tasklet("init_reduce_vector", {},
                                                       {"acc_out"},
                                                       "acc_out = 0")
        state.add_memlet_path(x_entry,
                              init_reduce_vector_tasklet,
                              memlet=dace.Memlet())
        state.add_memlet_path(init_reduce_vector_tasklet,
                              accumulate_product_read,
                              src_conn="acc_out",
                              memlet=dace.Memlet(f"accumulate_product[0]"))

        # Connect it to the tasklet
        state.add_memlet_path(accumulate_product_read,
                              reduce_vector_entry,
                              reduce_vector_tasklet,
                              dst_conn="acc_in",
                              memlet=dace.Memlet(f"accumulate_product[0]"))
        state.add_memlet_path(reduce_vector_tasklet,
                              reduce_vector_exit,
                              accumulate_product_write,
                              src_conn="acc_out",
                              memlet=dace.Memlet(f"accumulate_product[0]"))

        # Partial sums
        sdfg.add_array("partial_sums", (num_partial_sums, ),
                       desc_y.dtype,
                       storage=dace.StorageType.FPGA_Registers,
                       transient=True)
        partial_sum_read = state.add_read("partial_sums")
        partial_sum_write = state.add_access("partial_sums")

        # Output array
        sdfg.add_array("y_local", (tile_size_y, ),
                       desc_y.dtype,
                       storage=dace.StorageType.FPGA_Local,
                       transient=True)

        # Now we need to actually accumulate into a local register of y
        y_local_read = state.add_read("y_local")
        y_local_write = state.add_read("y_local")
        update_y_tasklet = state.add_tasklet(
            "update_y", {"y_in", "acc_in"}, {"acc_out"}, f"""\
prev = acc_in if ix >= {num_partial_sums} else 0
acc_out = prev + y_in""")
        state.add_memlet_path(accumulate_product_write,
                              update_y_tasklet,
                              dst_conn="y_in",
                              memlet=dace.Memlet(f"accumulate_product[0]"))
        state.add_memlet_path(
            partial_sum_read,
            x_entry,
            update_y_tasklet,
            dst_conn="acc_in",
            memlet=dace.Memlet(f"partial_sums[ix%{num_partial_sums}]"))
        state.add_memlet_path(y_tile_entry, y_local_read, memlet=dace.Memlet())
        state.add_memlet_path(y_entry, partial_sum_read, memlet=dace.Memlet())
        state.add_memlet_path(
            update_y_tasklet,
            x_exit,
            partial_sum_write,
            src_conn="acc_out",
            memlet=dace.Memlet(f"partial_sums[ix%{num_partial_sums}]"))

        # Reduce the partial sums
        reduce_sums_entry, reduce_sums_exit = state.add_map(
            "reduce_partial_sums", {"u": f"0:{num_partial_sums}"},
            schedule=dace.ScheduleType.FPGA_Device,
            unroll=True)
        reduce_sums_tasklet = state.add_tasklet(
            "reduce_partial_sums", {"sum_in", "val_in"}, {"sum_out"}, """
prev = sum_in if u > 0 else 0
sum_out = prev + val_in""")
        sdfg.add_array("accumulate_sum", (1, ),
                       desc_y.dtype,
                       transient=True,
                       storage=dace.StorageType.FPGA_Local)
        accumulate_sum_read = state.add_access("accumulate_sum")
        accumulate_sum_write = state.add_access("accumulate_sum")
        state.add_memlet_path(y_entry,
                              accumulate_sum_read,
                              memlet=dace.Memlet())
        state.add_memlet_path(accumulate_sum_read,
                              reduce_sums_entry,
                              reduce_sums_tasklet,
                              dst_conn="sum_in",
                              memlet=dace.Memlet("accumulate_sum[0]"))
        state.add_memlet_path(reduce_sums_tasklet,
                              reduce_sums_exit,
                              accumulate_sum_write,
                              src_conn="sum_out",
                              memlet=dace.Memlet("accumulate_sum[0]"))
        state.add_memlet_path(partial_sum_write,
                              reduce_sums_entry,
                              reduce_sums_tasklet,
                              dst_conn="val_in",
                              memlet=dace.Memlet("partial_sums[u]"))

        # Combine with y buffer
        combine_tasklet = state.add_tasklet(
            "combine_y", {"val", "buffer_in"}, {"buffer_out"}, """\
prev = buffer_in if tx > 0 else 0
buffer_out = prev + val""")
        state.add_memlet_path(accumulate_sum_write,
                              combine_tasklet,
                              dst_conn="val",
                              memlet=dace.Memlet("accumulate_sum[0]"))
        state.add_memlet_path(y_local_read,
                              x_tile_entry,
                              y_entry,
                              combine_tasklet,
                              dst_conn="buffer_in",
                              memlet=dace.Memlet("y_local[iy]"))

        state.add_memlet_path(combine_tasklet,
                              y_exit,
                              x_tile_exit,
                              y_local_write,
                              src_conn="buffer_out",
                              memlet=dace.Memlet(f"y_local[iy]"))

        subset = ("0" if isinstance(desc_y, dt.Stream) else
                  f"ty*{tile_size_y} + iy")
        write_y_entry, write_y_exit = state.add_map(
            "write_y", {"iy": f"0:{tile_size_y}"},
            schedule=dace.ScheduleType.FPGA_Device)
        write_y_tasklet = state.add_tasklet("write_y", {"y_buffer"},
                                            {"y_memory"}, "y_memory = y_buffer")
        state.add_memlet_path(y_local_write,
                              write_y_entry,
                              write_y_tasklet,
                              dst_conn="y_buffer",
                              memlet=dace.Memlet(f"y_local[iy]"))
        state.add_memlet_path(write_y_tasklet,
                              write_y_exit,
                              y_tile_exit,
                              write_y,
                              src_conn="y_memory",
                              memlet=dace.Memlet(f"_y[{subset}]"))

        return sdfg


@dace.library.expansion
class ExpandGemvFpgaTilesByColumn(ExpandTransformation):
    """
    FPGA-oriented expansion that reads the input matrix A in column-major
    order, such that consecutive values are accumulated into different
    registers, avoiding a loop-carried dependency due to accumulation.

    The matrix can optionally be tiled, where the tiles will be traversed in
    row-major order in order to bound the size of the output buffer to the tile
    size. The tile size on y must be larger than the latency of addition for
    the given data type.

    This expansion supports both transposed A and non-transposed A, but
    vectorization is only implemented for transposed A.
    """
    # This corresponds to gemv_v2 in FBLAS

    environments = []

    @staticmethod
    def expansion(node, state, sdfg, tile_size_x=None, tile_size_y=None):
        """
        :param node: Node to expand.
        :param parent_state: State that the node is in.
        :param parent_sdfg: SDFG that the node is in.
        :param tile_size_x: Tile size along the dimension of the vector x. If
                            set to None, no tiling is used, corresponding to
                            setting the tile size equal to the full size of x.
        :param tile_size_y: Tile size along the dimension of the vector y. If
                            set to None, no tiling is used, corresponding to
                            setting the tile size equal to the full size of y.
        """

        node.validate(sdfg, state)

        for e in state.in_edges(node):
            if e.dst_conn == "_A":
                desc_a = sdfg.arrays[e.data.data]
            elif e.dst_conn == "_x":
                desc_x = sdfg.arrays[e.data.data]
        for e in state.out_edges(node):
            if e.src_conn == "_y":
                desc_y = sdfg.arrays[e.data.data]

        sdfg = dace.SDFG("gemv")
        state = sdfg.add_state("gemv")

        alpha = node.alpha
        beta = node.beta

        # Create local versions of input data nodes
        desc_a = desc_a.clone()
        desc_a.transient = False
        sdfg.add_datadesc("_A", desc_a)
        desc_x = desc_x.clone()
        desc_x.transient = False
        sdfg.add_datadesc("_x", desc_x)
        desc_y = desc_y.clone()
        desc_y.transient = False
        sdfg.add_datadesc("_y", desc_y)

        if not node.transA and desc_a.dtype.veclen > 1:
            raise NotImplementedError(
                "Vectorization not implemented for non-transposed A.")

        # Create accesses
        read_a = state.add_read("_A")
        read_x = state.add_read("_x")
        if beta != 0:
            read_y = state.add_read("_y")
        write_y = state.add_write("_y")

        size_x = desc_x.shape[0]
        size_y = desc_y.shape[0]
        if tile_size_x is None:
            tile_size_x = size_x
        if tile_size_y is None:
            tile_size_y = size_y
        num_tiles_y = f"{size_y}/{tile_size_y}"
        num_tiles_x = f"{size_x}/{tile_size_x}"

        # Create y tile map
        y_tile_entry, y_tile_exit = state.add_map(
            "y_tiles", {"ty": f"0:{num_tiles_y}"},
            schedule=dace.ScheduleType.FPGA_Device)

        # Create buffer
        sdfg.add_array("y_local", (tile_size_y, ),
                       desc_y.dtype,
                       storage=dace.StorageType.FPGA_Local,
                       transient=True)
        y_local = state.add_access("y_local")
        y_local_write = state.add_access("y_local")

        # Initialize buffer
        init_entry, init_exit = state.add_map(
            "init", {"iy": f"0:{tile_size_y}"},
            schedule=dace.ScheduleType.FPGA_Device)
        if beta != 0:
            if isinstance(desc_y, dt.Stream):
                subset = "0"
            else:
                subset = f"ty*{tile_size_y}+iy"
            init_tasklet = state.add_tasklet(
                "init", {"y_in"}, {"y_out"},
                f"y_out = {desc_y.dtype.base_type.ctype}({beta}) * y_in")
            state.add_memlet_path(read_y,
                                  y_tile_entry,
                                  init_entry,
                                  init_tasklet,
                                  dst_conn="y_in",
                                  memlet=dace.Memlet(f"_y[{subset}]"))
            state.add_memlet_path(init_tasklet,
                                  init_exit,
                                  y_local,
                                  src_conn="y_out",
                                  memlet=dace.Memlet(f"y_local[iy]"))
        else:
            state.add_memlet_path(y_tile_entry,
                                  init_entry,
                                  memlet=dace.Memlet())
            init_tasklet = state.add_tasklet("init", {}, {"y_out"}, "y_out = 0")
            state.add_memlet_path(init_entry,
                                  init_tasklet,
                                  memlet=dace.Memlet())
            state.add_memlet_path(init_tasklet,
                                  init_exit,
                                  y_local,
                                  src_conn="y_out",
                                  memlet=dace.Memlet("y_local[iy]"))

        # Create x tile map
        x_tile_entry, x_tile_exit = state.add_map(
            "x_tiles", {"tx": f"0:{num_tiles_x}"},
            schedule=dace.ScheduleType.FPGA_Device)

        # Create loop over tile size in x
        x_entry, x_exit = state.add_map("x", {"ix": f"0:{tile_size_x}"},
                                        schedule=dace.ScheduleType.FPGA_Device)

        # Buffer a scalar value of x
        sdfg.add_array("x_local", (1, ),
                       desc_x.dtype,
                       transient=True,
                       storage=dace.StorageType.FPGA_Local)
        x_local = state.add_access("x_local")
        subset = "0" if isinstance(desc_x,
                                   dt.Stream) else f"tx*{tile_size_x}+ix"
        state.add_memlet_path(read_x,
                              y_tile_entry,
                              x_tile_entry,
                              x_entry,
                              x_local,
                              memlet=dace.Memlet(f"_x[{subset}]"))

        # Create loop over tile size in y
        y_entry, y_exit = state.add_map("y", {"iy": f"0:{tile_size_y}"},
                                        schedule=dace.ScheduleType.FPGA_Device)

        # Do computation
        tasklet = state.add_tasklet("gemv", {"A_in", "x_in", "y_in"}, {"y_out"},
                                    f"y_out = y_in + {alpha} * A_in * x_in")
        state.add_memlet_path(y_local,
                              x_tile_entry,
                              x_entry,
                              y_entry,
                              tasklet,
                              dst_conn="y_in",
                              memlet=dace.Memlet("y_local[iy]"))
        state.add_memlet_path(x_local,
                              y_entry,
                              tasklet,
                              dst_conn="x_in",
                              memlet=dace.Memlet("x_local[0]"))
        state.add_memlet_path(tasklet,
                              y_exit,
                              x_exit,
                              x_tile_exit,
                              y_local_write,
                              src_conn="y_out",
                              memlet=dace.Memlet("y_local[iy]"))
        if isinstance(desc_a, dt.Stream):
            subset = "0"
        elif node.transA:
            subset = f"tx * {tile_size_x} + ix, ty * {tile_size_y} + iy"
        else:
            subset = f"ty * {tile_size_y} + iy, tx * {tile_size_x} + ix"
        state.add_memlet_path(read_a,
                              y_tile_entry,
                              x_tile_entry,
                              x_entry,
                              y_entry,
                              tasklet,
                              dst_conn="A_in",
                              memlet=dace.Memlet(f"_A[{subset}]"))

        # Write out tile of y
        write_y_entry, write_y_exit = state.add_map(
            "write_y", {"iy": f"0:{tile_size_y}"},
            schedule=dace.ScheduleType.FPGA_Device)
        write_y_tasklet = state.add_tasklet("write_y", {"y_in"}, {"y_out"},
                                            "y_out = y_in")
        subset = ("0" if isinstance(desc_y, dt.Stream) else
                  f"ty * {tile_size_y} + iy")
        state.add_memlet_path(y_local_write,
                              write_y_entry,
                              write_y_tasklet,
                              dst_conn="y_in",
                              memlet=dace.Memlet("y_local[iy]"))
        state.add_memlet_path(write_y_tasklet,
                              write_y_exit,
                              y_tile_exit,
                              write_y,
                              src_conn="y_out",
                              memlet=dace.Memlet(f"_y[{subset}]"))

        return sdfg


@dace.library.expansion
class ExpandGemvCuBLAS(ExpandTransformation):

    environments = [environments.cublas.cuBLAS]

    @staticmethod
    def expansion(node: 'Gemv', state, sdfg, m=None, n=None, **kwargs):
        node.validate(sdfg, state)

        ((edge_a, outer_array_a, shape_a, strides_a), (edge_x, outer_array_x,
                                                       shape_x, strides_x),
         (edge_y, outer_array_y, shape_y,
          strides_y)) = _get_matmul_operands(node,
                                             state,
                                             sdfg,
                                             name_lhs="_A",
                                             name_rhs="_x",
                                             name_out="_y")
        dtype_a = outer_array_a.dtype.type
        dtype = outer_array_x.dtype.base_type
        veclen = outer_array_x.dtype.veclen
        m = m or node.m
        n = n or node.n
        if m is None:
            m = shape_y[0]
        if n is None:
            n = shape_x[0]

        transA = node.transA
        if strides_a[0] == 1:
            transA = not transA
            lda = strides_a[1]
        elif strides_a[1] == 1:
            lda = strides_a[0]
        else:
            warnings.warn('Matrix must be contiguous in at least '
                          'one dimension. Falling back to pure expansion.')
            return ExpandGemvPure.expansion(node,
                                            state,
                                            sdfg,
                                            m=m,
                                            n=n,
                                            **kwargs)

        trans = 'CUBLAS_OP_N' if transA else 'CUBLAS_OP_T'
        if not node.transA:
            m, n = n, m

        if veclen != 1:
            warnings.warn('Vector GEMV not supported, falling back to pure')
            return ExpandGemvPure.expansion(node,
                                            state,
                                            sdfg,
                                            m=m,
                                            n=n,
                                            **kwargs)

        func, ctype, runtimetype = blas_helpers.cublas_type_metadata(dtype)
        func += 'gemv'
        call_prefix = environments.cublas.cuBLAS.handle_setup_code(node)
        call_suffix = ''

        # Handle alpha / beta
        constants = {
            1.0:
            f"__state->cublas_handle.Constants(__dace_cuda_device).{runtimetype}Pone()",
            0.0:
            f"__state->cublas_handle.Constants(__dace_cuda_device).{runtimetype}Zero()",
        }
        if node.alpha not in constants or node.beta not in constants:
            # Deal with complex input constants
            if isinstance(node.alpha, complex):
                alpha = f'{dtype.ctype}({node.alpha.real}, {node.alpha.imag})'
            else:
                alpha = f'{dtype.ctype}({node.alpha})'
            if isinstance(node.beta, complex):
                beta = f'{dtype.ctype}({node.beta.real}, {node.beta.imag})'
            else:
                beta = f'{dtype.ctype}({node.beta})'

            # Set pointer mode to host
            call_prefix += f'''cublasSetPointerMode(__dace_cublas_handle, CUBLAS_POINTER_MODE_HOST);
            {dtype.ctype} alpha = {alpha};
            {dtype.ctype} beta = {beta};
            '''
            call_suffix += '''
cublasSetPointerMode(__dace_cublas_handle, CUBLAS_POINTER_MODE_DEVICE);
            '''
            alpha = f'({ctype} *)&alpha'
            beta = f'({ctype} *)&beta'
        else:
            alpha = constants[node.alpha]
            beta = constants[node.beta]

        code = (call_prefix + f"""
cublas{func}(__dace_cublas_handle, {trans}, {m}, {n}, {alpha}, _A, {lda},
             _x, {strides_x[0]}, {beta}, _y, {strides_y[0]});
                """ + call_suffix)

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)

        return tasklet


@dace.library.expansion
class ExpandGemvOpenBLAS(ExpandTransformation):

    environments = [environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node: 'Gemv', state, sdfg, m=None, n=None, **kwargs):
        from dace.sdfg.scope import is_devicelevel_gpu
        if is_devicelevel_gpu(sdfg, state, node):
            return ExpandGemvPure.expansion(node, state, sdfg)

        node.validate(sdfg, state)

        ((edge_a, outer_array_a, shape_a, strides_a), (edge_x, outer_array_x,
                                                       shape_x, strides_x),
         (edge_y, outer_array_y, shape_y,
          strides_y)) = _get_matmul_operands(node,
                                             state,
                                             sdfg,
                                             name_lhs="_A",
                                             name_rhs="_x",
                                             name_out="_y")
        dtype_a = outer_array_a.dtype.type
        dtype = outer_array_x.dtype.base_type
        veclen = outer_array_x.dtype.veclen
        m = m or node.m
        n = n or node.n
        if m is None:
            m = shape_y[0]
        if n is None:
            n = shape_x[0]

        transA = node.transA
        if strides_a[0] == 1:
            transA = not transA
            lda = strides_a[1]
        elif strides_a[1] == 1:
            lda = strides_a[0]
        else:
            warnings.warn('Matrix must be contiguous in at least '
                          'one dimension. Falling back to pure expansion.')
            return ExpandGemvPure.expansion(node,
                                            state,
                                            sdfg,
                                            m=m,
                                            n=n,
                                            **kwargs)

        layout = 'CblasColMajor'
        trans = 'CblasNoTrans' if transA else 'CblasTrans'
        if not node.transA:
            m, n = n, m

        if veclen != 1:
            warnings.warn('Vector GEMV not supported, falling back to pure.')
            return ExpandGemvPure.expansion(node,
                                            state,
                                            sdfg,
                                            m=m,
                                            n=n,
                                            **kwargs)

        func, ctype, runtimetype = blas_helpers.cublas_type_metadata(dtype)
        func = func.lower() + 'gemv'

        code = f"""cblas_{func}({layout}, {trans}, {m}, {n}, {node.alpha}, _A, {lda},
                                _x, {strides_x[0]}, {node.beta}, _y, {strides_y[0]});"""

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)

        return tasklet


@dace.library.expansion
class ExpandGemvMKL(ExpandTransformation):
    environments = [environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(*args, **kwargs):
        return ExpandGemvOpenBLAS.expansion(*args, **kwargs)


@dace.library.expansion
class ExpandGemvPBLAS(ExpandTransformation):

    environments = []

    @staticmethod
    def expansion(node: 'Gemv', state, sdfg, m=None, n=None, **kwargs):
        node.validate(sdfg, state)
        ((edge_a, outer_array_a, shape_a, strides_a), (edge_x, outer_array_x,
                                                       shape_x, strides_x),
         (edge_y, outer_array_y, shape_y,
          strides_y)) = _get_matmul_operands(node,
                                             state,
                                             sdfg,
                                             name_lhs="_A",
                                             name_rhs="_x",
                                             name_out="_y")
        dtype_a = outer_array_a.dtype.type
        dtype = outer_array_x.dtype.base_type
        veclen = outer_array_x.dtype.veclen
        m = m or node.m
        n = n or node.n
        if m is None:
            m = shape_y[0]
        if n is None:
            n = shape_x[0]

        transA = node.transA

        Px = dace.symbol('Px', dtype=dace.int32, integer=True, positive=True)
        Py = dace.symbol('Py', dtype=dace.int32, integer=True, positive=True)
        try:
            sdfg.add_symbol('Px', dace.int32)
            sdfg.add_symbol('Py', dace.int32)
        except FileExistsError:
            pass

        @dace.program
        def _gemNv_pblas(_A: dtype[m, n], _x: dtype[n], _y: dtype[m]):
            lA = np.empty((m // Px, n // Py), dtype=_A.dtype)
            lx = np.empty((n // Px, ), dtype=_x.dtype)
            dace.comm.BCScatter(_A, lA, (m // Px, n // Py))
            dace.comm.BCScatter(_x, lx, (n // Px, 1))
            ly = distr.MatMult(_A, _x, lA, lx, (m // Px, n // Py), (n // Px, 1))
            dace.comm.BCGather(ly, _y, (m // Px, 1))

        @dace.program
        def _gemTv_pblas(_A: dtype[m, n], _x: dtype[m], _y: dtype[n]):
            lA = np.empty((m // Px, n // Py), dtype=_A.dtype)
            lx = np.empty((m // Px, ), dtype=_x.dtype)
            dace.comm.BCScatter(_A, lA, (m // Px, n // Py))
            dace.comm.BCScatter(_x, lx, (m // Px, 1))
            ly = distr.MatMult(_x, _A, lx, lA, (m // Px, 1), (m // Px, n // Py))
            dace.comm.BCGather(ly, _y, (n // Px, 1))

        # NOTE: The following is done to avoid scalar promotion, which results
        # in ValueError: Node type "BlockCyclicScatter" not supported for
        # promotion
        if transA:
            sdfg = _gemTv_pblas.to_sdfg(strict=False)
        else:
            sdfg = _gemNv_pblas.to_sdfg(strict=False)
        sdfg.apply_strict_transformations()
        return sdfg


@dace.library.node
class Gemv(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "pure": ExpandGemvPure,
        "OpenBLAS": ExpandGemvOpenBLAS,
        "MKL": ExpandGemvMKL,
        "cuBLAS": ExpandGemvCuBLAS,
        "FPGA_Accumulate": ExpandGemvFpgaAccumulate,
        "FPGA_TilesByColumn": ExpandGemvFpgaTilesByColumn,
        "PBLAS": ExpandGemvPBLAS
    }
    default_implementation = None

    # Object fields
    alpha = properties.SymbolicProperty(allow_none=False, default=1)
    beta = properties.SymbolicProperty(allow_none=False, default=0)

    transA = properties.Property(
        dtype=bool, desc="Whether to transpose A before multiplying")

    n = properties.SymbolicProperty(allow_none=True, default=None)
    m = properties.SymbolicProperty(allow_none=True, default=None)

    def __init__(self, name, location=None, transA=False, alpha=1, beta=0):
        super().__init__(
            name,
            location=location,
            inputs={"_A", "_x", "_y"} if beta != 0 else {"_A", "_x"},
            outputs={"_y"})
        self.transA = transA
        self.alpha = alpha
        self.beta = beta

    def validate(self, sdfg, state):
        in_edges = state.in_edges(self)
        if len(in_edges) not in [2, 3]:
            raise ValueError("Expected 2 or 3 inputs to GEMV")
        size_y_in = None
        for _, _, _, dst_conn, memlet in state.in_edges(self):
            if dst_conn == "_A":
                subset = copy.deepcopy(memlet.subset)
                subset.squeeze()
                size_a = subset.size()
            if dst_conn == "_x":
                subset = copy.deepcopy(memlet.subset)
                subset.squeeze()
                size_x = subset.size()
            if dst_conn == "_y":
                subset = copy.deepcopy(memlet.subset)
                subset.squeeze()
                size_y_in = subset.size()

        if len(size_a) != 2 or len(size_x) != 1:
            raise ValueError(
                "Matrix-vector product only supported on matrix-vector input")

        a_cols = size_a[1] if not self.transA else size_a[0]
        a_rows = size_a[0] if not self.transA else size_a[1]

        if a_cols != size_x[0]:
            raise ValueError(f"Columns of A ({a_cols}) don't match "
                             f"size of x ({size_x[0]}).")

        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError(
                "Expected exactly one output from matrix-vector product")
        out_memlet = out_edges[0].data

        out_subset = copy.deepcopy(out_memlet.subset)
        out_subset.squeeze()
        size_y_out = out_subset.size()
        if size_y_in is not None and size_y_in != size_y_out:
            raise ValueError("Input y-vector must match output y-vector.")
        if (len(size_y_out) != 1 or size_y_out[0] != a_rows):
            raise ValueError("Vector input to GEMV must match matrix rows.")


# Numpy replacement
@oprepo.replaces('dace.libraries.blas.gemv')
@oprepo.replaces('dace.libraries.blas.Gemv')
def gemv_libnode(pv: 'ProgramVisitor',
                 sdfg: SDFG,
                 state: SDFGState,
                 A,
                 x,
                 y,
                 alpha,
                 beta,
                 trans=None):
    # Get properties
    if trans is None:
        trans = (sdfg.arrays[x].shape[0] == sdfg.arrays[A].shape[0])

    # Add nodes
    A_in, x_in = (state.add_read(name) for name in (A, x))
    y_out = state.add_write(y)

    libnode = Gemv('gemv', transA=trans, alpha=alpha, beta=beta)
    state.add_node(libnode)

    # Connect nodes
    state.add_edge(A_in, None, libnode, '_A', mm.Memlet(A))
    state.add_edge(x_in, None, libnode, '_x', mm.Memlet(x))
    state.add_edge(libnode, '_y', y_out, None, mm.Memlet(y))

    if beta != 0:
        y_in = state.add_read(y)
        state.add_edge(y_in, None, libnode, '_y', mm.Memlet(y))

    return []
