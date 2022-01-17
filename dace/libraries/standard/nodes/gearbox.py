# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import collections
import copy
import dace


@dace.library.expansion
class ExpandGearbox(dace.transformation.ExpandTransformation):

    environments = []

    @staticmethod
    def expansion(node: "Gearbox", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG):

        (in_edge, in_desc, out_edge, out_desc, is_pack, gear_factor) = node.validate(parent_sdfg, parent_state)

        is_elementwise = in_desc.dtype.base_type == out_desc.dtype.base_type

        sdfg = dace.SDFG(node.name)
        in_desc_inner = copy.deepcopy(in_desc)
        in_desc_inner.transient = False
        sdfg.add_datadesc(in_edge.dst_conn, in_desc_inner)
        out_desc_inner = copy.deepcopy(out_desc)
        out_desc_inner.transient = False
        sdfg.add_datadesc(out_edge.src_conn, out_desc_inner)

        state = sdfg.add_state(node.name)
        input_read = state.add_read(in_edge.dst_conn)
        output_write = state.add_write(out_edge.src_conn)
        vec_it = f"_{node.name}_w"
        entry, exit = state.add_map(node.name, {
            f"_{node.name}_i": f"0:{node.size}",
            vec_it: f"0:{gear_factor}"
        },
                                    schedule=node.schedule)
        buffer_name = f"{node.name}_buffer"

        if is_elementwise:

            dtype = in_desc.dtype.base_type

            if is_pack:
                small_veclen = in_desc.dtype.veclen
                large_veclen = out_desc.dtype.veclen
            else:
                large_veclen = in_desc.dtype.veclen
                small_veclen = out_desc.dtype.veclen

            sdfg.add_array(buffer_name, (large_veclen, ), dtype, storage=out_desc.storage, transient=True)

            nested_sdfg = dace.SDFG(f"{node.name}_nested")
            nested_in_desc = copy.deepcopy(in_desc)
            nested_in_desc.transient = False
            nested_out_desc = copy.deepcopy(out_desc)
            nested_out_desc.transient = False
            nested_sdfg.add_datadesc(f"_{in_edge.dst_conn}", nested_in_desc)
            nested_sdfg.add_datadesc(f"_{out_edge.src_conn}", nested_out_desc)
            read_state = nested_sdfg.add_state(f"{node.name}_read")
            write_state = nested_sdfg.add_state(f"{node.name}_write")

            read_nested = read_state.add_read(f"_{in_edge.dst_conn}")
            write_nested = write_state.add_write(f"_{out_edge.src_conn}")

            buffer_name_inner = f"{buffer_name}_inner"
            nested_sdfg.add_array(buffer_name_inner, (large_veclen, ), dtype, storage=out_desc.storage, transient=False)
            buffer_read = write_state.add_read(buffer_name_inner)
            buffer_write = read_state.add_write(buffer_name_inner)

            elem_it = f"_{node.name}_e"

            if is_pack:

                # Unpack the input vector into individual elements
                unpack_name = f"{node.name}_unpack"
                nested_sdfg.add_array(unpack_name, (small_veclen, ), dtype, storage=in_desc.storage, transient=True)
                unpack_access = read_state.add_write(unpack_name)
                read_state.add_memlet_path(read_nested, unpack_access, memlet=dace.Memlet(f"{read_nested.data}[0]"))

                # Now write the elements into the large buffer at the
                # appropriate indices
                unroll_entry, unroll_exit = read_state.add_map(f"{node.name}_elementwise",
                                                               {elem_it: f"0:{small_veclen}"},
                                                               schedule=node.schedule,
                                                               unroll=True)
                unroll_tasklet = read_state.add_tasklet(f"{node.name}_elementwise", {"unpack_in"}, {"buffer_out"},
                                                        "buffer_out = unpack_in")
                read_state.add_memlet_path(unpack_access,
                                           unroll_entry,
                                           unroll_tasklet,
                                           dst_conn="unpack_in",
                                           memlet=dace.Memlet(f"{unpack_name}[{elem_it}]"))
                read_state.add_memlet_path(unroll_tasklet,
                                           unroll_exit,
                                           buffer_write,
                                           src_conn="buffer_out",
                                           memlet=dace.Memlet(f"{buffer_name_inner}[{vec_it} * "
                                                              f"{small_veclen} + {elem_it}]"))

                # Only progress to the write state if we're on the last vector
                nested_sdfg.add_edge(read_state, write_state, dace.InterstateEdge(f"{vec_it} >= {gear_factor} - 1"))
                end_state = nested_sdfg.add_state(f"{node.name}_end")
                nested_sdfg.add_edge(read_state, end_state, dace.InterstateEdge(f"{vec_it} < {gear_factor} - 1"))
                nested_sdfg.add_edge(write_state, end_state, dace.InterstateEdge())

                # Write out
                write_state.add_memlet_path(buffer_read, write_nested, memlet=dace.Memlet(f"{write_nested.data}[0]"))

            else:  # Is unpack

                # Only read a new wide vector on the first iteration
                start_state = nested_sdfg.add_state(f"{node.name}_start")
                nested_sdfg.add_edge(start_state, read_state, dace.InterstateEdge(f"{vec_it} == 0"))
                nested_sdfg.add_edge(start_state, write_state, dace.InterstateEdge(f"{vec_it} != 0"))
                nested_sdfg.add_edge(read_state, write_state, dace.InterstateEdge())

                # Read new wide vector
                read_state.add_memlet_path(read_nested, buffer_write, memlet=dace.Memlet(f"{read_nested.data}[0]"))

                # Read out the appropriate elements and write them out
                pack_name = f"{node.name}_pack"
                nested_sdfg.add_array(pack_name, (small_veclen, ), dtype, storage=in_desc.storage, transient=True)
                pack_access = write_state.add_write(pack_name)
                unroll_entry, unroll_exit = write_state.add_map(f"{node.name}_elementwise",
                                                                {elem_it: f"0:{small_veclen}"},
                                                                schedule=node.schedule,
                                                                unroll=True)
                unroll_tasklet = write_state.add_tasklet(f"{node.name}_elementwise", {"buffer_in"}, {"pack_out"},
                                                         "pack_out = buffer_in")
                write_state.add_memlet_path(buffer_read,
                                            unroll_entry,
                                            unroll_tasklet,
                                            dst_conn="buffer_in",
                                            memlet=dace.Memlet(f"{buffer_name_inner}[{vec_it} * "
                                                               f"{small_veclen} + {elem_it}]"))
                write_state.add_memlet_path(unroll_tasklet,
                                            unroll_exit,
                                            pack_access,
                                            src_conn="pack_out",
                                            memlet=dace.Memlet(f"{pack_name}[{elem_it}]"))
                write_state.add_memlet_path(pack_access, write_nested, memlet=dace.Memlet(f"{write_nested.data}[0]"))

            nested_sdfg_node = state.add_nested_sdfg(nested_sdfg, sdfg,
                                                     {f"_{in_edge.dst_conn}", f"{buffer_name}_inner"},
                                                     {f"_{out_edge.src_conn}", f"{buffer_name}_inner"})
            buffer_read = state.add_read(buffer_name)
            buffer_write = state.add_write(buffer_name)
            state.add_memlet_path(input_read,
                                  entry,
                                  nested_sdfg_node,
                                  dst_conn=f"_{in_edge.dst_conn}",
                                  memlet=dace.Memlet(f"{input_read.data}[0]", dynamic=True))
            state.add_memlet_path(nested_sdfg_node,
                                  exit,
                                  output_write,
                                  src_conn=f"_{out_edge.src_conn}",
                                  memlet=dace.Memlet(f"{output_write.data}[0]", dynamic=True))
            state.add_memlet_path(buffer_read,
                                  entry,
                                  nested_sdfg_node,
                                  dst_conn=buffer_name_inner,
                                  memlet=dace.Memlet(f"{buffer_name}[0:{large_veclen}]"))
            state.add_memlet_path(nested_sdfg_node,
                                  exit,
                                  buffer_write,
                                  src_conn=buffer_name_inner,
                                  memlet=dace.Memlet(f"{buffer_name}[0:{large_veclen}]"))

        else:  # Not elementwise, one side is a vector of vectors

            vtype = out_desc.dtype if is_pack else in_desc.dtype

            sdfg.add_array(buffer_name, (1, ), vtype, storage=in_desc.storage, transient=True)
            buffer_read = state.add_read(buffer_name)
            buffer_write = state.add_write(buffer_name)

            tasklet = state.add_tasklet(
                node.name, {"val_in", "buffer_in"}, {"val_out", "buffer_out"}, f"""\
wide = buffer_in
wide[_{node.name}_w] = val_in
if _{node.name}_w == {gear_factor} - 1:
    val_out = wide
buffer_out = wide""" if is_pack else f"""\
wide = val_in if _{node.name}_w == 0 else buffer_in
val_out = wide[_{node.name}_w]
buffer_out = wide""")
            state.add_memlet_path(input_read,
                                  entry,
                                  tasklet,
                                  dst_conn="val_in",
                                  memlet=dace.Memlet(f"{in_edge.dst_conn}[0]", dynamic=not is_pack))
            state.add_memlet_path(buffer_read,
                                  entry,
                                  tasklet,
                                  dst_conn="buffer_in",
                                  memlet=dace.Memlet(f"{buffer_name}[0]"))
            state.add_memlet_path(tasklet,
                                  exit,
                                  output_write,
                                  src_conn="val_out",
                                  memlet=dace.Memlet(f"{out_edge.src_conn}[0]", dynamic=is_pack))
            state.add_memlet_path(tasklet,
                                  exit,
                                  buffer_write,
                                  src_conn="buffer_out",
                                  memlet=dace.Memlet(f"{buffer_name}[0]"))

        return sdfg


@dace.library.node
class Gearbox(dace.sdfg.nodes.LibraryNode):
    """
    Provides a library node that converts from a stream of type
    vector(vector(dtype, w0)) to a stream of type vector(dtype, w1), or vice
    versa. This is useful for achieving efficient memory reads on Xilinx FPGAs,
    where modules accessing memories should always read or write 512-bit
    vectors, which then potentially need to be narrowed down to the vector width
    of the computational kernel.

    The node expects to have a single input and a single output, where one end
    is of type vector(vector(dtype, w0)), and the other is of type
    vector(dtype, w1).
    """

    implementations = {
        "pure": ExpandGearbox,
    }
    default_implementation = "pure"

    # Properties
    size = dace.properties.SymbolicProperty(desc="Number of wide vectors to convert to/from narrow vectors.", default=0)

    def __init__(self, size, name=None, schedule=None, **kwargs):
        """
        :param size: Number of wide vectors to convert to/from narrow vectors.
                     For example, if converting n/16 reads (vector size 16) from
                     memory into n/4 elements (vector size 4), this parameter
                     should be set to n/16.
        """
        super().__init__(name=name or "gearbox", schedule=schedule or dace.ScheduleType.FPGA_Device, **kwargs)
        self.size = size
        if schedule is not None:
            self.schedule = schedule

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState):
        try:
            size = dace.symbolic.evaluate(self.size, sdfg.constants)
            if size < 1:
                raise ValueError(f"Invalid size parameter for {self}: {size}")
        except TypeError:
            pass  # Not a constant
        in_edge = state.in_edges(self)
        if len(in_edge) != 1:
            raise ValueError(f"Expected only one input edge, found {len(in_edge)} edges.")
        out_edge = state.out_edges(self)
        if len(out_edge) != 1:
            raise ValueError(
                f"Expected only one output edge, found {len(out_edge)} edges.")
        in_edge = in_edge[0]
        in_desc = sdfg.arrays[in_edge.data.data]
        if not isinstance(in_desc, dace.data.Stream):
            raise TypeError(f"Expected input to be a stream, got {type(in_desc)}.")
        out_edge = out_edge[0]
        out_desc = sdfg.arrays[out_edge.data.data]
        if not isinstance(out_desc, dace.data.Stream):
            raise TypeError(
                f"Expected output to be a stream, got {type(out_desc)}.")
        # The type of one side must be a vector of the other, or a vector of the
        # same type with a vector size that is a multiple of the other
        if (isinstance(in_desc.dtype, dace.vector) and in_desc.dtype.base_type == out_desc.dtype):
            is_pack = False  # Is unpack
            gear_factor = in_desc.dtype.veclen
        elif (isinstance(out_desc.dtype, dace.vector) and out_desc.dtype.base_type == in_desc.dtype):
            is_pack = True
            gear_factor = out_desc.dtype.veclen
        elif (isinstance(in_desc.dtype, dace.vector) and isinstance(out_desc.dtype, dace.vector)
              and in_desc.veclen // out_desc.veclen > 1 and in_desc.veclen % out_desc.veclen == 0):
            is_pack = False  # Is unpack
            gear_factor = in_desc.veclen // out_desc.veclen
        elif (isinstance(in_desc.dtype, dace.vector) and isinstance(out_desc.dtype, dace.vector)
              and out_desc.veclen // in_desc.veclen > 1 and out_desc.veclen % in_desc.veclen == 0):
            is_pack = True
            gear_factor = out_desc.veclen // in_desc.veclen
        else:
            raise TypeError(f"Cannot gearbox between {in_desc.dtype} for {in_edge.dst_conn}"
                            f" and {out_desc.dtype} for {out_edge.src_conn}.")
        return (in_edge, in_desc, out_edge, out_desc, is_pack, gear_factor)
