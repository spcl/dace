# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
Provides a library node that converts from a stream of type
vector(vector(dtype, w0)) to a stream of type vector(dtype, w1), or vice versa.
This is useful for achieving efficient memory reads on Xilinx FPGAs, where
modules accessing memories should always read or write512-bit vectors, which
then potentially need to be narrowed down to the vector width of the
computational kernel.

The node expects to have a single input and a single output, where one end is
of type vector(vector(dtype, w0)), and the other is of type vector(dtype, w1).
"""
import collections
import copy
import dace


@dace.library.expansion
class ExpandGearbox(dace.transformation.ExpandTransformation):

    environments = []

    @staticmethod
    def expansion(node: "Gearbox", parent_state: dace.SDFGState,
                  parent_sdfg: dace.SDFG):

        (in_edge, in_desc, out_edge, out_desc, is_pack,
         gear_factor) = node.validate(parent_sdfg, parent_state)

        if is_pack:
            vtype = out_desc.dtype
            etype = in_desc.dtype
        else:
            etype = out_desc.dtype
            vtype = in_desc.dtype

        sdfg = dace.SDFG("gearbox")
        in_desc_inner = copy.copy(in_desc)
        in_desc_inner.transient = False
        sdfg.add_datadesc(in_edge.dst_conn, in_desc_inner)
        out_desc_inner = copy.copy(out_desc)
        out_desc_inner.transient = False
        sdfg.add_datadesc(out_edge.src_conn, out_desc_inner)
        sdfg.add_array("gearbox_buffer", (1, ),
                       vtype,
                       storage=in_desc.storage,
                       transient=True)

        state = sdfg.add_state("gearbox")
        buffer_read = state.add_read("gearbox_buffer")
        buffer_write = state.add_write("gearbox_buffer")
        input_read = state.add_read(in_edge.dst_conn)
        output_write = state.add_write(out_edge.src_conn)
        iteration_space = collections.OrderedDict()
        for i, r in enumerate(node.ranges):
            if i == len(node.ranges) - 1:
                r = (r[0], r[1] / gear_factor, r[2])
            iteration_space[f"_i{i}_gearbox"] = r
        iteration_space["_w_gearbox"] = (0, gear_factor - 1, 1)
        entry, exit = state.add_map("gearbox",
                                    iteration_space,
                                    schedule=node.schedule)
        tasklet = state.add_tasklet(
            "gearbox", {
                "val_in",
                "buffer_in"
            }, {
                "val_out",
                "buffer_out"
            }, f"""\
wide = buffer_in
wide[_w_gearbox] = val_in
if _w_gearbox == {gear_factor} - 1:
    val_out = wide
buffer_out = wide""" if is_pack else """\
wide = val_in if _w_gearbox == 0 else buffer_in
val_out = wide[_w_gearbox]
buffer_out = wide""")
        state.add_memlet_path(input_read,
                              entry,
                              tasklet,
                              dst_conn="val_in",
                              memlet=dace.Memlet(f"{in_edge.dst_conn}[0]",
                                                 dynamic=not is_pack))
        state.add_memlet_path(buffer_read,
                              entry,
                              tasklet,
                              dst_conn="buffer_in",
                              memlet=dace.Memlet(f"gearbox_buffer[0]"))
        state.add_memlet_path(tasklet,
                              exit,
                              output_write,
                              src_conn="val_out",
                              memlet=dace.Memlet(f"{out_edge.src_conn}[0]",
                                                 dynamic=is_pack))
        state.add_memlet_path(tasklet,
                              exit,
                              buffer_write,
                              src_conn="buffer_out",
                              memlet=dace.Memlet(f"gearbox_buffer[0]"))

        return sdfg


@dace.library.node
class Gearbox(dace.sdfg.nodes.LibraryNode):

    implementations = {
        "pure": ExpandGearbox,
    }
    default_implementation = "pure"

    # Properties
    ranges = dace.properties.RangeProperty(
        desc="One or more ranges defining the iteration space.",
        default=dace.subsets.Range([]))

    def __init__(self, ranges, name=None, schedule=None, **kwargs):
        super().__init__(name=name or "gearbox",
                         schedule=schedule or dace.ScheduleType.FPGA_Device,
                         **kwargs)
        self.ranges = ranges
        if schedule is not None:
            self.schedule = schedule

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState):
        in_edge = state.in_edges(self)
        if len(in_edge) != 1:
            raise ValueError(
                f"Expected only one input edge, found {len(in_edge)} edges.")
        out_edge = state.out_edges(self)
        if len(out_edge) != 1:
            raise ValueError(
                f"Expected only one input edge, found {len(out_edge)} edges.")
        in_edge = in_edge[0]
        in_desc = sdfg.arrays[in_edge.data.data]
        if not isinstance(in_desc, dace.data.Stream):
            raise TypeError(
                f"Expected input to be a stream, got {type(in_desc)}.")
        out_edge = out_edge[0]
        out_desc = sdfg.arrays[out_edge.data.data]
        if not isinstance(out_desc, dace.data.Stream):
            raise TypeError(
                f"Expected input to be a stream, got {type(out_desc)}.")
        # The type of one side must be a vector of the other
        if (isinstance(in_desc.dtype, dace.vector)
                and in_desc.dtype.base_type == out_desc.dtype):
            is_pack = False  # Is unpack
            gear_factor = in_desc.dtype.veclen
        elif (isinstance(out_desc.dtype, dace.vector)
              and out_desc.dtype.base_type == in_desc.dtype):
            is_pack = True
            gear_factor = out_desc.dtype.veclen
        else:
            raise TypeError(
                f"Cannot gearbox between {in_desc.dtype} and {out_desc.dtype}.")
        return (in_edge, in_desc, out_edge, out_desc, is_pack, gear_factor)
