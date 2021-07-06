# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace.sdfg import utils
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.blas import environments
from dace import (config, data as dt, dtypes, memlet as mm, SDFG, SDFGState,
                  symbolic)
from dace.frontend.common import op_repository as oprepo
from dace.transformation.dataflow import hbm_transform

@dace.library.expansion
class ExpandAxpyVectorized(ExpandTransformation):
    """
    Generic expansion of AXPY with support for vectorized data types.
    """

    environments = []

    @staticmethod
    def expansion(node,
                  parent_state: SDFGState,
                  parent_sdfg,
                  schedule=dace.ScheduleType.Default):
        """
        :param node: Node to expand.
        :param parent_state: State that the node is in.
        :param parent_sdfg: SDFG that the node is in.
        :param schedule: The schedule to set on maps in the expansion. For FPGA
                         expansion, this should be set to FPGA_Device.
        """
        node.validate(parent_sdfg, parent_state)

        x_outer = parent_sdfg.arrays[next(
            parent_state.in_edges_by_connector(node, "_x")).data.data]
        y_outer = parent_sdfg.arrays[next(
            parent_state.in_edges_by_connector(node, "_y")).data.data]
        res_outer = parent_sdfg.arrays[next(
            parent_state.out_edges_by_connector(node, "_res")).data.data]

        a = node.a
        n = node.n / x_outer.dtype.veclen

        axpy_sdfg = dace.SDFG("axpy")
        axpy_state = axpy_sdfg.add_state("axpy")

        x_inner = x_outer.clone()
        x_inner.transient = False
        y_inner = y_outer.clone()
        y_inner.transient = False
        res_inner = res_outer.clone()
        res_inner.transient = False

        axpy_sdfg.add_datadesc("_x", x_inner)
        axpy_sdfg.add_datadesc("_y", y_inner)
        axpy_sdfg.add_datadesc("_res", res_inner)

        x_in = axpy_state.add_read("_x")
        y_in = axpy_state.add_read("_y")
        z_out = axpy_state.add_write("_res")

        vec_map_entry, vec_map_exit = axpy_state.add_map(
            "axpy", {"i": f"0:{n}"}, schedule=schedule)

        axpy_tasklet = axpy_state.add_tasklet(
            "axpy", ["x_conn", "y_conn"], ["z_conn"],
            f"z_conn = {a} * x_conn + y_conn")

        # Access container either as an array or as a stream
        index = "0" if isinstance(x_inner, dt.Stream) else "i"
        axpy_state.add_memlet_path(x_in,
                                   vec_map_entry,
                                   axpy_tasklet,
                                   dst_conn="x_conn",
                                   memlet=dace.Memlet(f"_x[{index}]"))

        index = "0" if isinstance(y_inner, dt.Stream) else "i"
        axpy_state.add_memlet_path(y_in,
                                   vec_map_entry,
                                   axpy_tasklet,
                                   dst_conn="y_conn",
                                   memlet=dace.Memlet(f"_y[{index}]"))

        index = "0" if isinstance(res_inner, dt.Stream) else "i"
        axpy_state.add_memlet_path(axpy_tasklet,
                                   vec_map_exit,
                                   z_out,
                                   src_conn="z_conn",
                                   memlet=dace.Memlet(f"_res[{index}]"))

        return axpy_sdfg


@dace.library.expansion
class ExpandAxpyFpga(ExpandTransformation):
    """
    FPGA expansion which uses the generic implementation, but sets the map
    schedule to be executed on FPGA.
    """

    environments = []

    @staticmethod
    def expansion(node, parent_state: SDFGState, parent_sdfg: SDFG, **kwargs):
        """
        :param node: Node to expand.
        :param parent_state: State that the node is in.
        :param parent_sdfg: SDFG that the node is in.
        """
        return ExpandAxpyVectorized.expansion(
            node,
            parent_state,
            parent_sdfg,
            schedule=dace.ScheduleType.FPGA_Device,
            **kwargs)

@dace.library.expansion
class ExpandAxpyFpgaHbm(ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node, parent_state: SDFGState, parent_sdfg: SDFG, param="k", **kwargs):
        sdfg: SDFG = ExpandAxpyFpga.expansion(node, parent_state, parent_sdfg, **kwargs)
        state: SDFGState = sdfg.states()[0]

        desc_x = parent_sdfg.arrays[parent_state.in_edges(node)[0].data.data]
        desc_y = parent_sdfg.arrays[parent_state.in_edges(node)[1].data.data]
        desc_res = parent_sdfg.arrays[parent_state.out_edges(node)[0].data.data]
        banks_x = desc_x.location["bank"]
        banks_y = desc_y.location["bank"]
        banks_res = desc_res.location["bank"]
        tmp_bank_c = utils.get_multibank_ranges_from_subset(
            utils.parse_location_bank(desc_x)[1], sdfg)
        bank_count = tmp_bank_c[1] - tmp_bank_c[0] # We know this is equal for all arrays it's checked in validation

        xform = hbm_transform.HbmTransform(sdfg.sdfg_id, -1, {}, -1)
        xform.outer_map_range = {param : f"0:{bank_count}"}
        xform.update_hbm_access_list.extend([(state, node, "k") for node in state.source_nodes()])
        xform.update_hbm_access_list.append((state, state.sink_nodes()[0], "k"))
        xform.update_array_list.extend([("_x", banks_x), ("_y", banks_y), ("_res", banks_res)])
        xform.apply(sdfg)

        return sdfg
        

@dace.library.node
class Axpy(dace.sdfg.nodes.LibraryNode):
    """
    Implements the BLAS AXPY operation, which computes a*x + y, where the
    vectors x and y are of size n. Expects input connectrs "_x" and "_y", and
    output connector "_res".
    """

    # Global properties
    implementations = {
        "pure": ExpandAxpyVectorized,
        "fpga": ExpandAxpyFpga,
        "fpga_hbm": ExpandAxpyFpgaHbm,
    }
    default_implementation = None

    # Object fields
    a = dace.properties.SymbolicProperty(allow_none=False,
                                         default=dace.symbolic.symbol("a"))
    n = dace.properties.SymbolicProperty(allow_none=False,
                                         default=dace.symbolic.symbol("n"))

    def __init__(self, name, a=None, n=None, *args, **kwargs):
        super().__init__(name,
                         *args,
                         inputs={"_x", "_y"},
                         outputs={"_res"},
                         **kwargs)
        self.a = a or dace.symbolic.symbol("a")
        self.n = n or dace.symbolic.symbol("n")

    def compare(self, other):

        if (self.veclen == other.veclen
                and self.implementation == other.implementation):

            return True
        else:
            return False

    def validate(self, sdfg: SDFG, state: SDFGState):

        in_edges = state.in_edges(self)
        if len(in_edges) != 2:
            raise ValueError("Expected exactly two inputs to axpy")

        in_memlets = [in_edges[0].data, in_edges[1].data]
        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one output from axpy")

        out_memlet = out_edges[0].data
        size = in_memlets[0].subset.size()

        if len(size) != 1 and self.implementation != "fpga_hbm":
            raise ValueError("axpy only supported on 1-dimensional arrays")
        elif len(size) != 2 and self.implementation == "fpga_hbm":
            raise ValueError("axpy for hbm only supports 2-dimensional arrays")

        if self.implementation == "fpga_hbm":
            desc_x = sdfg.arrays[in_memlets[0].data]
            desc_y = sdfg.arrays[in_memlets[1].data]
            desc_z = sdfg.arrays[out_memlet.data]
            parse_x = utils.parse_location_bank(desc_x)
            parse_y = utils.parse_location_bank(desc_y)
            parse_z = utils.parse_location_bank(desc_z)
            if(parse_x[0] != "HBM" or parse_y[0] != "HBM" or parse_z[0] != "HBM"):
                raise ValueError("All attached arrays must be on HBM for this "
                    "axpy implementation")
            low1, high1 = utils.get_multibank_ranges_from_subset(parse_x[1], sdfg)
            low2, high2 = utils.get_multibank_ranges_from_subset(parse_y[1], sdfg)
            low3, high3 = utils.get_multibank_ranges_from_subset(parse_z[1], sdfg)
            if (high1 - low1) != (high2 - low2) or (high2 - low2) != (high3 - low3):
                raise ValueError("All attached arrays should be distributed among "
                    "the same number of banks")

        if size != in_memlets[1].subset.size():
            raise ValueError("Inputs to axpy must have equal size")

        if size != out_memlet.subset.size():
            raise ValueError("Output of axpy must have same size as input")

        if (in_memlets[0].wcr is not None or in_memlets[1].wcr is not None
                or out_memlet.wcr is not None):
            raise ValueError("WCR on axpy memlets not supported")

        return True


# Numpy replacement
@oprepo.replaces('dace.libraries.blas.axpy')
@oprepo.replaces('dace.libraries.blas.Axpy')
def axpy_libnode(sdfg: SDFG, state: SDFGState, a, x, y, result):
    # Add nodes
    x_in, y_in = (state.add_read(name) for name in (x, y))
    res = state.add_write(result)

    libnode = Axpy('axpy', a=a)
    state.add_node(libnode)

    # Connect nodes
    state.add_edge(x_in, None, libnode, '_x', mm.Memlet(x))
    state.add_edge(y_in, None, libnode, '_y', mm.Memlet(y))
    state.add_edge(libnode, '_res', res, None, mm.Memlet(result))

    return []
