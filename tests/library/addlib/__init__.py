# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import dace
from dace import SDFGState, SDFG, library, dtypes
from dace.transformation.transformation import ExpandTransformation

library.register_library(__name__, "AddLib")


@dace.library.expansion
class ExpandAdd(ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node, parent_state: SDFGState, parent_sdfg: SDFG):
        in_edge = parent_state.in_edges(node)[0]
        out_edge = parent_state.out_edges(node)[0]

        sdfg = dace.SDFG("nested")
        sdfg.add_datadesc("_a", copy.deepcopy(parent_sdfg.arrays[in_edge.data.data]))
        sdfg.add_datadesc("_b", copy.deepcopy(parent_sdfg.arrays[out_edge.data.data]))
        sdfg.arrays["_a"].transient = False
        sdfg.arrays["_b"].transient = False
        state = sdfg.add_state()

        inp = state.add_access("_a")
        outp = state.add_access("_b")

        me, mx = state.add_map("useless_map", {"i": "0"})

        tasklet = state.add_tasklet("add", {"inp"}, {"outp"}, "outp = inp + 1")

        state.add_edge(inp, None, me, None, sdfg.make_array_memlet("_a"))
        state.add_edge(me, None, tasklet, "inp", sdfg.make_array_memlet("_a"))

        state.add_edge(tasklet, "outp", mx, None, dace.Memlet("_b[0]"))
        state.add_edge(mx, None, outp, None, dace.Memlet("_b[0]"))
        sdfg.fill_scope_connectors()

        return sdfg


@dace.library.node
class AddNode(dace.sdfg.nodes.LibraryNode):

    _dace_library_name = "AddLib"
    # Global properties
    implementations = {
        "pure": ExpandAdd,
    }
    default_implementation = 'pure'

    def __init__(self, name):
        super().__init__(name, inputs={'_a'}, outputs={'_b'})
