import dace
from dace import data, registry
from dace.sdfg import nodes, utils as sdutil
from dace.transformation import transformation as xf
import copy


@registry.autoregister_params(singlestate=True)#, strict=True)
class SqueezeViewRemove(xf.Transformation):
    in_array = xf.PatternNode(nodes.AccessNode)
    out_array = xf.PatternNode(nodes.AccessNode)

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(SqueezeViewRemove.in_array,
                                   SqueezeViewRemove.out_array)
        ]

    def can_be_applied(self,
                       state: dace.SDFGState,
                       candidate,
                       expr_index: int,
                       sdfg: dace.SDFG,
                       strict: bool = False):
        in_array = self.in_array(sdfg)
        out_array = self.out_array(sdfg)

        in_desc = in_array.desc(sdfg)
        out_desc = out_array.desc(sdfg)

        if state.out_degree(out_array) != 1:
            return False

        if not isinstance(out_desc, data.View):
            return False

        vedge = state.out_edges(out_array)[0]
        if vedge.data.data != out_array.data:  # Ensures subset comes from view
            return False
        view_subset = copy.deepcopy(vedge.data.subset)

        aedge = state.edges_between(in_array, out_array)[0]
        if aedge.data.data != in_array.data:
            return False
        array_subset = copy.deepcopy(aedge.data.subset)

        #asqdims = array_subset.squeeze()
        vsqdims = view_subset.squeeze()

        # Check that subsets are equivalent
        if array_subset != view_subset:
            return False

        # Verify strides after squeeze
        astrides = tuple(
            in_desc.strides
        )  #s for i, s in enumerate(in_desc.strides) if i not in asqdims)
        vstrides = tuple(s for i, s in enumerate(out_desc.strides)
                         if i in vsqdims)
        if astrides != vstrides:
            return False

        return True

    def apply(self, sdfg: dace.SDFG):
        state: dace.SDFGState = sdfg.nodes()[self.state_id]
        in_array = self.in_array(sdfg)
        out_array = self.out_array(sdfg)
        out_desc = out_array.desc(sdfg)

        vedge = state.out_edges(out_array)[0]
        view_subset = copy.deepcopy(vedge.data.subset)

        aedge = state.edges_between(in_array, out_array)[0]
        array_subset = copy.deepcopy(aedge.data.subset)

        vsqdims = view_subset.squeeze()

        # Modify data and subset on all outgoing edges
        for e in state.memlet_tree(vedge):
            e.data.data = in_array.data
            e.data.subset.squeeze(vsqdims)

        # Redirect original edge to point to data
        state.remove_edge(vedge)
        state.add_edge(in_array, vedge.src_conn, vedge.dst, vedge.dst_conn,
                       vedge.data)

        # Remove node and descriptor
        state.remove_node(out_array)
        sdfg.remove_data(out_array.data)
