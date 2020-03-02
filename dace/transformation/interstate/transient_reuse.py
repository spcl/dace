from dace import data, memlet, dtypes, registry, sdfg as sd
from dace.graph import nodes, nxutil, edges as ed
from dace.transformation import pattern_matching
from dace.properties import Property, make_properties, SubsetProperty
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState

@registry.autoregister
@make_properties
class TransientReuse(pattern_matching.Transformation):

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        return True

    @staticmethod
    def match_to_str(graph, candidate):
        return graph.label

    @staticmethod
    def expressions():
        return [sd.SDFG('_')]

    def expansion(node):
        pass

    def apply(self, sdfg):
        state: SDFGState = sdfg.nodes()[self.state_id]

        # Step 1: Find all transients.
        transients = []
        size = []
        for a in sdfg.arrays:
            if sdfg.arrays[a].transient == True:
                transients.append(a)
                size.append(sdfg.arrays[a].total_size)

        # Step 2: Determine all possible reuses and decide on a mapping.
        mapping = []
        mapping.append((transients[2],transients[0]))

        # Step 3: For each mapping redirect edges and rename memlets in the whole tree.
        for (old, new) in mapping:
            old_node = state.find_node(old)
            old_node.data = new
            for e in state.all_edges(old_node):
                for edge in state.memlet_tree(e):
                    if edge.data.data == old:
                        edge.data.data = new

            sdfg.remove_data(old)
