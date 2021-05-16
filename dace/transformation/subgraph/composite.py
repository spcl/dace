""" This module contains classes that implement a composite
    Subgraph Fusion - Stencil Tiling Transformation
"""


import dace

import dace.transformation.transformation as transformation
from dace.transformation.subgraph import SubgraphFusion, StencilTiling, MultiExpansion
import dace.transformation.subgraph.helpers as helpers

from dace import dtypes, registry, symbolic, subsets, data
from dace.properties import make_properties, Property, ShapeProperty
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.graph import SubgraphView

import warnings

@registry.autoregister_params(singlestate=True)
@make_properties
class CompositeFusion(transformation.SubgraphTransformation):
    """ MultiExpansion + SubgraphFusion in one Transformation
        Additional StencilTiling is also possible as a canonicalizing
        transformation before fusion.
    """

    debug = Property(desc="Debug mode", dtype=bool, default = False)

    allow_expansion = Property(desc="Allow MultiExpansion first",
                               dtype = bool,
                               default = True)

    allow_tiling = Property(desc="Allow StencilTiling (after MultiExpansion)",
                            dtype = bool,
                            default = False)

    transient_allocation = Property(
        desc="Storage Location to push transients to that are "
              "fully contained within the subgraph.",
        dtype=dtypes.StorageType,
        default=dtypes.StorageType.Default)

    schedule_innermaps = Property(desc="Schedule of inner fused maps",
                                  dtype=dtypes.ScheduleType,
                                  default=dtypes.ScheduleType.Default)

    stencil_unroll_loops = Property(desc="Unroll inner stencil loops if they have size > 1",
                                    dtype=bool,
                                    default=False)
    stencil_strides = ShapeProperty(dtype=tuple,
                                    default=(1, ),
                                    desc="Stencil tile stride")

    expansion_split = Property(desc="Allow MultiExpansion to split up maps, if enabled",
                               dtype = bool,
                               default = True)

    def can_be_applied(self, sdfg: SDFG, subgraph: SubgraphView) -> bool:
        graph = subgraph.graph
        if self.allow_expansion == True:
            
            if SubgraphFusion.can_be_applied(sdfg, subgraph):
                # try w/o copy first 
                return True 
            
            expansion = MultiExpansion(subgraph) 
            expansion.permutation_only = not self.expansion_split
            if MultiExpansion.can_be_applied(sdfg, subgraph):
                # deepcopy
                graph_indices = [i for (i,n) in enumerate(graph.nodes()) if n in subgraph]
                sdfg_copy = SDFG.from_json(sdfg.to_json())
                graph_copy = sdfg_copy.nodes()[sdfg.nodes().index(graph)]
                subgraph_copy = SubgraphView(graph_copy,
                [graph_copy.nodes()[i] for i in graph_indices])
            
                
                ##sdfg_copy.apply_transformations(MultiExpansion, states=[graph])
                #expansion = MultiExpansion(subgraph_copy)
                expansion.apply(sdfg_copy)

                if SubgraphFusion.can_be_applied(sdfg_copy, subgraph_copy):
                    return True 
               
                if self.allow_tiling and StencilTiling.can_be_applied(sdfg, subgraph):
                    return True 

        else:
            if SubgraphFusion.can_be_applied(sdfg, subgraph):
                return True
        
        if self.allow_tiling == True:
            if StencilTiling.can_be_applied(sdfg, subgraph):
                return True
                
        return False

    def apply(self, sdfg):
        subgraph = self.subgraph_view(sdfg)
        graph = subgraph.graph
        scope_dict = graph.scope_dict()
        map_entries = helpers.get_outermost_scope_maps(sdfg, graph, subgraph, scope_dict)
        first_entry = next(iter(map_entries))


        expansion = MultiExpansion(subgraph, self.sdfg_id, self.state_id) 
        expansion.permutation_only = not self.expansion_split
        if self.allow_expansion == True and expansion.can_be_applied(sdfg, subgraph):
            # expand first 
            #me = MultiExpansion(subgraph, self.sdfg_id, self.state_id)
            #me.debug = self.debug
            me.apply(sdfg)
        if SubgraphFusion.can_be_applied(sdfg, self.subgraph_view(sdfg)):
            sf = SubgraphFusion(subgraph, self.sdfg_id, self.state_id)
            # set SubgraphFusion properties
            sf.debug = self.debug
            sf.transient_allocation = self.transient_allocation
            sf.schedule_innermaps = self.schedule_innermaps
            sf.apply(sdfg)

        elif self.allow_tiling == True and StencilTiling.can_be_applied(sdfg, self.subgraph_view(sdfg)):
            st = StencilTiling(self.subgraph_view(sdfg), self.sdfg_id, self.state_id)
            # set StencilTiling properties
            st.debug = self.debug
            st.unroll_loops = self.stencil_unroll_loops
            st.strides= self.stencil_strides
            st.apply(sdfg)
            # StencilTiling: update nodes
            new_entries = st._outer_entries
            subgraph = helpers.subgraph_from_maps(sdfg, graph, new_entries)
            sf = SubgraphFusion(subgraph, self.sdfg_id, self.state_id)
            # set SubgraphFusion properties
            sf.debug = self.debug
            sf.transient_allocation = self.transient_allocation
            sf.schedule_innermaps = self.schedule_innermaps

            sf.apply(sdfg)

        else:
            warnings.warn("CompositeFusion::Apply did not perform as expected")
        self._global_map_entry = sf._global_map_entry

