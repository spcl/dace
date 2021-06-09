# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes that implement a composite
    Subgraph Fusion - Stencil Tiling Transformation
"""


import dace
from dace.transformation.subgraph import stencil_tiling

import dace.transformation.transformation as transformation
from dace.transformation.subgraph import SubgraphFusion, MultiExpansion
from dace.transformation.subgraph.stencil_tiling import StencilTiling
import dace.transformation.subgraph.helpers as helpers

from dace import dtypes, registry, symbolic, subsets, data
from dace.properties import EnumProperty, make_properties, Property, ShapeProperty
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

    transient_allocation = EnumProperty(
        desc="Storage Location to push transients to that are "
             "fully contained within the subgraph.",
        dtype=dtypes.StorageType,
        default=dtypes.StorageType.Default)

    schedule_innermaps = Property(desc="Schedule of inner fused maps",
                                  dtype=dtypes.ScheduleType,
                                  default=None,  
                                  allow_none=True)

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
            subgraph_fusion = SubgraphFusion(subgraph)
            if subgraph_fusion.can_be_applied(sdfg, subgraph):
                # try w/o copy first 
                return True 
            
            expansion = MultiExpansion(subgraph) 
            expansion.permutation_only = not self.expansion_split
            if expansion.can_be_applied(sdfg, subgraph):
                # deepcopy
                graph_indices = [i for (i,n) in enumerate(graph.nodes()) if n in subgraph]
                sdfg_copy = SDFG.from_json(sdfg.to_json())
                graph_copy = sdfg_copy.nodes()[sdfg.nodes().index(graph)]
                subgraph_copy = SubgraphView(graph_copy,
                [graph_copy.nodes()[i] for i in graph_indices])
            
                
                ##sdfg_copy.apply_transformations(MultiExpansion, states=[graph])
                #expansion = MultiExpansion(subgraph_copy)
                expansion.apply(sdfg_copy)

                subgraph_fusion = SubgraphFusion(subgraph_copy)
                if subgraph_fusion.can_be_applied(sdfg_copy, subgraph_copy):
                    return True 
               
                stencil_tiling = StencilTiling(subgraph_copy)
                if self.allow_tiling and stencil_tiling.can_be_applied(sdfg_copy, subgraph_copy):
                    return True 

        else:
            subgraph_fusion = SubgraphFusion(subgraph)
            if subgraph_fusion.can_be_applied(sdfg, subgraph):
                return True
        
        if self.allow_tiling == True:
            stencil_tiling = StencilTiling(subgraph)
            if stencil_tiling.can_be_applied(sdfg, subgraph):
                return True
                
        return False

    def apply(self, sdfg):
        subgraph = self.subgraph_view(sdfg)
        graph = subgraph.graph
        scope_dict = graph.scope_dict()
        map_entries = helpers.get_outermost_scope_maps(sdfg, graph, subgraph, scope_dict)
        first_entry = next(iter(map_entries))

        if self.allow_expansion:
            expansion = MultiExpansion(subgraph, self.sdfg_id, self.state_id) 
            expansion.permutation_only = not self.expansion_split
            if expansion.can_be_applied(sdfg, subgraph):
                expansion.apply(sdfg)

        sf = SubgraphFusion(subgraph, self.sdfg_id, self.state_id)
        if sf.can_be_applied(sdfg, self.subgraph_view(sdfg)):
            # set SubgraphFusion properties
            sf.debug = self.debug
            sf.transient_allocation = self.transient_allocation
            sf.schedule_innermaps = self.schedule_innermaps
            sf.apply(sdfg)
            self._global_map_entry = sf._global_map_entry
            return 
        
        elif self.allow_tiling == True:
            st = StencilTiling(subgraph, self.sdfg_id, self.state_id)
            if st.can_be_applied(sdfg, self.subgraph_view(sdfg)):
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
                self._global_map_entry = sf._global_map_entry
                return 


        warnings.warn("CompositeFusion::Apply did not perform as expected")

