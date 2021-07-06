from typing import Any, Dict, Iterable, List, Tuple, Union

import networkx
from networkx.algorithms.centrality import current_flow_betweenness
from dace import dtypes, properties, registry, subsets, symbolic
from dace.sdfg import utils, graph
from dace.transformation import transformation, interstate
from dace.sdfg import nodes as nd
from dace import SDFG, SDFGState, memlet

@registry.autoregister_params(singlestate=True)
@properties.make_properties
class HbmCopyTransform(transformation.Transformation):

    _src_node = nd.AccessNode("")
    _dst_node = nd.AccessNode("")

    #dtype=List[int]
    split_array_info = properties.Property(
        dtype=List,
        default=None,
        allow_none=True,
        desc="Describes how many times this array is split in each dimension. "
        "A value of 1 means that the array is not split in a dimension at all. "
        "If None, then the transform will try to split equally in each dimension."
    )
    
    def _get_split_size(self, virtualshape : Iterable, splitcount : List[int]) -> List[int]:
        """
        :returns: the shape of a part-array on one HBMbank
        """
        newshapelist = []
        for d in range(len(virtualshape)):
            if splitcount[d] != 1:
                newshapelist.append(virtualshape[d] // splitcount[d])
            else:
                newshapelist.append(virtualshape[d])
        return newshapelist

    @staticmethod
    def can_be_applied(graph: Union[SDFG, SDFGState], candidate: Dict['PatternNode', int],
        expr_index: int, sdfg: SDFG, strict: bool) -> bool:
        src = graph.nodes()[candidate[HbmCopyTransform._src_node]]
        dst = graph.nodes()[candidate[HbmCopyTransform._dst_node]]
        src_array = sdfg.arrays[src.data]
        dst_array = sdfg.arrays[dst.data]

        #same dimensions means HBM-array needs 1 dimension more
        collect_src = len(src_array.shape) - 1 == len(dst_array.shape)
        distribute_dst = len(src_array.shape) + 1 == len(dst_array.shape)
        if collect_src:
            try:
                tmp = int(src_array.shape[0])
            except:
                return False
        elif distribute_dst:
            try:
                tmp = int(dst_array.shape[0])
            except:
                return False
        return collect_src or distribute_dst

    @staticmethod
    def expressions():
        return [utils.node_path_graph(HbmCopyTransform._src_node, 
            HbmCopyTransform._dst_node)]

    def apply(self, sdfg: SDFG) -> Union[Any, None]:
        #Load/parse infos from the SDFG
        graph = sdfg.nodes()[self.state_id]
        src = graph.nodes()[self.subgraph[HbmCopyTransform._src_node]]
        dst = graph.nodes()[self.subgraph[HbmCopyTransform._dst_node]]
        src_array = sdfg.arrays[src.data]
        dst_array = sdfg.arrays[dst.data]
        collect_src = len(src_array.shape) - 1 == len(dst_array.shape) #If this is not true distribute_dst is (checked in can_apply)
        if collect_src:
            bank_count = int(src_array.shape[0])
            true_size = dst_array.shape
        else:
            bank_count = int(dst_array.shape[0])
            true_size = src_array.shape
        ndim = len(true_size)
        
        #Figure out how to split
        if self.split_array_info is None:
            tmp_split = round((bank_count)**(1 / ndim))
            if tmp_split**ndim != bank_count:
                raise RuntimeError("Splitting equally is not possible with "
                    "this array dimension and number of HBM-banks")
            split_info = [tmp_split]*ndim
        else:
            split_info = self.split_array_info
            if len(split_info) != ndim:
                raise RuntimeError("Length of split_array_info must match number of "
                    "dimensions")
        
        #create the copy-subgraph
        ndrange = dict()
        usable_params = ["i", "j", "k"]
        for i in range(ndim):
            ndrange[usable_params[i]] = f"0:{split_info[i]}"
        graph.remove_edge_and_connectors(graph.edges_between(src, dst)[0])
        copy_map_enter, copy_map_exit = graph.add_map("copy_map", ndrange,
            dtypes.ScheduleType.Unrolled)
        graph.add_edge(copy_map_enter, None, src, None, memlet.Memlet())
        graph.add_edge(dst, None, copy_map_exit, None, memlet.Memlet())

        target_size = [str(x) for x in self._get_split_size(true_size, split_info)]
        target_hbm_bank = []
        for i in range(ndim):
            current = usable_params[i]
            if i > 0:
                current += f"*{target_hbm_bank[i-1]}"
            target_hbm_bank.append(usable_params[i])
        target_offset = []
        for i in range(ndim):
            target_offset.append(f"{usable_params[i]}*{target_size[i]}")

        target_size_str = ", ".join([f"{x}:{y}" for x, y in zip([0]*ndim, target_size)])
        target_hbm_bank_str = "+ ".join(target_hbm_bank)
        target_offset_str = ", ".join([f"({x}):({x}+{y})" 
            for x, y in zip(target_offset, target_size)])
        if collect_src:
            copy_memlet = memlet.Memlet(
                f"{src.data}[{target_hbm_bank_str}, {target_size_str}]->"
                f"{target_offset_str}"
            )
        else:
            copy_memlet = memlet.Memlet(
                f"{src.data}[{target_offset_str}]->{target_hbm_bank_str}, "
                f"{target_size_str}"
            )
        graph.add_edge(src, None, dst, None, copy_memlet)
        
