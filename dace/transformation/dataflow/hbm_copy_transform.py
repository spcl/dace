from typing import Any, Dict, Iterable, List, Tuple, Union

from dace import dtypes, properties, registry
from dace.sdfg import utils
from dace.transformation import transformation
from dace.sdfg import nodes as nd
from dace import SDFG, SDFGState, memlet
import functools


@registry.autoregister_params(singlestate=True)
@properties.make_properties
class HbmCopyTransform(transformation.Transformation):
    """
    A transformation that allows to split an array and distribute on
    an array with one dimension more, or the reverse operation. Works in principle
    with arbitrary arrays, but it's real use case is to distribute data on many HBM-banks.
    Matches any 2 AccessNodes connected by any edge, if the dimensionality of the two accessed
    arrays differ by exactly one. The sizes of the arrays have to be large enough with
    respect to the split executed, but this is not verified.
    """

    _src_node = nd.AccessNode("")
    _dst_node = nd.AccessNode("")

    # dtype=List[int]
    split_array_info = properties.Property(
        dtype=List,
        default=None,
        allow_none=True,
        desc="Describes how many times this array is split in each dimension. "
        "A value of 1 means that the array is not split in a dimension at all. "
        "If None, then the transform will try to split equally in each dimension."
    )

    default_to_cpu_storage = properties.Property(
        dtype=bool,
        default=True,
        allow_none=False,
        desc="If set storage types will be set to CPU Heap if on Default"
    )

    def _get_split_size(self, virtual_shape: Iterable,
                        split_count: List[int]) -> List[int]:
        """
        :return: the shape of a part-array on one HBMbank
        """
        new_shape_list = []
        for d in range(len(virtual_shape)):
            if split_count[d] != 1:
                new_shape_list.append(virtual_shape[d] // split_count[d])
            else:
                new_shape_list.append(virtual_shape[d])
        return new_shape_list

    @staticmethod
    def can_be_applied(graph: Union[SDFG, SDFGState],
                       candidate: Dict['PatternNode', int], expr_index: int,
                       sdfg: SDFG, strict: bool) -> bool:
        src = graph.nodes()[candidate[HbmCopyTransform._src_node]]
        dst = graph.nodes()[candidate[HbmCopyTransform._dst_node]]
        src_array = sdfg.arrays[src.data]
        dst_array = sdfg.arrays[dst.data]

        # same dimensions means HBM-array needs 1 dimension more
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
        return [
            utils.node_path_graph(HbmCopyTransform._src_node,
                                  HbmCopyTransform._dst_node)
        ]

    def apply(self, sdfg: SDFG) -> Union[Any, None]:
        # Load/parse infos from the SDFG
        graph = sdfg.nodes()[self.state_id]
        src = graph.nodes()[self.subgraph[HbmCopyTransform._src_node]]
        dst = graph.nodes()[self.subgraph[HbmCopyTransform._dst_node]]
        src_array = sdfg.arrays[src.data]
        dst_array = sdfg.arrays[dst.data]
        collect_src = len(src_array.shape) - 1 == len(
            dst_array.shape
        )  # If this is not true we have to distribute to dst (checked in can_apply)
        if collect_src:
            bank_count = int(src_array.shape[0])
            true_size = dst_array.shape
        else:
            bank_count = int(dst_array.shape[0])
            true_size = src_array.shape
        ndim = len(true_size)

        #Initialize array defaults
        if self.default_to_cpu_storage:
            if sdfg.arrays[src.data].storage == dtypes.StorageType.Default:
                sdfg.arrays[src.data].storage = dtypes.StorageType.CPU_Heap
            if sdfg.arrays[dst.data].storage == dtypes.StorageType.Default:
                sdfg.arrays[dst.data].storage = dtypes.StorageType.CPU_Heap

        # Figure out how to split
        if self.split_array_info is None:
            tmp_split = round((bank_count)**(1 / ndim))
            split_info = [tmp_split] * ndim
        else:
            split_info = self.split_array_info
            if len(split_info) != ndim:
                raise RuntimeError(
                    "Length of split_array_info must match number of "
                    "dimensions")
        if functools.reduce(lambda a, b: a * b, split_info) != bank_count:
            raise RuntimeError(
                "Splitting is not possible with the selected splits"
                "and this number of HBM-banks (required number of banks "
                "!= actual number of banks)")

        # create the copy-subgraph
        ndrange = dict()
        usable_params = ["i", "j", "k"]
        for i in range(ndim):
            ndrange[usable_params[i]] = f"0:{split_info[i]}"
        graph.remove_edge_and_connectors(graph.edges_between(src, dst)[0])
        copy_map_enter, copy_map_exit = graph.add_map(
            "copy_map", ndrange, dtypes.ScheduleType.Unrolled)
        graph.add_edge(copy_map_enter, None, src, None, memlet.Memlet())
        graph.add_edge(dst, None, copy_map_exit, None, memlet.Memlet())

        target_size = [
            str(x) for x in self._get_split_size(true_size, split_info)
        ]
        target_hbm_bank = []
        for i in range(ndim):
            target_hbm_bank.append(usable_params[i])
            for j in range(i):
                target_hbm_bank[j] = f"{split_info[i]}*{target_hbm_bank[j]}"
        target_offset = []
        for i in range(ndim):
            target_offset.append(f"{usable_params[i]}*{target_size[i]}")

        target_size_str = ", ".join(
            [f"{x}:{y}" for x, y in zip([0] * ndim, target_size)])
        target_hbm_bank_str = "+ ".join(target_hbm_bank)
        target_offset_str = ", ".join(
            [f"({x}):({x}+{y})" for x, y in zip(target_offset, target_size)])
        if collect_src:
            copy_memlet = memlet.Memlet(
                f"{src.data}[{target_hbm_bank_str}, {target_size_str}]->"
                f"{target_offset_str}")
        else:
            copy_memlet = memlet.Memlet(
                f"{src.data}[{target_offset_str}]->{target_hbm_bank_str}, "
                f"{target_size_str}")
        graph.add_edge(src, None, dst, None, copy_memlet)
