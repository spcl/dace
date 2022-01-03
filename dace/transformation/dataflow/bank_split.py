# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Any, Dict, Iterable, List, Tuple, Union

from dace import data, dtypes, properties
from dace.sdfg import utils
from dace.transformation import transformation
from dace.sdfg import nodes as nd
from dace import SDFG, SDFGState, memlet
from dace import symbolic
import functools


@properties.make_properties
class BankSplit(transformation.SingleStateTransformation):
    """
    A transformation that allow splitting an array and distribute it on another
    array with one dimension more, or vice versa. Works with arbitrary arrays,
    but its intended use case is to distribute data on many HBM-banks.
    Matches any 2 AccessNodes connected by an edge, if the dimensionality of the two accessed
    arrays differ by exactly one. The sizes of the arrays have to be large enough with
    respect to the split executed, but this is not verified. While it is allowed to use symbolics
    for the shapes of the array, it is expected that each dimension is divisible by the number
    of splits specified.

    When appling an unrolled map is generated around the accessnodes, which copies the parts of
    the array to the target array.

    Examples:
    Distribute: Suppose for example we copy from A to B, where A has shape [100, 100] and B shape
    [10, 100, 10]. We can distribute A in that case to B using the transformation by setting
    split_array_info=[1, 10]. A will then be divided along it's second dimension into 10 parts
    of size [100, 10] and distributed on B.
    Gather: Suppose A has shape [4, 50, 50] and B has shape [100, 100]. If one sets
    split_array_info to [2, 2] and applies the transformation, it will split
    equally in all dimensions.
    Therefore A[0] will be copied to B[0:50, 0:50], A[1] to B[0:50, 50:100], A[2] to B[50:100, 0:50] and
    A[3] to B[50:100, 50:100].

    Note that simply reversing the AccessNodes for the arrays in the above examples would
    have lead to the inverse operation, i.e. the gather would become a distribute and
    the other way around.
    """

    _src_node = nd.AccessNode("")
    _dst_node = nd.AccessNode("")

    # dtype=List[int]
    split_array_info = properties.Property(
        dtype=List,
        default=None,
        allow_none=True,
        desc="Describes how many times this array is split in each dimension, "
        "where the k-th number describes how many times dimension k is split. "
        "If the k-th number is 1 this means that the array is not split in "
        "the k-th dimension at all. "
        "If None, then the transform will split the first dimension exactly shape[0] times.")

    default_to_storage = properties.Property(
        dtype=dtypes.StorageType,
        default=dtypes.StorageType.CPU_Heap,
        allow_none=False,
        desc="The storage type of involved arrays will be set to the value of this property if "
        "they have Default storage type. ")

    def _get_split_size(self, virtual_shape: Iterable, split_count: List[int]) -> List[int]:
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
    def can_be_applied(graph: Union[SDFG, SDFGState], candidate: Dict['PatternNode', int], expr_index: int, sdfg: SDFG,
                       permissive: bool) -> bool:
        src = graph.nodes()[candidate[BankSplit._src_node]]
        dst = graph.nodes()[candidate[BankSplit._dst_node]]
        src_array = sdfg.arrays[src.data]
        dst_array = sdfg.arrays[dst.data]

        plain_array = lambda array: isinstance(array, data.Array) and not isinstance(array, data.View)

        if not plain_array(src_array):
            return False
        if not plain_array(dst_array):
            return False

        # same dimensions means HBM-array needs 1 dimension more
        collect_src = len(src_array.shape) - 1 == len(dst_array.shape)
        distribute_dst = len(src_array.shape) + 1 == len(dst_array.shape)
        if collect_src and symbolic.issymbolic(src_array.shape[0], sdfg.constants):
            return False
        elif distribute_dst and symbolic.issymbolic(dst_array.shape[0], sdfg.constants):
            return False
        return collect_src or distribute_dst

    @staticmethod
    def expressions():
        return [utils.node_path_graph(BankSplit._src_node, BankSplit._dst_node)]

    def apply(self, sdfg: SDFG) -> Union[Any, None]:
        # Load/parse infos from the SDFG
        graph = sdfg.nodes()[self.state_id]
        src = graph.nodes()[self.subgraph[BankSplit._src_node]]
        dst = graph.nodes()[self.subgraph[BankSplit._dst_node]]
        src_array = sdfg.arrays[src.data]
        dst_array = sdfg.arrays[dst.data]
        collect_src = len(src_array.shape) - 1 == len(
            dst_array.shape)  # If this is not true we have to distribute to dst (checked in can_apply)
        if collect_src:
            bank_count = int(src_array.shape[0])
            true_size = dst_array.shape
        else:
            bank_count = int(dst_array.shape[0])
            true_size = src_array.shape
        ndim = len(true_size)

        # Move Default storage
        if sdfg.arrays[src.data].storage == dtypes.StorageType.Default:
            sdfg.arrays[src.data].storage = self.default_to_storage
        if sdfg.arrays[dst.data].storage == dtypes.StorageType.Default:
            sdfg.arrays[dst.data].storage = self.default_to_storage

        # Figure out how to split
        if self.split_array_info is None:
            split_info = [1] * ndim
            split_info[0] = bank_count
        else:
            split_info = self.split_array_info
            if len(split_info) != ndim:
                raise RuntimeError("Length of split_array_info must match number of " "dimensions")
        if functools.reduce(lambda a, b: a * b, split_info) != bank_count:
            raise RuntimeError("Splitting is not possible with the selected splits"
                               "and this number of HBM-banks (required number of banks "
                               "!= actual number of banks)")

        # create the copy-subgraph
        ndrange = dict()
        usable_params = []
        for i in range(ndim):
            usable_params.append(f"i{i}")
        for i in range(ndim):
            ndrange[usable_params[i]] = f"0:{split_info[i]}"
        graph.remove_edge_and_connectors(graph.edges_between(src, dst)[0])
        copy_map_enter, copy_map_exit = graph.add_map("hbm_bank_split", ndrange, dtypes.ScheduleType.Unrolled)
        graph.add_edge(copy_map_enter, None, src, None, memlet.Memlet())
        graph.add_edge(dst, None, copy_map_exit, None, memlet.Memlet())

        target_size = [str(x) for x in self._get_split_size(true_size, split_info)]
        target_hbm_bank = []
        for i in range(ndim):
            target_hbm_bank.append(usable_params[i])
            for j in range(i):
                target_hbm_bank[j] = f"{split_info[i]}*{target_hbm_bank[j]}"
        target_offset = []
        for i in range(ndim):
            target_offset.append(f"{usable_params[i]}*{target_size[i]}")

        target_size_str = ", ".join([f"{x}:{y}" for x, y in zip([0] * ndim, target_size)])
        target_hbm_bank_str = "+ ".join(target_hbm_bank)
        target_offset_str = ", ".join([f"({x}):({x}+{y})" for x, y in zip(target_offset, target_size)])
        if collect_src:
            copy_memlet = memlet.Memlet(f"{src.data}[{target_hbm_bank_str}, {target_size_str}]->"
                                        f"{target_offset_str}")
        else:
            copy_memlet = memlet.Memlet(f"{src.data}[{target_offset_str}]->{target_hbm_bank_str}, "
                                        f"{target_size_str}")
        graph.add_edge(src, None, dst, None, copy_memlet)
