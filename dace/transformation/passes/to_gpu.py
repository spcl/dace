# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes and functions that implement the grid-strided map tiling
    transformation."""

import ast
import copy
import dace
import typing
from dace.codegen.control_flow import ConditionalBlock, ControlFlowBlock
from dace.data import Property, make_properties
from dace.sdfg import is_devicelevel_gpu
from dace.transformation import pass_pipeline as ppl


@make_properties
class ToGPU(ppl.Pass):
    verbose: bool = Property(dtype=bool, default=False, desc="Print debug information")
    #duplicated_const_arrays: typing.Dict[str, str] = Property(
    #    dtype=dict,
    #    default=None,
    #    desc="Dictionary of duplicated constant arrays (key: old name, value: new name)"
    #)

    def __init__(
        self,
        verbose: bool = False,
        #duplicated_const_arrays: typing.Dict[str, str] = None,
    ):
        self.verbose = verbose
        super().__init__()

    def modifies(self) -> ppl.Modifies:
        return (
            ppl.Modifies.Nodes
            | ppl.Modifies.Edges
            | ppl.Modifies.AccessNodes
            | ppl.Modifies.Memlets
            | ppl.Modifies.Descriptors
        )

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False


    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: typing.Dict[str, typing.Any]) -> int:
        start_state = sdfg.start_state
        copies_in_first_state = dict()
        for edge in start_state.edges():
            if (isinstance(edge.src, dace.nodes.AccessNode) and
                isinstance(edge.dst, dace.nodes.AccessNode) and
                sdfg.arrays[edge.dst.data].storage == dace.dtypes.StorageType.GPU_Global
                and sdfg.arrays[edge.src.data].storage != dace.dtypes.StorageType.GPU_Global):
                copies_in_first_state[edge.src.data] = edge.dst.data

        # TODO: Filter unused names

        # If not copied, transient, and currently default storage, make it GPU glob
        descs = dict()
        for name, arr in sdfg.arrays.items():
            if isinstance(arr, dace.data.Array):
                if (arr.storage == dace.dtypes.StorageType.Default and arr.transient
                    and name not in copies_in_first_state and
                    name not in copies_in_first_state.values()):
                    arr.storage = dace.dtypes.StorageType.GPU_Global
            if isinstance(arr, dace.data.Scalar):
                arr.storage = dace.dtypes.StorageType.Register
            """
            # Do not generate GPU scalars for now
            # Only generate if a kernel really needs to write to a scalar
            if isinstance(arr, dace.data.Scalar):
                arr.storage = dace.dtypes.StorageType.Register
                if (arr.storage == dace.dtypes.StorageType.Register and arr.transient
                    and name not in copies_in_first_state and
                    name not in copies_in_first_state.values()):
                    # Add corresponding GPU scalar in case we need to use it
                    desc = dace.data.Scalar(
                        dtype=arr.dtype,
                        transient=True,
                        storage=dace.dtypes.StorageType.GPU_Global,
                    )
                    descs["gpu_" + name] = desc
                    an1 = start_state.add_access(name)
                    an2 = start_state.add_access("gpu_" + name)
                    start_state.add_edge(an1, None, an2, None,
                                            dace.memlet.Memlet(expr=name))
            """

        # Copy all default all CPU to GPU
        descs = dict()
        for name, arr in sdfg.arrays.items():
            if isinstance(arr, dace.data.Array) and name not in copies_in_first_state:
                if (arr.storage == dace.dtypes.StorageType.Default or
                    arr.storage == dace.dtypes.StorageType.CPU_Heap):
                    gpu_arr = copy.deepcopy(arr)
                    gpu_arr.storage = dace.dtypes.StorageType.GPU_Global
                    gpu_arr.transient = True
                    descs["gpu_" + name] = gpu_arr
                    an1 = start_state.add_access(name)
                    an2 = start_state.add_access("gpu_" + name)
                    start_state.add_edge(an1, None, an2, None,
                                         dace.memlet.Memlet(expr=name))

        for name, arr in descs.items():
            sdfg.arrays[name] = arr