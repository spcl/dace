# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``ComputePersistentKernelRange``: the host computes how many thread blocks of a cooperative kernel fit the device,
before launching it."""

import dace
from dace import dtypes, library, nodes, properties
from dace.sdfg.scope import is_devicelevel_gpu
from dace.transformation.transformation import ExpandTransformation

#: The out-connector that receives the number of blocks
NBLK_CONNECTOR = "resident_blocks"


@library.node
class ComputePersistentKernelRange(nodes.LibraryNode):
    """Writes to :data:`NBLK_CONNECTOR` how many blocks of the cooperative kernel whose map has ``kernel_handle`` are
    resident on the device at once, within the configured share (``persistent_map_SM_fraction`` and
    ``persistent_map_occupancy``). The kernel's grid takes it as a dynamic map range, so its grid barriers are legal.
    """

    implementations = {}
    default_implementation = "native"

    kernel_handle = properties.Property(dtype=str, default="", desc="The kernel_handle of the cooperative kernel map.")

    def __init__(self, name: str, *, kernel_handle: str = "", **kwargs):
        super().__init__(name, outputs={NBLK_CONNECTOR}, **kwargs)
        self.kernel_handle = kernel_handle

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        if not self.kernel_handle:
            raise ValueError(f"ComputePersistentKernelRange {self.name}: no kernel_handle")
        if set(self.in_connectors) or set(self.out_connectors) != {NBLK_CONNECTOR}:
            raise ValueError(f"ComputePersistentKernelRange {self.name}: takes no input and writes {NBLK_CONNECTOR}")
        if is_devicelevel_gpu(sdfg, state, self):
            raise ValueError(f"ComputePersistentKernelRange {self.name}: runs on the host, not in a kernel")


@library.register_expansion(ComputePersistentKernelRange, "native")
class ExpandNative(ExpandTransformation):
    """Calls the host function the CUDA/HIP code generator emits next to the kernel's launcher."""

    environments = []

    @staticmethod
    def expansion(
        node: ComputePersistentKernelRange, parent_state: dace.SDFGState, parent_sdfg: dace.SDFG
    ) -> nodes.Tasklet:
        from dace.codegen.targets.experimental_cuda import persistent_blocks_name

        return nodes.Tasklet(
            node.name,
            {},
            {NBLK_CONNECTOR: dtypes.int32},
            f"{NBLK_CONNECTOR} = {persistent_blocks_name(node.kernel_handle)}(__state);",
            language=dtypes.Language.CPP,
            side_effects=True,
        )
