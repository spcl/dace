# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Placement of an SDFG's top-level transients in a caller-owned byte workspace.

A caller that owns all memory (a benchmark judge, a library with its own allocator) passes one buffer of bytes,
the workspace, and its size. :class:`CarveTransientsFromWorkspace` chooses the transients the entry allocates once
per call and gives each a fixed byte offset into that buffer, aligned to :data:`WORKSPACE_ALIGNMENT`. The arrays keep
their descriptors -- shape, strides, data type -- so every access indexes them as before; only their allocation
becomes a pointer into the workspace, which the code generator writes from the :class:`WorkspaceLayout`.
"""

import dataclasses
from typing import Any

from dace import data as dt
from dace import dtypes, properties, symbolic
from dace.sdfg import nodes
from dace.sdfg.sdfg import SDFG
from dace.transformation import pass_pipeline as ppl

#: Each array placed in the workspace starts on this boundary, the alignment its own allocation would have had. The
#: caller passes a workspace aligned at least as strictly.
WORKSPACE_ALIGNMENT = 64

#: The storages a workspace of a given storage holds: a host workspace the heap arrays, a device one the device arrays.
HELD_STORAGES: dict[dtypes.StorageType, tuple[dtypes.StorageType, ...]] = {
    dtypes.StorageType.Default: (dtypes.StorageType.CPU_Heap, dtypes.StorageType.Default),
    dtypes.StorageType.CPU_Heap: (dtypes.StorageType.CPU_Heap, dtypes.StorageType.Default),
    dtypes.StorageType.GPU_Global: (dtypes.StorageType.GPU_Global,),
}


def aligned(nbytes: Any) -> Any:
    """``nbytes`` rounded up to the next multiple of :data:`WORKSPACE_ALIGNMENT`, as a symbolic expression."""
    return symbolic.int_ceil(nbytes, WORKSPACE_ALIGNMENT) * WORKSPACE_ALIGNMENT


@dataclasses.dataclass(frozen=True)
class WorkspaceLayout:
    """Where each carved array lives in the workspace.

    ``offsets`` maps a data name to its byte offset, ``descriptors`` the same name to the descriptor that was placed
    (a nested SDFG's array of the same name is not placed), and ``size_bytes`` bounds the bytes the workspace must
    hold: each array's bytes plus one :data:`WORKSPACE_ALIGNMENT`, plain arithmetic a caller evaluates without the
    rounding helpers the offsets use. Offsets and size are symbolic expressions over the entry's scalar arguments.
    """

    workspace: str
    size: str
    offsets: dict[str, Any]
    descriptors: dict[str, dt.Data]
    size_bytes: Any

    def offset(self, name: str | None, desc: dt.Data | None) -> Any | None:
        """The byte offset of ``desc`` if the layout placed it under ``name``, else ``None``."""
        if name is None or self.descriptors.get(name) is not desc:
            return None
        return self.offsets[name]

    def holds(self, desc: dt.Data) -> bool:
        """Whether ``desc`` lives in the workspace, so freeing it frees nothing."""
        return any(placed is desc for placed in self.descriptors.values())


def mapped_and_zeroed_data(sdfg: SDFG) -> tuple[set[str], set[str]]:
    """The data ``sdfg``'s own states access inside a host map, and the data an access node zeroes on allocation.

    A host map gives each iteration, each thread, its own allocation of a ``Scope`` array it touches. A kernel does
    not: the device array a kernel touches is allocated once, outside it, so only a map whose outermost scope is not
    a GPU schedule counts.
    """
    in_maps: set[str] = set()
    zeroed: set[str] = set()
    for state in sdfg.states():
        scopes = state.scope_dict()
        for node, scope in scopes.items():
            if isinstance(node, nodes.AccessNode):
                outermost = scope
                while outermost is not None and scopes[outermost] is not None:
                    outermost = scopes[outermost]
                if outermost is not None and outermost.map.schedule not in dtypes.ALL_GPU_SCHEDULES:
                    in_maps.add(node.data)
                if node.setzero:
                    zeroed.add(node.data)
    return in_maps, zeroed


def placeable_type(desc: dt.Data) -> bool:
    """Whether ``desc`` is data an allocation points at: an array, or a device scalar (a host scalar is a value)."""
    return type(desc) is dt.Array or (type(desc) is dt.Scalar and desc.storage is dtypes.StorageType.GPU_Global)


@properties.make_properties
class CarveTransientsFromWorkspace(ppl.Pass):
    """Give the transients the entry allocates once per call a slot of the workspace.

    An array is placed when it is a top-level transient (:func:`placeable_type`) in a storage the workspace holds
    (:data:`HELD_STORAGES`), sized by the entry's arguments alone, and allocated once per call: not a ``Scope``
    array a host map touches (allocated per iteration, per thread), not one zeroed on allocation unless its lifetime is
    already ``SDFG``, and not ``Global`` or ``External`` memory, which the program or the caller owns. Each placed
    array's lifetime becomes ``SDFG``: allocated on entry, whichever state or loop body allocated it before. Its slot
    is its own either way. A placed array is no longer pooled: it has no allocation for a pool to take or release.

    Arrays are placed in name order, so the layout is deterministic.
    """

    CATEGORY: str = "Memory"

    workspace = properties.Property(dtype=str, default="", desc="The workspace argument, a uint8 array.")
    size = properties.Property(dtype=str, default="", desc="The workspace's size argument, in bytes.")

    def __init__(self, workspace: str = "", size: str = ""):
        self.workspace = workspace
        self.size = size

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> WorkspaceLayout:
        held = HELD_STORAGES.get(sdfg.arrays[self.workspace].storage, ())
        arguments = set(sdfg.arglist())
        in_maps, zeroed = mapped_and_zeroed_data(sdfg)
        offsets: dict[str, Any] = {}
        descriptors: dict[str, dt.Data] = {}
        end: Any = 0
        bound: Any = 0
        for name, desc in sorted(sdfg.arrays.items()):
            if (
                not desc.transient
                or not placeable_type(desc)
                or desc.storage not in held
                or desc.lifetime in (dtypes.AllocationLifetime.Global, dtypes.AllocationLifetime.External)
                or (desc.lifetime is dtypes.AllocationLifetime.Scope and name in in_maps)
                or (desc.lifetime is not dtypes.AllocationLifetime.SDFG and name in zeroed)
                or not {str(s) for s in symbolic.symlist(desc.total_size)} <= arguments
            ):
                continue
            desc.lifetime, desc.pool = dtypes.AllocationLifetime.SDFG, False
            offsets[name], descriptors[name] = end, desc
            nbytes = desc.total_size * desc.dtype.bytes
            end, bound = end + aligned(nbytes), bound + nbytes + WORKSPACE_ALIGNMENT
        return WorkspaceLayout(self.workspace, self.size, offsets, descriptors, bound)
