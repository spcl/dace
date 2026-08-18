# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``CopyLibraryNode``: an explicit copy between two data containers.
"""
from typing import TYPE_CHECKING

from dace import library, nodes, dtypes
from dace.libraries.standard.helper import (CURRENT_STREAM_NAME, CPU_RESIDENT_STORAGES)
from dace.libraries.standard.nodes.copy.common import INPUT_CONNECTOR_NAME, OUTPUT_CONNECTOR_NAME
from dace.libraries.standard.nodes.copy.expansions import (ExpandAuto, ExpandMappedTasklet, ExpandMemcpyCPU,
                                                           ExpandMemcpyCUDA1D, ExpandMemcpyCUDA2D,
                                                           ExpandMemcpyCUDANDStrided, ExpandSharedMemoryCollective,
                                                           ExpandTasklet)

if TYPE_CHECKING:
    from dace.libraries.standard.nodes.copy.node import CopyLibraryNode


@library.node
class CopyLibraryNode(nodes.LibraryNode):
    """Library node representing a data copy between two access nodes. Implementations:
    ``MappedTasklet`` (element-wise tasklet, also the rank-mismatch/reshape and large-CPU-copy
    path), ``Tasklet`` (bare assignment, no map), ``MemcpyCPU`` (single ``std::memcpy``),
    ``MemcpyCUDA1D``/``2D`` (``cudaMemcpyAsync``/``cudaMemcpy2DAsync``), ``MemcpyCUDANDStrided``
    (Sequential map of ``cudaMemcpyAsync``), ``SharedMemoryCollective`` (``dace::CopyND`` +
    ``__syncthreads()``).

    Does NOT accept dynamic (Scalar) input connectors -- subset expressions must use symbols
    already in scope at construction time, so the auto selector reasons purely from static
    memlet subsets.
    """

    implementations = {
        "Auto": ExpandAuto,
        "MappedTasklet": ExpandMappedTasklet,
        "Tasklet": ExpandTasklet,
        "MemcpyCPU": ExpandMemcpyCPU,
        "MemcpyCUDA1D": ExpandMemcpyCUDA1D,
        "MemcpyCUDA2D": ExpandMemcpyCUDA2D,
        "MemcpyCUDANDStrided": ExpandMemcpyCUDANDStrided,
        "SharedMemoryCollective": ExpandSharedMemoryCollective,
    }
    default_implementation = 'Auto'

    INPUT_CONNECTOR_NAME = "_cpy_in"
    OUTPUT_CONNECTOR_NAME = "_cpy_out"

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs={INPUT_CONNECTOR_NAME}, outputs={OUTPUT_CONNECTOR_NAME}, **kwargs)

    def src_storage(self, state) -> dtypes.StorageType:
        """Storage of the array feeding ``_cpy_in``, or ``Default`` if unwired.

        :param state: state containing this libnode (owning SDFG is ``state.sdfg``).
        :returns: the source :class:`~dace.dtypes.StorageType`.
        """
        in_edges = [e for e in state.in_edges(self) if e.dst_conn == INPUT_CONNECTOR_NAME]
        if not in_edges:
            return dtypes.StorageType.Default
        outer = state.memlet_path(in_edges[0])[0].src
        if not isinstance(outer, nodes.AccessNode):
            return dtypes.StorageType.Default
        return state.sdfg.arrays[outer.data].storage

    def dst_storage(self, state) -> dtypes.StorageType:
        """Storage of the array fed by ``_cpy_out``, or ``Default`` if unwired.

        :param state: state containing this libnode (owning SDFG is ``state.sdfg``).
        :returns: the destination :class:`~dace.dtypes.StorageType`.
        """
        out_edges = [e for e in state.out_edges(self) if e.src_conn == OUTPUT_CONNECTOR_NAME]
        if not out_edges:
            return dtypes.StorageType.Default
        outer = state.memlet_path(out_edges[0])[-1].dst
        if not isinstance(outer, nodes.AccessNode):
            return dtypes.StorageType.Default
        return state.sdfg.arrays[outer.data].storage

    def validate(self, sdfg, state, allow_cross_storage=True):
        """Resolve in/out edges, names, and subsets: ``(inp_name, inp, in_subset, out_name, out,
        out_subset)``. Raises ``ValueError`` if not wired with exactly one input and one output
        data edge, dtypes mismatch, an extraneous non-reserved input connector wired, or (when
        ``allow_cross_storage`` is False) the two storages differ.

        :param sdfg: SDFG containing ``state``.
        :param state: state containing this libnode.
        :param allow_cross_storage: when False, require matching src/dst storages.
        :returns: ``(inp_name, inp, in_subset, out_name, out, out_subset)``.
        :raises ValueError: see above.
        """
        out_edges = [oe for oe in state.out_edges(self) if oe.src_conn == OUTPUT_CONNECTOR_NAME]
        if len(out_edges) != 1:
            raise ValueError(f"{type(self).__name__} expects exactly one "
                             f"``{OUTPUT_CONNECTOR_NAME}`` output edge.")
        oe = out_edges[0]
        out = sdfg.arrays[oe.data.data]
        out_subset = oe.data.subset
        out_name = oe.src_conn

        reserved = {INPUT_CONNECTOR_NAME, CURRENT_STREAM_NAME}
        extra = [ie.dst_conn for ie in state.in_edges(self) if ie.dst_conn not in reserved and not ie.data.is_empty()]
        if extra:
            raise ValueError(f"{type(self).__name__} does not accept dynamic input connectors; got {extra}. "
                             f"Subset expressions must use symbols already in scope.")

        in_edges = [ie for ie in state.in_edges(self) if ie.dst_conn == INPUT_CONNECTOR_NAME]
        if len(in_edges) != 1:
            raise ValueError(f"{type(self).__name__} expects exactly one data input edge "
                             f"connected to the ``{INPUT_CONNECTOR_NAME}`` connector.")
        ie = in_edges[0]
        inp = sdfg.arrays[ie.data.data]
        in_subset = ie.data.subset
        inp_name = ie.dst_conn

        if inp.dtype != out.dtype:
            raise ValueError(f"Input and output data types must match (got {inp.dtype} vs {out.dtype}).")

        # Two host storages differ only in the allocator, so a plain memcpy between them is correct;
        # only a CPU/GPU (or other target-specific) pairing genuinely needs a different expansion.
        host_pair = {inp.storage, out.storage} <= (CPU_RESIDENT_STORAGES | {dtypes.StorageType.Default})
        if not allow_cross_storage and inp.storage != out.storage and not host_pair:
            raise ValueError(f"Input and output storage types must match for this expansion "
                             f"(got {inp.storage} vs {out.storage}). Use a cross-storage "
                             f"expansion or the pure fallback.")

        return inp_name, inp, in_subset, out_name, out, out_subset
