# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``MemsetLibraryNode`` representing 0-memsets over contiguous subsets.

The CUDA expansion emits the ambient ``__dace_current_stream`` symbol; the
GPU stream scheduler binds it post-expansion (legacy codegen declares it
directly), so the libnode carries no stream input connector itself.
"""
from typing import List, Tuple

import dace
from dace import library, nodes
from dace.codegen.common import sym2cpp
from dace.sdfg.scope import is_devicelevel_gpu
from dace.transformation.transformation import ExpandTransformation
from .. import environments

from dace.libraries.standard.helper import (CURRENT_STREAM_NAME, auto_dispatch, collapse_shape_and_strides,
                                            is_parallel_cpu_transfer_size, make_parallel_cpu_transfer_sdfg)


def _make_memset_skeleton(node: "MemsetLibraryNode", parent_state: dace.SDFGState,
                          parent_sdfg: dace.SDFG) -> Tuple[dace.SDFG, dace.SDFGState, str, dace.data.Data, List]:
    """Build the shared SDFG skeleton for the mapped (``ExpandPure``) memset expansion.

    :param node: The memset library node being expanded.
    :param parent_state: The state containing ``node``.
    :param parent_sdfg: The SDFG containing ``parent_state``.
    :returns: ``(sdfg, state, out_name, out, map_lengths)``.
    """
    out_name, out, out_subset = node.validate(parent_sdfg, parent_state)
    out_shape_collapsed, out_strides_collapsed = collapse_shape_and_strides(out_subset, out.strides)

    sdfg = dace.SDFG(f"{node.label}_sdfg")
    sdfg.add_array(out_name, out_shape_collapsed, out.dtype, out.storage, strides=out_strides_collapsed)
    sdfg.schedule = dace.dtypes.ScheduleType.Sequential

    state = sdfg.add_state(f"{node.label}_state")
    map_lengths = [s for s in out_subset.size() if s != 1]

    return sdfg, state, out_name, out, map_lengths


def _make_memset_tasklet(node: "MemsetLibraryNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG, *,
                         cuda: bool) -> nodes.Tasklet:
    """Build a direct memset tasklet.

    Emits the stream-bound ``cudaMemsetAsync`` form when ``cuda`` is set,
    otherwise plain ``memset``.

    :param node: The memset library node being expanded.
    :param parent_state: The state containing ``node``.
    :param parent_sdfg: The SDFG containing ``parent_state``.
    :param cuda: Emit ``cudaMemsetAsync`` (else ``memset``).
    :returns: The memset tasklet.
    :raises ValueError: if the output subset is non-contiguous; the single-call
        ``cudaMemsetAsync`` / ``memset`` form would silently zero memory outside
        the subset. Use the ``pure`` expansion (mapped tasklet) for those.
    """
    out_name, out, out_subset = node.validate(parent_sdfg, parent_state)
    if not out_subset.is_contiguous_subset(out):
        raise ValueError(
            f"MemsetLibraryNode {'CUDA' if cuda else 'CPU'} expansion requires a contiguous subset; "
            f"got '{out_name}' subset {out_subset} on shape {tuple(out.shape)} strides {tuple(out.strides)}. "
            f"Use the 'pure' expansion (mapped tasklet) for non-contiguous regions.")

    nbytes = f"{sym2cpp(out_subset.num_elements())} * sizeof({out.dtype.ctype})"
    if cuda:
        code = f"cudaMemsetAsync({MemsetLibraryNode.OUTPUT_CONNECTOR_NAME}, 0, {nbytes}, {CURRENT_STREAM_NAME});"
    else:
        code = f"memset({MemsetLibraryNode.OUTPUT_CONNECTOR_NAME}, 0, {nbytes});"

    return nodes.Tasklet(node.name,
                         inputs={},
                         outputs={MemsetLibraryNode.OUTPUT_CONNECTOR_NAME: dace.dtypes.pointer(out.dtype)},
                         code=code,
                         language=dace.Language.CPP)


def select_memset_implementation(node: "MemsetLibraryNode", parent_state: dace.SDFGState,
                                 parent_sdfg: dace.SDFG) -> str:
    """Resolve an ``'Auto'`` ``MemsetLibraryNode`` implementation to a concrete one.

    Returns ``'pure'`` (Sequential element-zero map) in device scope since
    ``cudaMemsetAsync`` cannot be issued from a kernel, and for non-contiguous
    subsets where the single-call memset forms would zero outside the region;
    ``'CUDA'`` (``cudaMemsetAsync``) for host-issued GPU-destination contiguous
    memsets; ``'ParallelCPU'`` (OpenMP-chunked ``std::memset``) for a contiguous
    CPU main-memory zero whose size is symbolic or statically ``>=
    PARALLEL_COPY_MIN_BYTES``; otherwise ``'CPU'`` (single ``std::memset``).

    :param node: The memset library node being expanded.
    :param parent_state: The state containing ``node``.
    :param parent_sdfg: The SDFG containing ``parent_state``.
    :returns: One of ``'pure'``, ``'CUDA'``, ``'ParallelCPU'``, or ``'CPU'``.
    """
    _out_name, out, out_subset = node.validate(parent_sdfg, parent_state)

    if is_devicelevel_gpu(parent_sdfg, parent_state, node):
        return 'pure'

    if not out_subset.is_contiguous_subset(out):
        return 'pure'

    if out.storage == dace.dtypes.StorageType.GPU_Global:
        return 'CUDA'

    # Same-side CPU main-memory contiguous zero: parallelize above the byte
    # threshold (or when the size is symbolic), else a single ``std::memset``.
    # ``Register`` and other non-main-memory storages keep the serial call.
    allowed = dace.dtypes.CPU_RESIDENT_STORAGES | {dace.dtypes.StorageType.Default}
    if out.storage in allowed and is_parallel_cpu_transfer_size(out_subset.num_elements() * out.dtype.bytes):
        return 'ParallelCPU'
    return 'CPU'


@library.expansion
class ExpandAuto(ExpandTransformation):
    """Default expansion: dispatches to the implementation chosen by
    :func:`select_memset_implementation` based on the destination storage
    and the surrounding scope."""
    environments = []

    @staticmethod
    def expansion(node: "MemsetLibraryNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG):
        return auto_dispatch(node, parent_state, parent_sdfg, select_memset_implementation, MemsetLibraryNode)


@library.expansion
class ExpandPure(ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node: "MemsetLibraryNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> dace.SDFG:
        sdfg, state, out_name, out, map_lengths = _make_memset_skeleton(node, parent_state, parent_sdfg)

        # Inner-tasklet connector. Must not collide with the wrapper SDFG's
        # parameter array, which is named after the libnode's outer connector.
        inner_out = "_out"
        map_params = [f"__i{i}" for i in range(len(map_lengths))]
        map_rng = {i: f"0:{s}" for i, s in zip(map_params, map_lengths)}
        outputs = {inner_out: dace.memlet.Memlet(f"{out_name}[{','.join(map_params)}]")}
        schedule = (dace.dtypes.ScheduleType.GPU_Device
                    if out.storage == dace.dtypes.StorageType.GPU_Global else dace.dtypes.ScheduleType.Default)
        state.add_mapped_tasklet(f"{node.label}_tasklet",
                                 map_rng,
                                 dict(),
                                 f"{inner_out} = 0",
                                 outputs,
                                 schedule=schedule,
                                 external_edges=True)

        return sdfg


@library.expansion
class ExpandCUDA(ExpandTransformation):
    environments = [environments.CUDA]

    @staticmethod
    def expansion(node: "MemsetLibraryNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        return _make_memset_tasklet(node, parent_state, parent_sdfg, cuda=True)


@library.expansion
class ExpandCPU(ExpandTransformation):
    environments = [environments.CPU]

    @staticmethod
    def expansion(node: "MemsetLibraryNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        return _make_memset_tasklet(node, parent_state, parent_sdfg, cuda=False)


@library.expansion
class ExpandParallelCPU(ExpandTransformation):
    """OpenMP-parallel ``std::memset`` for a large contiguous CPU zero.

    Expands to a wrapper SDFG whose ``CPU_Multicore`` map (lowered by codegen to
    ``#pragma omp parallel for``) iterates ``ceil(N / chunk)`` chunks, each running one
    contiguous ``std::memset`` over its slice of up to
    :data:`~dace.libraries.standard.helper.PARALLEL_COPY_CHUNK_BYTES` bytes. Selected by
    the auto path for contiguous CPU memsets whose size is symbolic or statically
    ``>= PARALLEL_COPY_MIN_BYTES``. The OpenMP parallelism comes from the map schedule."""
    environments = [environments.CPU]

    @staticmethod
    def expansion(node: "MemsetLibraryNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> dace.SDFG:
        out_name, out, out_subset = node.validate(parent_sdfg, parent_state)
        if not out_subset.is_contiguous_subset(out):
            raise ValueError(
                f"MemsetLibraryNode ParallelCPU expansion requires a contiguous subset; got '{out_name}' subset "
                f"{out_subset} on shape {tuple(out.shape)} strides {tuple(out.strides)}. "
                f"Use the 'pure' expansion (mapped tasklet) for non-contiguous regions.")

        return make_parallel_cpu_transfer_sdfg(node.label,
                                               out.dtype,
                                               out.storage,
                                               out_subset.num_elements(),
                                               out_array=MemsetLibraryNode.OUTPUT_CONNECTOR_NAME)


@library.node
class MemsetLibraryNode(nodes.LibraryNode):
    """Library node representing a 0-memset over a contiguous output subset.

    Design rationale: the libnode does NOT accept dynamic (Scalar) input
    connectors -- the subset expression must use symbols already in scope at
    construction time. This keeps the contract simple and lets the auto
    selector reason purely from the static memlet subset.
    """

    implementations = {
        "Auto": ExpandAuto,
        "pure": ExpandPure,
        "CUDA": ExpandCUDA,
        "CPU": ExpandCPU,
        "ParallelCPU": ExpandParallelCPU,
    }
    default_implementation = 'Auto'

    # Connector name exposed for library node builders.
    OUTPUT_CONNECTOR_NAME = "_mset_out"

    def __init__(self, name: str, *args, **kwargs):
        super().__init__(name, *args, outputs={MemsetLibraryNode.OUTPUT_CONNECTOR_NAME}, **kwargs)

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> Tuple[str, dace.data.Data, dace.subsets.Range]:
        """Validate wiring and resolve the output edge.

        :param sdfg: The SDFG owning the data descriptors.
        :param state: The state containing this node.
        :returns: ``(out_name, out, out_subset)``.
        :raises ValueError: If the node lacks exactly one output edge or has
            any non-empty non-reserved input connector wired.
        """
        data_oes = [oe for oe in state.out_edges(self) if oe.src_conn == MemsetLibraryNode.OUTPUT_CONNECTOR_NAME]
        if len(data_oes) != 1:
            raise ValueError(f"{type(self).__name__} expects exactly one "
                             f"``{MemsetLibraryNode.OUTPUT_CONNECTOR_NAME}`` output edge.")

        # Reject any non-empty input connector: the libnode does not accept
        # dynamic inputs (see class docstring's design rationale).
        reserved = {CURRENT_STREAM_NAME}
        extra = [ie.dst_conn for ie in state.in_edges(self) if ie.dst_conn not in reserved and not ie.data.is_empty()]
        if extra:
            raise ValueError(f"{type(self).__name__} does not accept dynamic input connectors; got {extra}. "
                             f"Subset expressions must use symbols already in scope.")

        oe = data_oes[0]
        out = sdfg.arrays[oe.data.data]
        out_subset = oe.data.subset
        out_name = oe.src_conn

        return out_name, out, out_subset
