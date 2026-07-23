# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``MemsetLibraryNode`` representing 0-memsets."""
from typing import List, Tuple

import dace
from dace import library, nodes
from dace.codegen.common import sym2cpp
from dace.sdfg.scope import is_devicelevel_gpu
from dace.transformation.transformation import ExpandTransformation
from .. import environments

from dace.libraries.standard.helper import (CURRENT_STREAM_NAME, CPU_RESIDENT_STORAGES, GPU_RESIDENT_STORAGES,
                                            auto_dispatch, collapse_shape_and_strides, is_parallel_cpu_transfer_size)


def _make_memset_skeleton(node: "MemsetLibraryNode",
                          parent_state: dace.SDFGState) -> Tuple[dace.SDFG, dace.SDFGState, str, dace.data.Data, List]:
    """Build the shared SDFG skeleton for the mapped (``ExpandPure``) memset expansion ->
    ``(sdfg, state, out_name, out, map_lengths)``."""
    out_name, out, out_subset = node.validate(parent_state.sdfg, parent_state)
    out_shape_collapsed, out_strides_collapsed = collapse_shape_and_strides(out_subset, out.strides)

    sdfg = dace.SDFG(f"{node.label}_sdfg")
    sdfg.add_array(out_name, out_shape_collapsed, out.dtype, out.storage, strides=out_strides_collapsed)
    sdfg.schedule = dace.dtypes.ScheduleType.Sequential

    state = sdfg.add_state(f"{node.label}_state")
    # Reuse the array descriptor's collapsed shape as the map bounds, so rank/extents can't
    # diverge from the array.
    map_lengths = out_shape_collapsed

    return sdfg, state, out_name, out, map_lengths


def _make_memset_tasklet(node: "MemsetLibraryNode", parent_state: dace.SDFGState, *, cuda: bool) -> nodes.Tasklet:
    """Build a direct memset tasklet (``cudaMemsetAsync`` if ``cuda`` else ``memset``). Raises if
    the output subset is non-contiguous (single-call memset would zero outside the subset); use
    the ``pure`` expansion instead."""
    out_name, out, out_subset = node.validate(parent_state.sdfg, parent_state)
    if not out_subset.is_contiguous_subset(out):
        raise ValueError(
            f"MemsetLibraryNode {'CUDA' if cuda else 'CPU'} expansion requires a contiguous subset; "
            f"got '{out_name}' subset {out_subset} on shape {tuple(out.shape)} strides {tuple(out.strides)}. "
            f"Use the 'pure' expansion (mapped tasklet) for non-contiguous regions.")

    nbytes = f"{sym2cpp(out_subset.num_elements_exact())} * sizeof({out.dtype.ctype})"
    if cuda:
        code = f"cudaMemsetAsync({MemsetLibraryNode.OUTPUT_CONNECTOR_NAME}, 0, {nbytes}, {CURRENT_STREAM_NAME});"
    else:
        code = f"memset({MemsetLibraryNode.OUTPUT_CONNECTOR_NAME}, 0, {nbytes});"

    return nodes.Tasklet(node.name,
                         inputs={},
                         outputs={MemsetLibraryNode.OUTPUT_CONNECTOR_NAME: dace.dtypes.pointer(out.dtype)},
                         code=code,
                         language=dace.Language.CPP)


def select_memset_implementation(node: "MemsetLibraryNode", parent_state: dace.SDFGState) -> str:
    """Resolve an ``'Auto'`` ``MemsetLibraryNode`` implementation to ``'pure'``, ``'CUDA'``, or
    ``'CPU'``.

    ``'pure'``: device scope (no ``cudaMemsetAsync`` from a kernel), non-contiguous subsets, or a
    statically-large contiguous CPU zero (element map parallelizes across OpenMP at top level).
    ``'CUDA'``: host-issued GPU-destination contiguous memset. Else ``'CPU'`` (single
    ``std::memset``), including small/symbolic-size contiguous CPU zero.
    """
    _out_name, out, out_subset = node.validate(parent_state.sdfg, parent_state)

    if is_devicelevel_gpu(parent_state.sdfg, parent_state, node):
        if out_subset.num_elements_exact() == 1:
            return 'tasklet'
        return 'pure'

    if out_subset.num_elements_exact() == 1 and (out.storage in CPU_RESIDENT_STORAGES
                                                 or out.storage == dace.dtypes.StorageType.Register):
        return 'tasklet'

    if not out_subset.is_contiguous_subset(out):
        return 'pure'

    if out.storage == dace.dtypes.StorageType.GPU_Global:
        return 'CUDA'

    # CPU main-memory contiguous zero: only a size KNOWN at compile time to be large (static
    # count >= parallel_transfer_min_elements) takes the element map ('pure', OpenMP-parallel at
    # top level); small/symbolic size keeps a single memset ('CPU') -- no forking for a size that
    # may be tiny at runtime. Register / non-main-memory storages also stay serial.
    allowed = CPU_RESIDENT_STORAGES | {dace.dtypes.StorageType.Default}
    if out.storage in allowed and is_parallel_cpu_transfer_size(out_subset.num_elements()):
        return 'pure'
    return 'CPU'


@library.expansion
class ExpandAuto(ExpandTransformation):
    """Default expansion: dispatches via :func:`select_memset_implementation`."""
    environments = []

    @staticmethod
    def expansion(node: "MemsetLibraryNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG):
        return auto_dispatch(node, parent_state, select_memset_implementation, MemsetLibraryNode)


@library.expansion
class ExpandPure(ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node: "MemsetLibraryNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> dace.SDFG:
        sdfg, state, out_name, out, map_lengths = _make_memset_skeleton(node, parent_state)

        # Inner-tasklet connector -- must not collide with the wrapper SDFG's parameter array
        # (named after the libnode's outer connector).
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
        return _make_memset_tasklet(node, parent_state, cuda=True)


@library.expansion
class ExpandCPU(ExpandTransformation):
    environments = [environments.CPU]

    @staticmethod
    def expansion(node: "MemsetLibraryNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        return _make_memset_tasklet(node, parent_state, cuda=False)


@library.expansion
class ExpandTasklet(ExpandTransformation):
    """Single-element same-side scalar assignment"""
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        out_name, out, out_subset = node.validate(parent_sdfg, parent_state)
        out_volume = out_subset.num_elements_exact()
        if out_volume != 1:
            raise ValueError(f"Tasklet expansion requires single-element subsets "
                             f"(got output volume {out_volume}). "
                             f"Use MappedTasklet for multi-element copies.")

        # ``_out = 0`` is a valid device-side store inside a GPU kernel; from host scope a scalar
        # assignment can't write device memory -- route that case to the ``CUDA`` expansion.
        if (not is_devicelevel_gpu(parent_state.sdfg, parent_state, node) and out.storage in GPU_RESIDENT_STORAGES):
            raise ValueError(f"Tasklet expansion cannot zero GPU-resident storage ({out.storage}) for "
                             f"'{out_name}' from host scope; use the 'CUDA' Memset expansion instead.")

        return nodes.Tasklet(node.name,
                             inputs={},
                             outputs={MemsetLibraryNode.OUTPUT_CONNECTOR_NAME: out.dtype},
                             code=f"{MemsetLibraryNode.OUTPUT_CONNECTOR_NAME} = 0",
                             language=dace.Language.Python)


@library.node
class MemsetLibraryNode(nodes.LibraryNode):
    """Library node representing a 0-memset over a contiguous output subset.

    Does NOT accept dynamic (Scalar) input connectors: subset expressions must use symbols
    already in scope, so the auto selector reasons purely from the static memlet subset.
    """

    implementations = {
        "Auto": ExpandAuto,
        "pure": ExpandPure,
        "CUDA": ExpandCUDA,
        "CPU": ExpandCPU,
        "tasklet": ExpandTasklet
    }
    default_implementation = 'Auto'

    OUTPUT_CONNECTOR_NAME = "_mset_out"

    def __init__(self, name: str, *args, **kwargs):
        super().__init__(name, *args, outputs={MemsetLibraryNode.OUTPUT_CONNECTOR_NAME}, **kwargs)

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> Tuple[str, dace.data.Data, dace.subsets.Range]:
        """Validate wiring and resolve the output edge -> ``(out_name, out, out_subset)``. Raises
        if the node lacks exactly one output edge, or has a non-empty non-reserved input
        connector wired."""
        data_oes = [oe for oe in state.out_edges(self) if oe.src_conn == MemsetLibraryNode.OUTPUT_CONNECTOR_NAME]
        if len(data_oes) != 1:
            raise ValueError(f"{type(self).__name__} expects exactly one "
                             f"``{MemsetLibraryNode.OUTPUT_CONNECTOR_NAME}`` output edge.")

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
