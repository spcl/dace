# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Strided host/device copies as one flat ``gpuMemcpyAsync``: of the whole span the region occupies (``MemcpyCUDAHull``),
or of a packed staging buffer (``MemcpyCUDAStaged``)."""

import functools
import operator

import dace
from dace import dtypes, library, nodes, symbolic
from dace.codegen.common import get_gpu_backend, sym2cpp
from dace.libraries.standard import environments
from dace.libraries.standard.helper import CURRENT_STREAM_NAME
from dace.libraries.standard.nodes.copy.common import (
    INPUT_CONNECTOR_NAME,
    OUTPUT_CONNECTOR_NAME,
    _make_expansion_sdfg,
    _memcpy_kind,
    hull_elements,
    span_is_bounded,
)
from dace.libraries.standard.nodes.copy.node import CopyLibraryNode
from dace.transformation.transformation import ExpandTransformation


@library.register_expansion(CopyLibraryNode, "MemcpyCUDAHull")
class ExpandMemcpyCUDAHull(ExpandTransformation):
    """One ``gpuMemcpyAsync`` of the span from the region's first element to its last. Both sides lay the region out
    alike, so every element lands in its own slot; the elements between the rows land in the destination too, which
    the selector allows only where nothing reads them (``hull_copy_is_safe``)."""

    environments = [environments.CUDA]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        ctx = _make_expansion_sdfg(node, parent_state, allow_cross_storage=True)
        span = hull_elements(ctx.in_shape_collapsed, ctx.sdfg.arrays[ctx.inp_name].strides)
        code = (
            f"DACE_GPU_CHECK({get_gpu_backend()}MemcpyAsync({OUTPUT_CONNECTOR_NAME}, {INPUT_CONNECTOR_NAME}, "
            f"{sym2cpp(span)} * sizeof({ctx.inp.dtype.ctype}), {_memcpy_kind(ctx.inp, ctx.out)}, {CURRENT_STREAM_NAME}));"
        )
        return nodes.Tasklet(
            node.name,
            inputs={INPUT_CONNECTOR_NAME: dtypes.pointer(ctx.inp.dtype)},
            outputs={OUTPUT_CONNECTOR_NAME: dtypes.pointer(ctx.out.dtype)},
            code=code,
            language=dace.Language.CPP,
        )


@library.register_expansion(CopyLibraryNode, "MemcpyCUDAStaged")
class ExpandMemcpyCUDAStaged(ExpandTransformation):
    """A strided host/device copy as one ``gpuMemcpyAsync``, with the filtering done on the GPU where it can be.

    Host to device:

    1. **transfer** -- one tasklet moves the source to a device buffer and synchronizes the stream: the whole span the
       region occupies when that is at most twice the region (the span is contiguous, and only read), else the region
       packed by a host map first;
    2. **unpack** -- a device map copies the region out of the buffer into the destination.

    Device to host:

    1. **pack** -- a device map copies the region into a packed device buffer;
    2. **transfer** -- one tasklet moves exactly the region's bytes and synchronizes the stream;
    3. **unpack** -- a host map copies them into the destination, if its region is strided. Moving the span instead
       would overwrite the destination between the rows (``MemcpyCUDAHull`` does that only where nothing reads it).

    A side whose region is packed already is transferred directly. The buffers are allocated once for the program
    (``Persistent``: in ``__dace_init``, freed in ``__dace_exit``); a size that names a symbol only the copy's scope
    defines is known only per copy, so that buffer lives for the copy's scope."""

    environments = [environments.CUDA]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        ctx = _make_expansion_sdfg(node, parent_state, allow_cross_storage=True)
        sdfg = ctx.sdfg
        shape = ctx.in_shape_collapsed
        src_desc, dst_desc = sdfg.arrays[ctx.inp_name], sdfg.arrays[ctx.out_name]
        to_device = dst_desc.storage == dtypes.StorageType.GPU_Global
        # The program's own symbols, when the copy runs at its top level: what ``__dace_init`` can size a buffer by
        top_level = parent_sdfg.parent_sdfg is None and parent_state.entry_node(node) is None
        names = set().union(*(symbolic.free_symbols_and_functions(n) for n in shape))
        persistent = top_level and names <= set(parent_sdfg.free_symbols)
        lifetime = dtypes.AllocationLifetime.Persistent if persistent else dtypes.AllocationLifetime.Scope

        def buffer(name: str, storage: dtypes.StorageType, strides=None, total_size=None) -> str:
            """A transient beside ``name``: packed, or laid out with ``strides`` over ``total_size`` elements."""
            buffer_name, _ = sdfg.add_array(
                f"__cpy_stage_{name}",
                shape,
                sdfg.arrays[name].dtype,
                storage,
                strides=strides,
                total_size=total_size,
                transient=True,
                find_new_name=True,
                lifetime=lifetime,
            )
            return buffer_name

        def element_copy(state: dace.SDFGState, src: str, dst: str, label: str) -> None:
            """``dst[i] = src[i]`` over the region, on the side both live on."""
            on_device = sdfg.arrays[src].storage == dtypes.StorageType.GPU_Global
            params = [f"__cpy_i{d}" for d in range(len(shape))]
            index = ", ".join(params)
            state.add_mapped_tasklet(
                f"{node.label}_{label}",
                {p: (0, n - 1, 1) for p, n in zip(params, shape)},
                {"_in": dace.Memlet(f"{src}[{index}]")},
                "_out = _in",
                {"_out": dace.Memlet(f"{dst}[{index}]")},
                # Explicit on the host too: a Default map nested under a GPU schedule (the cast-on-copy staging)
                # would inherit it and run as a kernel over host buffers
                schedule=dtypes.ScheduleType.GPU_Device if on_device else dtypes.ScheduleType.CPU_Multicore,
                external_edges=True,
            )

        state = ctx.state
        count = functools.reduce(operator.mul, shape, 1)
        sent, received = ctx.inp_name, ctx.out_name
        if to_device:
            if not src_desc.is_packed_c_strides():
                if span_is_bounded(shape, src_desc.strides, parent_sdfg.facts()):
                    # The span, read as is: the device buffer keeps the source's layout
                    count = hull_elements(shape, src_desc.strides)
                    received = buffer(ctx.out_name, dst_desc.storage, src_desc.strides, count)
                else:
                    sent = buffer(ctx.inp_name, src_desc.storage)
                    element_copy(state, ctx.inp_name, sent, "pack")
                    state = sdfg.add_state_after(state, f"{node.label}_transfer")
            if received == ctx.out_name and not dst_desc.is_packed_c_strides():
                received = buffer(ctx.out_name, dst_desc.storage)
        else:
            if not src_desc.is_packed_c_strides():
                sent = buffer(ctx.inp_name, src_desc.storage)
                element_copy(state, ctx.inp_name, sent, "pack")
                state = sdfg.add_state_after(state, f"{node.label}_transfer")
            if not dst_desc.is_packed_c_strides():
                received = buffer(ctx.out_name, dst_desc.storage)

        backend = get_gpu_backend()
        tasklet = state.add_tasklet(
            f"{node.label}_transfer",
            {"_in": dtypes.pointer(ctx.inp.dtype)},
            {"_out": dtypes.pointer(ctx.out.dtype)},
            f"DACE_GPU_CHECK({backend}MemcpyAsync(_out, _in, {sym2cpp(count)} * sizeof({ctx.inp.dtype.ctype}), "
            f"{_memcpy_kind(ctx.inp, ctx.out)}, {CURRENT_STREAM_NAME}));\n"
            f"DACE_GPU_CHECK({backend}StreamSynchronize({CURRENT_STREAM_NAME}));",
            language=dace.Language.CPP,
        )
        state.add_edge(state.add_read(sent), None, tasklet, "_in", sdfg.make_array_memlet(sent))
        state.add_edge(tasklet, "_out", state.add_write(received), None, sdfg.make_array_memlet(received))

        if received != ctx.out_name:
            element_copy(sdfg.add_state_after(state, f"{node.label}_unpack"), received, ctx.out_name, "unpack")
        return sdfg
