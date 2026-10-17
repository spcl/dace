# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Strided N-D device copy, issued as a loop of contiguous ``gpuMemcpyAsync`` calls."""

from typing import TYPE_CHECKING

import dace
from dace import library, nodes, subsets, symbolic
from dace.codegen.common import get_gpu_backend, sym2cpp
from dace.libraries.standard import environments
from dace.libraries.standard.helper import CURRENT_STREAM_NAME, collapse_shape_and_strides
from dace.libraries.standard.nodes.copy.common import (
    INPUT_CONNECTOR_NAME,
    OUTPUT_CONNECTOR_NAME,
    _make_expansion_sdfg,
    _memcpy_kind,
)
from dace.libraries.standard.nodes.copy.node import CopyLibraryNode
from dace.transformation.transformation import ExpandTransformation

if TYPE_CHECKING:
    pass


@library.register_expansion(CopyLibraryNode, "MemcpyCUDANDStrided")
class ExpandMemcpyCUDANDStrided(ExpandTransformation):
    """Fallback for >=3D-strided cross-boundary copies that can't collapse to one
    ``gpuMemcpyAsync`` / ``cudaMemcpy2DAsync``: a Sequential map issuing one ``gpuMemcpyAsync``
    per row over every collapsed dim except the chunk axis (``stride == 1`` both sides)."""

    environments = [environments.CUDA]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        inp_name, inp, in_subset, out_name, out, out_subset = node.validate(
            parent_sdfg, parent_state, allow_cross_storage=True
        )
        in_shape_collapsed, in_strides_collapsed = collapse_shape_and_strides(in_subset, inp.strides)
        out_shape_collapsed, out_strides_collapsed = collapse_shape_and_strides(out_subset, out.strides)

        if len(in_shape_collapsed) != len(out_shape_collapsed):
            raise NotImplementedError(
                "ExpandCUDANDStrided requires src and dst to share the collapsed rank "
                f"(got {in_shape_collapsed} vs {out_shape_collapsed})."
            )
        ndims = len(in_shape_collapsed)
        if ndims < 1:
            raise NotImplementedError("ExpandCUDANDStrided requires at least one collapsed dimension.")

        # Chunk axis: innermost dim with stride 1 on both sides.
        chunk_dim = None
        for d in reversed(range(ndims)):
            if in_strides_collapsed[d] == 1 and out_strides_collapsed[d] == 1:
                chunk_dim = d
                break
        if chunk_dim is None:
            raise NotImplementedError(
                "ExpandCUDANDStrided requires at least one common stride-1 axis on both sides "
                f"(got src_strides={in_strides_collapsed}, dst_strides={out_strides_collapsed})."
            )

        ctype = inp.dtype.ctype
        chunk = sym2cpp(in_shape_collapsed[chunk_dim])
        kind = _memcpy_kind(inp, out)
        backend = get_gpu_backend()

        if ndims == 1:
            code = (
                f"DACE_GPU_CHECK({backend}MemcpyAsync({OUTPUT_CONNECTOR_NAME}, {INPUT_CONNECTOR_NAME}, "
                f"{chunk} * sizeof({ctype}), {kind}, {CURRENT_STREAM_NAME}));"
            )
            in_conns = {INPUT_CONNECTOR_NAME: dace.dtypes.pointer(inp.dtype)}
            return nodes.Tasklet(
                node.name,
                inputs=in_conns,
                outputs={OUTPUT_CONNECTOR_NAME: dace.dtypes.pointer(out.dtype)},
                code=code,
                language=dace.Language.CPP,
            )

        ctx = _make_expansion_sdfg(node, parent_state, allow_cross_storage=True)
        # The innermost axes go as one pitched 2D/3D copy where the layout allows, so the map only walks the
        # outer axes: a 3D interior is one call, not one per row. HIP refuses a pitched copy out of pooled
        # memory whose rows end exactly at the pitch, so a pooled operand keeps one gpuMemcpyAsync per row.
        pooled = backend == "hip" and (getattr(inp, "pool", False) or getattr(out, "pool", False))
        pitched = 0 if pooled or chunk_dim != ndims - 1 else _pitched_axes(in_strides_collapsed, out_strides_collapsed)
        if pitched:
            map_axes = list(range(ndims - pitched))
        else:
            map_axes = [d for d in range(ndims) if d != chunk_dim]
        map_params = [f"__cpy_i{d}" for d in map_axes]
        # Symbolic bounds, never a rendered string: ``sym2cpp`` spells a symbolic extent as C++
        # (``dace::math::ipow(R, K)``), and the range parser splits on ':', so the qualified name
        # comes back as four bogus tokens.
        map_ranges = {p: (0, ctx.in_shape_collapsed[d] - 1, 1) for d, p in zip(map_axes, map_params)}

        def _row_subset(shape):
            parts = []
            for d in range(ndims):
                if d in map_axes:
                    p = symbolic.symbol(map_params[map_axes.index(d)])
                    parts.append((p, p, 1))
                else:
                    parts.append((0, shape[d] - 1, 1))
            return subsets.Range(parts)

        in_memlet = dace.memlet.Memlet(data=ctx.inp_name, subset=_row_subset(ctx.in_shape_collapsed))
        out_memlet = dace.memlet.Memlet(data=ctx.out_name, subset=_row_subset(ctx.out_shape_collapsed))
        inner_in, inner_out = "_in", "_out"
        if pitched:
            code = _pitched_copy_code(
                backend,
                inner_out,
                inner_in,
                in_shape_collapsed[ndims - pitched :],
                in_strides_collapsed[ndims - pitched :],
                out_strides_collapsed[ndims - pitched :],
                ctype,
                kind,
            )
        else:
            code = (
                f"DACE_GPU_CHECK({backend}MemcpyAsync({inner_out}, {inner_in}, "
                f"{chunk} * sizeof({ctype}), {kind}, {CURRENT_STREAM_NAME}));"
            )
        if not map_axes:
            tasklet = ctx.state.add_tasklet(
                f"{node.label}_tasklet",
                {inner_in: dace.dtypes.pointer(inp.dtype)},
                {inner_out: dace.dtypes.pointer(out.dtype)},
                code,
                language=dace.Language.CPP,
            )
            ctx.state.add_edge(ctx.state.add_read(ctx.inp_name), None, tasklet, inner_in, in_memlet)
            ctx.state.add_edge(tasklet, inner_out, ctx.state.add_write(ctx.out_name), None, out_memlet)
            return ctx.sdfg

        inner_tasklet, map_entry, _map_exit = ctx.state.add_mapped_tasklet(
            name=f"{node.label}_tasklet",
            map_ranges=map_ranges,
            inputs={inner_in: in_memlet},
            code=code,
            outputs={inner_out: out_memlet},
            schedule=dace.dtypes.ScheduleType.Sequential,
            language=dace.Language.CPP,
            external_edges=True,
        )
        # Force pointer connectors so codegen types them T*, matching gpuMemcpyAsync's signature.
        inner_tasklet.in_connectors[inner_in] = dace.dtypes.pointer(inp.dtype)
        inner_tasklet.out_connectors[inner_out] = dace.dtypes.pointer(out.dtype)

        return ctx.sdfg


def _divides(outer, inner) -> bool:
    """Whether stride ``outer`` is a whole multiple of stride ``inner``, provably."""
    if inner == 0:
        return False
    _, denominator = symbolic.simplify(outer / inner).as_numer_denom()
    return denominator == 1


def _pitched_axes(src_strides, dst_strides) -> int:
    """How many innermost axes (stride 1 last on both sides) one pitched copy covers: 3 when each side's
    slice stride is a whole number of its rows, else 2."""
    if len(src_strides) >= 3 and all(_divides(st[-3], st[-2]) for st in (src_strides, dst_strides)):
        return 3
    return 2


def _pitched_copy_code(backend, dst, src, shape, src_strides, dst_strides, ctype, kind) -> str:
    """One checked ``Memcpy2DAsync`` (two axes) or ``Memcpy3DAsync`` (three axes) for a box of ``shape``."""
    width = f"{sym2cpp(shape[-1])} * sizeof({ctype})"
    if len(shape) == 2:
        return (
            f"DACE_GPU_CHECK({backend}Memcpy2DAsync({dst}, {sym2cpp(dst_strides[0])} * sizeof({ctype}), {src}, "
            f"{sym2cpp(src_strides[0])} * sizeof({ctype}), {width}, {sym2cpp(shape[0])}, {kind}, {CURRENT_STREAM_NAME}));"
        )
    rows = {
        side: sym2cpp(symbolic.simplify(st[0] / st[1])) for side, st in (("src", src_strides), ("dst", dst_strides))
    }
    return (
        f"{backend}Memcpy3DParms __cpy_parms = {{}};\n"
        f"__cpy_parms.srcPtr = make_{backend}PitchedPtr((void*){src}, {sym2cpp(src_strides[1])} * sizeof({ctype}), "
        f"{sym2cpp(shape[-1])}, {rows['src']});\n"
        f"__cpy_parms.dstPtr = make_{backend}PitchedPtr((void*){dst}, {sym2cpp(dst_strides[1])} * sizeof({ctype}), "
        f"{sym2cpp(shape[-1])}, {rows['dst']});\n"
        f"__cpy_parms.extent = make_{backend}Extent({width}, {sym2cpp(shape[1])}, {sym2cpp(shape[0])});\n"
        f"__cpy_parms.kind = {kind};\n"
        f"DACE_GPU_CHECK({backend}Memcpy3DAsync(&__cpy_parms, {CURRENT_STREAM_NAME}));"
    )
