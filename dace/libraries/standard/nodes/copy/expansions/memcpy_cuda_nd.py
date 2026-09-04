# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Strided N-D device copy, issued as a loop of contiguous ``gpuMemcpyAsync`` calls.
"""
from typing import TYPE_CHECKING

import dace
from dace import library, nodes, subsets, symbolic
from dace.codegen.common import sym2cpp, get_gpu_backend
from dace.libraries.standard import environments
from dace.libraries.standard.helper import (CURRENT_STREAM_NAME, collapse_shape_and_strides)
from dace.libraries.standard.nodes.copy.node import CopyLibraryNode
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.standard.nodes.copy.common import (_make_expansion_sdfg, _memcpy_kind, INPUT_CONNECTOR_NAME,
                                                       OUTPUT_CONNECTOR_NAME)

if TYPE_CHECKING:
    pass


@library.register_expansion(CopyLibraryNode, 'MemcpyCUDANDStrided')
class ExpandMemcpyCUDANDStrided(ExpandTransformation):
    """Fallback for >=3D-strided cross-boundary copies that can't collapse to one
    ``gpuMemcpyAsync`` / ``cudaMemcpy2DAsync``: a Sequential map issuing one ``gpuMemcpyAsync``
    per row over every collapsed dim except the chunk axis (``stride == 1`` both sides)."""
    environments = [environments.CUDA]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        inp_name, inp, in_subset, out_name, out, out_subset = node.validate(parent_sdfg,
                                                                            parent_state,
                                                                            allow_cross_storage=True)
        in_shape_collapsed, in_strides_collapsed = collapse_shape_and_strides(in_subset, inp.strides)
        out_shape_collapsed, out_strides_collapsed = collapse_shape_and_strides(out_subset, out.strides)

        if len(in_shape_collapsed) != len(out_shape_collapsed):
            raise NotImplementedError("ExpandCUDANDStrided requires src and dst to share the collapsed rank "
                                      f"(got {in_shape_collapsed} vs {out_shape_collapsed}).")
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
            raise NotImplementedError("ExpandCUDANDStrided requires at least one common stride-1 axis on both sides "
                                      f"(got src_strides={in_strides_collapsed}, dst_strides={out_strides_collapsed}).")

        ctype = inp.dtype.ctype
        chunk = sym2cpp(in_shape_collapsed[chunk_dim])
        kind = _memcpy_kind(inp, out)
        backend = get_gpu_backend()

        if ndims == 1:
            code = (f"DACE_GPU_CHECK({backend}MemcpyAsync({OUTPUT_CONNECTOR_NAME}, {INPUT_CONNECTOR_NAME}, "
                    f"{chunk} * sizeof({ctype}), {kind}, {CURRENT_STREAM_NAME}));")
            in_conns = {INPUT_CONNECTOR_NAME: dace.dtypes.pointer(inp.dtype)}
            return nodes.Tasklet(node.name,
                                 inputs=in_conns,
                                 outputs={OUTPUT_CONNECTOR_NAME: dace.dtypes.pointer(out.dtype)},
                                 code=code,
                                 language=dace.Language.CPP)

        ctx = _make_expansion_sdfg(node, parent_state, allow_cross_storage=True)
        map_axes = [d for d in range(ndims) if d != chunk_dim]
        map_params = [f"__cpy_i{d}" for d in map_axes]
        # Symbolic bounds, never a rendered string: ``sym2cpp`` spells a symbolic extent as C++
        # (``dace::math::ipow(R, K)``), and the range parser splits on ':', so the qualified name
        # comes back as four bogus tokens.
        map_ranges = {p: (0, ctx.in_shape_collapsed[d] - 1, 1) for d, p in zip(map_axes, map_params)}

        def _row_subset(shape):
            parts = []
            map_pi = 0
            for d in range(ndims):
                if d == chunk_dim:
                    parts.append((0, shape[d] - 1, 1))
                else:
                    p = symbolic.symbol(map_params[map_pi])
                    parts.append((p, p, 1))
                    map_pi += 1
            return subsets.Range(parts)

        in_memlet = dace.memlet.Memlet(data=ctx.inp_name, subset=_row_subset(ctx.in_shape_collapsed))
        out_memlet = dace.memlet.Memlet(data=ctx.out_name, subset=_row_subset(ctx.out_shape_collapsed))
        inner_in, inner_out = "_in", "_out"
        backend = get_gpu_backend()
        code = (f"DACE_GPU_CHECK({backend}MemcpyAsync({inner_out}, {inner_in}, "
                f"{chunk} * sizeof({ctype}), {kind}, {CURRENT_STREAM_NAME}));")

        inner_tasklet, map_entry, _map_exit = ctx.state.add_mapped_tasklet(name=f"{node.label}_tasklet",
                                                                           map_ranges=map_ranges,
                                                                           inputs={inner_in: in_memlet},
                                                                           code=code,
                                                                           outputs={inner_out: out_memlet},
                                                                           schedule=dace.dtypes.ScheduleType.Sequential,
                                                                           language=dace.Language.CPP,
                                                                           external_edges=True)
        # Force pointer connectors so codegen types them T*, matching gpuMemcpyAsync's signature.
        inner_tasklet.in_connectors[inner_in] = dace.dtypes.pointer(inp.dtype)
        inner_tasklet.out_connectors[inner_out] = dace.dtypes.pointer(out.dtype)

        return ctx.sdfg
