# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pitched 2-D device copy through ``cudaMemcpy2DAsync``.
"""
from typing import TYPE_CHECKING

import dace
from dace import library, nodes
from dace.codegen.common import sym2cpp, get_gpu_backend
from dace.libraries.standard import environments
from dace.libraries.standard.helper import (CURRENT_STREAM_NAME, collapse_shape_and_strides)
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.standard.nodes.copy.common import (_memcpy_kind, cuda2d_pitch_params, INPUT_CONNECTOR_NAME,
                                                       OUTPUT_CONNECTOR_NAME)

if TYPE_CHECKING:
    pass


@library.expansion
class ExpandMemcpyCUDA2D(ExpandTransformation):
    """2D strided copy via ``cudaMemcpy2DAsync`` between any GPU_Global/host storage combination:
    row-major contiguous rows, column-major contiguous columns, or outer stride a multiple of
    inner (degenerate)."""
    environments = [environments.CUDA]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        inp_name, inp, in_subset, out_name, out, out_subset = node.validate(parent_sdfg,
                                                                            parent_state,
                                                                            allow_cross_storage=True)

        in_shape_collapsed, in_strides_collapsed = collapse_shape_and_strides(in_subset, inp.strides)
        out_shape_collapsed, out_strides_collapsed = collapse_shape_and_strides(out_subset, out.strides)

        # 1D-collapsed shapes promote to (N, 1) so one cudaMemcpy2D call covers strided 1D patterns.
        if len(in_shape_collapsed) == 1 and len(out_shape_collapsed) == 1:
            in_shape_2d = [in_shape_collapsed[0], 1]
            out_shape_2d = [out_shape_collapsed[0], 1]
            in_strides_2d = [in_strides_collapsed[0], 1]
            out_strides_2d = [out_strides_collapsed[0], 1]
        elif len(in_shape_collapsed) == 2 and len(out_shape_collapsed) == 2:
            in_shape_2d = in_shape_collapsed
            out_shape_2d = out_shape_collapsed
            in_strides_2d = in_strides_collapsed
            out_strides_2d = out_strides_collapsed
        else:
            raise ValueError("MemcpyCUDA2D requires 1D or 2D collapsed shapes, got "
                             f"{in_shape_collapsed} (src) / {out_shape_collapsed} (dst).")

        kind = _memcpy_kind(inp, out)

        copy_shape = in_shape_2d
        src_strides = in_strides_2d
        dst_strides = out_strides_2d
        ctype = inp.dtype.ctype
        backend = get_gpu_backend()

        pitch = cuda2d_pitch_params(copy_shape, src_strides, dst_strides)
        if pitch is None:
            raise NotImplementedError(f"Unsupported 2D memory copy: shape={copy_shape}, "
                                      f"src_strides={src_strides}, dst_strides={dst_strides}.")
        dpitch_elems, spitch_elems, width_elems, height_elems = pitch
        dpitch = f"{sym2cpp(dpitch_elems)} * sizeof({ctype})"
        spitch = f"{sym2cpp(spitch_elems)} * sizeof({ctype})"
        width = f"{sym2cpp(width_elems)} * sizeof({ctype})"
        height = sym2cpp(height_elems)

        code = (f"{backend}Memcpy2DAsync({OUTPUT_CONNECTOR_NAME}, {dpitch}, {INPUT_CONNECTOR_NAME}, {spitch}, "
                f"{width}, {height}, {kind}, {CURRENT_STREAM_NAME});")

        in_conns = {INPUT_CONNECTOR_NAME: dace.dtypes.pointer(inp.dtype)}
        tasklet = nodes.Tasklet(node.name,
                                inputs=in_conns,
                                outputs={OUTPUT_CONNECTOR_NAME: dace.dtypes.pointer(out.dtype)},
                                code=code,
                                language=dace.Language.CPP)
        return tasklet
