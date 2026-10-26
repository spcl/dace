# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Copy through the Snitch cluster DMA engine."""

import dace
from dace import dtypes, library, nodes
from dace.codegen.common import sym2cpp
from dace.libraries.standard.helper import collapse_shape_and_strides
from dace.libraries.standard.nodes.copy.common import INPUT_CONNECTOR_NAME, OUTPUT_CONNECTOR_NAME
from dace.libraries.standard.nodes.copy.node import CopyLibraryNode
from dace.transformation.transformation import ExpandTransformation


@library.register_expansion(CopyLibraryNode, "MemcpySnitchDMA")
class ExpandMemcpySnitchDMA(ExpandTransformation):
    """One ``__builtin_sdma_start_oned`` for a contiguous region, ``__builtin_sdma_start_twod`` for rows of
    contiguous elements (a strided 1-D region is rows of one element), then a wait for the engine."""

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        _, inp, in_subset, _, out, out_subset = node.validate(parent_sdfg, parent_state, allow_cross_storage=True)
        shape, src = collapse_shape_and_strides(in_subset, inp.strides)
        _, dst = collapse_shape_and_strides(out_subset, out.strides)
        elem = f"sizeof({inp.dtype.ctype})"
        ptrs = f"(uint64_t)({INPUT_CONNECTOR_NAME}), (uint64_t)({OUTPUT_CONNECTOR_NAME})"
        if len(shape) == 1 and src[0] == 1 and dst[0] == 1:
            xfer = f"__builtin_sdma_start_oned({ptrs}, {elem} * ({sym2cpp(shape[0])}), 0);"
        elif len(shape) == 1 or (len(shape) == 2 and src[1] == 1 and dst[1] == 1):
            row = sym2cpp(shape[1]) if len(shape) == 2 else "1"
            xfer = (
                f"__builtin_sdma_start_twod({ptrs}, {elem} * ({row}), {elem} * ({sym2cpp(src[0])}), "
                f"{elem} * ({sym2cpp(dst[0])}), {sym2cpp(shape[0])}, 0);"
            )
        else:
            raise NotImplementedError(f"Snitch DMA copies one or two dimensions with contiguous rows; got {shape}")
        return nodes.Tasklet(
            node.name,
            inputs={INPUT_CONNECTOR_NAME: dtypes.pointer(inp.dtype)},
            outputs={OUTPUT_CONNECTOR_NAME: dtypes.pointer(out.dtype)},
            code=f"{xfer}\n__builtin_sdma_wait_for_idle();",
            language=dace.Language.CPP,
        )
