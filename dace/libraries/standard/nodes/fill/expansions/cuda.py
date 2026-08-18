# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Host-issued ``cudaMemsetAsync`` over GPU memory. Byte-splat values only."""
from typing import TYPE_CHECKING

import dace
from dace import library, nodes
from dace.codegen.common import sym2cpp
from dace.libraries.standard import environments
from dace.libraries.standard.helper import CURRENT_STREAM_NAME
from dace.libraries.standard.nodes.fill.common import OUTPUT_CONNECTOR_NAME, byte_pattern
from dace.transformation.transformation import ExpandTransformation

if TYPE_CHECKING:
    from dace.libraries.standard.nodes.fill.node import FillLibraryNode


@library.expansion
class ExpandCUDA(ExpandTransformation):
    environments = [environments.CUDA]

    @staticmethod
    def expansion(node: "FillLibraryNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        out_name, out, out_subset = node.validate(parent_state.sdfg, parent_state)
        if not out_subset.is_contiguous_subset(out):
            raise ValueError(f"FillLibraryNode CUDA expansion requires a contiguous subset; got '{out_name}' "
                             f"subset {out_subset} on shape {tuple(out.shape)} strides {tuple(out.strides)}. "
                             f"Use the 'pure' expansion (mapped tasklet) for non-contiguous regions.")

        pattern = byte_pattern(node.value, out.dtype)
        if pattern is None:
            raise ValueError(f"FillLibraryNode CUDA expansion requires a byte-splat value; {node.value!r} as "
                             f"{out.dtype} is not one. Use the 'pure' expansion (mapped tasklet).")

        nbytes = f"{sym2cpp(out_subset.num_elements_exact())} * sizeof({out.dtype.ctype})"
        code = f"cudaMemsetAsync({OUTPUT_CONNECTOR_NAME}, {pattern}, {nbytes}, {CURRENT_STREAM_NAME});"

        return nodes.Tasklet(node.name,
                             inputs={},
                             outputs={OUTPUT_CONNECTOR_NAME: dace.dtypes.pointer(out.dtype)},
                             code=code,
                             language=dace.Language.CPP)
