# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Single-call host fill: ``memset`` when the value is byte-splat, else ``std::fill_n``."""
from typing import TYPE_CHECKING

import dace
from dace import library, nodes
from dace.codegen.common import sym2cpp
from dace.libraries.standard import environments
from dace.libraries.standard.nodes.fill.common import OUTPUT_CONNECTOR_NAME, byte_pattern, cpp_literal
from dace.transformation.transformation import ExpandTransformation

if TYPE_CHECKING:
    from dace.libraries.standard.nodes.fill.node import FillLibraryNode


@library.expansion
class ExpandCPU(ExpandTransformation):
    environments = [environments.CPU]

    @staticmethod
    def expansion(node: "FillLibraryNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        out_name, out, out_subset = node.validate(parent_state.sdfg, parent_state)
        if not out_subset.is_contiguous_subset(out):
            raise ValueError(f"FillLibraryNode CPU expansion requires a contiguous subset; got '{out_name}' "
                             f"subset {out_subset} on shape {tuple(out.shape)} strides {tuple(out.strides)}. "
                             f"Use the 'pure' expansion (mapped tasklet) for non-contiguous regions.")

        count = sym2cpp(out_subset.num_elements_exact())
        pattern = byte_pattern(node.value, out.dtype)
        if pattern is not None:
            code = f"memset({OUTPUT_CONNECTOR_NAME}, {pattern}, {count} * sizeof({out.dtype.ctype}));"
        else:
            code = f"std::fill_n({OUTPUT_CONNECTOR_NAME}, {count}, {cpp_literal(node.value, out.dtype)});"

        return nodes.Tasklet(node.name,
                             inputs={},
                             outputs={OUTPUT_CONNECTOR_NAME: dace.dtypes.pointer(out.dtype)},
                             code=code,
                             language=dace.Language.CPP)
