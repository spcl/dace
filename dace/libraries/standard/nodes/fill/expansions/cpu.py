# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Single-call host fill through ``std::fill_n``."""

from typing import TYPE_CHECKING

import dace
from dace import library, nodes
from dace.codegen.common import sym2cpp
from dace.libraries.standard import environments
from dace.libraries.standard.nodes.fill.common import OUTPUT_CONNECTOR_NAME, VALUE_CONNECTOR_NAME, cpp_literal
from dace.libraries.standard.nodes.fill.node import FillLibraryNode
from dace.transformation.transformation import ExpandTransformation

if TYPE_CHECKING:
    pass


@library.register_expansion(FillLibraryNode, 'CPU')
class ExpandCPU(ExpandTransformation):
    environments = [environments.CPU]

    @staticmethod
    def expansion(node: "FillLibraryNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        out_name, out, out_subset = node.validate(parent_state.sdfg, parent_state)
        if not out_subset.is_contiguous_subset(out):
            raise ValueError(
                f"FillLibraryNode CPU expansion requires a contiguous subset; got '{out_name}' "
                f"subset {out_subset} on shape {tuple(out.shape)} strides {tuple(out.strides)}. "
                f"Use the 'pure' expansion (mapped tasklet) for non-contiguous regions."
            )

        value_info = node.value_descriptor(parent_state)
        if value_info is not None:
            # Dynamic scalar value supplied through the input connector.
            inputs = {VALUE_CONNECTOR_NAME: value_info[1].dtype}
            value_expr = VALUE_CONNECTOR_NAME
        else:
            # Both gcc and clang turn this into a memset at -O2 and above whenever the value's object
            # representation allows it, including through the one-byte fp8 wrappers, so spelling the
            # memset out here would only repeat what the build already does at the Release level dace
            # always compiles with.
            inputs = {}
            value_expr = cpp_literal(node.value, out.dtype)

        count = sym2cpp(out_subset.num_elements_exact())
        code = f"std::fill_n({OUTPUT_CONNECTOR_NAME}, {count}, {value_expr});"

        return nodes.Tasklet(
            node.name,
            inputs=inputs,
            outputs={OUTPUT_CONNECTOR_NAME: dace.dtypes.pointer(out.dtype)},
            code=code,
            language=dace.Language.CPP,
        )
