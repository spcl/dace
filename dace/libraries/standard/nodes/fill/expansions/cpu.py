# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Single-call host fill: ``std::fill_n`` in C++, ``memset`` or a loop in C."""
from typing import TYPE_CHECKING

import dace
from dace import library, nodes
from dace.codegen.common import sym2cpp
from dace.libraries.standard import environments
from dace.libraries.standard.nodes.fill.common import (OUTPUT_CONNECTOR_NAME, VALUE_CONNECTOR_NAME, byte_pattern,
                                                       c_literal, cpp_literal, memset_is_exact)
from dace.libraries.standard.nodes.fill.node import FillLibraryNode
from dace.transformation.transformation import ExpandTransformation

if TYPE_CHECKING:
    pass

#: The C fill loop's induction variable. Declared inside the tasklet's own braces, so nesting one
#: fill inside another is legal and neither reaches the surrounding scope.
C_FILL_INDEX = '__fill_i'


def c_memset(count: str, splat: int) -> str:
    """The fill as one ``memset``, for a value :func:`memset_is_exact` accepts.

    :param count: The element count, already printed.
    :param splat: The byte every element's object representation repeats.
    :returns: The tasklet body.
    """
    return f"memset({OUTPUT_CONNECTOR_NAME}, {splat}, ({count}) * sizeof({OUTPUT_CONNECTOR_NAME}[0]));"


def c_loop(count: str, value_expr: str) -> str:
    """The fill as a loop over the elements, which expresses any value of any type.

    :param count: The element count, already printed.
    :param value_expr: The fill value, already printed as a C expression.
    :returns: The tasklet body.
    """
    return (f"for (long long {C_FILL_INDEX} = 0; {C_FILL_INDEX} < ({count}); ++{C_FILL_INDEX}) "
            f"{{ {OUTPUT_CONNECTOR_NAME}[{C_FILL_INDEX}] = {value_expr}; }}")


@library.register_expansion(FillLibraryNode, 'CPU')
class ExpandCPU(ExpandTransformation):
    environments = [environments.CPU]

    @staticmethod
    def expansion(node: "FillLibraryNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        # Asked at EXPANSION time because a tasklet's text is fixed once it is built, and C has no
        # ``std::fill_n`` to call. Outside a standalone-C rendering this is the C++ spelling, which
        # is the behaviour every other caller has.
        from dace import cpf_lowering

        out_name, out, out_subset = node.validate(parent_state.sdfg, parent_state)
        if not out_subset.is_contiguous_subset(out):
            raise ValueError(f"FillLibraryNode CPU expansion requires a contiguous subset; got '{out_name}' "
                             f"subset {out_subset} on shape {tuple(out.shape)} strides {tuple(out.strides)}. "
                             f"Use the 'pure' expansion (mapped tasklet) for non-contiguous regions.")

        value_info = node.value_descriptor(parent_state)
        # Dynamic scalar value supplied through the input connector.
        inputs = {} if value_info is None else {VALUE_CONNECTOR_NAME: value_info[1].dtype}
        count = sym2cpp(out_subset.num_elements_exact())

        if not cpf_lowering.standalone_c():
            # Both gcc and clang turn this into a memset at -O2 and above whenever the value's object
            # representation allows it, including through the one-byte fp8 wrappers, so spelling the
            # memset out here would only repeat what the build already does at the Release level dace
            # always compiles with.
            value_expr = VALUE_CONNECTOR_NAME if value_info is not None else cpp_literal(node.value, out.dtype)
            code = f"std::fill_n({OUTPUT_CONNECTOR_NAME}, {count}, {value_expr});"
        elif value_info is None and memset_is_exact(node.value, out.dtype):
            # C has no type-generic fill call, and libstdc++ is what turns the C++ one into a
            # memset. Spelling the memset is therefore the C dialect's own decision, taken only
            # where the value's object representation makes it exact.
            code = c_memset(count, byte_pattern(node.value, out.dtype))
        else:
            # A dynamic value's representation is unknown here, so it can never be shown
            # byte-splat and the loop is the only exact form for it.
            value_expr = VALUE_CONNECTOR_NAME if value_info is not None else c_literal(node.value, out.dtype)
            code = c_loop(count, value_expr)

        return nodes.Tasklet(node.name,
                             inputs=inputs,
                             outputs={OUTPUT_CONNECTOR_NAME: dace.dtypes.pointer(out.dtype)},
                             code=code,
                             language=dace.Language.CPP)
