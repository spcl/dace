# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.data import Array
from dace.properties import Property, make_properties
from dace.libraries.standard.nodes import CodeLibraryNode
from dace.codegen.targets.cpp import cpp_offset_expr
import numpy as np
from typing import Dict


@make_properties
class MyNode(CodeLibraryNode):
    value_to_add = Property(dtype=int,
                            default=5,
                            desc="Value to add in custom code")

    def __init__(self):
        super().__init__(input_names=['inp'], output_names=['out'])

    def generate_code(self, inputs: Dict[str, Array], outputs: Dict[str,
                                                                    Array]):
        assert len(inputs) == 1
        assert len(outputs) == 1
        inarr = inputs['inp']
        outarr = outputs['out']
        assert len(inarr.shape) == len(outarr.shape)

        # Construct for loops
        code = ''
        for dim, shp in enumerate(inarr.shape):
            code += f'for (int i{dim} = 0; i{dim} < {shp}; ++i{dim}) {{\n'

        # Construct index expressions
        output_expr = ' + '.join(f'i{dim} * {stride}'
                                 for dim, stride in enumerate(outarr.strides))
        input_expr = ' + '.join(f'i{dim} * {stride}'
                                for dim, stride in enumerate(inarr.strides))

        code += \
            f'out[{output_expr}] = inp[{input_expr}] + {self.value_to_add};\n'

        # End for loops
        for dim in range(len(inarr.shape)):
            code += '}\n'

        return code


def test_custom_code_node():
    # Construct graph
    sdfg = dace.SDFG('custom_code')
    sdfg.add_array('A', [20, 30], dace.float64)
    sdfg.add_array('B', [20, 30], dace.float64)
    state = sdfg.add_state()

    a = state.add_read('A')
    node = MyNode()
    b = state.add_write('B')
    state.add_edge(a, None, node, 'inp',
                   dace.Memlet.simple('A', '10:20, 10:30'))
    state.add_edge(node, 'out', b, None, dace.Memlet.simple('B', '0:10, 0:20'))

    # Run graph with default node value
    A = np.random.rand(20, 30)
    B = np.random.rand(20, 30)
    sdfg(A=A, B=B)
    assert np.allclose(B[0:10, 0:20], A[10:20, 10:30] + 5)

    # Try again with a different value
    node.value_to_add = 7
    sdfg(A=A, B=B)
    assert np.allclose(B[0:10, 0:20], A[10:20, 10:30] + 7)


if __name__ == '__main__':
    test_custom_code_node()
