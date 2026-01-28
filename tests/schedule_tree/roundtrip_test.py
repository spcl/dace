# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests conversion of schedule trees to SDFGs.
"""
import dace
import numpy as np


def test_implicit_inline_and_constants():
    """
    Tests implicit inlining upon roundtrip conversion, as well as constants with conflicting names.
    """

    @dace
    def nester(A: dace.float64[20]):
        A[:] = 12

    @dace.program
    def tester(A: dace.float64[20, 20]):
        for i in dace.map[0:20]:
            nester(A[:, i])

    sdfg = tester.to_sdfg(simplify=False)

    # Inject constant into nested SDFG
    assert len(list(sdfg.all_sdfgs_recursive())) > 1
    sdfg.add_constant('cst', 13)  # Add an unused constant
    sdfg.sdfg_list[-1].add_constant('cst', 1, dace.data.Scalar(dace.float64))
    tasklet = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet))
    tasklet.code.as_string = tasklet.code.as_string.replace('12', 'cst')

    # Perform a roundtrip conversion
    stree = sdfg.as_schedule_tree()
    new_sdfg = stree.as_sdfg()

    assert len(list(new_sdfg.all_sdfgs_recursive())) == 1
    assert new_sdfg.constants['cst_0'].dtype == np.float64

    # Test SDFG
    a = np.random.rand(20, 20)
    new_sdfg(A=a)  # Tests arg_names
    assert np.allclose(a, 1)


if __name__ == '__main__':
    test_implicit_inline_and_constants()
