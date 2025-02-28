# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests optional (could be None) arrays and arguments. """
from typing import Optional, Union

import numpy as np
import pytest

import dace
from dace.sdfg.utils import inline_sdfgs
from dace.transformation.passes.dead_state_elimination import DeadStateElimination
from dace.transformation.passes.optional_arrays import OptionalArrayInference


def test_type_hint():
    assert Optional[dace.float64[20, 20]] == Union[dace.float64[20, 20], None]
    assert dace.float64[20, 20] != dace.float32[20, 20]
    assert (Union[None, dace.float64[20, 20], dace.float64[20, 21], dace.float64[20, 20],
                  None] == Optional[Union[dace.float64[20, 20], dace.float64[20, 21]]])


def test_optional_arg_hint():

    @dace.program
    def tester(a: Optional[dace.float64[1]], b: dace.float64[1]):
        transient = b + 1

    sdfg = tester.to_sdfg()
    assert sdfg.arrays['a'].optional is True
    # Depending on analysis passes, b may either not be optional or indeterminate
    assert (sdfg.arrays['b'].optional is False or sdfg.arrays['b'].optional is None)
    # Transients cannot be optional
    if 'transient' in sdfg.arrays:
        assert sdfg.arrays['transient'].optional is False


def test_optional_argcheck():
    desc = dace.float64[20, 20]
    desc.optional = False  # Explicitly set to non-optional

    @dace.program
    def tester(a: desc):
        pass

    with pytest.raises(TypeError):
        tester(None)


@pytest.mark.parametrize('isnone', (False, True))
def test_optional_dead_state(isnone):
    desc = dace.float64[20, 20]
    desc.optional = False  # Explicitly set to non-optional

    if isnone:

        @dace.program
        def tester(a: Optional[dace.float64[20]], b: desc):
            if a is None:
                return 1
            elif b is None:
                return 2
            else:
                return 3
    else:

        @dace.program
        def tester(a: Optional[dace.float64[20]], b: desc):
            if a is None:
                return 1
            elif b is not None:
                return 2
            else:
                return 3

    sdfg = tester.to_sdfg(simplify=False)
    inline_sdfgs(sdfg)
    DeadStateElimination().apply_pass(sdfg, {})
    assert all('b' not in str(e.data.condition.as_string) for e in sdfg.edges())


def test_optional_array_inference_via_simplify():

    @dace.program
    def nested(b):
        if b is not None:
            tmp = np.zeros_like(b)
            b[:] += 1 + tmp

    NotOptional = dace.float64[20]
    NotOptional.optional = False

    @dace.program
    def outer(yes: Optional[dace.float64[20]], no: NotOptional, maybe: dace.float64[20], always_read: dace.float64[20],
              cond: dace.int32):
        # Add loop to challenge unconditional traversal
        for _ in range(10):
            pass

        nested(always_read)
        if cond == 0:
            nested(yes)
        else:
            nested(no)

        if cond == 1:
            nested(maybe)

    sdfg = outer.to_sdfg(simplify=False)
    sdfg.validate()

    # Before pass, `maybe` and `always_read` are unknown
    assert sdfg.arrays['maybe'].optional is None
    assert sdfg.arrays['always_read'].optional is None

    result = OptionalArrayInference().apply_pass(sdfg, {})
    assert (0, 'always_read') in result

    # Check top-level SDFG
    assert sdfg.arrays['yes'].optional is True
    assert sdfg.arrays['no'].optional is False
    assert sdfg.arrays['maybe'].optional is None
    assert sdfg.arrays['always_read'].optional is False

    # Test nested SDFGs
    for state in sdfg.nodes():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                inputnode: dace.nodes.AccessNode = state.predecessors(node)[0]
                assert node.sdfg.arrays['b'].optional == inputnode.desc(sdfg).optional
                assert node.sdfg.arrays['tmp'].optional is False


def test_optional_array_inference_via_parse():

    @dace.program
    def nested(nested_none_arr: Optional[dace.float64[20]], nested_arr: dace.float64[20]):
        if nested_none_arr is None:
            nested_none_arr[:] = 10

        if nested_arr is None:
            nested_arr[:] = 10

    @dace.program
    def outer(none_arr: Optional[dace.float64[20]], arr: dace.float64[20], unknown):
        # Add loop to challenge unconditional traversal
        for _ in range(10):
            pass

        if none_arr is None:
            none_arr[:] = 10
        else:
            unknown[:] = 10

        if arr is None:
            arr[:] = 10

        nested(none_arr, nested_arr=arr)

    g_none_arr = None
    g_arr = np.zeros(3)
    unknown_arr = np.zeros(3)

    sdfg = outer.to_sdfg(g_none_arr, g_arr, unknown_arr, simplify=False)
    sdfg.validate()

    # Check arrays are properly optional is properly handle
    # with correct type hint
    assert sdfg.arrays['none_arr'].optional is True
    assert sdfg.arrays['arr'].optional is False
    assert sdfg.arrays['unknown'].optional is None
    assert sdfg.sdfg_list[1].arrays['nested_none_arr'].optional is True
    assert sdfg.sdfg_list[1].arrays['nested_arr'].optional is False

    # Check that the ConditionalOptionalArrayResolver pass in preprocess
    # combined with ConditionalCodeResolver and DeadCodeEliminator lead
    # to the branch elimination of None array
    arr_is_assigned = False
    none_arr_is_assigned = False
    unknown_is_assigned = False
    nested_arr_is_assigned = False
    nested_none_arr_is_assigned = False
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.AccessNode) and node.has_writes(state):
            arr_is_assigned |= (node.data == 'arr')
            none_arr_is_assigned |= (node.data == 'none_arr')
            unknown_is_assigned |= (node.data == 'unknown')
            nested_arr_is_assigned |= (node.data == 'nested_arr')
            nested_none_arr_is_assigned |= (node.data == 'nested_none_arr')

    assert arr_is_assigned
    assert not none_arr_is_assigned
    assert unknown_is_assigned
    assert nested_arr_is_assigned
    assert not nested_none_arr_is_assigned


if __name__ == '__main__':
    test_type_hint()
    test_optional_arg_hint()
    test_optional_argcheck()
    test_optional_dead_state(False)
    test_optional_dead_state(True)
    test_optional_array_inference_via_parse()
    test_optional_array_inference_via_simplify()
