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


def test_optional_array_inference():
    @dace.program
    def nested(b):
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


if __name__ == '__main__':
    test_type_hint()
    test_optional_arg_hint()
    test_optional_argcheck()
    test_optional_dead_state(False)
    test_optional_dead_state(True)
    test_optional_array_inference()
