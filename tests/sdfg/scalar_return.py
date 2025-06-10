# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest
from typing import Tuple

from dace.sdfg.validation import InvalidSDFGError


def single_retval_sdfg() -> dace.SDFG:

    @dace.program(auto_optimize=False, recreate_sdfg=True)
    def testee(A: dace.float64[20], ) -> dace.float64:
        return dace.float64(A[3])

    return testee.to_sdfg(validate=False)


def tuple_retval_sdfg() -> dace.SDFG:

    # This can not be used, as the frontend promotes the two scalars inside the tuple
    #  to arrays of length one.
    #@dace.program(auto_optimize=False, recreate_sdfg=True)
    #def testee(
    #    a: dace.float64,
    #    b: dace.float64,
    #) -> Tuple[dace.float64, dace.float64]:
    #    return a + b, a - b

    sdfg = dace.SDFG("scalar_tuple_return")
    state = sdfg.add_state("init", is_start_block=True)
    anames = ["a", "b"]
    sdfg.add_scalar(anames[0], dace.float64)
    sdfg.add_scalar(anames[1], dace.float64)
    sdfg.add_scalar("__return_0", dace.float64)
    sdfg.add_scalar("__return_1", dace.float64)
    acnodes = {aname: state.add_access(aname) for aname in anames}

    for iout, ops in enumerate(["+", "-"]):
        tskl = state.add_tasklet(
            "work",
            inputs={"__in0", "__in1"},
            outputs={"__out"},
            code=f"__out0 = __in0 {ops} __in1",
        )
        for isrc, src in enumerate(anames):
            state.add_edge(acnodes[src], None, tskl, f"__in{isrc}", dace.Memlet.simple(src, "0"))
        state.add_edge(
            tskl,
            "__out",
            state.add_write(f"__return_{iout}"),
            None,
            dace.Memlet.simple(f"__return_{iout}", "0"),
        )
    return sdfg


@pytest.mark.skip("Scalar return is not implemented")
def test_scalar_return():

    sdfg = single_retval_sdfg()
    assert isinstance(sdfg.arrays["__return"], dace.data.Scalar)

    sdfg.validate()
    A = np.random.rand(20)
    res = sdfg(A=A)
    assert isinstance(res, np.float64)
    assert A[3] == res


@pytest.mark.skip("Scalar return is not implemented")
def test_scalar_return_tuple():

    sdfg = tuple_retval_sdfg()
    assert all(isinstance(desc, dace.data.Scalar) for name, desc in sdfg.arrays.items() if name.startswith("__return"))

    sdfg.validate()
    a, b = np.float64(23.9), np.float64(10.0)
    res1, res2 = sdfg(a=a, b=b)
    assert all(isinstance(res, np.float64) for res in (ret1, ret2))
    assert np.isclose(res1 == (a + b))
    assert np.isclose(res2 == (a - b))


def test_scalar_return_validation():
    """Test if the validation actually works.

    Todo:
        Remove this test after scalar return values are implemented and enable
        the `test_scalar_return` and `test_scalar_return_tuple()` tests.
    """

    sdfg = single_retval_sdfg()
    with pytest.raises(
            InvalidSDFGError,
            match='Can not use scalar "__return" as return value.',
    ):
        sdfg.validate()

    sdfg = tuple_retval_sdfg()
    with pytest.raises(
            InvalidSDFGError,
            match='Can not use scalar "__return_(0|1)" as return value.',
    ):
        sdfg.validate()
