# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``compare_numpy_output`` is the oracle of every ufunc test, so an output it skips passes while wrong."""
import dace
import numpy as np
import pytest

from common import compare_numpy_output


def two_outputs(A: dace.float64[4]):
    return A + 1.0, A * 2.0


def run_dace_leg_as_python_tampering_the_second_output(monkeypatch, tamper):
    """The harness's dace leg becomes the plain Python function with its second output passed through ``tamper``."""

    def program(device):

        def wrap(func):

            def run(**arguments):
                first, second = func(**arguments)
                return first, tamper(second)

            return run

        return wrap

    monkeypatch.setattr(dace, 'program', program)


@pytest.mark.parametrize('tamper', [
    pytest.param(lambda second: second + 1.0, id='value'),
    pytest.param(lambda second: second.astype(np.float32), id='dtype'),
])
def test_a_wrong_second_tuple_element_is_reported(monkeypatch, tamper):
    run_dace_leg_as_python_tampering_the_second_output(monkeypatch, tamper)
    with pytest.raises(AssertionError, match=r'^\(1, '):
        compare_numpy_output(check_dtype=True)(two_outputs)()


def test_an_untampered_tuple_result_passes(monkeypatch):
    """The control for the test above: the stand-in dace leg alone does not trip the harness."""
    run_dace_leg_as_python_tampering_the_second_output(monkeypatch, lambda second: second)
    compare_numpy_output(check_dtype=True)(two_outputs)()
