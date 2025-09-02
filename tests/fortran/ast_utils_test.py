import pytest

from dace.frontend.fortran.ast_internal_classes import Real_Literal_Node

from dace.frontend.fortran.ast_utils import TaskletWriter


def test_floatlit2string():

    def parse(fl: str) -> float:
        t = TaskletWriter([], [])  # The parameters won't matter.
        return t.floatlit2string(Real_Literal_Node(value=fl))

    assert parse('1.0') == '1.0'
    assert parse('1.') == '1.0'
    assert parse('1.e5') == '100000.0'
    assert parse('1.d5') == '100000.0'
    assert parse('1._kinder') == '1.0'
    assert parse('1.e5_kinder') == '100000.0'
    with pytest.raises(AssertionError):
        parse('1.d5_kinder')
    with pytest.raises(AssertionError):
        parse('1._kinder_kinder')
    with pytest.raises(ValueError, match="could not convert string to float"):
        parse('1.2.0')
    with pytest.raises(ValueError, match="could not convert string to float"):
        parse('1.d0d0')
    with pytest.raises(ValueError, match="could not convert string to float"):
        parse('foo')
