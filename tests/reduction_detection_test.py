# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.frontend.operations import is_op_associative, is_op_commutative, detect_reduction_type

def _test_type(wcr_str, red_type):
    assert detect_reduction_type('lambda a,b: %s' % wcr_str) == red_type


def _test_comm_assoc(wcr_str, comm, assoc):
    assert is_op_commutative('lambda a,b: %s' % wcr_str) == comm
    assert is_op_associative('lambda a,b: %s' % wcr_str) == assoc


def test_expr_type():
    _test_type('a + b', dace.ReductionType.Sum)
    _test_type('a * b', dace.ReductionType.Product)
    _test_type('min(a, b)', dace.ReductionType.Min)
    _test_type('max(a, b)', dace.ReductionType.Max)
    _test_type('a | b', dace.ReductionType.Bitwise_Or)
    _test_type('a ^ b', dace.ReductionType.Bitwise_Xor)
    _test_type('a & b', dace.ReductionType.Bitwise_And)
    _test_type('a or b', dace.ReductionType.Logical_Or)
    _test_type('a != b', dace.ReductionType.Logical_Xor)
    _test_type('a and b', dace.ReductionType.Logical_And)
    #_test_type('b if b[0] < a[0] else a', dace.ReductionType.Min_Location)
    #_test_type('b if b[0] > a[0] else a', dace.ReductionType.Max_Location)

    _test_type('a * b + b', dace.ReductionType.Custom)
    _test_type('a / b', dace.ReductionType.Custom)


def test_op_properties():
    # Test for associativity / commutativity of operations
    _test_comm_assoc('a + b', True, True)
    _test_comm_assoc('a * b', True, True)
    _test_comm_assoc('(a + b) / 2', True, False)
    _test_comm_assoc('a', False, True)
    _test_comm_assoc('a * b + b', False, False)
    _test_comm_assoc('a / b', False, False)


if __name__ == '__main__':
    test_expr_type()
    test_op_properties()
