import dace


def test_type(wcr_str, red_type):
    assert dace.detect_reduction_type('lambda a,b: %s' % wcr_str) == red_type


def test_comm_assoc(wcr_str, comm, assoc):
    assert dace.is_op_commutative('lambda a,b: %s' % wcr_str) == comm
    assert dace.is_op_associative('lambda a,b: %s' % wcr_str) == assoc


if __name__ == '__main__':
    test_type('a + b', dace.ReductionType.Sum)
    test_type('a * b', dace.ReductionType.Product)
    test_type('min(a, b)', dace.ReductionType.Min)
    test_type('max(a, b)', dace.ReductionType.Max)
    test_type('a | b', dace.ReductionType.Bitwise_Or)
    test_type('a ^ b', dace.ReductionType.Bitwise_Xor)
    test_type('a & b', dace.ReductionType.Bitwise_And)
    test_type('a or b', dace.ReductionType.Logical_Or)
    test_type('a != b', dace.ReductionType.Logical_Xor)
    test_type('a and b', dace.ReductionType.Logical_And)
    #test_type('b if b[0] < a[0] else a', dace.ReductionType.Min_Location)
    #test_type('b if b[0] > a[0] else a', dace.ReductionType.Max_Location)

    test_type('a * b + b', dace.ReductionType.Custom)
    test_type('a / b', dace.ReductionType.Custom)

    # Test for associativity / commutativity of operations
    test_comm_assoc('a + b', True, True)
    test_comm_assoc('a * b', True, True)
    test_comm_assoc('(a + b) / 2', True, False)
    test_comm_assoc('a', False, True)
    test_comm_assoc('a * b + b', False, False)
    test_comm_assoc('a / b', False, False)

    print('PASSED')
