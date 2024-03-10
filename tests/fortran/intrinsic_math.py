import pytest
import numpy as np
from dace.frontend.fortran import fortran_parser

def test_fortran_frontend_math_abs():
    """
    Test that the generated SDFG correctly computes the absolute value.
    """
    code = """
            PROGRAM intrinsic_math_abs_test
                IMPLICIT NONE
                double precision, dimension(2) :: d
                double precision, dimension(2) :: res
                CALL intrinsic_math_test_function(d, res)
            END
            SUBROUTINE intrinsic_math_test_function(d, res)
                double precision, dimension(2) :: d
                double precision, dimension(2) :: res
                res(1) = ABS(d(1))
                res(2) = ABS(d(2))
            END SUBROUTINE intrinsic_math_test_function
    """
    sdfg = fortran_parser.create_sdfg_from_string(code, "abs_test", False)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    size = 2
    d = np.full([size], 42, order="F", dtype=np.float64)
    d[0] = -99
    d[1] = -1000
    res = np.full([2], 42, order="F", dtype=np.float64)
    # Execute the compiled SDFG
    sdfg(d=d, res=res)

    # Assert the expected result
    assert res[0] == 99
    assert res[1] == 1000

# def test_fortran_frontend_math_log():
#     """
#     Test that the generated SDFG correctly computes the natural logarithm.
#     """
#     code = """
#             PROGRAM intrinsic_math_log_test
#                 IMPLICIT NONE
#                 double precision, dimension(2) :: d
#                 double precision, dimension(2) :: res
#                 CALL intrinsic_math_test_function(d, res)
#             END
#             SUBROUTINE intrinsic_math_test_function(d, res)
#                 double precision, dimension(2) :: d
#                 double precision, dimension(2) :: res
#                 res(1) = LOG(d(1))
#                 res(2) = LOG(d(2))
#             END SUBROUTINE intrinsic_math_test_function
#     """
#     sdfg = fortran_parser.create_sdfg_from_string(code, "log_test", False)
#     sdfg.simplify(verbose=True)
#     sdfg.compile()

#     size = 2
#     d = np.full([size], 1.0, order="F", dtype=np.float64)
#     d[0] = np.e  # e^1 = 2.718281...
#     d[1] = np.e**2  # e^2
#     res = np.full([2], 0.0, order="F", dtype=np.float64)
#     # Execute the compiled SDFG
#     sdfg(d=d, res=res)

#     # Assert the expected result
#     assert np.isclose(res[0], 1.0)  # log(e) = 1
#     assert np.isclose(res[1], 2.0)  # log(e^2) = 2

if __name__ == "__main__":
    pytest.main()