# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

from dace.frontend.fortran import fortran_parser


def test_fortran_frontend_loop_region_basic_loop():
    test_name = "loop_test"
    test_string = """
    PROGRAM loop_test_program
        implicit none
        double precision a(10,10)
        double precision b(10,10)
        double precision c(10,10)

        CALL loop_test_function(a,b,c)
    end

    SUBROUTINE loop_test_function(a,b,c)
        double precision :: a(10,10)
        double precision :: b(10,10)
        double precision :: c(10,10)

        INTEGER :: JK,JL
        DO JK=1,10
            DO JL=1,10
                c(JK,JL) = a(JK,JL) + b(JK,JL)
            ENDDO
        ENDDO
    end SUBROUTINE loop_test_function
    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, test_name)

    a_test = np.full([10, 10], 2, order="F", dtype=np.float64)
    b_test = np.full([10, 10], 3, order="F", dtype=np.float64)
    c_test = np.zeros([10, 10], order="F", dtype=np.float64)
    sdfg(a=a_test, b=b_test, c=c_test)

    validate = np.full([10, 10], 5, order="F", dtype=np.float64)

    assert np.allclose(c_test, validate)


if __name__ == '__main__':
    test_fortran_frontend_loop_region_basic_loop()
