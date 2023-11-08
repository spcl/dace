# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np

from dace.frontend.fortran import ast_transforms, fortran_parser

def test_fortran_frontend_offset_normalizer_1d():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    test_string = """
                    PROGRAM index_offset_test
                    implicit none
                    double precision, dimension(50:54) :: d
                    CALL index_test_function(d)
                    end

                    SUBROUTINE index_test_function(d)
                    double precision, dimension(50:54) :: d

                    do i=50,54
                        d(i) = i * 2.0
                    end do

                    END SUBROUTINE index_test_function
                    """

    # Test to verify that offset is normalized correctly
    ast, own_ast = fortran_parser.create_ast_from_string(test_string, "index_offset_test", True, True)

    for subroutine in ast.subroutine_definitions:

        loop = subroutine.execution_part.execution[1]
        idx_assignment = loop.body.execution[1]
        assert idx_assignment.rval.rval.value == "50"

    # Now test to verify it executes correctly

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "index_offset_test", True)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    assert len(sdfg.data('d').shape) == 1
    assert sdfg.data('d').shape[0] == 5

    a = np.full([5], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    for i in range(0,5):
        assert a[i] == (50+i)* 2

def test_fortran_frontend_offset_normalizer_2d():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    test_string = """
                    PROGRAM index_offset_test
                    implicit none
                    double precision, dimension(50:54,7:9) :: d
                    CALL index_test_function(d)
                    end

                    SUBROUTINE index_test_function(d)
                    double precision, dimension(50:54,7:9) :: d

                    do i=50,54
                        do j=7,9
                            d(i, j) = i * 2.0 + 3 * j
                        end do
                    end do

                    END SUBROUTINE index_test_function
                    """

    # Test to verify that offset is normalized correctly
    ast, own_ast = fortran_parser.create_ast_from_string(test_string, "index_offset_test", True, True)

    for subroutine in ast.subroutine_definitions:

        loop = subroutine.execution_part.execution[1]
        nested_loop = loop.body.execution[1]

        idx = nested_loop.body.execution[1]
        assert idx.lval.name == 'tmp_index_0'
        assert idx.rval.rval.value == "50"

        idx2 = nested_loop.body.execution[3]
        assert idx2.lval.name == 'tmp_index_1'
        assert idx2.rval.rval.value == "7"

    # Now test to verify it executes correctly

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "index_offset_test", True)
    sdfg.simplify(verbose=True)
    sdfg.compile()

    assert len(sdfg.data('d').shape) == 2
    assert sdfg.data('d').shape[0] == 5
    assert sdfg.data('d').shape[1] == 3

    a = np.full([5,3], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    for i in range(0,5):
        for j in range(0,3):
            assert a[i, j] == (50+i) * 2 + 3 * (7 + j)

def test_fortran_frontend_offset_normalizer_2d_arr2loop():
    """
    Tests that the Fortran frontend can parse array accesses and that the accessed indices are correct.
    """
    test_string = """
                    PROGRAM index_offset_test
                    implicit none
                    double precision, dimension(50:54,7:9) :: d
                    CALL index_test_function(d)
                    end

                    SUBROUTINE index_test_function(d)
                    double precision, dimension(50:54,7:9) :: d

                    do i=50,54
                        d(i, :) = i * 2.0
                    end do

                    END SUBROUTINE index_test_function
                    """

    # Test to verify that offset is normalized correctly
    ast, own_ast = fortran_parser.create_ast_from_string(test_string, "index_offset_test", True, True)

    for subroutine in ast.subroutine_definitions:

        loop = subroutine.execution_part.execution[1]
        nested_loop = loop.body.execution[1]

        idx = nested_loop.body.execution[1]
        assert idx.lval.name == 'tmp_index_0'
        assert idx.rval.rval.value == "50"

        idx2 = nested_loop.body.execution[3]
        assert idx2.lval.name == 'tmp_index_1'
        assert idx2.rval.rval.value == "7"

    # Now test to verify it executes correctly with no normalization

    sdfg = fortran_parser.create_sdfg_from_string(test_string, "index_offset_test", True)
    sdfg.save('test.sdfg')
    sdfg.simplify(verbose=True)
    sdfg.compile()

    assert len(sdfg.data('d').shape) == 2
    assert sdfg.data('d').shape[0] == 5
    assert sdfg.data('d').shape[1] == 3

    a = np.full([5,3], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    for i in range(0,5):
        for j in range(0,3):
            assert a[i, j] == (50 + i) * 2

if __name__ == "__main__":

    test_fortran_frontend_offset_normalizer_1d()
    test_fortran_frontend_offset_normalizer_2d()
    test_fortran_frontend_offset_normalizer_2d_arr2loop()
