# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

from fparser.common.readfortran import FortranStringReader
from fparser.common.readfortran import FortranFileReader
from fparser.two.parser import ParserFactory
import sys, os
import numpy as np
import pytest

from dace import SDFG, SDFGState, nodes, dtypes, data, subsets, symbolic
from dace.frontend.fortran import fortran_parser
from fparser.two.symbol_table import SymbolTable
from dace.sdfg import utils as sdutil

import dace.frontend.fortran.ast_components as ast_components
import dace.frontend.fortran.ast_transforms as ast_transforms
import dace.frontend.fortran.ast_utils as ast_utils
import dace.frontend.fortran.ast_internal_classes as ast_internal_classes


def test_fortran_frontend_basic_type():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    test_string = """
                    PROGRAM type_test
                    implicit none
                    
                    TYPE simple_type
                        REAL:: w(5,5,5),z(5)
                        INTEGER:: a       
                        REAL :: name  
                    END TYPE simple_type


                    REAL :: d(5,5)
                    CALL type_test_function(d)
                    end

                    SUBROUTINE type_test_function(d)
                    REAL d(5,5)
                    TYPE(simple_type) :: s
                    s%w(1,1,1)=5.5
                    !t=s
                    d(2,1)=5.5+s%w(1,1,1)
                    
                    END SUBROUTINE type_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "type_test")
    sdfg.simplify(verbose=True)
    a = np.full([4, 5], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)



def test_fortran_frontend_basic_type2():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    test_string = """
                    PROGRAM type_test2
                    implicit none
                    
                    TYPE simple_type
                        REAL:: w(5,5,5),z(5)
                        INTEGER:: a         
                    END TYPE simple_type

                    TYPE comlex_type
                        TYPE(simple_type):: s
                        REAL:: b
                    END TYPE comlex_type

                    TYPE meta_type
                        TYPE(comlex_type):: cc
                        REAL:: omega
                    END TYPE meta_type

                    REAL :: d(5,5)
                    CALL type_test2_function(d)
                    end

                    SUBROUTINE type_test2_function(d)
                    REAL d(5,5)
                    TYPE(simple_type) :: stype(3)
                    TYPE(comlex_type) :: c
                    TYPE(meta_type) :: m
                    
                    !c%b=1.0
                    c%s%w(1,1,1)=5.5
                    !m%cc%s%a=17
                    !s(1)%w(1,1,1)=5.5+c%b
                    d(2,1)=c%s%w(1,1,1)
                    !+s(1)%w(1,1,1)
                    
                    END SUBROUTINE type_test2_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "type_test")
    sdfg.simplify(verbose=True)
    a = np.full([4, 5], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)


def test_fortran_frontend_circular_type():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    test_string = """
                    PROGRAM type_test
                    implicit none
                    
                    
                    type a_t
                        real :: w(5,5,5)
                        type(b_t), pointer :: b
                    end type a_t

                    type b_t
                        type(a_t)          :: a
                        integer              :: x
                    end type b_t

                    type c_t
                        type(d_t),pointer    :: ab
                        integer              :: xz
                    end type c_t

                    type d_t
                        type(c_t)          :: ac
                        integer              :: xy
                    end type d_t

                    REAL :: d(5,5)

                    CALL circular_type_test_function(d)
                    end

                    SUBROUTINE circular_type_test_function(d)
                    REAL d(5,5)
                    TYPE(a_t) :: s
                    TYPE(b_t) :: b(3)
                    
                    s%w(1,1,1)=5.5
                    !s%b=>b(1)
                    !s%b%a=>s
                    b(1)%x=1
                    d(2,1)=5.5+s%w(1,1,1)
                    
                    END SUBROUTINE circular_type_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "type_test")
    sdfg.simplify(verbose=True)
    a = np.full([4, 5], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)

if __name__ == "__main__":

    #test_fortran_frontend_basic_type()
    test_fortran_frontend_basic_type2()

    #test_fortran_frontend_circular_type()