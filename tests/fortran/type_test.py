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

from dace.transformation.passes.lift_struct_views import LiftStructViews
from dace.transformation import pass_pipeline as ppl


def test_fortran_frontend_basic_type():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    test_string = """
        PROGRAM type_test
            implicit none

            TYPE simple_type
                REAL :: w(5,5,5), z(5)
                INTEGER :: a
                REAL :: name
            END TYPE simple_type

            REAL :: d(5,5)
            CALL type_test_function(d)
        end

        SUBROUTINE type_test_function(d)
            REAL d(5,5)
            TYPE(simple_type) :: s
            s%w(1,1,1) = 5.5
            d(2,1) = 5.5 + s%w(1,1,1)
        END SUBROUTINE type_test_function
    """
    sources={}
    sources["type_test"]=test_string
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "type_test",sources=sources)
    sdfg.simplify(verbose=True)
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)



def test_fortran_frontend_basic_type2():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    test_string = """
                    PROGRAM type2_test
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
                    CALL type2_test_function(d)
                    end

                    SUBROUTINE type2_test_function(d)
                    REAL d(5,5)
                    TYPE(simple_type) :: s(3)
                    TYPE(comlex_type) :: c
                    TYPE(meta_type) :: m
                    
                    c%b=1.0
                    c%s%w(1,1,1)=5.5
                    m%cc%s%a=17
                    s(1)%w(1,1,1)=5.5+c%b
                    d(2,1)=c%s%w(1,1,1)+s(1)%w(1,1,1)
                    
                    END SUBROUTINE type2_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "type2_test")
    sdfg.validate()
    sdfg.simplify(verbose=True)
    a = np.full([4, 5], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)


def test_fortran_frontend_type_symbol():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    test_string = """
                    PROGRAM type_symbol_test
                    implicit none
                    
                    TYPE simple_type
                        REAL:: z(5)
                        INTEGER:: a         
                    END TYPE simple_type

                    
                    REAL :: d(5,5)
                    CALL type_symbol_test_function(d)
                    end

                    SUBROUTINE type_symbol_test_function(d)
                    TYPE(simple_type) :: st 
                    REAL :: d(5,5)
                    st%a=10
                    CALL internal_function(d,st)
                    
                    END SUBROUTINE type_symbol_test_function

                    
                    SUBROUTINE internal_function(d,st)
                    REAL d(5,5)
                    TYPE(simple_type) :: st
                    REAL bob(st%a) 
                    bob(1)=5.5
                    d(2,1)=2*bob(1)
                    
                    END SUBROUTINE internal_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "type_symbol_test",sources={"type_symbol_test":test_string})
    sdfg.validate()
    sdfg.simplify(verbose=True)
    a = np.full([4, 5], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)

def test_fortran_frontend_type_pardecl():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    test_string = """
                    PROGRAM type_pardecl_test
                    implicit none
                    
                    TYPE simple_type
                        REAL:: z(5,5,5)
                        INTEGER:: a         
                    END TYPE simple_type

                    
                    REAL :: d(5,5)
                    CALL type_pardecl_test_function(d)
                    end

                    SUBROUTINE type_pardecl_test_function(d)
                    TYPE(simple_type) :: st 
                    REAL :: d(5,5)
                    st%a=10
                    CALL internal_function(d,st)
                    
                    END SUBROUTINE type_pardecl_test_function

                    
                    SUBROUTINE internal_function(d,st)
                    REAL d(5,5)
                    TYPE(simple_type) :: st
                    REAL bob(st%a) 
                    INTEGER, PARAMETER :: n=5
                    REAL BOB2(n)
                    bob(1)=5.5
                    bob2(1)=5.5
                    st%z(1,:,2:3)=bob(1)
                    d(2,1)=bob(1)+bob2
                    
                    END SUBROUTINE internal_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "type_pardecl_test",sources={"type_pardecl_test":test_string})
    sdfg.validate()
    sdfg.simplify(verbose=True)
    a = np.full([4, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)

def test_fortran_frontend_type_struct():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    test_string = """
                    PROGRAM type_struct_test
                    implicit none
                    
                    TYPE simple_type
                        REAL:: z(5,5,5)
                        INTEGER:: a   
                        REAL :: unkown(:)      
                        !INTEGER :: unkown_size
                    END TYPE simple_type

                    
                    REAL :: d(5,5)
                    CALL type_struct_test_function(d)
                    end

                    SUBROUTINE type_struct_test_function(d)
                    TYPE(simple_type) :: st 
                    REAL :: d(5,5)
                    st%a=10
                    CALL internal_function(d,st)
                    
                    END SUBROUTINE type_struct_test_function

                    
                    SUBROUTINE internal_function(d,st)
                    st.a.shape=[st.a_size]
                    REAL d(5,5)
                    TYPE(simple_type) :: st
                    REAL bob(st%a) 
                    INTEGER, PARAMETER :: n=5
                    REAL BOB2(n)
                    bob(1)=5.5
                    bob2(1)=5.5
                    st%z(1,:,2:3)=bob(1)
                    d(2,1)=bob(1)+bob2(1)
                    
                    END SUBROUTINE internal_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "type_struct_test",sources={"type_struct_test":test_string})
    sdfg.validate()
    sdfg.simplify(verbose=True)
    a = np.full([4, 5], 42, order="F", dtype=np.float32)
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



def test_fortran_frontend_type_in_call():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    test_string = """
        PROGRAM type_in_call_test
            implicit none

            TYPE simple_type
                REAL :: w(5,5,5), z(5)
                INTEGER :: a
                REAL :: name
            END TYPE simple_type

            REAL :: d(5,5)
            CALL type_in_call_test_function(d)
        end

        SUBROUTINE type_in_call_test_function(d)
            REAL d(5,5)
            TYPE(simple_type) :: s
            REAL,POINTER :: tmp(:,:,:)
            tmp=>s%w
            tmp(1,1,1) = 11.0
            d(2,1) = max(1.0, tmp(1,1,1))
        END SUBROUTINE type_in_call_test_function
    """
    sources={}
    sources["type_test"]=test_string
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "type_in_call_test",sources=sources)
    sdfg.simplify(verbose=True)
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)

def test_fortran_frontend_type_array():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    test_string = """
        PROGRAM type_in_call_test
            implicit none

            TYPE simple_type3
                INTEGER :: a
            END TYPE simple_type3

            TYPE simple_type2
                type(simple_type3) :: w(7:12,8:13)
            END TYPE simple_type2

            TYPE simple_type
                type(simple_type2) :: name
            END TYPE simple_type

            REAL :: d(5,5)
            CALL type_in_call_test_function(d)
        end

        SUBROUTINE type_in_call_test_function(d)
            REAL :: d(5,5)
            TYPE(simple_type) :: s

            CALL type_in_call_test_function2(s)
            d(1,1) = s%name%w(8,10)%a
        END SUBROUTINE type_in_call_test_function

        SUBROUTINE type_in_call_test_function2(s)
            TYPE(simple_type) :: s

            s%name%w(8,10)%a = 42
        END SUBROUTINE type_in_call_test_function2
    """
    sources={}
    sources["type_test"]=test_string
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "type_in_call_test",sources=sources, normalize_offsets=True)
    sdfg.simplify(verbose=True)
    sdfg.save('test.sdfg')
    sdfg.compile()

    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    print(a)

def test_fortran_frontend_type_array2():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    test_string = """
        PROGRAM type_in_call_test
            implicit none

            TYPE simple_type3
                INTEGER :: a
            END TYPE simple_type3

            TYPE simple_type2
                type(simple_type3) :: w(7:12,8:13)
                integer :: wx(7:12,8:13)
            END TYPE simple_type2

            TYPE simple_type
                type(simple_type2) :: name
            END TYPE simple_type

            REAL :: d(5,5)
            CALL type_in_call_test_function(d)
        end

        SUBROUTINE type_in_call_test_function(d)
            REAL :: d(5,5)
            integer :: x(3,3,3)
            TYPE(simple_type) :: s

            CALL type_in_call_test_function2(s,x)
            !d(1,1) = s%name%w(8, x(3,3,3))%a
            d(1,2) = s%name%wx(8, x(3,3,3))
        END SUBROUTINE type_in_call_test_function

        SUBROUTINE type_in_call_test_function2(s,x)
            TYPE(simple_type) :: s
            integer :: x(3,3,3)

            x(3,3,3) = 10
            !s%name%w(8,x(3,3,3))%a = 42
            s%name%wx(8,x(3,3,3)) = 43
        END SUBROUTINE type_in_call_test_function2
    """
    sources={}
    sources["type_test"]=test_string
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "type_in_call_test",sources=sources, normalize_offsets=True)
    sdfg.save("before.sdfg")
    sdfg.simplify(verbose=True)
    sdfg.save("after.sdfg")
    sdfg.compile()

    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    print(a)

def test_fortran_frontend_type_pointer():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    test_string = """
        PROGRAM type_pointer_test
            implicit none

            TYPE simple_type
                REAL :: w(5,5,5), z(5)
                INTEGER :: a
                REAL :: name
            END TYPE simple_type

            REAL :: d(5,5)
            CALL type_pointer_test_function(d)
        end

        SUBROUTINE type_pointer_test_function(d)
            REAL d(5,5)
            TYPE(simple_type) :: s
            REAL, DIMENSION(:,:,:), POINTER :: tmp
            tmp=>s%w
            tmp(1,1,1) = 11.0
            d(2,1) = max(1.0, tmp(1,1,1))
        END SUBROUTINE type_pointer_test_function
    """
    sources={}
    sources["type_pointer_test"]=test_string
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "type_pointer_test",sources=sources)
    sdfg.simplify(verbose=True)
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)


def test_fortran_frontend_type_arg():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    test_string = """
        PROGRAM type_arg_test
            implicit none

          
            TYPE simple_type
                REAL :: w(5,5)
            END TYPE simple_type

             TYPE simple_type2
                type(simple_type) :: pprog(10)
            END TYPE simple_type2

            REAL :: d(5,5)
            CALL type_arg_test_function(d)
            print *, d(1,1)
        end

        SUBROUTINE type_arg_test_function(d)
            REAL :: d(5,5)
            TYPE(simple_type2) :: p_prog

            CALL type_arg_test_f2(p_prog%pprog(1))
            d(1,1) = p_prog%pprog(1)%w(1,1)
        END SUBROUTINE type_arg_test_function

        SUBROUTINE type_arg_test_f2(stuff)
            TYPE(simple_type) :: stuff
            CALL deepest(stuff%w)
            
        END SUBROUTINE type_arg_test_f2

        SUBROUTINE deepest(my_arr)
            REAL :: my_arr(:,:)

            my_arr(1,1) = 42
        END SUBROUTINE deepest

    """
    sources={}
    sources["type_arg_test"]=test_string
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "type_arg_test",sources=sources, normalize_offsets=True)
    sdfg.simplify(verbose=True)
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    print(a)



def test_fortran_frontend_type_arg2():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    test_string = """
        PROGRAM type_arg2_test
            implicit none

          
            TYPE simple_type
                REAL :: w(5,5)
            END TYPE simple_type

             TYPE simple_type2
                type(simple_type) :: pprog(10)
            END TYPE simple_type2

            REAL :: d(5,5)
            CALL type_arg2_test_function(d)
            print *, d(1,1)
        end

        SUBROUTINE type_arg2_test_function(d)
            REAL :: d(5,5)
            TYPE(simple_type2) :: p_prog

            CALL deepest(p_prog%pprog(1)%w)
            d(1,1) = p_prog%pprog(1)%w(1,1)
        END SUBROUTINE type_arg2_test_function

        SUBROUTINE deepest(my_arr)
            REAL :: my_arr(:,:)

            my_arr(1,1) = 42
        END SUBROUTINE deepest

    """
    sources={}
    sources["type_arg2_test"]=test_string
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "type_arg2_test",sources=sources, normalize_offsets=True)
    sdfg.simplify(verbose=True)
    a = np.full([5, 5], 42, order="F", dtype=np.float32)
    sdfg(d=a)
    print(a)



def test_fortran_frontend_type_view():
    """
    Tests that the Fortran frontend can parse the simplest type declaration and make use of it in a computation.
    """
    test_string = """
                    PROGRAM type_view_test
                    implicit none
                    
                    TYPE simple_type
                        REAL:: z(5)
                        INTEGER:: a         
                    END TYPE simple_type

                    
                    REAL :: d(5,5)
                    CALL type_view_test_function(d)
                    end

                    SUBROUTINE type_view_test_function(d)
                    TYPE(simple_type) :: st 
                    REAL :: d(5,5)
                    st%z(1)=5.5
                    CALL internal_function(d,st%z)
                    
                    END SUBROUTINE type_view_test_function

                    
                    SUBROUTINE internal_function(d,sta)
                    REAL d(5,5)
                    REAL sta(:)
                    d(2,1)=2*sta(1)
                    
                    END SUBROUTINE internal_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "type_view_test",sources={"type_view_test":test_string},normalize_offsets=True)
    sdfg.validate()
    sdfg.simplify(verbose=True)
    a = np.full([4, 5], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0, 0] == 42)
    assert (a[1, 0] == 11)
    assert (a[2, 0] == 42)


if __name__ == "__main__":
    #test_fortran_frontend_basic_type()
    #test_fortran_frontend_basic_type2()
    #test_fortran_frontend_type_symbol()
    #test_fortran_frontend_type_pardecl()
    #test_fortran_frontend_type_struct()
    #test_fortran_frontend_circular_type()
    #test_fortran_frontend_type_in_call()
    #test_fortran_frontend_type_array()
    #test_fortran_frontend_type_array2()
    #test_fortran_frontend_type_pointer()
    #test_fortran_frontend_type_arg()
    test_fortran_frontend_type_view()
    #test_fortran_frontend_type_arg2()
