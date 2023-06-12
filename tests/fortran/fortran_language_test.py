# Copyright 2023 ETH Zurich and the DaCe authors. All rights reserved.

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


def test_fortran_frontend_real_kind_selector():
    test_string = """
                    PROGRAM real_kind_selector_test
                    implicit none
                    INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13,300)
                    INTEGER, PARAMETER :: JPIM = SELECTED_INT_KIND(9)
                    REAL(KIND=JPRB) d(4)
                    CALL real_kind_selector_test_function(d)
                    end

                    SUBROUTINE real_kind_selector_test_function(d)
                    REAL(KIND=JPRB) d(4)
                    INTEGER(KIND=JPIM) i

                    i=7
                    d(2)=5.5+i
                    
                    END SUBROUTINE real_kind_selector_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "real_kind_selector_test")
    sdfg.simplify(verbose=True)
    a = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert (a[0] == 42)
    assert (a[1] == 12.5)
    assert (a[2] == 42)


def test_fortran_frontend_if1():
    test_string = """
                    PROGRAM if1_test
                    implicit none
                    double precision d(3,4,5)
                    CALL if1_test_function(d)
                    end

                    SUBROUTINE if1_test_function(d)
                    double precision d(3,4,5),ZFAC(10)
                    integer JK,JL,RTT,NSSOPT
                    integer ZTP1(10,10)
                    JL=1
                    JK=1
                    ZTP1(JL,JK)=1.0
                    RTT=2
                    NSSOPT=1

                    IF (ZTP1(JL,JK)>=RTT .OR. NSSOPT==0) THEN
                      ZFAC(1)  = 1.0
                    ELSE
                      ZFAC(1)  = 2.0
                    ENDIF
                    d(1,1,1)=ZFAC(1)
                                    
                    END SUBROUTINE if1_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "if1_test")
    sdfg.simplify(verbose=True)
    d = np.full([3, 4, 5], 42, order="F", dtype=np.float64)
    sdfg(d=d)
    assert (d[0, 0, 0] == 2)


def test_fortran_frontend_loop1():
    test_string = """
                    PROGRAM loop1_test
                    implicit none
                    double precision d(3,4,5)
                    CALL loop1_test_function(d)
                    end

                    SUBROUTINE loop1_test_function(d)
                   double precision d(3,4,5),ZFAC(10)
                   INTEGER :: a, JK, JL,JM
                   INTEGER, PARAMETER :: KLEV=10, N=10,NCLV=3

                   double precision :: RLMIN,ZVQX(NCLV)
                   LOGICAL :: LLCOOLJ,LLFALL(NCLV)
                   LLFALL(:)= .FALSE.
                   ZVQX(:)= 0.0
                   ZVQX(2)= 1.0
                   DO JM=1,NCLV
                    IF (ZVQX(JM)>0.0) LLFALL(JM)=.TRUE. ! falling species
                   ENDDO

                   d(1,1,1)=LLFALL(1)
                   d(1,1,2)=LLFALL(2)                 
                   END SUBROUTINE loop1_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "loop1_test")
    sdfg.simplify(verbose=True)
    d = np.full([3, 4, 5], 42, order="F", dtype=np.float64)
    sdfg(d=d)
    assert (d[0, 0, 0] == 0)
    assert (d[0, 0, 1] == 1)


def test_fortran_frontend_function_statement1():
    test_string = """
                    PROGRAM function_statement1_test
                    implicit none
                    double precision d(3,4,5)
                    CALL function_statement1_test_function(d)
                    end

                    SUBROUTINE function_statement1_test_function(d)
                   double precision d(3,4,5)
                   double precision :: PTARE,RTT(2),FOEDELTA,FOELDCP
                   double precision :: RALVDCP(2),RALSDCP(2),RES

                    FOEDELTA (PTARE) = MAX (0.0,SIGN(1.0,PTARE-RTT(1)))
                    FOELDCP ( PTARE ) = FOEDELTA(PTARE)*RALVDCP(1) + (1.0-FOEDELTA(PTARE))*RALSDCP(1)

                    RTT(1)=4.5
                    RALVDCP(1)=4.9
                    RALSDCP(1)=5.1
                    d(1,1,1)=FOELDCP(3.0)
                    RES=FOELDCP(3.0)
                   d(1,1,2)=RES                 
                   END SUBROUTINE function_statement1_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "function_statement1_test")
    sdfg.simplify(verbose=True)
    d = np.full([3, 4, 5], 42, order="F", dtype=np.float64)
    sdfg(d=d)
    assert (d[0, 0, 0] == 5.1)
    assert (d[0, 0, 1] == 5.1)


def test_fortran_frontend_pow1():
    test_string = """
                    PROGRAM pow1_test
                    implicit none
                    double precision d(3,4,5)
                    CALL pow1_test_function(d)
                    end

                    SUBROUTINE pow1_test_function(d)
                   double precision d(3,4,5)
                  double precision :: ZSIGK(2), ZHRC(2),RAMID(2)

                  ZSIGK(1)=4.8
                  RAMID(1)=0.0
                  ZHRC(1)=12.34
                  IF(ZSIGK(1) > 0.8) THEN
                          ZHRC(1)=RAMID(1)+(1.0-RAMID(1))*((ZSIGK(1)-0.8)/0.2)**2
                  ENDIF
                   d(1,1,2)=ZHRC(1)                 
                   END SUBROUTINE pow1_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "pow1_test")
    sdfg.simplify(verbose=True)
    d = np.full([3, 4, 5], 42, order="F", dtype=np.float64)
    sdfg(d=d)
    assert (d[0, 0, 1] == 400)


def test_fortran_frontend_pow2():
    test_string = """
                    PROGRAM pow2_test
                    implicit none
                    double precision d(3,4,5)
                    CALL pow2_test_function(d)
                    end

                    SUBROUTINE pow2_test_function(d)
                   double precision d(3,4,5)
                  double precision :: ZSIGK(2), ZHRC(2),RAMID(2)

                  ZSIGK(1)=4.8
                  RAMID(1)=0.0
                  ZHRC(1)=12.34
                  IF(ZSIGK(1) > 0.8) THEN
                          ZHRC(1)=RAMID(1)+(1.0-RAMID(1))*((ZSIGK(1)-0.8)/0.01)**1.5
                  ENDIF
                   d(1,1,2)=ZHRC(1)                 
                   END SUBROUTINE pow2_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "pow2_test")
    sdfg.simplify(verbose=True)
    d = np.full([3, 4, 5], 42, order="F", dtype=np.float64)
    sdfg(d=d)
    assert (d[0, 0, 1] == 8000)


def test_fortran_frontend_sign1():
    test_string = """
                    PROGRAM sign1_test
                    implicit none
                    double precision d(3,4,5)
                    CALL sign1_test_function(d)
                    end

                    SUBROUTINE sign1_test_function(d)
                   double precision d(3,4,5)
                  double precision :: ZSIGK(2), ZHRC(2),RAMID(2)

                  ZSIGK(1)=4.8
                  RAMID(1)=0.0
                  ZHRC(1)=-12.34
                   d(1,1,2)=SIGN(ZSIGK(1),ZHRC(1))                 
                   END SUBROUTINE sign1_test_function
                    """
    sdfg = fortran_parser.create_sdfg_from_string(test_string, "sign1_test")
    sdfg.simplify(verbose=True)
    d = np.full([3, 4, 5], 42, order="F", dtype=np.float64)
    sdfg(d=d)
    assert (d[0, 0, 1] == -4.8)


if __name__ == "__main__":

    test_fortran_frontend_real_kind_selector()
    test_fortran_frontend_if1()
    test_fortran_frontend_loop1()
    test_fortran_frontend_function_statement1()

    test_fortran_frontend_pow1()
    test_fortran_frontend_pow2()
    test_fortran_frontend_sign1()
