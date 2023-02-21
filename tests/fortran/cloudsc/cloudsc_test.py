# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.frontend.fortran import fortran_parser
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes import RemoveUnusedSymbols
from importlib import import_module
import numpy as np
from numpy import f2py
import os
import sys
import tempfile


def get_fortran(source: str, program_name: str, subroutine_name: str, fortran_extension: str = '.f90'):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cwd = os.getcwd()
        os.chdir(tmp_dir)
        f2py.compile(source, modulename=program_name, verbose=True, extension=fortran_extension)
        sys.path.append(tmp_dir)
        module = import_module(program_name)
        function = getattr(module, subroutine_name)
        os.chdir(cwd)
        return function


def get_sdfg(source: str, program_name: str) -> dace.SDFG:

    intial_sdfg = fortran_parser.create_sdfg_from_string(source, program_name)
    
    # Find first NestedSDFG
    sdfg = None
    for state in intial_sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                sdfg = node.sdfg
                break
    if not sdfg:
        raise ValueError("SDFG not found.")

    sdfg.parent = None
    sdfg.parent_sdfg = None
    sdfg.parent_nsdfg_node = None
    sdfg.reset_sdfg_list()

    my_simplify = Pipeline([RemoveUnusedSymbols()])
    my_simplify.apply_pass(sdfg, {})

    return sdfg


def test_enthalpy_flux_due_to_precipitation():

    fsource = """
PROGRAM enthalpy_flux_due_to_precipitation

    INTEGER, PARAMETER :: JPIM = SELECTED_INT_KIND(9)
    ! INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13, 300)

    INTEGER(KIND=JPIM), PARAMETER  :: KLON = 100
    INTEGER(KIND=JPIM), PARAMETER  :: KLEV = 100

    INTEGER(KIND=JPIM)  :: KIDIA 
    INTEGER(KIND=JPIM)  :: KFDIA 
    ! REAL(KIND=JPRB)     :: RLVTT
    ! REAL(KIND=JPRB)     :: RLSTT
    ! REAL(KIND=JPRB)     :: PFPLSL(KLON,KLEV+1)
    ! REAL(KIND=JPRB)     :: PFPLSN(KLON,KLEV+1)
    ! REAL(KIND=JPRB)     :: PFHPSL(KLON,KLEV+1)
    ! REAL(KIND=JPRB)     :: PFHPSN(KLON,KLEV+1)
    DOUBLE PRECISION RLVTT
    DOUBLE PRECISION RLSTT
    DOUBLE PRECISION PFPLSL(KLON,KLEV+1)
    DOUBLE PRECISION PFPLSN(KLON,KLEV+1)
    DOUBLE PRECISION PFHPSL(KLON,KLEV+1)
    DOUBLE PRECISION PFHPSN(KLON,KLEV+1)

    CALL enthalpy_flux_due_to_precipitation_routine(KLON, KLEV, KIDIA, KFDIA, RLVTT, RLSTT, PFPLSL, PFPLSN, PFHPSL, PFHPSN)

END

SUBROUTINE enthalpy_flux_due_to_precipitation_routine(KLON, KLEV, KIDIA, KFDIA, RLVTT, RLSTT, PFPLSL, PFPLSN, PFHPSL, PFHPSN)

    INTEGER, PARAMETER :: JPIM = SELECTED_INT_KIND(9)
    ! INTEGER, PARAMETER :: JPRB = SELECTED_REAL_KIND(13, 300)

    INTEGER(KIND=JPIM)  :: KLON
    INTEGER(KIND=JPIM)  :: KLEV
    INTEGER(KIND=JPIM)  :: KIDIA 
    INTEGER(KIND=JPIM)  :: KFDIA 
    ! REAL(KIND=JPRB)     :: RLVTT
    ! REAL(KIND=JPRB)     :: RLSTT
    ! REAL(KIND=JPRB)     :: PFPLSL(KLON,KLEV+1)
    ! REAL(KIND=JPRB)     :: PFPLSN(KLON,KLEV+1)
    ! REAL(KIND=JPRB)     :: PFHPSL(KLON,KLEV+1)
    ! REAL(KIND=JPRB)     :: PFHPSN(KLON,KLEV+1)
    DOUBLE PRECISION RLVTT
    DOUBLE PRECISION RLSTT
    DOUBLE PRECISION PFPLSL(KLON,KLEV+1)
    DOUBLE PRECISION PFPLSN(KLON,KLEV+1)
    DOUBLE PRECISION PFHPSL(KLON,KLEV+1)
    DOUBLE PRECISION PFHPSN(KLON,KLEV+1)

    INTEGER(KIND=JPIM)  :: JK, JL

    DO JK=1,KLEV+1
        DO JL=KIDIA,KFDIA
            PFHPSL(JL,JK) = -RLVTT*PFPLSL(JL,JK)
            PFHPSN(JL,JK) = -RLSTT*PFPLSN(JL,JK)
        ENDDO
    ENDDO

END SUBROUTINE enthalpy_flux_due_to_precipitation_routine
    """

    ffunc = get_fortran(fsource, 'enthalpy_flux_due_to_precipitation', 'enthalpy_flux_due_to_precipitation_routine')
    sdfg = get_sdfg(fsource, 'enthalpy_flux_due_to_precipitation')

    rng = np.random.default_rng(42)

    klon = 10
    klev = 10
    kidia = 2
    kfdia = 8

    rlvtt = rng.random()
    rlstt = rng.random()

    pfplsl = np.asfortranarray(rng.random((klon, klev+1)))
    pfplsn = np.asfortranarray(rng.random((klon, klev+1)))

    pfhpsl_f = np.zeros((klon, klev+1), order="F")
    pfhpsn_f = np.zeros((klon, klev+1), order="F")
    pfhpsl_d = np.zeros((klon, klev+1), order="F")
    pfhpsn_d = np.zeros((klon, klev+1), order="F")

    ffunc(klon=klon, klev=klev, kidia=kidia, kfdia=kfdia, rlvtt=rlvtt, rlstt=rlstt, pfplsl=pfplsl, pfplsn=pfplsn, pfhpsl=pfhpsl_f, pfhpsn=pfhpsn_f)
    sdfg(KLON=klon, KLEV=klev, KIDIA=kidia, KFDIA=kfdia, RLVTT=rlvtt, RLSTT=rlstt, PFPLSL=pfplsl, PFPLSN=pfplsn, PFHPSL=pfhpsl_d, PFHPSN=pfhpsn_d)

    assert np.allclose(pfhpsl_f, pfhpsl_d)
    assert np.allclose(pfhpsn_f, pfhpsn_d)

if __name__ == "__main__":
    test_enthalpy_flux_due_to_precipitation()
