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


def test_fluxes():

    fsource = """
PROGRAM fluxes

    INTEGER, PARAMETER :: JPIM = SELECTED_INT_KIND(9)

    INTEGER(KIND=JPIM), PARAMETER  :: KLON = 100
    INTEGER(KIND=JPIM), PARAMETER  :: KLEV = 100

    INTEGER(KIND=JPIM)  :: KIDIA 
    INTEGER(KIND=JPIM)  :: KFDIA 
    ! DOUBLE PRECISION RLVTT
    ! DOUBLE PRECISION RLSTT
    ! DOUBLE PRECISION PFPLSL(KLON,KLEV+1)
    ! DOUBLE PRECISION PFPLSN(KLON,KLEV+1)
    ! DOUBLE PRECISION PFHPSL(KLON,KLEV+1)
    ! DOUBLE PRECISION PFHPSN(KLON,KLEV+1)

    DOUBLE PRECISION PFSQLF(KLON,KLEV+1)
    DOUBLE PRECISION PFSQIF(KLON,KLEV+1)
    DOUBLE PRECISION PFCQLNG(KLON,KLEV+1)
    DOUBLE PRECISION PFCQNNG(KLON,KLEV+1)
    DOUBLE PRECISION PFSQRF(KLON,KLEV+1)
    DOUBLE PRECISION PFSQSF(KLON,KLEV+1)
    DOUBLE PRECISION PFCQRNG(KLON,KLEV+1)
    DOUBLE PRECISION PFCQSNG(KLON,KLEV+1)
    DOUBLE PRECISION PFSQLTUR(KLON,KLEV+1)
    DOUBLE PRECISION PFSQITUR(KLON,KLEV+1)

    DOUBLE PRECISION PVFL(KLON,KLEV)
    DOUBLE PRECISION PVFI(KLON,KLEV)
    DOUBLE PRECISION PAPH(KLON,KLEV+1)
    DOUBLE PRECISION PLUDE(KLON,KLEV) 

    INTEGER(KIND=JPIM), PARAMETER  :: NCLV = 100
    INTEGER(KIND=JPIM)  :: NCLDQL
    INTEGER(KIND=JPIM)  :: NCLDQI
    INTEGER(KIND=JPIM)  :: NCLDQR
    INTEGER(KIND=JPIM)  :: NCLDQS
    INTEGER(KIND=JPIM)  :: NCLDQV

    DOUBLE PRECISION ZQX0(KLON,KLEV,NCLV)
    DOUBLE PRECISION ZLNEG(KLON,KLEV,NCLV)
    DOUBLE PRECISION ZQXN2D(KLON,KLEV,NCLV)
    DOUBLE PRECISION ZFOEALFA(KLON,KLEV+1) 

    DOUBLE PRECISION ZRG_R, ZQTMST, PTSPHY


    CALL fluxes_routine(&
        & KLON, KLEV, KIDIA, KFDIA,&
        & PFSQLF,   PFSQIF ,  PFCQNNG,  PFCQLNG,&
        & PFSQRF,   PFSQSF ,  PFCQRNG,  PFCQSNG,&
        & PFSQLTUR, PFSQITUR ,&
        & NCLV, NCLDQL, NCLDQI, NCLDQR, NCLDQS, NCLDQV,&
        & PVFL, PVFI, PAPH, PLUDE,&
        & ZQX0, ZLNEG, ZQXN2D, ZFOEALFA,&
        & ZRG_R, ZQTMST, PTSPHY)

END

SUBROUTINE fluxes_routine(&
    & KLON, KLEV, KIDIA, KFDIA,&
    & PFSQLF,   PFSQIF ,  PFCQNNG,  PFCQLNG,&
    & PFSQRF,   PFSQSF ,  PFCQRNG,  PFCQSNG,&
    & PFSQLTUR, PFSQITUR ,&
    & NCLV, NCLDQL, NCLDQI, NCLDQR, NCLDQS, NCLDQV,&
    & PVFL, PVFI, PAPH, PLUDE,&
    & ZQX0, ZLNEG, ZQXN2D, ZFOEALFA,&
    & ZRG_R, ZQTMST, PTSPHY)

    INTEGER, PARAMETER :: JPIM = SELECTED_INT_KIND(9)

    INTEGER(KIND=JPIM)  :: KLON
    INTEGER(KIND=JPIM)  :: KLEV
    INTEGER(KIND=JPIM)  :: KIDIA 
    INTEGER(KIND=JPIM)  :: KFDIA 
    ! DOUBLE PRECISION RLVTT
    ! DOUBLE PRECISION RLSTT
    ! DOUBLE PRECISION PFPLSL(KLON,KLEV+1)
    ! DOUBLE PRECISION PFPLSN(KLON,KLEV+1)
    ! DOUBLE PRECISION PFHPSL(KLON,KLEV+1)
    ! DOUBLE PRECISION PFHPSN(KLON,KLEV+1)

    DOUBLE PRECISION PFSQLF(KLON,KLEV+1)
    DOUBLE PRECISION PFSQIF(KLON,KLEV+1)
    DOUBLE PRECISION PFCQLNG(KLON,KLEV+1)
    DOUBLE PRECISION PFCQNNG(KLON,KLEV+1)
    DOUBLE PRECISION PFSQRF(KLON,KLEV+1)
    DOUBLE PRECISION PFSQSF(KLON,KLEV+1)
    DOUBLE PRECISION PFCQRNG(KLON,KLEV+1)
    DOUBLE PRECISION PFCQSNG(KLON,KLEV+1)
    DOUBLE PRECISION PFSQLTUR(KLON,KLEV+1)
    DOUBLE PRECISION PFSQITUR(KLON,KLEV+1)

    DOUBLE PRECISION PVFL(KLON,KLEV)
    DOUBLE PRECISION PVFI(KLON,KLEV)
    DOUBLE PRECISION PAPH(KLON,KLEV+1)
    DOUBLE PRECISION PLUDE(KLON,KLEV) 

    INTEGER(KIND=JPIM)  :: NCLV
    INTEGER(KIND=JPIM)  :: NCLDQL
    INTEGER(KIND=JPIM)  :: NCLDQI
    INTEGER(KIND=JPIM)  :: NCLDQR
    INTEGER(KIND=JPIM)  :: NCLDQS
    INTEGER(KIND=JPIM)  :: NCLDQV

    DOUBLE PRECISION ZQX0(KLON,KLEV,NCLV)
    DOUBLE PRECISION ZLNEG(KLON,KLEV,NCLV)
    DOUBLE PRECISION ZQXN2D(KLON,KLEV,NCLV)
    DOUBLE PRECISION ZFOEALFA(KLON,KLEV+1) 

    DOUBLE PRECISION ZRG_R, ZGDPH_R, ZQTMST, ZALFAW, PTSPHY

    INTEGER(KIND=JPIM)  :: JK, JL

    DO JL=KIDIA,KFDIA
        PFSQLF(JL,1)  = 0.0
        PFSQIF(JL,1)  = 0.0
        PFSQRF(JL,1)  = 0.0
        PFSQSF(JL,1)  = 0.0
        PFCQLNG(JL,1) = 0.0
        PFCQNNG(JL,1) = 0.0
        PFCQRNG(JL,1) = 0.0 !rain
        PFCQSNG(JL,1) = 0.0 !snow
        ! fluxes due to turbulence
        PFSQLTUR(JL,1) = 0.0
        PFSQITUR(JL,1) = 0.0
    ENDDO

    DO JK=1,KLEV
        DO JL=KIDIA,KFDIA

            ZGDPH_R = -ZRG_R*(PAPH(JL,JK+1)-PAPH(JL,JK))*ZQTMST
            PFSQLF(JL,JK+1)  = PFSQLF(JL,JK)
            PFSQIF(JL,JK+1)  = PFSQIF(JL,JK)
            PFSQRF(JL,JK+1)  = PFSQLF(JL,JK)
            PFSQSF(JL,JK+1)  = PFSQIF(JL,JK)
            PFCQLNG(JL,JK+1) = PFCQLNG(JL,JK)
            PFCQNNG(JL,JK+1) = PFCQNNG(JL,JK)
            PFCQRNG(JL,JK+1) = PFCQLNG(JL,JK)
            PFCQSNG(JL,JK+1) = PFCQNNG(JL,JK)
            PFSQLTUR(JL,JK+1) = PFSQLTUR(JL,JK)
            PFSQITUR(JL,JK+1) = PFSQITUR(JL,JK)

            ZALFAW=ZFOEALFA(JL,JK)

            ! Liquid , LS scheme minus detrainment
            PFSQLF(JL,JK+1)=PFSQLF(JL,JK+1)+ &
            &(ZQXN2D(JL,JK,NCLDQL)-ZQX0(JL,JK,NCLDQL)+PVFL(JL,JK)*PTSPHY-ZALFAW*PLUDE(JL,JK))*ZGDPH_R
            ! liquid, negative numbers
            PFCQLNG(JL,JK+1)=PFCQLNG(JL,JK+1)+ZLNEG(JL,JK,NCLDQL)*ZGDPH_R

            ! liquid, vertical diffusion
            PFSQLTUR(JL,JK+1)=PFSQLTUR(JL,JK+1)+PVFL(JL,JK)*PTSPHY*ZGDPH_R

            ! Rain, LS scheme 
            PFSQRF(JL,JK+1)=PFSQRF(JL,JK+1)+(ZQXN2D(JL,JK,NCLDQR)-ZQX0(JL,JK,NCLDQR))*ZGDPH_R 
            ! rain, negative numbers
            PFCQRNG(JL,JK+1)=PFCQRNG(JL,JK+1)+ZLNEG(JL,JK,NCLDQR)*ZGDPH_R

            ! Ice , LS scheme minus detrainment
            PFSQIF(JL,JK+1)=PFSQIF(JL,JK+1)+ &
            & (ZQXN2D(JL,JK,NCLDQI)-ZQX0(JL,JK,NCLDQI)+PVFI(JL,JK)*PTSPHY-(1.0-ZALFAW)*PLUDE(JL,JK))*ZGDPH_R
            ! ice, negative numbers
            PFCQNNG(JL,JK+1)=PFCQNNG(JL,JK+1)+ZLNEG(JL,JK,NCLDQI)*ZGDPH_R

            ! ice, vertical diffusion
            PFSQITUR(JL,JK+1)=PFSQITUR(JL,JK+1)+PVFI(JL,JK)*PTSPHY*ZGDPH_R

            ! snow, LS scheme
            PFSQSF(JL,JK+1)=PFSQSF(JL,JK+1)+(ZQXN2D(JL,JK,NCLDQS)-ZQX0(JL,JK,NCLDQS))*ZGDPH_R 
            ! snow, negative numbers
            PFCQSNG(JL,JK+1)=PFCQSNG(JL,JK+1)+ZLNEG(JL,JK,NCLDQS)*ZGDPH_R
        ENDDO
    ENDDO

END SUBROUTINE fluxes_routine
    """

    ffunc = get_fortran(fsource, 'fluxes', 'fluxes_routine')
    sdfg = get_sdfg(fsource, 'fluxes')

    rng = np.random.default_rng(42)

    klon, klev, nclv = 10, 10, 10
    kidia, kfdia = 2, 8
    ncldql, ncldqi, ncldqr, ncldqs, ncldqv = 3, 4, 5, 6, 7

    inputs = dict()
    inputs['KLON'] = klon
    inputs['KLEV'] = klev
    inputs['KIDIA'] = kidia
    inputs['KFDIA'] = kfdia
    inputs['NCLV'] = nclv
    inputs['NCLDQL'] = ncldql
    inputs['NCLDQI'] = ncldqi
    inputs['NCLDQR'] = ncldqr
    inputs['NCLDQS'] = ncldqs
    inputs['NCLDQV'] = ncldqv

    for name in ('PAPH', 'ZFOEALFA'):
        inputs[name] = np.asfortranarray(rng.random((klon, klev+1)))
    for name in ('PVFL', 'PVFI', 'PLUDE'):
        inputs[name] = np.asfortranarray(rng.random((klon, klev)))
    for name in ('ZQX0', 'ZLNEG', 'ZQXN2D'):
        inputs[name] = np.asfortranarray(rng.random((klon, klev, nclv)))
    for name in ('ZRG_R', 'ZQTMST', 'PTSPHY'):
        inputs[name] = rng.random()
    
    outputs_f = dict()
    outputs_d = dict()

    for name in ('PFSQLF', 'PFSQIF', 'PFCQNNG', 'PFCQLNG', 'PFSQRF', 'PFSQSF', 'PFCQRNG', 'PFCQSNG', 'PFSQLTUR', 'PFSQITUR'):
        farr = np.asfortranarray(rng.random((klon, klev+1)))
        darr = np.copy(farr)
        outputs_f[name] = farr
        outputs_d[name] = darr

    ffunc(**{k.lower(): v for k, v in inputs.items()}, **{k.lower(): v for k, v in outputs_f.items()})
    sdfg(**inputs, **outputs_d)

    for k in outputs_f.keys():
        farr = outputs_f[k]
        darr = outputs_f[k]
        assert np.allclose(farr, darr)

if __name__ == "__main__":
    test_enthalpy_flux_due_to_precipitation()
    test_fluxes()
