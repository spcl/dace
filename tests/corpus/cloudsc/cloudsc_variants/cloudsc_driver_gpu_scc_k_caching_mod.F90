! (C) Copyright 1988- ECMWF.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
!
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation
! nor does it submit to any jurisdiction.

MODULE CLOUDSC_DRIVER_GPU_SCC_K_CACHING_MOD

  USE PARKIND1, ONLY: JPIM, JPRB, JPRL
  USE YOMPHYDER, ONLY: STATE_TYPE
  USE YOECLDP, ONLY : NCLV, YRECLDP, TECLDP
  USE CLOUDSC_MPI_MOD, ONLY: NUMPROC, IRANK
  USE TIMER_MOD, ONLY : PERFORMANCE_TIMER, GET_THREAD_NUM

  USE CLOUDSC_GPU_SCC_K_CACHING_MOD, ONLY: CLOUDSC_SCC_K_CACHING

  IMPLICIT NONE

CONTAINS

  SUBROUTINE CLOUDSC_DRIVER_GPU_SCC_K_CACHING( &
     & NUMOMP, NPROMA, NLEV, NGPTOT, NGPBLKS, NGPTOTG, KFLDX, PTSPHY, &
     & PT, PQ, &
     & BUFFER_CML, BUFFER_TMP, BUFFER_LOC, &
     & PVFA, PVFL, PVFI, PDYNA, PDYNL, PDYNI, &
     & PHRSW,    PHRLW, &
     & PVERVEL,  PAP,      PAPH, &
     & PLSM,     LDCUM,    KTYPE, &
     & PLU,      PLUDE,    PSNDE,    PMFU,     PMFD, &
     & PA, &
     & PCLV,     PSUPSAT,&
     & PLCRIT_AER,PICRIT_AER, PRE_ICE, &
     & PCCN,     PNICE,&
     & PCOVPTOT, PRAINFRAC_TOPRFZ, &
     & PFSQLF,   PFSQIF ,  PFCQNNG,  PFCQLNG, &
     & PFSQRF,   PFSQSF ,  PFCQRNG,  PFCQSNG, &
     & PFSQLTUR, PFSQITUR, &
     & PFPLSL,   PFPLSN,   PFHPSL,   PFHPSN &
     & )
    ! Driver routine that invokes the optimized CLAW-based CLOUDSC GPU kernel

    INTEGER(KIND=JPIM)             :: JL
    INTEGER(KIND=JPIM)             :: NUMOMP, NPROMA, NLEV, NGPTOT, NGPBLKS, NGPTOTG
    INTEGER(KIND=JPIM)             :: KFLDX
    REAL(KIND=JPRB)                :: PTSPHY       ! Physics timestep
    REAL(KIND=JPRB), INTENT(IN)    :: PT(NPROMA, NLEV, NGPBLKS) ! T at start of callpar
    REAL(KIND=JPRB), INTENT(IN)    :: PQ(NPROMA, NLEV, NGPBLKS) ! Q at start of callpar
    REAL(KIND=JPRB), INTENT(INOUT) :: BUFFER_CML(NPROMA,NLEV,3+NCLV,NGPBLKS) ! Storage buffer for TENDENCY_CML
    REAL(KIND=JPRB), INTENT(INOUT) :: BUFFER_TMP(NPROMA,NLEV,3+NCLV,NGPBLKS) ! Storage buffer for TENDENCY_TMP
    REAL(KIND=JPRB), INTENT(INOUT) :: BUFFER_LOC(NPROMA,NLEV,3+NCLV,NGPBLKS) ! Storage buffer for TENDENCY_LOC
    REAL(KIND=JPRB), INTENT(IN)    :: PVFA(NPROMA, NLEV, NGPBLKS)     ! CC from VDF scheme
    REAL(KIND=JPRB), INTENT(IN)    :: PVFL(NPROMA, NLEV, NGPBLKS)     ! Liq from VDF scheme
    REAL(KIND=JPRB), INTENT(IN)    :: PVFI(NPROMA, NLEV, NGPBLKS)     ! Ice from VDF scheme
    REAL(KIND=JPRB), INTENT(IN)    :: PDYNA(NPROMA, NLEV, NGPBLKS)    ! CC from Dynamics
    REAL(KIND=JPRB), INTENT(IN)    :: PDYNL(NPROMA, NLEV, NGPBLKS)    ! Liq from Dynamics
    REAL(KIND=JPRB), INTENT(IN)    :: PDYNI(NPROMA, NLEV, NGPBLKS)    ! Liq from Dynamics
    REAL(KIND=JPRB), INTENT(IN)    :: PHRSW(NPROMA, NLEV, NGPBLKS)    ! Short-wave heating rate
    REAL(KIND=JPRB), INTENT(IN)    :: PHRLW(NPROMA, NLEV, NGPBLKS)    ! Long-wave heating rate
    REAL(KIND=JPRB), INTENT(IN)    :: PVERVEL(NPROMA, NLEV, NGPBLKS)  !Vertical velocity
    REAL(KIND=JPRL), INTENT(IN)    :: PAP(NPROMA, NLEV, NGPBLKS)      ! Pressure on full levels  ! SC2026: JPRL — value(s) exceed FP16 max
    REAL(KIND=JPRL), INTENT(IN)    :: PAPH(NPROMA, NLEV+1, NGPBLKS) ! Pressure on half levels  ! SC2026: JPRL — value(s) exceed FP16 max
    REAL(KIND=JPRB), INTENT(IN)    :: PLSM(NPROMA, NGPBLKS)    ! Land fraction (0-1)
    LOGICAL, INTENT(IN)            :: LDCUM(NPROMA, NGPBLKS)    ! Convection active
    INTEGER(KIND=JPIM), INTENT(IN) :: KTYPE(NPROMA, NGPBLKS)    ! Convection type 0,1,2
    REAL(KIND=JPRB), INTENT(IN)    :: PLU(NPROMA, NLEV, NGPBLKS)      ! Conv. condensate
    REAL(KIND=JPRB), INTENT(INOUT) :: PLUDE(NPROMA, NLEV, NGPBLKS)    ! Conv. detrained water
    REAL(KIND=JPRB), INTENT(IN)    :: PSNDE(NPROMA, NLEV, NGPBLKS)    ! Conv. detrained snow
    REAL(KIND=JPRB), INTENT(IN)    :: PMFU(NPROMA, NLEV, NGPBLKS)     ! Conv. mass flux up
    REAL(KIND=JPRB), INTENT(IN)    :: PMFD(NPROMA, NLEV, NGPBLKS)     ! Conv. mass flux down
    REAL(KIND=JPRB), INTENT(IN)    :: PA(NPROMA, NLEV, NGPBLKS)       ! Original Cloud fraction (t)
    REAL(KIND=JPRL), INTENT(IN)    :: PCLV(NPROMA, NLEV, NCLV, NGPBLKS)  ! SC2026: JPRL — values underflow FP16
    REAL(KIND=JPRB), INTENT(IN)    :: PSUPSAT(NPROMA, NLEV, NGPBLKS)
    REAL(KIND=JPRB), INTENT(IN)    :: PLCRIT_AER(NPROMA, NLEV, NGPBLKS)
    REAL(KIND=JPRB), INTENT(IN)    :: PICRIT_AER(NPROMA, NLEV, NGPBLKS)
    REAL(KIND=JPRB), INTENT(IN)    :: PRE_ICE(NPROMA, NLEV, NGPBLKS)
    REAL(KIND=JPRB), INTENT(IN)    :: PCCN(NPROMA, NLEV, NGPBLKS)     ! liquid cloud condensation nuclei
    REAL(KIND=JPRB), INTENT(IN)    :: PNICE(NPROMA, NLEV, NGPBLKS)    ! ice number concentration (cf. CCN)

    REAL(KIND=JPRB), INTENT(INOUT) :: PCOVPTOT(NPROMA, NLEV, NGPBLKS)    ! Precip fraction
    REAL(KIND=JPRB), INTENT(OUT) :: PRAINFRAC_TOPRFZ(NPROMA, NGPBLKS)
    ! Flux diagnostics for DDH budget
    REAL(KIND=JPRB), INTENT(OUT) :: PFSQLF(NPROMA, NLEV+1, NGPBLKS)    ! Flux of liquid
    REAL(KIND=JPRB), INTENT(OUT) :: PFSQIF(NPROMA, NLEV+1, NGPBLKS)    ! Flux of ice
    REAL(KIND=JPRB), INTENT(OUT) :: PFCQLNG(NPROMA, NLEV+1, NGPBLKS)   ! -ve corr for liq
    REAL(KIND=JPRB), INTENT(OUT) :: PFCQNNG(NPROMA, NLEV+1, NGPBLKS)   ! -ve corr for ice
    REAL(KIND=JPRB), INTENT(OUT) :: PFSQRF(NPROMA, NLEV+1, NGPBLKS)    ! Flux diagnostics
    REAL(KIND=JPRB), INTENT(OUT) :: PFSQSF(NPROMA, NLEV+1, NGPBLKS)    !    for DDH, generic
    REAL(KIND=JPRB), INTENT(OUT) :: PFCQRNG(NPROMA, NLEV+1, NGPBLKS)   ! rain
    REAL(KIND=JPRB), INTENT(OUT) :: PFCQSNG(NPROMA, NLEV+1, NGPBLKS)   ! snow
    REAL(KIND=JPRB), INTENT(OUT) :: PFSQLTUR(NPROMA, NLEV+1, NGPBLKS)  ! liquid flux due to VDF
    REAL(KIND=JPRB), INTENT(OUT) :: PFSQITUR(NPROMA, NLEV+1, NGPBLKS)  ! ice flux due to VDF
    REAL(KIND=JPRB), INTENT(OUT) :: PFPLSL(NPROMA, NLEV+1, NGPBLKS)    ! liq+rain sedim flux
    REAL(KIND=JPRB), INTENT(OUT) :: PFPLSN(NPROMA, NLEV+1, NGPBLKS)    ! ice+snow sedim flux
    REAL(KIND=JPRB), INTENT(OUT) :: PFHPSL(NPROMA, NLEV+1, NGPBLKS)    ! Enthalpy flux for liq
    REAL(KIND=JPRB), INTENT(OUT) :: PFHPSN(NPROMA, NLEV+1, NGPBLKS)    ! ice number concentration (cf. CCN)

    INTEGER(KIND=JPIM) :: JKGLO,IBL,ICEND
    TYPE(PERFORMANCE_TIMER) :: TIMER
    INTEGER(KIND=JPIM) :: TID ! thread id from 0 .. NUMOMP - 1

    ! Local copy of cloud parameters for offload
    TYPE(TECLDP) :: LOCAL_YRECLDP

    NGPBLKS = (NGPTOT / NPROMA) + MIN(MOD(NGPTOT,NPROMA), 1)
1003 format(5x,'NUMPROC=',i0,', NUMOMP=',i0,', NGPTOTG=',i0,', NPROMA=',i0,', NGPBLKS=',i0)
    if (irank == 0) then
      write(0,1003) NUMPROC,NUMOMP,NGPTOTG,NPROMA,NGPBLKS
    end if

    ! Global timer for the parallel region
    CALL TIMER%START(NUMOMP)

    ! Workaround for PGI / OpenACC oddities:
    ! Create a local copy of the parameter struct to ensure they get
    ! moved to the device the in ``acc data`` clause below
    LOCAL_YRECLDP = YRECLDP

!$acc data &
!$acc copyin(local_yrecldp) &
!$acc present( &
!$acc   pt,pq,buffer_cml,buffer_tmp,buffer_loc,pvfa, &
!$acc   pvfl,pvfi,pdyna,pdynl,pdyni,phrsw,phrlw,pvervel, &
!$acc   pap,paph,plsm,ldcum,ktype,plu,plude,psnde, &
!$acc   pmfu,pmfd,pa,pclv,psupsat,plcrit_aer,picrit_aer, &
!$acc   pre_ice,pccn,pnice,pcovptot,prainfrac_toprfz, &
!$acc   pfsqlf,pfsqif,pfcqnng, &
!$acc   pfcqlng,pfsqrf,pfsqsf,pfcqrng,pfcqsng,pfsqltur, &
!$acc   pfsqitur,pfplsl,pfplsn,pfhpsl,pfhpsn)

    ! Local timer for each thread
    TID = GET_THREAD_NUM()
    CALL TIMER%THREAD_START(TID)

!$acc parallel loop gang vector_length(NPROMA)
    DO JKGLO=1,NGPTOT,NPROMA
       IBL=(JKGLO-1)/NPROMA+1
       ICEND=MIN(NPROMA,NGPTOT-JKGLO+1)

      !$acc loop vector
      DO JL=1,ICEND
       CALL CLOUDSC_SCC_K_CACHING &
        & (1, ICEND, NPROMA, NLEV, PTSPHY,&
        & PT(:,:,IBL), PQ(:,:,IBL), &
        & BUFFER_TMP(:,:,1,IBL), BUFFER_TMP(:,:,3,IBL), BUFFER_TMP(:,:,2,IBL), BUFFER_TMP(:,:,4:8,IBL), &
        & BUFFER_LOC(:,:,1,IBL), BUFFER_LOC(:,:,3,IBL), BUFFER_LOC(:,:,2,IBL), BUFFER_LOC(:,:,4:8,IBL), &
        & PVFA(:,:,IBL), PVFL(:,:,IBL), PVFI(:,:,IBL), PDYNA(:,:,IBL), PDYNL(:,:,IBL), PDYNI(:,:,IBL), &
        & PHRSW(:,:,IBL),    PHRLW(:,:,IBL),&
        & PVERVEL(:,:,IBL),  PAP(:,:,IBL),      PAPH(:,:,IBL),&
        & PLSM(:,IBL),       LDCUM(:,IBL),      KTYPE(:,IBL), &
        & PLU(:,:,IBL),      PLUDE(:,:,IBL),    PSNDE(:,:,IBL),    PMFU(:,:,IBL),     PMFD(:,:,IBL),&
        !---prognostic fields
        & PA(:,:,IBL),       PCLV(:,:,:,IBL),   PSUPSAT(:,:,IBL),&
        !-- arrays for aerosol-cloud interactions
        & PLCRIT_AER(:,:,IBL),PICRIT_AER(:,:,IBL),&
        & PRE_ICE(:,:,IBL),&
        & PCCN(:,:,IBL),     PNICE(:,:,IBL),&
        !---diagnostic output
        & PCOVPTOT(:,:,IBL), PRAINFRAC_TOPRFZ(:,IBL),&
        !---resulting fluxes
        & PFSQLF(:,:,IBL),   PFSQIF (:,:,IBL),  PFCQNNG(:,:,IBL),  PFCQLNG(:,:,IBL),&
        & PFSQRF(:,:,IBL),   PFSQSF (:,:,IBL),  PFCQRNG(:,:,IBL),  PFCQSNG(:,:,IBL),&
        & PFSQLTUR(:,:,IBL), PFSQITUR (:,:,IBL), &
        & PFPLSL(:,:,IBL),   PFPLSN(:,:,IBL),   PFHPSL(:,:,IBL),   PFHPSN(:,:,IBL),&
        & YRECLDP=LOCAL_YRECLDP, JL=JL)
      ENDDO
    ENDDO
!$acc end parallel loop
!$acc wait

    CALL TIMER%THREAD_END(TID)

!$acc end data

    CALL TIMER%END()

    ! On GPUs, adding block-level column totals is cumbersome and
    ! error prone, and of little value due to the large number of
    ! processing "thread teams". Instead we register the total here.
    CALL TIMER%THREAD_LOG(TID=TID, IGPC=NGPTOT)

    CALL TIMER%PRINT_PERFORMANCE(NPROMA, NGPBLKS, NGPTOT)

  END SUBROUTINE CLOUDSC_DRIVER_GPU_SCC_K_CACHING

END MODULE CLOUDSC_DRIVER_GPU_SCC_K_CACHING_MOD
