! Multistep wrapper for CLOUDSC GPU SCC k-caching kernel.
!
! Runs the unmodified ECMWF cloudsc kernel in a forward-Euler time loop
! with fixed total physical time (PTSPHY from input.h5) subdivided into
! NSTEPS substeps.  Prognostic fields (PT, PQ, PA, PCLV) are dumped to
! HDF5 after every step for offline comparison.
!
! Usage:  dwarf-cloudsc-gpu-scc-k-caching-multistep NUMOMP NGPTOTG NPROMA NSTEPS [TPHYS] [NSUB]
!
! This file is NOT part of the upstream ECMWF dwarf-P-cloudsc distribution.
! It was written for the SC2026 precision study.

PROGRAM DWARF_CLOUDSC_MULTISTEP

USE PARKIND1, ONLY: JPIM, JPRB, JPRD, JPRL
USE YOECLDP, ONLY: NCLV
USE CLOUDSC_MPI_MOD, ONLY: CLOUDSC_MPI_INIT, CLOUDSC_MPI_END, NUMPROC, IRANK
USE CLOUDSC_GLOBAL_STATE_MOD, ONLY: CLOUDSC_GLOBAL_STATE
USE CLOUDSC_DRIVER_GPU_SCC_K_CACHING_MOD, ONLY: CLOUDSC_DRIVER_GPU_SCC_K_CACHING
USE CLOUDSC_OUTPUT_MOD, ONLY: CLOUDSC_OUTPUT_OPEN, CLOUDSC_OUTPUT_WRITE_STEP, CLOUDSC_OUTPUT_CLOSE

IMPLICIT NONE

CHARACTER(LEN=20) :: CLARG
INTEGER(KIND=JPIM) :: IARGS, LENARG

INTEGER(KIND=JPIM) :: NUMOMP   = 1      ! OpenMP threads (GPU: always 1)
INTEGER(KIND=JPIM) :: NGPTOTG  = 16384  ! Total grid-point columns
INTEGER(KIND=JPIM) :: NPROMA   = 64     ! Blocking factor
INTEGER(KIND=JPIM) :: NGPTOT            ! Local grid points (= NGPTOTG for single rank)
INTEGER(KIND=JPIM) :: NSTEPS   = 1      ! Number of outer physics steps
INTEGER(KIND=JPIM) :: NSUB     = 1      ! Number of sub-substeps within each outer step
INTEGER(KIND=JPIM) :: JSTEP             ! Outer step loop counter
INTEGER(KIND=JPIM) :: JSUB              ! Inner sub-substep loop counter
INTEGER(KIND=JPIM) :: JB, JK, JL, JM   ! Loop indices for state update
REAL(KIND=JPRB)    :: ZTSPHY_SUB        ! Outer substep dt = PTSPHY / NSTEPS
REAL(KIND=JPRB)    :: ZTSPHY_INNER      ! Inner dt = ZTSPHY_SUB / NSUB
CHARACTER(LEN=128) :: OUT_FILENAME      ! Output HDF5 filename
CHARACTER(LEN=128) :: TIMING_FILENAME   ! Timing CSV filename
CHARACTER(LEN=8)   :: PRECISION_TAG     ! 'fp16', 'fp32', or 'fp64'

! Timing
INTEGER(KIND=8)    :: ICLOCK_START, ICLOCK_STEP, ICLOCK_END, ICLOCK_RATE
INTEGER(KIND=8)    :: ICLOCK_H2D, ICLOCK_AUX, ICLOCK_SUB
REAL(KIND=JPRD)    :: ZTIME_TOTAL, ZTIME_STEP
REAL(KIND=JPRD)    :: ZTIME_H2D, ZTIME_KERNEL, ZTIME_UPDATE, ZTIME_D2H
INTEGER, PARAMETER :: IOTIMING = 42     ! Unit for timing CSV

! NaN diagnostics
INTEGER(KIND=JPIM) :: NNAN_T, NNAN_Q, NNAN_A, NNAN_CLD

TYPE(CLOUDSC_GLOBAL_STATE) :: GLOBAL_STATE

#include "abor1.intfb.h"

! --- Parse command-line arguments ---

IARGS = COMMAND_ARGUMENT_COUNT()

IF (IARGS >= 1) THEN
  CALL GET_COMMAND_ARGUMENT(1, CLARG, LENARG)
  READ(CLARG(1:LENARG),*) NUMOMP
  IF (NUMOMP <= 0) NUMOMP = 1
END IF

CALL CLOUDSC_MPI_INIT(NUMOMP)

IF (IARGS >= 2) THEN
  CALL GET_COMMAND_ARGUMENT(2, CLARG, LENARG)
  READ(CLARG(1:LENARG),*) NGPTOTG
END IF

NGPTOT = (NGPTOTG - 1) / NUMPROC + 1
IF (IRANK == NUMPROC - 1) THEN
  NGPTOT = NGPTOTG - (NUMPROC - 1) * NGPTOT
END IF

IF (IARGS >= 3) THEN
  CALL GET_COMMAND_ARGUMENT(3, CLARG, LENARG)
  READ(CLARG(1:LENARG),*) NPROMA
END IF

IF (IARGS >= 4) THEN
  CALL GET_COMMAND_ARGUMENT(4, CLARG, LENARG)
  READ(CLARG(1:LENARG),*) NSTEPS
  IF (NSTEPS < 1) NSTEPS = 1
END IF

! --- Load serialized input state ---

CALL GLOBAL_STATE%LOAD(NPROMA, NGPTOT, NGPTOTG)

! --- Override PTSPHY if 5th argument given ---

IF (IARGS >= 5) THEN
  CALL GET_COMMAND_ARGUMENT(5, CLARG, LENARG)
  READ(CLARG(1:LENARG),*) GLOBAL_STATE%PTSPHY
END IF

! --- Parse NSUB (6th argument, optional, default=1) ---

IF (IARGS >= 6) THEN
  CALL GET_COMMAND_ARGUMENT(6, CLARG, LENARG)
  READ(CLARG(1:LENARG),*) NSUB
  IF (NSUB < 1) NSUB = 1
END IF

! --- Compute substep dt ---

ZTSPHY_SUB = GLOBAL_STATE%PTSPHY / REAL(NSTEPS, JPRB)
ZTSPHY_INNER = ZTSPHY_SUB / REAL(NSUB, JPRB)

IF (IRANK == 0) THEN
  WRITE(0,'(1X,A)')           '============================================'
  WRITE(0,'(1X,A)')           '  CLOUDSC GPU SCC-k-caching  (multistep)'
  WRITE(0,'(1X,A,I0)')        '  NGPTOTG  = ', NGPTOTG
  WRITE(0,'(1X,A,I0)')        '  NPROMA   = ', NPROMA
  WRITE(0,'(1X,A,I0)')        '  NSTEPS   = ', NSTEPS
  WRITE(0,'(1X,A,I0)')        '  NSUB     = ', NSUB
  WRITE(0,'(1X,A,ES12.5)')    '  PTSPHY   = ', GLOBAL_STATE%PTSPHY
  WRITE(0,'(1X,A,ES12.5)')    '  dt_sub   = ', ZTSPHY_SUB
  WRITE(0,'(1X,A,ES12.5)')    '  dt_inner = ', ZTSPHY_INNER
#ifdef HALF
  WRITE(0,'(1X,A)')           '  Precision: FP16 (HALF)'
#elif defined(SINGLE)
  WRITE(0,'(1X,A)')           '  Precision: FP32 (SINGLE)'
#else
  WRITE(0,'(1X,A)')           '  Precision: FP64'
#endif
  WRITE(0,'(1X,A)')           '============================================'
END IF

! --- Open HDF5 output ---

IF (IRANK == 0) THEN
#ifdef HALF_RESTRICTED
  PRECISION_TAG = 'fp16r'
#elif defined(HALF)
  PRECISION_TAG = 'fp16'
#elif defined(SINGLE)
  PRECISION_TAG = 'fp32'
#else
  PRECISION_TAG = 'fp64'
#endif
  IF (NSUB == 1) THEN
    WRITE(OUT_FILENAME, '(A,A,A,I0,A,I0,A,I0,A)') &
      & 'cloudsc_output_', TRIM(PRECISION_TAG), '_', NSTEPS, 'steps_', NGPTOTG, 'col_', GLOBAL_STATE%KLEV, 'lev.h5'
  ELSE
    WRITE(OUT_FILENAME, '(A,A,A,I0,A,I0,A,I0,A,I0,A)') &
      & 'cloudsc_output_', TRIM(PRECISION_TAG), '_', NSTEPS, 'steps_', NGPTOTG, 'col_', GLOBAL_STATE%KLEV, 'lev_nsub', NSUB, '.h5'
  END IF
  CALL CLOUDSC_OUTPUT_OPEN(TRIM(OUT_FILENAME))

  ! Open timing CSV (one row per step + summary)
  IF (NSUB == 1) THEN
    WRITE(TIMING_FILENAME, '(A,A,A,I0,A,I0,A,I0,A)') &
      & 'cloudsc_timing_', TRIM(PRECISION_TAG), '_', NSTEPS, 'steps_', NGPTOTG, 'col_', GLOBAL_STATE%KLEV, 'lev.csv'
  ELSE
    WRITE(TIMING_FILENAME, '(A,A,A,I0,A,I0,A,I0,A,I0,A)') &
      & 'cloudsc_timing_', TRIM(PRECISION_TAG), '_', NSTEPS, 'steps_', NGPTOTG, 'col_', GLOBAL_STATE%KLEV, 'lev_nsub', NSUB, '.csv'
  END IF
  OPEN(UNIT=IOTIMING, FILE=TRIM(TIMING_FILENAME), STATUS='REPLACE', ACTION='WRITE')
  WRITE(IOTIMING, '(A)') 'step,wall_ms,kernel_ms,update_ms,d2h_ms'

  ! Write initial state as step 0
  CALL CLOUDSC_OUTPUT_WRITE_STEP(0, NPROMA, GLOBAL_STATE%KLEV, NCLV, &
    & GLOBAL_STATE%NBLOCKS, GLOBAL_STATE%PT, GLOBAL_STATE%PQ, &
    & GLOBAL_STATE%PA, GLOBAL_STATE%PCLV)
END IF

! =============================================
!  Time loop: forward Euler with CLOUDSC kernel
! =============================================
!
!  At each substep the unmodified kernel:
!    1. Predicts state:  ZTP1 = PT + dt * B_TMP   (applies external tendencies)
!    2. Runs cloud microphysics on predicted state
!    3. Stores local tendency dX/dt in B_LOC       (B_TMP untouched)
!
!  We then apply the full tendency to advance the prognostic state:
!    X_new = X + dt * (B_TMP + B_LOC)
!
!  B_TMP is constant (frozen external forcing from input.h5).
!  B_LOC is zeroed before each kernel call (kernel also zeroes internally,
!  but we clear the host buffer to be safe with the ACC copy-in).
!
!  This matches ICON's forward-Euler sequential-splitting integration
!  (cf. mo_nh_interface_nwp.f90, mo_util_phys.f90).

CALL SYSTEM_CLOCK(ICLOCK_START, ICLOCK_RATE)

! --- Persistent GPU data region ---
! All arrays are uploaded once (copyin) and stay on-device for the entire
! time loop.  The driver uses !$acc present(...) to access them.
! We only transfer back to host for HDF5 output and NaN checks.

CALL SYSTEM_CLOCK(ICLOCK_H2D)

!$acc data &
!$acc copy( &
!$acc   GLOBAL_STATE%PT, GLOBAL_STATE%PQ, &
!$acc   GLOBAL_STATE%PA, GLOBAL_STATE%PCLV) &
!$acc copyin( &
!$acc   GLOBAL_STATE%B_CML, GLOBAL_STATE%B_TMP, &
!$acc   GLOBAL_STATE%PVFA, GLOBAL_STATE%PVFL, GLOBAL_STATE%PVFI, &
!$acc   GLOBAL_STATE%PDYNA, GLOBAL_STATE%PDYNL, GLOBAL_STATE%PDYNI, &
!$acc   GLOBAL_STATE%PHRSW, GLOBAL_STATE%PHRLW, &
!$acc   GLOBAL_STATE%PVERVEL, GLOBAL_STATE%PAP, GLOBAL_STATE%PAPH, &
!$acc   GLOBAL_STATE%PLSM, GLOBAL_STATE%LDCUM, GLOBAL_STATE%KTYPE, &
!$acc   GLOBAL_STATE%PLU, GLOBAL_STATE%PSNDE, &
!$acc   GLOBAL_STATE%PMFU, GLOBAL_STATE%PMFD, &
!$acc   GLOBAL_STATE%PSUPSAT, &
!$acc   GLOBAL_STATE%PLCRIT_AER, GLOBAL_STATE%PICRIT_AER, &
!$acc   GLOBAL_STATE%PRE_ICE, &
!$acc   GLOBAL_STATE%PCCN, GLOBAL_STATE%PNICE) &
!$acc copy( &
!$acc   GLOBAL_STATE%B_LOC, GLOBAL_STATE%PLUDE, &
!$acc   GLOBAL_STATE%PCOVPTOT, GLOBAL_STATE%PRAINFRAC_TOPRFZ) &
!$acc create( &
!$acc   GLOBAL_STATE%PFSQLF, GLOBAL_STATE%PFSQIF, &
!$acc   GLOBAL_STATE%PFCQNNG, GLOBAL_STATE%PFCQLNG, &
!$acc   GLOBAL_STATE%PFSQRF, GLOBAL_STATE%PFSQSF, &
!$acc   GLOBAL_STATE%PFCQRNG, GLOBAL_STATE%PFCQSNG, &
!$acc   GLOBAL_STATE%PFSQLTUR, GLOBAL_STATE%PFSQITUR, &
!$acc   GLOBAL_STATE%PFPLSL, GLOBAL_STATE%PFPLSN, &
!$acc   GLOBAL_STATE%PFHPSL, GLOBAL_STATE%PFHPSN)

!$acc wait
CALL SYSTEM_CLOCK(ICLOCK_AUX)
ZTIME_H2D = REAL(ICLOCK_AUX - ICLOCK_H2D, JPRD) / REAL(ICLOCK_RATE, JPRD)
IF (IRANK == 0) THEN
  WRITE(0,'(1X,A,F10.3,A)') '  H2D transfer: ', ZTIME_H2D*1000.0_JPRD, ' ms'
  WRITE(IOTIMING, '(A,A,F12.4,A)') 'h2d', ',', ZTIME_H2D*1000.0_JPRD, ',,,'
END IF

! --- Warmup: one discarded kernel call to prime GPU caches and JIT ---
IF (IRANK == 0) WRITE(0,'(1X,A)') '  Warmup kernel call (discarded)...'
CALL CLOUDSC_DRIVER_GPU_SCC_K_CACHING(NUMOMP, NPROMA, GLOBAL_STATE%KLEV, &
     & NGPTOT, GLOBAL_STATE%NBLOCKS, NGPTOTG, &
     & GLOBAL_STATE%KFLDX, ZTSPHY_INNER, &
     & GLOBAL_STATE%PT, GLOBAL_STATE%PQ, &
     & GLOBAL_STATE%B_CML,   GLOBAL_STATE%B_TMP, GLOBAL_STATE%B_LOC, &
     & GLOBAL_STATE%PVFA,    GLOBAL_STATE%PVFL,  GLOBAL_STATE%PVFI, &
     & GLOBAL_STATE%PDYNA,   GLOBAL_STATE%PDYNL, GLOBAL_STATE%PDYNI, &
     & GLOBAL_STATE%PHRSW,   GLOBAL_STATE%PHRLW, &
     & GLOBAL_STATE%PVERVEL, GLOBAL_STATE%PAP,   GLOBAL_STATE%PAPH, &
     & GLOBAL_STATE%PLSM,    GLOBAL_STATE%LDCUM, GLOBAL_STATE%KTYPE, &
     & GLOBAL_STATE%PLU,     GLOBAL_STATE%PLUDE, GLOBAL_STATE%PSNDE, &
     & GLOBAL_STATE%PMFU,    GLOBAL_STATE%PMFD, &
     & GLOBAL_STATE%PA, &
     & GLOBAL_STATE%PCLV,    GLOBAL_STATE%PSUPSAT,&
     & GLOBAL_STATE%PLCRIT_AER, GLOBAL_STATE%PICRIT_AER, GLOBAL_STATE%PRE_ICE, &
     & GLOBAL_STATE%PCCN,     GLOBAL_STATE%PNICE,&
     & GLOBAL_STATE%PCOVPTOT, GLOBAL_STATE%PRAINFRAC_TOPRFZ, &
     & GLOBAL_STATE%PFSQLF,   GLOBAL_STATE%PFSQIF ,  GLOBAL_STATE%PFCQNNG,  GLOBAL_STATE%PFCQLNG, &
     & GLOBAL_STATE%PFSQRF,   GLOBAL_STATE%PFSQSF ,  GLOBAL_STATE%PFCQRNG,  GLOBAL_STATE%PFCQSNG, &
     & GLOBAL_STATE%PFSQLTUR, GLOBAL_STATE%PFSQITUR, &
     & GLOBAL_STATE%PFPLSL,   GLOBAL_STATE%PFPLSN,   GLOBAL_STATE%PFHPSL,   GLOBAL_STATE%PFHPSN &
     & )
!$acc wait
! Zero B_LOC after warmup (kernel wrote tendencies into it)
GLOBAL_STATE%B_LOC(:,:,:,:) = 0.0_JPRB
!$acc update device(GLOBAL_STATE%B_LOC)

DO JSTEP = 1, NSTEPS

  IF (IRANK == 0) THEN
    WRITE(0,'(1X,A,I0,A,I0)') '  Step ', JSTEP, ' / ', NSTEPS
  END IF

  CALL SYSTEM_CLOCK(ICLOCK_STEP)
  ZTIME_KERNEL = 0.0_JPRD
  ZTIME_UPDATE = 0.0_JPRD

  ! Inner sub-substep loop: NSUB kernel calls per outer step
  DO JSUB = 1, NSUB

    ! Call the kernel — all arrays are already present on device
    CALL SYSTEM_CLOCK(ICLOCK_AUX)
    CALL CLOUDSC_DRIVER_GPU_SCC_K_CACHING(NUMOMP, NPROMA, GLOBAL_STATE%KLEV, &
         & NGPTOT, GLOBAL_STATE%NBLOCKS, NGPTOTG, &
         & GLOBAL_STATE%KFLDX, ZTSPHY_INNER, &
         & GLOBAL_STATE%PT, GLOBAL_STATE%PQ, &
         & GLOBAL_STATE%B_CML,   GLOBAL_STATE%B_TMP, GLOBAL_STATE%B_LOC, &
         & GLOBAL_STATE%PVFA,    GLOBAL_STATE%PVFL,  GLOBAL_STATE%PVFI, &
         & GLOBAL_STATE%PDYNA,   GLOBAL_STATE%PDYNL, GLOBAL_STATE%PDYNI, &
         & GLOBAL_STATE%PHRSW,   GLOBAL_STATE%PHRLW, &
         & GLOBAL_STATE%PVERVEL, GLOBAL_STATE%PAP,   GLOBAL_STATE%PAPH, &
         & GLOBAL_STATE%PLSM,    GLOBAL_STATE%LDCUM, GLOBAL_STATE%KTYPE, &
         & GLOBAL_STATE%PLU,     GLOBAL_STATE%PLUDE, GLOBAL_STATE%PSNDE, &
         & GLOBAL_STATE%PMFU,    GLOBAL_STATE%PMFD, &
         & GLOBAL_STATE%PA, &
         & GLOBAL_STATE%PCLV,    GLOBAL_STATE%PSUPSAT,&
         & GLOBAL_STATE%PLCRIT_AER, GLOBAL_STATE%PICRIT_AER, GLOBAL_STATE%PRE_ICE, &
         & GLOBAL_STATE%PCCN,     GLOBAL_STATE%PNICE,&
         & GLOBAL_STATE%PCOVPTOT, GLOBAL_STATE%PRAINFRAC_TOPRFZ, &
         & GLOBAL_STATE%PFSQLF,   GLOBAL_STATE%PFSQIF ,  GLOBAL_STATE%PFCQNNG,  GLOBAL_STATE%PFCQLNG, &
         & GLOBAL_STATE%PFSQRF,   GLOBAL_STATE%PFSQSF ,  GLOBAL_STATE%PFCQRNG,  GLOBAL_STATE%PFCQSNG, &
         & GLOBAL_STATE%PFSQLTUR, GLOBAL_STATE%PFSQITUR, &
         & GLOBAL_STATE%PFPLSL,   GLOBAL_STATE%PFPLSN,   GLOBAL_STATE%PFHPSL,   GLOBAL_STATE%PFHPSN &
         & )

    !$acc wait
    CALL SYSTEM_CLOCK(ICLOCK_SUB)
    ZTIME_KERNEL = ZTIME_KERNEL + REAL(ICLOCK_SUB - ICLOCK_AUX, JPRD) / REAL(ICLOCK_RATE, JPRD)

    ! Apply tendencies: forward Euler state update (CPU-side)
    ! B_LOC layout: index 1=T, 2=A, 3=Q, 4:(3+NCLV)=CLD
    !$acc update host(GLOBAL_STATE%PT, GLOBAL_STATE%PQ, GLOBAL_STATE%PA, &
    !$acc   GLOBAL_STATE%PCLV, GLOBAL_STATE%B_TMP, GLOBAL_STATE%B_LOC)
    DO JB = 1, GLOBAL_STATE%NBLOCKS
      DO JK = 1, GLOBAL_STATE%KLEV
        DO JL = 1, NPROMA
          GLOBAL_STATE%PT(JL, JK, JB) = GLOBAL_STATE%PT(JL, JK, JB) &
            & + ZTSPHY_INNER * (GLOBAL_STATE%B_TMP(JL, JK, 1, JB) &
            &                 + GLOBAL_STATE%B_LOC(JL, JK, 1, JB))
          GLOBAL_STATE%PQ(JL, JK, JB) = GLOBAL_STATE%PQ(JL, JK, JB) &
            & + ZTSPHY_INNER * (GLOBAL_STATE%B_TMP(JL, JK, 3, JB) &
            &                 + GLOBAL_STATE%B_LOC(JL, JK, 3, JB))
          GLOBAL_STATE%PA(JL, JK, JB) = GLOBAL_STATE%PA(JL, JK, JB) &
            & + ZTSPHY_INNER * (GLOBAL_STATE%B_TMP(JL, JK, 2, JB) &
            &                 + GLOBAL_STATE%B_LOC(JL, JK, 2, JB))
          DO JM = 1, NCLV
            GLOBAL_STATE%PCLV(JL, JK, JM, JB) = GLOBAL_STATE%PCLV(JL, JK, JM, JB) &
              & + ZTSPHY_INNER * (GLOBAL_STATE%B_TMP(JL, JK, 3+JM, JB) &
              &                 + GLOBAL_STATE%B_LOC(JL, JK, 3+JM, JB))
          END DO
        END DO
      END DO
    END DO
    !$acc update device(GLOBAL_STATE%PT, GLOBAL_STATE%PQ, &
    !$acc   GLOBAL_STATE%PA, GLOBAL_STATE%PCLV)

    ! Zero B_LOC for next kernel call (skip after very last sub-substep of last outer step)
    IF (JSUB < NSUB .OR. JSTEP < NSTEPS) THEN
      GLOBAL_STATE%B_LOC(:,:,:,:) = 0.0_JPRB
      !$acc update device(GLOBAL_STATE%B_LOC)
    END IF

    CALL SYSTEM_CLOCK(ICLOCK_AUX)
    ZTIME_UPDATE = ZTIME_UPDATE + REAL(ICLOCK_AUX - ICLOCK_SUB, JPRD) / REAL(ICLOCK_RATE, JPRD)

  END DO  ! JSUB

  ! NaN/Inf check — compile with -DNAN_CHECK to enable
#ifdef NAN_CHECK
  IF (IRANK == 0 .AND. JSTEP <= 2) THEN
    !$acc update host(GLOBAL_STATE%B_LOC, GLOBAL_STATE%PT, GLOBAL_STATE%PQ)
    NNAN_T = COUNT(GLOBAL_STATE%B_LOC(:,:,1,:) /= GLOBAL_STATE%B_LOC(:,:,1,:))
    NNAN_A = COUNT(GLOBAL_STATE%B_LOC(:,:,2,:) /= GLOBAL_STATE%B_LOC(:,:,2,:))
    NNAN_Q = COUNT(GLOBAL_STATE%B_LOC(:,:,3,:) /= GLOBAL_STATE%B_LOC(:,:,3,:))
    NNAN_CLD = COUNT(GLOBAL_STATE%B_LOC(:,:,4:,:) /= GLOBAL_STATE%B_LOC(:,:,4:,:))
    IF (NNAN_T + NNAN_Q + NNAN_A + NNAN_CLD > 0) THEN
      WRITE(0,'(1X,A,I0,A)') 'FATAL: Step ', JSTEP, ' B_LOC contains NaN after kernel call'
      WRITE(0,'(5X,A,I0)') 'T:   NaN=', NNAN_T
      WRITE(0,'(5X,A,I0)') 'Q:   NaN=', NNAN_Q
      WRITE(0,'(5X,A,I0)') 'A:   NaN=', NNAN_A
      WRITE(0,'(5X,A,I0)') 'CLD: NaN=', NNAN_CLD
      STOP 1
    END IF
    NNAN_T = COUNT(GLOBAL_STATE%PT(:,:,:) /= GLOBAL_STATE%PT(:,:,:))
    NNAN_Q = COUNT(GLOBAL_STATE%PQ(:,:,:) /= GLOBAL_STATE%PQ(:,:,:))
    IF (NNAN_T + NNAN_Q > 0) THEN
      WRITE(0,'(1X,A,I0,A,I0,A,I0)') 'FATAL: Step ', JSTEP, &
        & ' state pre-update: PT NaN=', NNAN_T, '  PQ NaN=', NNAN_Q
      STOP 1
    END IF
  END IF
#endif

  ! Transfer prognostics to host for HDF5 output
  CALL SYSTEM_CLOCK(ICLOCK_AUX)
  IF (IRANK == 0) THEN
    !$acc update host(GLOBAL_STATE%PT, GLOBAL_STATE%PQ, &
    !$acc             GLOBAL_STATE%PA, GLOBAL_STATE%PCLV)
  END IF
  !$acc wait
  CALL SYSTEM_CLOCK(ICLOCK_END)
  ZTIME_D2H = REAL(ICLOCK_END - ICLOCK_AUX, JPRD) / REAL(ICLOCK_RATE, JPRD)
  ZTIME_STEP = ZTIME_KERNEL + ZTIME_UPDATE + ZTIME_D2H

  IF (IRANK == 0) THEN
    WRITE(0,'(1X,A,I0,A,F10.3,A,A,F8.3,A,F8.3,A,F8.3,A)') &
      & '  Step ', JSTEP, ' wall: ', ZTIME_STEP*1000.0_JPRD, ' ms', &
      & '  (kernel=', ZTIME_KERNEL*1000.0_JPRD, &
      & '  update=', ZTIME_UPDATE*1000.0_JPRD, &
      & '  d2h=', ZTIME_D2H*1000.0_JPRD, ')'
    WRITE(IOTIMING, '(I0,4(A,F12.4))') JSTEP, &
      & ',', ZTIME_STEP*1000.0_JPRD, &
      & ',', ZTIME_KERNEL*1000.0_JPRD, &
      & ',', ZTIME_UPDATE*1000.0_JPRD, &
      & ',', ZTIME_D2H*1000.0_JPRD

    CALL CLOUDSC_OUTPUT_WRITE_STEP(JSTEP, NPROMA, GLOBAL_STATE%KLEV, NCLV, &
      & GLOBAL_STATE%NBLOCKS, GLOBAL_STATE%PT, GLOBAL_STATE%PQ, &
      & GLOBAL_STATE%PA, GLOBAL_STATE%PCLV)
  END IF

END DO

!$acc end data

CALL SYSTEM_CLOCK(ICLOCK_END)
ZTIME_TOTAL = REAL(ICLOCK_END - ICLOCK_START, JPRD) / REAL(ICLOCK_RATE, JPRD)

! --- Close output and finalize ---

IF (IRANK == 0) THEN
  CALL CLOUDSC_OUTPUT_CLOSE()
END IF

IF (IRANK == 0) THEN
  ! Write summary row and close timing CSV
  WRITE(IOTIMING, '(A,A,F12.4,A)') 'total', ',', ZTIME_TOTAL*1000.0_JPRD, ',,,'
  CLOSE(IOTIMING)

  WRITE(0,'(1X,A)')           '============================================'
  WRITE(0,'(1X,A,I0,A)')      '  Completed ', NSTEPS, ' timesteps.'
  WRITE(0,'(1X,A,F10.3,A)')   '  Total wall time:    ', ZTIME_TOTAL*1000.0_JPRD, ' ms'
  WRITE(0,'(1X,A,F10.3,A)')   '  Mean step time:     ', ZTIME_TOTAL*1000.0_JPRD/REAL(NSTEPS,JPRD), ' ms'
  WRITE(0,'(1X,A,ES12.5,A)')  '  Throughput:         ', REAL(NGPTOTG,JPRD)*REAL(NSTEPS,JPRD)/ZTIME_TOTAL, ' col·steps/s'
  WRITE(0,'(1X,A)')           '============================================'
END IF

CALL CLOUDSC_MPI_END()

END PROGRAM DWARF_CLOUDSC_MULTISTEP
