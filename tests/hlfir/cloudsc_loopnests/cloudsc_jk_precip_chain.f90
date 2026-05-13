SUBROUTINE jk_precip_chain(kidia, kfdia, klon, klev, nclv, ncldqr, ncldqs, ncldtop, &
                           pap, paph, ztp1, za, zqxn, zfallsink_in, &
                           zpfplsx, zqpretot, zcovptot, &
                           rd, rg, ptsphy, zepsec)
  ! Multi-JK reproducer extracted from cloudsc.F90 -- the suspect precip
  ! chain that produces the 1-ulp ZQPRETOT drift at JK=NCLDTOP+1 and
  ! cascades into the ZCOVPTOT threshold flip at JK=21.
  !
  ! Structure (mirrors cloudsc.F90 lines 1766..3700 stripped to the
  ! precip-only thread, no source/sink accumulation, no LU solver):
  !
  !   1. ZRHO setup (line 1832): ZRHO = PAP / (RD * ZTP1)
  !   2. ZDTGDP setup (~1830): ZDTGDP = PTSPHY * (PAPH(JK+1) - PAPH(JK)) / RG
  !   3. ZRDTGDP = 1 / ZDTGDP
  !   4. Line 3608: ZPFPLSX(JL,JK+1,JM) = ZFALLSINK(JL,JM) * ZQXN(JL,JM) * ZRDTGDP(JL)
  !   5. Line 3614: ZQPRETOT(JL) = ZPFPLSX(JL,JK+1,NCLDQS) + ZPFPLSX(JL,JK+1,NCLDQR)
  !   6. Line 3617-3619: clear ZCOVPTOT if ZQPRETOT < ZEPSEC
  !   7. (At JK > NCLDTOP) line 2675-2692: ZCOVPTOT max-overlap update from ZA
  !
  ! ZFALLSINK and ZQXN are *inputs* (per-JK from caller) -- this skips
  ! the LU solver entirely so we isolate the JK-loop-carried portion
  ! of the divergence chain from the LU solver portion.
  !
  ! Output: ZCOVPTOT at end of vertical column.  If the bridge reproduces
  ! the cloudsc_full ZCOVPTOT divergence pattern with this stripped-down
  ! kernel, the bug is in the JK-loop carry of ZCOVPTOT / ZPFPLSX itself.
  ! If it's bit-exact, the bug is in the upstream ZFALLSINK / ZQXN
  ! computation (which the LU solver feeds into) -- a different chain.

  IMPLICIT NONE
  INTEGER(KIND = 4), VALUE :: kidia, kfdia, klon, klev, nclv, ncldqr, ncldqs, ncldtop
  REAL(KIND = 8), INTENT(IN)  :: pap(klon, klev)              ! pressure full levels
  REAL(KIND = 8), INTENT(IN)  :: paph(klon, klev + 1)         ! pressure half levels
  REAL(KIND = 8), INTENT(IN)  :: ztp1(klon, klev)             ! temperature
  REAL(KIND = 8), INTENT(IN)  :: za(klon, klev)               ! cloud fraction
  REAL(KIND = 8), INTENT(IN)  :: zqxn(klon, klev, nclv)       ! new q values (would-be LU output)
  REAL(KIND = 8), INTENT(IN)  :: zfallsink_in(klon, klev, nclv) ! sedimentation sink
  REAL(KIND = 8), INTENT(OUT) :: zpfplsx(klon, klev + 1, nclv) ! precip flux
  REAL(KIND = 8), INTENT(OUT) :: zqpretot(klon, klev)          ! total precip per JK
  REAL(KIND = 8), INTENT(OUT) :: zcovptot(klon, klev)          ! cover per JK (running)
  REAL(KIND = 8), VALUE       :: rd, rg, ptsphy, zepsec

  INTEGER(KIND = 4) :: jl, jk, jm
  REAL(KIND = 8) :: zrho(klon)
  REAL(KIND = 8) :: zdtgdp(klon)
  REAL(KIND = 8) :: zrdtgdp(klon)
  REAL(KIND = 8) :: zcovptot_cur(klon)

  zpfplsx(:, :, :) = 0.0_8
  zqpretot(:, :)   = 0.0_8
  zcovptot(:, :)   = 0.0_8
  zcovptot_cur(:)  = 0.0_8

  DO jk = ncldtop, klev
    DO jl = kidia, kfdia
      zrho(jl)    = pap(jl, jk) / (rd * ztp1(jl, jk))
      zdtgdp(jl)  = ptsphy * (paph(jl, jk + 1) - paph(jl, jk)) / rg
      zrdtgdp(jl) = 1.0_8 / zdtgdp(jl)
    END DO

    ! Line-3608 3-way multiply per species; only fires for JK >= NCLDTOP.
    DO jm = 1, nclv
      DO jl = kidia, kfdia
        zpfplsx(jl, jk + 1, jm) = zfallsink_in(jl, jk, jm) * zqxn(jl, jk, jm) * zrdtgdp(jl)
      END DO
    END DO

    ! Line-3614 ZQPRETOT total (snow + rain).
    DO jl = kidia, kfdia
      zqpretot(jl, jk) = zpfplsx(jl, jk + 1, ncldqs) + zpfplsx(jl, jk + 1, ncldqr)
    END DO

    ! Line-3617 reset cover if no precip.
    DO jl = kidia, kfdia
      IF (zqpretot(jl, jk) < zepsec) THEN
        zcovptot_cur(jl) = 0.0_8
      END IF
    END DO

    ! Lines 2675-2692: max-overlap ZCOVPTOT update (only for JK > NCLDTOP
    ! so the JK-1 read is valid).
    IF (jk > ncldtop) THEN
      DO jl = kidia, kfdia
        IF (zqpretot(jl, jk) > zepsec) THEN
          zcovptot_cur(jl) = 1.0_8 - ((1.0_8 - zcovptot_cur(jl)) * &
                              (1.0_8 - MAX(za(jl, jk), za(jl, jk - 1))) / &
                              (1.0_8 - MIN(za(jl, jk - 1), 1.0_8 - 1.0E-06_8)))
        END IF
      END DO
    END IF

    DO jl = kidia, kfdia
      zcovptot(jl, jk) = zcovptot_cur(jl)
    END DO
  END DO
END SUBROUTINE jk_precip_chain
