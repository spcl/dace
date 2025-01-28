MODULE serde
  IMPLICIT NONE
  INTERFACE serialize
    MODULE PROCEDURE aerosol_type_2s, cloud_type_2s, cloud_optics_type_2s, gas_type_2s, pdf_sampler_type_2s, config_type_2s, flux_type_2s, thermodynamics_type_2s, randomnumberstream_2s, single_level_type_2s, real__8_2s3, real__8_2s2, integer_2s1, real__8_2s1, integer__4_2s1, logical_2s, integer1_2s, integer2_2s, integer4_2s, integer8_2s, real4_2s, real8_2s
  END INTERFACE serialize
  CONTAINS
  SUBROUTINE write_to(path, s)
    CHARACTER(LEN = *), INTENT(IN) :: path
    CHARACTER(LEN = *), INTENT(IN) :: s
    INTEGER :: io
    OPEN(NEWUNIT = io, FILE = path, STATUS = "replace", ACTION = "write")
    WRITE(io, *) s
    CLOSE(UNIT = io)
  END SUBROUTINE write_to
  FUNCTION aerosol_type_2s(x) RESULT(s)
    USE radiation_aerosol, ONLY: aerosol_type
    TYPE(aerosol_type), TARGET, INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: kmeta, kmeta_0, kmeta_1, kmeta_2, kmeta_3, kmeta_4, kmeta_5, kmeta_6, kmeta_7, kmeta_8, kmeta_9
    s = s // '# od_sw' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % od_sw)) // NEW_LINE('A')
    IF (ALLOCATED(x % od_sw)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(3) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(SIZE(x % od_sw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(LBOUND(x % od_sw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % od_sw) // NEW_LINE('A')
    END IF
    s = s // '# ssa_sw' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % ssa_sw)) // NEW_LINE('A')
    IF (ALLOCATED(x % ssa_sw)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(3) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(SIZE(x % ssa_sw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(LBOUND(x % ssa_sw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % ssa_sw) // NEW_LINE('A')
    END IF
    s = s // '# g_sw' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % g_sw)) // NEW_LINE('A')
    IF (ALLOCATED(x % g_sw)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(3) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(SIZE(x % g_sw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(LBOUND(x % g_sw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % g_sw) // NEW_LINE('A')
    END IF
    s = s // '# od_lw' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % od_lw)) // NEW_LINE('A')
    IF (ALLOCATED(x % od_lw)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(3) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(SIZE(x % od_lw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(LBOUND(x % od_lw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % od_lw) // NEW_LINE('A')
    END IF
    s = s // '# ssa_lw' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % ssa_lw)) // NEW_LINE('A')
    IF (ALLOCATED(x % ssa_lw)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(3) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(SIZE(x % ssa_lw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(LBOUND(x % ssa_lw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % ssa_lw) // NEW_LINE('A')
    END IF
  END FUNCTION aerosol_type_2s
  FUNCTION cloud_type_2s(x) RESULT(s)
    USE radiation_cloud, ONLY: cloud_type
    TYPE(cloud_type), TARGET, INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: kmeta, kmeta_0, kmeta_1, kmeta_2, kmeta_3, kmeta_4, kmeta_5, kmeta_6, kmeta_7, kmeta_8, kmeta_9
    s = s // '# mixing_ratio' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % mixing_ratio)) // NEW_LINE('A')
    IF (ALLOCATED(x % mixing_ratio)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(3) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(SIZE(x % mixing_ratio, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(LBOUND(x % mixing_ratio, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % mixing_ratio) // NEW_LINE('A')
    END IF
    s = s // '# q_liq' // NEW_LINE('A')
    s = s // '# assoc' // NEW_LINE('A') // serialize(ASSOCIATED(x % q_liq)) // NEW_LINE('A')
    IF (ASSOCIATED(x % q_liq)) THEN
      kmeta = 0
      DO kmeta_0 = LBOUND(x % mixing_ratio, 1), UBOUND(x % mixing_ratio, 1)
        IF (ASSOCIATED(x % q_liq, x % mixing_ratio(kmeta_0, :, :))) THEN
          kmeta = 1
          s = s // "=> x%mixing_ratio("
          s = s // serialize(kmeta_0)
          s = s // ':'
          s = s // ':'
          s = s // "))" // NEW_LINE('A')
        END IF
      END DO
      DO kmeta_1 = LBOUND(x % mixing_ratio, 2), UBOUND(x % mixing_ratio, 2)
        IF (ASSOCIATED(x % q_liq, x % mixing_ratio(:, kmeta_1, :))) THEN
          kmeta = 1
          s = s // "=> x%mixing_ratio("
          s = s // ':'
          s = s // serialize(kmeta_1)
          s = s // ':'
          s = s // "))" // NEW_LINE('A')
        END IF
      END DO
      DO kmeta_2 = LBOUND(x % mixing_ratio, 3), UBOUND(x % mixing_ratio, 3)
        IF (ASSOCIATED(x % q_liq, x % mixing_ratio(:, :, kmeta_2))) THEN
          kmeta = 1
          s = s // "=> x%mixing_ratio("
          s = s // ':'
          s = s // ':'
          s = s // serialize(kmeta_2)
          s = s // "))" // NEW_LINE('A')
        END IF
      END DO
      IF (ASSOCIATED(x % q_liq, x % overlap_param(:, :))) THEN
        kmeta = 1
        s = s // "=> x%overlap_param("
        s = s // ':'
        s = s // ':'
        s = s // "))" // NEW_LINE('A')
      END IF
      IF (kmeta == 0) THEN
        s = s // "=> missing" // NEW_LINE('A')
      END IF
    END IF
    s = s // '# q_ice' // NEW_LINE('A')
    s = s // '# assoc' // NEW_LINE('A') // serialize(ASSOCIATED(x % q_ice)) // NEW_LINE('A')
    IF (ASSOCIATED(x % q_ice)) THEN
      kmeta = 0
      DO kmeta_0 = LBOUND(x % mixing_ratio, 1), UBOUND(x % mixing_ratio, 1)
        IF (ASSOCIATED(x % q_ice, x % mixing_ratio(kmeta_0, :, :))) THEN
          kmeta = 1
          s = s // "=> x%mixing_ratio("
          s = s // serialize(kmeta_0)
          s = s // ':'
          s = s // ':'
          s = s // "))" // NEW_LINE('A')
        END IF
      END DO
      DO kmeta_1 = LBOUND(x % mixing_ratio, 2), UBOUND(x % mixing_ratio, 2)
        IF (ASSOCIATED(x % q_ice, x % mixing_ratio(:, kmeta_1, :))) THEN
          kmeta = 1
          s = s // "=> x%mixing_ratio("
          s = s // ':'
          s = s // serialize(kmeta_1)
          s = s // ':'
          s = s // "))" // NEW_LINE('A')
        END IF
      END DO
      DO kmeta_2 = LBOUND(x % mixing_ratio, 3), UBOUND(x % mixing_ratio, 3)
        IF (ASSOCIATED(x % q_ice, x % mixing_ratio(:, :, kmeta_2))) THEN
          kmeta = 1
          s = s // "=> x%mixing_ratio("
          s = s // ':'
          s = s // ':'
          s = s // serialize(kmeta_2)
          s = s // "))" // NEW_LINE('A')
        END IF
      END DO
      IF (ASSOCIATED(x % q_ice, x % overlap_param(:, :))) THEN
        kmeta = 1
        s = s // "=> x%overlap_param("
        s = s // ':'
        s = s // ':'
        s = s // "))" // NEW_LINE('A')
      END IF
      IF (kmeta == 0) THEN
        s = s // "=> missing" // NEW_LINE('A')
      END IF
    END IF
    s = s // '# re_liq' // NEW_LINE('A')
    s = s // '# assoc' // NEW_LINE('A') // serialize(ASSOCIATED(x % re_liq)) // NEW_LINE('A')
    IF (ASSOCIATED(x % re_liq)) THEN
      kmeta = 0
      DO kmeta_0 = LBOUND(x % mixing_ratio, 1), UBOUND(x % mixing_ratio, 1)
        IF (ASSOCIATED(x % re_liq, x % mixing_ratio(kmeta_0, :, :))) THEN
          kmeta = 1
          s = s // "=> x%mixing_ratio("
          s = s // serialize(kmeta_0)
          s = s // ':'
          s = s // ':'
          s = s // "))" // NEW_LINE('A')
        END IF
      END DO
      DO kmeta_1 = LBOUND(x % mixing_ratio, 2), UBOUND(x % mixing_ratio, 2)
        IF (ASSOCIATED(x % re_liq, x % mixing_ratio(:, kmeta_1, :))) THEN
          kmeta = 1
          s = s // "=> x%mixing_ratio("
          s = s // ':'
          s = s // serialize(kmeta_1)
          s = s // ':'
          s = s // "))" // NEW_LINE('A')
        END IF
      END DO
      DO kmeta_2 = LBOUND(x % mixing_ratio, 3), UBOUND(x % mixing_ratio, 3)
        IF (ASSOCIATED(x % re_liq, x % mixing_ratio(:, :, kmeta_2))) THEN
          kmeta = 1
          s = s // "=> x%mixing_ratio("
          s = s // ':'
          s = s // ':'
          s = s // serialize(kmeta_2)
          s = s // "))" // NEW_LINE('A')
        END IF
      END DO
      IF (ASSOCIATED(x % re_liq, x % overlap_param(:, :))) THEN
        kmeta = 1
        s = s // "=> x%overlap_param("
        s = s // ':'
        s = s // ':'
        s = s // "))" // NEW_LINE('A')
      END IF
      IF (kmeta == 0) THEN
        s = s // "=> missing" // NEW_LINE('A')
      END IF
    END IF
    s = s // '# re_ice' // NEW_LINE('A')
    s = s // '# assoc' // NEW_LINE('A') // serialize(ASSOCIATED(x % re_ice)) // NEW_LINE('A')
    IF (ASSOCIATED(x % re_ice)) THEN
      kmeta = 0
      DO kmeta_0 = LBOUND(x % mixing_ratio, 1), UBOUND(x % mixing_ratio, 1)
        IF (ASSOCIATED(x % re_ice, x % mixing_ratio(kmeta_0, :, :))) THEN
          kmeta = 1
          s = s // "=> x%mixing_ratio("
          s = s // serialize(kmeta_0)
          s = s // ':'
          s = s // ':'
          s = s // "))" // NEW_LINE('A')
        END IF
      END DO
      DO kmeta_1 = LBOUND(x % mixing_ratio, 2), UBOUND(x % mixing_ratio, 2)
        IF (ASSOCIATED(x % re_ice, x % mixing_ratio(:, kmeta_1, :))) THEN
          kmeta = 1
          s = s // "=> x%mixing_ratio("
          s = s // ':'
          s = s // serialize(kmeta_1)
          s = s // ':'
          s = s // "))" // NEW_LINE('A')
        END IF
      END DO
      DO kmeta_2 = LBOUND(x % mixing_ratio, 3), UBOUND(x % mixing_ratio, 3)
        IF (ASSOCIATED(x % re_ice, x % mixing_ratio(:, :, kmeta_2))) THEN
          kmeta = 1
          s = s // "=> x%mixing_ratio("
          s = s // ':'
          s = s // ':'
          s = s // serialize(kmeta_2)
          s = s // "))" // NEW_LINE('A')
        END IF
      END DO
      IF (ASSOCIATED(x % re_ice, x % overlap_param(:, :))) THEN
        kmeta = 1
        s = s // "=> x%overlap_param("
        s = s // ':'
        s = s // ':'
        s = s // "))" // NEW_LINE('A')
      END IF
      IF (kmeta == 0) THEN
        s = s // "=> missing" // NEW_LINE('A')
      END IF
    END IF
    s = s // '# fraction' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % fraction)) // NEW_LINE('A')
    IF (ALLOCATED(x % fraction)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % fraction, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % fraction, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % fraction) // NEW_LINE('A')
    END IF
    s = s // '# fractional_std' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % fractional_std)) // NEW_LINE('A')
    IF (ALLOCATED(x % fractional_std)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % fractional_std, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % fractional_std, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % fractional_std) // NEW_LINE('A')
    END IF
    s = s // '# overlap_param' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % overlap_param)) // NEW_LINE('A')
    IF (ALLOCATED(x % overlap_param)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % overlap_param, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % overlap_param, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % overlap_param) // NEW_LINE('A')
    END IF
  END FUNCTION cloud_type_2s
  FUNCTION cloud_optics_type_2s(x) RESULT(s)
    USE radiation_cloud_optics_data, ONLY: cloud_optics_type
    TYPE(cloud_optics_type), TARGET, INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: kmeta, kmeta_0, kmeta_1, kmeta_2, kmeta_3, kmeta_4, kmeta_5, kmeta_6, kmeta_7, kmeta_8, kmeta_9
    s = s // '# liq_coeff_lw' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % liq_coeff_lw)) // NEW_LINE('A')
    IF (ALLOCATED(x % liq_coeff_lw)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % liq_coeff_lw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % liq_coeff_lw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % liq_coeff_lw) // NEW_LINE('A')
    END IF
    s = s // '# liq_coeff_sw' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % liq_coeff_sw)) // NEW_LINE('A')
    IF (ALLOCATED(x % liq_coeff_sw)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % liq_coeff_sw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % liq_coeff_sw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % liq_coeff_sw) // NEW_LINE('A')
    END IF
    s = s // '# ice_coeff_lw' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % ice_coeff_lw)) // NEW_LINE('A')
    IF (ALLOCATED(x % ice_coeff_lw)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % ice_coeff_lw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % ice_coeff_lw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % ice_coeff_lw) // NEW_LINE('A')
    END IF
    s = s // '# ice_coeff_sw' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % ice_coeff_sw)) // NEW_LINE('A')
    IF (ALLOCATED(x % ice_coeff_sw)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % ice_coeff_sw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % ice_coeff_sw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % ice_coeff_sw) // NEW_LINE('A')
    END IF
  END FUNCTION cloud_optics_type_2s
  FUNCTION gas_type_2s(x) RESULT(s)
    USE radiation_gas, ONLY: gas_type
    TYPE(gas_type), TARGET, INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: kmeta, kmeta_0, kmeta_1, kmeta_2, kmeta_3, kmeta_4, kmeta_5, kmeta_6, kmeta_7, kmeta_8, kmeta_9
    s = s // '# mixing_ratio' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % mixing_ratio)) // NEW_LINE('A')
    IF (ALLOCATED(x % mixing_ratio)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(3) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(SIZE(x % mixing_ratio, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(LBOUND(x % mixing_ratio, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % mixing_ratio) // NEW_LINE('A')
    END IF
  END FUNCTION gas_type_2s
  FUNCTION pdf_sampler_type_2s(x) RESULT(s)
    USE radiation_pdf_sampler, ONLY: pdf_sampler_type
    TYPE(pdf_sampler_type), TARGET, INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: kmeta, kmeta_0, kmeta_1, kmeta_2, kmeta_3, kmeta_4, kmeta_5, kmeta_6, kmeta_7, kmeta_8, kmeta_9
    s = s // '# ncdf' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % ncdf) // NEW_LINE('A')
    s = s // '# nfsd' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % nfsd) // NEW_LINE('A')
    s = s // '# fsd1' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % fsd1) // NEW_LINE('A')
    s = s // '# inv_fsd_interval' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % inv_fsd_interval) // NEW_LINE('A')
    s = s // '# val' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % val)) // NEW_LINE('A')
    IF (ALLOCATED(x % val)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % val, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % val, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % val) // NEW_LINE('A')
    END IF
  END FUNCTION pdf_sampler_type_2s
  FUNCTION config_type_2s(x) RESULT(s)
    USE radiation_config, ONLY: config_type
    TYPE(config_type), TARGET, INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: kmeta, kmeta_0, kmeta_1, kmeta_2, kmeta_3, kmeta_4, kmeta_5, kmeta_6, kmeta_7, kmeta_8, kmeta_9
    s = s // '# i_emiss_from_band_lw' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % i_emiss_from_band_lw)) // NEW_LINE('A')
    IF (ALLOCATED(x % i_emiss_from_band_lw)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(SIZE(x % i_emiss_from_band_lw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(LBOUND(x % i_emiss_from_band_lw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % i_emiss_from_band_lw) // NEW_LINE('A')
    END IF
    s = s // '# sw_albedo_weights' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % sw_albedo_weights)) // NEW_LINE('A')
    IF (ALLOCATED(x % sw_albedo_weights)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % sw_albedo_weights, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % sw_albedo_weights, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % sw_albedo_weights) // NEW_LINE('A')
    END IF
    s = s // '# i_band_from_g_lw' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % i_band_from_g_lw)) // NEW_LINE('A')
    IF (ALLOCATED(x % i_band_from_g_lw)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(SIZE(x % i_band_from_g_lw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(LBOUND(x % i_band_from_g_lw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % i_band_from_g_lw) // NEW_LINE('A')
    END IF
    s = s // '# i_band_from_reordered_g_lw' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % i_band_from_reordered_g_lw)) // NEW_LINE('A')
    IF (ALLOCATED(x % i_band_from_reordered_g_lw)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(SIZE(x % i_band_from_reordered_g_lw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(LBOUND(x % i_band_from_reordered_g_lw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % i_band_from_reordered_g_lw) // NEW_LINE('A')
    END IF
    s = s // '# i_band_from_reordered_g_sw' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % i_band_from_reordered_g_sw)) // NEW_LINE('A')
    IF (ALLOCATED(x % i_band_from_reordered_g_sw)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(SIZE(x % i_band_from_reordered_g_sw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(LBOUND(x % i_band_from_reordered_g_sw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % i_band_from_reordered_g_sw) // NEW_LINE('A')
    END IF
    s = s // '# cloud_optics' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % cloud_optics) // NEW_LINE('A')
    s = s // '# pdf_sampler' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % pdf_sampler) // NEW_LINE('A')
  END FUNCTION config_type_2s
  FUNCTION flux_type_2s(x) RESULT(s)
    USE radiation_flux, ONLY: flux_type
    TYPE(flux_type), TARGET, INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: kmeta, kmeta_0, kmeta_1, kmeta_2, kmeta_3, kmeta_4, kmeta_5, kmeta_6, kmeta_7, kmeta_8, kmeta_9
    s = s // '# lw_up' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % lw_up)) // NEW_LINE('A')
    IF (ALLOCATED(x % lw_up)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % lw_up, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % lw_up, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % lw_up) // NEW_LINE('A')
    END IF
    s = s // '# lw_dn' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % lw_dn)) // NEW_LINE('A')
    IF (ALLOCATED(x % lw_dn)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % lw_dn, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % lw_dn, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % lw_dn) // NEW_LINE('A')
    END IF
    s = s // '# sw_up' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % sw_up)) // NEW_LINE('A')
    IF (ALLOCATED(x % sw_up)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % sw_up, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % sw_up, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % sw_up) // NEW_LINE('A')
    END IF
    s = s // '# sw_dn' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % sw_dn)) // NEW_LINE('A')
    IF (ALLOCATED(x % sw_dn)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % sw_dn, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % sw_dn, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % sw_dn) // NEW_LINE('A')
    END IF
    s = s // '# sw_dn_direct' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % sw_dn_direct)) // NEW_LINE('A')
    IF (ALLOCATED(x % sw_dn_direct)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % sw_dn_direct, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % sw_dn_direct, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % sw_dn_direct) // NEW_LINE('A')
    END IF
    s = s // '# lw_up_clear' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % lw_up_clear)) // NEW_LINE('A')
    IF (ALLOCATED(x % lw_up_clear)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % lw_up_clear, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % lw_up_clear, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % lw_up_clear) // NEW_LINE('A')
    END IF
    s = s // '# lw_dn_clear' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % lw_dn_clear)) // NEW_LINE('A')
    IF (ALLOCATED(x % lw_dn_clear)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % lw_dn_clear, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % lw_dn_clear, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % lw_dn_clear) // NEW_LINE('A')
    END IF
    s = s // '# sw_up_clear' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % sw_up_clear)) // NEW_LINE('A')
    IF (ALLOCATED(x % sw_up_clear)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % sw_up_clear, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % sw_up_clear, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % sw_up_clear) // NEW_LINE('A')
    END IF
    s = s // '# sw_dn_clear' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % sw_dn_clear)) // NEW_LINE('A')
    IF (ALLOCATED(x % sw_dn_clear)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % sw_dn_clear, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % sw_dn_clear, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % sw_dn_clear) // NEW_LINE('A')
    END IF
    s = s // '# sw_dn_direct_clear' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % sw_dn_direct_clear)) // NEW_LINE('A')
    IF (ALLOCATED(x % sw_dn_direct_clear)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % sw_dn_direct_clear, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % sw_dn_direct_clear, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % sw_dn_direct_clear) // NEW_LINE('A')
    END IF
    s = s // '# lw_dn_surf_g' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % lw_dn_surf_g)) // NEW_LINE('A')
    IF (ALLOCATED(x % lw_dn_surf_g)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % lw_dn_surf_g, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % lw_dn_surf_g, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % lw_dn_surf_g) // NEW_LINE('A')
    END IF
    s = s // '# lw_dn_surf_clear_g' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % lw_dn_surf_clear_g)) // NEW_LINE('A')
    IF (ALLOCATED(x % lw_dn_surf_clear_g)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % lw_dn_surf_clear_g, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % lw_dn_surf_clear_g, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % lw_dn_surf_clear_g) // NEW_LINE('A')
    END IF
    s = s // '# sw_dn_diffuse_surf_g' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % sw_dn_diffuse_surf_g)) // NEW_LINE('A')
    IF (ALLOCATED(x % sw_dn_diffuse_surf_g)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % sw_dn_diffuse_surf_g, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % sw_dn_diffuse_surf_g, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % sw_dn_diffuse_surf_g) // NEW_LINE('A')
    END IF
    s = s // '# sw_dn_direct_surf_g' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % sw_dn_direct_surf_g)) // NEW_LINE('A')
    IF (ALLOCATED(x % sw_dn_direct_surf_g)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % sw_dn_direct_surf_g, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % sw_dn_direct_surf_g, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % sw_dn_direct_surf_g) // NEW_LINE('A')
    END IF
    s = s // '# sw_dn_diffuse_surf_clear_g' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % sw_dn_diffuse_surf_clear_g)) // NEW_LINE('A')
    IF (ALLOCATED(x % sw_dn_diffuse_surf_clear_g)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % sw_dn_diffuse_surf_clear_g, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % sw_dn_diffuse_surf_clear_g, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % sw_dn_diffuse_surf_clear_g) // NEW_LINE('A')
    END IF
    s = s // '# sw_dn_direct_surf_clear_g' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % sw_dn_direct_surf_clear_g)) // NEW_LINE('A')
    IF (ALLOCATED(x % sw_dn_direct_surf_clear_g)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % sw_dn_direct_surf_clear_g, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % sw_dn_direct_surf_clear_g, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % sw_dn_direct_surf_clear_g) // NEW_LINE('A')
    END IF
    s = s // '# sw_dn_surf_band' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % sw_dn_surf_band)) // NEW_LINE('A')
    IF (ALLOCATED(x % sw_dn_surf_band)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % sw_dn_surf_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % sw_dn_surf_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % sw_dn_surf_band) // NEW_LINE('A')
    END IF
    s = s // '# sw_dn_direct_surf_band' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % sw_dn_direct_surf_band)) // NEW_LINE('A')
    IF (ALLOCATED(x % sw_dn_direct_surf_band)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % sw_dn_direct_surf_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % sw_dn_direct_surf_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % sw_dn_direct_surf_band) // NEW_LINE('A')
    END IF
    s = s // '# sw_dn_surf_clear_band' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % sw_dn_surf_clear_band)) // NEW_LINE('A')
    IF (ALLOCATED(x % sw_dn_surf_clear_band)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % sw_dn_surf_clear_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % sw_dn_surf_clear_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % sw_dn_surf_clear_band) // NEW_LINE('A')
    END IF
    s = s // '# sw_dn_direct_surf_clear_band' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % sw_dn_direct_surf_clear_band)) // NEW_LINE('A')
    IF (ALLOCATED(x % sw_dn_direct_surf_clear_band)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % sw_dn_direct_surf_clear_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % sw_dn_direct_surf_clear_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % sw_dn_direct_surf_clear_band) // NEW_LINE('A')
    END IF
    s = s // '# cloud_cover_lw' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % cloud_cover_lw)) // NEW_LINE('A')
    IF (ALLOCATED(x % cloud_cover_lw)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(SIZE(x % cloud_cover_lw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(LBOUND(x % cloud_cover_lw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % cloud_cover_lw) // NEW_LINE('A')
    END IF
    s = s // '# cloud_cover_sw' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % cloud_cover_sw)) // NEW_LINE('A')
    IF (ALLOCATED(x % cloud_cover_sw)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(SIZE(x % cloud_cover_sw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(LBOUND(x % cloud_cover_sw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % cloud_cover_sw) // NEW_LINE('A')
    END IF
  END FUNCTION flux_type_2s
  FUNCTION thermodynamics_type_2s(x) RESULT(s)
    USE radiation_thermodynamics, ONLY: thermodynamics_type
    TYPE(thermodynamics_type), TARGET, INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: kmeta, kmeta_0, kmeta_1, kmeta_2, kmeta_3, kmeta_4, kmeta_5, kmeta_6, kmeta_7, kmeta_8, kmeta_9
    s = s // '# pressure_hl' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % pressure_hl)) // NEW_LINE('A')
    IF (ALLOCATED(x % pressure_hl)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % pressure_hl, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % pressure_hl, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % pressure_hl) // NEW_LINE('A')
    END IF
    s = s // '# temperature_hl' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % temperature_hl)) // NEW_LINE('A')
    IF (ALLOCATED(x % temperature_hl)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % temperature_hl, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % temperature_hl, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % temperature_hl) // NEW_LINE('A')
    END IF
    s = s // '# pressure_fl' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % pressure_fl)) // NEW_LINE('A')
    IF (ALLOCATED(x % pressure_fl)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % pressure_fl, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % pressure_fl, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % pressure_fl) // NEW_LINE('A')
    END IF
    s = s // '# temperature_fl' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % temperature_fl)) // NEW_LINE('A')
    IF (ALLOCATED(x % temperature_fl)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % temperature_fl, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % temperature_fl, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % temperature_fl) // NEW_LINE('A')
    END IF
  END FUNCTION thermodynamics_type_2s
  FUNCTION randomnumberstream_2s(x) RESULT(s)
    USE random_numbers_mix, ONLY: randomnumberstream
    TYPE(randomnumberstream), TARGET, INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: kmeta, kmeta_0, kmeta_1, kmeta_2, kmeta_3, kmeta_4, kmeta_5, kmeta_6, kmeta_7, kmeta_8, kmeta_9
    s = s // '# iused' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % iused) // NEW_LINE('A')
    s = s // '# inittest' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % inittest) // NEW_LINE('A')
    s = s // '# ix' // NEW_LINE('A')
    s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
    s = s // "# size" // NEW_LINE('A')
    DO kmeta = 1, 1
      s = s // serialize(SIZE(x % ix, kmeta)) // NEW_LINE('A')
    END DO
    s = s // "# lbound" // NEW_LINE('A')
    DO kmeta = 1, 1
      s = s // serialize(LBOUND(x % ix, kmeta)) // NEW_LINE('A')
    END DO
    s = s // NEW_LINE('A') // serialize(x % ix) // NEW_LINE('A')
    s = s // '# zrm' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % zrm) // NEW_LINE('A')
  END FUNCTION randomnumberstream_2s
  FUNCTION single_level_type_2s(x) RESULT(s)
    USE radiation_single_level, ONLY: single_level_type
    TYPE(single_level_type), TARGET, INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: kmeta, kmeta_0, kmeta_1, kmeta_2, kmeta_3, kmeta_4, kmeta_5, kmeta_6, kmeta_7, kmeta_8, kmeta_9
    s = s // '# cos_sza' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % cos_sza)) // NEW_LINE('A')
    IF (ALLOCATED(x % cos_sza)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(SIZE(x % cos_sza, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(LBOUND(x % cos_sza, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % cos_sza) // NEW_LINE('A')
    END IF
    s = s // '# skin_temperature' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % skin_temperature)) // NEW_LINE('A')
    IF (ALLOCATED(x % skin_temperature)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(SIZE(x % skin_temperature, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(LBOUND(x % skin_temperature, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % skin_temperature) // NEW_LINE('A')
    END IF
    s = s // '# sw_albedo' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % sw_albedo)) // NEW_LINE('A')
    IF (ALLOCATED(x % sw_albedo)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % sw_albedo, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % sw_albedo, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % sw_albedo) // NEW_LINE('A')
    END IF
    s = s // '# sw_albedo_direct' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % sw_albedo_direct)) // NEW_LINE('A')
    IF (ALLOCATED(x % sw_albedo_direct)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % sw_albedo_direct, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % sw_albedo_direct, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % sw_albedo_direct) // NEW_LINE('A')
    END IF
    s = s // '# lw_emissivity' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % lw_emissivity)) // NEW_LINE('A')
    IF (ALLOCATED(x % lw_emissivity)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % lw_emissivity, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % lw_emissivity, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % lw_emissivity) // NEW_LINE('A')
    END IF
    s = s // '# spectral_solar_scaling' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % spectral_solar_scaling)) // NEW_LINE('A')
    IF (ALLOCATED(x % spectral_solar_scaling)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(SIZE(x % spectral_solar_scaling, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(LBOUND(x % spectral_solar_scaling, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % spectral_solar_scaling) // NEW_LINE('A')
    END IF
    s = s // '# iseed' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % iseed)) // NEW_LINE('A')
    IF (ALLOCATED(x % iseed)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(SIZE(x % iseed, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(LBOUND(x % iseed, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % iseed) // NEW_LINE('A')
    END IF
  END FUNCTION single_level_type_2s
  FUNCTION real__8_2s3(a) RESULT(s)
    REAL(KIND = 8), INTENT(IN) :: a(:, :, :)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k, k1, k2, k3
    s = "# entries" // NEW_LINE('A')
    DO k1 = LBOUND(a, 1), UBOUND(a, 1)
      DO k2 = LBOUND(a, 2), UBOUND(a, 2)
        DO k3 = LBOUND(a, 3), UBOUND(a, 3)
          s = s // serialize(a(k1, k2, k3)) // NEW_LINE('A')
        END DO
      END DO
    END DO
  END FUNCTION real__8_2s3
  FUNCTION real__8_2s2(a) RESULT(s)
    REAL(KIND = 8), INTENT(IN) :: a(:, :)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k, k1, k2
    s = "# entries" // NEW_LINE('A')
    DO k1 = LBOUND(a, 1), UBOUND(a, 1)
      DO k2 = LBOUND(a, 2), UBOUND(a, 2)
        s = s // serialize(a(k1, k2)) // NEW_LINE('A')
      END DO
    END DO
  END FUNCTION real__8_2s2
  FUNCTION integer_2s1(a) RESULT(s)
    INTEGER, INTENT(IN) :: a(:)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k, k1
    s = "# entries" // NEW_LINE('A')
    DO k1 = LBOUND(a, 1), UBOUND(a, 1)
      s = s // serialize(a(k1)) // NEW_LINE('A')
    END DO
  END FUNCTION integer_2s1
  FUNCTION real__8_2s1(a) RESULT(s)
    REAL(KIND = 8), INTENT(IN) :: a(:)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k, k1
    s = "# entries" // NEW_LINE('A')
    DO k1 = LBOUND(a, 1), UBOUND(a, 1)
      s = s // serialize(a(k1)) // NEW_LINE('A')
    END DO
  END FUNCTION real__8_2s1
  FUNCTION integer__4_2s1(a) RESULT(s)
    INTEGER(KIND = 4), INTENT(IN) :: a(:)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k, k1
    s = "# entries" // NEW_LINE('A')
    DO k1 = LBOUND(a, 1), UBOUND(a, 1)
      s = s // serialize(a(k1)) // NEW_LINE('A')
    END DO
  END FUNCTION integer__4_2s1
  FUNCTION logical_2s(x) RESULT(s)
    LOGICAL, INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    ALLOCATE(CHARACTER(LEN = 50)::s)
    WRITE(s, *) x
    s = TRIM(s)
  END FUNCTION logical_2s
  FUNCTION integer1_2s(x) RESULT(s)
    INTEGER(KIND = 1), INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    ALLOCATE(CHARACTER(LEN = 50)::s)
    WRITE(s, *) x
    s = TRIM(s)
  END FUNCTION integer1_2s
  FUNCTION integer2_2s(x) RESULT(s)
    INTEGER(KIND = 2), INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    ALLOCATE(CHARACTER(LEN = 50)::s)
    WRITE(s, *) x
    s = TRIM(s)
  END FUNCTION integer2_2s
  FUNCTION integer4_2s(x) RESULT(s)
    INTEGER(KIND = 4), INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    ALLOCATE(CHARACTER(LEN = 50)::s)
    WRITE(s, *) x
    s = TRIM(s)
  END FUNCTION integer4_2s
  FUNCTION integer8_2s(x) RESULT(s)
    INTEGER(KIND = 8), INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    ALLOCATE(CHARACTER(LEN = 50)::s)
    WRITE(s, *) x
    s = TRIM(s)
  END FUNCTION integer8_2s
  FUNCTION real4_2s(x) RESULT(s)
    REAL(KIND = 4), INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    ALLOCATE(CHARACTER(LEN = 50)::s)
    WRITE(s, *) x
    s = TRIM(s)
  END FUNCTION real4_2s
  FUNCTION real8_2s(x) RESULT(s)
    REAL(KIND = 8), INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    ALLOCATE(CHARACTER(LEN = 50)::s)
    WRITE(s, *) x
    s = TRIM(s)
  END FUNCTION real8_2s
END MODULE serde