MODULE serde
  IMPLICIT NONE
  INTERFACE serialize
    MODULE PROCEDURE :: character_2s
    MODULE PROCEDURE aerosol_type_2s, cloud_type_2s, cloud_optics_type_2s, gas_type_2s, config_type_2s, flux_type_2s, thermodynamics_type_2s, single_level_type_2s, real__8_2s3, real__8_2s2, integer__4_2s1, logical_2s, integer1_2s, integer2_2s, integer4_2s, integer8_2s, real4_2s, real8_2s
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
  FUNCTION add_line(r, l) RESULT(s)
    CHARACTER(LEN = *), INTENT(IN) :: r
    CHARACTER(LEN = *), INTENT(IN) :: l
    CHARACTER(LEN = :), ALLOCATABLE :: s
    s = r // TRIM(l) // NEW_LINE('A')
  END FUNCTION add_line
  FUNCTION character_2s(x) RESULT(s)
    CHARACTER(LEN = *), INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    ALLOCATE(CHARACTER(LEN = LEN(x) + 1)::s)
    WRITE(s, '(g0)') TRIM(x)
    s = TRIM(s)
  END FUNCTION character_2s
  FUNCTION aerosol_type_2s(x) RESULT(s)
    USE radiation_aerosol, ONLY: aerosol_type
    TYPE(aerosol_type), TARGET, INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: kmeta, kmeta_0, kmeta_1, kmeta_2, kmeta_3, kmeta_4, kmeta_5, kmeta_6, kmeta_7, kmeta_8, kmeta_9
    s = ""
    s = add_line(s, '# od_sw')
    s = add_line(s, '# alloc')
    s = add_line(s, serialize(ALLOCATED(x % od_sw)))
    IF (ALLOCATED(x % od_sw)) THEN
      s = add_line(s, "# rank")
      s = add_line(s, serialize(3))
      s = add_line(s, "# size")
      DO kmeta = 1, 3
        s = add_line(s, serialize(SIZE(x % od_sw, kmeta)))
      END DO
      s = add_line(s, "# lbound")
      DO kmeta = 1, 3
        s = add_line(s, serialize(LBOUND(x % od_sw, kmeta)))
      END DO
      s = add_line(s, serialize(x % od_sw))
    END IF
    s = add_line(s, '# ssa_sw')
    s = add_line(s, '# alloc')
    s = add_line(s, serialize(ALLOCATED(x % ssa_sw)))
    IF (ALLOCATED(x % ssa_sw)) THEN
      s = add_line(s, "# rank")
      s = add_line(s, serialize(3))
      s = add_line(s, "# size")
      DO kmeta = 1, 3
        s = add_line(s, serialize(SIZE(x % ssa_sw, kmeta)))
      END DO
      s = add_line(s, "# lbound")
      DO kmeta = 1, 3
        s = add_line(s, serialize(LBOUND(x % ssa_sw, kmeta)))
      END DO
      s = add_line(s, serialize(x % ssa_sw))
    END IF
    s = add_line(s, '# g_sw')
    s = add_line(s, '# alloc')
    s = add_line(s, serialize(ALLOCATED(x % g_sw)))
    IF (ALLOCATED(x % g_sw)) THEN
      s = add_line(s, "# rank")
      s = add_line(s, serialize(3))
      s = add_line(s, "# size")
      DO kmeta = 1, 3
        s = add_line(s, serialize(SIZE(x % g_sw, kmeta)))
      END DO
      s = add_line(s, "# lbound")
      DO kmeta = 1, 3
        s = add_line(s, serialize(LBOUND(x % g_sw, kmeta)))
      END DO
      s = add_line(s, serialize(x % g_sw))
    END IF
    s = add_line(s, '# od_lw')
    s = add_line(s, '# alloc')
    s = add_line(s, serialize(ALLOCATED(x % od_lw)))
    IF (ALLOCATED(x % od_lw)) THEN
      s = add_line(s, "# rank")
      s = add_line(s, serialize(3))
      s = add_line(s, "# size")
      DO kmeta = 1, 3
        s = add_line(s, serialize(SIZE(x % od_lw, kmeta)))
      END DO
      s = add_line(s, "# lbound")
      DO kmeta = 1, 3
        s = add_line(s, serialize(LBOUND(x % od_lw, kmeta)))
      END DO
      s = add_line(s, serialize(x % od_lw))
    END IF
    s = add_line(s, '# ssa_lw')
    s = add_line(s, '# alloc')
    s = add_line(s, serialize(ALLOCATED(x % ssa_lw)))
    IF (ALLOCATED(x % ssa_lw)) THEN
      s = add_line(s, "# rank")
      s = add_line(s, serialize(3))
      s = add_line(s, "# size")
      DO kmeta = 1, 3
        s = add_line(s, serialize(SIZE(x % ssa_lw, kmeta)))
      END DO
      s = add_line(s, "# lbound")
      DO kmeta = 1, 3
        s = add_line(s, serialize(LBOUND(x % ssa_lw, kmeta)))
      END DO
      s = add_line(s, serialize(x % ssa_lw))
    END IF
  END FUNCTION aerosol_type_2s
  FUNCTION cloud_type_2s(x) RESULT(s)
    USE radiation_cloud, ONLY: cloud_type
    TYPE(cloud_type), TARGET, INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: kmeta, kmeta_0, kmeta_1, kmeta_2, kmeta_3, kmeta_4, kmeta_5, kmeta_6, kmeta_7, kmeta_8, kmeta_9
    s = ""
    s = add_line(s, '# mixing_ratio')
    s = add_line(s, '# alloc')
    s = add_line(s, serialize(ALLOCATED(x % mixing_ratio)))
    IF (ALLOCATED(x % mixing_ratio)) THEN
      s = add_line(s, "# rank")
      s = add_line(s, serialize(3))
      s = add_line(s, "# size")
      DO kmeta = 1, 3
        s = add_line(s, serialize(SIZE(x % mixing_ratio, kmeta)))
      END DO
      s = add_line(s, "# lbound")
      DO kmeta = 1, 3
        s = add_line(s, serialize(LBOUND(x % mixing_ratio, kmeta)))
      END DO
      s = add_line(s, serialize(x % mixing_ratio))
    END IF
    s = add_line(s, '# q_liq')
    s = add_line(s, '# assoc')
    s = add_line(s, serialize(ASSOCIATED(x % q_liq)))
    IF (ASSOCIATED(x % q_liq)) THEN
      kmeta = 0
      DO kmeta_0 = LBOUND(x % mixing_ratio, 1), UBOUND(x % mixing_ratio, 1)
        IF (ASSOCIATED(x % q_liq, x % mixing_ratio(kmeta_0, :, :))) THEN
          kmeta = 1
          s = s // "=> x%mixing_ratio("
          s = s // serialize(kmeta_0) // ',' // ':' // ',' // ':'
          s = s // "))" // NEW_LINE('A')
        END IF
      END DO
      DO kmeta_1 = LBOUND(x % mixing_ratio, 2), UBOUND(x % mixing_ratio, 2)
        IF (ASSOCIATED(x % q_liq, x % mixing_ratio(:, kmeta_1, :))) THEN
          kmeta = 1
          s = s // "=> x%mixing_ratio("
          s = s // ':' // ',' // serialize(kmeta_1) // ',' // ':'
          s = s // "))" // NEW_LINE('A')
        END IF
      END DO
      DO kmeta_2 = LBOUND(x % mixing_ratio, 3), UBOUND(x % mixing_ratio, 3)
        IF (ASSOCIATED(x % q_liq, x % mixing_ratio(:, :, kmeta_2))) THEN
          kmeta = 1
          s = s // "=> x%mixing_ratio("
          s = s // ':' // ',' // ':' // ',' // serialize(kmeta_2)
          s = s // "))" // NEW_LINE('A')
        END IF
      END DO
      IF (ASSOCIATED(x % q_liq, x % fraction(:, :))) THEN
        kmeta = 1
        s = s // "=> x%fraction("
        s = s // ':' // ',' // ':'
        s = s // "))" // NEW_LINE('A')
      END IF
      IF (kmeta == 0) THEN
        s = add_line(s, "=> missing")
      END IF
    END IF
    s = add_line(s, '# q_ice')
    s = add_line(s, '# assoc')
    s = add_line(s, serialize(ASSOCIATED(x % q_ice)))
    IF (ASSOCIATED(x % q_ice)) THEN
      kmeta = 0
      DO kmeta_0 = LBOUND(x % mixing_ratio, 1), UBOUND(x % mixing_ratio, 1)
        IF (ASSOCIATED(x % q_ice, x % mixing_ratio(kmeta_0, :, :))) THEN
          kmeta = 1
          s = s // "=> x%mixing_ratio("
          s = s // serialize(kmeta_0) // ',' // ':' // ',' // ':'
          s = s // "))" // NEW_LINE('A')
        END IF
      END DO
      DO kmeta_1 = LBOUND(x % mixing_ratio, 2), UBOUND(x % mixing_ratio, 2)
        IF (ASSOCIATED(x % q_ice, x % mixing_ratio(:, kmeta_1, :))) THEN
          kmeta = 1
          s = s // "=> x%mixing_ratio("
          s = s // ':' // ',' // serialize(kmeta_1) // ',' // ':'
          s = s // "))" // NEW_LINE('A')
        END IF
      END DO
      DO kmeta_2 = LBOUND(x % mixing_ratio, 3), UBOUND(x % mixing_ratio, 3)
        IF (ASSOCIATED(x % q_ice, x % mixing_ratio(:, :, kmeta_2))) THEN
          kmeta = 1
          s = s // "=> x%mixing_ratio("
          s = s // ':' // ',' // ':' // ',' // serialize(kmeta_2)
          s = s // "))" // NEW_LINE('A')
        END IF
      END DO
      IF (ASSOCIATED(x % q_ice, x % fraction(:, :))) THEN
        kmeta = 1
        s = s // "=> x%fraction("
        s = s // ':' // ',' // ':'
        s = s // "))" // NEW_LINE('A')
      END IF
      IF (kmeta == 0) THEN
        s = add_line(s, "=> missing")
      END IF
    END IF
    s = add_line(s, '# re_liq')
    s = add_line(s, '# assoc')
    s = add_line(s, serialize(ASSOCIATED(x % re_liq)))
    IF (ASSOCIATED(x % re_liq)) THEN
      kmeta = 0
      DO kmeta_0 = LBOUND(x % mixing_ratio, 1), UBOUND(x % mixing_ratio, 1)
        IF (ASSOCIATED(x % re_liq, x % mixing_ratio(kmeta_0, :, :))) THEN
          kmeta = 1
          s = s // "=> x%mixing_ratio("
          s = s // serialize(kmeta_0) // ',' // ':' // ',' // ':'
          s = s // "))" // NEW_LINE('A')
        END IF
      END DO
      DO kmeta_1 = LBOUND(x % mixing_ratio, 2), UBOUND(x % mixing_ratio, 2)
        IF (ASSOCIATED(x % re_liq, x % mixing_ratio(:, kmeta_1, :))) THEN
          kmeta = 1
          s = s // "=> x%mixing_ratio("
          s = s // ':' // ',' // serialize(kmeta_1) // ',' // ':'
          s = s // "))" // NEW_LINE('A')
        END IF
      END DO
      DO kmeta_2 = LBOUND(x % mixing_ratio, 3), UBOUND(x % mixing_ratio, 3)
        IF (ASSOCIATED(x % re_liq, x % mixing_ratio(:, :, kmeta_2))) THEN
          kmeta = 1
          s = s // "=> x%mixing_ratio("
          s = s // ':' // ',' // ':' // ',' // serialize(kmeta_2)
          s = s // "))" // NEW_LINE('A')
        END IF
      END DO
      IF (ASSOCIATED(x % re_liq, x % fraction(:, :))) THEN
        kmeta = 1
        s = s // "=> x%fraction("
        s = s // ':' // ',' // ':'
        s = s // "))" // NEW_LINE('A')
      END IF
      IF (kmeta == 0) THEN
        s = add_line(s, "=> missing")
      END IF
    END IF
    s = add_line(s, '# re_ice')
    s = add_line(s, '# assoc')
    s = add_line(s, serialize(ASSOCIATED(x % re_ice)))
    IF (ASSOCIATED(x % re_ice)) THEN
      kmeta = 0
      DO kmeta_0 = LBOUND(x % mixing_ratio, 1), UBOUND(x % mixing_ratio, 1)
        IF (ASSOCIATED(x % re_ice, x % mixing_ratio(kmeta_0, :, :))) THEN
          kmeta = 1
          s = s // "=> x%mixing_ratio("
          s = s // serialize(kmeta_0) // ',' // ':' // ',' // ':'
          s = s // "))" // NEW_LINE('A')
        END IF
      END DO
      DO kmeta_1 = LBOUND(x % mixing_ratio, 2), UBOUND(x % mixing_ratio, 2)
        IF (ASSOCIATED(x % re_ice, x % mixing_ratio(:, kmeta_1, :))) THEN
          kmeta = 1
          s = s // "=> x%mixing_ratio("
          s = s // ':' // ',' // serialize(kmeta_1) // ',' // ':'
          s = s // "))" // NEW_LINE('A')
        END IF
      END DO
      DO kmeta_2 = LBOUND(x % mixing_ratio, 3), UBOUND(x % mixing_ratio, 3)
        IF (ASSOCIATED(x % re_ice, x % mixing_ratio(:, :, kmeta_2))) THEN
          kmeta = 1
          s = s // "=> x%mixing_ratio("
          s = s // ':' // ',' // ':' // ',' // serialize(kmeta_2)
          s = s // "))" // NEW_LINE('A')
        END IF
      END DO
      IF (ASSOCIATED(x % re_ice, x % fraction(:, :))) THEN
        kmeta = 1
        s = s // "=> x%fraction("
        s = s // ':' // ',' // ':'
        s = s // "))" // NEW_LINE('A')
      END IF
      IF (kmeta == 0) THEN
        s = add_line(s, "=> missing")
      END IF
    END IF
    s = add_line(s, '# fraction')
    s = add_line(s, '# alloc')
    s = add_line(s, serialize(ALLOCATED(x % fraction)))
    IF (ALLOCATED(x % fraction)) THEN
      s = add_line(s, "# rank")
      s = add_line(s, serialize(2))
      s = add_line(s, "# size")
      DO kmeta = 1, 2
        s = add_line(s, serialize(SIZE(x % fraction, kmeta)))
      END DO
      s = add_line(s, "# lbound")
      DO kmeta = 1, 2
        s = add_line(s, serialize(LBOUND(x % fraction, kmeta)))
      END DO
      s = add_line(s, serialize(x % fraction))
    END IF
  END FUNCTION cloud_type_2s
  FUNCTION cloud_optics_type_2s(x) RESULT(s)
    USE radiation_cloud_optics_data, ONLY: cloud_optics_type
    TYPE(cloud_optics_type), TARGET, INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: kmeta, kmeta_0, kmeta_1, kmeta_2, kmeta_3, kmeta_4, kmeta_5, kmeta_6, kmeta_7, kmeta_8, kmeta_9
    s = ""
    s = add_line(s, '# liq_coeff_lw')
    s = add_line(s, '# alloc')
    s = add_line(s, serialize(ALLOCATED(x % liq_coeff_lw)))
    IF (ALLOCATED(x % liq_coeff_lw)) THEN
      s = add_line(s, "# rank")
      s = add_line(s, serialize(2))
      s = add_line(s, "# size")
      DO kmeta = 1, 2
        s = add_line(s, serialize(SIZE(x % liq_coeff_lw, kmeta)))
      END DO
      s = add_line(s, "# lbound")
      DO kmeta = 1, 2
        s = add_line(s, serialize(LBOUND(x % liq_coeff_lw, kmeta)))
      END DO
      s = add_line(s, serialize(x % liq_coeff_lw))
    END IF
    s = add_line(s, '# liq_coeff_sw')
    s = add_line(s, '# alloc')
    s = add_line(s, serialize(ALLOCATED(x % liq_coeff_sw)))
    IF (ALLOCATED(x % liq_coeff_sw)) THEN
      s = add_line(s, "# rank")
      s = add_line(s, serialize(2))
      s = add_line(s, "# size")
      DO kmeta = 1, 2
        s = add_line(s, serialize(SIZE(x % liq_coeff_sw, kmeta)))
      END DO
      s = add_line(s, "# lbound")
      DO kmeta = 1, 2
        s = add_line(s, serialize(LBOUND(x % liq_coeff_sw, kmeta)))
      END DO
      s = add_line(s, serialize(x % liq_coeff_sw))
    END IF
    s = add_line(s, '# ice_coeff_lw')
    s = add_line(s, '# alloc')
    s = add_line(s, serialize(ALLOCATED(x % ice_coeff_lw)))
    IF (ALLOCATED(x % ice_coeff_lw)) THEN
      s = add_line(s, "# rank")
      s = add_line(s, serialize(2))
      s = add_line(s, "# size")
      DO kmeta = 1, 2
        s = add_line(s, serialize(SIZE(x % ice_coeff_lw, kmeta)))
      END DO
      s = add_line(s, "# lbound")
      DO kmeta = 1, 2
        s = add_line(s, serialize(LBOUND(x % ice_coeff_lw, kmeta)))
      END DO
      s = add_line(s, serialize(x % ice_coeff_lw))
    END IF
    s = add_line(s, '# ice_coeff_sw')
    s = add_line(s, '# alloc')
    s = add_line(s, serialize(ALLOCATED(x % ice_coeff_sw)))
    IF (ALLOCATED(x % ice_coeff_sw)) THEN
      s = add_line(s, "# rank")
      s = add_line(s, serialize(2))
      s = add_line(s, "# size")
      DO kmeta = 1, 2
        s = add_line(s, serialize(SIZE(x % ice_coeff_sw, kmeta)))
      END DO
      s = add_line(s, "# lbound")
      DO kmeta = 1, 2
        s = add_line(s, serialize(LBOUND(x % ice_coeff_sw, kmeta)))
      END DO
      s = add_line(s, serialize(x % ice_coeff_sw))
    END IF
  END FUNCTION cloud_optics_type_2s
  FUNCTION gas_type_2s(x) RESULT(s)
    USE radiation_gas, ONLY: gas_type
    TYPE(gas_type), TARGET, INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: kmeta, kmeta_0, kmeta_1, kmeta_2, kmeta_3, kmeta_4, kmeta_5, kmeta_6, kmeta_7, kmeta_8, kmeta_9
    s = ""
  END FUNCTION gas_type_2s
  FUNCTION config_type_2s(x) RESULT(s)
    USE radiation_config, ONLY: config_type
    TYPE(config_type), TARGET, INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: kmeta, kmeta_0, kmeta_1, kmeta_2, kmeta_3, kmeta_4, kmeta_5, kmeta_6, kmeta_7, kmeta_8, kmeta_9
    s = ""
    s = add_line(s, '# i_emiss_from_band_lw')
    s = add_line(s, '# alloc')
    s = add_line(s, serialize(ALLOCATED(x % i_emiss_from_band_lw)))
    IF (ALLOCATED(x % i_emiss_from_band_lw)) THEN
      s = add_line(s, "# rank")
      s = add_line(s, serialize(1))
      s = add_line(s, "# size")
      DO kmeta = 1, 1
        s = add_line(s, serialize(SIZE(x % i_emiss_from_band_lw, kmeta)))
      END DO
      s = add_line(s, "# lbound")
      DO kmeta = 1, 1
        s = add_line(s, serialize(LBOUND(x % i_emiss_from_band_lw, kmeta)))
      END DO
      s = add_line(s, serialize(x % i_emiss_from_band_lw))
    END IF
    s = add_line(s, '# sw_albedo_weights')
    s = add_line(s, '# alloc')
    s = add_line(s, serialize(ALLOCATED(x % sw_albedo_weights)))
    IF (ALLOCATED(x % sw_albedo_weights)) THEN
      s = add_line(s, "# rank")
      s = add_line(s, serialize(2))
      s = add_line(s, "# size")
      DO kmeta = 1, 2
        s = add_line(s, serialize(SIZE(x % sw_albedo_weights, kmeta)))
      END DO
      s = add_line(s, "# lbound")
      DO kmeta = 1, 2
        s = add_line(s, serialize(LBOUND(x % sw_albedo_weights, kmeta)))
      END DO
      s = add_line(s, serialize(x % sw_albedo_weights))
    END IF
    s = add_line(s, '# i_band_from_reordered_g_lw')
    s = add_line(s, '# alloc')
    s = add_line(s, serialize(ALLOCATED(x % i_band_from_reordered_g_lw)))
    IF (ALLOCATED(x % i_band_from_reordered_g_lw)) THEN
      s = add_line(s, "# rank")
      s = add_line(s, serialize(1))
      s = add_line(s, "# size")
      DO kmeta = 1, 1
        s = add_line(s, serialize(SIZE(x % i_band_from_reordered_g_lw, kmeta)))
      END DO
      s = add_line(s, "# lbound")
      DO kmeta = 1, 1
        s = add_line(s, serialize(LBOUND(x % i_band_from_reordered_g_lw, kmeta)))
      END DO
      s = add_line(s, serialize(x % i_band_from_reordered_g_lw))
    END IF
    s = add_line(s, '# i_band_from_reordered_g_sw')
    s = add_line(s, '# alloc')
    s = add_line(s, serialize(ALLOCATED(x % i_band_from_reordered_g_sw)))
    IF (ALLOCATED(x % i_band_from_reordered_g_sw)) THEN
      s = add_line(s, "# rank")
      s = add_line(s, serialize(1))
      s = add_line(s, "# size")
      DO kmeta = 1, 1
        s = add_line(s, serialize(SIZE(x % i_band_from_reordered_g_sw, kmeta)))
      END DO
      s = add_line(s, "# lbound")
      DO kmeta = 1, 1
        s = add_line(s, serialize(LBOUND(x % i_band_from_reordered_g_sw, kmeta)))
      END DO
      s = add_line(s, serialize(x % i_band_from_reordered_g_sw))
    END IF
    s = add_line(s, '# cloud_optics')
    s = add_line(s, serialize(x % cloud_optics))
  END FUNCTION config_type_2s
  FUNCTION flux_type_2s(x) RESULT(s)
    USE radiation_flux, ONLY: flux_type
    TYPE(flux_type), TARGET, INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: kmeta, kmeta_0, kmeta_1, kmeta_2, kmeta_3, kmeta_4, kmeta_5, kmeta_6, kmeta_7, kmeta_8, kmeta_9
    s = ""
    s = add_line(s, '# sw_dn_diffuse_surf_g')
    s = add_line(s, '# alloc')
    s = add_line(s, serialize(ALLOCATED(x % sw_dn_diffuse_surf_g)))
    IF (ALLOCATED(x % sw_dn_diffuse_surf_g)) THEN
      s = add_line(s, "# rank")
      s = add_line(s, serialize(2))
      s = add_line(s, "# size")
      DO kmeta = 1, 2
        s = add_line(s, serialize(SIZE(x % sw_dn_diffuse_surf_g, kmeta)))
      END DO
      s = add_line(s, "# lbound")
      DO kmeta = 1, 2
        s = add_line(s, serialize(LBOUND(x % sw_dn_diffuse_surf_g, kmeta)))
      END DO
      s = add_line(s, serialize(x % sw_dn_diffuse_surf_g))
    END IF
    s = add_line(s, '# sw_dn_direct_surf_g')
    s = add_line(s, '# alloc')
    s = add_line(s, serialize(ALLOCATED(x % sw_dn_direct_surf_g)))
    IF (ALLOCATED(x % sw_dn_direct_surf_g)) THEN
      s = add_line(s, "# rank")
      s = add_line(s, serialize(2))
      s = add_line(s, "# size")
      DO kmeta = 1, 2
        s = add_line(s, serialize(SIZE(x % sw_dn_direct_surf_g, kmeta)))
      END DO
      s = add_line(s, "# lbound")
      DO kmeta = 1, 2
        s = add_line(s, serialize(LBOUND(x % sw_dn_direct_surf_g, kmeta)))
      END DO
      s = add_line(s, serialize(x % sw_dn_direct_surf_g))
    END IF
    s = add_line(s, '# sw_dn_diffuse_surf_clear_g')
    s = add_line(s, '# alloc')
    s = add_line(s, serialize(ALLOCATED(x % sw_dn_diffuse_surf_clear_g)))
    IF (ALLOCATED(x % sw_dn_diffuse_surf_clear_g)) THEN
      s = add_line(s, "# rank")
      s = add_line(s, serialize(2))
      s = add_line(s, "# size")
      DO kmeta = 1, 2
        s = add_line(s, serialize(SIZE(x % sw_dn_diffuse_surf_clear_g, kmeta)))
      END DO
      s = add_line(s, "# lbound")
      DO kmeta = 1, 2
        s = add_line(s, serialize(LBOUND(x % sw_dn_diffuse_surf_clear_g, kmeta)))
      END DO
      s = add_line(s, serialize(x % sw_dn_diffuse_surf_clear_g))
    END IF
    s = add_line(s, '# sw_dn_direct_surf_clear_g')
    s = add_line(s, '# alloc')
    s = add_line(s, serialize(ALLOCATED(x % sw_dn_direct_surf_clear_g)))
    IF (ALLOCATED(x % sw_dn_direct_surf_clear_g)) THEN
      s = add_line(s, "# rank")
      s = add_line(s, serialize(2))
      s = add_line(s, "# size")
      DO kmeta = 1, 2
        s = add_line(s, serialize(SIZE(x % sw_dn_direct_surf_clear_g, kmeta)))
      END DO
      s = add_line(s, "# lbound")
      DO kmeta = 1, 2
        s = add_line(s, serialize(LBOUND(x % sw_dn_direct_surf_clear_g, kmeta)))
      END DO
      s = add_line(s, serialize(x % sw_dn_direct_surf_clear_g))
    END IF
    s = add_line(s, '# sw_dn_surf_band')
    s = add_line(s, '# alloc')
    s = add_line(s, serialize(ALLOCATED(x % sw_dn_surf_band)))
    IF (ALLOCATED(x % sw_dn_surf_band)) THEN
      s = add_line(s, "# rank")
      s = add_line(s, serialize(2))
      s = add_line(s, "# size")
      DO kmeta = 1, 2
        s = add_line(s, serialize(SIZE(x % sw_dn_surf_band, kmeta)))
      END DO
      s = add_line(s, "# lbound")
      DO kmeta = 1, 2
        s = add_line(s, serialize(LBOUND(x % sw_dn_surf_band, kmeta)))
      END DO
      s = add_line(s, serialize(x % sw_dn_surf_band))
    END IF
    s = add_line(s, '# sw_dn_direct_surf_band')
    s = add_line(s, '# alloc')
    s = add_line(s, serialize(ALLOCATED(x % sw_dn_direct_surf_band)))
    IF (ALLOCATED(x % sw_dn_direct_surf_band)) THEN
      s = add_line(s, "# rank")
      s = add_line(s, serialize(2))
      s = add_line(s, "# size")
      DO kmeta = 1, 2
        s = add_line(s, serialize(SIZE(x % sw_dn_direct_surf_band, kmeta)))
      END DO
      s = add_line(s, "# lbound")
      DO kmeta = 1, 2
        s = add_line(s, serialize(LBOUND(x % sw_dn_direct_surf_band, kmeta)))
      END DO
      s = add_line(s, serialize(x % sw_dn_direct_surf_band))
    END IF
    s = add_line(s, '# sw_dn_surf_clear_band')
    s = add_line(s, '# alloc')
    s = add_line(s, serialize(ALLOCATED(x % sw_dn_surf_clear_band)))
    IF (ALLOCATED(x % sw_dn_surf_clear_band)) THEN
      s = add_line(s, "# rank")
      s = add_line(s, serialize(2))
      s = add_line(s, "# size")
      DO kmeta = 1, 2
        s = add_line(s, serialize(SIZE(x % sw_dn_surf_clear_band, kmeta)))
      END DO
      s = add_line(s, "# lbound")
      DO kmeta = 1, 2
        s = add_line(s, serialize(LBOUND(x % sw_dn_surf_clear_band, kmeta)))
      END DO
      s = add_line(s, serialize(x % sw_dn_surf_clear_band))
    END IF
    s = add_line(s, '# sw_dn_direct_surf_clear_band')
    s = add_line(s, '# alloc')
    s = add_line(s, serialize(ALLOCATED(x % sw_dn_direct_surf_clear_band)))
    IF (ALLOCATED(x % sw_dn_direct_surf_clear_band)) THEN
      s = add_line(s, "# rank")
      s = add_line(s, serialize(2))
      s = add_line(s, "# size")
      DO kmeta = 1, 2
        s = add_line(s, serialize(SIZE(x % sw_dn_direct_surf_clear_band, kmeta)))
      END DO
      s = add_line(s, "# lbound")
      DO kmeta = 1, 2
        s = add_line(s, serialize(LBOUND(x % sw_dn_direct_surf_clear_band, kmeta)))
      END DO
      s = add_line(s, serialize(x % sw_dn_direct_surf_clear_band))
    END IF
  END FUNCTION flux_type_2s
  FUNCTION thermodynamics_type_2s(x) RESULT(s)
    USE radiation_thermodynamics, ONLY: thermodynamics_type
    TYPE(thermodynamics_type), TARGET, INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: kmeta, kmeta_0, kmeta_1, kmeta_2, kmeta_3, kmeta_4, kmeta_5, kmeta_6, kmeta_7, kmeta_8, kmeta_9
    s = ""
    s = add_line(s, '# pressure_hl')
    s = add_line(s, '# alloc')
    s = add_line(s, serialize(ALLOCATED(x % pressure_hl)))
    IF (ALLOCATED(x % pressure_hl)) THEN
      s = add_line(s, "# rank")
      s = add_line(s, serialize(2))
      s = add_line(s, "# size")
      DO kmeta = 1, 2
        s = add_line(s, serialize(SIZE(x % pressure_hl, kmeta)))
      END DO
      s = add_line(s, "# lbound")
      DO kmeta = 1, 2
        s = add_line(s, serialize(LBOUND(x % pressure_hl, kmeta)))
      END DO
      s = add_line(s, serialize(x % pressure_hl))
    END IF
  END FUNCTION thermodynamics_type_2s
  FUNCTION single_level_type_2s(x) RESULT(s)
    USE radiation_single_level, ONLY: single_level_type
    TYPE(single_level_type), TARGET, INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: kmeta, kmeta_0, kmeta_1, kmeta_2, kmeta_3, kmeta_4, kmeta_5, kmeta_6, kmeta_7, kmeta_8, kmeta_9
    s = ""
    s = add_line(s, '# sw_albedo')
    s = add_line(s, '# alloc')
    s = add_line(s, serialize(ALLOCATED(x % sw_albedo)))
    IF (ALLOCATED(x % sw_albedo)) THEN
      s = add_line(s, "# rank")
      s = add_line(s, serialize(2))
      s = add_line(s, "# size")
      DO kmeta = 1, 2
        s = add_line(s, serialize(SIZE(x % sw_albedo, kmeta)))
      END DO
      s = add_line(s, "# lbound")
      DO kmeta = 1, 2
        s = add_line(s, serialize(LBOUND(x % sw_albedo, kmeta)))
      END DO
      s = add_line(s, serialize(x % sw_albedo))
    END IF
    s = add_line(s, '# sw_albedo_direct')
    s = add_line(s, '# alloc')
    s = add_line(s, serialize(ALLOCATED(x % sw_albedo_direct)))
    IF (ALLOCATED(x % sw_albedo_direct)) THEN
      s = add_line(s, "# rank")
      s = add_line(s, serialize(2))
      s = add_line(s, "# size")
      DO kmeta = 1, 2
        s = add_line(s, serialize(SIZE(x % sw_albedo_direct, kmeta)))
      END DO
      s = add_line(s, "# lbound")
      DO kmeta = 1, 2
        s = add_line(s, serialize(LBOUND(x % sw_albedo_direct, kmeta)))
      END DO
      s = add_line(s, serialize(x % sw_albedo_direct))
    END IF
    s = add_line(s, '# lw_emissivity')
    s = add_line(s, '# alloc')
    s = add_line(s, serialize(ALLOCATED(x % lw_emissivity)))
    IF (ALLOCATED(x % lw_emissivity)) THEN
      s = add_line(s, "# rank")
      s = add_line(s, serialize(2))
      s = add_line(s, "# size")
      DO kmeta = 1, 2
        s = add_line(s, serialize(SIZE(x % lw_emissivity, kmeta)))
      END DO
      s = add_line(s, "# lbound")
      DO kmeta = 1, 2
        s = add_line(s, serialize(LBOUND(x % lw_emissivity, kmeta)))
      END DO
      s = add_line(s, serialize(x % lw_emissivity))
    END IF
  END FUNCTION single_level_type_2s
  FUNCTION real__8_2s3(a) RESULT(s)
    REAL(KIND = 8), INTENT(IN) :: a(:, :, :)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k, k1, k2, k3
    s = ""
    s = add_line(s, "# entries")
    DO k1 = LBOUND(a, 1), UBOUND(a, 1)
      DO k2 = LBOUND(a, 2), UBOUND(a, 2)
        DO k3 = LBOUND(a, 3), UBOUND(a, 3)
          s = add_line(s, serialize(a(k1, k2, k3)))
        END DO
      END DO
    END DO
  END FUNCTION real__8_2s3
  FUNCTION real__8_2s2(a) RESULT(s)
    REAL(KIND = 8), INTENT(IN) :: a(:, :)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k, k1, k2
    s = ""
    s = add_line(s, "# entries")
    DO k1 = LBOUND(a, 1), UBOUND(a, 1)
      DO k2 = LBOUND(a, 2), UBOUND(a, 2)
        s = add_line(s, serialize(a(k1, k2)))
      END DO
    END DO
  END FUNCTION real__8_2s2
  FUNCTION integer__4_2s1(a) RESULT(s)
    INTEGER(KIND = 4), INTENT(IN) :: a(:)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k, k1
    s = ""
    s = add_line(s, "# entries")
    DO k1 = LBOUND(a, 1), UBOUND(a, 1)
      s = add_line(s, serialize(a(k1)))
    END DO
  END FUNCTION integer__4_2s1
  FUNCTION logical_2s(x) RESULT(s)
    LOGICAL, INTENT(IN) :: x
    INTEGER :: y
    CHARACTER(LEN = :), ALLOCATABLE :: s
    ALLOCATE(CHARACTER(LEN = 50)::s)
    y = x
    WRITE(s, '(g0)') y
    s = TRIM(s)
  END FUNCTION logical_2s
  FUNCTION integer1_2s(x) RESULT(s)
    INTEGER(KIND = 1), INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    ALLOCATE(CHARACTER(LEN = 50)::s)
    WRITE(s, '(g0)') x
    s = TRIM(s)
  END FUNCTION integer1_2s
  FUNCTION integer2_2s(x) RESULT(s)
    INTEGER(KIND = 2), INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    ALLOCATE(CHARACTER(LEN = 50)::s)
    WRITE(s, '(g0)') x
    s = TRIM(s)
  END FUNCTION integer2_2s
  FUNCTION integer4_2s(x) RESULT(s)
    INTEGER(KIND = 4), INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    ALLOCATE(CHARACTER(LEN = 50)::s)
    WRITE(s, '(g0)') x
    s = TRIM(s)
  END FUNCTION integer4_2s
  FUNCTION integer8_2s(x) RESULT(s)
    INTEGER(KIND = 8), INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    ALLOCATE(CHARACTER(LEN = 50)::s)
    WRITE(s, '(g0)') x
    s = TRIM(s)
  END FUNCTION integer8_2s
  FUNCTION real4_2s(x) RESULT(s)
    REAL(KIND = 4), INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    ALLOCATE(CHARACTER(LEN = 50)::s)
    WRITE(s, '(g0)') x
    s = TRIM(s)
  END FUNCTION real4_2s
  FUNCTION real8_2s(x) RESULT(s)
    REAL(KIND = 8), INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    ALLOCATE(CHARACTER(LEN = 50)::s)
    WRITE(s, '(g0)') x
    s = TRIM(s)
  END FUNCTION real8_2s
END MODULE serde