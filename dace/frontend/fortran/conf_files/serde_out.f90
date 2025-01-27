MODULE serde
  IMPLICIT NONE
  INTERFACE serialize
    MODULE PROCEDURE int_2s, int_a2s, int_aa2s, int_aaa2s
    MODULE PROCEDURE real_2s, real_a2s, real_aa2s, real_aaa2s
    MODULE PROCEDURE double_2s, double_a2s, double_aa2s, double_aaa2s
    MODULE PROCEDURE logical_2s, logical_a2s, logical_aa2s, logical_aaa2s
    MODULE PROCEDURE aerosol_type_2s, cloud_type_2s, cloud_optics_type_2s, gas_type_2s, pdf_sampler_type_2s, config_type_2s, flux_type_2s, thermodynamics_type_2s, randomnumberstream_2s, single_level_type_2s
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
  FUNCTION rank_2s_(a) RESULT(s)
    CLASS(*), INTENT(IN) :: a(*)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k
    s = "# rank" // NEW_LINE('A') // serialize(rank(a)) // NEW_LINE('A')
    s = s // "# size" // NEW_LINE('A')
    DO k = 1, rank(a)
      s = s // serialize(SIZE(a, k)) // NEW_LINE('A')
    END DO
    s = s // "# lbound" // NEW_LINE('A')
    DO k = 1, rank(a)
      s = s // serialize(LBOUND(a, k)) // NEW_LINE('A')
    END DO
  END FUNCTION rank_2s_
  FUNCTION int_2s(i) RESULT(s)
    INTEGER, INTENT(IN) :: i
    CHARACTER(LEN = :), ALLOCATABLE :: s
    ALLOCATE(CHARACTER(LEN = 50)::s)
    WRITE(s, *) i
    s = TRIM(s)
  END FUNCTION int_2s
  FUNCTION int_a2s(i) RESULT(s)
    INTEGER, INTENT(IN) :: i(:)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k1
    s = rank_2s_(i) // "# entries" // NEW_LINE('A')
    DO k1 = 1, SIZE(i, 1)
      s = s // serialize(i(k1)) // NEW_LINE('A')
    END DO
  END FUNCTION int_a2s
  FUNCTION int_aa2s(i) RESULT(s)
    INTEGER, INTENT(IN) :: i(:, :)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k1, k2
    s = rank_2s_(i) // "# entries" // NEW_LINE('A')
    DO k1 = 1, SIZE(i, 1)
      DO k2 = 1, SIZE(i, 2)
        s = s // serialize(i(k1, k2)) // NEW_LINE('A')
      END DO
    END DO
  END FUNCTION int_aa2s
  FUNCTION int_aaa2s(i) RESULT(s)
    INTEGER, INTENT(IN) :: i(:, :, :)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k1, k2, k3
    s = rank_2s_(i) // "# entries" // NEW_LINE('A')
    DO k1 = 1, SIZE(i, 1)
      DO k2 = 1, SIZE(i, 2)
        DO k3 = 1, SIZE(i, 3)
          s = s // serialize(i(k1, k2, k3)) // NEW_LINE('A')
        END DO
      END DO
    END DO
  END FUNCTION int_aaa2s
  FUNCTION int_za2s(i) RESULT(s)
    INTEGER, ALLOCATABLE, INTENT(IN) :: i(:)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k1
    s = "# allocated" // NEW_LINE('A') // serialize(ALLOCATED(i)) // NEW_LINE('A')
    IF (ALLOCATED(i)) THEN
      s = rank_2s_(i) // "# entries" // NEW_LINE('A')
      DO k1 = 1, SIZE(i, 1)
        s = s // serialize(i(k1)) // NEW_LINE('A')
      END DO
    END IF
  END FUNCTION int_za2s
  FUNCTION int_zaa2s(i) RESULT(s)
    INTEGER, ALLOCATABLE, INTENT(IN) :: i(:, :)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k1, k2
    s = "# allocated" // NEW_LINE('A') // serialize(ALLOCATED(i)) // NEW_LINE('A')
    IF (ALLOCATED(i)) THEN
      s = rank_2s_(i) // "# entries" // NEW_LINE('A')
      DO k1 = 1, SIZE(i, 1)
        DO k2 = 1, SIZE(i, 2)
          s = s // serialize(i(k1, k2)) // NEW_LINE('A')
        END DO
      END DO
    END IF
  END FUNCTION int_zaa2s
  FUNCTION int_zaaa2s(i) RESULT(s)
    INTEGER, ALLOCATABLE, INTENT(IN) :: i(:, :, :)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k1, k2, k3
    s = "# allocated" // NEW_LINE('A') // serialize(ALLOCATED(i)) // NEW_LINE('A')
    IF (ALLOCATED(i)) THEN
      s = rank_2s_(i) // "# entries" // NEW_LINE('A')
      DO k1 = 1, SIZE(i, 1)
        DO k2 = 1, SIZE(i, 2)
          DO k3 = 1, SIZE(i, 3)
            s = s // serialize(i(k1, k2, k3)) // NEW_LINE('A')
          END DO
        END DO
      END DO
    END IF
  END FUNCTION int_zaaa2s
  FUNCTION real_2s(r) RESULT(s)
    REAL, INTENT(IN) :: r
    CHARACTER(LEN = :), ALLOCATABLE :: s
    ALLOCATE(CHARACTER(LEN = 50)::s)
    WRITE(s, *) r
    s = TRIM(s)
  END FUNCTION real_2s
  FUNCTION real_a2s(i) RESULT(s)
    REAL, INTENT(IN) :: i(:)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k1
    s = rank_2s_(i) // "# entries" // NEW_LINE('A')
    DO k1 = 1, SIZE(i, 1)
      s = s // serialize(i(k1)) // NEW_LINE('A')
    END DO
  END FUNCTION real_a2s
  FUNCTION real_aa2s(i) RESULT(s)
    REAL, INTENT(IN) :: i(:, :)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k1, k2
    s = rank_2s_(i) // "# entries" // NEW_LINE('A')
    DO k1 = 1, SIZE(i, 1)
      DO k2 = 1, SIZE(i, 2)
        s = s // serialize(i(k1, k2)) // NEW_LINE('A')
      END DO
    END DO
  END FUNCTION real_aa2s
  FUNCTION real_aaa2s(i) RESULT(s)
    REAL, INTENT(IN) :: i(:, :, :)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k1, k2, k3
    s = rank_2s_(i) // "# entries" // NEW_LINE('A')
    DO k1 = 1, SIZE(i, 1)
      DO k2 = 1, SIZE(i, 2)
        DO k3 = 1, SIZE(i, 3)
          s = s // serialize(i(k1, k2, k3)) // NEW_LINE('A')
        END DO
      END DO
    END DO
  END FUNCTION real_aaa2s
  FUNCTION real_za2s(i) RESULT(s)
    REAL, ALLOCATABLE, INTENT(IN) :: i(:)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k1
    s = "# allocated" // NEW_LINE('A') // serialize(ALLOCATED(i)) // NEW_LINE('A')
    IF (ALLOCATED(i)) THEN
      s = rank_2s_(i) // "# entries" // NEW_LINE('A')
      DO k1 = 1, SIZE(i, 1)
        s = s // serialize(i(k1)) // NEW_LINE('A')
      END DO
    END IF
  END FUNCTION real_za2s
  FUNCTION real_zaa2s(i) RESULT(s)
    REAL, ALLOCATABLE, INTENT(IN) :: i(:, :)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k1, k2
    s = "# allocated" // NEW_LINE('A') // serialize(ALLOCATED(i)) // NEW_LINE('A')
    IF (ALLOCATED(i)) THEN
      s = rank_2s_(i) // "# entries" // NEW_LINE('A')
      DO k1 = 1, SIZE(i, 1)
        DO k2 = 1, SIZE(i, 2)
          s = s // serialize(i(k1, k2)) // NEW_LINE('A')
        END DO
      END DO
    END IF
  END FUNCTION real_zaa2s
  FUNCTION real_zaaa2s(i) RESULT(s)
    REAL, ALLOCATABLE, INTENT(IN) :: i(:, :, :)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k1, k2, k3
    s = "# allocated" // NEW_LINE('A') // serialize(ALLOCATED(i)) // NEW_LINE('A')
    IF (ALLOCATED(i)) THEN
      s = rank_2s_(i) // "# entries" // NEW_LINE('A')
      DO k1 = 1, SIZE(i, 1)
        DO k2 = 1, SIZE(i, 2)
          DO k3 = 1, SIZE(i, 3)
            s = s // serialize(i(k1, k2, k3)) // NEW_LINE('A')
          END DO
        END DO
      END DO
    END IF
  END FUNCTION real_zaaa2s
  FUNCTION double_2s(r) RESULT(s)
    DOUBLE PRECISION, INTENT(IN) :: r
    CHARACTER(LEN = :), ALLOCATABLE :: s
    ALLOCATE(CHARACTER(LEN = 50)::s)
    WRITE(s, *) r
    s = TRIM(s)
  END FUNCTION double_2s
  FUNCTION double_a2s(i) RESULT(s)
    DOUBLE PRECISION, INTENT(IN) :: i(:)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k1
    s = rank_2s_(i) // "# entries" // NEW_LINE('A')
    DO k1 = 1, SIZE(i, 1)
      s = s // serialize(i(k1)) // NEW_LINE('A')
    END DO
  END FUNCTION double_a2s
  FUNCTION double_aa2s(i) RESULT(s)
    DOUBLE PRECISION, INTENT(IN) :: i(:, :)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k1, k2
    s = rank_2s_(i) // "# entries" // NEW_LINE('A')
    DO k1 = 1, SIZE(i, 1)
      DO k2 = 1, SIZE(i, 2)
        s = s // serialize(i(k1, k2)) // NEW_LINE('A')
      END DO
    END DO
  END FUNCTION double_aa2s
  FUNCTION double_aaa2s(i) RESULT(s)
    DOUBLE PRECISION, INTENT(IN) :: i(:, :, :)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k1, k2, k3
    s = rank_2s_(i) // "# entries" // NEW_LINE('A')
    DO k1 = 1, SIZE(i, 1)
      DO k2 = 1, SIZE(i, 2)
        DO k3 = 1, SIZE(i, 3)
          s = s // serialize(i(k1, k2, k3)) // NEW_LINE('A')
        END DO
      END DO
    END DO
  END FUNCTION double_aaa2s
  FUNCTION double_za2s(i) RESULT(s)
    DOUBLE PRECISION, ALLOCATABLE, INTENT(IN) :: i(:)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k1
    s = "# allocated" // NEW_LINE('A') // serialize(ALLOCATED(i)) // NEW_LINE('A')
    IF (ALLOCATED(i)) THEN
      s = rank_2s_(i) // "# entries" // NEW_LINE('A')
      DO k1 = 1, SIZE(i, 1)
        s = s // serialize(i(k1)) // NEW_LINE('A')
      END DO
    END IF
  END FUNCTION double_za2s
  FUNCTION double_zaa2s(i) RESULT(s)
    DOUBLE PRECISION, ALLOCATABLE, INTENT(IN) :: i(:, :)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k1, k2
    s = "# allocated" // NEW_LINE('A') // serialize(ALLOCATED(i)) // NEW_LINE('A')
    IF (ALLOCATED(i)) THEN
      s = rank_2s_(i) // "# entries" // NEW_LINE('A')
      DO k1 = 1, SIZE(i, 1)
        DO k2 = 1, SIZE(i, 2)
          s = s // serialize(i(k1, k2)) // NEW_LINE('A')
        END DO
      END DO
    END IF
  END FUNCTION double_zaa2s
  FUNCTION double_zaaa2s(i) RESULT(s)
    DOUBLE PRECISION, ALLOCATABLE, INTENT(IN) :: i(:, :, :)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k1, k2, k3
    s = "# allocated" // NEW_LINE('A') // serialize(ALLOCATED(i)) // NEW_LINE('A')
    IF (ALLOCATED(i)) THEN
      s = rank_2s_(i) // "# entries" // NEW_LINE('A')
      DO k1 = 1, SIZE(i, 1)
        DO k2 = 1, SIZE(i, 2)
          DO k3 = 1, SIZE(i, 3)
            s = s // serialize(i(k1, k2, k3)) // NEW_LINE('A')
          END DO
        END DO
      END DO
    END IF
  END FUNCTION double_zaaa2s
  FUNCTION logical_2s(l) RESULT(s)
    LOGICAL, INTENT(IN) :: l
    CHARACTER(LEN = :), ALLOCATABLE :: s
    ALLOCATE(CHARACTER(LEN = 50)::s)
    WRITE(s, *) l
    s = TRIM(s)
  END FUNCTION logical_2s
  FUNCTION logical_a2s(i) RESULT(s)
    LOGICAL, INTENT(IN) :: i(:)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k1
    s = rank_2s_(i) // "# entries" // NEW_LINE('A')
    DO k1 = 1, SIZE(i, 1)
      s = s // serialize(i(k1)) // NEW_LINE('A')
    END DO
  END FUNCTION logical_a2s
  FUNCTION logical_aa2s(i) RESULT(s)
    LOGICAL, INTENT(IN) :: i(:, :)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k1, k2
    s = rank_2s_(i) // "# entries" // NEW_LINE('A')
    DO k1 = 1, SIZE(i, 1)
      DO k2 = 1, SIZE(i, 2)
        s = s // serialize(i(k1, k2)) // NEW_LINE('A')
      END DO
    END DO
  END FUNCTION logical_aa2s
  FUNCTION logical_aaa2s(i) RESULT(s)
    LOGICAL, INTENT(IN) :: i(:, :, :)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k1, k2, k3
    s = rank_2s_(i) // "# entries" // NEW_LINE('A')
    DO k1 = 1, SIZE(i, 1)
      DO k2 = 1, SIZE(i, 2)
        DO k3 = 1, SIZE(i, 3)
          s = s // serialize(i(k1, k2, k3)) // NEW_LINE('A')
        END DO
      END DO
    END DO
  END FUNCTION logical_aaa2s
  FUNCTION logical_za2s(i) RESULT(s)
    LOGICAL, ALLOCATABLE, INTENT(IN) :: i(:)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k1
    s = "# allocated" // NEW_LINE('A') // serialize(ALLOCATED(i)) // NEW_LINE('A')
    IF (ALLOCATED(i)) THEN
      s = rank_2s_(i) // "# entries" // NEW_LINE('A')
      DO k1 = 1, SIZE(i, 1)
        s = s // serialize(i(k1)) // NEW_LINE('A')
      END DO
    END IF
  END FUNCTION logical_za2s
  FUNCTION logical_zaa2s(i) RESULT(s)
    LOGICAL, ALLOCATABLE, INTENT(IN) :: i(:, :)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k1, k2
    s = "# allocated" // NEW_LINE('A') // serialize(ALLOCATED(i)) // NEW_LINE('A')
    IF (ALLOCATED(i)) THEN
      s = rank_2s_(i) // "# entries" // NEW_LINE('A')
      DO k1 = 1, SIZE(i, 1)
        DO k2 = 1, SIZE(i, 2)
          s = s // serialize(i(k1, k2)) // NEW_LINE('A')
        END DO
      END DO
    END IF
  END FUNCTION logical_zaa2s
  FUNCTION logical_zaaa2s(i) RESULT(s)
    LOGICAL, ALLOCATABLE, INTENT(IN) :: i(:, :, :)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k1, k2, k3
    s = "# allocated" // NEW_LINE('A') // serialize(ALLOCATED(i)) // NEW_LINE('A')
    IF (ALLOCATED(i)) THEN
      s = rank_2s_(i) // "# entries" // NEW_LINE('A')
      DO k1 = 1, SIZE(i, 1)
        DO k2 = 1, SIZE(i, 2)
          DO k3 = 1, SIZE(i, 3)
            s = s // serialize(i(k1, k2, k3)) // NEW_LINE('A')
          END DO
        END DO
      END DO
    END IF
  END FUNCTION logical_zaaa2s
  FUNCTION aerosol_type_2s(x) RESULT(s)
    USE radiation_aerosol, ONLY: aerosol_type
    TYPE(aerosol_type), INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    s = '# od_sw' // NEW_LINE("A") // serialize(x % od_sw) // NEW_LINE("A")
    s = s // '# ssa_sw' // NEW_LINE("A") // serialize(x % ssa_sw) // NEW_LINE("A")
    s = s // '# g_sw' // NEW_LINE("A") // serialize(x % g_sw) // NEW_LINE("A")
    s = s // '# od_lw' // NEW_LINE("A") // serialize(x % od_lw) // NEW_LINE("A")
    s = s // '# ssa_lw' // NEW_LINE("A") // serialize(x % ssa_lw) // NEW_LINE("A")
  END FUNCTION aerosol_type_2s
  FUNCTION cloud_type_2s(x) RESULT(s)
    USE radiation_cloud, ONLY: cloud_type
    TYPE(cloud_type), INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    s = '# mixing_ratio' // NEW_LINE("A") // serialize(x % mixing_ratio) // NEW_LINE("A")
    s = s // '# q_liq' // NEW_LINE("A") // serialize(x % q_liq) // NEW_LINE("A")
    s = s // '# q_ice' // NEW_LINE("A") // serialize(x % q_ice) // NEW_LINE("A")
    s = s // '# re_liq' // NEW_LINE("A") // serialize(x % re_liq) // NEW_LINE("A")
    s = s // '# re_ice' // NEW_LINE("A") // serialize(x % re_ice) // NEW_LINE("A")
    s = s // '# fraction' // NEW_LINE("A") // serialize(x % fraction) // NEW_LINE("A")
    s = s // '# fractional_std' // NEW_LINE("A") // serialize(x % fractional_std) // NEW_LINE("A")
    s = s // '# overlap_param' // NEW_LINE("A") // serialize(x % overlap_param) // NEW_LINE("A")
  END FUNCTION cloud_type_2s
  FUNCTION cloud_optics_type_2s(x) RESULT(s)
    USE radiation_cloud_optics_data, ONLY: cloud_optics_type
    TYPE(cloud_optics_type), INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    s = '# liq_coeff_lw' // NEW_LINE("A") // serialize(x % liq_coeff_lw) // NEW_LINE("A")
    s = s // '# liq_coeff_sw' // NEW_LINE("A") // serialize(x % liq_coeff_sw) // NEW_LINE("A")
    s = s // '# ice_coeff_lw' // NEW_LINE("A") // serialize(x % ice_coeff_lw) // NEW_LINE("A")
    s = s // '# ice_coeff_sw' // NEW_LINE("A") // serialize(x % ice_coeff_sw) // NEW_LINE("A")
  END FUNCTION cloud_optics_type_2s
  FUNCTION gas_type_2s(x) RESULT(s)
    USE radiation_gas, ONLY: gas_type
    TYPE(gas_type), INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    s = '# mixing_ratio' // NEW_LINE("A") // serialize(x % mixing_ratio) // NEW_LINE("A")
  END FUNCTION gas_type_2s
  FUNCTION pdf_sampler_type_2s(x) RESULT(s)
    USE radiation_pdf_sampler, ONLY: pdf_sampler_type
    TYPE(pdf_sampler_type), INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    s = '# ncdf' // NEW_LINE("A") // serialize(x % ncdf) // NEW_LINE("A")
    s = s // '# nfsd' // NEW_LINE("A") // serialize(x % nfsd) // NEW_LINE("A")
    s = s // '# fsd1' // NEW_LINE("A") // serialize(x % fsd1) // NEW_LINE("A")
    s = s // '# inv_fsd_interval' // NEW_LINE("A") // serialize(x % inv_fsd_interval) // NEW_LINE("A")
    s = s // '# val' // NEW_LINE("A") // serialize(x % val) // NEW_LINE("A")
  END FUNCTION pdf_sampler_type_2s
  FUNCTION config_type_2s(x) RESULT(s)
    USE radiation_config, ONLY: config_type
    TYPE(config_type), INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    s = '# i_emiss_from_band_lw' // NEW_LINE("A") // serialize(x % i_emiss_from_band_lw) // NEW_LINE("A")
    s = s // '# sw_albedo_weights' // NEW_LINE("A") // serialize(x % sw_albedo_weights) // NEW_LINE("A")
    s = s // '# i_band_from_g_lw' // NEW_LINE("A") // serialize(x % i_band_from_g_lw) // NEW_LINE("A")
    s = s // '# i_band_from_reordered_g_lw' // NEW_LINE("A") // serialize(x % i_band_from_reordered_g_lw) // NEW_LINE("A")
    s = s // '# i_band_from_reordered_g_sw' // NEW_LINE("A") // serialize(x % i_band_from_reordered_g_sw) // NEW_LINE("A")
    s = s // '# cloud_optics' // NEW_LINE("A") // serialize(x % cloud_optics) // NEW_LINE("A")
    s = s // '# pdf_sampler' // NEW_LINE("A") // serialize(x % pdf_sampler) // NEW_LINE("A")
  END FUNCTION config_type_2s
  FUNCTION flux_type_2s(x) RESULT(s)
    USE radiation_flux, ONLY: flux_type
    TYPE(flux_type), INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    s = '# lw_up' // NEW_LINE("A") // serialize(x % lw_up) // NEW_LINE("A")
    s = s // '# lw_dn' // NEW_LINE("A") // serialize(x % lw_dn) // NEW_LINE("A")
    s = s // '# sw_up' // NEW_LINE("A") // serialize(x % sw_up) // NEW_LINE("A")
    s = s // '# sw_dn' // NEW_LINE("A") // serialize(x % sw_dn) // NEW_LINE("A")
    s = s // '# sw_dn_direct' // NEW_LINE("A") // serialize(x % sw_dn_direct) // NEW_LINE("A")
    s = s // '# lw_up_clear' // NEW_LINE("A") // serialize(x % lw_up_clear) // NEW_LINE("A")
    s = s // '# lw_dn_clear' // NEW_LINE("A") // serialize(x % lw_dn_clear) // NEW_LINE("A")
    s = s // '# sw_up_clear' // NEW_LINE("A") // serialize(x % sw_up_clear) // NEW_LINE("A")
    s = s // '# sw_dn_clear' // NEW_LINE("A") // serialize(x % sw_dn_clear) // NEW_LINE("A")
    s = s // '# sw_dn_direct_clear' // NEW_LINE("A") // serialize(x % sw_dn_direct_clear) // NEW_LINE("A")
    s = s // '# lw_dn_surf_g' // NEW_LINE("A") // serialize(x % lw_dn_surf_g) // NEW_LINE("A")
    s = s // '# lw_dn_surf_clear_g' // NEW_LINE("A") // serialize(x % lw_dn_surf_clear_g) // NEW_LINE("A")
    s = s // '# sw_dn_diffuse_surf_g' // NEW_LINE("A") // serialize(x % sw_dn_diffuse_surf_g) // NEW_LINE("A")
    s = s // '# sw_dn_direct_surf_g' // NEW_LINE("A") // serialize(x % sw_dn_direct_surf_g) // NEW_LINE("A")
    s = s // '# sw_dn_diffuse_surf_clear_g' // NEW_LINE("A") // serialize(x % sw_dn_diffuse_surf_clear_g) // NEW_LINE("A")
    s = s // '# sw_dn_direct_surf_clear_g' // NEW_LINE("A") // serialize(x % sw_dn_direct_surf_clear_g) // NEW_LINE("A")
    s = s // '# sw_dn_surf_band' // NEW_LINE("A") // serialize(x % sw_dn_surf_band) // NEW_LINE("A")
    s = s // '# sw_dn_direct_surf_band' // NEW_LINE("A") // serialize(x % sw_dn_direct_surf_band) // NEW_LINE("A")
    s = s // '# sw_dn_surf_clear_band' // NEW_LINE("A") // serialize(x % sw_dn_surf_clear_band) // NEW_LINE("A")
    s = s // '# sw_dn_direct_surf_clear_band' // NEW_LINE("A") // serialize(x % sw_dn_direct_surf_clear_band) // NEW_LINE("A")
    s = s // '# cloud_cover_lw' // NEW_LINE("A") // serialize(x % cloud_cover_lw) // NEW_LINE("A")
    s = s // '# cloud_cover_sw' // NEW_LINE("A") // serialize(x % cloud_cover_sw) // NEW_LINE("A")
  END FUNCTION flux_type_2s
  FUNCTION thermodynamics_type_2s(x) RESULT(s)
    USE radiation_thermodynamics, ONLY: thermodynamics_type
    TYPE(thermodynamics_type), INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    s = '# pressure_hl' // NEW_LINE("A") // serialize(x % pressure_hl) // NEW_LINE("A")
    s = s // '# temperature_hl' // NEW_LINE("A") // serialize(x % temperature_hl) // NEW_LINE("A")
    s = s // '# pressure_fl' // NEW_LINE("A") // serialize(x % pressure_fl) // NEW_LINE("A")
    s = s // '# temperature_fl' // NEW_LINE("A") // serialize(x % temperature_fl) // NEW_LINE("A")
  END FUNCTION thermodynamics_type_2s
  FUNCTION randomnumberstream_2s(x) RESULT(s)
    USE random_numbers_mix, ONLY: randomnumberstream
    TYPE(randomnumberstream), INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    s = '# iused' // NEW_LINE("A") // serialize(x % iused) // NEW_LINE("A")
    s = s // '# inittest' // NEW_LINE("A") // serialize(x % inittest) // NEW_LINE("A")
    s = s // '# ix' // NEW_LINE("A") // serialize(x % ix) // NEW_LINE("A")
    s = s // '# zrm' // NEW_LINE("A") // serialize(x % zrm) // NEW_LINE("A")
  END FUNCTION randomnumberstream_2s
  FUNCTION single_level_type_2s(x) RESULT(s)
    USE radiation_single_level, ONLY: single_level_type
    TYPE(single_level_type), INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    s = '# cos_sza' // NEW_LINE("A") // serialize(x % cos_sza) // NEW_LINE("A")
    s = s // '# skin_temperature' // NEW_LINE("A") // serialize(x % skin_temperature) // NEW_LINE("A")
    s = s // '# sw_albedo' // NEW_LINE("A") // serialize(x % sw_albedo) // NEW_LINE("A")
    s = s // '# sw_albedo_direct' // NEW_LINE("A") // serialize(x % sw_albedo_direct) // NEW_LINE("A")
    s = s // '# lw_emissivity' // NEW_LINE("A") // serialize(x % lw_emissivity) // NEW_LINE("A")
    s = s // '# spectral_solar_scaling' // NEW_LINE("A") // serialize(x % spectral_solar_scaling) // NEW_LINE("A")
    s = s // '# iseed' // NEW_LINE("A") // serialize(x % iseed) // NEW_LINE("A")
  END FUNCTION single_level_type_2s
END MODULE serde