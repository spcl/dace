MODULE serde
  IMPLICIT NONE
  INTERFACE serialize
    MODULE PROCEDURE :: character_2s
    MODULE PROCEDURE rng_type_2s, randomnumberstream_2s, netcdf_file_2s, aerosol_optics_type_2s, cloud_optics_type_2s, ckd_gas_type_2s, gas_type_2s, pdf_sampler_type_2s, spectral_definition_type_2s, ckd_model_type_2s, general_cloud_optics_type_2s, config_type_2s, aerosol_type_2s, flux_type_2s, single_level_type_2s, thermodynamics_type_2s, cloud_type_2s, real__8_2s1, integer_2s1, real__8_2s2, real__8_2s3, real__8_2s4, logical_2s1, dt_ckd_gas_type_2s1, character__511_2s1, dt_general_cloud_optics_type_2s1, logical_2s, integer1_2s, integer2_2s, integer4_2s, integer8_2s, real4_2s, real8_2s
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
  FUNCTION character_2s(x) RESULT(s)
    CHARACTER(LEN = *), INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    ALLOCATE(CHARACTER(LEN = LEN(x))::s)
    WRITE(s, *) x
    s = TRIM(s)
  END FUNCTION character_2s
  FUNCTION rng_type_2s(x) RESULT(s)
    USE radiation_random_numbers, ONLY: rng_type
    TYPE(rng_type), TARGET, INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: kmeta, kmeta_0, kmeta_1, kmeta_2, kmeta_3, kmeta_4, kmeta_5, kmeta_6, kmeta_7, kmeta_8, kmeta_9
    s = s // '# itype' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % itype) // NEW_LINE('A')
    s = s // '# istate' // NEW_LINE('A')
    s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
    s = s // "# size" // NEW_LINE('A')
    DO kmeta = 1, 1
      s = s // serialize(SIZE(x % istate, kmeta)) // NEW_LINE('A')
    END DO
    s = s // "# lbound" // NEW_LINE('A')
    DO kmeta = 1, 1
      s = s // serialize(LBOUND(x % istate, kmeta)) // NEW_LINE('A')
    END DO
    s = s // NEW_LINE('A') // serialize(x % istate) // NEW_LINE('A')
    s = s // '# nmaxstreams' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % nmaxstreams) // NEW_LINE('A')
    s = s // '# iseed' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % iseed) // NEW_LINE('A')
  END FUNCTION rng_type_2s
  FUNCTION randomnumberstream_2s(x) RESULT(s)
    USE random_numbers_mix, ONLY: randomnumberstream
    TYPE(randomnumberstream), TARGET, INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: kmeta, kmeta_0, kmeta_1, kmeta_2, kmeta_3, kmeta_4, kmeta_5, kmeta_6, kmeta_7, kmeta_8, kmeta_9
  END FUNCTION randomnumberstream_2s
  FUNCTION netcdf_file_2s(x) RESULT(s)
    USE easy_netcdf, ONLY: netcdf_file
    TYPE(netcdf_file), TARGET, INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: kmeta, kmeta_0, kmeta_1, kmeta_2, kmeta_3, kmeta_4, kmeta_5, kmeta_6, kmeta_7, kmeta_8, kmeta_9
    s = s // '# ncid' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % ncid) // NEW_LINE('A')
    s = s // '# iverbose' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % iverbose) // NEW_LINE('A')
    s = s // '# do_transpose_2d' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % do_transpose_2d) // NEW_LINE('A')
    s = s // '# is_write_mode' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % is_write_mode) // NEW_LINE('A')
    s = s // '# is_define_mode' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % is_define_mode) // NEW_LINE('A')
    s = s // '# is_double_precision' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % is_double_precision) // NEW_LINE('A')
    s = s // '# do_permute_3d' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % do_permute_3d) // NEW_LINE('A')
    s = s // '# do_permute_4d' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % do_permute_4d) // NEW_LINE('A')
    s = s // '# i_permute_3d' // NEW_LINE('A')
    s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
    s = s // "# size" // NEW_LINE('A')
    DO kmeta = 1, 1
      s = s // serialize(SIZE(x % i_permute_3d, kmeta)) // NEW_LINE('A')
    END DO
    s = s // "# lbound" // NEW_LINE('A')
    DO kmeta = 1, 1
      s = s // serialize(LBOUND(x % i_permute_3d, kmeta)) // NEW_LINE('A')
    END DO
    s = s // NEW_LINE('A') // serialize(x % i_permute_3d) // NEW_LINE('A')
    s = s // '# i_permute_4d' // NEW_LINE('A')
    s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
    s = s // "# size" // NEW_LINE('A')
    DO kmeta = 1, 1
      s = s // serialize(SIZE(x % i_permute_4d, kmeta)) // NEW_LINE('A')
    END DO
    s = s // "# lbound" // NEW_LINE('A')
    DO kmeta = 1, 1
      s = s // serialize(LBOUND(x % i_permute_4d, kmeta)) // NEW_LINE('A')
    END DO
    s = s // NEW_LINE('A') // serialize(x % i_permute_4d) // NEW_LINE('A')
    s = s // '# file_name' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % file_name) // NEW_LINE('A')
  END FUNCTION netcdf_file_2s
  FUNCTION aerosol_optics_type_2s(x) RESULT(s)
    USE radiation_aerosol_optics_data, ONLY: aerosol_optics_type
    TYPE(aerosol_optics_type), TARGET, INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: kmeta, kmeta_0, kmeta_1, kmeta_2, kmeta_3, kmeta_4, kmeta_5, kmeta_6, kmeta_7, kmeta_8, kmeta_9
    s = s // '# iclass' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % iclass)) // NEW_LINE('A')
    IF (ALLOCATED(x % iclass)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(SIZE(x % iclass, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(LBOUND(x % iclass, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % iclass) // NEW_LINE('A')
    END IF
    s = s // '# itype' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % itype)) // NEW_LINE('A')
    IF (ALLOCATED(x % itype)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(SIZE(x % itype, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(LBOUND(x % itype, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % itype) // NEW_LINE('A')
    END IF
    s = s // '# wavenumber1_sw' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % wavenumber1_sw)) // NEW_LINE('A')
    IF (ALLOCATED(x % wavenumber1_sw)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(SIZE(x % wavenumber1_sw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(LBOUND(x % wavenumber1_sw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % wavenumber1_sw) // NEW_LINE('A')
    END IF
    s = s // '# wavenumber2_sw' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % wavenumber2_sw)) // NEW_LINE('A')
    IF (ALLOCATED(x % wavenumber2_sw)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(SIZE(x % wavenumber2_sw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(LBOUND(x % wavenumber2_sw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % wavenumber2_sw) // NEW_LINE('A')
    END IF
    s = s // '# wavenumber1_lw' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % wavenumber1_lw)) // NEW_LINE('A')
    IF (ALLOCATED(x % wavenumber1_lw)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(SIZE(x % wavenumber1_lw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(LBOUND(x % wavenumber1_lw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % wavenumber1_lw) // NEW_LINE('A')
    END IF
    s = s // '# wavenumber2_lw' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % wavenumber2_lw)) // NEW_LINE('A')
    IF (ALLOCATED(x % wavenumber2_lw)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(SIZE(x % wavenumber2_lw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(LBOUND(x % wavenumber2_lw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % wavenumber2_lw) // NEW_LINE('A')
    END IF
    s = s // '# mass_ext_sw_phobic' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % mass_ext_sw_phobic)) // NEW_LINE('A')
    IF (ALLOCATED(x % mass_ext_sw_phobic)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % mass_ext_sw_phobic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % mass_ext_sw_phobic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % mass_ext_sw_phobic) // NEW_LINE('A')
    END IF
    s = s // '# ssa_sw_phobic' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % ssa_sw_phobic)) // NEW_LINE('A')
    IF (ALLOCATED(x % ssa_sw_phobic)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % ssa_sw_phobic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % ssa_sw_phobic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % ssa_sw_phobic) // NEW_LINE('A')
    END IF
    s = s // '# g_sw_phobic' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % g_sw_phobic)) // NEW_LINE('A')
    IF (ALLOCATED(x % g_sw_phobic)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % g_sw_phobic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % g_sw_phobic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % g_sw_phobic) // NEW_LINE('A')
    END IF
    s = s // '# mass_ext_lw_phobic' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % mass_ext_lw_phobic)) // NEW_LINE('A')
    IF (ALLOCATED(x % mass_ext_lw_phobic)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % mass_ext_lw_phobic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % mass_ext_lw_phobic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % mass_ext_lw_phobic) // NEW_LINE('A')
    END IF
    s = s // '# ssa_lw_phobic' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % ssa_lw_phobic)) // NEW_LINE('A')
    IF (ALLOCATED(x % ssa_lw_phobic)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % ssa_lw_phobic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % ssa_lw_phobic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % ssa_lw_phobic) // NEW_LINE('A')
    END IF
    s = s // '# g_lw_phobic' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % g_lw_phobic)) // NEW_LINE('A')
    IF (ALLOCATED(x % g_lw_phobic)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % g_lw_phobic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % g_lw_phobic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % g_lw_phobic) // NEW_LINE('A')
    END IF
    s = s // '# mass_ext_sw_philic' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % mass_ext_sw_philic)) // NEW_LINE('A')
    IF (ALLOCATED(x % mass_ext_sw_philic)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(3) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(SIZE(x % mass_ext_sw_philic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(LBOUND(x % mass_ext_sw_philic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % mass_ext_sw_philic) // NEW_LINE('A')
    END IF
    s = s // '# ssa_sw_philic' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % ssa_sw_philic)) // NEW_LINE('A')
    IF (ALLOCATED(x % ssa_sw_philic)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(3) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(SIZE(x % ssa_sw_philic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(LBOUND(x % ssa_sw_philic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % ssa_sw_philic) // NEW_LINE('A')
    END IF
    s = s // '# g_sw_philic' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % g_sw_philic)) // NEW_LINE('A')
    IF (ALLOCATED(x % g_sw_philic)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(3) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(SIZE(x % g_sw_philic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(LBOUND(x % g_sw_philic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % g_sw_philic) // NEW_LINE('A')
    END IF
    s = s // '# mass_ext_lw_philic' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % mass_ext_lw_philic)) // NEW_LINE('A')
    IF (ALLOCATED(x % mass_ext_lw_philic)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(3) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(SIZE(x % mass_ext_lw_philic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(LBOUND(x % mass_ext_lw_philic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % mass_ext_lw_philic) // NEW_LINE('A')
    END IF
    s = s // '# ssa_lw_philic' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % ssa_lw_philic)) // NEW_LINE('A')
    IF (ALLOCATED(x % ssa_lw_philic)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(3) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(SIZE(x % ssa_lw_philic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(LBOUND(x % ssa_lw_philic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % ssa_lw_philic) // NEW_LINE('A')
    END IF
    s = s // '# g_lw_philic' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % g_lw_philic)) // NEW_LINE('A')
    IF (ALLOCATED(x % g_lw_philic)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(3) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(SIZE(x % g_lw_philic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(LBOUND(x % g_lw_philic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % g_lw_philic) // NEW_LINE('A')
    END IF
    s = s // '# wavelength_mono' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % wavelength_mono)) // NEW_LINE('A')
    IF (ALLOCATED(x % wavelength_mono)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(SIZE(x % wavelength_mono, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(LBOUND(x % wavelength_mono, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % wavelength_mono) // NEW_LINE('A')
    END IF
    s = s // '# mass_ext_mono_phobic' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % mass_ext_mono_phobic)) // NEW_LINE('A')
    IF (ALLOCATED(x % mass_ext_mono_phobic)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % mass_ext_mono_phobic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % mass_ext_mono_phobic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % mass_ext_mono_phobic) // NEW_LINE('A')
    END IF
    s = s // '# ssa_mono_phobic' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % ssa_mono_phobic)) // NEW_LINE('A')
    IF (ALLOCATED(x % ssa_mono_phobic)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % ssa_mono_phobic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % ssa_mono_phobic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % ssa_mono_phobic) // NEW_LINE('A')
    END IF
    s = s // '# g_mono_phobic' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % g_mono_phobic)) // NEW_LINE('A')
    IF (ALLOCATED(x % g_mono_phobic)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % g_mono_phobic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % g_mono_phobic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % g_mono_phobic) // NEW_LINE('A')
    END IF
    s = s // '# lidar_ratio_mono_phobic' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % lidar_ratio_mono_phobic)) // NEW_LINE('A')
    IF (ALLOCATED(x % lidar_ratio_mono_phobic)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % lidar_ratio_mono_phobic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % lidar_ratio_mono_phobic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % lidar_ratio_mono_phobic) // NEW_LINE('A')
    END IF
    s = s // '# mass_ext_mono_philic' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % mass_ext_mono_philic)) // NEW_LINE('A')
    IF (ALLOCATED(x % mass_ext_mono_philic)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(3) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(SIZE(x % mass_ext_mono_philic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(LBOUND(x % mass_ext_mono_philic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % mass_ext_mono_philic) // NEW_LINE('A')
    END IF
    s = s // '# ssa_mono_philic' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % ssa_mono_philic)) // NEW_LINE('A')
    IF (ALLOCATED(x % ssa_mono_philic)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(3) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(SIZE(x % ssa_mono_philic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(LBOUND(x % ssa_mono_philic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % ssa_mono_philic) // NEW_LINE('A')
    END IF
    s = s // '# g_mono_philic' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % g_mono_philic)) // NEW_LINE('A')
    IF (ALLOCATED(x % g_mono_philic)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(3) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(SIZE(x % g_mono_philic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(LBOUND(x % g_mono_philic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % g_mono_philic) // NEW_LINE('A')
    END IF
    s = s // '# lidar_ratio_mono_philic' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % lidar_ratio_mono_philic)) // NEW_LINE('A')
    IF (ALLOCATED(x % lidar_ratio_mono_philic)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(3) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(SIZE(x % lidar_ratio_mono_philic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(LBOUND(x % lidar_ratio_mono_philic, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % lidar_ratio_mono_philic) // NEW_LINE('A')
    END IF
    s = s // '# rh_lower' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % rh_lower)) // NEW_LINE('A')
    IF (ALLOCATED(x % rh_lower)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(SIZE(x % rh_lower, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(LBOUND(x % rh_lower, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % rh_lower) // NEW_LINE('A')
    END IF
    s = s // '# description_phobic_str' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % description_phobic_str) // NEW_LINE('A')
    s = s // '# description_philic_str' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % description_philic_str) // NEW_LINE('A')
    s = s // '# ntype' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % ntype) // NEW_LINE('A')
    s = s // '# n_type_phobic' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % n_type_phobic) // NEW_LINE('A')
    s = s // '# n_type_philic' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % n_type_philic) // NEW_LINE('A')
    s = s // '# nrh' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % nrh) // NEW_LINE('A')
    s = s // '# n_bands_lw' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % n_bands_lw) // NEW_LINE('A')
    s = s // '# n_bands_sw' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % n_bands_sw) // NEW_LINE('A')
    s = s // '# n_mono_wl' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % n_mono_wl) // NEW_LINE('A')
    s = s // '# use_hydrophilic' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % use_hydrophilic) // NEW_LINE('A')
    s = s // '# use_monochromatic' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % use_monochromatic) // NEW_LINE('A')
  END FUNCTION aerosol_optics_type_2s
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
    s = s // '# liq_coeff_gen' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % liq_coeff_gen)) // NEW_LINE('A')
    IF (ALLOCATED(x % liq_coeff_gen)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(SIZE(x % liq_coeff_gen, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(LBOUND(x % liq_coeff_gen, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % liq_coeff_gen) // NEW_LINE('A')
    END IF
    s = s // '# ice_coeff_gen' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % ice_coeff_gen)) // NEW_LINE('A')
    IF (ALLOCATED(x % ice_coeff_gen)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(SIZE(x % ice_coeff_gen, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(LBOUND(x % ice_coeff_gen, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % ice_coeff_gen) // NEW_LINE('A')
    END IF
  END FUNCTION cloud_optics_type_2s
  FUNCTION ckd_gas_type_2s(x) RESULT(s)
    USE radiation_ecckd_gas, ONLY: ckd_gas_type
    TYPE(ckd_gas_type), TARGET, INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: kmeta, kmeta_0, kmeta_1, kmeta_2, kmeta_3, kmeta_4, kmeta_5, kmeta_6, kmeta_7, kmeta_8, kmeta_9
    s = s // '# i_gas_code' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % i_gas_code) // NEW_LINE('A')
    s = s // '# i_conc_dependence' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % i_conc_dependence) // NEW_LINE('A')
    s = s // '# molar_abs' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % molar_abs)) // NEW_LINE('A')
    IF (ALLOCATED(x % molar_abs)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(3) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(SIZE(x % molar_abs, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(LBOUND(x % molar_abs, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % molar_abs) // NEW_LINE('A')
    END IF
    s = s // '# molar_abs_conc' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % molar_abs_conc)) // NEW_LINE('A')
    IF (ALLOCATED(x % molar_abs_conc)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(4) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 4
        s = s // serialize(SIZE(x % molar_abs_conc, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 4
        s = s // serialize(LBOUND(x % molar_abs_conc, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % molar_abs_conc) // NEW_LINE('A')
    END IF
    s = s // '# reference_mole_frac' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % reference_mole_frac) // NEW_LINE('A')
    s = s // '# log_mole_frac1' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % log_mole_frac1) // NEW_LINE('A')
    s = s // '# d_log_mole_frac' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % d_log_mole_frac) // NEW_LINE('A')
    s = s // '# n_mole_frac' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % n_mole_frac) // NEW_LINE('A')
  END FUNCTION ckd_gas_type_2s
  FUNCTION gas_type_2s(x) RESULT(s)
    USE radiation_gas, ONLY: gas_type
    TYPE(gas_type), TARGET, INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: kmeta, kmeta_0, kmeta_1, kmeta_2, kmeta_3, kmeta_4, kmeta_5, kmeta_6, kmeta_7, kmeta_8, kmeta_9
    s = s // '# iunits' // NEW_LINE('A')
    s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
    s = s // "# size" // NEW_LINE('A')
    DO kmeta = 1, 1
      s = s // serialize(SIZE(x % iunits, kmeta)) // NEW_LINE('A')
    END DO
    s = s // "# lbound" // NEW_LINE('A')
    DO kmeta = 1, 1
      s = s // serialize(LBOUND(x % iunits, kmeta)) // NEW_LINE('A')
    END DO
    s = s // NEW_LINE('A') // serialize(x % iunits) // NEW_LINE('A')
    s = s // '# scale_factor' // NEW_LINE('A')
    s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
    s = s // "# size" // NEW_LINE('A')
    DO kmeta = 1, 1
      s = s // serialize(SIZE(x % scale_factor, kmeta)) // NEW_LINE('A')
    END DO
    s = s // "# lbound" // NEW_LINE('A')
    DO kmeta = 1, 1
      s = s // serialize(LBOUND(x % scale_factor, kmeta)) // NEW_LINE('A')
    END DO
    s = s // NEW_LINE('A') // serialize(x % scale_factor) // NEW_LINE('A')
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
    s = s // '# is_present' // NEW_LINE('A')
    s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
    s = s // "# size" // NEW_LINE('A')
    DO kmeta = 1, 1
      s = s // serialize(SIZE(x % is_present, kmeta)) // NEW_LINE('A')
    END DO
    s = s // "# lbound" // NEW_LINE('A')
    DO kmeta = 1, 1
      s = s // serialize(LBOUND(x % is_present, kmeta)) // NEW_LINE('A')
    END DO
    s = s // NEW_LINE('A') // serialize(x % is_present) // NEW_LINE('A')
    s = s // '# is_well_mixed' // NEW_LINE('A')
    s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
    s = s // "# size" // NEW_LINE('A')
    DO kmeta = 1, 1
      s = s // serialize(SIZE(x % is_well_mixed, kmeta)) // NEW_LINE('A')
    END DO
    s = s // "# lbound" // NEW_LINE('A')
    DO kmeta = 1, 1
      s = s // serialize(LBOUND(x % is_well_mixed, kmeta)) // NEW_LINE('A')
    END DO
    s = s // NEW_LINE('A') // serialize(x % is_well_mixed) // NEW_LINE('A')
    s = s // '# ntype' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % ntype) // NEW_LINE('A')
    s = s // '# ncol' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % ncol) // NEW_LINE('A')
    s = s // '# nlev' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % nlev) // NEW_LINE('A')
    s = s // '# icode' // NEW_LINE('A')
    s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
    s = s // "# size" // NEW_LINE('A')
    DO kmeta = 1, 1
      s = s // serialize(SIZE(x % icode, kmeta)) // NEW_LINE('A')
    END DO
    s = s // "# lbound" // NEW_LINE('A')
    DO kmeta = 1, 1
      s = s // serialize(LBOUND(x % icode, kmeta)) // NEW_LINE('A')
    END DO
    s = s // NEW_LINE('A') // serialize(x % icode) // NEW_LINE('A')
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
  FUNCTION spectral_definition_type_2s(x) RESULT(s)
    USE radiation_spectral_definition, ONLY: spectral_definition_type
    TYPE(spectral_definition_type), TARGET, INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: kmeta, kmeta_0, kmeta_1, kmeta_2, kmeta_3, kmeta_4, kmeta_5, kmeta_6, kmeta_7, kmeta_8, kmeta_9
    s = s // '# nwav' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % nwav) // NEW_LINE('A')
    s = s // '# ng' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % ng) // NEW_LINE('A')
    s = s // '# wavenumber1' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % wavenumber1)) // NEW_LINE('A')
    IF (ALLOCATED(x % wavenumber1)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(SIZE(x % wavenumber1, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(LBOUND(x % wavenumber1, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % wavenumber1) // NEW_LINE('A')
    END IF
    s = s // '# wavenumber2' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % wavenumber2)) // NEW_LINE('A')
    IF (ALLOCATED(x % wavenumber2)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(SIZE(x % wavenumber2, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(LBOUND(x % wavenumber2, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % wavenumber2) // NEW_LINE('A')
    END IF
    s = s // '# gpoint_fraction' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % gpoint_fraction)) // NEW_LINE('A')
    IF (ALLOCATED(x % gpoint_fraction)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % gpoint_fraction, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % gpoint_fraction, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % gpoint_fraction) // NEW_LINE('A')
    END IF
    s = s // '# reference_temperature' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % reference_temperature) // NEW_LINE('A')
    s = s // '# solar_spectral_irradiance' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % solar_spectral_irradiance)) // NEW_LINE('A')
    IF (ALLOCATED(x % solar_spectral_irradiance)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(SIZE(x % solar_spectral_irradiance, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(LBOUND(x % solar_spectral_irradiance, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % solar_spectral_irradiance) // NEW_LINE('A')
    END IF
    s = s // '# nband' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % nband) // NEW_LINE('A')
    s = s // '# wavenumber1_band' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % wavenumber1_band)) // NEW_LINE('A')
    IF (ALLOCATED(x % wavenumber1_band)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(SIZE(x % wavenumber1_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(LBOUND(x % wavenumber1_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % wavenumber1_band) // NEW_LINE('A')
    END IF
    s = s // '# wavenumber2_band' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % wavenumber2_band)) // NEW_LINE('A')
    IF (ALLOCATED(x % wavenumber2_band)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(SIZE(x % wavenumber2_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(LBOUND(x % wavenumber2_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % wavenumber2_band) // NEW_LINE('A')
    END IF
    s = s // '# i_band_number' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % i_band_number)) // NEW_LINE('A')
    IF (ALLOCATED(x % i_band_number)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(SIZE(x % i_band_number, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(LBOUND(x % i_band_number, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % i_band_number) // NEW_LINE('A')
    END IF
  END FUNCTION spectral_definition_type_2s
  FUNCTION ckd_model_type_2s(x) RESULT(s)
    USE radiation_ecckd, ONLY: ckd_model_type
    TYPE(ckd_model_type), TARGET, INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: kmeta, kmeta_0, kmeta_1, kmeta_2, kmeta_3, kmeta_4, kmeta_5, kmeta_6, kmeta_7, kmeta_8, kmeta_9
    s = s // '# ngas' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % ngas) // NEW_LINE('A')
    s = s // '# single_gas' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % single_gas)) // NEW_LINE('A')
    IF (ALLOCATED(x % single_gas)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(SIZE(x % single_gas, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(LBOUND(x % single_gas, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % single_gas) // NEW_LINE('A')
    END IF
    s = s // '# i_gas_mapping' // NEW_LINE('A')
    s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
    s = s // "# size" // NEW_LINE('A')
    DO kmeta = 1, 1
      s = s // serialize(SIZE(x % i_gas_mapping, kmeta)) // NEW_LINE('A')
    END DO
    s = s // "# lbound" // NEW_LINE('A')
    DO kmeta = 1, 1
      s = s // serialize(LBOUND(x % i_gas_mapping, kmeta)) // NEW_LINE('A')
    END DO
    s = s // NEW_LINE('A') // serialize(x % i_gas_mapping) // NEW_LINE('A')
    s = s // '# npress' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % npress) // NEW_LINE('A')
    s = s // '# ntemp' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % ntemp) // NEW_LINE('A')
    s = s // '# log_pressure1' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % log_pressure1) // NEW_LINE('A')
    s = s // '# d_log_pressure' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % d_log_pressure) // NEW_LINE('A')
    s = s // '# temperature1' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % temperature1)) // NEW_LINE('A')
    IF (ALLOCATED(x % temperature1)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(SIZE(x % temperature1, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(LBOUND(x % temperature1, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % temperature1) // NEW_LINE('A')
    END IF
    s = s // '# d_temperature' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % d_temperature) // NEW_LINE('A')
    s = s // '# nplanck' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % nplanck) // NEW_LINE('A')
    s = s // '# temperature1_planck' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % temperature1_planck)) // NEW_LINE('A')
    IF (ALLOCATED(x % temperature1_planck)) THEN
      s = s // NEW_LINE('A') // serialize(x % temperature1_planck) // NEW_LINE('A')
    END IF
    s = s // '# d_temperature_planck' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % d_temperature_planck)) // NEW_LINE('A')
    IF (ALLOCATED(x % d_temperature_planck)) THEN
      s = s // NEW_LINE('A') // serialize(x % d_temperature_planck) // NEW_LINE('A')
    END IF
    s = s // '# planck_function' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % planck_function)) // NEW_LINE('A')
    IF (ALLOCATED(x % planck_function)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % planck_function, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % planck_function, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % planck_function) // NEW_LINE('A')
    END IF
    s = s // '# norm_solar_irradiance' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % norm_solar_irradiance)) // NEW_LINE('A')
    IF (ALLOCATED(x % norm_solar_irradiance)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(SIZE(x % norm_solar_irradiance, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(LBOUND(x % norm_solar_irradiance, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % norm_solar_irradiance) // NEW_LINE('A')
    END IF
    s = s // '# norm_amplitude_solar_irradiance' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % norm_amplitude_solar_irradiance)) // NEW_LINE('A')
    IF (ALLOCATED(x % norm_amplitude_solar_irradiance)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(SIZE(x % norm_amplitude_solar_irradiance, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(LBOUND(x % norm_amplitude_solar_irradiance, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % norm_amplitude_solar_irradiance) // NEW_LINE('A')
    END IF
    s = s // '# rayleigh_molar_scat' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % rayleigh_molar_scat)) // NEW_LINE('A')
    IF (ALLOCATED(x % rayleigh_molar_scat)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(SIZE(x % rayleigh_molar_scat, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(LBOUND(x % rayleigh_molar_scat, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % rayleigh_molar_scat) // NEW_LINE('A')
    END IF
    s = s // '# ng' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % ng) // NEW_LINE('A')
    s = s // '# spectral_def' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % spectral_def) // NEW_LINE('A')
    s = s // '# is_sw' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % is_sw) // NEW_LINE('A')
  END FUNCTION ckd_model_type_2s
  FUNCTION general_cloud_optics_type_2s(x) RESULT(s)
    USE radiation_general_cloud_optics_data, ONLY: general_cloud_optics_type
    TYPE(general_cloud_optics_type), TARGET, INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: kmeta, kmeta_0, kmeta_1, kmeta_2, kmeta_3, kmeta_4, kmeta_5, kmeta_6, kmeta_7, kmeta_8, kmeta_9
    s = s // '# mass_ext' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % mass_ext)) // NEW_LINE('A')
    IF (ALLOCATED(x % mass_ext)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % mass_ext, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % mass_ext, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % mass_ext) // NEW_LINE('A')
    END IF
    s = s // '# ssa' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % ssa)) // NEW_LINE('A')
    IF (ALLOCATED(x % ssa)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % ssa, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % ssa, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % ssa) // NEW_LINE('A')
    END IF
    s = s // '# asymmetry' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % asymmetry)) // NEW_LINE('A')
    IF (ALLOCATED(x % asymmetry)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % asymmetry, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % asymmetry, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % asymmetry) // NEW_LINE('A')
    END IF
    s = s // '# n_effective_radius' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % n_effective_radius) // NEW_LINE('A')
    s = s // '# effective_radius_0' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % effective_radius_0) // NEW_LINE('A')
    s = s // '# d_effective_radius' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % d_effective_radius) // NEW_LINE('A')
    s = s // '# type_name' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % type_name) // NEW_LINE('A')
    s = s // '# use_bands' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % use_bands) // NEW_LINE('A')
  END FUNCTION general_cloud_optics_type_2s
  FUNCTION config_type_2s(x) RESULT(s)
    USE radiation_config, ONLY: config_type
    TYPE(config_type), TARGET, INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: kmeta, kmeta_0, kmeta_1, kmeta_2, kmeta_3, kmeta_4, kmeta_5, kmeta_6, kmeta_7, kmeta_8, kmeta_9
    s = s // '# use_spectral_solar_scaling' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % use_spectral_solar_scaling) // NEW_LINE('A')
    s = s // '# use_spectral_solar_cycle' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % use_spectral_solar_cycle) // NEW_LINE('A')
    s = s // '# directory_name' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % directory_name) // NEW_LINE('A')
    s = s // '# use_general_cloud_optics' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % use_general_cloud_optics) // NEW_LINE('A')
    s = s // '# use_general_aerosol_optics' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % use_general_aerosol_optics) // NEW_LINE('A')
    s = s // '# cloud_fraction_threshold' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % cloud_fraction_threshold) // NEW_LINE('A')
    s = s // '# cloud_mixing_ratio_threshold' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % cloud_mixing_ratio_threshold) // NEW_LINE('A')
    s = s // '# i_overlap_scheme' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % i_overlap_scheme) // NEW_LINE('A')
    s = s // '# use_beta_overlap' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % use_beta_overlap) // NEW_LINE('A')
    s = s // '# use_vectorizable_generator' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % use_vectorizable_generator) // NEW_LINE('A')
    s = s // '# i_cloud_pdf_shape' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % i_cloud_pdf_shape) // NEW_LINE('A')
    s = s // '# cloud_inhom_decorr_scaling' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % cloud_inhom_decorr_scaling) // NEW_LINE('A')
    s = s // '# clear_to_thick_fraction' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % clear_to_thick_fraction) // NEW_LINE('A')
    s = s // '# overhead_sun_factor' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % overhead_sun_factor) // NEW_LINE('A')
    s = s // '# min_gas_od_lw' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % min_gas_od_lw) // NEW_LINE('A')
    s = s // '# min_gas_od_sw' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % min_gas_od_sw) // NEW_LINE('A')
    s = s // '# max_gas_od_3d' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % max_gas_od_3d) // NEW_LINE('A')
    s = s // '# max_cloud_od' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % max_cloud_od) // NEW_LINE('A')
    s = s // '# do_lw_cloud_scattering' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % do_lw_cloud_scattering) // NEW_LINE('A')
    s = s // '# do_lw_aerosol_scattering' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % do_lw_aerosol_scattering) // NEW_LINE('A')
    s = s // '# nregions' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % nregions) // NEW_LINE('A')
    s = s // '# i_solver_sw' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % i_solver_sw) // NEW_LINE('A')
    s = s // '# i_solver_lw' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % i_solver_lw) // NEW_LINE('A')
    s = s // '# do_sw_delta_scaling_with_gases' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % do_sw_delta_scaling_with_gases) // NEW_LINE('A')
    s = s // '# i_gas_model_sw' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % i_gas_model_sw) // NEW_LINE('A')
    s = s // '# i_gas_model_lw' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % i_gas_model_lw) // NEW_LINE('A')
    s = s // '# mono_lw_wavelength' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % mono_lw_wavelength) // NEW_LINE('A')
    s = s // '# mono_lw_total_od' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % mono_lw_total_od) // NEW_LINE('A')
    s = s // '# mono_sw_total_od' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % mono_sw_total_od) // NEW_LINE('A')
    s = s // '# mono_sw_single_scattering_albedo' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % mono_sw_single_scattering_albedo) // NEW_LINE('A')
    s = s // '# mono_sw_asymmetry_factor' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % mono_sw_asymmetry_factor) // NEW_LINE('A')
    s = s // '# mono_lw_single_scattering_albedo' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % mono_lw_single_scattering_albedo) // NEW_LINE('A')
    s = s // '# mono_lw_asymmetry_factor' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % mono_lw_asymmetry_factor) // NEW_LINE('A')
    s = s // '# i_liq_model' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % i_liq_model) // NEW_LINE('A')
    s = s // '# i_ice_model' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % i_ice_model) // NEW_LINE('A')
    s = s // '# do_nearest_spectral_sw_albedo' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % do_nearest_spectral_sw_albedo) // NEW_LINE('A')
    s = s // '# do_nearest_spectral_lw_emiss' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % do_nearest_spectral_lw_emiss) // NEW_LINE('A')
    s = s // '# sw_albedo_wavelength_bound' // NEW_LINE('A')
    s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
    s = s // "# size" // NEW_LINE('A')
    DO kmeta = 1, 1
      s = s // serialize(SIZE(x % sw_albedo_wavelength_bound, kmeta)) // NEW_LINE('A')
    END DO
    s = s // "# lbound" // NEW_LINE('A')
    DO kmeta = 1, 1
      s = s // serialize(LBOUND(x % sw_albedo_wavelength_bound, kmeta)) // NEW_LINE('A')
    END DO
    s = s // NEW_LINE('A') // serialize(x % sw_albedo_wavelength_bound) // NEW_LINE('A')
    s = s // '# lw_emiss_wavelength_bound' // NEW_LINE('A')
    s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
    s = s // "# size" // NEW_LINE('A')
    DO kmeta = 1, 1
      s = s // serialize(SIZE(x % lw_emiss_wavelength_bound, kmeta)) // NEW_LINE('A')
    END DO
    s = s // "# lbound" // NEW_LINE('A')
    DO kmeta = 1, 1
      s = s // serialize(LBOUND(x % lw_emiss_wavelength_bound, kmeta)) // NEW_LINE('A')
    END DO
    s = s // NEW_LINE('A') // serialize(x % lw_emiss_wavelength_bound) // NEW_LINE('A')
    s = s // '# i_sw_albedo_index' // NEW_LINE('A')
    s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
    s = s // "# size" // NEW_LINE('A')
    DO kmeta = 1, 1
      s = s // serialize(SIZE(x % i_sw_albedo_index, kmeta)) // NEW_LINE('A')
    END DO
    s = s // "# lbound" // NEW_LINE('A')
    DO kmeta = 1, 1
      s = s // serialize(LBOUND(x % i_sw_albedo_index, kmeta)) // NEW_LINE('A')
    END DO
    s = s // NEW_LINE('A') // serialize(x % i_sw_albedo_index) // NEW_LINE('A')
    s = s // '# i_lw_emiss_index' // NEW_LINE('A')
    s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
    s = s // "# size" // NEW_LINE('A')
    DO kmeta = 1, 1
      s = s // serialize(SIZE(x % i_lw_emiss_index, kmeta)) // NEW_LINE('A')
    END DO
    s = s // "# lbound" // NEW_LINE('A')
    DO kmeta = 1, 1
      s = s // serialize(LBOUND(x % i_lw_emiss_index, kmeta)) // NEW_LINE('A')
    END DO
    s = s // NEW_LINE('A') // serialize(x % i_lw_emiss_index) // NEW_LINE('A')
    s = s // '# do_lw' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % do_lw) // NEW_LINE('A')
    s = s // '# do_sw' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % do_sw) // NEW_LINE('A')
    s = s // '# do_clear' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % do_clear) // NEW_LINE('A')
    s = s // '# do_sw_direct' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % do_sw_direct) // NEW_LINE('A')
    s = s // '# do_3d_effects' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % do_3d_effects) // NEW_LINE('A')
    s = s // '# cloud_type_name' // NEW_LINE('A')
    s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
    s = s // "# size" // NEW_LINE('A')
    DO kmeta = 1, 1
      s = s // serialize(SIZE(x % cloud_type_name, kmeta)) // NEW_LINE('A')
    END DO
    s = s // "# lbound" // NEW_LINE('A')
    DO kmeta = 1, 1
      s = s // serialize(LBOUND(x % cloud_type_name, kmeta)) // NEW_LINE('A')
    END DO
    s = s // NEW_LINE('A') // serialize(x % cloud_type_name) // NEW_LINE('A')
    s = s // '# use_thick_cloud_spectral_averaging' // NEW_LINE('A')
    s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
    s = s // "# size" // NEW_LINE('A')
    DO kmeta = 1, 1
      s = s // serialize(SIZE(x % use_thick_cloud_spectral_averaging, kmeta)) // NEW_LINE('A')
    END DO
    s = s // "# lbound" // NEW_LINE('A')
    DO kmeta = 1, 1
      s = s // serialize(LBOUND(x % use_thick_cloud_spectral_averaging, kmeta)) // NEW_LINE('A')
    END DO
    s = s // NEW_LINE('A') // serialize(x % use_thick_cloud_spectral_averaging) // NEW_LINE('A')
    s = s // '# i_3d_sw_entrapment' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % i_3d_sw_entrapment) // NEW_LINE('A')
    s = s // '# do_3d_lw_multilayer_effects' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % do_3d_lw_multilayer_effects) // NEW_LINE('A')
    s = s // '# do_lw_side_emissivity' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % do_lw_side_emissivity) // NEW_LINE('A')
    s = s // '# max_3d_transfer_rate' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % max_3d_transfer_rate) // NEW_LINE('A')
    s = s // '# min_cloud_effective_size' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % min_cloud_effective_size) // NEW_LINE('A')
    s = s // '# overhang_factor' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % overhang_factor) // NEW_LINE('A')
    s = s // '# use_expm_everywhere' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % use_expm_everywhere) // NEW_LINE('A')
    s = s // '# use_aerosols' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % use_aerosols) // NEW_LINE('A')
    s = s // '# n_aerosol_types' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % n_aerosol_types) // NEW_LINE('A')
    s = s // '# i_aerosol_type_map' // NEW_LINE('A')
    s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
    s = s // "# size" // NEW_LINE('A')
    DO kmeta = 1, 1
      s = s // serialize(SIZE(x % i_aerosol_type_map, kmeta)) // NEW_LINE('A')
    END DO
    s = s // "# lbound" // NEW_LINE('A')
    DO kmeta = 1, 1
      s = s // serialize(LBOUND(x % i_aerosol_type_map, kmeta)) // NEW_LINE('A')
    END DO
    s = s // NEW_LINE('A') // serialize(x % i_aerosol_type_map) // NEW_LINE('A')
    s = s // '# do_save_radiative_properties' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % do_save_radiative_properties) // NEW_LINE('A')
    s = s // '# do_save_spectral_flux' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % do_save_spectral_flux) // NEW_LINE('A')
    s = s // '# do_surface_sw_spectral_flux' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % do_surface_sw_spectral_flux) // NEW_LINE('A')
    s = s // '# do_toa_spectral_flux' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % do_toa_spectral_flux) // NEW_LINE('A')
    s = s // '# do_lw_derivatives' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % do_lw_derivatives) // NEW_LINE('A')
    s = s // '# do_save_gpoint_flux' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % do_save_gpoint_flux) // NEW_LINE('A')
    s = s // '# do_setup_ifsrrtm' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % do_setup_ifsrrtm) // NEW_LINE('A')
    s = s // '# do_fu_lw_ice_optics_bug' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % do_fu_lw_ice_optics_bug) // NEW_LINE('A')
    s = s // '# iverbosesetup' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % iverbosesetup) // NEW_LINE('A')
    s = s // '# iverbose' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % iverbose) // NEW_LINE('A')
    s = s // '# do_canopy_fluxes_sw' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % do_canopy_fluxes_sw) // NEW_LINE('A')
    s = s // '# do_canopy_fluxes_lw' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % do_canopy_fluxes_lw) // NEW_LINE('A')
    s = s // '# use_canopy_full_spectrum_sw' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % use_canopy_full_spectrum_sw) // NEW_LINE('A')
    s = s // '# use_canopy_full_spectrum_lw' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % use_canopy_full_spectrum_lw) // NEW_LINE('A')
    s = s // '# do_canopy_gases_sw' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % do_canopy_gases_sw) // NEW_LINE('A')
    s = s // '# do_canopy_gases_lw' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % do_canopy_gases_lw) // NEW_LINE('A')
    s = s // '# ice_optics_override_file_name' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % ice_optics_override_file_name) // NEW_LINE('A')
    s = s // '# liq_optics_override_file_name' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % liq_optics_override_file_name) // NEW_LINE('A')
    s = s // '# aerosol_optics_override_file_name' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % aerosol_optics_override_file_name) // NEW_LINE('A')
    s = s // '# gas_optics_sw_override_file_name' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % gas_optics_sw_override_file_name) // NEW_LINE('A')
    s = s // '# gas_optics_lw_override_file_name' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % gas_optics_lw_override_file_name) // NEW_LINE('A')
    s = s // '# ssi_override_file_name' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % ssi_override_file_name) // NEW_LINE('A')
    s = s // '# use_updated_solar_spectrum' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % use_updated_solar_spectrum) // NEW_LINE('A')
    s = s // '# cloud_pdf_override_file_name' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % cloud_pdf_override_file_name) // NEW_LINE('A')
    s = s // '# do_cloud_aerosol_per_sw_g_point' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % do_cloud_aerosol_per_sw_g_point) // NEW_LINE('A')
    s = s // '# do_cloud_aerosol_per_lw_g_point' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % do_cloud_aerosol_per_lw_g_point) // NEW_LINE('A')
    s = s // '# do_weighted_surface_mapping' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % do_weighted_surface_mapping) // NEW_LINE('A')
    s = s // '# is_consolidated' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % is_consolidated) // NEW_LINE('A')
    s = s // '# g_frac_sw' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % g_frac_sw)) // NEW_LINE('A')
    IF (ALLOCATED(x % g_frac_sw)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % g_frac_sw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % g_frac_sw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % g_frac_sw) // NEW_LINE('A')
    END IF
    s = s // '# g_frac_lw' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % g_frac_lw)) // NEW_LINE('A')
    IF (ALLOCATED(x % g_frac_lw)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % g_frac_lw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % g_frac_lw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % g_frac_lw) // NEW_LINE('A')
    END IF
    s = s // '# i_albedo_from_band_sw' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % i_albedo_from_band_sw)) // NEW_LINE('A')
    IF (ALLOCATED(x % i_albedo_from_band_sw)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(SIZE(x % i_albedo_from_band_sw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(LBOUND(x % i_albedo_from_band_sw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % i_albedo_from_band_sw) // NEW_LINE('A')
    END IF
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
    s = s // '# lw_emiss_weights' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % lw_emiss_weights)) // NEW_LINE('A')
    IF (ALLOCATED(x % lw_emiss_weights)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % lw_emiss_weights, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % lw_emiss_weights, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % lw_emiss_weights) // NEW_LINE('A')
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
    s = s // '# i_band_from_g_sw' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % i_band_from_g_sw)) // NEW_LINE('A')
    IF (ALLOCATED(x % i_band_from_g_sw)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(SIZE(x % i_band_from_g_sw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(LBOUND(x % i_band_from_g_sw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % i_band_from_g_sw) // NEW_LINE('A')
    END IF
    s = s // '# i_g_from_reordered_g_lw' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % i_g_from_reordered_g_lw)) // NEW_LINE('A')
    IF (ALLOCATED(x % i_g_from_reordered_g_lw)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(SIZE(x % i_g_from_reordered_g_lw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(LBOUND(x % i_g_from_reordered_g_lw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % i_g_from_reordered_g_lw) // NEW_LINE('A')
    END IF
    s = s // '# i_g_from_reordered_g_sw' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % i_g_from_reordered_g_sw)) // NEW_LINE('A')
    IF (ALLOCATED(x % i_g_from_reordered_g_sw)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(SIZE(x % i_g_from_reordered_g_sw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(LBOUND(x % i_g_from_reordered_g_sw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % i_g_from_reordered_g_sw) // NEW_LINE('A')
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
    s = s // '# i_spec_from_reordered_g_lw' // NEW_LINE('A')
    s = s // '# assoc' // NEW_LINE('A') // serialize(ASSOCIATED(x % i_spec_from_reordered_g_lw)) // NEW_LINE('A')
    IF (ASSOCIATED(x % i_spec_from_reordered_g_lw)) THEN
      kmeta = 0
      IF (ASSOCIATED(x % i_spec_from_reordered_g_lw, x % i_band_from_reordered_g_sw(:))) THEN
        kmeta = 1
        s = s // "=> x%i_band_from_reordered_g_sw("
        s = s // ':'
        s = s // "))" // NEW_LINE('A')
      END IF
      IF (kmeta == 0) THEN
        s = s // "=> missing" // NEW_LINE('A')
      END IF
    END IF
    s = s // '# i_spec_from_reordered_g_sw' // NEW_LINE('A')
    s = s // '# assoc' // NEW_LINE('A') // serialize(ASSOCIATED(x % i_spec_from_reordered_g_sw)) // NEW_LINE('A')
    IF (ASSOCIATED(x % i_spec_from_reordered_g_sw)) THEN
      kmeta = 0
      IF (ASSOCIATED(x % i_spec_from_reordered_g_sw, x % i_band_from_reordered_g_sw(:))) THEN
        kmeta = 1
        s = s // "=> x%i_band_from_reordered_g_sw("
        s = s // ':'
        s = s // "))" // NEW_LINE('A')
      END IF
      IF (kmeta == 0) THEN
        s = s // "=> missing" // NEW_LINE('A')
      END IF
    END IF
    s = s // '# n_canopy_bands_sw' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % n_canopy_bands_sw) // NEW_LINE('A')
    s = s // '# n_canopy_bands_lw' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % n_canopy_bands_lw) // NEW_LINE('A')
    s = s // '# gas_optics_sw' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % gas_optics_sw) // NEW_LINE('A')
    s = s // '# gas_optics_lw' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % gas_optics_lw) // NEW_LINE('A')
    s = s // '# cloud_optics' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % cloud_optics) // NEW_LINE('A')
    s = s // '# n_cloud_types' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % n_cloud_types) // NEW_LINE('A')
    s = s // '# cloud_optics_sw' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % cloud_optics_sw)) // NEW_LINE('A')
    IF (ALLOCATED(x % cloud_optics_sw)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(SIZE(x % cloud_optics_sw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(LBOUND(x % cloud_optics_sw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % cloud_optics_sw) // NEW_LINE('A')
    END IF
    s = s // '# cloud_optics_lw' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % cloud_optics_lw)) // NEW_LINE('A')
    IF (ALLOCATED(x % cloud_optics_lw)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(1) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(SIZE(x % cloud_optics_lw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 1
        s = s // serialize(LBOUND(x % cloud_optics_lw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % cloud_optics_lw) // NEW_LINE('A')
    END IF
    s = s // '# aerosol_optics' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % aerosol_optics) // NEW_LINE('A')
    s = s // '# pdf_sampler' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % pdf_sampler) // NEW_LINE('A')
    s = s // '# ice_optics_file_name' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % ice_optics_file_name) // NEW_LINE('A')
    s = s // '# liq_optics_file_name' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % liq_optics_file_name) // NEW_LINE('A')
    s = s // '# aerosol_optics_file_name' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % aerosol_optics_file_name) // NEW_LINE('A')
    s = s // '# gas_optics_sw_file_name' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % gas_optics_sw_file_name) // NEW_LINE('A')
    s = s // '# gas_optics_lw_file_name' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % gas_optics_lw_file_name) // NEW_LINE('A')
    s = s // '# ssi_file_name' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % ssi_file_name) // NEW_LINE('A')
    s = s // '# cloud_pdf_file_name' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % cloud_pdf_file_name) // NEW_LINE('A')
    s = s // '# n_g_sw' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % n_g_sw) // NEW_LINE('A')
    s = s // '# n_g_lw' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % n_g_lw) // NEW_LINE('A')
    s = s // '# n_bands_sw' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % n_bands_sw) // NEW_LINE('A')
    s = s // '# n_bands_lw' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % n_bands_lw) // NEW_LINE('A')
    s = s // '# n_spec_sw' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % n_spec_sw) // NEW_LINE('A')
    s = s // '# n_spec_lw' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % n_spec_lw) // NEW_LINE('A')
    s = s // '# n_wav_frac_sw' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % n_wav_frac_sw) // NEW_LINE('A')
    s = s // '# n_wav_frac_lw' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % n_wav_frac_lw) // NEW_LINE('A')
    s = s // '# n_g_lw_if_scattering' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % n_g_lw_if_scattering) // NEW_LINE('A')
    s = s // '# n_bands_lw_if_scattering' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % n_bands_lw_if_scattering) // NEW_LINE('A')
    s = s // '# is_homogeneous' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % is_homogeneous) // NEW_LINE('A')
    s = s // '# do_clouds' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % do_clouds) // NEW_LINE('A')
  END FUNCTION config_type_2s
  FUNCTION aerosol_type_2s(x) RESULT(s)
    USE radiation_aerosol, ONLY: aerosol_type
    TYPE(aerosol_type), TARGET, INTENT(IN) :: x
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
    s = s // '# g_lw' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % g_lw)) // NEW_LINE('A')
    IF (ALLOCATED(x % g_lw)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(3) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(SIZE(x % g_lw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(LBOUND(x % g_lw, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % g_lw) // NEW_LINE('A')
    END IF
    s = s // '# istartlev' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % istartlev) // NEW_LINE('A')
    s = s // '# iendlev' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % iendlev) // NEW_LINE('A')
    s = s // '# is_direct' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % is_direct) // NEW_LINE('A')
  END FUNCTION aerosol_type_2s
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
    s = s // '# lw_up_band' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % lw_up_band)) // NEW_LINE('A')
    IF (ALLOCATED(x % lw_up_band)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(3) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(SIZE(x % lw_up_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(LBOUND(x % lw_up_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % lw_up_band) // NEW_LINE('A')
    END IF
    s = s // '# lw_dn_band' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % lw_dn_band)) // NEW_LINE('A')
    IF (ALLOCATED(x % lw_dn_band)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(3) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(SIZE(x % lw_dn_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(LBOUND(x % lw_dn_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % lw_dn_band) // NEW_LINE('A')
    END IF
    s = s // '# sw_up_band' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % sw_up_band)) // NEW_LINE('A')
    IF (ALLOCATED(x % sw_up_band)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(3) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(SIZE(x % sw_up_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(LBOUND(x % sw_up_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % sw_up_band) // NEW_LINE('A')
    END IF
    s = s // '# sw_dn_band' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % sw_dn_band)) // NEW_LINE('A')
    IF (ALLOCATED(x % sw_dn_band)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(3) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(SIZE(x % sw_dn_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(LBOUND(x % sw_dn_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % sw_dn_band) // NEW_LINE('A')
    END IF
    s = s // '# sw_dn_direct_band' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % sw_dn_direct_band)) // NEW_LINE('A')
    IF (ALLOCATED(x % sw_dn_direct_band)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(3) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(SIZE(x % sw_dn_direct_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(LBOUND(x % sw_dn_direct_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % sw_dn_direct_band) // NEW_LINE('A')
    END IF
    s = s // '# lw_up_clear_band' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % lw_up_clear_band)) // NEW_LINE('A')
    IF (ALLOCATED(x % lw_up_clear_band)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(3) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(SIZE(x % lw_up_clear_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(LBOUND(x % lw_up_clear_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % lw_up_clear_band) // NEW_LINE('A')
    END IF
    s = s // '# lw_dn_clear_band' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % lw_dn_clear_band)) // NEW_LINE('A')
    IF (ALLOCATED(x % lw_dn_clear_band)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(3) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(SIZE(x % lw_dn_clear_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(LBOUND(x % lw_dn_clear_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % lw_dn_clear_band) // NEW_LINE('A')
    END IF
    s = s // '# sw_up_clear_band' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % sw_up_clear_band)) // NEW_LINE('A')
    IF (ALLOCATED(x % sw_up_clear_band)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(3) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(SIZE(x % sw_up_clear_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(LBOUND(x % sw_up_clear_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % sw_up_clear_band) // NEW_LINE('A')
    END IF
    s = s // '# sw_dn_clear_band' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % sw_dn_clear_band)) // NEW_LINE('A')
    IF (ALLOCATED(x % sw_dn_clear_band)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(3) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(SIZE(x % sw_dn_clear_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(LBOUND(x % sw_dn_clear_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % sw_dn_clear_band) // NEW_LINE('A')
    END IF
    s = s // '# sw_dn_direct_clear_band' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % sw_dn_direct_clear_band)) // NEW_LINE('A')
    IF (ALLOCATED(x % sw_dn_direct_clear_band)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(3) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(SIZE(x % sw_dn_direct_clear_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(LBOUND(x % sw_dn_direct_clear_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % sw_dn_direct_clear_band) // NEW_LINE('A')
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
    s = s // '# lw_up_toa_g' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % lw_up_toa_g)) // NEW_LINE('A')
    IF (ALLOCATED(x % lw_up_toa_g)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % lw_up_toa_g, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % lw_up_toa_g, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % lw_up_toa_g) // NEW_LINE('A')
    END IF
    s = s // '# lw_up_toa_clear_g' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % lw_up_toa_clear_g)) // NEW_LINE('A')
    IF (ALLOCATED(x % lw_up_toa_clear_g)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % lw_up_toa_clear_g, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % lw_up_toa_clear_g, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % lw_up_toa_clear_g) // NEW_LINE('A')
    END IF
    s = s // '# sw_dn_toa_g' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % sw_dn_toa_g)) // NEW_LINE('A')
    IF (ALLOCATED(x % sw_dn_toa_g)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % sw_dn_toa_g, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % sw_dn_toa_g, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % sw_dn_toa_g) // NEW_LINE('A')
    END IF
    s = s // '# sw_up_toa_g' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % sw_up_toa_g)) // NEW_LINE('A')
    IF (ALLOCATED(x % sw_up_toa_g)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % sw_up_toa_g, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % sw_up_toa_g, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % sw_up_toa_g) // NEW_LINE('A')
    END IF
    s = s // '# sw_up_toa_clear_g' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % sw_up_toa_clear_g)) // NEW_LINE('A')
    IF (ALLOCATED(x % sw_up_toa_clear_g)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % sw_up_toa_clear_g, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % sw_up_toa_clear_g, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % sw_up_toa_clear_g) // NEW_LINE('A')
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
    s = s // '# lw_up_toa_band' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % lw_up_toa_band)) // NEW_LINE('A')
    IF (ALLOCATED(x % lw_up_toa_band)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % lw_up_toa_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % lw_up_toa_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % lw_up_toa_band) // NEW_LINE('A')
    END IF
    s = s // '# lw_up_toa_clear_band' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % lw_up_toa_clear_band)) // NEW_LINE('A')
    IF (ALLOCATED(x % lw_up_toa_clear_band)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % lw_up_toa_clear_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % lw_up_toa_clear_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % lw_up_toa_clear_band) // NEW_LINE('A')
    END IF
    s = s // '# sw_dn_toa_band' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % sw_dn_toa_band)) // NEW_LINE('A')
    IF (ALLOCATED(x % sw_dn_toa_band)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % sw_dn_toa_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % sw_dn_toa_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % sw_dn_toa_band) // NEW_LINE('A')
    END IF
    s = s // '# sw_up_toa_band' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % sw_up_toa_band)) // NEW_LINE('A')
    IF (ALLOCATED(x % sw_up_toa_band)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % sw_up_toa_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % sw_up_toa_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % sw_up_toa_band) // NEW_LINE('A')
    END IF
    s = s // '# sw_up_toa_clear_band' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % sw_up_toa_clear_band)) // NEW_LINE('A')
    IF (ALLOCATED(x % sw_up_toa_clear_band)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % sw_up_toa_clear_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % sw_up_toa_clear_band, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % sw_up_toa_clear_band) // NEW_LINE('A')
    END IF
    s = s // '# lw_dn_surf_canopy' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % lw_dn_surf_canopy)) // NEW_LINE('A')
    IF (ALLOCATED(x % lw_dn_surf_canopy)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % lw_dn_surf_canopy, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % lw_dn_surf_canopy, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % lw_dn_surf_canopy) // NEW_LINE('A')
    END IF
    s = s // '# sw_dn_diffuse_surf_canopy' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % sw_dn_diffuse_surf_canopy)) // NEW_LINE('A')
    IF (ALLOCATED(x % sw_dn_diffuse_surf_canopy)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % sw_dn_diffuse_surf_canopy, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % sw_dn_diffuse_surf_canopy, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % sw_dn_diffuse_surf_canopy) // NEW_LINE('A')
    END IF
    s = s // '# sw_dn_direct_surf_canopy' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % sw_dn_direct_surf_canopy)) // NEW_LINE('A')
    IF (ALLOCATED(x % sw_dn_direct_surf_canopy)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % sw_dn_direct_surf_canopy, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % sw_dn_direct_surf_canopy, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % sw_dn_direct_surf_canopy) // NEW_LINE('A')
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
    s = s // '# lw_derivatives' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % lw_derivatives)) // NEW_LINE('A')
    IF (ALLOCATED(x % lw_derivatives)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % lw_derivatives, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % lw_derivatives, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % lw_derivatives) // NEW_LINE('A')
    END IF
  END FUNCTION flux_type_2s
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
    s = s // '# lw_emission' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % lw_emission)) // NEW_LINE('A')
    IF (ALLOCATED(x % lw_emission)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % lw_emission, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % lw_emission, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % lw_emission) // NEW_LINE('A')
    END IF
    s = s // '# solar_irradiance' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % solar_irradiance) // NEW_LINE('A')
    s = s // '# spectral_solar_cycle_multiplier' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % spectral_solar_cycle_multiplier) // NEW_LINE('A')
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
    s = s // '# is_simple_surface' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % is_simple_surface) // NEW_LINE('A')
  END FUNCTION single_level_type_2s
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
    s = s // '# h2o_sat_liq' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % h2o_sat_liq)) // NEW_LINE('A')
    IF (ALLOCATED(x % h2o_sat_liq)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % h2o_sat_liq, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % h2o_sat_liq, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % h2o_sat_liq) // NEW_LINE('A')
    END IF
    s = s // '# rrtm_pass_temppres_fl' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % rrtm_pass_temppres_fl) // NEW_LINE('A')
  END FUNCTION thermodynamics_type_2s
  FUNCTION cloud_type_2s(x) RESULT(s)
    USE radiation_cloud, ONLY: cloud_type
    TYPE(cloud_type), TARGET, INTENT(IN) :: x
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: kmeta, kmeta_0, kmeta_1, kmeta_2, kmeta_3, kmeta_4, kmeta_5, kmeta_6, kmeta_7, kmeta_8, kmeta_9
    s = s // '# ntype' // NEW_LINE('A')
    s = s // NEW_LINE('A') // serialize(x % ntype) // NEW_LINE('A')
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
    s = s // '# effective_radius' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % effective_radius)) // NEW_LINE('A')
    IF (ALLOCATED(x % effective_radius)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(3) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(SIZE(x % effective_radius, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 3
        s = s // serialize(LBOUND(x % effective_radius, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % effective_radius) // NEW_LINE('A')
    END IF
    s = s // '# q_liq' // NEW_LINE('A')
    s = s // '# assoc' // NEW_LINE('A') // serialize(ASSOCIATED(x % q_liq)) // NEW_LINE('A')
    IF (ASSOCIATED(x % q_liq)) THEN
      kmeta = 0
      DO kmeta_0 = LBOUND(x % effective_radius, 1), UBOUND(x % effective_radius, 1)
        IF (ASSOCIATED(x % q_liq, x % effective_radius(kmeta_0, :, :))) THEN
          kmeta = 1
          s = s // "=> x%effective_radius("
          s = s // serialize(kmeta_0)
          s = s // ':'
          s = s // ':'
          s = s // "))" // NEW_LINE('A')
        END IF
      END DO
      DO kmeta_1 = LBOUND(x % effective_radius, 2), UBOUND(x % effective_radius, 2)
        IF (ASSOCIATED(x % q_liq, x % effective_radius(:, kmeta_1, :))) THEN
          kmeta = 1
          s = s // "=> x%effective_radius("
          s = s // ':'
          s = s // serialize(kmeta_1)
          s = s // ':'
          s = s // "))" // NEW_LINE('A')
        END IF
      END DO
      DO kmeta_2 = LBOUND(x % effective_radius, 3), UBOUND(x % effective_radius, 3)
        IF (ASSOCIATED(x % q_liq, x % effective_radius(:, :, kmeta_2))) THEN
          kmeta = 1
          s = s // "=> x%effective_radius("
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
      DO kmeta_0 = LBOUND(x % effective_radius, 1), UBOUND(x % effective_radius, 1)
        IF (ASSOCIATED(x % q_ice, x % effective_radius(kmeta_0, :, :))) THEN
          kmeta = 1
          s = s // "=> x%effective_radius("
          s = s // serialize(kmeta_0)
          s = s // ':'
          s = s // ':'
          s = s // "))" // NEW_LINE('A')
        END IF
      END DO
      DO kmeta_1 = LBOUND(x % effective_radius, 2), UBOUND(x % effective_radius, 2)
        IF (ASSOCIATED(x % q_ice, x % effective_radius(:, kmeta_1, :))) THEN
          kmeta = 1
          s = s // "=> x%effective_radius("
          s = s // ':'
          s = s // serialize(kmeta_1)
          s = s // ':'
          s = s // "))" // NEW_LINE('A')
        END IF
      END DO
      DO kmeta_2 = LBOUND(x % effective_radius, 3), UBOUND(x % effective_radius, 3)
        IF (ASSOCIATED(x % q_ice, x % effective_radius(:, :, kmeta_2))) THEN
          kmeta = 1
          s = s // "=> x%effective_radius("
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
      DO kmeta_0 = LBOUND(x % effective_radius, 1), UBOUND(x % effective_radius, 1)
        IF (ASSOCIATED(x % re_liq, x % effective_radius(kmeta_0, :, :))) THEN
          kmeta = 1
          s = s // "=> x%effective_radius("
          s = s // serialize(kmeta_0)
          s = s // ':'
          s = s // ':'
          s = s // "))" // NEW_LINE('A')
        END IF
      END DO
      DO kmeta_1 = LBOUND(x % effective_radius, 2), UBOUND(x % effective_radius, 2)
        IF (ASSOCIATED(x % re_liq, x % effective_radius(:, kmeta_1, :))) THEN
          kmeta = 1
          s = s // "=> x%effective_radius("
          s = s // ':'
          s = s // serialize(kmeta_1)
          s = s // ':'
          s = s // "))" // NEW_LINE('A')
        END IF
      END DO
      DO kmeta_2 = LBOUND(x % effective_radius, 3), UBOUND(x % effective_radius, 3)
        IF (ASSOCIATED(x % re_liq, x % effective_radius(:, :, kmeta_2))) THEN
          kmeta = 1
          s = s // "=> x%effective_radius("
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
      DO kmeta_0 = LBOUND(x % effective_radius, 1), UBOUND(x % effective_radius, 1)
        IF (ASSOCIATED(x % re_ice, x % effective_radius(kmeta_0, :, :))) THEN
          kmeta = 1
          s = s // "=> x%effective_radius("
          s = s // serialize(kmeta_0)
          s = s // ':'
          s = s // ':'
          s = s // "))" // NEW_LINE('A')
        END IF
      END DO
      DO kmeta_1 = LBOUND(x % effective_radius, 2), UBOUND(x % effective_radius, 2)
        IF (ASSOCIATED(x % re_ice, x % effective_radius(:, kmeta_1, :))) THEN
          kmeta = 1
          s = s // "=> x%effective_radius("
          s = s // ':'
          s = s // serialize(kmeta_1)
          s = s // ':'
          s = s // "))" // NEW_LINE('A')
        END IF
      END DO
      DO kmeta_2 = LBOUND(x % effective_radius, 3), UBOUND(x % effective_radius, 3)
        IF (ASSOCIATED(x % re_ice, x % effective_radius(:, :, kmeta_2))) THEN
          kmeta = 1
          s = s // "=> x%effective_radius("
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
    s = s // '# inv_cloud_effective_size' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % inv_cloud_effective_size)) // NEW_LINE('A')
    IF (ALLOCATED(x % inv_cloud_effective_size)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % inv_cloud_effective_size, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % inv_cloud_effective_size, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % inv_cloud_effective_size) // NEW_LINE('A')
    END IF
    s = s // '# inv_inhom_effective_size' // NEW_LINE('A')
    s = s // '# alloc' // NEW_LINE('A') // serialize(ALLOCATED(x % inv_inhom_effective_size)) // NEW_LINE('A')
    IF (ALLOCATED(x % inv_inhom_effective_size)) THEN
      s = s // "# rank" // NEW_LINE('A') // serialize(2) // NEW_LINE('A')
      s = s // "# size" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(SIZE(x % inv_inhom_effective_size, kmeta)) // NEW_LINE('A')
      END DO
      s = s // "# lbound" // NEW_LINE('A')
      DO kmeta = 1, 2
        s = s // serialize(LBOUND(x % inv_inhom_effective_size, kmeta)) // NEW_LINE('A')
      END DO
      s = s // NEW_LINE('A') // serialize(x % inv_inhom_effective_size) // NEW_LINE('A')
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
  FUNCTION real__8_2s1(a) RESULT(s)
    REAL(KIND = 8), INTENT(IN) :: a(:)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k, k1
    s = "# entries" // NEW_LINE('A')
    DO k1 = LBOUND(a, 1), UBOUND(a, 1)
      s = s // serialize(a(k1)) // NEW_LINE('A')
    END DO
  END FUNCTION real__8_2s1
  FUNCTION integer_2s1(a) RESULT(s)
    INTEGER, INTENT(IN) :: a(:)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k, k1
    s = "# entries" // NEW_LINE('A')
    DO k1 = LBOUND(a, 1), UBOUND(a, 1)
      s = s // serialize(a(k1)) // NEW_LINE('A')
    END DO
  END FUNCTION integer_2s1
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
  FUNCTION real__8_2s4(a) RESULT(s)
    REAL(KIND = 8), INTENT(IN) :: a(:, :, :, :)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k, k1, k2, k3, k4
    s = "# entries" // NEW_LINE('A')
    DO k1 = LBOUND(a, 1), UBOUND(a, 1)
      DO k2 = LBOUND(a, 2), UBOUND(a, 2)
        DO k3 = LBOUND(a, 3), UBOUND(a, 3)
          DO k4 = LBOUND(a, 4), UBOUND(a, 4)
            s = s // serialize(a(k1, k2, k3, k4)) // NEW_LINE('A')
          END DO
        END DO
      END DO
    END DO
  END FUNCTION real__8_2s4
  FUNCTION logical_2s1(a) RESULT(s)
    LOGICAL, INTENT(IN) :: a(:)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k, k1
    s = "# entries" // NEW_LINE('A')
    DO k1 = LBOUND(a, 1), UBOUND(a, 1)
      s = s // serialize(a(k1)) // NEW_LINE('A')
    END DO
  END FUNCTION logical_2s1
  FUNCTION dt_ckd_gas_type_2s1(a) RESULT(s)
    USE radiation_ecckd_gas, ONLY: ckd_gas_type
    TYPE(ckd_gas_type), INTENT(IN) :: a(:)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k, k1
    s = "# entries" // NEW_LINE('A')
    DO k1 = LBOUND(a, 1), UBOUND(a, 1)
      s = s // serialize(a(k1)) // NEW_LINE('A')
    END DO
  END FUNCTION dt_ckd_gas_type_2s1
  FUNCTION character__511_2s1(a) RESULT(s)
    CHARACTER(LEN = 511), INTENT(IN) :: a(:)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k, k1
    s = "# entries" // NEW_LINE('A')
    DO k1 = LBOUND(a, 1), UBOUND(a, 1)
      s = s // serialize(a(k1)) // NEW_LINE('A')
    END DO
  END FUNCTION character__511_2s1
  FUNCTION dt_general_cloud_optics_type_2s1(a) RESULT(s)
    USE radiation_general_cloud_optics_data, ONLY: general_cloud_optics_type
    TYPE(general_cloud_optics_type), INTENT(IN) :: a(:)
    CHARACTER(LEN = :), ALLOCATABLE :: s
    INTEGER :: k, k1
    s = "# entries" // NEW_LINE('A')
    DO k1 = LBOUND(a, 1), UBOUND(a, 1)
      s = s // serialize(a(k1)) // NEW_LINE('A')
    END DO
  END FUNCTION dt_general_cloud_optics_type_2s1
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