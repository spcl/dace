# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np
from dace.frontend.fortran import ast_internal_classes

from tests.fortran.fortran_test_helper import  SourceCodeBuilder
from dace.frontend.fortran.fortran_parser import create_singular_sdfg_from_string

def test_fortran_frontend_elemental_ecrad():
    sources, main = SourceCodeBuilder().add_file("""

MODULE test_interface
    IMPLICIT NONE

    CONTAINS

    ELEMENTAL SUBROUTINE delta_eddington_scat_od(od_var_486, scat_od_var_487, g_var_488)
        REAL(KIND = 8), INTENT(INOUT) :: od_var_486, scat_od_var_487, g_var_488
        REAL(KIND = 8) :: f_var_489

        f_var_489 = g_var_488 * g_var_488
        od_var_486 = od_var_486 - scat_od_var_487 * f_var_489
        scat_od_var_487 = scat_od_var_487 * (1.0D0 - f_var_489)
        g_var_488 = g_var_488 / (1.0D0 + g_var_488)
    END SUBROUTINE delta_eddington_scat_od

END MODULE

SUBROUTINE main(od_sw_liq, scat_od_sw_liq, g_sw_liq)

    USE test_interface, ONLY: delta_eddington_scat_od

    REAL(KIND = 8), DIMENSION(14) :: od_sw_liq, scat_od_sw_liq, g_sw_liq

    CALL delta_eddington_scat_od(od_sw_liq, scat_od_sw_liq, g_sw_liq)
END SUBROUTINE

""", 'main').check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main', normalize_offsets=True)
    sdfg.simplify()
    sdfg.compile()

    size = 14
    arg1 = np.full([size], 42, order="F", dtype=np.float64)
    arg2 = np.full([size], 43, order="F", dtype=np.float64)
    arg3 = np.full([size], 44, order="F", dtype=np.float64)

    for idx in range(size):

        arg1[idx] = 42 + idx
        arg2[idx] = 2 * idx
        arg3[idx] = 100 - idx

    f_var = arg3 * arg3
    res1 = arg1 - arg2 * f_var
    res2 = arg2 * (1.0 - f_var)
    res3 = arg3 / (1.0 + arg3)

    sdfg(od_sw_liq=arg1, scat_od_sw_liq=arg2, g_sw_liq=arg3)

    assert np.allclose(arg1, res1)
    assert np.allclose(arg2, res2)
    assert np.allclose(arg3, res3)

def test_fortran_frontend_elemental_ecrad_scalar():
    sources, main = SourceCodeBuilder().add_file("""

MODULE test_interface
    IMPLICIT NONE

    TYPE :: pdf_sampler_type
      INTEGER :: ncdf, nfsd
      REAL(KIND = 8) :: fsd1, inv_fsd_interval
      REAL(KIND = 8), ALLOCATABLE, DIMENSION(:, :) :: val
    END TYPE pdf_sampler_type

    CONTAINS

    ELEMENTAL SUBROUTINE sample_from_pdf(this_var_340, fsd_var_341, cdf_var_342, val_var_343)
      CLASS(pdf_sampler_type), INTENT(IN) :: this_var_340
      REAL(KIND = 8), INTENT(IN) :: fsd_var_341, cdf_var_342
      REAL(KIND = 8), INTENT(OUT) :: val_var_343
      INTEGER :: ifsd_var_344, icdf_var_345
      REAL(KIND = 8) :: wfsd_var_346, wcdf_var_347
      wcdf_var_347 = cdf_var_342 * (this_var_340 % ncdf - 1) + 1.0D0
      wfsd_var_346 = (fsd_var_341 - this_var_340 % fsd1) * this_var_340 % inv_fsd_interval + 1.0D0
      val_var_343 = (1.0D0 - wcdf_var_347) * (1.0D0 - wfsd_var_346) 
    END SUBROUTINE sample_from_pdf

END MODULE

SUBROUTINE main(fsd, cdf, val)

    USE test_interface, ONLY: sample_from_pdf, pdf_sampler_type

    REAL(KIND = 8), DIMENSION(5) :: fsd, cdf, val
    TYPE(pdf_sampler_type) :: class_object

    class_object%ncdf = 3
    class_object%fsd1 = 42
    class_object%inv_fsd_interval = 43

    CALL sample_from_pdf(class_object, fsd, cdf, val)
END SUBROUTINE

""", 'main').check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main', normalize_offsets=True)
    sdfg.simplify()
    sdfg.compile()

    size = 5
    arg1 = np.full([size], 42, order="F", dtype=np.float64)
    arg2 = np.full([size], 43, order="F", dtype=np.float64)
    arg3 = np.full([size], 44, order="F", dtype=np.float64)

    for idx in range(size):

        arg1[idx] = 42 + idx
        arg2[idx] = 2 * idx

    ncdf = 3
    fsd1 = 42
    inv_fsd_interval = 43
    wcdf = arg2 * (ncdf - 1) + 1.0
    wfsd = (arg1 - fsd1) * inv_fsd_interval + 1.0
    val = (1.0 - wcdf) * (1.0 - wfsd) 

    sdfg(fsd=arg1, cdf=arg2, val=arg3)

    assert np.allclose(val, arg3)

def test_fortran_frontend_elemental_ecrad_range():
    sources, main = SourceCodeBuilder().add_file("""

MODULE test_interface
    IMPLICIT NONE

    TYPE :: pdf_sampler_type
      INTEGER :: ncdf, nfsd
      REAL(KIND = 8) :: fsd1, inv_fsd_interval
      REAL(KIND = 8), ALLOCATABLE, DIMENSION(:, :) :: val
    END TYPE pdf_sampler_type

    CONTAINS

    ELEMENTAL SUBROUTINE sample_from_pdf(this_var_340, fsd_var_341, cdf_var_342, val_var_343)
      CLASS(pdf_sampler_type), INTENT(IN) :: this_var_340
      REAL(KIND = 8), INTENT(IN) :: fsd_var_341, cdf_var_342
      REAL(KIND = 8), INTENT(OUT) :: val_var_343
      INTEGER :: ifsd_var_344, icdf_var_345
      REAL(KIND = 8) :: wfsd_var_346, wcdf_var_347
      wcdf_var_347 = cdf_var_342 * (this_var_340 % ncdf - 1) + 1.0D0
      wfsd_var_346 = (fsd_var_341 - this_var_340 % fsd1) * this_var_340 % inv_fsd_interval + 1.0D0
      val_var_343 = (1.0D0 - wcdf_var_347) * (1.0D0 - wfsd_var_346) 
    END SUBROUTINE sample_from_pdf

END MODULE

SUBROUTINE main(iters, fsd, cdf, val)

    USE test_interface, ONLY: sample_from_pdf, pdf_sampler_type

    REAL(KIND = 8), DIMENSION(5) :: fsd, cdf, val
    TYPE(pdf_sampler_type) :: class_object
    integer, dimension(2):: iters
    integer:: iter_up, iter_low

    iter_up = iters(1)
    iter_low = iters(2)

    class_object%ncdf = 3
    class_object%fsd1 = 42
    class_object%inv_fsd_interval = 43

    CALL sample_from_pdf(class_object, fsd(iter_up - iter_low : iter_up - 1), cdf(1:iter_low), val(iter_up - iter_low: iter_up - 1))
END SUBROUTINE

""", 'main').check_with_gfortran().get()
    sdfg = create_singular_sdfg_from_string(sources, 'main', normalize_offsets=True)
    sdfg.simplify()
    sdfg.compile()

    size = 5
    arg1 = np.full([size], 42, order="F", dtype=np.float64)
    arg2 = np.full([size], 43, order="F", dtype=np.float64)
    arg3 = np.full([size], 44, order="F", dtype=np.float64)
    iters = np.full([2], 0, order="F", dtype=np.int32)
    iters = [5, 4]

    for idx in range(size):

        arg1[idx] = 42 + idx
        arg2[idx] = 2 * idx

    ncdf = 3
    fsd1 = 42
    inv_fsd_interval = 43
    wcdf = arg2 * (ncdf - 1) + 1.0
    wfsd = (arg1 - fsd1) * inv_fsd_interval + 1.0
    val = (1.0 - wcdf) * (1.0 - wfsd) 

    sdfg(iters=iters, fsd=arg1, cdf=arg2, val=arg3)

    print(val)
    print(arg3)

    assert np.allclose(val[1:4], arg3[1:4])

if __name__ == "__main__":
    #test_fortran_frontend_elemental_ecrad()
    #test_fortran_frontend_elemental_ecrad_scalar()
    test_fortran_frontend_elemental_ecrad_range()
