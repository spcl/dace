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

if __name__ == "__main__":
    test_fortran_frontend_elemental_ecrad()
