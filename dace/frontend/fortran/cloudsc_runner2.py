# Copyright 2023 ETH Zurich and the DaCe authors. All rights reserved.

from fparser.common.readfortran import FortranStringReader
from fparser.common.readfortran import FortranFileReader
from fparser.two.parser import ParserFactory
import sys, os
import numpy as np

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from dace import SDFG, SDFGState, nodes, dtypes, data, subsets, symbolic
from dace.frontend.fortran import fortran_parser
from fparser.two.symbol_table import SymbolTable

import dace.frontend.fortran.ast_components as ast_components
import dace.frontend.fortran.ast_transforms as ast_transforms
import dace.frontend.fortran.ast_utils as ast_utils
import dace.frontend.fortran.ast_internal_classes as ast_internal_classes


def find_path_recursive(base_dir):
    dirs = os.listdir(base_dir)
    fortran_files = []
    for path in dirs:
        if os.path.isdir(os.path.join(base_dir, path)):
            fortran_files.extend(find_path_recursive(os.path.join(base_dir, path)))
        if os.path.isfile(os.path.join(base_dir, path)) and (path.endswith(".F90") or path.endswith(".f90")):
            fortran_files.append(os.path.join(base_dir, path))
    return fortran_files


if __name__ == "__main__":
    base_dir = "/home/alex/ecrad/"
    #base_dir = "/mnt/c/Users/AlexWork/icon_f2dace/src"
    fortran_files = find_path_recursive(base_dir)
  
    #print(fortran_files)
    inc_list = ["/home/alex/ecrad/include"]
    #inc_list = ["/mnt/c/Users/AlexWork/icon_f2dace/src/include"]
    
    #sdfg = fortran_parser.create_sdfg_from_fortran_file_with_options(
    #    "/mnt/c/Users/AlexWork/icon_f2dace/src/shared/mo_util_texthash.f90",
    #     include_list=inc_list,
    #    source_list=fortran_files)
    fortran_parser.create_sdfg_from_fortran_file_with_options(
        #"/mnt/c/Users/AlexWork/icon_f2dace/src/atm_dyn_iconam/mo_solve_nonhydro.f90",
        #"/home/alex/ecrad/driver/ecrad_driver.F90",
        #"/home/alex/ecrad/radiation/radiation_homogeneous_lw.F90",
        #"/home/alex/ecrad/ifs/yoe_spectral_planck.F90",
        #"/home/alex/ecrad/ifs/radiation_scheme.F90",
        "/home/alex/ecrad/radiation/radiation_interface.F90",
        #"/mnt/c/Users/AlexWork/icon_f2dace/src/shared/mo_loopindices.f90",
        #"/mnt/c/Users/AlexWork/icon_f2dace/src/parallel_infrastructure/mo_mpi.f90",
        #"/mnt/c/Users/AlexWork/icon_f2dace/src/shared/mo_fortran_tools.f90",
        #"/mnt/c/Users/AlexWork/icon_f2dace/src/shared/mo_math_utility_solvers2.f90",
        #"/mnt/c/Users/AlexWork/icon_f2dace/src/shared/mo_math_utilities.f90",
        #"/mnt/c/Users/AlexWork/icon_f2dace/src/configure_model/mo_parallel_config.f90",
        #"/mnt/c/Users/AlexWork/icon_f2dace/src/shared/mo_exception.f90",
        include_list=inc_list,
        source_list=fortran_files,icon_sources_dir="/home/alex/ecrad/",
        icon_sdfgs_dir="/home/alex/fcdc/ecrad_f2dace/sdfgs",normalize_offsets=True)
    #sdfg = fortran_parser.create_sdfg_from_fortran_file_with_options(
        #"/home/alex/ecrad/ecrad-prep/driver/ecrad_driver.F90",
        #"/home/alex/ecrad/ecrad-prep/radiation/radiation_interface.F90",
        #/home/alex/ecrad/ecrad-prep/radiation/radiation_matrix.F90",
        #include_list=inc_list,
        #source_list=fortran_files)
    

    #sdfg.view()