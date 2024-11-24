# Copyright 2023 ETH Zurich and the DaCe authors. All rights reserved.

from fparser.common.readfortran import FortranStringReader
from fparser.common.readfortran import FortranFileReader as ffr

from fparser.two.parser import ParserFactory as pf
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

from dace import symbolic as sym
import sympy as sp

from fparser.two import Fortran2003 as f03
from fparser.two import Fortran2008 as f08

def find_path_recursive(base_dir):
    dirs = os.listdir(base_dir)
    fortran_files = []
    for path in dirs:
        if os.path.isdir(os.path.join(base_dir, path)):
            fortran_files.extend(find_path_recursive(os.path.join(base_dir, path)))
        if os.path.isfile(os.path.join(base_dir, path)) and (path.endswith(".F90") or path.endswith(".f90")):
            fortran_files.append(os.path.join(base_dir, path))
    return fortran_files

def read_lines_between(file_path: str, start_str: str, end_str: str) -> list[str]:
    lines_between = []
    with open(file_path, 'r') as file:
        capture = False
        for line in file:
            if start_str in line:
                capture = True
                continue
            if end_str in line:
                if capture:
                    capture = False
                    break
            if capture:
                lines_between.append(line.strip())
    return lines_between[1:]

def parse_assignments(assignments: list[str]) -> list[tuple[str, str]]:
    parsed_assignments = []
    for assignment in assignments:
        # Remove comments
        assignment = assignment.split('!')[0].strip()
        if '=' in assignment:
            a, b = assignment.split('=', 1)
            parsed_assignments.append((a.strip(), b.strip()))
    return parsed_assignments



if __name__ == "__main__":
    base_dir_ecrad = "/home/alex/icon-model/externals/ecrad"
    base_dir_icon = "/home/alex/icon-model/src"
    fortran_files = find_path_recursive(base_dir_ecrad)
    ast_builder = ast_components.InternalFortranAst()
    parser = pf().create(std="f2008")



    strings = read_lines_between("/home/alex/icon-model/run/exp.exclaim_ape_R2B09", "! radiation_nml: radiation scheme", "/")
    parsed_strings = parse_assignments(strings)

    parkind_ast= parser(ffr(file_candidate="/home/alex/icon-model/src/shared/mo_kind.f90"))
    parkinds=ast_builder.create_ast(parkind_ast)
    
    reader = ffr(file_candidate="/home/alex/icon-model/src/namelists/mo_radiation_nml.f90")
    namelist_ast = parser(reader)
    namelist_internal_ast=ast_builder.create_ast(namelist_ast)
    lister=ast_transforms.AssignmentLister(parsed_strings)

    replacements=1
    step=1
    while replacements>0:
        lister.reset()
        lister.visit(namelist_internal_ast)
        prop=ast_transforms.AssignmentPropagator(lister.simple_assignments)
        namelist_internal_ast=prop.visit(namelist_internal_ast)
        replacements=prop.replacements
        if_eval=ast_transforms.IfEvaluator()
        namelist_internal_ast=if_eval.visit(namelist_internal_ast)
        replacements+=if_eval.replacements
        print("Made "+ str(replacements) + " replacements in step " +  str(step) + " Prop: "+ str(prop.replacements) + " If: " + str(if_eval.replacements))
        step+=1

    #adding enums from radiotion config
    adiation_config_ast= parser(ffr(file_candidate="/home/alex/icon-model/src/configure_model/mo_radiation_config.f90"))
    radiation_config_internal_ast=ast_builder.create_ast(adiation_config_ast)
    enum_propagator=ast_transforms.PropagateEnums()
    enum_propagator.visit(radiation_config_internal_ast)

    #namelist_assignments.insert(0,("amd", "28.970"))
    
    ecrad_init_ast= parser(ffr(file_candidate="/home/alex/icon-model/src/atm_phy_nwp/mo_nwp_ecrad_init.f90"))
    ecrad_internal_ast=ast_builder.create_ast(ecrad_init_ast)
    #clearing acc check
    ecrad_internal_ast.modules[0].subroutine_definitions.pop(1)
    ecrad_internal_ast=enum_propagator.generic_visit(ecrad_internal_ast)
    lister2=ast_transforms.AssignmentLister(parsed_strings)
    replacements=1
    step=1
    while replacements>0:
        lister2.reset()
        ecrad_internal_ast=enum_propagator.generic_visit(ecrad_internal_ast)
        lister2.visit(ecrad_internal_ast)
        prop=ast_transforms.AssignmentPropagator(lister2.simple_assignments+lister.simple_assignments)
        ecrad_internal_ast=prop.visit(ecrad_internal_ast)
        replacements=prop.replacements
        if_eval=ast_transforms.IfEvaluator()
        ecrad_internal_ast=if_eval.visit(ecrad_internal_ast)
        replacements+=if_eval.replacements
        print("Made "+ str(replacements) + " replacements in step " +  str(step) + " Prop: "+ str(prop.replacements) + " If: " + str(if_eval.replacements))
        step+=1


    #namelist_internal_ast=IfEvaluator().visit(namelist_internal_ast)
    base_dir = "/home/alex/icon-model/externals/ecrad/"
    #base_dir = "/mnt/c/Users/AlexWork/icon_f2dace/src"
    fortran_files = find_path_recursive(base_dir)
  
    #print(fortran_files)
    inc_list = ["/home/alex/icon-model/externals/ecrad/include"]
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
        "/home/alex/icon-model/externals/ecrad/radiation/radiation_interface.F90",
        #"/mnt/c/Users/AlexWork/icon_f2dace/src/shared/mo_loopindices.f90",
        #"/mnt/c/Users/AlexWork/icon_f2dace/src/parallel_infrastructure/mo_mpi.f90",
        #"/mnt/c/Users/AlexWork/icon_f2dace/src/shared/mo_fortran_tools.f90",
        #"/mnt/c/Users/AlexWork/icon_f2dace/src/shared/mo_math_utility_solvers2.f90",
        #"/mnt/c/Users/AlexWork/icon_f2dace/src/shared/mo_math_utilities.f90",
        #"/mnt/c/Users/AlexWork/icon_f2dace/src/configure_model/mo_parallel_config.f90",
        #"/mnt/c/Users/AlexWork/icon_f2dace/src/shared/mo_exception.f90",
        include_list=inc_list,
        source_list=fortran_files,icon_sources_dir="/home/alex/icon-model/externals/ecradicon-model/external/ecrad/",
        icon_sdfgs_dir="/home/alex/fcdc/ecrad_f2dace/sdfgs",normalize_offsets=True, propagation_info=lister.simple_assignments+lister2.simple_assignments)
    

  