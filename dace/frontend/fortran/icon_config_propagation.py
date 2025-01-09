# Copyright 2023 ETH Zurich and the DaCe authors. All rights reserved.

import os
import sys
from pathlib import Path
from typing import List

from fparser.common.readfortran import FortranFileReader as ffr
from fparser.two.parser import ParserFactory as pf

from dace.frontend.fortran.ast_desugaring import ConstTypeInjection
from dace.frontend.fortran.config_propagation_data import deserialse_v2
from dace.frontend.fortran.fortran_parser import ParseConfig, create_fparser_ast

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from dace.frontend.fortran import fortran_parser


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


def config_injection_list(root: str = 'dace/frontend/fortran/conf_files') -> List[ConstTypeInjection]:
    cfgs = [
        (('radiation_aerosol', 'aerosol_type'), f"{root}/aerosol_obj.txt"),
        (('radiation_cloud', 'cloud_type'), f"{root}/cloud_obj.txt"),
        (('radiation_config', 'config_type'), f"{root}/config_type_obj.txt"),
        (('radiation_flux', 'flux_type'), f"{root}/flux_obj.txt"),
        (('radiation_gas', 'gas_type'), f"{root}/gas_obj.txt"),
        (('radiation_single_level', 'single_level_type'), f"{root}/single_level_obj.txt"),
        (('radiation_thermodynamics', 'thermodynamics_type'), f"{root}/thermodynamics_obj.txt"),
    ]
    injs = []
    for k, v in cfgs:
        injs.extend(deserialse_v2(Path(v).read_text(), k))
    return injs


if __name__ == "__main__":

    base_icon_path = sys.argv[1]
    icon_file = sys.argv[2]
    sdfgs_dir = sys.argv[3]
    if len(sys.argv) > 4:
        already_parsed_ast = sys.argv[4]
    else:
        already_parsed_ast = None

    base_dir_ecrad = f"{base_icon_path}/externals/ecrad"
    base_dir_icon = f"{base_icon_path}/src"
    fortran_files = find_path_recursive(base_dir_ecrad)
    inc_list = [f"{base_icon_path}/externals/ecrad/include"]

    # Construct the primary ECRad AST.
    parse_cfg = ParseConfig(
        main=Path(f"{base_icon_path}/{icon_file}"),
        sources=[Path(f) for f in fortran_files],
        entry_points=[('radiation_interface', 'radiation')],
        config_injections=config_injection_list('conf_files'),
    )
    #already_parsed_ast=None
    if already_parsed_ast is None:
        ecrad_ast = create_fparser_ast(parse_cfg)
        already_parsed_ast_bool = False
    else:
        mini_parser=pf().create(std="f2008")
        ecrad_ast = mini_parser(ffr(file_candidate=already_parsed_ast))
        already_parsed_ast_bool = True

    # ast_builder = ast_components.InternalFortranAst()
    # parser = pf().create(std="f2008")

    # # Update configuration with user changes
    # strings = read_lines_between(
    #     f"{base_icon_path}/run/exp.exclaim_ape_R2B09",
    #     "! radiation_nml: radiation scheme",
    #     "/"
    # )
    # parsed_strings = parse_assignments(strings)

    # parkind_ast = parser(ffr(file_candidate=f"{base_icon_path}/src/shared/mo_kind.f90"))
    # parkinds = ast_builder.create_ast(parkind_ast)

    # reader = ffr(file_candidate=f"{base_icon_path}/src/namelists/mo_radiation_nml.f90")
    # namelist_ast = parser(reader)
    # namelist_internal_ast = ast_builder.create_ast(namelist_ast)
    # # this creates the initial list of assignments
    # # this does not consider conditional control-flow
    # # it excludes condiditions like "if"
    # #
    # # assignments are in form of (variable, value)
    # # we try to replace instances of variable with "value"
    # # value can be almost anything
    # #
    # lister = ast_transforms.AssignmentLister(parsed_strings)

    # replacements = 1
    # step = 1
    # # we replace if conditions iteratively until no more changes
    # # are done to the source code
    # while replacements > 0:
    #     lister.reset()
    #     lister.visit(namelist_internal_ast)

    #     # Propagate assignments
    #     prop = ast_transforms.AssignmentPropagator(lister.simple_assignments)
    #     namelist_internal_ast = prop.visit(namelist_internal_ast)
    #     replacements = prop.replacements

    #     # We try to evaluate if conditions. If we can evaluate to true/false,
    #     # then we replace the if condition with the exact path
    #     if_eval = ast_transforms.IfEvaluator()
    #     namelist_internal_ast = if_eval.visit(namelist_internal_ast)
    #     replacements += if_eval.replacements
    #     print("Made " + str(replacements) + " replacements in step " + str(step) + " Prop: " + str(
    #         prop.replacements) + " If: " + str(if_eval.replacements))
    #     step += 1

    # # adding enums from radiation config
    # adiation_config_ast = parser(
    #     ffr(file_candidate=f"{base_icon_path}/src/configure_model/mo_radiation_config.f90")
    # )
    # radiation_config_internal_ast = ast_builder.create_ast(adiation_config_ast)
    # # replace long complex enum names with integers
    # enum_propagator = ast_transforms.PropagateEnums()
    # enum_propagator.visit(radiation_config_internal_ast)

    # # namelist_assignments.insert(0,("amd", "28.970"))

    # # Repeat the
    # ecrad_init_ast = parser(ffr(file_candidate=f"{base_icon_path}/src/atm_phy_nwp/mo_nwp_ecrad_init.f90"))
    # ecrad_internal_ast = ast_builder.create_ast(ecrad_init_ast)
    # # clearing acc check
    # ecrad_internal_ast.modules[0].subroutine_definitions.pop(1)
    # ecrad_internal_ast = enum_propagator.generic_visit(ecrad_internal_ast)
    # lister2 = ast_transforms.AssignmentLister(parsed_strings)
    # replacements = 1
    # step = 1
    # while replacements > 0:
    #     lister2.reset()
    #     ecrad_internal_ast = enum_propagator.generic_visit(ecrad_internal_ast)
    #     lister2.visit(ecrad_internal_ast)
    #     prop = ast_transforms.AssignmentPropagator(lister2.simple_assignments + lister.simple_assignments)
    #     ecrad_internal_ast = prop.visit(ecrad_internal_ast)
    #     replacements = prop.replacements
    #     if_eval = ast_transforms.IfEvaluator()
    #     ecrad_internal_ast = if_eval.visit(ecrad_internal_ast)
    #     replacements += if_eval.replacements
    #     print("Made " + str(replacements) + " replacements in step " + str(step) + " Prop: " + str(
    #         prop.replacements) + " If: " + str(if_eval.replacements))
    #     step += 1

    # # TODO: a couple of manual replacements
    # lister2.simple_assignments.append(
    #     (ast_internal.Data_Ref_Node(parent_ref=ast_internal.Name_Node(name="ecrad_conf"),
    #                                 part_ref=ast_internal.Name_Node(name="do_save_radiative_properties")),
    #      ast_internal.Bool_Literal_Node(value='False')))

    # # this is defined internally in the program as "false"
    # # We remove it to simplify the code
    # lister2.simple_assignments.append(
    #     (ast_internal.Name_Node(name="lhook"),
    #      ast_internal.Bool_Literal_Node(value='False')))

    # propagation_info = lister.simple_assignments + lister2.simple_assignments

    # # let's fix the propagation info for ECRAD
    # for i in propagation_info:
    #     if isinstance(i[0], ast_internal.Data_Ref_Node):
    #         i[0].parent_ref.name = i[0].parent_ref.name.replace("ecrad_conf", "config")

    # radiation_config_ast = parser(
    #     ffr(file_candidate=f"{base_icon_path}/externals/ecrad/radiation/radiation_config.F90"))

    # enum_propagator_files = [
    #     f"{base_icon_path}/src/shared/mo_kind.f90",
    #     f"{base_icon_path}/externals/ecrad/ifsaux/parkind1.F90",
    #     f"{base_icon_path}/externals/ecrad/ifsaux/ecradhook.F90"
    # ]

    cfg = fortran_parser.FindUsedFunctionsConfig(
        root='radiation',
        needed_functions=[['radiation_interface', 'radiation']],
        skip_functions=['radiation_monochromatic', 'radiation_cloudless_sw',
                        'radiation_tripleclouds_sw', 'radiation_homogeneous_sw']
    )

    # generate_propagation_info(propagation_info)

    # previous steps were used to generate the initial list of assignments for ECRAD
    # this includes user config and internal enumerations of ICON
    # the previous ASTs can be now disregarded
    # we only keep the list of assignments and propagate it to ECRAD parsing.
    print(f"{base_icon_path}/{icon_file}")
    already_parsed_ast_bool = False
    fortran_parser.create_sdfg_from_fortran_file_with_options(
        parse_cfg,
        ecrad_ast,
        sdfgs_dir=sdfgs_dir,
        subroutine_name="radiation",
        # subroutine_name="cloud_generator",
        normalize_offsets=True,
        #propagation_info=propagation_info,
        #enum_propagator_ast=radiation_config_ast,
        #enum_propagator_files=enum_propagator_files,
        used_functions_config=cfg,
        already_parsed_ast=already_parsed_ast_bool,
        config_injections=config_injection_list('conf_files')
    )
