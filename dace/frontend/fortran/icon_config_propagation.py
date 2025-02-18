# Copyright 2023 ETH Zurich and the DaCe authors. All rights reserved.

import os
import sys
from pathlib import Path
from typing import List

from fparser.common.readfortran import FortranFileReader as ffr
from fparser.two.parser import ParserFactory as pf

from dace.frontend.fortran.ast_desugaring import ConstTypeInjection
from dace.frontend.fortran.config_propagation_data import deserialize_v2
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
        injs.extend(deserialize_v2(Path(v).read_text(), k))
    return injs


if __name__ == "__main__":

    base_icon_path = sys.argv[1]
    icon_file = sys.argv[2]
    sdfgs_dir = sys.argv[3]
    if len(sys.argv) > 4:
        already_parsed_ast = sys.argv[4]
    else:
        already_parsed_ast = None

    if len(sys.argv) > 5:
        entry_point_module = sys.argv[5]
        entry_point_function = sys.argv[6]
    else:
        entry_point_module = 'radiation_interface'
        entry_point_function = 'radiation'

    base_dir_ecrad = f"{base_icon_path}/externals/ecrad"
    fortran_files = find_path_recursive(base_dir_ecrad)

    # Construct the primary ECRAD AST.
    parse_cfg = ParseConfig(
        sources=[Path(f) for f in fortran_files],
        entry_points=[(entry_point_module, entry_point_function)],
        config_injections=config_injection_list('dace/frontend/fortran/conf_files'),
    )
    if already_parsed_ast is None:
        ecrad_ast = create_fparser_ast(parse_cfg)
        already_parsed_ast_bool = False
    else:
        mini_parser = pf().create(std="f2008")
        ecrad_ast = mini_parser(ffr(file_candidate=already_parsed_ast))
        already_parsed_ast_bool = True

    already_parsed_ast_bool = False
    fortran_parser.create_sdfg_from_fortran_file_with_options(
        parse_cfg,
        ecrad_ast,
        sdfgs_dir=sdfgs_dir,
        normalize_offsets=True,
        already_parsed_ast=already_parsed_ast_bool,
    )
