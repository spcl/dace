# Copyright 2023 ETH Zurich and the DaCe authors. All rights reserved.

import os
import sys
from pathlib import Path

from dace.frontend.fortran.config_propagation_data import ecrad_config_injection_list
from dace.frontend.fortran.fortran_parser import ParseConfig, create_fparser_ast, \
    create_sdfg_from_fortran_file_with_options
from dace.frontend.fortran.gen_serde import find_all_f90_files

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

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
        entry_point_module = 'radiation_mcica_lw'
        entry_point_function = 'solver_mcica_lw'

    if already_parsed_ast:
        fortran_files = find_all_f90_files(Path(already_parsed_ast))
    else:
        fortran_files = find_all_f90_files(Path(base_icon_path).joinpath('externals/ecrad'))

    # Construct the primary ECRAD AST.
    parse_cfg = ParseConfig(
        sources=[Path(f) for f in fortran_files],
        entry_points=[(entry_point_module, entry_point_function)],
        config_injections=ecrad_config_injection_list(os.path.join(DIR_PATH, 'conf_files'))
    )
    ecrad_ast = create_fparser_ast(parse_cfg)

    create_sdfg_from_fortran_file_with_options(
        parse_cfg,
        ecrad_ast,
        sdfgs_dir=sdfgs_dir,
        normalize_offsets=True,
        already_parsed_ast=False,
    )
