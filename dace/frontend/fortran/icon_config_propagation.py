# Copyright 2023 ETH Zurich and the DaCe authors. All rights reserved.

import sys
from pathlib import Path
from typing import List

from dace.frontend.fortran.ast_desugaring import ConstTypeInjection
from dace.frontend.fortran.config_propagation_data import deserialize
from dace.frontend.fortran.fortran_parser import ParseConfig, create_fparser_ast, \
    create_sdfg_from_fortran_file_with_options
from dace.frontend.fortran.gen_serde import find_all_f90_files


def config_injection_list(root: str = 'dace/frontend/fortran/conf_files') -> List[ConstTypeInjection]:
    cfgs = [Path(root).joinpath(f).read_text() for f in [
        'config.ti', 'aerosol.ti', 'cloud.ti', 'flux.ti', 'gas.ti', 'single_level.ti', 'thermodynamics.ti']]
    injs = [deserialize(l.strip()) for c in cfgs for l in c.splitlines() if l.strip()]
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

    if already_parsed_ast:
        fortran_files = find_all_f90_files(Path(already_parsed_ast))
    else:
        fortran_files = find_all_f90_files(Path(base_icon_path).joinpath('externals/ecrad'))

    # Construct the primary ECRAD AST.
    parse_cfg = ParseConfig(
        sources=[Path(f) for f in fortran_files],
        entry_points=[(entry_point_module, entry_point_function)],
        config_injections=config_injection_list('dace/frontend/fortran/conf_files'),
    )
    ecrad_ast = create_fparser_ast(parse_cfg)

    create_sdfg_from_fortran_file_with_options(
        parse_cfg,
        ecrad_ast,
        sdfgs_dir=sdfgs_dir,
        normalize_offsets=True,
        already_parsed_ast=False,
    )
