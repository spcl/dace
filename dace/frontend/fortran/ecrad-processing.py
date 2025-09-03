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


if __name__ == "__main__":
    
    #sdfg=SDFG.from_file("/home/alex/fcdc/ecrad_f2dace/sdfgs/solver_mcica_lw_3321_validated_dbg.sdfgz")
    sdfg=SDFG.from_file("/home/alex/dace/dies_in_simplify6.sdfgz")
    sdfg.validate()
    #sdfg=SDFG.from_file("/home/alex/fcdc/ecrad_f2dace/sdfgs/radiation_validated_simplified.sdfgz")
    sdfg.simplify(verbose=True,validate_all=True)
    sdfg.validate()

    print("Done")
    sdfg.save("/home/alex/fcdc/ecrad_f2dace/sdfgs/radiation_validated_simplified2.sdfgz")
    print("Saved")
    sdfg.compile()