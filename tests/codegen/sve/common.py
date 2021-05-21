# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from tests.codegen.sve.vectorization import vectorize

SHOULD_EXECUTE_SVE = False


def get_code(program, sve_param):
    sdfg = program.to_sdfg()
    vectorize(sdfg, sve_param)
    return sdfg.generate_code()[0].clean_code
