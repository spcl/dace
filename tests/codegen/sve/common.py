# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace.dtypes import vector
import dace
from dace.transformation.dataflow.sve.vectorization import SVEVectorization

def vectorize(program):
    sdfg = program.to_sdfg(strict=True)
    sdfg.apply_transformations(SVEVectorization)
    return sdfg

def get_code(program):
    return vectorize(program).generate_code()[0].clean_code
