# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest
import copy
import os

from dace.transformation import *
from dace.transformation.auto import *
from dace.transformation.dataflow import *
from dace.transformation.dataflow.sve.vectorization import SVEVectorization
from dace.transformation.estimator import *
from dace.transformation.estimator.enumeration import *
from dace.transformation.interstate import *
from dace.transformation.passes import *
from dace.transformation.passes.analysis import *
from dace.transformation.passes.analysis.scope_data_and_symbol_analysis import ScopeDataAndSymbolAnalysis
from dace.transformation.passes.simplification import *
from dace.transformation.subgraph import *


def get_small_input_data(sdfg: dace.SDFG) -> dict:
        """
        Generate small input data for the given SDFG with non-trivial entries.
        """
        input_data = {}
        sym_data = {}
        for sym in sdfg.symbols:
            if sym not in sdfg.constants:
                sym_data[sym] = 3
                input_data[sym] = 3

        for argName, argType in sdfg.arglist().items():
            if isinstance(argType, dace.data.Scalar):
                input_data[argName] = 3
                continue

            shape = []
            for entry in argType.shape:
                shape.append(
                    dace.symbolic.evaluate(entry, {**sdfg.constants, **sym_data})
                )
            shape = tuple(shape)
            arr = dace.ndarray(shape=shape, dtype=argType.dtype)
            # prime number to avoid nice alignment
            arr[:] = (np.arange(arr.size).reshape(arr.shape) + 3) % 7 + 3
            input_data[argName] = arr
        return input_data


def check_transformation(orig_sdfg: dace.SDFG, xform: type):
    # Apply and validate
    llmr_sdfg = copy.deepcopy(orig_sdfg)
    apps = llmr_sdfg.apply_transformations_repeated(xform)
    llmr_sdfg.validate()

    # If no application was made, skip the rest of the test
    if apps == 0:
         return

    # Execute both SDFGs
    input_data_orig = get_small_input_data(orig_sdfg)
    input_data_llmr = copy.deepcopy(input_data_orig)
    orig_sdfg(**input_data_orig)
    llmr_sdfg(**input_data_llmr)

    # No difference should be observable
    assert (sum(not np.array_equal(input_data_orig[argName], input_data_llmr[argName])
                for argName, argType in llmr_sdfg.arglist().items()) == 0
            ), f"Output differs after transformation!"


if __name__ == "__main__":
    corpus = []
    for filename in os.listdir(os.path.dirname(__file__) + "/sdfg_corpus"):
      corpus.append(dace.SDFG.from_file(os.path.join(os.path.dirname(__file__), "sdfg_corpus", filename)))

    xform_classes = Pass.subclasses_recursive()

    for xform in xform_classes:
        for sdfg in corpus:
            try:
                check_transformation(sdfg, xform)
            except Exception as e:
                print(f"Transformation {xform.__name__} failed on SDFG {sdfg.name} with exception {e}.")
                break
