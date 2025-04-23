# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests all transformations using the SDFG corpus to ensure correctness."""

import dace
import os
import pytest

# Import everything from the dace/transformation folder recursively, so we can check for subclasses
import importlib
import pkgutil
import dace.transformation
from dace.transformation.pass_pipeline import Pass, Pipeline


def import_all_submodules(package):
    """
    Recursively import all submodules of a package.
    """
    for _, name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        module = importlib.import_module(name)
        if hasattr(module, "__all__"):
            for item in module.__all__:
                if item not in globals():
                    globals()[item] = getattr(module, item)


import_all_submodules(dace.transformation)


# Returns a list of all SDFG files in the sdfg_corpus directory
def get_sdfg_paths():
    file_loc = os.path.dirname(__file__)
    sdfg_dir = os.path.join(file_loc, "../sdfg_corpus")
    return [
        os.path.join(sdfg_dir, f) for f in os.listdir(sdfg_dir) if f.endswith(".sdfg")
    ]


# Returns a list of all transformations
def get_transformations():
    full_pass_set = Pass.subclasses_recursive()
    # If the class has any subclasses, it's probably a base class
    # and we don't want to test it
    usable_transformations = [c for c in full_pass_set if len(c.__subclasses__()) == 0]
    return usable_transformations


# Store transformation classes and SDFGs that have already failed
_failed_transformations = set()
_failed_sdfgs = set()


# Tests all transformations using the SDFG corpus
@pytest.mark.timeout(10)
@pytest.mark.parametrize("transformation_cls", get_transformations())
@pytest.mark.parametrize("sdfg_path", get_sdfg_paths())
def test_transformation(transformation_cls, sdfg_path):
    # If this transformation_cls has failed before, skip the test
    if transformation_cls in _failed_transformations:
        pytest.skip(f"Skipping because {transformation_cls.__name__} already failed")
    # If this SDFG has failed before, skip the test
    if sdfg_path in _failed_sdfgs:
        pytest.skip(f"Skipping because {sdfg_path} already failed")

    # Filter large SDFGs
    if os.path.getsize(sdfg_path) > 10 * 1024:
        # The SDFG is too large, so we skip the test
        _failed_sdfgs.add(sdfg_path)
        pytest.skip(f"SDFG {sdfg_path} is too large")

    # First check if the provided SDFG is valid
    try:
        orig_sdfg = dace.SDFG.from_file(sdfg_path)
        orig_sdfg.validate()
        orig_sdfg.compile()
    except Exception as e:
        # The SDFG in the corpus is not valid, so we skip the test
        # Check the origin of the test
        _failed_sdfgs.add(sdfg_path)
        pytest.skip(f"SDFG {sdfg_path} is invalid: {e}")

    try:
        # Apply the transformation
        sdfg = dace.SDFG.from_file(sdfg_path)
        if issubclass(transformation_cls, dace.transformation.PatternTransformation):
            sdfg.apply_transformations_repeated(transformation_cls)
        else:
            pipe = Pipeline([transformation_cls()])  # To resolve dependencies
            pipe.apply_pass(sdfg, {})

        # Check if the transformed SDFG is valid
        sdfg.validate()
        sdfg.compile()

        # TODO: Check numerical correctness

    except Exception as e:
        # The transformation failed, so we add it to the failed transformations set
        _failed_transformations.add(transformation_cls)
        pytest.fail(f"Transformation {transformation_cls.__name__} failed: {e}")


if __name__ == "__main__":
    for transformation_cls in get_transformations():
        for sdfg_path in get_sdfg_paths():
            test_transformation(transformation_cls, sdfg_path)
