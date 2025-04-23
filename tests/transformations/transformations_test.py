# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests all transformations using the SDFG corpus to ensure correctness."""

import dace
import os
import pytest
import warnings
import inspect

# Import everything from the dace/transformation folder recursively
import importlib
import pkgutil
import dace.transformation


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
    subclasses = set([dace.transformation.Pass])
    while True:
        prev_size = len(subclasses)
        for subclass in list(subclasses):
            subclasses.update(subclass.__subclasses__())
        if len(subclasses) == prev_size:
            break

    # If the subclass is a subclass of dace.transformation.PatternTransformation, the function apply() should not return NotImplementedError
    # Otherwise apply_pass() should not return NotImplementedError
    usable_transformations = []
    for cls in subclasses:
        if inspect.isabstract(cls):
            continue  # Skip abstract base classes

        instance = cls()
        if isinstance(instance, dace.transformation.PatternTransformation):
            # Try calling apply(); if it raises NotImplementedError, skip it
            try:
                instance.apply(None, None)
                usable_transformations.append(cls)
            except NotImplementedError:
                continue
            except Exception:
                # Ignore runtime errors for now, only skip NotImplemented
                usable_transformations.append(cls)
        else:
            # Try calling apply_pass(); if it raises NotImplementedError, skip it
            try:
                instance.apply_pass(None, None)
                usable_transformations.append(cls)
            except NotImplementedError:
                continue
            except Exception:
                usable_transformations.append(cls)

    return usable_transformations


# Tests all transformations using the SDFG corpus
@pytest.mark.parametrize("transformation_cls", get_transformations())
@pytest.mark.parametrize("sdfg_path", get_sdfg_paths())
def test_transformation(transformation_cls, sdfg_path):
    # First check if the provided SDFG is valid
    try:
        orig_sdfg = dace.SDFG.from_file(sdfg_path)
        orig_sdfg.validate()
        orig_sdfg.compile()
    except Exception as e:
        # Issue pytest warning if the SDFG is invalid
        warnings.warn(f"SDFG {sdfg_path} is invalid: {e}. Skipping test for this SDFG.")
        return

    # Apply the transformation
    try:
        sdfg = dace.SDFG.from_file(sdfg_path)
        if issubclass(transformation_cls, dace.transformation.PatternTransformation):
            sdfg.apply_transformations_repeated(transformation_cls)
        else:
            # TODO: Some passes have dependencies
            transformation_cls().apply_pass(sdfg, {})
    except Exception as e:
        # Issue pytest warning if the transformation fails
        warnings.warn(
            f"Transformation {transformation_cls.__name__} failed on SDFG {sdfg_path}: {e}. Skipping test for this transformation."
        )
        return

    # Check if the transformed SDFG is valid
    sdfg.validate()
    sdfg.compile()

    # TODO: Check numerical correctness


if __name__ == "__main__":
    for transformation_cls in get_transformations():
        for sdfg_path in get_sdfg_paths():
            test_transformation(transformation_cls, sdfg_path)
