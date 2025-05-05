# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests all transformations using the SDFG corpus to ensure correctness."""

import dace
import os
import pytest
import warnings

# Import everything from the dace/transformation folder recursively, so we can check for subclasses
import importlib
import pkgutil
import dace.transformation
import dace.transformation.passes
import dace.transformation.interstate
from dace.transformation.pass_pipeline import Pass, Pipeline

from functools import lru_cache
import coverage
import json
from collections import defaultdict


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


# Tests all transformations using the SDFG corpus
@pytest.mark.timeout(10)
@pytest.mark.parametrize("transformation_cls", get_transformations())
@pytest.mark.parametrize("sdfg_path", get_sdfg_paths())
def test_transformation(transformation_cls, sdfg_path):
    # First check if the provided SDFG is valid
    try:
        orig_sdfg = dace.SDFG.from_file(sdfg_path)
        orig_sdfg.validate()
        orig_sdfg.compile()
    except Exception as e:
        # The SDFG in the corpus is not valid, so we skip the test
        # Check the origin of the test
        warnings.warn(f"SDFG {sdfg_path} is invalid: {e}")
        return

    # Apply the transformation
    sdfg = dace.SDFG.from_file(sdfg_path)
    if issubclass(transformation_cls, dace.transformation.PatternTransformation):
        # Limit to a 10 executions
        # sdfg.apply_transformations_repeated(transformation_cls)
        for i in range(10):
            app = sdfg.apply_transformations(transformation_cls)
            if app == 0:
                break
    else:
        pipe = Pipeline([transformation_cls()])  # To resolve dependencies
        pipe.apply_pass(sdfg, {})

    # Check if the transformed SDFG is valid
    sdfg.validate()
    sdfg.compile()

    # TODO: Check numerical correctness


# Loads a json coverage report and returns a dictionary of covered lines
@lru_cache(maxsize=None)
def load_coverage_lines(report_path):
    with open(report_path) as f:
        data = json.load(f)

    covered = defaultdict(set)
    for file in data.get("files", {}):
        lines = data["files"][file]["executed_lines"]
        covered[file].update(lines)

    return covered


# Computes the percentage of overlap between two coverage reports using the lower line count as the denominator
def compute_overlap(cov1, cov2):
    overlap_lines = 0
    shared_files = set(cov1.keys()) & set(cov2.keys())

    for file in shared_files:
        lines1 = cov1[file]
        lines2 = cov2[file]
        overlap_lines += len(lines1 & lines2)

    total_cov1 = sum(len(lines) for lines in cov1.values())
    total_cov2 = sum(len(lines) for lines in cov2.values())
    total_lines = min(total_cov1, total_cov2)
    if total_lines == 0:
        return 0.0
    return overlap_lines / total_lines


if __name__ == "__main__":
    # XXX: This main will try to reduce the corpus by measuring the codecoverage. It does not perform the testing itself.
    # Run the test suite with coverage
    for sdfg_path in get_sdfg_paths():
        sdfg_name = os.path.basename(sdfg_path).replace(".sdfg", "")
        cov_datafile = f"coverage/{sdfg_name}"
        os.makedirs(os.path.dirname(cov_datafile), exist_ok=True)
        cov = coverage.Coverage(
            data_file=cov_datafile, source=["dace"], data_suffix="cov"
        )
        cov.start()

        for transformation_cls in get_transformations():
            try:
                test_transformation(transformation_cls, sdfg_path)
            except Exception as e:
                continue
        cov.stop()
        cov.save()
        cov.json_report(outfile=f"coverage/{sdfg_name}.json")
        print(".", end="")
    print("")

    # Remove SDFGs with high overlap
    sdfg_paths = get_sdfg_paths()

    # Filter out SDFGs that are not in the coverage directory
    sdfg_paths = [
        path for path in sdfg_paths if os.path.exists(f"coverage/{os.path.basename(path).replace(".sdfg", "")}.json")
    ]

    removable_sdfgs = []
    changed = True

    while changed:
        changed = False
        for sdfg_path1 in sdfg_paths:
            combined = defaultdict(set)
            for sdfg_path2 in sdfg_paths:
                if sdfg_path1 == sdfg_path2:
                    continue
                sdfg_name2 = os.path.basename(sdfg_path2).replace(".sdfg", "")
                cov2 = load_coverage_lines(f"coverage/{sdfg_name2}.json")
                for file, lines in cov2.items():
                    combined[file].update(lines)

            sdfg_name1 = os.path.basename(sdfg_path1).replace(".sdfg", "")
            cov1 = load_coverage_lines(f"coverage/{sdfg_name1}.json")
            overlap = compute_overlap(cov1, combined)

            if overlap > 0.99:
                removable_sdfgs.append(sdfg_path1)

        # Sort the removable SDFGs by size and remove the largest one from sdfg_paths
        removable_sdfgs.sort(key=lambda x: os.path.getsize(x), reverse=True)
        if removable_sdfgs:
            os.remove(removable_sdfgs[0])
            sdfg_paths.remove(removable_sdfgs[0])
            removable_sdfgs = []
            changed = True
