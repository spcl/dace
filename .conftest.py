# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""Run pytest with this file renamed as conftest.py to generate a corpus of all SDFGs used in the tests."""

import dace
import os
import pytest

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

# Tracker for created SDFGs
created_sdfgs = []

_original_init = dace.SDFG.__init__
def _tracked_init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    created_sdfgs.append(self)
dace.SDFG.__init__ = _tracked_init

# Prevent transformations from being applied
def _no_transformation(self, *args, **kwargs):
    return 0
def _cannot_apply(self, *args, **kwargs):
    return False
def _no_transformation_cls(*args, **kwargs):
    return 0
def _cannot_apply_cls(*args, **kwargs):
    return False
dace.SDFG.apply_transformations = _no_transformation
dace.SDFG.apply_transformations_repeated = _no_transformation
dace.SDFG.apply_transformations_once_everywhere = _no_transformation

subclasses = set([dace.transformation.PatternTransformation])
while True:
    prev_size = len(subclasses)
    for subclass in list(subclasses):
        subclasses.update(subclass.__subclasses__())
    if len(subclasses) == prev_size:
        break
for subclass in subclasses:
    subclass.apply = _no_transformation
    subclass.can_be_applied = _cannot_apply
    subclass.apply_to = _no_transformation_cls
    subclass.can_be_applied_to = _cannot_apply_cls

# Prevent passes from being applied
def _no_pass(self, *args, **kwargs):
    return
subclasses = set([dace.transformation.Pass])
while True:
    prev_size = len(subclasses)
    for subclass in list(subclasses):
        subclasses.update(subclass.__subclasses__())
    if len(subclasses) == prev_size:
        break
for subclass in subclasses:
    subclass.apply_pass = _no_pass


# Saves an SDFG to the corpus
def save_sdfg(sdfg: dace.SDFG):
    # Save sdfg in ./corpus using the name of the SDFG
    # if the name already exists, try out increasing _<number> until it is unique
    path = f"{os.path.dirname(__file__)}/corpus"
    os.makedirs(path, exist_ok=True)
    svname = sdfg.name
    i = 0
    while os.path.exists(os.path.join(path, svname + ".sdfg")):
        svname = sdfg.name + "_" + str(i)
        i += 1
    sdfg.save(os.path.join(path, svname + ".sdfg"))


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_teardown(item, nextitem):
    current_file = item.fspath
    next_file = getattr(nextitem, "fspath", None)
    if current_file != next_file:
      base_filename = os.path.splitext(os.path.basename(str(current_file)))[0]
      base_filename = base_filename.replace("test_", "").replace("_test", "").replace("_tests", "").replace("tests_", "")
      for sdfg in created_sdfgs:
              # Save the SDFG to the corpus
              sdfg.name = sdfg.name.replace("test_", "").replace("_test", "").replace("_tests", "").replace("tests_", "")
              if not sdfg.name.startswith(base_filename):
                  sdfg.name = base_filename + "_" + sdfg.name
              save_sdfg(sdfg)
      # Clear the list of created SDFGs
      created_sdfgs.clear()
        
