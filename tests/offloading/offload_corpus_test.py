# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The GPU pipeline over the npbench kernels, up to the compiler: ``validate`` and ``generate_code`` need no GPU.

The whole ``auto_optimize`` pipeline runs, not the pass alone, because it picks library implementations before it
offloads. Numerical agreement with numpy stays in ``tests/npbench``, behind ``-m gpu``.
"""
import importlib.util
import pathlib
import sys
from types import ModuleType

import pytest

import dace
from dace.transformation.auto.auto_optimize import auto_optimize

#: ``tests/npbench`` is a directory of test files, not a package: its modules are loaded by path.
NPBENCH_ROOT = pathlib.Path(__file__).resolve().parent.parent / 'npbench'

PROGRAM_SUFFIX = '_kernel'


def load_module(path: pathlib.Path) -> ModuleType | None:
    """Import one npbench test module from its path, or None if it will not import."""
    name = 'npbench_corpus_' + path.relative_to(NPBENCH_ROOT).with_suffix('').as_posix().replace('/', '_')
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        return None
    return module


def npbench_programs() -> dict[str, dace.frontend.python.parser.DaceProgram]:
    """Every ``@dace.program`` under ``tests/npbench``, also the ones whose own GPU test is disabled upstream."""
    found = {}
    for path in sorted(NPBENCH_ROOT.rglob('*_test.py')):
        module = load_module(path)
        if module is None:
            continue
        for attr in sorted(module.__dict__):
            if not attr.endswith(PROGRAM_SUFFIX):
                continue
            obj = module.__dict__[attr]
            if isinstance(obj, dace.frontend.python.parser.DaceProgram):
                found[f'{path.stem}-{attr}'] = obj
    return found


PROGRAMS = npbench_programs()


def test_the_corpus_is_not_empty() -> None:
    """A collection bug would otherwise turn this whole file into a silent no-op."""
    assert len(PROGRAMS) > 20, f'expected the npbench corpus, found {len(PROGRAMS)} programs'


@pytest.mark.parametrize('program', PROGRAMS.values(), ids=list(PROGRAMS))
def test_the_offloaded_kernel_validates_and_emits(program: dace.frontend.python.parser.DaceProgram) -> None:
    """The GPU pipeline leaves a graph that validates and generates code."""
    sdfg = auto_optimize(program.to_sdfg(), dace.dtypes.DeviceType.GPU)
    sdfg.validate()
    assert sdfg.generate_code(), f'{program.name} generated nothing'


if __name__ == '__main__':
    test_the_corpus_is_not_empty()
    for program in PROGRAMS.values():
        test_the_offloaded_kernel_validates_and_emits(program)
