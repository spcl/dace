# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The GPU pipeline over the npbench kernels, up to the compiler: ``validate`` and ``generate_code`` need no GPU.

The whole ``auto_optimize`` pipeline runs, not the pass alone, because it picks library implementations before it
offloads. Numerical agreement with numpy stays in ``tests/npbench``, behind ``-m gpu``.
"""

import importlib
import pathlib
from types import ModuleType

import pytest

import dace
from dace.transformation.auto.auto_optimize import auto_optimize

#: ``tests/npbench`` is a directory of test files, not a package: its modules are the namespace package
#: ``tests.npbench``, importable from the repository root.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
NPBENCH_ROOT = REPO_ROOT / "tests" / "npbench"

PROGRAM_SUFFIX = "_kernel"


def load_module(path: pathlib.Path) -> ModuleType | None:
    """Import one npbench test module, or None if it will not import."""
    try:
        return importlib.import_module(".".join(path.relative_to(REPO_ROOT).with_suffix("").parts))
    except Exception:
        return None


def npbench_programs() -> dict[str, dace.frontend.python.parser.DaceProgram]:
    """Every ``@dace.program`` under ``tests/npbench``, also the ones whose own GPU test is disabled upstream."""
    found = {}
    for path in sorted(NPBENCH_ROOT.rglob("*_test.py")):
        module = load_module(path)
        if module is None:
            continue
        for attr in sorted(module.__dict__):
            if not attr.endswith(PROGRAM_SUFFIX):
                continue
            obj = module.__dict__[attr]
            if isinstance(obj, dace.frontend.python.parser.DaceProgram):
                found[f"{path.stem}-{attr}"] = obj
    return found


PROGRAMS = npbench_programs()


def test_the_corpus_is_not_empty() -> None:
    """A collection bug would otherwise turn this whole file into a silent no-op."""
    assert len(PROGRAMS) > 20, f"expected the npbench corpus, found {len(PROGRAMS)} programs"


@pytest.mark.parametrize("program", PROGRAMS.values(), ids=list(PROGRAMS))
def test_the_offloaded_kernel_validates_and_emits(program: dace.frontend.python.parser.DaceProgram) -> None:
    """The GPU pipeline leaves a graph that validates and generates code."""
    sdfg = auto_optimize(program.to_sdfg(), dace.dtypes.DeviceType.GPU)
    sdfg.validate()
    assert sdfg.generate_code(), f"{program.name} generated nothing"


if __name__ == "__main__":
    test_the_corpus_is_not_empty()
    for program in PROGRAMS.values():
        test_the_offloaded_kernel_validates_and_emits(program)
