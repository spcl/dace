# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The GPU pipeline over the npbench kernels, as far as a box without a GPU can take it.

``offload_to_accelerator_graphs_test.py`` and ``host_maps_test.py`` pin the pass's rules on graphs built to
show one rule each. This file asks the other question -- whether those rules hold on kernels nobody
wrote them for.

It runs the SAME pipeline ``tests/npbench`` runs -- ``auto_optimize`` for the GPU, which is what
calls ``apply_gpu_transformations`` -- and stops before the compiler. That last part is the point:
those tests are all marked ``gpu``, so on a machine without one the offloading gets no corpus
coverage at all, while ``validate`` plus ``generate_code`` needs neither a GPU nor a compiler and
still catches the whole placement family -- a descriptor moved to the device that host code still
reads fails ``validate`` with the container named, and a name minted into a reserved namespace
fails at code generation.

Running the whole pipeline rather than the pass alone is deliberate. ``auto_optimize`` picks library
implementations (``set_fast_implementations``) before it offloads, and a raw parsed SDFG has none:
offloading that directly leaves a host LAPACK call under a GPU schedule, which is a graph the
pipeline never actually produces. Testing it would report bugs that cannot happen.

The numerical question -- does the offloaded kernel compute what numpy computes -- stays in
``tests/npbench``, behind ``-m gpu``, because answering it needs the device.
"""
import importlib.util
import pathlib
import sys
from types import ModuleType

import pytest

import dace
from dace.transformation.auto.auto_optimize import auto_optimize

#: Kernels are reached by loading each test module by path: ``tests/npbench`` is a directory of test
#: files, not a package, so it cannot be walked as one.
NPBENCH_ROOT = pathlib.Path(__file__).resolve().parent.parent / 'npbench'

#: npbench test modules name their ``@dace.program`` ``<something>_kernel``.
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
    except Exception:  # an optional dependency missing is not a statement about offloading
        sys.modules.pop(name, None)
        return None
    return module


def gpu_skip_reason(module: ModuleType) -> str:
    """Why upstream disabled this kernel's own GPU test, or '' if it did not.

    A kernel whose ``test_gpu`` is skipped is one DaCe already knows is broken -- mandelbrot2 is
    issue #1139, lenet raises std::runtime_error. Reporting those here would be reporting somebody
    else's open bug as this pass's, so the corpus follows the same verdict and stays in step with it
    automatically when upstream re-enables one.
    """
    test = module.__dict__.get('test_gpu')
    for mark in vars(test).get('pytestmark', ()) if test is not None else ():
        if mark.name == 'skip':
            return mark.kwargs.get('reason', 'skipped upstream')
    return ''


def npbench_programs() -> list:
    """Every ``@dace.program`` reachable under ``tests/npbench``, as pytest params."""
    found = []
    for path in sorted(NPBENCH_ROOT.rglob('*_test.py')):
        module = load_module(path)
        if module is None:
            continue
        skip = gpu_skip_reason(module)
        marks = [pytest.mark.skip(reason=f'{path.stem}: {skip}')] if skip else []
        for attr in sorted(module.__dict__):
            if not attr.endswith(PROGRAM_SUFFIX):
                continue
            obj = module.__dict__[attr]
            if isinstance(obj, dace.frontend.python.parser.DaceProgram):
                found.append(pytest.param(obj, marks=marks, id=f'{path.stem}-{attr}'))
    return found


PROGRAMS = npbench_programs()


def test_the_corpus_is_not_empty() -> None:
    """A collection bug would otherwise turn this whole file into a silent no-op."""
    assert len(PROGRAMS) > 20, f'expected the npbench corpus, found {len(PROGRAMS)} programs'


@pytest.mark.parametrize('program', PROGRAMS)
def test_the_offloaded_kernel_validates_and_emits(program: dace.frontend.python.parser.DaceProgram) -> None:
    """The GPU pipeline leaves a graph that validates and generates code.

    ``validate`` is the load-bearing half: it names the container when a descriptor is moved to the
    device that host code still reads. ``generate_code`` is what catches a name minted into a
    namespace the runtime reserves.
    """
    try:
        # Parsed the way tests/npbench parses: simplification follows the configuration, so this
        # sees the same graph the pipeline does rather than a stricter one.
        sdfg = program.to_sdfg()
    except Exception as exc:  # a kernel the frontend cannot build is not a statement about offloading
        pytest.skip(f'{program.name} does not parse: {type(exc).__name__}')

    sdfg = auto_optimize(sdfg, dace.dtypes.DeviceType.GPU)
    sdfg.validate()
    assert sdfg.generate_code(), f'{program.name} generated nothing'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
