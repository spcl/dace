# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Loading and calling compiled SDFGs built as nanobind modules.

The compiled artifact is imported through ``importlib`` (never a top-level
``import``) and registered under the ``dace.generated.*`` namespace. The
user-facing wrapper is a thin Python shell around the generated handle.
"""
import importlib.util
import os
import sys

GENERATED_NAMESPACE = 'dace.generated'


def load_nanobind_module(library_path, module_name: str):
    """Imports the nanobind module at ``library_path``.

    The spec is created under the unprefixed ``module_name`` (so the loader
    resolves the matching ``PyInit_<name>``; the file name itself is
    irrelevant), then the module is registered in ``sys.modules`` under
    ``dace.generated.<name>``. An already-loaded module is reused, since one
    module can serve many handles.
    """
    qualified = f'{GENERATED_NAMESPACE}.{module_name}'
    if qualified in sys.modules:
        return sys.modules[qualified]

    library_path = os.fspath(library_path)
    spec = importlib.util.spec_from_file_location(module_name, library_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot create an import spec for the compiled SDFG module at {library_path}.')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[qualified] = module
    return module


class NanobindCompiledSDFG:
    """Thin Python shell over a generated nanobind module (one handle per instance).

    Argument marshalling, lazy initialization, and the GIL-released call into
    ``__program_<name>`` all happen inside the generated C++ handle.
    """

    def __init__(self, sdfg, module, arg_names):
        self._sdfg = sdfg
        self._module = module
        self._arg_names = list(arg_names or [])
        self._handle = module.make_compiled_sdfg()

    @property
    def sdfg(self):
        return self._sdfg

    @property
    def module(self):
        return self._module

    def __call__(self, *args, **kwargs):
        if args:
            if len(args) > len(self._arg_names):
                raise TypeError(f'Too many positional arguments (got {len(args)}, '
                                f'expected at most {len(self._arg_names)}).')
            for name, value in zip(self._arg_names, args):
                if name in kwargs:
                    raise TypeError(f'Argument "{name}" passed both positionally and as a keyword.')
                kwargs[name] = value
        return self._handle(**kwargs)

    def initialize(self, **kwargs):
        self._handle.initialize(**kwargs)

    def finalize(self):
        self._handle.finalize()


def load_nanobind_compiled_sdfg(library_path, sdfg) -> NanobindCompiledSDFG:
    """Loads the compiled module for ``sdfg`` and mints a fresh handle."""
    module = load_nanobind_module(library_path, sdfg.name)
    return NanobindCompiledSDFG(sdfg, module, sdfg.arg_names)
