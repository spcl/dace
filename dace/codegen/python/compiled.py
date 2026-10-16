# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Load and call the module the Python backend writes."""

import importlib.util
import pathlib
from typing import Any

import numpy as np

from dace import SDFG, symbolic


class PythonCompiledSDFG:
    """A callable over the function of a generated module, with the calling convention of ``CompiledSDFG``.

    Arguments bind by keyword or in ``argnames`` order. The ``__return`` arrays are allocated per call unless passed,
    and returned: one array alone, several as a tuple; a scalar is a one-element array.
    """

    __slots__ = ("sdfg", "path", "function", "argnames")

    def __init__(self, sdfg: SDFG, path: pathlib.Path):
        self.sdfg = sdfg
        self.path = pathlib.Path(path)
        spec = importlib.util.spec_from_file_location(f"dace_python_{sdfg.name}", self.path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.function = getattr(module, sdfg.name)
        self.argnames = [name for name in sdfg.arglist() if not name.startswith("__return")]

    def return_names(self) -> list[str]:
        if "__return" in self.sdfg.arrays:
            return ["__return"]
        count = sum(1 for name in self.sdfg.arrays if name.startswith("__return_"))
        return [f"__return_{i}" for i in range(count)]

    def __call__(self, *args, **kwargs) -> Any:
        if len(args) > len(self.argnames):
            raise TypeError(f"{self.sdfg.name} takes {len(self.argnames)} arguments, got {len(args)}")
        kwargs = dict(zip(self.argnames, args)) | kwargs
        symbols = {name: value for name, value in kwargs.items() if name not in self.sdfg.arrays}
        symbols |= self.sdfg.constants
        returns = self.return_names()
        for name in returns:
            if name not in kwargs:
                desc = self.sdfg.arrays[name]
                shape = tuple(int(symbolic.evaluate(s, symbols)) for s in desc.shape)
                kwargs[name] = np.empty(shape, desc.dtype.as_numpy_dtype())
        arglist = self.sdfg.arglist()
        missing = [name for name in arglist if name not in kwargs]
        if missing:
            raise TypeError(f"{self.sdfg.name}: missing arguments {missing}")
        self.function(**{name: kwargs[name] for name in arglist})
        values = [kwargs[name] for name in returns]
        if not values:
            return None
        return values[0] if len(values) == 1 else tuple(values)
