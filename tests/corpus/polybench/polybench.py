# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""polybench corpus loader (SDFG inputs from the python frontend).

Each polybench kernel module in this package declares, at module level:

* ``sizes``      -- a list of 5 dataset dicts ``{symbol: value}`` (mini..extra-large),
* ``args``       -- a list of ``(shape, dtype)`` specs aligned with the kernel's
  positional parameters (``shape`` entries may be :class:`dace.symbol`\\ s),
* ``init_array`` -- ``init_array(*arrays, **lowercase_symbol_values)`` filling them,
* a single ``@dace.program`` kernel.

Unlike npbench these kernels carry no numpy oracle (the original suite validated
against a C dump), so correctness here is *value-preserving*: a transform pipeline
is compared against the untransformed baseline SDFG run on identical inputs.

The legacy absl ``polybench.main`` harness is gone -- the kernels import it only
inside their ``if __name__ == '__main__'`` blocks (CLI use), so loading a module
pulls in no extra dependency.
"""
import importlib
import pkgutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import dace
from dace.frontend.python.parser import DaceProgram

#: Smallest dataset (``sizes`` index 0) -- keeps the corpus sweep fast.
DEFAULT_SIZE_INDEX: int = 0


@dataclass(frozen=True)
class PolybenchKernel:
    modpath: str
    program_name: str

    @property
    def name(self) -> str:
        return self.modpath.rsplit(".", 1)[-1]


def _this_package() -> str:
    return __name__.rsplit(".", 1)[0]


def _module(kernel: PolybenchKernel):
    return importlib.import_module(kernel.modpath)


def _program(mod) -> DaceProgram:
    progs = [v for v in vars(mod).values() if isinstance(v, DaceProgram)]
    return progs[0]


def collect(name: Optional[str] = None) -> List[PolybenchKernel]:
    """Discover polybench kernels in this package (skips the loader + ``__init__``)."""
    pkg = importlib.import_module(_this_package())
    kernels: List[PolybenchKernel] = []
    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name in ("polybench", "__init__"):
            continue
        modpath = f"{_this_package()}.{info.name}"
        mod = importlib.import_module(modpath)
        progs = [v for v in vars(mod).values() if isinstance(v, DaceProgram)]
        if not progs or not hasattr(mod, "sizes") or not hasattr(mod, "args") or not hasattr(mod, "init_array"):
            continue
        kernels.append(PolybenchKernel(modpath=modpath, program_name=progs[0].name))
    kernels.sort(key=lambda k: k.name)
    if name is not None:
        kernels = [k for k in kernels if k.name == name]
    return kernels


def make_inputs(kernel: PolybenchKernel,
                size_index: int = DEFAULT_SIZE_INDEX) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Allocate + initialize one input set; return ``(call_arrays, symbol_values)``.

    ``call_arrays`` maps each kernel parameter name to its ndarray (``args`` order is
    aligned with the program's ``argnames``); ``symbol_values`` maps each dataset
    symbol (e.g. ``N``) to its concrete value.
    """
    mod = _module(kernel)
    program = _program(mod)
    psize = {str(k): v for k, v in mod.sizes[size_index].items()}
    arrays = []
    for shape, dtype in mod.args:
        concrete = [psize[str(s)] if isinstance(s, dace.symbol) else s for s in shape]
        arrays.append(dace.ndarray(concrete, dtype))
    mod.init_array(*arrays, **{k.lower(): v for k, v in psize.items()})
    call_arrays = {n: a for n, a in zip(program.argnames, arrays)}
    return call_arrays, psize


def fresh_sdfg(kernel: PolybenchKernel, *, simplify: bool = True) -> dace.SDFG:
    """A fresh, unoptimized SDFG built from the kernel's ``@dace.program``."""
    sdfg = _program(_module(kernel)).to_sdfg(simplify=simplify)
    sdfg.name = f"{kernel.name}_{sdfg.name}"
    return sdfg


def reference(kernel: PolybenchKernel, call_arrays: Dict[str, np.ndarray], psize: Dict[str,
                                                                                       int]) -> Dict[str, np.ndarray]:
    """Run the untransformed baseline SDFG on copies of the inputs; return the
    resulting arrays (the value-preserving ground truth)."""
    base = fresh_sdfg(kernel)
    out = {n: a.copy() for n, a in call_arrays.items()}
    base.compile()(**out, **psize)
    return out


def run(sdfg: dace.SDFG, call_arrays: Dict[str, np.ndarray], psize: Dict[str, int]) -> Dict[str, np.ndarray]:
    """Compile + run ``sdfg`` on copies of the inputs; return the resulting arrays."""
    out = {n: a.copy() for n, a in call_arrays.items()}
    sdfg.compile()(**out, **psize)
    return out


def outputs_match(ref: Dict[str, np.ndarray],
                  got: Dict[str, np.ndarray],
                  *,
                  rtol: float = 1e-9,
                  atol: float = 1e-9) -> bool:
    """Compare two result dicts over every floating array (integer arrays exact)."""
    for name, r in ref.items():
        g = got[name]
        if np.issubdtype(r.dtype, np.floating):
            if not np.allclose(r, g, rtol=rtol, atol=atol, equal_nan=True):
                return False
        elif not np.array_equal(r, g):
            return False
    return True


# Inert CLI shim: the kernels' ``__main__`` blocks ``import polybench`` and call
# ``polybench.main(...)``; the corpus never triggers ``__main__``, but keep a no-op
# so running a kernel as a script fails loudly only on intent, not on import.
def main(*args, **kwargs):  # noqa: D401 - legacy CLI entry, intentionally a no-op
    raise SystemExit("polybench.main: the absl CLI harness was removed; use the corpus loader API instead.")
