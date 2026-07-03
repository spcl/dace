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

#: Dataset symbols are clamped to this maximum for a fast value-preserving check.
SIZE_CAP: int = 16


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
    """Discover polybench kernels recursively across the category subfolders
    (datamining / linear_algebra / medley / stencils); skips the loader + ``__init__``."""
    pkg = importlib.import_module(_this_package())
    kernels: List[PolybenchKernel] = []
    for info in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        if info.name.rsplit(".", 1)[-1] in ("polybench", "__init__"):
            continue
        try:
            mod = importlib.import_module(info.name)
        except Exception:
            continue
        progs = [v for v in vars(mod).values() if isinstance(v, DaceProgram)]
        if not progs or not hasattr(mod, "sizes") or not hasattr(mod, "args") or not hasattr(mod, "init_array"):
            continue
        kernels.append(PolybenchKernel(modpath=info.name, program_name=progs[0].name))
    kernels.sort(key=lambda k: k.name)
    if name is not None:
        kernels = [k for k in kernels if k.name == name]
    return kernels


def make_inputs(kernel: PolybenchKernel,
                size_index: int = DEFAULT_SIZE_INDEX,
                cap: Optional[int] = SIZE_CAP) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Allocate + initialize one input set; return ``(call_arrays, symbol_values)``.

    ``call_arrays`` maps each kernel parameter name to its ndarray (``args`` order is
    aligned with the program's ``argnames``); ``symbol_values`` maps each dataset
    symbol (e.g. ``N``) to its concrete value. ``cap`` clamps dataset symbols (default
    ``SIZE_CAP`` for a fast value-preserving check); pass ``cap=None`` for the full
    ``sizes[size_index]`` preset (the perf/speedup test needs realistic sizes).
    """
    mod = _module(kernel)
    program = _program(mod)
    # Clamp dataset symbols to a small size for a fast numerical-correctness check
    # (the same value feeds the baseline and the candidate run, so value-preservation
    # is unaffected); polybench ``mini`` is still up to ~2000 on some kernels.
    psize = {str(k): (int(v) if cap is None else min(int(v), cap)) for k, v in mod.sizes[size_index].items()}
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


def _tol_for(dtype) -> Tuple[float, float]:
    """``(rtol, atol)`` appropriate to an array's numeric precision. Integers / bools
    compare exactly (``0, 0`` -> ``array_equal``); ``float32`` (e.g. ``deriche``) uses an
    fp32-appropriate tolerance so vectorization reassociation is not flagged as a bug;
    ``float64`` (the polybench majority) uses a tight fp64 tolerance. A single global
    tolerance is wrong when the corpus mixes precisions."""
    dt = np.dtype(dtype)
    if dt.kind in "iub":
        return 0.0, 0.0
    single = (dt.kind == "f" and dt.itemsize <= 4) or (dt.kind == "c" and dt.itemsize <= 8)
    return (1e-5, 1e-6) if single else (1e-9, 1e-11)


def outputs_match(ref: Dict[str, np.ndarray],
                  got: Dict[str, np.ndarray],
                  *,
                  rtol: float = None,
                  atol: float = None) -> bool:
    """Compare two result dicts with a DTYPE-AWARE tolerance (:func:`_tol_for`): fp64
    tight, fp32 fp32-appropriate, integers exact. Pass explicit ``rtol`` / ``atol`` to
    override the per-array default."""
    for name, r in ref.items():
        g = got[name]
        ra, ga = np.asarray(r), np.asarray(g)
        rt, at = _tol_for(ra.dtype)
        if rtol is not None:
            rt = rtol
        if atol is not None:
            at = atol
        if rt == 0.0 and at == 0.0:
            if not np.array_equal(ra, ga):
                return False
        elif not np.allclose(ra, ga, rtol=rt, atol=at, equal_nan=True):
            return False
    return True


# Inert CLI shim: the kernels' ``__main__`` blocks ``import polybench`` and call
# ``polybench.main(...)``; the corpus never triggers ``__main__``, but keep a no-op
# so running a kernel as a script fails loudly only on intent, not on import.
def main(*args, **kwargs):  # noqa: D401 - legacy CLI entry, intentionally a no-op
    raise SystemExit("polybench.main: the absl CLI harness was removed; use the corpus loader API instead.")
