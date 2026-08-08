# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""polybench corpus loader (SDFG inputs from the python frontend).

Each polybench kernel module in this package declares, at module level:

* ``sizes``      -- a list of 5 dataset dicts ``{symbol: value}`` (mini..extra-large),
* ``args``       -- a list of ``(shape, dtype)`` specs aligned with the kernel's
  positional parameters (``shape`` entries may be :class:`dace.symbol`\\ s),
* ``init_array`` -- ``init_array(*arrays, **lowercase_symbol_values)`` filling them,
* a single ``@dace.program`` kernel.

There are TWO references, and they are both kept:

* :func:`reference` runs the untransformed baseline SDFG -- compiled C++, a DaCe-vs-DaCe
  ground truth. It is what the corpus has always compared against (*value preservation*).
* :func:`numpy_reference` runs the in-repo numpy formulation from
  :mod:`tests.corpus.polybench.polybench_numpy`, so polybench and npbench can share ONE
  honest denominator -- parallel numpy -- instead of secretly dividing by compiled C++.
  :func:`numpy_call` hands a perf harness the same reference as ``(fn, kwargs)``, timeable
  exactly like an arm. WARNING: Not every kernel's numpy reference is array-level; see
  ``polybench_numpy.VECTORIZATION`` before putting one in a "vs numpy" figure.

The legacy absl ``polybench.main`` harness is gone -- the kernels import it only
inside their ``if __name__ == '__main__'`` blocks (CLI use), so loading a module
pulls in no extra dependency.
"""
import importlib
import inspect
import pkgutil
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

import dace
from dace.frontend.python.parser import DaceProgram
from tests.corpus.polybench import polybench_numpy

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


def program(kernel: PolybenchKernel) -> DaceProgram:
    """The kernel's ``@dace.program`` object (source-level inspection; no SDFG build)."""
    return _program(_module(kernel))


def collect(name: Optional[str] = None) -> List[PolybenchKernel]:
    """Discover polybench kernels recursively across the category subfolders
    (datamining / linear_algebra / medley / stencils); skips the loader + ``__init__``."""
    pkg = importlib.import_module(_this_package())
    kernels: List[PolybenchKernel] = []
    for info in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        if info.name.rsplit(".", 1)[-1] in ("polybench", "polybench_numpy", "__init__"):
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


def paper_size_row(mod) -> Dict[str, int]:
    """The module's ``paper_sizes`` row, or its largest ``sizes`` entry as a fallback.

    polybench's own five sizes are mini..extra-large, none of which is the shape the papers
    report; the ``paper_sizes`` row carries that (ported from the npbench repo's ``bench_info``
    descriptors, which pin a ``paper`` dataset per kernel).
    """
    return vars(mod).get('paper_sizes') or mod.sizes[-1]


def make_inputs(kernel: PolybenchKernel,
                size_index: int = DEFAULT_SIZE_INDEX,
                cap: Optional[int] = SIZE_CAP,
                paper: bool = False) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Allocate + initialize one input set; return ``(call_arrays, symbol_values)``.

    ``call_arrays`` maps each kernel parameter name to its ndarray (``args`` order is
    aligned with the program's ``argnames``); ``symbol_values`` maps each dataset
    symbol (e.g. ``N``) to its concrete value. ``cap`` clamps dataset symbols (default
    ``SIZE_CAP`` for a fast value-preserving check); pass ``cap=None`` for the full
    ``sizes[size_index]`` preset (the perf/speedup test needs realistic sizes).
    ``paper=True`` takes the published ``paper_sizes`` row instead of ``sizes[size_index]``.
    """
    mod = _module(kernel)
    program = _program(mod)
    # Clamp dataset symbols to a small size for a fast numerical-correctness check
    # (the same value feeds the baseline and the candidate run, so value-preservation
    # is unaffected); polybench ``mini`` is still up to ~2000 on some kernels.
    row = paper_size_row(mod) if paper else mod.sizes[size_index]
    psize = {str(k): (int(v) if cap is None else min(int(v), cap)) for k, v in row.items()}
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


def numpy_call(kernel: PolybenchKernel, call_arrays: Dict[str, np.ndarray],
               psize: Dict[str, int]) -> Tuple[Callable[..., None], Dict[str, object]]:
    """``(fn, kwargs)`` for repeated *timed* invocation of this kernel's numpy reference.

    Shaped like :func:`tests.corpus.corpus_suite.compiled_call` so a perf harness can treat the
    denominator exactly like an arm: the copies and the name resolution happen HERE, once, outside
    the timed region. The reference's parameters are resolved by name against the input arrays and
    the dataset symbols, the same rule the tsvc_2_5 oracle adapter uses.

    WARNING: The polybench kernels write their inputs, so repetition ``k+1`` would otherwise see what
    repetition ``k`` left behind; call :func:`restore_inputs` between repetitions (outside the
    timed region) when the kernel's result depends on its input state.
    WARNING: Time it WITHOUT pinning BLAS threads: the agreed baseline is *parallel* numpy.
    """
    fn = polybench_numpy.REFERENCES[kernel.name]
    pool: Dict[str, object] = {**{n: a.copy() for n, a in call_arrays.items()}, **psize}
    return fn, {p: pool[p] for p in inspect.signature(fn).parameters}


def restore_inputs(kwargs: Dict[str, object], call_arrays: Dict[str, np.ndarray]) -> None:
    """Reset the array arguments in ``kwargs`` to their pristine input values, IN PLACE.

    In place rather than re-copying so a timing loop reuses one allocation per array; run it
    between repetitions but outside the timed region, or the memcpy lands in the denominator.

    ARRAY arguments only, as the name says: npbench passes some of its inputs as python/numpy
    scalars (``cavity_flow`` dt/dx/dy, ``compute`` a/b/c), which are immutable and cannot be
    written back into. They are pass-by-value, so a callee cannot have changed them either.
    """
    for name, pristine in call_arrays.items():
        target = kwargs.get(name)
        if isinstance(target, np.ndarray) and isinstance(pristine, np.ndarray):
            np.copyto(target, pristine)


def numpy_reference(kernel: PolybenchKernel, call_arrays: Dict[str, np.ndarray],
                    psize: Dict[str, int]) -> Dict[str, np.ndarray]:
    """Run the numpy reference on copies of the inputs; return the resulting arrays.

    The second ground truth, alongside :func:`reference`'s untransformed SDFG: same inputs, same
    output dict shape, so the two can be cross-checked against each other kernel by kernel.
    """
    out = {n: a.copy() for n, a in call_arrays.items()}
    fn = polybench_numpy.REFERENCES[kernel.name]
    pool: Dict[str, object] = {**out, **psize}
    fn(**{p: pool[p] for p in inspect.signature(fn).parameters})
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


#: Reassociation floor for the DEFAULT absolute tolerance, as a fraction of an array's own
#: largest magnitude. ``_tol_for``'s absolute term is a constant, so it stops covering the
#: REFERENCE's own rounding once the dataset grows: at the tsvc_2_5 paper preset a 589824-long
#: fp64 prefix sum leaves the scalar oracle 2.7e-11 off a long-double reference against the
#: canonicalize candidate's 2.8e-12, so the 1e-11 constant failed the more accurate of the two.
#: The comparison stays elementwise -- the floor only frees an element that CANCELLED to near zero
#: from a relative tolerance about its own tiny value. Same constant as ``tests/library/scan_test.py``.
REASSOC_SCALE = 1e-12


def atol_for(ra: np.ndarray, at: float) -> float:
    """``at`` raised to this array's reassociation floor. See :data:`REASSOC_SCALE`.

    Only ever applied to the dtype-derived DEFAULT: an explicitly-passed tolerance is the
    caller's own bound and is taken as given, and an exact (``0.0``) gate stays exact. A
    reference that is not all-finite sizes nothing either -- tsvc_2_5's ``wf_north_west`` /
    ``wf_triangular`` overflow to ``inf`` at the paper preset, and a floor read off that
    would be ~1e295 wide -- so it keeps the constant.
    """
    if at == 0.0:
        return 0.0
    if ra.size and not np.all(np.isfinite(ra)):
        return at
    return max(at, REASSOC_SCALE * float(np.max(np.abs(ra), initial=0.0)))


def outputs_match(ref: Dict[str, np.ndarray],
                  got: Dict[str, np.ndarray],
                  *,
                  rtol: float = None,
                  atol: float = None) -> bool:
    """Compare two result dicts with a DTYPE-AWARE tolerance (:func:`_tol_for`): fp64
    tight, fp32 fp32-appropriate, integers exact. The default absolute term is raised to
    the array's reassociation floor (:data:`REASSOC_SCALE`). Pass explicit ``rtol`` /
    ``atol`` to override the per-array default; an explicit ``atol`` is used as given."""
    for name, r in ref.items():
        g = got[name]
        ra, ga = np.asarray(r), np.asarray(g)
        rt, at = _tol_for(ra.dtype)
        at = atol if atol is not None else atol_for(ra, at)
        if rtol is not None:
            rt = rtol
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
