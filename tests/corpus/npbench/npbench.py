# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""npbench corpus loader.

Each benchmark is a self-contained module under this package's dwarf subfolders
(e.g. ``map_reduce/go_fast.py``, ``ml/lenet.py``) exposing a ``CORPUS`` descriptor
fused from the npbench repo: the ``initialize`` input generator, the numpy
``reference`` kernel, the ``@dace.program`` kernel, and the ``S`` dataset preset
(+ any non-array ``scalars``).

This loader generates inputs at preset ``S``, computes the numpy reference, runs a
(transformed) SDFG built from the kernel on identical inputs, and compares -- a
real numerical-correctness check. Kernel parameters are matched to the initialized
arrays / scalar params / dataset symbols by NAME; outputs are taken from the
program's return value (mapped to ``output_args`` in order) or, for in-place
kernels, from the mutated arrays.
"""
import importlib
import inspect
import pkgutil
from typing import Dict, List, Optional, Tuple

import numpy as np

import dace


def _package():
    return importlib.import_module(__name__.rsplit(".", 1)[0])


def collect(name: Optional[str] = None) -> List[dict]:
    """Discover benchmark ``CORPUS`` descriptors recursively across dwarf folders."""
    pkg = _package()
    found: List[dict] = []
    for info in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        if info.name.rsplit(".", 1)[-1] in ("npbench", "__init__"):
            continue
        try:
            mod = importlib.import_module(info.name)
        except Exception:
            continue
        descriptor = vars(mod).get("CORPUS")
        if isinstance(descriptor, dict):
            found.append(descriptor)
    found.sort(key=lambda c: (c.get("dwarf", ""), c["name"]))
    if name is not None:
        found = [c for c in found if c["name"] == name]
    return found


#: Dataset symbols are clamped to this maximum for the corpus -- preset ``S`` is
#: sized for performance benchmarking (e.g. N=400000) and far too large for a fast
#: numerical-correctness check. Clamping the ints (the same value feeds the numpy
#: reference and the SDFG, so correctness is preserved) keeps compile+run tractable.
SIZE_CAP = 16


def _capped_sizes(c: dict, cap: Optional[int] = SIZE_CAP) -> Dict[str, object]:
    if cap is None:
        return dict(c["sizes"])
    return {
        k: (min(v, cap) if isinstance(v, int) and not isinstance(v, bool) else v)
        for k, v in c["sizes"].items()
    }


def make_inputs(c: dict, cap: Optional[int] = SIZE_CAP) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Initialize the named arrays at the dataset size; return ``(arrays, params)``
    where ``params`` holds the dataset symbols + any scalar kernel arguments.

    ``cap`` clamps integer dataset symbols (default ``SIZE_CAP`` for a fast numerical
    check); pass ``cap=None`` to use the full preset (e.g. the perf/speedup test, which
    needs realistic sizes)."""
    sizes = _capped_sizes(c, cap)
    args = [sizes[a] for a in c["input_args"]]
    rets = c["initialize"](*args)
    if not isinstance(rets, tuple):
        rets = (rets, )
    arrays = dict(zip(c["array_args"], rets))
    params = dict(c.get("scalars", {}))
    # Include ALL dataset sizes -- floats too (e.g. mandelbrot's xmin/horizon,
    # nbody's dt/G/softening, cavity/channel's rho/nu/F). The reference and kernel
    # resolve these by name; SDFG symbol binding in ``run_outputs`` filters floats
    # out separately, so it is safe to keep them in ``params``.
    params.update(sizes)
    return arrays, params


def _map_call(fn_or_program, arrays, params):
    """Build call kwargs for a function/program: each parameter name is resolved
    from the initialized arrays first, then the scalar/symbol params."""
    if hasattr(fn_or_program, "argnames"):
        names = fn_or_program.argnames
    else:
        names = list(inspect.signature(fn_or_program).parameters)
    pool = {**arrays, **params}
    return {n: pool[n] for n in names if n in pool}


def _collect_outputs(c, ret, work):
    out: Dict[str, np.ndarray] = {}
    rets = ret if isinstance(ret, tuple) else (ret, )
    for i, name in enumerate(c["output_args"]):
        if ret is not None and i < len(rets) and rets[i] is not None:
            out[name] = np.asarray(rets[i])
        else:
            out[name] = work[name]
    return out


def reference_outputs(c: dict, arrays: Dict[str, np.ndarray], params: Dict[str, object]) -> Dict[str, np.ndarray]:
    """Run the numpy reference on copies of the inputs; return the ``output_args``."""
    work = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in arrays.items()}
    call = _map_call(c["reference"], work, params)
    ret = c["reference"](**call)
    return _collect_outputs(c, ret, work)


def fresh_sdfg(c: dict, *, simplify: bool = True) -> dace.SDFG:
    sdfg = c["program"].to_sdfg(simplify=simplify)
    sdfg.name = f"{c['name']}_{sdfg.name}"
    return sdfg


def run_outputs(c: dict, sdfg: dace.SDFG, arrays: Dict[str, np.ndarray], params: Dict[str,
                                                                                      object]) -> Dict[str, np.ndarray]:
    """Compile + run ``sdfg`` on copies of the inputs; return the ``output_args``."""
    work = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in arrays.items()}
    call = _map_call(c["program"], work, params)
    # The SDFG also needs the dataset symbols it was parametrized over.
    symbols = {k: v for k, v in params.items() if not isinstance(v, float)}
    ret = sdfg.compile()(**call, **{k: v for k, v in symbols.items() if k not in call})
    return _collect_outputs(c, ret, work)


def outputs_match(ref: Dict[str, np.ndarray],
                  got: Dict[str, np.ndarray],
                  *,
                  rtol: float = 1e-3,
                  atol: float = 1e-4) -> bool:
    """Compare reference vs candidate ``output_args`` (relaxed tol -- npbench S is fp32)."""
    for name, r in ref.items():
        g = got.get(name)
        if g is None or not np.allclose(np.asarray(r), np.asarray(g), rtol=rtol, atol=atol, equal_nan=True):
            return False
    return True
