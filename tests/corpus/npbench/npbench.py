# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""npbench corpus loader.

Each benchmark is a self-contained module in this package exposing a ``CORPUS``
descriptor (see e.g. :mod:`tests.corpus.npbench.go_fast`) fused from the npbench
repo: the ``initialize`` input generator, the numpy ``reference`` kernel, the
``@dace.program`` kernel, and the ``S`` dataset preset.

This loader provides the uniform machinery to (a) generate inputs at preset ``S``,
(b) compute the numpy reference outputs, and (c) run a (transformed) SDFG built
from the kernel on identical inputs and compare -- a real numerical-correctness
check. The kernel's parameters are matched to the initialized arrays / dataset
symbols by NAME; outputs are taken from the program's return value (mapped to
``output_args`` in order) or, for in-place kernels, from the mutated arrays.
"""
import importlib
import inspect
import pkgutil
from typing import Dict, List, Optional, Tuple

import numpy as np

import dace


def _package() -> str:
    return __name__.rsplit(".", 1)[0]


def collect(name: Optional[str] = None) -> List[dict]:
    """Discover benchmark ``CORPUS`` descriptors in this package."""
    pkg = importlib.import_module(_package())
    found: List[dict] = []
    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name in ("npbench", "__init__"):
            continue
        mod = importlib.import_module(f"{_package()}.{info.name}")
        descriptor = vars(mod).get("CORPUS")
        if isinstance(descriptor, dict):
            found.append(descriptor)
    found.sort(key=lambda c: c["name"])
    if name is not None:
        found = [c for c in found if c["name"] == name]
    return found


def make_inputs(c: dict) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Initialize the named input arrays at preset ``S`` plus the symbol values."""
    args = [c["sizes"][a] for a in c["input_args"]]
    rets = c["initialize"](*args)
    if not isinstance(rets, tuple):
        rets = (rets, )
    arrays = dict(zip(c["array_args"], rets))
    # Dataset symbols the kernel/SDFG needs (scalars not produced by initialize).
    symbols = {k: v for k, v in c["sizes"].items() if not isinstance(v, float)}
    return arrays, symbols


def reference_outputs(c: dict, arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Run the numpy reference on copies of the inputs; return the ``output_args``."""
    work = {k: v.copy() for k, v in arrays.items()}
    ref = c["reference"]
    params = list(inspect.signature(ref).parameters)
    ret = ref(*[work[p] for p in params if p in work])
    out: Dict[str, np.ndarray] = {}
    rets = ret if isinstance(ret, tuple) else (ret, )
    for i, name in enumerate(c["output_args"]):
        if ret is not None and i < len(rets) and rets[i] is not None:
            out[name] = np.asarray(rets[i])
        else:
            out[name] = work[name]
    return out


def fresh_sdfg(c: dict, *, simplify: bool = True) -> dace.SDFG:
    sdfg = c["program"].to_sdfg(simplify=simplify)
    sdfg.name = f"{c['name']}_{sdfg.name}"
    return sdfg


def run_outputs(c: dict, sdfg: dace.SDFG, arrays: Dict[str, np.ndarray], symbols: Dict[str,
                                                                                       int]) -> Dict[str, np.ndarray]:
    """Compile + run ``sdfg`` on copies of the inputs; return the ``output_args``."""
    work = {k: v.copy() for k, v in arrays.items()}
    call = {n: work[n] for n in c["program"].argnames if n in work}
    ret = sdfg.compile()(**call, **symbols)
    out: Dict[str, np.ndarray] = {}
    rets = ret if isinstance(ret, tuple) else (ret, )
    for i, name in enumerate(c["output_args"]):
        if ret is not None and i < len(rets) and rets[i] is not None:
            out[name] = np.asarray(rets[i])
        else:
            out[name] = work[name]
    return out


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
