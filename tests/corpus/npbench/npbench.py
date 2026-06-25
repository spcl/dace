# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""npbench corpus loader (non-polybench groups: misc / deep_learning / weather).

Each npbench benchmark module ships a ``run_<name>(device_type)`` driver that wires
initialization, SDFG construction, ``auto_optimize`` and reference checking together
in a bespoke way (the kernel program name, the init function, whether the kernel
writes in place or returns, which arguments are arrays vs. symbols all vary per
module). Rather than re-deriving that wiring per kernel, this loader REUSES each
module's own driver via an ``auto_optimize`` capture shim:

* Replace ``auto_optimize`` in the module namespace with a proxy that records the
  exact ``(args, kwargs)`` the driver passes to the compiled SDFG (deep-copied
  *before* the run, so in-place kernels don't corrupt the snapshot) and the
  reference outputs produced by the known-correct optimized run.
* The corpus then builds a FRESH ``program.to_sdfg()`` from the same kernel and
  replays those captured inputs through any transform pipeline, comparing against
  the captured reference.

The polybench npbench group is intentionally excluded -- those kernels are covered
by the dedicated :mod:`tests.corpus.polybench` corpus (harness convention).
"""
import copy
import importlib
import pkgutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import dace
from dace.frontend.python.parser import DaceProgram
from dace.transformation.auto import auto_optimize as _ao_mod

#: npbench groups that are NOT polybench (polybench lives in its own corpus).
GROUPS: Tuple[str, ...] = ("misc", "deep_learning", "weather_stencils")

#: Package the benchmark modules are imported from.
SOURCE_PACKAGE: str = "tests.npbench"


@dataclass(frozen=True)
class NPBenchKernel:
    """One npbench benchmark: its module, driver, and kernel ``@dace.program``."""
    modpath: str
    run_fn: str
    program_name: str

    @property
    def name(self) -> str:
        # Strip the ``_test`` suffix for a clean corpus id (e.g. ``go_fast``).
        return self.modpath.rsplit(".", 1)[-1].removesuffix("_test")


def _iter_module_paths() -> List[str]:
    paths: List[str] = []
    for group in GROUPS:
        try:
            pkg = importlib.import_module(f"{SOURCE_PACKAGE}.{group}")
        except ModuleNotFoundError:
            continue
        for info in pkgutil.iter_modules(pkg.__path__):
            if info.name.endswith("_test"):
                paths.append(f"{SOURCE_PACKAGE}.{group}.{info.name}")
    return sorted(paths)


def _module_programs(mod) -> Dict[str, DaceProgram]:
    return {n: v for n, v in vars(mod).items() if isinstance(v, DaceProgram)}


def _pick_run_fn(mod) -> Optional[str]:
    runs = sorted(n for n in vars(mod) if n.startswith("run_") and callable(vars(mod)[n]))
    return runs[0] if runs else None


def _pick_program(mod, run_fn: str) -> Optional[str]:
    progs = _module_programs(mod)
    if not progs:
        return None
    if len(progs) == 1:
        return next(iter(progs))
    # Multiple module-level programs (rare): prefer one whose name appears in the
    # driver's source, else ``kernel``, else the first defined.
    import inspect
    try:
        src = inspect.getsource(vars(mod)[run_fn])
    except (OSError, TypeError):
        src = ""
    for name in progs:
        if name in src:
            return name
    return "kernel" if "kernel" in progs else next(iter(progs))


def collect(name: Optional[str] = None) -> List[NPBenchKernel]:
    """Discover npbench (non-polybench) kernels. ``name`` filters by corpus id."""
    kernels: List[NPBenchKernel] = []
    for modpath in _iter_module_paths():
        mod = importlib.import_module(modpath)
        run_fn = _pick_run_fn(mod)
        if run_fn is None:
            continue
        prog = _pick_program(mod, run_fn)
        if prog is None:
            continue
        kernels.append(NPBenchKernel(modpath=modpath, run_fn=run_fn, program_name=prog))
    if name is not None:
        kernels = [k for k in kernels if k.name == name]
    return kernels


@dataclass
class Captured:
    """The replayable inputs + known-correct reference for one kernel."""
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    ref_return: Any
    ref_args: Tuple[Any, ...]  # in-place-mutated positional args after the reference run
    ref_kwargs: Dict[str, Any]  # in-place-mutated kwarg arrays after the reference run


def capture(kernel: NPBenchKernel) -> Captured:
    """Run the module's ``run_*`` driver with an ``auto_optimize`` capture shim and
    return the inputs it fed the SDFG plus the reference outputs of the optimized
    (known-correct) run.
    """
    mod = importlib.import_module(kernel.modpath)
    real_ao = _ao_mod.auto_optimize
    grabbed: Dict[str, Any] = {}

    class _Proxy:

        def __init__(self, sdfg):
            self._sdfg = sdfg

        def __call__(self, *args, **kwargs):
            # Snapshot inputs BEFORE the run (in-place kernels mutate them).
            grabbed["args"] = copy.deepcopy(args)
            grabbed["kwargs"] = copy.deepcopy(kwargs)
            grabbed["sdfg"] = self._sdfg
            return self._sdfg(*args, **kwargs)

        def __getattr__(self, item):
            return getattr(self._sdfg, item)

    def _fake_ao(sdfg, device, *a, **k):
        return _Proxy(real_ao(sdfg, device, *a, **k))

    # The driver did ``from ...auto_optimize import auto_optimize`` -- patch the
    # name in the module namespace (and the source module, belt-and-braces).
    had_attr = "auto_optimize" in vars(mod)
    saved = vars(mod).get("auto_optimize")
    mod.auto_optimize = _fake_ao
    try:
        getattr(mod, kernel.run_fn)(dace.dtypes.DeviceType.CPU)
    finally:
        if had_attr:
            mod.auto_optimize = saved
        else:
            del mod.auto_optimize

    if "sdfg" not in grabbed:
        raise RuntimeError(f"{kernel.name}: driver did not call the SDFG (no inputs captured)")

    # Produce the reference outputs from the known-correct optimized SDFG.
    ref_args = copy.deepcopy(grabbed["args"])
    ref_kwargs = copy.deepcopy(grabbed["kwargs"])
    ref_return = grabbed["sdfg"](*copy.deepcopy(ref_args), **ref_kwargs)
    return Captured(args=grabbed["args"],
                    kwargs=grabbed["kwargs"],
                    ref_return=ref_return,
                    ref_args=ref_args,
                    ref_kwargs=ref_kwargs)


def fresh_sdfg(kernel: NPBenchKernel, *, simplify: bool = True) -> dace.SDFG:
    """A fresh, unoptimized SDFG built from the kernel's ``@dace.program``."""
    mod = importlib.import_module(kernel.modpath)
    program = _module_programs(mod)[kernel.program_name]
    sdfg = program.to_sdfg(simplify=simplify)
    sdfg.name = f"{kernel.name}_{sdfg.name}"
    return sdfg


def run_and_collect(sdfg: dace.SDFG, cap: Captured) -> Tuple[Any, Tuple, Dict]:
    """Compile + run ``sdfg`` on fresh copies of the captured inputs; return
    ``(return_value, mutated_positional_args, mutated_kwargs)``."""
    args = copy.deepcopy(cap.args)
    kwargs = copy.deepcopy(cap.kwargs)
    ret = sdfg.compile()(*args, **kwargs)
    return ret, args, kwargs


def outputs_match(cap: Captured, got: Tuple[Any, Tuple, Dict], *, rtol: float = 1e-6, atol: float = 1e-6) -> bool:
    """Compare a ``run_and_collect`` result against the captured reference, over
    the return value and every in-place-mutated floating array."""
    got_ret, got_args, got_kwargs = got

    def _close(a, b) -> bool:
        if isinstance(a, np.ndarray) and np.issubdtype(a.dtype, np.floating):
            return bool(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True))
        if isinstance(a, np.ndarray):  # integer / bool arrays: exact
            return bool(np.array_equal(a, b))
        if isinstance(a, (float, np.floating)):
            return bool(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True))
        return a == b

    if isinstance(cap.ref_return, (list, tuple)):
        if not isinstance(got_ret, (list, tuple)) or len(got_ret) != len(cap.ref_return):
            return False
        if not all(_close(r, g) for r, g in zip(cap.ref_return, got_ret)):
            return False
    elif cap.ref_return is not None:
        if not _close(cap.ref_return, got_ret):
            return False
    for r, g in zip(cap.ref_args, got_args):
        if not _close(r, g):
            return False
    for key, r in cap.ref_kwargs.items():
        if not _close(r, got_kwargs.get(key)):
            return False
    return True
