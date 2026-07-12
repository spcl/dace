# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CloudSC / ICON loopnest snippets for the canonicalization sweep.

The full CloudSC SDFG is far too large for a quick canonicalization check, so this
loader reuses the *characteristic loopnest snippets* already maintained as faithful
``@dace.program`` ports under ``tests/passes/vectorization/cloudsc/`` (the same ones
the vectorization tests exercise). Each snippet is a self-contained ``@dace.program``;
its argument shapes/dtypes are introspected from the function annotations.

Like polybench, these snippets carry no numpy oracle -- correctness here is
*value-preserving*: a canonicalized SDFG is compared against
the untransformed baseline SDFG run on identical inputs.
"""
import importlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import dace
from dace.frontend.python.parser import DaceProgram

#: Test modules that hold the snippet ``@dace.program`` definitions.
SNIPPET_MODULES = [
    "tests.passes.vectorization.cloudsc.test_cloudsc_loopnests",
    "tests.passes.vectorization.cloudsc.test_cloudsc",
    "tests.passes.vectorization.cloudsc.test_icon_loopnests",
]

#: One small uniform size for every dataset symbol (KLEV / KLON / NB / NPROMA / ...).
#: Using a single value keeps gather index tables (filled with ``randint(0, SIZE)``)
#: in-bounds for whichever dimension they address.
SIZE: int = 8


@dataclass(frozen=True)
class Snippet:
    modpath: str
    program_name: str

    @property
    def name(self) -> str:
        return self.program_name


def _program(s: Snippet) -> DaceProgram:
    mod = importlib.import_module(s.modpath)
    return vars(mod)[s.program_name]


def collect(name: Optional[str] = None) -> List[Snippet]:
    """Discover the snippet ``@dace.program`` objects across the snippet modules."""
    snippets: List[Snippet] = []
    seen = set()
    for modpath in SNIPPET_MODULES:
        try:
            mod = importlib.import_module(modpath)
        except Exception:
            continue
        for vname, v in vars(mod).items():
            if isinstance(v, DaceProgram) and v.name not in seen:
                seen.add(v.name)
                snippets.append(Snippet(modpath=modpath, program_name=vname))
    snippets.sort(key=lambda s: s.name)
    if name is not None:
        snippets = [s for s in snippets if s.name == name]
    return snippets


def _symbol_values(prog: DaceProgram) -> Dict[str, int]:
    """Every free symbol in every argument shape -> the uniform ``SIZE``."""
    syms = set()
    for t in prog.f.__annotations__.values():
        for dim in getattr(t, "shape", ()):  # scalars have no shape
            syms |= {str(s) for s in dace.symbolic.pystr_to_symbolic(dim).free_symbols}
    return {s: SIZE for s in syms}


def make_inputs(prog: DaceProgram) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Allocate + initialize one input set; return ``(call_arrays, symbol_values)``.

    Integer arrays are filled with ``randint(0, SIZE)`` (valid gather indices for any
    SIZE-sized dimension); float arrays with uniform random values. Scalar arguments
    get a representative value.
    """
    rng = np.random.default_rng(42)
    symvals = _symbol_values(prog)
    arrays: Dict[str, np.ndarray] = {}
    for argname, t in prog.f.__annotations__.items():
        if argname == "return":
            continue
        shape = getattr(t, "shape", None)
        np_dtype = np.dtype(t.dtype.as_numpy_dtype())
        if shape is None or len(shape) == 0:  # scalar argument
            arrays[argname] = np_dtype.type(1)
            continue
        concrete = [int(dace.symbolic.evaluate(d, symvals)) for d in shape]
        if np.issubdtype(np_dtype, np.integer):
            arrays[argname] = rng.integers(0, SIZE, size=concrete).astype(np_dtype)
        else:
            arrays[argname] = rng.random(size=concrete).astype(np_dtype)
    return arrays, symvals


def fresh_sdfg(prog: DaceProgram, *, simplify: bool = True) -> dace.SDFG:
    sdfg = prog.to_sdfg(simplify=simplify)
    sdfg.name = f"{prog.name}_{sdfg.name}"
    return sdfg


def reference(prog: DaceProgram, call_arrays: Dict[str, np.ndarray], symvals: Dict[str, int]) -> Dict[str, np.ndarray]:
    """Run the untransformed baseline SDFG on copies of the inputs (value-preserving ground truth)."""
    base = fresh_sdfg(prog)
    out = {n: (a.copy() if isinstance(a, np.ndarray) else a) for n, a in call_arrays.items()}
    base.compile()(**out, **symvals)
    return {n: v for n, v in out.items() if isinstance(v, np.ndarray)}


def run(sdfg: dace.SDFG, call_arrays: Dict[str, np.ndarray], symvals: Dict[str, int]) -> Dict[str, np.ndarray]:
    out = {n: (a.copy() if isinstance(a, np.ndarray) else a) for n, a in call_arrays.items()}
    sdfg.compile()(**out, **symvals)
    return {n: v for n, v in out.items() if isinstance(v, np.ndarray)}


def outputs_match(ref: Dict[str, np.ndarray],
                  got: Dict[str, np.ndarray],
                  *,
                  rtol: float = 1e-9,
                  atol: float = 1e-9) -> bool:
    for name, r in ref.items():
        g = got[name]
        if np.issubdtype(r.dtype, np.floating):
            if not np.allclose(r, g, rtol=rtol, atol=atol, equal_nan=True):
                return False
        elif not np.array_equal(r, g):
            return False
    return True
