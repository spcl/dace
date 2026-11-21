# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Per-stage debugging harness for the canonicalize pipeline."""

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from dace import SDFG, dtypes, symbolic
from dace import data as dt
from dace.transformation.passes.canonicalize.pipeline import build_stages

#: Default small value assigned to every free (sizing) symbol.
DEFAULT_SYMBOL_VALUE: int = 3
#: Default small-magnitude range for random floating-point input data.
DEFAULT_FLOAT_RANGE: tuple[float, float] = (0.25, 1.0)
#: Default small range for random integer input data.
DEFAULT_INT_RANGE: tuple[int, int] = (1, 4)


@dataclass
class StageCheckResult:
    """Outcome of validating + numerically checking one canonicalize stage."""

    index: int
    label: str
    pass_name: str
    valid: bool
    validation_error: str | None
    numerically_correct: bool | None
    max_abs_diff: float | None
    run_error: str | None

    @property
    def ok(self) -> bool:
        """``True`` iff the stage stayed valid and numerically correct."""
        return self.valid and self.numerically_correct is True

    def __str__(self) -> str:
        v = "valid" if self.valid else f"INVALID({self.validation_error})"
        if self.run_error is not None:
            n = f"RUN-ERROR({self.run_error})"
        elif self.numerically_correct is None:
            n = "not-run"
        elif self.numerically_correct:
            n = f"numeric-ok(maxdiff={self.max_abs_diff:.2e})"
        else:
            n = f"NUMERIC-MISMATCH(maxdiff={self.max_abs_diff:.2e})"
        return f"[{self.index:2}] {self.label:24s} {self.pass_name:36s} {v} | {n}"


def _resolve_shape(shape: tuple[Any, ...], symbol_values: dict[str, int]) -> tuple[int, ...]:
    """Substitute symbol values into a (symbolic) descriptor shape."""
    out: list[int] = []
    for dim in shape:
        val = symbolic.evaluate(dim, symbol_values) if symbolic.issymbolic(dim) else dim
        out.append(int(val))
    return tuple(out)


def _random_for_dtype(dtype: dtypes.typeclass, shape: tuple[int, ...], rng: np.random.Generator) -> Any:
    """Build small-magnitude random data of ``dtype`` and ``shape``."""
    np_dtype = dtype.as_numpy_dtype()
    if np.issubdtype(np_dtype, np.integer):
        lo, hi = DEFAULT_INT_RANGE
        data = rng.integers(lo, hi + 1, size=shape)
    elif np.issubdtype(np_dtype, np.complexfloating):
        lo, hi = DEFAULT_FLOAT_RANGE
        data = rng.uniform(lo, hi, size=shape) + 1j * rng.uniform(lo, hi, size=shape)
    else:
        lo, hi = DEFAULT_FLOAT_RANGE
        data = rng.uniform(lo, hi, size=shape)
    if shape == ():
        return np_dtype.type(data)
    return data.astype(np_dtype)


def _build_random_inputs(
    sdfg: SDFG, symbol_value: int, rng: np.random.Generator
) -> tuple[dict[str, int], dict[str, Any]]:
    """Build ``(symbol_values, data_values)`` for one run of ``sdfg``."""
    free_syms = dict.fromkeys(str(s) for s in sdfg.free_symbols)
    symbols: dict[str, int] = dict.fromkeys(free_syms, symbol_value)

    arrays: dict[str, Any] = {}
    for name, desc in sdfg.arglist().items():
        if name in symbols:
            continue  # a sizing symbol, handled above
        if not isinstance(desc, (dt.Array, dt.Scalar)):
            continue
        if isinstance(desc, dt.Scalar):
            arrays[name] = _random_for_dtype(desc.dtype, (), rng)
        else:
            shape = _resolve_shape(desc.shape, symbols)
            arrays[name] = _random_for_dtype(desc.dtype, shape, rng)
    return symbols, arrays


def _run_capture(sdfg: SDFG, symbols: dict[str, int], arrays: dict[str, Any], tag: str) -> dict[str, np.ndarray]:
    """Run a fresh copy of ``sdfg`` on copies of the inputs."""
    run_sdfg = copy.deepcopy(sdfg)
    run_sdfg.name = f"{run_sdfg.name}_{tag}"
    call_args = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in arrays.items()}
    call_args.update(symbols)
    run_sdfg(**call_args)
    return {k: v for k, v in call_args.items() if isinstance(v, np.ndarray)}


def _compare(ref: dict[str, np.ndarray], got: dict[str, np.ndarray], rtol: float, atol: float) -> tuple[bool, float]:
    """Compare two output dicts arraywise."""
    max_diff = 0.0
    all_close = True
    for name, ref_val in ref.items():
        if name not in got:
            continue
        got_val = got[name]
        diff = float(np.max(np.abs(ref_val.astype(np.float64) - got_val.astype(np.float64)))) if ref_val.size else 0.0
        max_diff = max(max_diff, diff)
        if not np.allclose(ref_val, got_val, rtol=rtol, atol=atol, equal_nan=True):
            all_close = False
    return all_close, max_diff


def canonicalize_with_stage_checks(
    sdfg: SDFG,
    symbol_value: int = DEFAULT_SYMBOL_VALUE,
    seed: int = 0,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    stop_on_failure: bool = False,
) -> list[StageCheckResult]:
    """Canonicalize ``sdfg`` stage-by-stage, checking validity + numerical equivalence after every stage."""
    original = copy.deepcopy(sdfg)
    rng = np.random.default_rng(seed)
    symbols, arrays = _build_random_inputs(original, symbol_value, rng)

    try:
        reference = _run_capture(original, symbols, arrays, "ref")
    except Exception as e:
        raise RuntimeError(f"cannot run the original SDFG to get a numerical reference: {e}") from e

    results: list[StageCheckResult] = []
    work = copy.deepcopy(sdfg)
    for index, (label, unit) in enumerate(build_stages()):
        unit.apply_pass(work, {})

        valid, verr = True, None
        try:
            work.validate()
        except Exception as e:
            valid, verr = False, str(e).splitlines()[0]

        num_ok: bool | None = None
        max_diff: float | None = None
        rerr: str | None = None
        if valid:
            try:
                got = _run_capture(work, symbols, arrays, f"s{index}")
                num_ok, max_diff = _compare(reference, got, rtol, atol)
            except Exception as e:
                rerr = str(e).splitlines()[0]

        results.append(
            StageCheckResult(
                index=index,
                label=label,
                pass_name=type(unit).__name__,
                valid=valid,
                validation_error=verr,
                numerically_correct=num_ok,
                max_abs_diff=max_diff,
                run_error=rerr,
            )
        )
        if stop_on_failure and not results[-1].ok:
            break
    return results


def first_failing_stage(sdfg: SDFG, **kwargs: Any) -> StageCheckResult | None:
    """Convenience: return the first stage that broke validity or values, or ``None`` if the whole pipeline stayed valid
    + numerically correct."""
    kwargs.setdefault("stop_on_failure", True)
    results = canonicalize_with_stage_checks(sdfg, **kwargs)
    return next((r for r in results if not r.ok), None)
