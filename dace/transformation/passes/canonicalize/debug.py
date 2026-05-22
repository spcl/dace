# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Per-stage debugging harness for the canonicalize pipeline.

Runs the canonicalize stages one at a time and, after each stage, checks
that the SDFG is still (i) structurally valid and (ii) numerically
equivalent to the un-canonicalized original on small random inputs. This
pinpoints the exact stage that breaks validity or changes results -- the
two failure modes the canonicalize work keeps hitting.

The harness generates inputs with every free (sizing) symbol set to a
small value and every data array/scalar filled with small-magnitude
random numbers (so transcendental kernels do not overflow), runs the
original SDFG once to get the reference outputs, then re-runs the
cumulative post-stage SDFG after each stage and compares.

Typical use::

    from dace.transformation.passes.canonicalize.debug import (
        canonicalize_with_stage_checks)
    results = canonicalize_with_stage_checks(sdfg)
    for r in results:
        print(r)
    # first failing stage:
    bad = next((r for r in results if not r.ok), None)
"""
import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dace import SDFG, data as dt, dtypes, symbolic
from dace.transformation.passes.canonicalize.pipeline import _build_stages

#: Default small value assigned to every free (sizing) symbol.
DEFAULT_SYMBOL_VALUE: int = 3
#: Default small-magnitude range for random floating-point input data.
DEFAULT_FLOAT_RANGE: Tuple[float, float] = (0.25, 1.0)
#: Default small range for random integer input data.
DEFAULT_INT_RANGE: Tuple[int, int] = (1, 4)


@dataclass
class StageCheckResult:
    """Outcome of validating + numerically checking one canonicalize stage.

    :param index: Stage position in the flat pipeline (0-based).
    :param label: Stage label (e.g. ``'fuse'``).
    :param pass_name: Class name of the applied pass.
    :param valid: Whether ``sdfg.validate()`` passed after the stage.
    :param validation_error: Validation exception text, or ``None``.
    :param numerically_correct: ``True``/``False`` if the post-stage SDFG ran
        and matched the reference; ``None`` if it could not be run.
    :param max_abs_diff: Largest absolute output difference vs. the reference
        across all compared arrays, or ``None`` if not run.
    :param run_error: Compile/run exception text, or ``None``.
    """
    index: int
    label: str
    pass_name: str
    valid: bool
    validation_error: Optional[str]
    numerically_correct: Optional[bool]
    max_abs_diff: Optional[float]
    run_error: Optional[str]

    @property
    def ok(self) -> bool:
        """``True`` iff the stage stayed valid and numerically correct."""
        return self.valid and self.numerically_correct is True

    def __str__(self) -> str:
        v = 'valid' if self.valid else f'INVALID({self.validation_error})'
        if self.run_error is not None:
            n = f'RUN-ERROR({self.run_error})'
        elif self.numerically_correct is None:
            n = 'not-run'
        elif self.numerically_correct:
            n = f'numeric-ok(maxdiff={self.max_abs_diff:.2e})'
        else:
            n = f'NUMERIC-MISMATCH(maxdiff={self.max_abs_diff:.2e})'
        return f'[{self.index:2}] {self.label:24s} {self.pass_name:36s} {v} | {n}'


def _resolve_shape(shape: Tuple[Any, ...], symbol_values: Dict[str, int]) -> Tuple[int, ...]:
    """Substitute symbol values into a (symbolic) descriptor shape.

    :param shape: Descriptor shape; each dimension is an int or a symbolic
        expression.
    :param symbol_values: Concrete value for every free symbol in ``shape``.
    :returns: The fully concrete shape.
    """
    out: List[int] = []
    for dim in shape:
        val = symbolic.evaluate(dim, symbol_values) if symbolic.issymbolic(dim) else dim
        out.append(int(val))
    return tuple(out)


def _random_for_dtype(dtype: dtypes.typeclass, shape: Tuple[int, ...], rng: np.random.Generator) -> Any:
    """Build small-magnitude random data of ``dtype`` and ``shape``.

    :param dtype: Element type of the data.
    :param shape: Concrete shape; an empty tuple yields a scalar.
    :param rng: Random generator drawn from for the data.
    :returns: A scalar of ``dtype`` if ``shape`` is empty, else an ndarray.
    """
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


def _build_random_inputs(sdfg: SDFG, symbol_value: int,
                         rng: np.random.Generator) -> Tuple[Dict[str, int], Dict[str, Any]]:
    """Build ``(symbol_values, data_values)`` for one run of ``sdfg``.

    Every free symbol is set to ``symbol_value``; every non-transient
    array/scalar is filled with small random data.

    :param sdfg: The SDFG whose argument list drives input construction.
    :param symbol_value: Value assigned to every free symbol.
    :param rng: Random generator drawn from for the data.
    :returns: ``(symbols, arrays)`` -- two dicts keyed by argument name.
    """
    free_syms = {str(s) for s in sdfg.free_symbols}
    symbols: Dict[str, int] = {s: symbol_value for s in free_syms}

    arrays: Dict[str, Any] = {}
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


def _run_capture(sdfg: SDFG, symbols: Dict[str, int], arrays: Dict[str, Any], tag: str) -> Dict[str, np.ndarray]:
    """Run a fresh copy of ``sdfg`` on copies of the inputs.

    A copy of every input is passed so callers can reuse the same inputs
    across runs without the in-place array writes of one run leaking into
    the next.

    :param sdfg: The SDFG to run.
    :param symbols: Concrete value for every free symbol.
    :param arrays: Input array/scalar values, keyed by argument name.
    :param tag: Suffix appended to the run copy's name to keep it unique.
    :returns: The post-run array values (outputs), keyed by argument name.
    """
    run_sdfg = copy.deepcopy(sdfg)
    run_sdfg.name = f'{run_sdfg.name}_{tag}'
    call_args = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in arrays.items()}
    call_args.update(symbols)
    run_sdfg(**call_args)
    return {k: v for k, v in call_args.items() if isinstance(v, np.ndarray)}


def _compare(ref: Dict[str, np.ndarray], got: Dict[str, np.ndarray], rtol: float, atol: float) -> Tuple[bool, float]:
    """Compare two output dicts arraywise.

    :param ref: Reference outputs from the un-canonicalized SDFG.
    :param got: Outputs from the post-stage SDFG.
    :param rtol: Relative tolerance for the comparison.
    :param atol: Absolute tolerance for the comparison.
    :returns: ``(all_close, max_abs_diff)`` over the arrays present in both.
    """
    max_diff = 0.0
    all_close = True
    for name, ref_val in ref.items():
        if name not in got:
            continue
        got_val = got[name]
        diff = float(np.max(np.abs(ref_val.astype(np.float64) - got_val.astype(np.float64)))) \
            if ref_val.size else 0.0
        max_diff = max(max_diff, diff)
        if not np.allclose(ref_val, got_val, rtol=rtol, atol=atol, equal_nan=True):
            all_close = False
    return all_close, max_diff


def canonicalize_with_stage_checks(sdfg: SDFG,
                                   symbol_value: int = DEFAULT_SYMBOL_VALUE,
                                   seed: int = 0,
                                   rtol: float = 1e-6,
                                   atol: float = 1e-9,
                                   stop_on_failure: bool = False) -> List[StageCheckResult]:
    """Canonicalize ``sdfg`` stage-by-stage, checking validity + numerical
    equivalence after every stage.

    The input ``sdfg`` is NOT mutated; a copy is canonicalized internally.

    :param sdfg: The SDFG to debug-canonicalize.
    :param symbol_value: Small value assigned to every free symbol.
    :param seed: RNG seed for the random inputs.
    :param rtol: Relative tolerance for the output comparison.
    :param atol: Absolute tolerance for the output comparison.
    :param stop_on_failure: Stop at the first invalid / mismatching stage.
    :returns: One :class:`StageCheckResult` per applied stage, in order.
    :raises RuntimeError: If the original (un-canonicalized) SDFG cannot be
        run to produce a reference (the harness needs a baseline).
    """
    original = copy.deepcopy(sdfg)
    rng = np.random.default_rng(seed)
    symbols, arrays = _build_random_inputs(original, symbol_value, rng)

    try:
        reference = _run_capture(original, symbols, arrays, 'ref')
    except Exception as e:
        raise RuntimeError(f'cannot run the original SDFG to get a numerical reference: {e}') from e

    results: List[StageCheckResult] = []
    work = copy.deepcopy(sdfg)
    for index, (label, unit) in enumerate(_build_stages()):
        unit.apply_pass(work, {})

        valid, verr = True, None
        try:
            work.validate()
        except Exception as e:
            valid, verr = False, str(e).splitlines()[0]

        num_ok: Optional[bool] = None
        max_diff: Optional[float] = None
        rerr: Optional[str] = None
        if valid:
            try:
                got = _run_capture(work, symbols, arrays, f's{index}')
                num_ok, max_diff = _compare(reference, got, rtol, atol)
            except Exception as e:
                rerr = str(e).splitlines()[0]

        results.append(
            StageCheckResult(index=index,
                             label=label,
                             pass_name=type(unit).__name__,
                             valid=valid,
                             validation_error=verr,
                             numerically_correct=num_ok,
                             max_abs_diff=max_diff,
                             run_error=rerr))
        if stop_on_failure and not results[-1].ok:
            break
    return results


def first_failing_stage(sdfg: SDFG, **kwargs) -> Optional[StageCheckResult]:
    """Convenience: return the first stage that broke validity or values,
    or ``None`` if the whole pipeline stayed valid + numerically correct.

    :param sdfg: The SDFG to debug-canonicalize.
    :param kwargs: Forwarded to :func:`canonicalize_with_stage_checks`.
    :returns: The first failing :class:`StageCheckResult`, or ``None``.
    """
    kwargs.setdefault('stop_on_failure', True)
    results = canonicalize_with_stage_checks(sdfg, **kwargs)
    return next((r for r in results if not r.ok), None)
