# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The polybench numpy reference must agree with the untransformed-SDFG reference.

Two things are checked:

* every collected kernel HAS a numpy reference and a declared vectorization class,
* the numpy reference agrees with the SDFG reference within the corpus's own dtype-aware tolerance
  and moves EXACTLY the arrays the SDFG reference moves (a reference that quietly computes nothing
  would otherwise pass the comparison by turning it into ``inputs == inputs``).

NOTE: the second check is written as "same arrays moved as the SDFG reference", not "at least one
array moved", because two polybench kernels are IDENTITY MAPS on their own ``init_array`` input:
``heat_3d`` (the initial condition is linear in i/j/k, so the 7-point second-difference stencil is
identically zero) and ``seidel_2d`` (the 9-point average of its bilinear initial condition
reproduces itself, to rounding). Both references correctly reproduce that, and pinning the numpy
side to the SDFG side is the strongest claim this file can honestly make.

The dataset is the capped one ``make_inputs`` defaults to, the same shape the numerical corpus test
uses, so the whole corpus fits in a normal test run.
"""

import numpy as np
import pytest

from tests.corpus.polybench import polybench as PB
from tests.corpus.polybench import polybench_numpy as PN

KERNELS = [k.name for k in PB.collect()]
#: The SDFG reference of a LAPACK kernel links LAPACKE.
SDFG_REFERENCE_CASES = [pytest.param(k.name, marks=pytest.mark.lapack if k.lapack else ()) for k in PB.collect()]


def worst_difference(ref: dict[str, np.ndarray], got: dict[str, np.ndarray]) -> tuple[str, float, float]:
    """``(array name, max abs diff, max rel diff)`` over the compared arrays -- so a failure names
    WHICH output diverged and by how much, instead of only that something did."""
    name, worst_abs, worst_rel = "", 0.0, 0.0
    for key, expected in ref.items():
        a = np.asarray(expected, dtype=np.float64)
        b = np.asarray(got[key], dtype=np.float64)
        if a.shape != b.shape or not a.size:
            continue
        diff = np.abs(a - b)
        rel = float(np.max(diff / np.maximum(np.abs(a), np.finfo(np.float64).tiny)))
        if rel > worst_rel:
            name, worst_rel = key, rel
        worst_abs = max(worst_abs, float(np.max(diff)))
    return name, worst_abs, worst_rel


def moved(before: dict[str, np.ndarray], after: dict[str, np.ndarray]) -> set[str]:
    """Names of the arrays a reference actually wrote a different value into."""
    return {n for n in before if not np.array_equal(np.asarray(before[n]), np.asarray(after[n]))}


def test_every_kernel_has_a_numpy_reference():
    assert sorted(PN.REFERENCES) == sorted(KERNELS)
    assert sorted(PN.VECTORIZATION) == sorted(KERNELS)


@pytest.mark.parametrize("name", SDFG_REFERENCE_CASES)
def test_numpy_reference_matches_sdfg_reference(name):
    kernel = PB.collect(name)[0]
    arrays, psize = PB.make_inputs(kernel)
    sdfg_ref = PB.reference(kernel, arrays, psize)
    numpy_ref = PB.numpy_reference(kernel, arrays, psize)
    assert moved(arrays, numpy_ref) == moved(arrays, sdfg_ref), (
        f"{name}: the numpy reference moved {sorted(moved(arrays, numpy_ref))} but the SDFG reference "
        f"moved {sorted(moved(arrays, sdfg_ref))}; a reference that writes nothing certifies nothing"
    )
    worst, abs_diff, rel_diff = worst_difference(sdfg_ref, numpy_ref)
    assert PB.outputs_match(sdfg_ref, numpy_ref), (
        f"{name}: numpy reference disagrees with the untransformed-SDFG reference on {worst!r} "
        f"(max abs {abs_diff:.3e}, max rel {rel_diff:.3e}); one of the two is wrong"
    )


if __name__ == "__main__":
    test_every_kernel_has_a_numpy_reference()
    for kernel_name in KERNELS:
        test_numpy_reference_matches_sdfg_reference(kernel_name)
