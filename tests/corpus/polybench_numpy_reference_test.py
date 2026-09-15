# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The polybench numpy reference must agree with the untransformed-SDFG reference.

``polybench.numpy_reference`` is now BOTH a correctness oracle and the timing denominator for the
polybench half of a "vs parallel numpy" figure. A numpy formulation that quietly computes something
else would invalidate every polybench result at once -- including the recorded 30/30
value-preservation run -- so each kernel is pinned against the reference the corpus has always used:
the untransformed SDFG, on identical inputs.

Three things are checked, and they fail for different reasons:

* every collected kernel HAS a numpy reference and a declared vectorization class (a kernel added
  without one would otherwise silently fall out of the figure),
* the numpy reference agrees with the SDFG reference within the corpus's own dtype-aware tolerance
  (a disagreement is a real divergence in one of the two, never a reason to loosen the tolerance),
* the numpy reference moves EXACTLY the arrays the SDFG reference moves (a reference that quietly
  computes nothing while the kernel computes would otherwise pass the comparison above by turning
  it into ``inputs == inputs``), and ``numpy_call`` + ``restore_inputs`` reproduce the same result
  on a second invocation -- which is what makes it timeable over many repetitions on pristine
  inputs.

NOTE: the second check is written as "same arrays moved as the SDFG reference", not "at least one
array moved", because two polybench kernels are IDENTITY MAPS on their own ``init_array`` input:
``heat_3d`` (the initial condition is linear in i/j/k, so the 7-point second-difference stencil is
identically zero) and ``seidel_2d`` (the 9-point average of its bilinear initial condition
reproduces itself, to rounding). Both references correctly reproduce that, and pinning the numpy
side to the SDFG side is the strongest claim this file can honestly make. Whether those two kernels
belong in a speedup figure at all is a corpus question, not a reference question.

The dataset is the capped one ``make_inputs`` defaults to, the same shape the numerical corpus test
uses, so the whole corpus fits in a normal test run.
"""
from typing import Dict, Set, Tuple

import numpy as np
import pytest

from tests.corpus.polybench import polybench as PB
from tests.corpus.polybench import polybench_numpy as PN

KERNELS = [k.name for k in PB.collect()]


def worst_difference(ref: Dict[str, np.ndarray], got: Dict[str, np.ndarray]) -> Tuple[str, float, float]:
    """``(array name, max abs diff, max rel diff)`` over the compared arrays -- so a failure names
    WHICH output diverged and by how much, instead of only that something did."""
    name, worst_abs, worst_rel = '', 0.0, 0.0
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


def moved(before: Dict[str, np.ndarray], after: Dict[str, np.ndarray]) -> Set[str]:
    """Names of the arrays a reference actually wrote a different value into."""
    return {n for n in before if not np.array_equal(np.asarray(before[n]), np.asarray(after[n]))}


def test_every_kernel_has_a_numpy_reference():
    assert sorted(PN.REFERENCES) == sorted(KERNELS)
    assert sorted(PN.VECTORIZATION) == sorted(KERNELS)


@pytest.mark.parametrize('name', KERNELS)
def test_numpy_reference_matches_sdfg_reference(name):
    kernel = PB.collect(name)[0]
    arrays, psize = PB.make_inputs(kernel)
    sdfg_ref = PB.reference(kernel, arrays, psize)
    numpy_ref = PB.numpy_reference(kernel, arrays, psize)
    assert moved(arrays, numpy_ref) == moved(
        arrays,
        sdfg_ref), (f'{name}: the numpy reference moved {sorted(moved(arrays, numpy_ref))} but the SDFG reference '
                    f'moved {sorted(moved(arrays, sdfg_ref))}; a reference that writes nothing certifies nothing')
    worst, abs_diff, rel_diff = worst_difference(sdfg_ref, numpy_ref)
    assert PB.outputs_match(
        sdfg_ref, numpy_ref), (f'{name}: numpy reference disagrees with the untransformed-SDFG reference on {worst!r} '
                               f'(max abs {abs_diff:.3e}, max rel {rel_diff:.3e}); one of the two is wrong')


@pytest.mark.parametrize('name', KERNELS)
def test_numpy_call_repeats_on_pristine_inputs(name):
    """A timed denominator runs the reference many times. The polybench kernels write their
    inputs, so without ``restore_inputs`` repetition 2 would compute from repetition 1's output --
    different data, different time, and the number would not mean what the figure says it does."""
    kernel = PB.collect(name)[0]
    arrays, psize = PB.make_inputs(kernel)
    fn, kwargs = PB.numpy_call(kernel, arrays, psize)
    fn(**kwargs)
    first = {n: np.asarray(v).copy() for n, v in kwargs.items() if isinstance(v, np.ndarray)}
    PB.restore_inputs(kwargs, arrays)
    fn(**kwargs)
    second = {n: np.asarray(kwargs[n]) for n in first}
    assert PB.outputs_match(first, second), \
        f'{name}: the second timed repetition did not reproduce the first; inputs were not pristine'


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
