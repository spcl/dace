# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""MPR against a whole application: CloudSC.

The unit tests in this directory each render one pattern. This one renders the ECMWF cloud
microphysics scheme -- thousands of states, every control-flow construct, hundreds of arrays -- and
asserts the same one sentence: the emitted C++ builds with a bare host compiler and reproduces the
SDFG's numbers.

Why a whole application is a different test from the sum of the small ones. A refusal MPR should
raise (a state-struct access, a runtime symbol, a lifetime that cannot be demoted) surfaces on the
first construct that needs it, and a kernel built from three maps never reaches most of them. The
value here is coverage of MPR's REFUSALS as much as of its output: whatever CloudSC contains,
either it renders or the failure names the construct.

Comparison is against the SDFG's own compiled output on identical physical inputs, not against a
stored baseline: MPR's contract is stated relative to the SDFG, so the SDFG is the oracle. Both
sides run single-threaded from sequential schedules under IEEE flags, which is what makes a
machine-precision tolerance meaningful -- a parallel reduction reorders floating-point accumulation
run to run, and a tolerance loose enough to absorb that would also absorb a real defect.

There is deliberately no parallel-form test here. CloudSC as built has ZERO maps -- it is all
LoopRegions -- so an assertion that the rendering carries ``#pragma omp parallel for`` would be
testing a property the input never had. Giving it one costs a 49-minute ``canonicalize`` run for
129 maps, which does not belong on top of the numeric leg; ``test_emission.py`` already asserts the
pragma on SDFGs whose maps exist.

Marked ``integration``: building CloudSC is minutes and the rendered translation unit is large. It
is NOT skipped -- on a box with a compiler it is expected to run and pass.
"""
import contextlib
import copy
import ctypes
from typing import Any, Dict

import numpy as np
import pytest

import dace
from dace import data as dt
from dace.codegen.mpr import render

from tests.codegen.mpr.conftest import assert_standalone, call_standalone, compile_standalone
from tests.corpus.cloudsc.generate_data_for_cloudsc import (IEEE_CPU_ARGS, build_cloudsc_sdfg, compare_outputs,
                                                            generate_cloudsc_inputs, make_sequential)

#: Machine precision. Both sides are sequential IEEE builds of the same computation, so they agree
#: bit-for-bit; this is the CloudSC harness's own established criterion, not a tolerance chosen to
#: make a discrepancy pass.
RTOL = ATOL = 1e-15

#: Flags added to the MPR build so it matches the reference's floating-point regime. The reference
#: compiles under :data:`IEEE_CPU_ARGS`; without ``-ffp-contract=off`` the host compiler is free to
#: fuse a multiply-add here and not there, and the two sides would differ by an ulp for a reason
#: that has nothing to do with MPR.
IEEE_FLAGS = ('-fno-fast-math', '-ffp-contract=off')


@contextlib.contextmanager
def ieee_build():
    """Compile the reference SDFG with deterministic IEEE flags, restoring the prior setting."""
    saved = dace.Config.get('compiler', 'cpu', 'args')
    try:
        dace.Config.set('compiler', 'cpu', 'args', value=IEEE_CPU_ARGS)
        yield
    finally:
        dace.Config.set('compiler', 'cpu', 'args', value=saved)


def entry_arguments(sdfg: dace.SDFG, values: Dict[str, Any]) -> Dict[str, Any]:
    """``values`` restricted and shaped to what ``sdfg``'s MPR entry point takes.

    Two adjustments, both of which would otherwise be silent. The CloudSC input generator passes a
    length-1 array as a plain number, which ctypes cannot write through -- so it is widened back to
    the array the descriptor describes. And an argument the generator did not produce is raised
    here rather than defaulted: the entry point would read uninitialized memory.
    """
    arguments: Dict[str, Any] = {}
    missing = []
    for name, desc in sdfg.arglist().items():
        if name not in values:
            missing.append(name)
            continue
        value = values[name]
        if not isinstance(desc, dt.Scalar) and np.isscalar(value):
            value = np.array([value], dtype=desc.dtype.as_numpy_dtype())
            values[name] = value  # the caller compares this buffer afterwards, so keep the widened one
        arguments[name] = value
    assert not missing, (f'the CloudSC input generator produced no value for {missing}, which the MPR entry '
                         'point takes; it would run on uninitialized memory')
    return arguments


@pytest.mark.integration
def test_cloudsc_renders_standalone_and_reproduces_the_sdfg():
    """Render CloudSC, build it with no include path, and compare every output array."""
    reference = build_cloudsc_sdfg(simplify=False)
    make_sequential(reference)

    # Row-major already, and asserted to be by ``cloudsc_input_data_test.py``: both sides read the
    # buffer through a raw pointer with the descriptor's strides, and ``call_standalone`` refuses
    # anything else rather than copying, which would send the kernel's writes into the copy.
    reference_values = generate_cloudsc_inputs(reference, seed=0)
    pristine = copy.deepcopy(reference_values)
    with ieee_build():
        reference(**reference_values)

    rendering = render(copy.deepcopy(reference), validate=False)
    assert_standalone(rendering.code, 'cloudsc')
    assert 'extern "C" void %s(' % reference.name in rendering.code

    library = ctypes.CDLL(compile_standalone(rendering.code, 'cloudsc', extra_flags=IEEE_FLAGS))
    mpr_values = copy.deepcopy(pristine)
    call_standalone(library, rendering.sdfg, entry_arguments(rendering.sdfg, mpr_values))

    report = compare_outputs(reference_values, mpr_values, rtol=RTOL, atol=ATOL)
    assert report, 'nothing was compared -- the two runs share no array, so this asserts nothing'
    mismatched = {name: (abs_err, rel_err) for name, (abs_err, rel_err, ok) in report.items() if not ok}
    assert not mismatched, (
        'MPR output diverges from the SDFG on ' +
        ', '.join(f'{name} (abs={abs_err:.3e} rel={rel_err:.3e})'
                  for name, (abs_err, rel_err) in sorted(mismatched.items(), key=lambda kv: -kv[1][1])[:5]))


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'integration'])
