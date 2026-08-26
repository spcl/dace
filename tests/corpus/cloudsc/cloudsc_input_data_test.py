# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The generated CloudSC input set has to be admissible, in memory and in physics.

Two properties, and the memory one is the one that bit. CloudSC is a port of a Fortran dwarf, so
the generator built every array column-major -- but DaCe describes them row-major (``pt`` is
``(klon, 1)``) and hands the compiled function a bare pointer. Nothing transposes and nothing
complains: the interop layer rejects a numpy VIEW, never a wrong-strided array that owns its
buffer, so column-major memory was read as its own transpose. Every vertical profile became a
horizontal one, and the microphysics ran away to ``5.2e9`` and to NaN on input that is perfectly
sane when read the right way round.

The physical property is pressure. Every other input is drawn uniformly inside its reference
window, which is fine for a field whose elements are independent; pressure is not such a field.
The kernel reads a layer thickness as a difference of consecutive half levels and divides by it, so
a non-monotone profile is inadmissible rather than merely unrealistic -- it drives ``zbeta1``
negative and the kernel then raises a negative number to the power ``0.5777``.

Both faults used to be invisible, because the runtime's ``max`` swallowed the NaN: ``max(x, 0.0)``
returned ``0.0`` for a NaN ``x``. It no longer does -- it keeps the earlier operand, as Python and
``std::`` do -- so bad input now shows up as a NaN output instead of hiding in a plausible one.
That is also why the data had to be fixed rather than the order: an oracle containing NaNs makes
every comparison against it hold trivially, wherever the NaN reached.

Marked ``integration`` automatically by path (``tests/conftest.py``), with the rest of CloudSC.
"""
import numpy as np

from dace.symbolic import evaluate
from tests.corpus.cloudsc.generate_data_for_cloudsc import (CLOUDSC_INPUT_RANGES, CLOUDSC_SYMBOLS, build_cloudsc_sdfg,
                                                            generate_cloudsc_inputs)

#: No CloudSC field is a large number in SI units: the biggest are pressures (~1e5 Pa) and the snow
#: enthalpy flux (~2e3 W/m2). A ceiling three orders above that leaves every legitimate result
#: untouched while still catching a runaway -- the transposed run reached 5.2e9.
PHYSICAL_CEILING = 1e6


def test_every_generated_array_matches_its_descriptor_layout():
    """Generated memory is laid out the way the SDFG says it is.

    Checked against the descriptor's own strides rather than against ``flags.c_contiguous``, so it
    stays true if a descriptor ever declares something else. This is the assertion the transposed
    run needed: DaCe's argument check rejects a view, so it never fired on an owned array whose
    strides were simply the wrong way round, and the kernel read the transpose in silence.
    """
    sdfg = build_cloudsc_sdfg(simplify=False)
    values = generate_cloudsc_inputs(sdfg, seed=0)

    checked = []
    for name, value in sorted(values.items()):
        descriptor = sdfg.arrays.get(name)
        if descriptor is None or not isinstance(value, np.ndarray) or value.ndim == 0:
            continue
        expected = tuple(int(evaluate(stride, CLOUDSC_SYMBOLS)) * value.itemsize for stride in descriptor.strides)
        assert value.strides == expected, \
            f'{name} is laid out {value.strides} but the descriptor declares {expected}; ' \
            'the kernel would read a transpose of it'
        assert value.base is None, f'{name} is a view, which the DaCe argument check rejects'
        checked.append(name)

    assert len(checked) > 20, f'only {len(checked)} arrays were checked, so this asserts almost nothing'


def test_pressure_is_a_monotone_hydrostatic_profile():
    """``paph`` increases from the model top to the surface, and ``pap`` sits between its neighbours.

    Stated as the three properties the kernel actually depends on -- a positive layer thickness, a
    full level inside the half levels bracketing it, and both inside the reference window -- rather
    than as a comparison against the generator's own formula, which would only restate it.
    """
    values = generate_cloudsc_inputs(build_cloudsc_sdfg(simplify=False), seed=0)
    half, full = values['paph'], values['pap']

    thickness = np.diff(half, axis=0)
    assert (thickness > 0).all(), \
        f'{int((thickness <= 0).sum())} layers have a non-positive thickness; the kernel divides by it'
    assert (half[:-1] < full).all() and (full < half[1:]).all(), \
        'a full level lies outside the half levels bracketing it'
    for name, array in (('paph', half), ('pap', full)):
        low, high = CLOUDSC_INPUT_RANGES[name]
        assert low <= array.min() and array.max() <= high, \
            f'{name} leaves its reference window [{low}, {high}]: [{array.min()}, {array.max()}]'


def test_the_reference_run_stays_inside_the_double_precision_envelope():
    """CloudSC on the generated inputs neither overflows nor underflows.

    Three claims, because "finite" alone is too weak to be worth asserting: no non-finite value
    (an overflow to infinity, or an invalid operation), no subnormal magnitude (a gradual underflow
    -- a subnormal result has already lost most of its significand, so a comparison against it
    means much less than it appears to), and nothing above :data:`PHYSICAL_CEILING`, which is what
    a runaway looks like before it reaches infinity.

    The premise is asserted first: inputs that already carried a NaN, a subnormal or a runaway
    would make all three pass without saying anything about the kernel.
    """
    sdfg = build_cloudsc_sdfg(simplify=False)
    values = generate_cloudsc_inputs(sdfg, seed=0)
    tiny = np.finfo(np.float64).tiny

    def offences():
        found = {}
        for name, value in sorted(values.items()):
            if not isinstance(value, np.ndarray) or value.dtype.kind != 'f' or value.size == 0:
                continue
            finite = np.isfinite(value)
            magnitude = np.abs(value[finite])
            nonzero = magnitude[magnitude > 0.0]
            reasons = []
            if not finite.all():
                reasons.append(f'{int((~finite).sum())} non-finite')
            if nonzero.size and nonzero.min() < tiny:
                reasons.append(f'{int((nonzero < tiny).sum())} subnormal')
            if magnitude.size and magnitude.max() > PHYSICAL_CEILING:
                reasons.append(f'|max| = {magnitude.max():.4g}')
            if reasons:
                found[name] = ', '.join(reasons)
        return found

    assert not offences(), 'the generated INPUTS are already outside the envelope'
    sdfg(**values)
    assert not offences(), 'CloudSC left the double-precision envelope'
