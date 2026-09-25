# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A physically-conditioned input distribution for the inlined CloudSC kernel.

DRAFT, AND MEASURED AS A NET REGRESSION -- referenced by no test on purpose. It fixes what it aimed
at: ``pfplsl`` and ``pfhpsl`` stop being identically zero, and the worst amplifier ``pfsqrf`` falls
5.7x. It breaks three other things, and until those are fixed the existing generator is the better
one. (1) The four ``pfcq*ng`` arrays go ALL-ZERO: they accumulate the mass the ``< rlmin`` guards
dump, and a margin that floors condensate above ``rlmin`` removes every dump -- the margin has to
STRADDLE ``rlmin``, not clear it. (2) ``tendency_loc_t`` amplification reaches 1.1e14 because purely
relative ``tendency_tmp_*`` increments create cells whose contributions cancel to ~1e-15; they need
an absolute floor. (3) ``prainfrac_toprfz`` is STILL identically zero -- the rain/snow split leaves
the cell above the 273.16 K crossing pure snow, which is exactly where the detector reads ``qr``.

Also worth knowing before trusting any tolerance on this kernel: the UN-TRANSFORMED reference
already amplifies a 1e-13 input perturbation by 5.5e4 on ``pfsqrf``. A legal reassociation at 1e-14
moves that output by 5e-10, past the 1e-10 the vectorize leg asserts. If that survives conditioning,
the tolerance is measuring the kernel's conditioning rather than the transform.

``generate_data_for_cloudsc.generate_cloudsc_inputs`` draws every field uniformly inside the
``[min, max]`` bounding box the dwarf reference dataset happens to occupy. A bounding box of a whole
three-dimensional field is not a pointwise constraint, and drawing from it independently per cell
destroys every correlation the kernel's branches key off:

* ``pt`` never exceeds ``267.5 K``, so ``ztp1 > ydcst_rtt`` (``273.16``) is FALSE everywhere. The
  melting branch (``cloudsc.py:1006``), the freezing-level detection that writes
  ``prainfrac_toprfz`` (``cloudsc.py:1025``) and the whole liquid-precipitation path are dead code,
  and ``pfplsl``, ``pfhpsl`` and ``prainfrac_toprfz`` come back IDENTICALLY ZERO -- an output array
  with no nonzero entry carries no relative-error signal at all.
* ``pt`` also has no vertical structure, so a cell can be at ``267 K`` and ``272 Pa`` at once.
  ``zqsmix`` there is ``0.7 kg/kg`` and hits the ``min(..., 0.5)`` saturation cap
  (``cloudsc.py:413``), which pins the derivative to zero.
* ``pq`` is drawn independently of ``pt`` and ``pap``, so relative humidity ranges over six orders
  of magnitude and supersaturation appears in the stratosphere.
* ``ptsphy * tendency_tmp_a`` reaches ``1.0`` while ``pa`` is at most ``1.0``, so ``za`` goes
  negative in a large fraction of cells and the ``za < yrecldp_ramin`` guard (``cloudsc.py:371``)
  wipes the condensate. ``ptsphy * tendency_tmp_cld`` is likewise larger than ``pclv`` itself.
* ``ldcum``, ``ktype``, ``pmfu``, ``plu`` and ``plude`` are drawn independently, so convective
  detrainment happens in columns that have no convection.

This module builds the same fields from a vertical profile instead. Temperature follows the US
Standard Atmosphere 1976 anchors interpolated in ``log(p)``, humidity is a relative humidity times
the kernel's OWN saturation function, condensate lives in cloud slabs with the phase split taken
from the kernel's own ``zfoealfa``, and every ``tendency_tmp_*`` is a bounded RELATIVE increment of
the field it increments. Values are additionally repelled from the kernel's discrete thresholds, so
a floating-point reassociation cannot flip a branch.

Deterministic: everything derives from ``seed`` through one ``numpy.random.default_rng``. Every
array is built ROW-MAJOR, the standing invariant for CloudSC data.
"""
from typing import Dict, List, Tuple, Union

import numpy as np
import sympy

import dace

from tests.corpus.cloudsc.generate_data_for_cloudsc import (CLOUDSC_CONSTANTS, CLOUDSC_INPUT_RANGES, CLOUDSC_SYMBOLS,
                                                            pressure_profile)

Field = Dict[str, np.ndarray]
Inputs = Dict[str, Union[np.ndarray, int, float]]

#: US Standard Atmosphere 1976 layer boundaries, ``(pressure [Pa], temperature [K])``, from the
#: surface up to 84.852 km. Temperature between anchors is linear in ``log(p)``.
STANDARD_ATMOSPHERE: Tuple[Tuple[float, float], ...] = (
    (101325.0, 288.15),
    (22632.1, 216.65),
    (5474.89, 216.65),
    (868.019, 228.65),
    (110.906, 270.65),
    (66.9389, 270.65),
    (3.95642, 214.65),
    (0.37338, 186.87),
)

#: Pressure of the 11 km anchor: the profile's tropopause. Cloud slabs and the per-column surface
#: temperature anomaly live below it.
TROPOPAUSE_PRESSURE: float = 22632.1

#: Every nonzero condensate is at least this multiple of ``yrecldp_rlmin``, so the
#: ``< rlmin`` dump branches (``cloudsc.py:371``, ``:388``) are decided by physics, not by noise.
CONDENSATE_MARGIN: float = 100.0

#: Minimum distance, in kelvin, between any generated temperature and any phase threshold the
#: kernel compares against.
PHASE_MARGIN: float = 1e-3

#: Minimum cloud fraction inside a cloud slab. Far above ``yrecldp_ramin`` (``1e-8``) even after the
#: largest ``tendency_tmp_a`` increment.
MIN_CLOUD_FRACTION: float = 0.05

#: Largest relative change any ``tendency_tmp_*`` may make to the field it increments over one
#: ``ptsphy``. Keeps ``zqx`` and ``za`` on the same side of every guard as the field they came from.
MAX_RELATIVE_INCREMENT: float = 0.15


def instantiate_dim(dim: object) -> int:
    """Resolve one declared shape dimension to a concrete size via :data:`CLOUDSC_SYMBOLS`."""
    if isinstance(dim, (int, sympy.Number)):
        return int(dim)
    if isinstance(dim, dace.symbol):
        return CLOUDSC_SYMBOLS[str(dim)]
    return int(sympy.sympify(dim).subs(CLOUDSC_SYMBOLS))


def liquid_fraction(t: np.ndarray) -> np.ndarray:
    """The kernel's ``zfoealfa`` (``cloudsc.py:401``): the liquid share of condensate at ``t``."""
    rtice = CLOUDSC_CONSTANTS['ydthf_rtice']
    rtwat = CLOUDSC_CONSTANTS['ydthf_rtwat']
    scale = CLOUDSC_CONSTANTS['ydthf_rtwat_rtice_r']
    return np.minimum(1.0, ((np.maximum(rtice, np.minimum(rtwat, t)) - rtice) * scale)**2)


def saturation_mixing_ratio(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    """The kernel's ``zqsmix`` (``cloudsc.py:405``): mixed-phase saturation specific humidity."""
    alfa = liquid_fraction(t)
    rtt = CLOUDSC_CONSTANTS['ydcst_rtt']
    over_water = np.exp(CLOUDSC_CONSTANTS['ydthf_r3les'] * (t - rtt) / (t - CLOUDSC_CONSTANTS['ydthf_r4les']))
    over_ice = np.exp(CLOUDSC_CONSTANTS['ydthf_r3ies'] * (t - rtt) / (t - CLOUDSC_CONSTANTS['ydthf_r4ies']))
    vapor = np.minimum(CLOUDSC_CONSTANTS['ydthf_r2es'] * (alfa * over_water + (1.0 - alfa) * over_ice) / p, 0.5)
    return vapor / (1.0 - CLOUDSC_CONSTANTS['ydcst_retv'] * vapor)


def phase_thresholds() -> Tuple[float, ...]:
    """Temperatures the kernel branches on: melting, homogeneous freezing, and the mixed-phase band."""
    return (CLOUDSC_CONSTANTS['ydcst_rtt'], CLOUDSC_CONSTANTS['yrecldp_rthomo'], CLOUDSC_CONSTANTS['ydthf_rtice'],
            CLOUDSC_CONSTANTS['ydthf_rtwat'])


def repel_from_thresholds(values: np.ndarray, thresholds: Tuple[float, ...], margin: float) -> np.ndarray:
    """Push every element at least ``margin`` away from every threshold, keeping which side it is on."""
    out = np.array(values, dtype=np.float64, order='C')
    for threshold in thresholds:
        delta = out - threshold
        near = np.abs(delta) < margin
        out[near] = threshold + margin * np.where(delta[near] < 0.0, -1.0, 1.0)
    return out


def standard_temperature(p: np.ndarray) -> np.ndarray:
    """US Standard Atmosphere temperature at pressure ``p``, interpolated linearly in ``log(p)``."""
    anchors = np.array(STANDARD_ATMOSPHERE, dtype=np.float64)
    order = np.argsort(anchors[:, 0])
    return np.interp(np.log(p), np.log(anchors[order, 0]), anchors[order, 1])


def build_conditioned_fields(klev: int, klon: int, nclv: int, seed: int) -> Field:
    """Every CloudSC input field, built from one vertical profile per column.

    The returned arrays are the PHYSICAL fields, keyed by the kernel's own argument names and
    shaped as the kernel declares them. Pure -- the same ``(klev, klon, nclv, seed)`` always gives
    the same arrays -- so the invariants can be asserted without an SDFG.

    :param klev: number of full levels.
    :param klon: number of columns.
    :param nclv: number of cloud species (5: liquid, ice, rain, snow, vapour).
    :param seed: seed for the single generator every draw comes from.
    :returns: mapping from kernel argument name to a row-major array.
    """
    rng = np.random.default_rng(seed)
    fields: Field = {}

    paph = pressure_profile('paph', [klev + 1, klon])
    pap = pressure_profile('pap', [klev, klon])
    fields['paph'] = paph
    fields['pap'] = pap
    sigma = pap / paph[klev, :][None, :]

    # Temperature: standard profile, plus a per-column surface anomaly that decays to zero at the
    # tropopause, plus a small per-cell jitter so the columns are not smooth copies of one another.
    below = np.clip((pap - TROPOPAUSE_PRESSURE) / (paph[klev, :][None, :] - TROPOPAUSE_PRESSURE), 0.0, 1.0)
    anomaly = rng.uniform(-8.0, 8.0, size=klon)[None, :]
    pt = standard_temperature(pap) + below * anomaly + rng.uniform(-0.35, 0.35, size=(klev, klon))
    pt = repel_from_thresholds(pt, phase_thresholds(), PHASE_MARGIN)
    fields['pt'] = pt

    # Humidity: a relative humidity that falls with height, against the kernel's own saturation
    # function, with a stratospheric floor of ~3 ppmv instead of a saturation-scaled value.
    qsat = saturation_mixing_ratio(pt, pap)
    relative_humidity = np.clip(0.75 * below + 0.10 + 0.20 * rng.standard_normal((klev, klon)), 0.02, 0.90)
    pq = np.maximum(relative_humidity * qsat, 3.0e-6)
    pq = np.where(pap < TROPOPAUSE_PRESSURE, np.minimum(pq, 6.0e-6), pq)
    fields['pq'] = np.array(pq, order='C')

    # Cloud slabs: contiguous tropospheric level ranges with a raised-sine shape, so cloud fraction
    # and condensate rise and fall together instead of being independent noise.
    troposphere = np.nonzero(pap[:, 0] >= TROPOPAUSE_PRESSURE)[0]
    top_level, bottom_level = int(troposphere[0]), int(troposphere[-1])
    cloud_fraction = np.zeros((klev, klon))
    condensate = np.zeros((klev, klon))
    cloud_top = np.full(klon, klev, dtype=np.int64)
    for column in range(klon):
        for _ in range(int(rng.integers(1, 4))):
            depth = int(rng.integers(2, 7))
            start = int(rng.integers(top_level, max(bottom_level - depth, top_level) + 1))
            levels = np.arange(start, min(start + depth, klev))
            shape = np.sin(np.pi * (levels - start + 0.5) / depth)
            peak_fraction = float(rng.uniform(0.3, 0.95))
            peak_condensate = float(np.exp(rng.uniform(np.log(5.0e-6), np.log(4.0e-4))))
            cloud_fraction[levels,
                           column] = np.maximum(cloud_fraction[levels, column],
                                                MIN_CLOUD_FRACTION + (peak_fraction - MIN_CLOUD_FRACTION) * shape)
            condensate[levels, column] = np.maximum(condensate[levels, column], peak_condensate * (0.2 + 0.8 * shape))
            cloud_top[column] = min(cloud_top[column], int(levels[0]))
    fields['pa'] = np.array(np.clip(cloud_fraction, 0.0, 1.0), order='C')

    # Precipitation: present from the highest cloud top downwards, split rain / snow across the
    # melting layer, and never left between zero and the kernel's rlmin dump threshold.
    rlmin = CLOUDSC_CONSTANTS['yrecldp_rlmin']
    floor = CONDENSATE_MARGIN * rlmin
    level_index = np.arange(klev)[:, None]
    falling = level_index >= cloud_top[None, :]
    depth_below = np.maximum(level_index - cloud_top[None, :], 0)
    precipitation = np.where(falling,
                             rng.uniform(2.0e-6, 2.0e-5, size=(klev, klon)) * (1.0 - np.exp(-depth_below / 3.0)), 0.0)
    rain_share = np.clip((pt - (CLOUDSC_CONSTANTS['ydcst_rtt'] - 2.0)) / 4.0, 0.0, 1.0)
    rain = precipitation * rain_share
    snow = precipitation * (1.0 - rain_share)
    rain = np.where(rain < floor, 0.0, rain)
    snow = np.where(snow < floor, 0.0, snow)

    alfa = liquid_fraction(pt)
    liquid = np.where(condensate < floor, 0.0, condensate * alfa)
    ice = np.where(condensate < floor, 0.0, condensate * (1.0 - alfa))
    liquid = np.where(liquid < floor, 0.0, liquid)
    ice = np.where(ice < floor, 0.0, ice)
    # A cloudy cell must keep ql + qi above rlmin, or cloudsc.py:371 dumps it and zeroes za.
    cloudy = (liquid + ice) >= floor
    fields['pa'] = np.array(np.where(cloudy, fields['pa'], 0.0), order='C')

    pclv = np.zeros((nclv, klev, klon))
    pclv[CLOUDSC_SYMBOLS['ncldql'] - 1] = liquid
    pclv[CLOUDSC_SYMBOLS['ncldqi'] - 1] = ice
    pclv[CLOUDSC_SYMBOLS['ncldqr'] - 1] = rain
    pclv[CLOUDSC_SYMBOLS['ncldqs'] - 1] = snow
    fields['pclv'] = np.array(pclv, order='C')

    # Dynamics tendencies: bounded RELATIVE increments, so ptsphy * tendency never moves a field
    # across a guard the field itself is safely on one side of.
    ptsphy = CLOUDSC_CONSTANTS['ptsphy']
    rate = MAX_RELATIVE_INCREMENT / ptsphy
    fields['tendency_tmp_t'] = np.array(rng.uniform(-0.5, 0.5, size=(klev, klon)) / ptsphy, order='C')
    fields['tendency_tmp_q'] = np.array(rate * rng.uniform(-1.0, 1.0, size=(klev, klon)) * fields['pq'], order='C')
    fields['tendency_tmp_a'] = np.array(rate * rng.uniform(-1.0, 1.0, size=(klev, klon)) * fields['pa'], order='C')
    fields['tendency_tmp_cld'] = np.array(rate * rng.uniform(-1.0, 1.0, size=(nclv, klev, klon)) * fields['pclv'],
                                          order='C')

    # Convection: one decision per column drives ldcum, ktype and every convective profile, so
    # detrainment cannot appear in a column that has no convection.
    convective = rng.random(klon) < 0.4
    fields['ldcum'] = np.array(convective.astype(np.int32), order='C')
    fields['ktype'] = np.array(np.where(convective, rng.integers(1, 3, size=klon), 0).astype(np.int32), order='C')
    updraft_shape = np.where(falling, np.sin(np.pi * np.clip(depth_below / max(klev - 1, 1), 0.0, 1.0)), 0.0)
    active = convective[None, :] & falling
    fields['pmfu'] = np.array(np.where(active, CLOUDSC_INPUT_RANGES['pmfu'][1] * updraft_shape, 0.0), order='C')
    fields['plu'] = np.array(np.where(active, CLOUDSC_INPUT_RANGES['plu'][1] * updraft_shape, 0.0), order='C')
    detrainable = np.zeros((klev, klon), dtype=bool)
    detrainable[:-1] = active[:-1] & (fields['plu'][1:] > 0.0)
    plude = np.where(detrainable, CLOUDSC_INPUT_RANGES['plude'][1] * updraft_shape, 0.0)
    fields['plude'] = np.array(np.where(plude < floor, 0.0, plude), order='C')

    # Detrained supersaturation: a sparse source inside convective cloud, not a value in every cell.
    supersaturated = active & cloudy & (rng.random((klev, klon)) < 0.2)
    psupsat = np.where(supersaturated, rng.uniform(1.0e-7, CLOUDSC_INPUT_RANGES['psupsat'][1], size=(klev, klon)), 0.0)
    fields['psupsat'] = np.array(psupsat, order='C')

    # Fields the kernel reads only in the flux diagnostics or the forcing: keep the dwarf's band but
    # give them a vertical envelope, and never an exact zero, so no cumulative flux starts dead.
    envelope = 4.0 * sigma * (1.0 - sigma) + 0.05
    fields['pvervel'] = np.array(CLOUDSC_INPUT_RANGES['pvervel'][1] * envelope * rng.uniform(-1.0, 1.0, (klev, klon)),
                                 order='C')
    fields['phrlw'] = np.array(CLOUDSC_INPUT_RANGES['phrlw'][0] * envelope * rng.uniform(0.05, 1.0, (klev, klon)),
                               order='C')
    fields['pvfl'] = np.array(CLOUDSC_INPUT_RANGES['pvfl'][1] * envelope * rng.uniform(-1.0, 1.0, (klev, klon)),
                              order='C')
    fields['pvfi'] = np.array(CLOUDSC_INPUT_RANGES['pvfi'][1] * envelope * rng.uniform(-1.0, 1.0, (klev, klon)),
                              order='C')
    fields['pvfa'] = np.array(CLOUDSC_INPUT_RANGES['pvfa'][1] * envelope * rng.uniform(-1.0, 1.0, (klev, klon)),
                              order='C')
    return fields


def generate_conditioned_cloudsc_inputs(sdfg: dace.SDFG, seed: int = 0) -> Inputs:
    """A conditioned input set for ``sdfg``, drop-in for ``generate_cloudsc_inputs``.

    Fields :func:`build_conditioned_fields` builds are taken from it; the physical constants and the
    shape / index symbols come from ``generate_data_for_cloudsc`` unchanged; every remaining
    non-transient array (kernel outputs, and the aerosol and downdraft fields the dwarf reference
    has uniformly zero) is zero-initialized.

    :param sdfg: the CloudSC SDFG whose non-transient arrays are filled.
    :param seed: seed for the single generator every draw comes from.
    :returns: a kwargs dict of arrays, scalars and symbol values.
    """
    klev, klon, nclv = CLOUDSC_SYMBOLS['klev'], CLOUDSC_SYMBOLS['klon'], CLOUDSC_SYMBOLS['nclv']
    fields = build_conditioned_fields(klev, klon, nclv, seed)

    arrays: Dict[str, np.ndarray] = {}
    for name, desc in sdfg.arrays.items():
        if desc.transient:
            continue
        dims: List[int] = [instantiate_dim(d) for d in desc.shape]
        is_int = 'int' in str(desc.dtype)
        if name in fields:
            arrays[name] = fields[name]
        elif name in CLOUDSC_CONSTANTS:
            value = CLOUDSC_CONSTANTS[name]
            arrays[name] = np.full(dims,
                                   int(value) if is_int else value,
                                   dtype=np.int32 if is_int else np.float64,
                                   order='C')
        elif is_int:
            data = np.zeros(dims, order='C').astype(np.int32)
            if name in CLOUDSC_SYMBOLS:
                data.flat[0] = CLOUDSC_SYMBOLS[name]
            arrays[name] = data
        else:
            arrays[name] = np.zeros(dims, order='C')

    inputs: Inputs = {name: (data.flat[0] if data.size == 1 else data) for name, data in arrays.items()}
    inputs.update(CLOUDSC_SYMBOLS)
    return inputs
