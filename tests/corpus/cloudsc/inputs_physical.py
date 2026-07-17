# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Physically-plausible CloudSC inputs, ported to the pure-Python ``cloudsc_py`` program.

Ported from dace-fortran ``tests/cloudsc/full/_registries.py::get_inputs_physical``, which samples
every field from a physical distribution (ECMWF/IFS canonical constants, a mid-latitude temperature
profile, strictly-monotone pressures, Magnus saturation-humidity ceilings) so the kernel runs in
its valid regime. Its own test records 0 mismatched cells and 0 NaN/Inf over 12 seeds -- the
physical generator, not the earlier uniform-random one that drove non-physical divergences.

Two shape differences from the Fortran source, handled here:

* The Fortran arrays are ``(KLON, KLEV[, NCLV], NBLOCKS)`` in Fortran order; ``cloudsc_py`` is
  ``(klev, klon)`` / ``(nclv, klev, klon)`` / ``(klev+1, klon)`` with NO block dimension. This is
  NBLOCKS = 1 with the block axis dropped and the column/level axes in the Python order.
* Field names are uppercase in Fortran; ``cloudsc_py`` uses lowercase. The physical rule is keyed
  on ``name.upper()`` so the two stay a single source of truth.

Marshaling is by DESCRIPTOR TYPE: a ``dace.data.Scalar`` crosses the ABI as a scalar value; a
length-1 ``Array`` (shape ``(1,)``) crosses as a 1-element array. Keying on ``size == 1`` conflates
the two and hands an array to a scalar argument (the ``ydcst_rcpd`` bug).
"""
from typing import Dict, Union

import numpy as np

import dace

from tests.corpus.cloudsc.generate_data_for_cloudsc import (CLOUDSC_CONSTANTS, CLOUDSC_INPUT_RANGES,
                                                            CLOUDSC_INT_RANGES, CLOUDSC_SYMBOLS)

#: Water triple-point temperature [K]; Magnus reference.
_RTT = 273.16

#: 1-based species slots in the ``cld`` dimension (NCLDQL/I/R/S/V from the IFS).
_NCLDQL, _NCLDQI, _NCLDQR, _NCLDQS, _NCLDQV = 1, 2, 3, 4, 5


def _pressure_half_levels(klev: int) -> np.ndarray:
    """Monotone-increasing half-level pressure [Pa], ~2 hPa at model top to ~1010 hPa at surface."""
    p_top, p_sfc = 2.0e2, 1.01e5
    eta = np.linspace(0.0, 1.0, klev + 1)
    return p_top + (p_sfc - p_top) * (0.85 * eta**3 + 0.15 * eta)


def _temperature_profile(klev: int, rng: np.random.Generator) -> np.ndarray:
    """Mid-latitude temperature [K] over full levels, clipped to [180, 320]."""
    eta = np.linspace(0.0, 1.0, klev)
    t_strat, t_sfc, trop = 215.0, 293.0, 0.28
    prof = np.where(eta < trop, t_strat + (220.0 - t_strat) * (eta / trop),
                    220.0 + (t_sfc - 220.0) * ((eta - trop) / (1.0 - trop)))
    prof = prof + rng.normal(0.0, 0.5, klev)
    return np.clip(prof, 180.0, 320.0)


def _level_axis(sym_shape) -> int:
    """The axis whose SYMBOLIC dimension is the vertical one (``klev`` or ``klev + 1``), or -1.

    Keyed on the symbol NAME, not the instantiated size: the corpus default is ``klev == klon == 32``,
    so a size-based check cannot tell the column axis from the level axis (and picks the wrong one for
    ``paph``'s ``(klon, klev+1)``, a broadcast mismatch).
    """
    for ax, dim in enumerate(sym_shape):
        if 'klev' in str(dim).lower():
            return ax
    return -1


def _resolve_symbols(sdfg: dace.SDFG) -> Dict[str, int]:
    """Concrete values for every free symbol the SDFG needs, so any cloudsc-family SDFG (the Python
    program, the CPU Fortran ``cloudscouter``, the GPU SCC kernel) is runnable.

    The three frontends differ in their free symbols: the CPU Fortran adds ``nblocks``/``kfldx``; the
    GPU SCC kernel adds ``jl`` (the column this thread handles) and integer ``yrecldp_*`` config
    fields. This is a TRANSFORM-PRESERVATION test -- the reference is the untransformed SDFG's own
    output on these inputs -- so a config symbol needs a VALID value (in range, no div-by-zero), not
    a physically canonical one; both the reference and every transformed variant use the same value.

    :param sdfg: the SDFG whose free symbols are resolved.
    :returns: ``{symbol_name: int}`` covering every free symbol.
    """
    klev = CLOUDSC_SYMBOLS['klev']
    klon = CLOUDSC_SYMBOLS['klon']
    known = {**CLOUDSC_SYMBOLS, 'nblocks': 1, 'kfldx': 1, 'jl': 1}
    resolved: Dict[str, int] = {}
    for sym in sdfg.free_symbols:
        name = str(sym)
        if name in known:
            resolved[name] = known[name]
        elif name.endswith('ncldtop'):
            resolved[name] = 1              # top cloud level, in [1, klev]
        elif name.endswith('nssopt') or name.endswith('nshapep') or name.endswith('nshapeq'):
            resolved[name] = 0              # microphysics option switches
        elif 'ncld' in name or name.endswith('nclv'):
            resolved[name] = min(5, klev)   # species-count-like integer config
        else:
            resolved[name] = 1              # any remaining integer config symbol
    resolved.setdefault('klev', klev)
    resolved.setdefault('klon', klon)
    return resolved


def _broadcast_profile(profile: np.ndarray, shape, level_ax: int) -> np.ndarray:
    """Tile a per-level 1-D ``profile`` across ``shape``, laid along ``level_ax``."""
    view = profile.reshape([len(profile) if ax == level_ax else 1 for ax in range(len(shape))])
    return np.asfortranarray(np.broadcast_to(view, shape).astype(np.float64))


def get_inputs_physical_py(sdfg: dace.SDFG, seed: int = 0) -> Dict[str, Union[np.ndarray, int, float]]:
    """Build one physically-plausible input set for ``cloudsc_py`` (NBLOCKS = 1).

    :param sdfg: the CloudSC SDFG whose non-transient arrays are filled.
    :param seed: RNG seed.
    :returns: kwargs dict keyed by the program's own (lowercase) argument names.
    """
    import sympy

    rng = np.random.default_rng(seed)
    symbols = _resolve_symbols(sdfg)
    klev = symbols['klev']
    klon = symbols['klon']

    def inst(dim) -> int:
        """Instantiate one shape dimension against the resolved symbols (handles ``klev + 1`` etc)."""
        if isinstance(dim, int):
            return dim
        return int(sympy.sympify(str(dim)).subs(symbols))

    paph_prof = _pressure_half_levels(klev)                 # (klev+1,)
    pap_prof = 0.5 * (paph_prof[:-1] + paph_prof[1:])       # (klev,)
    t_prof = _temperature_profile(klev, rng)                # (klev,)
    esat = 611.21 * np.exp(17.502 * (t_prof - _RTT) / (t_prof - 32.19))
    qsat = 0.622 * esat / np.maximum(pap_prof - esat, 1.0)  # (klev,) saturation specific humidity

    def field(shape, lo, hi):
        return np.asfortranarray((lo + (hi - lo) * rng.random(shape)).astype(np.float64))

    def qsat_scaled(shape, level_ax, lo, hi):
        rh = rng.uniform(lo, hi, shape)
        prof = qsat.reshape([klev if ax == level_ax else 1 for ax in range(len(shape))])
        return np.asfortranarray((rh * prof).astype(np.float64))

    inputs: Dict[str, Union[np.ndarray, int, float]] = {}
    for name, desc in sdfg.arrays.items():
        if desc.transient:
            continue
        shape = tuple(inst(d) for d in desc.shape)
        upper = name.upper()
        lax = _level_axis(desc.shape)
        is_int = 'int' in str(desc.dtype)
        is_bool = desc.dtype == dace.bool

        if name in CLOUDSC_SYMBOLS:
            arr = np.zeros(shape, dtype=np.int32, order='F')
            arr.flat[0] = CLOUDSC_SYMBOLS[name]
            inputs[name] = arr
        elif upper == 'PT':
            inputs[name] = _broadcast_profile(t_prof, shape, lax)
        elif upper == 'PAP':
            inputs[name] = _broadcast_profile(pap_prof, shape, lax)
        elif upper == 'PAPH':
            inputs[name] = _broadcast_profile(paph_prof, shape, lax)
        elif upper == 'PQ':
            inputs[name] = qsat_scaled(shape, lax, 0.2, 0.9)
        elif upper == 'PA':
            inputs[name] = np.asfortranarray(np.clip(rng.beta(1.5, 4.0, shape), 0.0, 1.0))
        elif upper == 'PCLV':
            pclv = np.zeros(shape, dtype=np.float64)
            sp_ax = next(ax for ax, sz in enumerate(shape) if sz == CLOUDSC_SYMBOLS['nclv'])
            sl = [slice(None)] * len(shape)
            for sp, scale in ((_NCLDQL - 1, 3.0e-4), (_NCLDQI - 1, 2.0e-4), (_NCLDQR - 1, 1.0e-4),
                              (_NCLDQS - 1, 1.0e-4)):
                sl[sp_ax] = sp
                pclv[tuple(sl)] = rng.uniform(0.0, scale, pclv[tuple(sl)].shape)
            sl[sp_ax] = _NCLDQV - 1
            vshape = pclv[tuple(sl)].shape
            vsym = [d for ax,d in enumerate(desc.shape) if ax != sp_ax]
            pclv[tuple(sl)] = qsat_scaled(vshape, _level_axis(vsym), 0.2, 0.9)
            inputs[name] = np.asfortranarray(pclv)
        elif upper == 'PSUPSAT':
            inputs[name] = field(shape, 0.0, 1.0e-5)
        elif upper in ('PVFA', 'PDYNA'):
            inputs[name] = field(shape, -1.0e-5, 1.0e-5)
        elif upper in ('PVFL', 'PVFI', 'PDYNL', 'PDYNI'):
            inputs[name] = field(shape, -1.0e-6, 1.0e-6)
        elif upper in ('PHRSW', 'PHRLW'):
            inputs[name] = field(shape, -6.0e-5, 6.0e-5)
        elif upper == 'PVERVEL':
            inputs[name] = field(shape, -1.0, 1.0)
        elif upper == 'PLSM':
            inputs[name] = np.asfortranarray(rng.integers(0, 2, shape).astype(np.float64))
        elif upper == 'LDCUM':
            inputs[name] = np.asfortranarray(rng.integers(0, 2, shape).astype(np.int32))
        elif upper == 'KTYPE':
            inputs[name] = np.asfortranarray(rng.integers(0, 3, shape).astype(np.int32))
        elif upper in ('PLU', 'PLUDE'):
            inputs[name] = field(shape, 0.0, 5.0e-4)
        elif upper == 'PSNDE':
            inputs[name] = field(shape, 0.0, 1.0e-4)
        elif upper in ('PMFU', 'PMFD'):
            sgn = 1.0 if upper == 'PMFU' else -1.0
            inputs[name] = np.asfortranarray(sgn * rng.uniform(0.0, 0.5, shape))
        elif upper == 'PCCN':
            inputs[name] = field(shape, 20.0e6, 300.0e6)
        elif upper == 'PNICE':
            inputs[name] = field(shape, 1.0e3, 1.0e5)
        elif upper in ('PLCRIT_AER', 'PICRIT_AER'):
            inputs[name] = field(shape, 1.0e-6, 1.0e-3)
        elif upper == 'PRE_ICE':
            inputs[name] = field(shape, 1.0e-5, 1.0e-4)
        elif name.startswith('tendency_'):
            inputs[name] = field(shape, -1.0e-7, 1.0e-7)
        elif name == 'ptsphy':
            inputs[name] = np.full(shape, 50.0, dtype=np.float64, order='F')  # physics timestep [s]
        elif name == 'kidia':
            inputs[name] = np.full(shape, 1, dtype=np.int32, order='F')
        elif name == 'kfdia':
            inputs[name] = np.full(shape, klon, dtype=np.int32, order='F')
        elif name in CLOUDSC_CONSTANTS:
            v = CLOUDSC_CONSTANTS[name]
            inputs[name] = np.full(shape, int(v) if is_int else v, dtype=np.int32 if is_int else np.float64, order='F')
        elif is_bool:
            inputs[name] = np.asfortranarray(rng.integers(0, 2, shape).astype(np.bool_))
        elif is_int:
            lo, hi = CLOUDSC_INT_RANGES.get(name, (1, 1))
            inputs[name] = rng.integers(lo, hi + 1, size=shape).astype(np.int32, order='F')
        else:
            rng_win = CLOUDSC_INPUT_RANGES.get(name)
            inputs[name] = field(shape, *rng_win) if rng_win else np.zeros(shape, dtype=np.float64, order='F')

    # Never hand an exact 0.0 to a continuous float field -- nudge to a tiny same-signed epsilon.
    zero_eps = 1.0e-12
    for k, v in inputs.items():
        if isinstance(v, np.ndarray) and v.dtype.kind == 'f' and k.upper() != 'PLSM':
            zero = np.abs(v) < zero_eps
            if zero.any():
                sgn = np.sign(v)
                sgn[sgn == 0.0] = 1.0
                v[zero] = sgn[zero] * zero_eps

    # Marshal by DESCRIPTOR TYPE: Scalar -> scalar value; Array (even shape (1,)) -> ndarray.
    marshalled: Dict[str, Union[np.ndarray, int, float]] = {}
    for name, desc in sdfg.arrays.items():
        if desc.transient or name not in inputs:
            continue
        value = inputs[name]
        if isinstance(desc, dace.data.Scalar):
            scalar = np.asarray(value).flat[0]
            marshalled[name] = int(scalar) if 'int' in str(desc.dtype) else float(scalar)
        else:
            marshalled[name] = np.asfortranarray(value)

    marshalled.update({k: v for k, v in symbols.items() if k not in marshalled})
    return marshalled
