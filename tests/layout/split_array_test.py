# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace

import dace
import numpy as np
import pytest
from dace.transformation.layout.split_array import SplitArray
from itertools import product

klev = dace.symbol('klev', dtype=dace.int32)
klon = dace.symbol('klon', dtype=dace.int32)
nclv = dace.symbol('nclv', dtype=dace.int32)
kfdia = dace.symbol('kfdia', dtype=dace.int32)
kidia = dace.symbol('kidia', dtype=dace.int32)
ncldql = dace.symbol('ncldql', dtype=dace.int32)
ncldqi = dace.symbol('ncldqi', dtype=dace.int32)
ncldqr = dace.symbol('ncldqr', dtype=dace.int32)
ncldqs = dace.symbol('ncldqs', dtype=dace.int32)
ncldqv = dace.symbol('ncldqv', dtype=dace.int32)

NAME_ORDER = ["ncldql", "ncldqi", "ncldqr", "ncldqs", "ncldqv"]
NAME_MAP = {i: NAME_ORDER[i] for i in range(5)}

CONDENSE_CST = dict(
    retv=0.6077,
    rtice=250.16,
    rtwat=273.16,
    rtwat_rtice_r=1.0 / (273.16 - 250.16),
    r5alvcp=5.4697e6,
    r4les=35.86,
    r5alscp=6.3147e6,
    r4ies=7.66,
    rthomo=250.0,
    rlmin=1e-12,
)


@dace.program
def condense_kernel(
    za: dace.float64[klev, klon],
    zdqs: dace.float64[klon],
    zqsmix: dace.float64[klev, klon],
    zqv: dace.float64[klev, klon],
    ztp1: dace.float64[klev, klon],
    zsolqa: dace.float64[nclv, nclv, klon],
    zqxfg: dace.float64[nclv, klon],
    retv: dace.float64,
    rtice: dace.float64,
    rtwat: dace.float64,
    rtwat_rtice_r: dace.float64,
    r5alvcp: dace.float64,
    r4les: dace.float64,
    r5alscp: dace.float64,
    r4ies: dace.float64,
    rthomo: dace.float64,
    rlmin: dace.float64,
):
    for jk in range(klev):
        for jl in range(klon):
            if za[jk, jl] > 1e-14:
                if zdqs[jl] <= -rlmin:
                    lc = max(-zdqs[jl], 0.0)
                    af = min(1.0, ((max(rtice, min(rtwat, ztp1[jk, jl])) - rtice) * rtwat_rtice_r)**2)
                    zcor = 1.0 / (1.0 - retv * zqsmix[jk, jl])
                    cdm_full = (zqv[jk, jl] - zqsmix[jk, jl]) / (1.0 + zcor * zqsmix[jk, jl] *
                                                                 (af * r5alvcp / (ztp1[jk, jl] - r4les)**2 +
                                                                  (1.0 - af) * r5alscp / (ztp1[jk, jl] - r4ies)**2))
                    cdm_part = (zqv[jk, jl] - za[jk, jl] * zqsmix[jk, jl]) / za[jk, jl]
                    if za[jk, jl] > 0.99:
                        cdm = cdm_full
                    else:
                        cdm = cdm_part
                    lc = za[jk, jl] * max(min(lc, cdm), 0.0)
                    if lc >= rlmin:
                        if ztp1[jk, jl] > rthomo:
                            zsolqa[ncldqv - 1, ncldql - 1, jl] = zsolqa[ncldqv - 1, ncldql - 1, jl] + lc
                            zsolqa[ncldql - 1, ncldqv - 1, jl] = zsolqa[ncldql - 1, ncldqv - 1, jl] - lc
                            zqxfg[ncldql - 1, jl] = zqxfg[ncldql - 1, jl] + lc
                        else:
                            zsolqa[ncldqv - 1, ncldqi - 1, jl] = zsolqa[ncldqv - 1, ncldqi - 1, jl] + lc
                            zsolqa[ncldqi - 1, ncldqv - 1, jl] = zsolqa[ncldqi - 1, ncldqv - 1, jl] - lc
                            zqxfg[ncldqi - 1, jl] = zqxfg[ncldqi - 1, jl] + lc


def condense_ref(za, zdqs, zqsmix, zqv, ztp1, zsolqa, zqxfg, cst, sym):
    KLEV, KLON = sym['klev'], sym['klon']
    QL, QI, QV = sym['ncldql'] - 1, sym['ncldqi'] - 1, sym['ncldqv'] - 1
    retv = cst['retv']
    rtice = cst['rtice']
    rtwat = cst['rtwat']
    rr = cst['rtwat_rtice_r']
    r5a = cst['r5alvcp']
    r4l = cst['r4les']
    r5s = cst['r5alscp']
    r4i = cst['r4ies']
    rth = cst['rthomo']
    rlm = cst['rlmin']

    for jk in range(KLEV):
        for jl in range(KLON):
            if za[jk, jl] > 1e-14 and zdqs[jl] <= -rlm:
                lc = max(-zdqs[jl], 0.0)
                af = min(1.0, ((max(rtice, min(rtwat, ztp1[jk, jl])) - rtice) * rr)**2)
                zcor = 1.0 / (1.0 - retv * zqsmix[jk, jl])
                cdm_f = (zqv[jk, jl] - zqsmix[jk, jl]) / (1.0 + zcor * zqsmix[jk, jl] *
                                                          (af * r5a / (ztp1[jk, jl] - r4l)**2 + (1.0 - af) * r5s /
                                                           (ztp1[jk, jl] - r4i)**2))
                cdm_p = (zqv[jk, jl] - za[jk, jl] * zqsmix[jk, jl]) / za[jk, jl]
                cdm = cdm_f if za[jk, jl] > 0.99 else cdm_p
                lc = za[jk, jl] * max(min(lc, cdm), 0.0)
                if lc >= rlm:
                    if ztp1[jk, jl] > rth:
                        zsolqa[QV, QL, jl] += lc
                        zsolqa[QL, QV, jl] -= lc
                        zqxfg[QL, jl] += lc
                    else:
                        zsolqa[QV, QI, jl] += lc
                        zsolqa[QI, QV, jl] -= lc
                        zqxfg[QI, jl] += lc


@dace.program
def melt_kernel(
    zqxfg: dace.float64[nclv, klon],
    zsolqa: dace.float64[nclv, nclv, klon],
    zmeltmax: dace.float64[klon],
    zicetot: dace.float64[klon],
    imelt: dace.int32[nclv],
):
    zepsec = 1e-14
    for jm in range(nclv):
        for jl in range(klon):
            if zmeltmax[jl] > zepsec and zicetot[jl] > zepsec:
                zalfa2 = zqxfg[jm, jl] / zicetot[jl]
                zmelt = min(zqxfg[jm, jl], zalfa2 * zmeltmax[jl])
                zqxfg[jm, jl] = zqxfg[jm, jl] - zmelt
                zqxfg[imelt[jm] - 1, jl] = zqxfg[imelt[jm] - 1, jl] + zmelt
                zsolqa[jm, imelt[jm] - 1, jl] = zsolqa[jm, imelt[jm] - 1, jl] + zmelt
                zsolqa[imelt[jm] - 1, jm, jl] = zsolqa[imelt[jm] - 1, jm, jl] - zmelt


def melt_ref(zqxfg, zsolqa, zmeltmax, zicetot, imelt, sym):
    NCLV, KLON = sym['nclv'], sym['klon']
    zepsec = 1e-14
    for jm in range(NCLV):
        for jl in range(KLON):
            if zmeltmax[jl] > zepsec and zicetot[jl] > zepsec:
                zalfa2 = zqxfg[jm, jl] / zicetot[jl]
                zmelt = min(zqxfg[jm, jl], zalfa2 * zmeltmax[jl])
                zqxfg[jm, jl] -= zmelt
                zqxfg[imelt[jm] - 1, jl] += zmelt
                zsolqa[jm, imelt[jm] - 1, jl] += zmelt
                zsolqa[imelt[jm] - 1, jm, jl] -= zmelt


@dace.program
def init_tendency(
    tendency_loc_cld: dace.float64[nclv, klev, klon],
    kidia: dace.int32,
    kfdia: dace.int32,
):
    for jm in range(1, nclv - 1 + 1):
        for jk in range(1, klev + 1):
            for jl in range(kidia, kfdia + 1):
                tendency_loc_cld[jm - 1, jk - 1, jl - kidia] = 0.0


def init_tendency_ref(
    tendency_loc_cld,
    kidia,
    kfdia,
    KLEV_VAL,
):
    for jm in range(1, 5 - 1 + 1):
        for jk in range(1, KLEV_VAL + 1):
            for jl in range(kidia, kfdia + 1):
                tendency_loc_cld[jm - 1, jk - 1, jl - kidia] = 0.0


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------
def xfill(shape, lo=0.0, hi=1.0):
    a = np.empty(shape, np.float64)
    n = a.size
    i = np.arange(1, n + 1, dtype=np.uint64)
    i ^= i << np.uint64(13)
    i ^= i >> np.uint64(7)
    i ^= i << np.uint64(17)
    a.ravel()[:] = lo + (hi - lo) * (i % np.uint64(10000)).astype(np.float64) / 10000.0
    return a


def clone(d):
    return {k: v.copy() for k, v in d.items()}


def _make_split_name(base, *indices):
    return f"{base}_{'_'.join(NAME_MAP[i] for i in indices)}"


def _check_split_result(split_args, ref_arrays, nclv_v, names):
    for name in names:
        ref = ref_arrays[name]

        if name == 'zsolqa':
            rebuilt = np.array([[split_args[_make_split_name(name, i, j)] for j in range(nclv_v)]
                                for i in range(nclv_v)])
        elif name in ('zqxfg', 'tendency_loc_cld'):
            rebuilt = np.array([split_args[_make_split_name(name, i)] for i in range(nclv_v)])
        else:
            rebuilt = split_args[name]

        np.testing.assert_allclose(rebuilt,
                                   ref,
                                   atol=1e-14,
                                   err_msg=f"{name}: max diff = {np.max(np.abs(rebuilt - ref))}")


def test_condense():
    KLEV, KLON, NCLV = 16, 512, 5
    QL, QI, QR, QS, QV = 1, 2, 3, 4, 5
    SYM = dict(klev=KLEV, klon=KLON, nclv=NCLV, ncldql=QL, ncldqi=QI, ncldqr=QR, ncldqs=QS, ncldqv=QV)

    def _make_inputs():
        return dict(
            za=xfill((KLEV, KLON), 0.0, 1.0),
            zdqs=xfill((KLON, ), -0.01, 0.01),
            zqsmix=xfill((KLEV, KLON), 0.001, 0.02),
            zqv=xfill((KLEV, KLON), 0.001, 0.02),
            ztp1=xfill((KLEV, KLON), 200.0, 300.0),
            zsolqa=xfill((NCLV, NCLV, KLON), -0.001, 0.001),
            zqxfg=xfill((NCLV, KLON), 0.0, 0.01),
        )

    inp_ref = _make_inputs()
    inp_dace = clone(inp_ref)

    condense_ref(inp_ref['za'], inp_ref['zdqs'], inp_ref['zqsmix'], inp_ref['zqv'], inp_ref['ztp1'], inp_ref['zsolqa'],
                 inp_ref['zqxfg'], CONDENSE_CST, SYM)

    csdfg = condense_kernel.compile(**SYM)
    csdfg(**inp_dace, **CONDENSE_CST, **SYM)

    np.testing.assert_allclose(inp_dace['zsolqa'], inp_ref['zsolqa'], atol=1e-14)
    np.testing.assert_allclose(inp_dace['zqxfg'], inp_ref['zqxfg'], atol=1e-14)

    inp_ref = _make_inputs()
    inp_split = clone(inp_ref)

    condense_ref(inp_ref['za'], inp_ref['zdqs'], inp_ref['zqsmix'], inp_ref['zqv'], inp_ref['ztp1'], inp_ref['zsolqa'],
                 inp_ref['zqxfg'], CONDENSE_CST, SYM)

    symbol_map = {
        "nclv": 5,
        "ncldql": 1,
        "ncldqi": 2,
        "ncldqr": 3,
        "ncldqs": 4,
        "ncldqv": 5,
    }
    name_map = {"nclv": NAME_ORDER}
    sdfg = condense_kernel.to_sdfg()
    sdfg.name = "condense_original"
    sdfg.compile()
    SplitArray(symbol_map=symbol_map, name_map=name_map).apply_pass(sdfg, {})
    sdfg.name = "condense_split"
    csdfg = sdfg.compile()

    # Build args depending on whether pass ran
    args = dict(**SYM, **CONDENSE_CST)
    for i in range(NCLV):
        for j in range(NCLV):
            args[_make_split_name('zsolqa', i, j)] = inp_split['zsolqa'][i, j].copy()
    for i in range(NCLV):
        args[_make_split_name('zqxfg', i)] = inp_split['zqxfg'][i].copy()
    args['za'] = inp_split['za']
    args['zdqs'] = inp_split['zdqs']
    args['zqsmix'] = inp_split['zqsmix']
    args['zqv'] = inp_split['zqv']
    args['ztp1'] = inp_split['ztp1']

    csdfg(**args)
    _check_split_result(args, inp_ref, NCLV, ['zsolqa', 'zqxfg'])


def test_melt():
    KLON, NCLV = 512, 5
    QI, QS, QR = 2, 4, 3
    SYM = dict(klon=KLON, nclv=NCLV, ncldqi=QI, ncldqs=QS, ncldqr=QR)

    def _make_inputs(KLON, NCLV, QI, QS, QR):
        iphase = np.ones(NCLV, dtype=np.int32)
        imelt = np.ones(NCLV, dtype=np.int32)
        iphase[QI - 1] = 2
        imelt[QI - 1] = QR
        iphase[QS - 1] = 2
        imelt[QS - 1] = QR
        return dict(
            zqxfg=xfill((NCLV, KLON), 0.0, 0.01),
            zsolqa=xfill((NCLV, NCLV, KLON), -0.001, 0.001),
            zmeltmax=xfill((KLON, ), 0.0, 0.005),
            zicetot=xfill((KLON, ), 0.0, 0.02),
            iphase=iphase,
            imelt=imelt,
        )

    """DaCe program without any pass matches numpy reference."""
    inp_ref = _make_inputs(KLON, NCLV, QI, QS, QR)
    inp_dace = clone(inp_ref)

    melt_ref(inp_ref['zqxfg'], inp_ref['zsolqa'], inp_ref['zmeltmax'], inp_ref['zicetot'], inp_ref['imelt'], SYM)

    csdfg = melt_kernel.compile(**SYM)
    csdfg(**inp_dace, **SYM)

    np.testing.assert_allclose(inp_dace['zsolqa'], inp_ref['zsolqa'], atol=1e-14)
    np.testing.assert_allclose(inp_dace['zqxfg'], inp_ref['zqxfg'], atol=1e-14)
    """DaCe program + SplitArray pass matches numpy reference."""
    inp_ref = _make_inputs(KLON, NCLV, QI, QS, QR)
    inp_split = clone(inp_ref)

    melt_ref(inp_ref['zqxfg'], inp_ref['zsolqa'], inp_ref['zmeltmax'], inp_ref['zicetot'], inp_ref['imelt'], SYM)

    symbol_map = {
        "nclv": 5,
    }
    name_map = {"nclv": NAME_ORDER}
    sdfg = melt_kernel.to_sdfg()
    SplitArray(symbol_map=symbol_map, name_map=name_map).apply_pass(sdfg, {})
    csdfg = sdfg.compile()

    args = dict(**SYM)
    sdfg_keys = set(sdfg.arrays.keys())
    if 'zsolqa_0' in sdfg_keys:
        for i in range(NCLV):
            args[f'zsolqa_{i}'] = inp_split['zsolqa'][i].copy()
            args[f'zqxfg_{i}'] = inp_split['zqxfg'][i].copy()
        for i in range(NCLV):
            args[f'iphase_{i}'] = inp_split['iphase'][i].copy()
            args[f'imelt_{i}'] = inp_split['imelt'][i].copy()
        args['zmeltmax'] = inp_split['zmeltmax']
        args['zicetot'] = inp_split['zicetot']
    else:
        args.update(inp_split)

    csdfg(**args)
    _check_split_result(args, inp_ref, NCLV, ['zsolqa', 'zqxfg'])


def test_init_tendency():
    KLEV, KLON, NCLV = 10, 512, 5
    KIDIA, KFDIA = 100, 200
    SYM = dict(klev=KLEV, klon=KLON, nclv=NCLV)

    def _make_inputs():
        return dict(tendency_loc_cld=xfill((NCLV, KLEV, KLON), -1.0, 1.0), )

    inp_ref = _make_inputs()
    inp_dace = clone(inp_ref)

    init_tendency_ref(inp_ref['tendency_loc_cld'], KIDIA, KFDIA, KLEV)

    sdfg_plain = init_tendency.compile()
    sdfg_plain(**inp_dace, kidia=KIDIA, kfdia=KFDIA, **SYM)

    np.testing.assert_allclose(inp_dace['tendency_loc_cld'], inp_ref['tendency_loc_cld'], atol=1e-14)

    inp_ref = _make_inputs()
    inp_split = clone(inp_ref)

    init_tendency_ref(inp_ref['tendency_loc_cld'], KIDIA, KFDIA, KLEV)

    # Split along the first dimension (size = nclv-1) into separate 2D arrays
    symbol_map = {
        "nclv": 5,
    }
    name_map = {"nclv": NAME_ORDER}

    sdfg = init_tendency.to_sdfg()
    SplitArray(symbol_map=symbol_map, name_map=name_map).apply_pass(sdfg, {})
    csdfg = sdfg.compile()

    # Build arguments: either original array or the split components
    args = dict(kidia=KIDIA, kfdia=KFDIA, **SYM)
    for i in range(NCLV):
        args[f'tendency_loc_cld_{NAME_MAP[i]}'] = inp_split['tendency_loc_cld'][i].copy()

    csdfg(**args)

    _check_split_result(args, inp_ref, NCLV, ['tendency_loc_cld'])


if __name__ == "__main__":
    test_condense()
    test_melt()
    test_init_tendency()
