# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Plain-numpy reference implementations of the TSVC corpus kernels.

Auto-derived from the ``@dace.program`` kernel bodies in :mod:`tests.corpus.tsvc`
(decorators + type annotations stripped, ``dace.map[...]`` rewritten to
``range(...)``). Each function mutates its array arguments in place and is the
scalar oracle for the corresponding kernel: a transform (canonicalize, LoopToMap,
vectorize) is correct iff its output matches the numpy reference here.

Call as ``REFERENCES[name](**arrays, **scalar_params, **symbols)`` -- the extra
``**_`` swallows any symbol the body does not use.
"""
from math import sin, cos, log, exp, pow  # noqa: F401

import numpy as np  # noqa: F401

VLEN = 8


def s000_d_single(a, b, *, LEN_1D, **_):
    for i in range(LEN_1D):
        a[i] = b[i] + 1.0


def s111_d_single(a, b, *, LEN_1D, **_):
    for i in range(1, LEN_1D, 2):
        a[i] = a[i - 1] + b[i]


def s1111_d_single(a, b, c, d, *, LEN_1D, **_):
    for i in range(0, LEN_1D // 2):
        a[2 * i] = c[i] * b[i] + d[i] * b[i] + c[i] * c[i] + d[i] * b[i] + d[i] * c[i]


def s1112_d_single(a, b, *, LEN_1D, **_):
    for i in range(LEN_1D - 1, -1, -1):
        a[i] = b[i] + 1.0


def s1113_d_single(a, b, *, LEN_1D, **_):
    for i in range(LEN_1D):
        a[i] = a[LEN_1D // 2] + b[i]


def s1115_d_single(aa, bb, cc, *, LEN_2D, **_):
    for i in range(LEN_2D):
        for j in range(LEN_2D):
            aa[i, j] = aa[i, j] * cc[j, i] + bb[i, j]


def s1119_d_single(aa, bb, *, LEN_2D, **_):
    for i in range(1, LEN_2D):
        for j in range(LEN_2D):
            aa[i, j] = aa[i - 1, j] + bb[i, j]


def s112_d_single(a, b, *, LEN_1D, **_):
    for i in range(LEN_1D - 2, -1, -1):
        a[i + 1] = a[i] + b[i]


def s113_d_single(a, b, *, LEN_1D, **_):
    for i in range(1, LEN_1D):
        a[i] = a[0] + b[i]


def s114_d_single(aa, bb, *, LEN_2D, **_):
    for i in range(LEN_2D // VLEN):
        for j in range(i * VLEN):
            aa[i, j] = aa[j, i] + bb[i, j]


def s115_d_single(a, aa, *, LEN_2D, **_):
    for j in range(LEN_2D):
        for i in range(j + 1, LEN_2D):
            a[i] = a[i] - aa[j, i] * a[j]


def s116_d_single(a, *, LEN_1D, **_):
    for i in range(0, LEN_1D - 4, 4):
        a[i] = a[i + 1] * a[i]
        a[i + 1] = a[i + 2] * a[i + 1]
        a[i + 2] = a[i + 3] * a[i + 2]
        a[i + 3] = a[i + 4] * a[i + 3]


def s1161_d_single(a, b, c, d, e, *, LEN_1D, **_):
    for i in range(LEN_1D):
        if c[i] < 0.0:
            b[i] = a[i] + d[i] * d[i]
        else:
            a[i] = c[i] + d[i] * e[i]


def s118_d_single(a, bb, *, LEN_2D, **_):
    for i in range(1, LEN_2D):
        for j in range(0, i):
            a[i] = a[i] + bb[j, i] * a[i - j - 1]


def s119_d_single(aa, bb, *, LEN_2D, **_):
    for i in range(1, LEN_2D):
        for j in range(1, LEN_2D):
            aa[i, j] = aa[i - 1, j - 1] + bb[i, j]


def s121_d_single(a, b, *, LEN_1D, **_):
    for i in range(LEN_1D - 1):
        j = i + 1
        a[i] = a[j] + b[i]


def s1213_d_single(a, b, c, d, *, LEN_1D, **_):
    for i in range(1, LEN_1D - 1):
        a[i] = b[i - 1] + c[i]
        b[i] = a[i + 1] * d[i]


def s122_d_single(a, b, n1, n3, *, LEN_1D, **_):
    j = 1
    k = 0
    for i in range(n1 - 1, LEN_1D, n3):
        k = k + j
        a[i] = a[i] + b[LEN_1D - k]


def s1221_d_single(a, b, *, LEN_1D, **_):
    for i in range(4, LEN_1D):
        b[i] = b[i - 4] + a[i]


def s123_d_single(a, b, c, d, e, *, LEN_1D, **_):
    j = -1
    for i in range(LEN_1D // 2):
        j = j + 1
        a[j] = b[i] + d[i] * e[i]
        if c[i] > 0.0:
            j = j + 1
            a[j] = c[i] + d[i] * e[i]


def s1232_d_single(aa, bb, cc, *, LEN_2D, **_):
    for j in range(LEN_2D):
        for i in range(j * VLEN, LEN_2D):
            aa[i, j] = bb[i, j] + cc[i, j]


def s124_d_single(a, b, c, d, e, *, LEN_1D, **_):
    j = -1
    for i in range(LEN_1D):
        if b[i] > 0.0:
            j = j + 1
            a[j] = b[i] + d[i] * e[i]
        else:
            j = j + 1
            a[j] = c[i] + d[i] * e[i]


def s1244_d_single(a, b, c, d, *, LEN_1D, **_):
    for i in range(LEN_1D - 1):
        a[i] = b[i] + c[i] * c[i] + b[i] * b[i] + c[i]
        d[i] = a[i] + a[i + 1]


def s125_d_single(flat_2d_array, aa, bb, cc, *, LEN_2D, **_):
    k = -1
    for i in range(LEN_2D):
        for j in range(LEN_2D):
            k = k + 1
            flat_2d_array[k] = aa[i, j] + bb[i, j] * cc[i, j]


def s1251_d_single(a, b, c, d, e, *, LEN_1D, **_):
    for i in range(LEN_1D):
        s = b[i] + c[i]
        b[i] = a[i] + d[i]
        a[i] = s * e[i]


def s126_d_single(bb, flat_2d_array, cc, *, LEN_2D, **_):
    k = 1
    for i in range(LEN_2D):
        for j in range(1, LEN_2D):
            bb[j, i] = bb[j - 1, i] + flat_2d_array[k - 1] * cc[j, i]
            k = k + 1
        k = k + 1


def s127_d_single(a, b, c, d, e, *, LEN_1D, **_):
    for i in range(0, LEN_1D // 2):
        a[2 * i] = b[i] + c[i] * d[i]
        a[2 * i + 1] = b[i] + d[i] * e[i]


def s1279_d_single(a, b, c, d, e, *, LEN_1D, **_):
    for i in range(LEN_1D):
        if a[i] < 0.0:
            if b[i] > a[i]:
                c[i] = c[i] + d[i] * e[i]


def s128_d_single(a, b, c, d, *, LEN_1D, **_):
    j = -1
    for i in range(LEN_1D // 2):
        k = j + 1
        a[i] = b[k] - d[i]
        j = k + 1
        b[k] = a[i] + c[k]


def s1281_d_single(a, b, c, d, e, *, LEN_1D, **_):
    for i in range(LEN_1D):
        x = b[i] * c[i] + a[i] * d[i] + e[i]
        a[i] = x - 1.0
        b[i] = x


def s131_d_single(a, b, *, LEN_1D, **_):
    for i in range(LEN_1D - 1):
        a[i] = a[i + 1] + b[i]


def s13110_d_single(aa, bb, *, LEN_2D, **_):
    maxv = aa[0, 0]
    xindex = 0
    yindex = 0
    for i in range(LEN_2D):
        for j in range(LEN_2D):
            if aa[i, j] > maxv:
                maxv = aa[i, j]
                xindex = i
                yindex = j
    chksum = maxv + float(xindex) + float(yindex)
    tmp = chksum
    tmp = tmp
    bb[0, 0] = chksum


def s132_d_single(aa, b, c, *, LEN_2D, **_):
    for i in range(1, LEN_2D):
        aa[0, i] = aa[1, i - 1] + b[i] * c[1]


def s1351_d_single(a, b, c, *, LEN_1D, **_):
    for i in range(LEN_1D):
        a[i] = b[i] + c[i]


def s141_d_single(bb, flat_2d_array, *, LEN_2D, **_):
    for i in range(LEN_2D):
        k = (i + 1) * i // 2 + i
        for j in range(i, LEN_2D):
            flat_2d_array[k] = flat_2d_array[k] + bb[j, i]
            k = k + j + 1


def s1421_d_single(b, a, *, LEN_1D, **_):
    half = LEN_1D // 2
    for i in range(half):
        b[i] = b[half + i] + a[i]


def s151_d_single(a, b, *, LEN_1D, **_):
    for i in range(LEN_1D - 1):
        a[i] = a[i + 1] + b[i]


def s152_d_single(a, b, c, d, e, *, LEN_1D, **_):
    for i in range(LEN_1D):
        b[i] = d[i] * e[i]
    for i in range(LEN_1D):
        a[i] = a[i] + b[i] * c[i]


def s161_d_single(a, b, c, d, e, *, LEN_1D, **_):
    for i in range(LEN_1D - 1):
        if b[i] < 0.0:
            c[i + 1] = a[i] + d[i] * d[i]
        else:
            a[i] = c[i] + d[i] * e[i]


def s162_d_single(a, b, c, k, *, LEN_1D, **_):
    if k > 0:
        for i in range(0, LEN_1D - k):
            a[i] = a[i + k] + b[i] * c[i]


def s171_d_single(a, b, inc, *, LEN_1D, **_):
    for i in range(LEN_1D):
        a[i * inc] = a[i * inc] + b[i]


def s172_d_single(a, b, n1, n3, *, LEN_1D, **_):
    for i in range(n1 - 1, LEN_1D, n3):
        a[i] = a[i] + b[i]


def s173_d_single(a, b, *, LEN_1D, **_):
    for i in range(LEN_1D // 2):
        a[i + LEN_1D // 2] = a[i] + b[i]


def s174_d_single(a, b, M, **_):
    for i in range(M):
        a[i + M] = a[i] + b[i]


def s175_d_single(a, b, inc, *, LEN_1D, **_):
    for i in range(0, LEN_1D - inc, inc):
        a[i] = a[i + inc] + b[i]


def s176_d_single(a, b, c, *, LEN_1D, **_):
    m = LEN_1D // 2
    for j in range(LEN_1D // 2):
        for i in range(m):
            a[i] = a[i] + b[i + m - j - 1] * c[j]


def s2101_d_single(aa, bb, cc, *, LEN_2D, **_):
    for i in range(LEN_2D):
        aa[i, i] = aa[i, i] + bb[i, i] * cc[i, i]


def s2102_d_single(aa, *, LEN_2D, **_):
    for i in range(LEN_2D):
        for j in range(LEN_2D):
            aa[j, i] = 0.0
        aa[i, i] = 1.0


def s211_d_single(a, b, c, d, e, *, LEN_1D, **_):
    for i in range(1, LEN_1D - 1):
        a[i] = b[i - 1] + c[i] * d[i]
        b[i] = b[i + 1] - e[i] * d[i]


def s2111_d_single(aa, *, LEN_2D, **_):
    for j in range(1, LEN_2D):
        for i in range(1, LEN_2D):
            aa[j, i] = (aa[j, i - 1] + aa[j - 1, i]) / 1.9


def s212_d_single(a, b, c, d, *, LEN_1D, **_):
    for i in range(LEN_1D - 1):
        a[i] = a[i] * c[i]
        b[i] = b[i] + a[i + 1] * d[i]


def s221_d_single(a, b, c, d, *, LEN_1D, **_):
    for i in range(1, LEN_1D):
        a[i] = a[i] + c[i] * d[i]
        b[i] = b[i - 1] + a[i] + d[i]


def s222_d_single(a, b, c, e, *, LEN_1D, **_):
    for i in range(1, LEN_1D):
        a[i] = a[i] + b[i] * c[i]
        e[i] = e[i - 1] * e[i - 1]
        a[i] = a[i] - b[i] * c[i]


def s2233_d_single(aa, bb, cc, *, LEN_2D, **_):
    for i in range(8, LEN_2D):
        for j in range(8, LEN_2D):
            aa[j, i] = aa[j - 1, i] + cc[j, i]
        for j in range(8, LEN_2D):
            bb[i, j] = bb[i - 1, j] + cc[i, j]


def s2244_d_single(a, b, c, e, *, LEN_1D, **_):
    a[LEN_1D - 1] = b[LEN_1D - 2] + e[LEN_1D - 2]
    for i in range(LEN_1D - 1):
        a[i] = b[i] + c[i]


def s2251_d_single(a, b, c, d, e, *, LEN_1D, **_):
    s = 0.0
    for i in range(LEN_1D):
        a[i] = s * e[i]
        s = b[i] + c[i]
        b[i] = a[i] + d[i]


def s2275_d_single(a, b, c, d, aa, bb, cc, *, LEN_2D, **_):
    for i in range(LEN_2D):
        for j in range(LEN_2D):
            aa[j, i] = aa[j, i] + bb[j, i] * cc[j, i]
        a[i] = b[i] + c[i] * d[i]


def s231_d_single(aa, bb, *, LEN_2D, **_):
    for i in range(LEN_2D):
        for j in range(1, LEN_2D):
            aa[j, i] = aa[j - 1, i] + bb[j, i]


def s232_d_single(aa, bb, *, LEN_2D, **_):
    for j in range(1, LEN_2D):
        for i in range(1, j + 1):
            aa[j, i] = aa[j, i - 1] * aa[j, i - 1] + bb[j, i]


def s233_d_single(aa, bb, cc, *, LEN_2D, **_):
    for i in range(8, LEN_2D):
        for j in range(8, LEN_2D):
            aa[j, i] = aa[j - 1, i] + cc[j, i]
        for j in range(8, LEN_2D):
            bb[j, i] = bb[j, i - 1] + cc[j, i]


def s235_d_single(a, b, c, aa, bb, *, LEN_2D, **_):
    for i in range(LEN_2D):
        a[i] = a[i] + b[i] * c[i]
        for j in range(1, LEN_2D):
            aa[j, i] = aa[j - 1, i] + bb[j, i] * a[i]


def s241_d_single(a, b, c, d, *, LEN_1D, **_):
    for i in range(LEN_1D - 1):
        a[i] = b[i] * c[i] * d[i]
        b[i] = a[i] * a[i + 1] * d[i]


def s242_d_single(a, b, c, d, *, LEN_1D, **_):
    for i in range(1, LEN_1D):
        a[i] = a[i - 1] + 0.5 + 1.0 + b[i] + c[i] + d[i]


def s243_d_single(a, b, c, d, e, *, LEN_1D, **_):
    for i in range(LEN_1D - 1):
        a[i] = b[i] + c[i] * d[i]
        b[i] = a[i] + d[i] * e[i]
        a[i] = b[i] + a[i + 1] * d[i]


def s244_d_single(a, b, c, d, *, LEN_1D, **_):
    for i in range(LEN_1D - 1):
        a[i] = b[i] + c[i] * d[i]
        b[i] = c[i] + b[i]
        a[i + 1] = b[i] + a[i + 1] * d[i]


def s251_d_single(a, b, c, d, *, LEN_1D, **_):
    for i in range(LEN_1D):
        s = b[i] + c[i] * d[i]
        a[i] = s * s


def s252_d_single(a, b, c, *, LEN_1D, **_):
    t = 0.0
    for i in range(LEN_1D):
        s = b[i] * c[i]
        a[i] = s + t
        t = s


def s253_d_single(a, b, c, d, *, LEN_1D, **_):
    for i in range(LEN_1D):
        if a[i] > b[i]:
            s = a[i] - b[i] * d[i]
            c[i] = c[i] + s
            a[i] = s


def s254_d_single(a, b, *, LEN_1D, **_):
    x = b[LEN_1D - 1]
    for i in range(LEN_1D):
        a[i] = (b[i] + x) * 0.5
        x = b[i]


def s255_d_single(a, b, *, LEN_1D, **_):
    x = b[LEN_1D - 1]
    y = b[LEN_1D - 2]
    for i in range(LEN_1D):
        a[i] = (b[i] + x + y) * 0.333
        y = x
        x = b[i]


def s256_d_single(a, aa, bb, d, *, LEN_2D, **_):
    for i in range(LEN_2D):
        for j in range(1, LEN_2D):
            a[j] = 1.0 - a[j - 1]
            aa[j, i] = a[j] + bb[j, i] * d[j]


def s257_d_single(a, aa, bb, *, LEN_2D, **_):
    for i in range(8, LEN_2D):
        for j in range(LEN_2D):
            a[i] = aa[j, i] - a[i - 1]
            aa[j, i] = a[i] + bb[j, i]


def s258_d_single(a, b, c, d, e, aa, *, LEN_2D, **_):
    s = 0.0
    for i in range(LEN_2D):
        if a[i] > 0.0:
            s = d[i] * d[i]
        b[i] = s * c[i] + d[i]
        e[i] = (s + 1.0) * aa[0, i]


def s261_d_single(a, b, c, d, *, LEN_1D, **_):
    for i in range(1, LEN_1D):
        t = a[i] + b[i]
        a[i] = t + c[i - 1]
        c[i] = c[i] * d[i]


def s271_d_single(a, b, c, *, LEN_1D, **_):
    for i in range(LEN_1D):
        if b[i] > 0.0:
            a[i] = a[i] + b[i] * c[i]


def s2710_d_single(a, b, c, d, e, x, *, LEN_1D, **_):
    for i in range(LEN_1D):
        if a[i] > b[i]:
            a[i] = a[i] + b[i] * d[i]
            if LEN_1D > 10:
                c[i] = c[i] + d[i] * d[i]
            else:
                c[i] = d[i] * e[i] + 1.0
        else:
            b[i] = a[i] + e[i] * e[i]
            if x[0] > 0.0:
                c[i] = a[i] + d[i] * d[i]
            else:
                c[i] = c[i] + e[i] * e[i]


def s2711_d_single(a, b, c, *, LEN_1D, **_):
    for i in range(LEN_1D):
        if b[i] != 0.0:
            a[i] = a[i] + b[i] * c[i]


def s2712_d_single(a, b, c, *, LEN_1D, **_):
    for i in range(LEN_1D):
        if a[i] > b[i]:
            a[i] = a[i] + b[i] * c[i]


def s272_d_single(a, b, c, d, e, threshold, *, LEN_1D, **_):
    for i in range(LEN_1D):
        if e[i] >= threshold:
            a[i] = a[i] + c[i] * d[i]
            b[i] = b[i] + c[i] * c[i]


def s273_d_single(a, b, c, d, e, *, LEN_1D, **_):
    for i in range(LEN_1D):
        a[i] = a[i] + d[i] * e[i]
        if a[i] < 0.0:
            b[i] = b[i] + d[i] * e[i]
        c[i] = c[i] + a[i] * d[i]


def s274_d_single(a, b, c, d, e, *, LEN_1D, **_):
    for i in range(LEN_1D):
        a[i] = c[i] + e[i] * d[i]
        if a[i] > 0.0:
            b[i] = a[i] + b[i]
        else:
            a[i] = d[i] * e[i]


def s275_d_single(aa, bb, cc, *, LEN_2D, **_):
    for i in range(LEN_2D):
        if aa[0, i] > 0.0:
            for j in range(1, LEN_2D):
                aa[j, i] = aa[j - 1, i] + bb[j, i] * cc[j, i]


def s276_d_single(a, b, c, d, *, LEN_1D, **_):
    mid = LEN_1D // 2
    for i in range(LEN_1D):
        if i + 1 < mid:
            a[i] = a[i] + b[i] * c[i]
        else:
            a[i] = a[i] + b[i] * d[i]


def s277_d_single(a, b, c, d, e, *, LEN_1D, **_):
    for i in range(LEN_1D - 1):
        if a[i] < 0.0:
            if b[i] < 0.0:
                a[i] = a[i] + c[i] * d[i]
            b[i + 1] = c[i] + d[i] * e[i]


def s278_d_single(a, b, c, d, e, *, LEN_1D, **_):
    for i in range(LEN_1D):
        if a[i] > 0.0:
            c[i] = -c[i] + d[i] * e[i]
        else:
            b[i] = -b[i] + d[i] * e[i]
        a[i] = b[i] + c[i] * d[i]


def s279_d_single(a, b, c, d, e, *, LEN_1D, **_):
    for i in range(LEN_1D):
        if a[i] > 0.0:
            c[i] = -c[i] + e[i] * e[i]
        else:
            b[i] = -b[i] + d[i] * d[i]
            if b[i] > a[i]:
                c[i] = c[i] + d[i] * e[i]
        a[i] = b[i] + c[i] * d[i]


def s281_d_single(a, b, c, *, LEN_1D, **_):
    for i in range(LEN_1D):
        x = a[LEN_1D - i - 1] + b[i] * c[i]
        a[i] = x - 1.0
        b[i] = x


def s291_d_single(a, b, *, LEN_1D, **_):
    a[0] = (b[0] + b[LEN_1D - 1]) * 0.5
    for i in range(1, LEN_1D):
        a[i] = (b[i] + b[i - 1]) * 0.5


def s292_d_single(a, b, *, LEN_1D, **_):
    a[0] = (b[0] + b[LEN_1D - 1] + b[LEN_1D - 2]) * 0.333
    a[1] = (b[1] + b[0] + b[LEN_1D - 1]) * 0.333
    for i in range(2, LEN_1D):
        a[i] = (b[i] + b[i - 1] + b[i - 2]) * 0.333


def s293_d_single(a, *, LEN_1D, **_):
    a0 = a[0]
    for i in range(LEN_1D):
        a[i] = a0


def s311_d_single(a, sum_out, *, LEN_1D, **_):
    sum_out[0] = 0.0
    for i in range(LEN_1D):
        sum_out[0] = sum_out[0] + a[i]


def s3110_d_single(aa, bb, *, LEN_2D, **_):
    maxv = aa[0, 0]
    xindex = 0
    yindex = 0
    for i in range(LEN_2D):
        for j in range(LEN_2D):
            if aa[i, j] > maxv:
                maxv = aa[i, j]
                xindex = i
                yindex = j
    chksum = maxv + float(xindex) + float(yindex)
    tmp = chksum
    tmp = tmp
    bb[0, 0] = chksum


def s3111_d_single(a, b, *, LEN_1D, **_):
    sum_val = 0.0
    for i in range(LEN_1D):
        if a[i] > 0.0:
            sum_val = sum_val + a[i]
    b[0] = sum_val


def s31111_d_single(a, b, *, LEN_1D, **_):
    sum_val = 0.0
    for base in range(0, LEN_1D - 3, 4):
        partial = 0.0
        partial = partial + a[base + 0]
        partial = partial + a[base + 1]
        partial = partial + a[base + 2]
        partial = partial + a[base + 3]
        sum_val = sum_val + partial
    b[0] = sum_val


def s3112_d_single(a, b, *, LEN_1D, **_):
    sum = 0.0
    for i in range(LEN_1D):
        sum = sum + a[i]
        b[i] = sum


def s3113_d_single(a, b, *, LEN_1D, **_):
    maxv = 0.0
    maxv = abs(a[0])
    for i in range(LEN_1D):
        av = abs(a[i])
        if av > maxv:
            maxv = av
    b[0] = maxv


def s312_d_single(a, result, *, LEN_1D, **_):
    prod = 1.0
    for i in range(LEN_1D):
        prod = prod * a[i]
    result[0] = prod


def s313_d_single(a, b, dot, *, LEN_1D, **_):
    dot[0] = 0.0
    for i in range(LEN_1D):
        dot[0] = dot[0] + a[i] * b[i]


def s314_d_single(a, result, *, LEN_1D, **_):
    x = a[0]
    for i in range(1, LEN_1D):
        if a[i] > x:
            x = a[i]
    result[0] = x


def s315_d_single(a, result, *, LEN_1D, **_):
    for i in range(LEN_1D):
        a[i] = float(i * 7 % LEN_1D)
    x = a[0]
    index = 0
    for i in range(LEN_1D):
        if a[i] > x:
            x = a[i]
            index = i
    a[0] = x + float(index)
    result[0] = a[0]


def s316_d_single(a, result, *, LEN_1D, **_):
    x = a[0]
    for i in range(1, LEN_1D):
        if a[i] < x:
            x = a[i]
    result[0] = x


def s317_d_single(q, *, LEN_1D, **_):
    q[0] = 1.0
    for i in range(LEN_1D // 2):
        q[0] = q[0] * 0.99


def s318_d_single(a, result, inc, *, LEN_1D, **_):
    k = 0
    index = 0
    maxv = abs(a[0])
    k = k + inc
    for i in range(1, LEN_1D):
        v = abs(a[k])
        if v > maxv:
            index = i
            maxv = v
        k = k + inc
    result[0] = maxv + float(index)


def s319_d_single(a, b, c, d, e, *, LEN_1D, **_):
    sum_val = 0.0
    for i in range(LEN_1D):
        a[i] = c[i] + d[i]
        sum_val = sum_val + a[i]
        b[i] = c[i] + e[i]
        sum_val = sum_val + b[i]
    b[0] = sum_val


def s321_d_single(a, b, *, LEN_1D, **_):
    for i in range(1, LEN_1D):
        a[i] = a[i] + a[i - 1] * b[i]


def s322_d_single(a, b, c, *, LEN_1D, **_):
    for i in range(2, LEN_1D):
        a[i] = a[i] + a[i - 1] * b[i] + a[i - 2] * c[i]


def s323_d_single(a, b, c, d, e, *, LEN_1D, **_):
    for i in range(1, LEN_1D):
        a[i] = b[i - 1] + c[i] * d[i]
        b[i] = a[i] + c[i] * e[i]


def s3251_d_single(a, b, c, d, e, *, LEN_1D, **_):
    for i in range(LEN_1D - 1):
        a[i + 1] = b[i] + c[i]
        b[i] = c[i] * e[i]
        d[i] = a[i] * e[i]


def s331_d_single(a, b, *, LEN_1D, **_):
    j = -1
    j = -1
    for i in range(LEN_1D):
        if a[i] < 0.0:
            j = i
    b[0] = j


def s332_d_single(a, result, threshold, *, LEN_1D, **_):
    index = -2
    value = -1.0
    for i in range(LEN_1D):
        if a[i] > threshold:
            index = i
            value = a[i]
            break
    result[0] = value + float(index)


def s341_d_single(a, b, *, LEN_1D, **_):
    j = -1
    for i in range(LEN_1D):
        if b[i] > 0.0:
            j = j + 1
            a[j] = b[i]


def s342_d_single(a, b, *, LEN_1D, **_):
    j = -1
    for i in range(LEN_1D):
        if a[i] > 0.0:
            j = j + 1
            a[i] = b[j]


def s343_d_single(aa, bb, flat_2d_array, *, LEN_2D, **_):
    k = -1
    for i in range(LEN_2D):
        for j in range(LEN_2D):
            if bb[j, i] > 0.0:
                k = k + 1
                flat_2d_array[k] = aa[j, i]


def s351_d_single(a, b, c, *, LEN_1D, **_):
    alpha = c[0]
    for i in range(0, LEN_1D - 3, 4):
        a[i] = a[i] + alpha * b[i]
        a[i + 1] = a[i + 1] + alpha * b[i + 1]
        a[i + 2] = a[i + 2] + alpha * b[i + 2]
        a[i + 3] = a[i + 3] + alpha * b[i + 3]


def s352_d_single(a, b, c, *, LEN_1D, **_):
    dot = 0.0
    dot = 0.0
    for i in range(0, LEN_1D - 4, 5):
        dot = dot + (a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3] + a[i + 4] * b[i + 4])
    c[0] = dot


def s353_d_single(a, b, c, ip, *, LEN_1D, **_):
    alpha = c[0]
    for i in range(0, LEN_1D - 3, 4):
        a[i] = a[i] + alpha * b[ip[i]]
        a[i + 1] = a[i + 1] + alpha * b[ip[i + 1]]
        a[i + 2] = a[i + 2] + alpha * b[ip[i + 2]]
        a[i + 3] = a[i + 3] + alpha * b[ip[i + 3]]


def s4112_d_single(a, b, ip, *, LEN_1D, **_):
    for i in range(LEN_1D):
        a[i] = a[i] + b[ip[i]] * 2.0


def s4113_d_single(a, b, c, ip, *, LEN_1D, **_):
    for i in range(LEN_1D):
        a[ip[i]] = b[ip[i]] + c[i]


def s4114_d_single(a, b, c, d_, ip, n1, *, LEN_1D, **_):
    for i in range(n1 - 1, LEN_1D):
        k = ip[i]
        a[i] = b[i] + c[LEN_1D - k - 1] * d_[i]


def s4115_d_single(a, b, ip, sum_out, *, LEN_1D, **_):
    sum_val = 0.0
    sum_val = 0.0
    for i in range(LEN_1D):
        sum_val = sum_val + a[i] * b[ip[i]]
    sum_out[0] = sum_val


def s4116_d_single(a, aa, ip, j, inc, sum_out, *, LEN_2D, **_):
    sum_val = 0.0
    sum_val = 0.0
    for i in range(LEN_2D - 1):
        off = inc + i
        sum_val = sum_val + a[off] * aa[j - 1, ip[i]]
    sum_out[0] = sum_val


def s4117_d_single(a, b, c, d, *, LEN_1D, **_):
    for i in range(LEN_1D):
        j = i // 2
        a[i] = b[i] + c[j] * d[i]


def s4121_d_single(a, b, c, *, LEN_1D, **_):
    for i in range(LEN_1D):
        a[i] = a[i] + b[i] * c[i]


def s421_d_single(a, flat_2d_array, *, LEN_1D, **_):
    for i in range(LEN_1D - 1):
        flat_2d_array[i] = flat_2d_array[i + 1] + a[i]


def s422_d_single(a, flat_2d_array, *, LEN_1D, **_):
    for i in range(LEN_1D):
        flat_2d_array[4 + i] = flat_2d_array[8 + i] + a[i]


def s423_d_single(a, flat_2d_array, *, LEN_1D, **_):
    vl = 64
    for i in range(LEN_1D - 1):
        flat_2d_array[i + 1] = flat_2d_array[vl + i] + a[i]


def s424_d_single(a, xx, flat, *, LEN_1D, **_):
    for i in range(LEN_1D - 1):
        xx[i + 1] = flat[i] + a[i]


def s431_d_single(a, b, *, LEN_1D, **_):
    for i in range(LEN_1D):
        a[i] = a[i] + b[i]


def s441_d_single(a, b, c, d, *, LEN_1D, **_):
    for i in range(LEN_1D):
        if d[i] < 0.0:
            a[i] = a[i] + b[i] * c[i]
        elif d[i] == 0.0:
            a[i] = a[i] + b[i] * b[i]
        else:
            a[i] = a[i] + c[i] * c[i]


def s442_d_single(a, b, c, d, e, indx, *, LEN_1D, **_):
    for i in range(LEN_1D):
        if indx[i] == 1:
            a[i] = a[i] + b[i] * b[i]
        elif indx[i] == 2:
            a[i] = a[i] + c[i] * c[i]
        elif indx[i] == 3:
            a[i] = a[i] + d[i] * d[i]
        elif indx[i] == 4:
            a[i] = a[i] + e[i] * e[i]


def s443_d_single(a, b, c, d, *, LEN_1D, **_):
    for i in range(LEN_1D):
        if d[i] <= 0.0:
            a[i] = a[i] + b[i] * c[i]
        else:
            a[i] = a[i] + b[i] * b[i]


def s451_d_single(a, b, c, *, LEN_1D, **_):
    for i in range(LEN_1D):
        a[i] = sin(b[i]) + cos(c[i])


def s452_d_single(a, b, c, *, LEN_1D, **_):
    for i in range(LEN_1D):
        a[i] = b[i] + c[i] * (i + 1)


def s453_d_single(a, b, *, LEN_1D, **_):
    s = 0.0
    for i in range(LEN_1D):
        s = s + 2.0
        a[i] = s * b[i]


def s471_d_single(x, b, c, d, e, *, LEN_1D, **_):
    for i in range(LEN_1D):
        x[i] = b[i] + d[i] * d[i]
        b[i] = c[i] + d[i] * e[i]


def s481_d_single(a, b, c, d, *, LEN_1D, **_):
    for i in range(LEN_1D):
        if d[i] < 0.0:
            break
        a[i] = a[i] + b[i] * c[i]


def s482_d_single(a, b, c, *, LEN_1D, **_):
    for i in range(LEN_1D):
        a[i] = a[i] + b[i] * c[i]
        if c[i] > b[i]:
            break


def s491_d_single(a, b, c, d, ip, *, LEN_1D, **_):
    for i in range(LEN_1D):
        a[ip[i]] = b[i] + c[i] * d[i]


def va_d_single(a, b, *, LEN_1D, **_):
    for i in range(LEN_1D):
        a[i] = b[i]


def vag_d_single(a, b, ip, *, LEN_1D, **_):
    for i in range(LEN_1D):
        a[i] = b[ip[i]]


def vas_d_single(a, b, ip, *, LEN_1D, **_):
    for i in range(LEN_1D):
        a[ip[i]] = b[i]


def vbor_d_single(a, b, c, d, e, x, *, LEN_2D, **_):
    for i in range(LEN_2D):
        a1 = a[i]
        b1 = b[i]
        c1 = c[i]
        d1 = d[i]
        e1 = e[i]
        f1 = a[i]
        a1 = a1 * b1 * c1 + a1 * b1 * d1 + a1 * b1 * e1 + a1 * b1 * f1 + a1 * c1 * d1 + a1 * c1 * e1 + a1 * c1 * f1 + a1 * d1 * e1 + a1 * d1 * f1 + a1 * e1 * f1
        b1 = b1 * c1 * d1 + b1 * c1 * e1 + b1 * c1 * f1 + b1 * d1 * e1 + b1 * d1 * f1 + b1 * e1 * f1
        c1 = c1 * d1 * e1 + c1 * d1 * f1 + c1 * e1 * f1
        d1 = d1 * e1 * f1
        x[i] = a1 * b1 * c1 * d1


def vdotr_d_single(a, b, dot_out, *, LEN_1D, **_):
    dot_out[0] = 0.0
    dot_out[0] = 0.0
    for i in range(LEN_1D):
        dot_out[0] = dot_out[0] + a[i] * b[i]


def vif_d_single(a, b, *, LEN_1D, **_):
    for i in range(LEN_1D):
        if b[i] > 0.0:
            a[i] = b[i]


def vpv_d_single(a, b, *, LEN_1D, **_):
    for i in range(LEN_1D):
        a[i] = a[i] + b[i]


def vpvpv_d_single(a, b, c, *, LEN_1D, **_):
    for i in range(LEN_1D):
        a[i] = a[i] + b[i] + c[i]


def vpvts_d_single(a, b, *, LEN_1D, S, **_):
    for i in range(LEN_1D):
        a[i] = a[i] + b[i] * S


def vpvtv_d_single(a, b, c, *, LEN_1D, **_):
    for i in range(LEN_1D):
        a[i] = a[i] + b[i] * c[i]


def vsumr_d_single(a, sum_out, *, LEN_1D, **_):
    s = 0.0
    s = 0.0
    for i in range(LEN_1D):
        s = s + a[i]
    sum_out[0] = s


def vtv_d_single(a, b, *, LEN_1D, **_):
    for i in range(LEN_1D):
        a[i] = a[i] * b[i]


def vtvtv_d_single(a, b, c, *, LEN_1D, **_):
    for i in range(LEN_1D):
        a[i] = a[i] * b[i] * c[i]


#: name -> numpy reference function.
REFERENCES = {
    's000_d_single': s000_d_single,
    's111_d_single': s111_d_single,
    's1111_d_single': s1111_d_single,
    's1112_d_single': s1112_d_single,
    's1113_d_single': s1113_d_single,
    's1115_d_single': s1115_d_single,
    's1119_d_single': s1119_d_single,
    's112_d_single': s112_d_single,
    's113_d_single': s113_d_single,
    's114_d_single': s114_d_single,
    's115_d_single': s115_d_single,
    's116_d_single': s116_d_single,
    's1161_d_single': s1161_d_single,
    's118_d_single': s118_d_single,
    's119_d_single': s119_d_single,
    's121_d_single': s121_d_single,
    's1213_d_single': s1213_d_single,
    's122_d_single': s122_d_single,
    's1221_d_single': s1221_d_single,
    's123_d_single': s123_d_single,
    's1232_d_single': s1232_d_single,
    's124_d_single': s124_d_single,
    's1244_d_single': s1244_d_single,
    's125_d_single': s125_d_single,
    's1251_d_single': s1251_d_single,
    's126_d_single': s126_d_single,
    's127_d_single': s127_d_single,
    's1279_d_single': s1279_d_single,
    's128_d_single': s128_d_single,
    's1281_d_single': s1281_d_single,
    's131_d_single': s131_d_single,
    's13110_d_single': s13110_d_single,
    's132_d_single': s132_d_single,
    's1351_d_single': s1351_d_single,
    's141_d_single': s141_d_single,
    's1421_d_single': s1421_d_single,
    's151_d_single': s151_d_single,
    's152_d_single': s152_d_single,
    's161_d_single': s161_d_single,
    's162_d_single': s162_d_single,
    's171_d_single': s171_d_single,
    's172_d_single': s172_d_single,
    's173_d_single': s173_d_single,
    's174_d_single': s174_d_single,
    's175_d_single': s175_d_single,
    's176_d_single': s176_d_single,
    's2101_d_single': s2101_d_single,
    's2102_d_single': s2102_d_single,
    's211_d_single': s211_d_single,
    's2111_d_single': s2111_d_single,
    's212_d_single': s212_d_single,
    's221_d_single': s221_d_single,
    's222_d_single': s222_d_single,
    's2233_d_single': s2233_d_single,
    's2244_d_single': s2244_d_single,
    's2251_d_single': s2251_d_single,
    's2275_d_single': s2275_d_single,
    's231_d_single': s231_d_single,
    's232_d_single': s232_d_single,
    's233_d_single': s233_d_single,
    's235_d_single': s235_d_single,
    's241_d_single': s241_d_single,
    's242_d_single': s242_d_single,
    's243_d_single': s243_d_single,
    's244_d_single': s244_d_single,
    's251_d_single': s251_d_single,
    's252_d_single': s252_d_single,
    's253_d_single': s253_d_single,
    's254_d_single': s254_d_single,
    's255_d_single': s255_d_single,
    's256_d_single': s256_d_single,
    's257_d_single': s257_d_single,
    's258_d_single': s258_d_single,
    's261_d_single': s261_d_single,
    's271_d_single': s271_d_single,
    's2710_d_single': s2710_d_single,
    's2711_d_single': s2711_d_single,
    's2712_d_single': s2712_d_single,
    's272_d_single': s272_d_single,
    's273_d_single': s273_d_single,
    's274_d_single': s274_d_single,
    's275_d_single': s275_d_single,
    's276_d_single': s276_d_single,
    's277_d_single': s277_d_single,
    's278_d_single': s278_d_single,
    's279_d_single': s279_d_single,
    's281_d_single': s281_d_single,
    's291_d_single': s291_d_single,
    's292_d_single': s292_d_single,
    's293_d_single': s293_d_single,
    's311_d_single': s311_d_single,
    's3110_d_single': s3110_d_single,
    's3111_d_single': s3111_d_single,
    's31111_d_single': s31111_d_single,
    's3112_d_single': s3112_d_single,
    's3113_d_single': s3113_d_single,
    's312_d_single': s312_d_single,
    's313_d_single': s313_d_single,
    's314_d_single': s314_d_single,
    's315_d_single': s315_d_single,
    's316_d_single': s316_d_single,
    's317_d_single': s317_d_single,
    's318_d_single': s318_d_single,
    's319_d_single': s319_d_single,
    's321_d_single': s321_d_single,
    's322_d_single': s322_d_single,
    's323_d_single': s323_d_single,
    's3251_d_single': s3251_d_single,
    's331_d_single': s331_d_single,
    's332_d_single': s332_d_single,
    's341_d_single': s341_d_single,
    's342_d_single': s342_d_single,
    's343_d_single': s343_d_single,
    's351_d_single': s351_d_single,
    's352_d_single': s352_d_single,
    's353_d_single': s353_d_single,
    's4112_d_single': s4112_d_single,
    's4113_d_single': s4113_d_single,
    's4114_d_single': s4114_d_single,
    's4115_d_single': s4115_d_single,
    's4116_d_single': s4116_d_single,
    's4117_d_single': s4117_d_single,
    's4121_d_single': s4121_d_single,
    's421_d_single': s421_d_single,
    's422_d_single': s422_d_single,
    's423_d_single': s423_d_single,
    's424_d_single': s424_d_single,
    's431_d_single': s431_d_single,
    's441_d_single': s441_d_single,
    's442_d_single': s442_d_single,
    's443_d_single': s443_d_single,
    's451_d_single': s451_d_single,
    's452_d_single': s452_d_single,
    's453_d_single': s453_d_single,
    's471_d_single': s471_d_single,
    's481_d_single': s481_d_single,
    's482_d_single': s482_d_single,
    's491_d_single': s491_d_single,
    'va_d_single': va_d_single,
    'vag_d_single': vag_d_single,
    'vas_d_single': vas_d_single,
    'vbor_d_single': vbor_d_single,
    'vdotr_d_single': vdotr_d_single,
    'vif_d_single': vif_d_single,
    'vpv_d_single': vpv_d_single,
    'vpvpv_d_single': vpvpv_d_single,
    'vpvts_d_single': vpvts_d_single,
    'vpvtv_d_single': vpvtv_d_single,
    'vsumr_d_single': vsumr_d_single,
    'vtv_d_single': vtv_d_single,
    'vtvtv_d_single': vtvtv_d_single,
}
