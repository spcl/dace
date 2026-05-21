# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""TSVC 1D ``s``-prefixed kernels (bulk), consolidated from the former
block1/block2/block3 batches. Curated kernels live in ``test_selected``;
the non-``s`` vector kernels live in ``test_vector_ops`` — neither is
duplicated here.

Each kernel is parametrised across ``LEN_1D`` in ``{{64, 65}}`` (64
divisible-by-W=8, 65 forces a remainder), ``remainder_strategy`` in
``{{scalar, masked}}`` and ``branch_mode`` in ``{{merge, fp_factor}}``.
Reference correctness is pinned against the unvectorized SDFG.
"""
import copy

import dace
import numpy as np
import pytest

from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU
from tests.passes.vectorization.helpers.tsvc_matrix import build_tsvc_matrix

LEN_1D = dace.symbol("LEN_1D")


@dace.program
def s116_d_single(a: dace.float64[LEN_1D]):
    for i in range(0, LEN_1D - 4, 4):
        a[i] = a[i + 1] * a[i]
        a[i + 1] = a[i + 2] * a[i + 1]
        a[i + 2] = a[i + 3] * a[i + 2]
        a[i + 3] = a[i + 4] * a[i + 3]

@dace.program
def s1161_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    for i in range(LEN_1D):
        if c[i] < 0.0:
            b[i] = a[i] + d[i] * d[i]
        else:
            a[i] = c[i] + d[i] * e[i]

@dace.program
def s1213_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
):
    for i in range(1, LEN_1D - 1):
        a[i] = b[i - 1] + c[i]
        b[i] = a[i + 1] * d[i]

@dace.program
def s1221_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(4, LEN_1D):
        b[i] = b[i - 4] + a[i]

@dace.program
def s123_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    j = -1
    for i in range(LEN_1D // 2):
        j = j + 1
        a[j] = b[i] + d[i] * e[i]
        if c[i] > 0.0:
            j = j + 1
            a[j] = c[i] + d[i] * e[i]

@dace.program
def s124_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    j = -1
    for i in range(LEN_1D):
        if b[i] > 0.0:
            j = j + 1
            a[j] = b[i] + d[i] * e[i]
        else:
            j = j + 1
            a[j] = c[i] + d[i] * e[i]

@dace.program
def s1251_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    for i in range(LEN_1D):
        s = b[i] + c[i]
        b[i] = a[i] + d[i]
        a[i] = s * e[i]

@dace.program
def s1279_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    for i in range(LEN_1D):
        if a[i] < 0.0:
            if b[i] > a[i]:
                c[i] = c[i] + d[i] * e[i]

@dace.program
def s128_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
):
    j = -1
    for i in range(LEN_1D // 2):
        k = j + 1
        a[i] = b[k] - d[i]
        j = k + 1
        b[k] = a[i] + c[k]

@dace.program
def s1281_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    for i in range(LEN_1D):
        x = (b[i] * c[i]) + (a[i] * d[i]) + e[i]
        a[i] = x - 1.0
        b[i] = x

@dace.program
def s1421_d_single(b: dace.float64[LEN_1D], a: dace.float64[LEN_1D]):
    half = LEN_1D // 2
    for i in range(half):
        b[i] = b[half + i] + a[i]

@dace.program
def s151_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(LEN_1D - 1):
        a[i] = a[i + 1] + b[i]

@dace.program
def s152_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    for i in range(LEN_1D):
        b[i] = d[i] * e[i]
    for i in range(LEN_1D):
        a[i] = a[i] + b[i] * c[i]

@dace.program
def s161_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    # Upstream TSVC s161 loops ``for (i = 0; i < LEN_1D-1; ++i)`` so the
    # ``c[i+1]`` write stays in bounds; the original port mis-transcribed
    # this as ``range(LEN_1D)``, writing ``c[LEN_1D]`` OOB even unvectorized.
    for i in range(LEN_1D - 1):
        if b[i] < 0.0:
            c[i + 1] = a[i] + d[i] * d[i]
        else:
            a[i] = c[i] + d[i] * e[i]

@dace.program
def s176_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    m = LEN_1D // 2
    for j in range(LEN_1D // 2):
        for i in range(m):
            a[i] = a[i] + b[i + m - j - 1] * c[j]

@dace.program
def s221_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
):
    for i in range(1, LEN_1D):
        a[i] = a[i] + c[i] * d[i]
        b[i] = b[i - 1] + a[i] + d[i]

@dace.program
def s222_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    for i in range(1, LEN_1D):
        a[i] = a[i] + b[i] * c[i]
        e[i] = e[i - 1] * e[i - 1]
        a[i] = a[i] - b[i] * c[i]

@dace.program
def s2244_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    a[LEN_1D - 1] = b[LEN_1D - 2] + e[LEN_1D - 2]
    for i in range(LEN_1D - 1):
        a[i] = b[i] + c[i]

@dace.program
def s2251_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    s = 0.0
    for i in range(LEN_1D):
        a[i] = s * e[i]
        s = b[i] + c[i]
        b[i] = a[i] + d[i]

@dace.program
def s242_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
):
    for i in range(1, LEN_1D):
        a[i] = a[i - 1] + 0.5 + 1.0 + b[i] + c[i] + d[i]

@dace.program
def s253_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
):
    for i in range(LEN_1D):
        if a[i] > b[i]:
            s = a[i] - b[i] * d[i]
            c[i] = c[i] + s
            a[i] = s

@dace.program
def s254_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    x = b[LEN_1D - 1]
    for i in range(LEN_1D):
        a[i] = (b[i] + x) * 0.5
        x = b[i]

@dace.program
def s255_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    x = b[LEN_1D - 1]
    y = b[LEN_1D - 2]
    for i in range(LEN_1D):
        a[i] = (b[i] + x + y) * 0.333
        y = x
        x = b[i]

@dace.program
def s261_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
):
    for i in range(1, LEN_1D):
        t = a[i] + b[i]
        a[i] = t + c[i - 1]
        c[i] = c[i] * d[i]

@dace.program
def s2710_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
    x: dace.float64[LEN_1D],
):
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

@dace.program
def s2711_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    for i in range(LEN_1D):
        if b[i] != 0.0:
            a[i] = a[i] + b[i] * c[i]

@dace.program
def s2712_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    for i in range(LEN_1D):
        if a[i] > b[i]:
            a[i] = a[i] + b[i] * c[i]

@dace.program
def s273_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    for i in range(LEN_1D):
        a[i] = a[i] + d[i] * e[i]
        if a[i] < 0.0:
            b[i] = b[i] + d[i] * e[i]
        c[i] = c[i] + a[i] * d[i]

@dace.program
def s274_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    for i in range(LEN_1D):
        a[i] = c[i] + e[i] * d[i]
        if a[i] > 0.0:
            b[i] = a[i] + b[i]
        else:
            a[i] = d[i] * e[i]

@dace.program
def s276_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
):
    mid = LEN_1D // 2
    for i in range(LEN_1D):
        if i + 1 < mid:
            a[i] = a[i] + b[i] * c[i]
        else:
            a[i] = a[i] + b[i] * d[i]

@dace.program
def s277_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    for i in range(LEN_1D - 1):
        if a[i] < 0.0:
            if b[i] < 0.0:
                a[i] = a[i] + c[i] * d[i]
            b[i + 1] = c[i] + d[i] * e[i]

@dace.program
def s278_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    for i in range(LEN_1D):
        if a[i] > 0.0:
            c[i] = -c[i] + d[i] * e[i]
        else:
            b[i] = -b[i] + d[i] * e[i]
        a[i] = b[i] + c[i] * d[i]

@dace.program
def s279_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    for i in range(LEN_1D):
        if a[i] > 0.0:
            c[i] = -c[i] + e[i] * e[i]
        else:
            b[i] = -b[i] + d[i] * d[i]
            if b[i] > a[i]:
                c[i] = c[i] + d[i] * e[i]
        a[i] = b[i] + c[i] * d[i]

@dace.program
def s281_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    for i in range(LEN_1D):
        x = a[LEN_1D - i - 1] + b[i] * c[i]
        a[i] = x - 1.0
        b[i] = x

@dace.program
def s291_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    a[0] = (b[0] + b[LEN_1D - 1]) * 0.5
    for i in range(1, LEN_1D):
        a[i] = (b[i] + b[i - 1]) * 0.5

@dace.program
def s292_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    a[0] = (b[0] + b[LEN_1D - 1] + b[LEN_1D - 2]) * 0.333
    a[1] = (b[1] + b[0] + b[LEN_1D - 1]) * 0.333
    for i in range(2, LEN_1D):
        a[i] = (b[i] + b[i - 1] + b[i - 2]) * 0.333

@dace.program
def s3111_d_single(a: dace.float64[LEN_1D], b: dace.float64[2]):
    sum_val = 0.0
    for i in range(LEN_1D):
        if a[i] > 0.0:
            sum_val = sum_val + a[i]
    b[0] = sum_val

@dace.program
def s31111_d_single(a: dace.float64[LEN_1D], b: dace.float64[2]):
    sum_val = 0.0
    for base in range(0, LEN_1D, 4):
        partial = 0.0
        partial = partial + a[base + 0]
        partial = partial + a[base + 1]
        partial = partial + a[base + 2]
        partial = partial + a[base + 3]
        sum_val = sum_val + partial
    b[0] = sum_val

@dace.program
def s3112_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    sum = 0.0
    for i in range(LEN_1D):
        sum = sum + a[i]
        b[i] = sum

@dace.program
def s3113_d_single(a: dace.float64[LEN_1D], b: dace.float64[2]):
    maxv = dace.float64(0)
    maxv = abs(a[0])
    for i in range(LEN_1D):
        av = abs(a[i])
        if av > maxv:
            maxv = av
    b[0] = maxv

@dace.program
def s313_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], dot: dace.float64[1]):
    dot[0] = 0.0
    for i in range(LEN_1D):
        dot[0] = dot[0] + a[i] * b[i]

@dace.program
def s314_d_single(a: dace.float64[LEN_1D], result: dace.float64[1]):
    x = a[0]
    for i in range(1, LEN_1D):
        if a[i] > x:
            x = a[i]
    result[0] = x

@dace.program
def s315_d_single(a: dace.float64[LEN_1D], result: dace.float64[1]):
    for i in range(LEN_1D):
        a[i] = float((i * 7) % LEN_1D)
    x = a[0]
    index = 0
    for i in range(LEN_1D):
        if a[i] > x:
            x = a[i]
            index = i
    a[0] = x + float(index)
    result[0] = a[0]

@dace.program
def s316_d_single(a: dace.float64[LEN_1D], result: dace.float64[1]):
    x = a[0]
    for i in range(1, LEN_1D):
        if a[i] < x:
            x = a[i]
    result[0] = x

@dace.program
def s319_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    sum_val = 0.0
    for i in range(LEN_1D):
        a[i] = c[i] + d[i]
        sum_val = sum_val + a[i]
        b[i] = c[i] + e[i]
        sum_val = sum_val + b[i]
    b[0] = sum_val

@dace.program
def s321_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(1, LEN_1D):
        a[i] = a[i] + a[i - 1] * b[i]

@dace.program
def s322_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    for i in range(2, LEN_1D):
        a[i] = a[i] + a[i - 1] * b[i] + a[i - 2] * c[i]

@dace.program
def s323_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    for i in range(1, LEN_1D):
        a[i] = b[i - 1] + c[i] * d[i]
        b[i] = a[i] + c[i] * e[i]

@dace.program
def s331_d_single(a: dace.float64[LEN_1D], b: dace.float64[2]):
    j = -1
    j = -1
    for i in range(LEN_1D):
        if a[i] < 0.0:
            j = i
    b[0] = j

@dace.program
def s341_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    j = -1
    for i in range(LEN_1D):
        if b[i] > 0.0:
            j = j + 1
            a[j] = b[i]

@dace.program
def s342_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    j = -1
    for i in range(LEN_1D):
        if a[i] > 0.0:
            j = j + 1
            a[i] = b[j]

@dace.program
def s351_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    alpha = c[0]
    for i in range(0, LEN_1D, 4):
        a[i] = a[i] + alpha * b[i]
        a[i + 1] = a[i + 1] + alpha * b[i + 1]
        a[i + 2] = a[i + 2] + alpha * b[i + 2]
        a[i + 3] = a[i + 3] + alpha * b[i + 3]

@dace.program
def s352_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[2]):
    dot = 0.0
    dot = 0.0
    for i in range(0, LEN_1D, 4):
        dot = dot + (a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3] +
                     a[i + 4] * b[i + 4])
    c[0] = dot

@dace.program
def s4117_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
):
    for i in range(LEN_1D):
        j = i // 2
        a[i] = b[i] + c[j] * d[i]

@dace.program
def s421_d_single(a: dace.float64[LEN_1D], flat_2d_array: dace.float64[LEN_1D]):
    for i in range(LEN_1D - 1):
        flat_2d_array[i] = flat_2d_array[i + 1] + a[i]

@dace.program
def s422_d_single(a: dace.float64[LEN_1D], flat_2d_array: dace.float64[LEN_1D * LEN_1D]):
    for i in range(LEN_1D):
        flat_2d_array[4 + i] = flat_2d_array[8 + i] + a[i]

@dace.program
def s423_d_single(a: dace.float64[LEN_1D], flat_2d_array: dace.float64[LEN_1D]):
    vl = 64
    for i in range(LEN_1D - 1):
        flat_2d_array[i + 1] = flat_2d_array[vl + i] + a[i]

@dace.program
def s424_d_single(a: dace.float64[LEN_1D], xx: dace.float64[LEN_1D], flat: dace.float64[LEN_1D]):
    for i in range(LEN_1D - 1):
        xx[i + 1] = flat[i] + a[i]

@dace.program
def s443_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
):
    for i in range(LEN_1D):
        if d[i] <= 0.0:
            a[i] = a[i] + b[i] * c[i]
        else:
            a[i] = a[i] + b[i] * b[i]

@dace.program
def s451_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    for i in range(LEN_1D):
        a[i] = sin(b[i]) + cos(c[i])

@dace.program
def s452_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    for i in range(LEN_1D):
        a[i] = b[i] + c[i] * (i + 1)

@dace.program
def s471_d_single(
    x: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    for i in range(LEN_1D):
        x[i] = b[i] + d[i] * d[i]
        b[i] = c[i] + d[i] * e[i]

_KERNELS = [
    (s116_d_single, ['a']),
    (s1161_d_single, ['a', 'b', 'c', 'd', 'e']),
    (s1213_d_single, ['a', 'b', 'c', 'd']),
    (s1221_d_single, ['a', 'b']),
    (s123_d_single, ['a', 'b', 'c', 'd', 'e']),
    (s124_d_single, ['a', 'b', 'c', 'd', 'e']),
    (s1251_d_single, ['a', 'b', 'c', 'd', 'e']),
    (s1279_d_single, ['a', 'b', 'c', 'd', 'e']),
    (s128_d_single, ['a', 'b', 'c', 'd']),
    (s1281_d_single, ['a', 'b', 'c', 'd', 'e']),
    (s1421_d_single, ['b', 'a']),
    (s151_d_single, ['a', 'b']),
    (s152_d_single, ['a', 'b', 'c', 'd', 'e']),
    (s161_d_single, ['a', 'b', 'c', 'd', 'e']),
    (s176_d_single, ['a', 'b', 'c']),
    (s221_d_single, ['a', 'b', 'c', 'd']),
    (s222_d_single, ['a', 'b', 'c', 'e']),
    (s2244_d_single, ['a', 'b', 'c', 'e']),
    (s2251_d_single, ['a', 'b', 'c', 'd', 'e']),
    (s242_d_single, ['a', 'b', 'c', 'd']),
    (s253_d_single, ['a', 'b', 'c', 'd']),
    (s254_d_single, ['a', 'b']),
    (s255_d_single, ['a', 'b']),
    (s261_d_single, ['a', 'b', 'c', 'd']),
    (s2710_d_single, ['a', 'b', 'c', 'd', 'e', 'x']),
    (s2711_d_single, ['a', 'b', 'c']),
    (s2712_d_single, ['a', 'b', 'c']),
    (s273_d_single, ['a', 'b', 'c', 'd', 'e']),
    (s274_d_single, ['a', 'b', 'c', 'd', 'e']),
    (s276_d_single, ['a', 'b', 'c', 'd']),
    (s277_d_single, ['a', 'b', 'c', 'd', 'e']),
    (s278_d_single, ['a', 'b', 'c', 'd', 'e']),
    (s279_d_single, ['a', 'b', 'c', 'd', 'e']),
    (s281_d_single, ['a', 'b', 'c']),
    (s291_d_single, ['a', 'b']),
    (s292_d_single, ['a', 'b']),
    (s3111_d_single, ['a', 'b']),
    (s31111_d_single, ['a', 'b']),
    (s3112_d_single, ['a', 'b']),
    (s3113_d_single, ['a', 'b']),
    (s313_d_single, ['a', 'b', 'dot']),
    (s314_d_single, ['a', 'result']),
    (s315_d_single, ['a', 'result']),
    (s316_d_single, ['a', 'result']),
    (s319_d_single, ['a', 'b', 'c', 'd', 'e']),
    (s321_d_single, ['a', 'b']),
    (s322_d_single, ['a', 'b', 'c']),
    (s323_d_single, ['a', 'b', 'c', 'd', 'e']),
    (s331_d_single, ['a', 'b']),
    (s341_d_single, ['a', 'b']),
    (s342_d_single, ['a', 'b']),
    (s351_d_single, ['a', 'b', 'c']),
    (s352_d_single, ['a', 'b', 'c']),
    (s4117_d_single, ['a', 'b', 'c', 'd']),
    (s421_d_single, ['a', 'flat_2d_array']),
    (s422_d_single, ['a', 'flat_2d_array']),
    (s423_d_single, ['a', 'flat_2d_array']),
    (s424_d_single, ['a', 'xx', 'flat']),
    (s443_d_single, ['a', 'b', 'c', 'd']),
    (s451_d_single, ['a', 'b', 'c']),
    (s452_d_single, ['a', 'b', 'c']),
    (s471_d_single, ['x', 'b', 'c', 'd', 'e']),
]


def _allocate(name: str, n: int) -> np.ndarray:
    """Length-1 outputs are written via subset ``[0]`` on a full
    ``LEN_1D`` array; allocate full ``n`` so kernels using either
    ``out[0]`` or ``out[i]`` can do so without OOB."""
    if name in ("result", "sum_out", "dot_out", "max_out", "min_out", "prod"):
        return np.zeros(n, dtype=np.float64)
    return np.random.rand(n).astype(np.float64)


_MATRIX, _IDS = build_tsvc_matrix(_KERNELS, (64, 65))


@pytest.mark.parametrize("kernel,argnames,remainder_strategy,branch_mode,len_1d_val", _MATRIX, ids=_IDS)
def test_tsvc_1d_bulk(kernel, argnames, remainder_strategy, branch_mode, len_1d_val):
    arrays_ref = {name: _allocate(name, len_1d_val) for name in argnames}
    arrays_vec = {name: arr.copy() for name, arr in arrays_ref.items()}

    sdfg_name = f"{kernel.name}_1db_{branch_mode}_{remainder_strategy}_{len_1d_val}"
    # Deep-copy before any mutation: to_sdfg() may return a shared cached
    # SDFG a prior variant already mutated in place.
    sdfg = copy.deepcopy(kernel.to_sdfg(simplify=False))
    sdfg.name = sdfg_name + "_ref"
    sdfg.simplify(validate=True, validate_all=True)
    sdfg.apply_transformations_repeated(LoopToMap())
    sdfg.simplify()

    vsdfg = copy.deepcopy(sdfg)
    vsdfg.name = sdfg_name + "_vec"

    if branch_mode == "fp_factor":
        branch_kwargs = dict(use_fp_factor=True, branch_normalization=False)
    else:
        branch_kwargs = dict(use_fp_factor=False, branch_normalization=True)

    try:
        VectorizeCPU(vector_width=8,
                     fail_on_unvectorizable=False,
                     remainder_strategy=remainder_strategy,
                     **branch_kwargs).apply_pass(vsdfg, {})
    except NotImplementedError as ex:
        pytest.skip(f"vectorize NotImplementedError on {kernel.name}: {ex}")

    c_ref = sdfg.compile()
    c_vec = vsdfg.compile()

    c_ref(**arrays_ref, LEN_1D=len_1d_val)
    c_vec(**arrays_vec, LEN_1D=len_1d_val)

    for name in argnames:
        diff = np.max(np.abs(arrays_ref[name] - arrays_vec[name]))
        assert diff < 1e-10, f"{kernel.name}/{name}: max abs diff = {diff}"
