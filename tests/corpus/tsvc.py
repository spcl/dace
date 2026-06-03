# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""TSVC kernel corpus -- kernels + selection + input generation only.

The single source of the 151 TSVC ``_d_single`` kernels (the stripped, no-ITERATIONS
forms copied from VectraArtifacts). Its sole responsibility is to *provide the Python
inputs to generate and run test SDFGs*: the kernel ``@dace.program`` s, a
:func:`collect` selector over them, and faithful per-kernel input allocation. It
contains no transform, no config matrix and no assertion -- each consumer (the
canonicalize, loop_to_map and vectorization test files) owns those.

Register a kernel with the ``tsvc_kernel`` decorator *outside* ``dace.program``; it
is an identity registrar (appends a :class:`TSVCKernel` descriptor and returns the
program unchanged), so existing transforms see the same SDFG.

The per-arg shape codes in ``args`` are documentation/tags only -- :func:`allocate`
derives the real shapes and dtypes from the kernel's own SDFG ``arglist()``.

``regime`` is ``"1d"`` (sweep ``LEN_1D`` in ``{64, 65}``, ``LEN_2D`` fixed at 16 for
any 2D scratch) or ``"2d"`` (sweep ``LEN_2D`` in ``{16, 17}``). Intrinsic ``tags``
are derived at registration (``branch``/``reduction``/``gather``/``2d``) for
:func:`collect`.
"""
import copy
import dataclasses
import inspect
import os
import re
import textwrap
import ast
from typing import Callable, Dict, FrozenSet, List, Optional, Tuple
# Some kernels (e.g. s451) call ``sin``/``cos`` unqualified; the original
# VectraArtifacts source imports them here, so do the same.
from math import sin, cos  # noqa: F401  -- resolved unqualified inside kernel bodies

import numpy as np

import dace

LEN_1D = dace.symbol("LEN_1D")
LEN_2D = dace.symbol("LEN_2D")
#: Scalar multiplier symbol used by a few kernels (e.g. ``vpvts``).
S = dace.symbol("S")
VLEN = 8

#: Fixed ``LEN_2D`` for the 2D scratch arrays of an otherwise 1D-swept kernel.
LEN_2D_FIXED = 16
#: Value bound to ``S`` when a kernel uses it. Irrelevant to a reference-vs-candidate
#: comparison (both sides see the same ``S``); just must be concrete.
S_VALUE = 2

#: Arg codes for gather-index arrays: read-only, never value-compared.
INDEX_CODES = frozenset({"I1", "I2"})
#: Arg codes whose array extent needs the ``LEN_1D`` symbol.
_LEN_1D_CODES = frozenset({"F1", "I1"})
#: Arg codes whose array extent needs the ``LEN_2D`` symbol.
_LEN_2D_CODES = frozenset({"F2", "F2v", "FL2", "F1L2", "I2"})


@dataclasses.dataclass(frozen=True)
class TSVCKernel:
    """One corpus kernel and the metadata needed to instantiate it."""
    program: Callable  #: the ``@dace.program`` (unchanged by registration)
    args: Dict[str, str]  #: arg name -> shape code (documentation/tags only)
    params: Dict[str, object]  #: scalar parameter name -> value ("N4" -> LEN//4)
    regime: str  #: "1d" or "2d"
    tags: FrozenSet[str]  #: intrinsic, derived tags for :func:`collect`

    @property
    def name(self) -> str:
        return self.program.f.__name__


_REGISTRY: List[TSVCKernel] = []


def _derived_tags(program, args: Dict[str, str]) -> FrozenSet[str]:
    """Intrinsic tags read off the kernel: ``branch``/``reduction``/``gather``."""
    tags = set()
    codes = set(args.values())
    if codes & INDEX_CODES:
        tags.add("gather")
    if "R1" in codes:
        tags.add("reduction")
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(program.f)))
        if any(isinstance(n, ast.If) for n in ast.walk(tree)):
            tags.add("branch")
    except Exception:  # noqa: BLE001 -- cannot classify; leave branch unset
        pass
    return frozenset(tags)


def tsvc_kernel(*,
                args: Dict[str, str],
                params: Optional[Dict[str, object]] = None,
                regime: str = "1d",
                tags: Tuple[str, ...] = ()):
    """Register ``program`` in the corpus and return it unchanged.

    :param args: arg name -> shape code.
    :param params: scalar parameter name -> value.
    :param regime: ``"1d"`` or ``"2d"``.
    :param tags: extra tags to union with the derived ones.
    """

    def deco(program):
        regime_tag = {"2d"} if regime == "2d" else set()
        all_tags = _derived_tags(program, args) | set(tags) | regime_tag
        _REGISTRY.append(TSVCKernel(program, args, dict(params or {}), regime, frozenset(all_tags)))
        return program

    return deco


def collect(*,
            regime: Optional[str] = None,
            tags: Optional[Tuple[str, ...]] = None,
            name: Optional[str] = None) -> List[TSVCKernel]:
    """Filtered view of the corpus.

    :param regime: keep only this regime (``"1d"``/``"2d"``).
    :param tags: keep kernels carrying *all* of these tags.
    :param name: keep kernels whose name matches this regex.
    :returns: the matching kernels, in registration order.
    """
    out = list(_REGISTRY)
    if regime is not None:
        out = [k for k in out if k.regime == regime]
    if tags:
        want = set(tags)
        out = [k for k in out if want <= k.tags]
    if name is not None:
        out = [k for k in out if re.search(name, k.name)]
    return out


@tsvc_kernel(args={'a': 'F1', 'b': 'F1'}, params={}, regime='1d')
@dace.program
def s000_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(LEN_1D):
        a[i] = b[i] + 1.0


@tsvc_kernel(args={'a': 'F1', 'b': 'F1'}, params={}, regime='1d')
@dace.program
def s111_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(1, LEN_1D, 2):
        a[i] = a[i - 1] + b[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1'}, params={}, regime='1d')
@dace.program
def s1111_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
):
    for i in dace.map[0:LEN_1D // 2]:
        a[2 * i] = (c[i] * b[i] + d[i] * b[i] + c[i] * c[i] + d[i] * b[i] + d[i] * c[i])


@tsvc_kernel(args={'a': 'F1', 'b': 'F1'}, params={}, regime='1d')
@dace.program
def s1112_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(LEN_1D - 1, -1, -1):
        a[i] = b[i] + 1.0


@tsvc_kernel(args={'a': 'F1', 'b': 'F1'}, params={}, regime='1d')
@dace.program
def s1113_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(LEN_1D):
        a[i] = a[LEN_1D // 2] + b[i]


@tsvc_kernel(args={'aa': 'F2', 'bb': 'F2', 'cc': 'F2'}, params={}, regime='2d')
@dace.program
def s1115_d_single(
    aa: dace.float64[LEN_2D, LEN_2D],
    bb: dace.float64[LEN_2D, LEN_2D],
    cc: dace.float64[LEN_2D, LEN_2D],
):
    for i in range(LEN_2D):
        for j in range(LEN_2D):
            aa[i, j] = aa[i, j] * cc[j, i] + bb[i, j]


@tsvc_kernel(args={'aa': 'F2', 'bb': 'F2'}, params={}, regime='2d')
@dace.program
def s1119_d_single(aa: dace.float64[LEN_2D, LEN_2D], bb: dace.float64[LEN_2D, LEN_2D]):
    for i in range(1, LEN_2D):
        for j in range(LEN_2D):
            aa[i, j] = aa[i - 1, j] + bb[i, j]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1'}, params={}, regime='1d')
@dace.program
def s112_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(LEN_1D - 2, -1, -1):
        a[i + 1] = a[i] + b[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1'}, params={}, regime='1d')
@dace.program
def s113_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(1, LEN_1D):
        a[i] = a[0] + b[i]


@tsvc_kernel(args={'aa': 'F2', 'bb': 'F2'}, params={}, regime='2d')
@dace.program
def s114_d_single(aa: dace.float64[LEN_2D, LEN_2D], bb: dace.float64[LEN_2D, LEN_2D]):
    for i in range(LEN_2D // VLEN):
        for j in range(i * VLEN):
            aa[i, j] = aa[j, i] + bb[i, j]


@tsvc_kernel(args={'a': 'F2v', 'aa': 'F2'}, params={}, regime='2d')
@dace.program
def s115_d_single(a: dace.float64[LEN_2D], aa: dace.float64[LEN_2D, LEN_2D]):
    for j in range(LEN_2D):
        for i in range(j + 1, LEN_2D):
            a[i] = a[i] - aa[j, i] * a[j]


@tsvc_kernel(args={'a': 'F1'}, params={}, regime='1d')
@dace.program
def s116_d_single(a: dace.float64[LEN_1D]):
    for i in range(0, LEN_1D - 4, 4):
        a[i] = a[i + 1] * a[i]
        a[i + 1] = a[i + 2] * a[i + 1]
        a[i + 2] = a[i + 3] * a[i + 2]
        a[i + 3] = a[i + 4] * a[i + 3]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1', 'e': 'F1'}, params={}, regime='1d')
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


@tsvc_kernel(args={'a': 'F2v', 'bb': 'F2'}, params={}, regime='2d')
@dace.program
def s118_d_single(a: dace.float64[LEN_2D], bb: dace.float64[LEN_2D, LEN_2D]):
    for i in range(1, LEN_2D):
        for j in range(0, i):
            a[i] = a[i] + bb[j, i] * a[i - j - 1]


@tsvc_kernel(args={'aa': 'F2', 'bb': 'F2'}, params={}, regime='2d')
@dace.program
def s119_d_single(aa: dace.float64[LEN_2D, LEN_2D], bb: dace.float64[LEN_2D, LEN_2D]):
    for i in range(1, LEN_2D):
        for j in range(1, LEN_2D):
            aa[i, j] = aa[i - 1, j - 1] + bb[i, j]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1'}, params={}, regime='1d')
@dace.program
def s121_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(LEN_1D - 1):
        j = i + 1
        a[i] = a[j] + b[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1'}, params={}, regime='1d')
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


@tsvc_kernel(args={'a': 'F1', 'b': 'F1'}, params={'n1': 1, 'n3': 2}, regime='1d')
@dace.program
def s122_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], n1: dace.int64, n3: dace.int64):
    j = 1
    k = 0
    for i in range(n1 - 1, LEN_1D, n3):
        k = k + j
        a[i] = a[i] + b[LEN_1D - k]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1'}, params={}, regime='1d')
@dace.program
def s1221_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(4, LEN_1D):
        b[i] = b[i - 4] + a[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1', 'e': 'F1'}, params={}, regime='1d')
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


@tsvc_kernel(args={'aa': 'F2', 'bb': 'F2', 'cc': 'F2'}, params={}, regime='2d')
@dace.program
def s1232_d_single(
    aa: dace.float64[LEN_2D, LEN_2D],
    bb: dace.float64[LEN_2D, LEN_2D],
    cc: dace.float64[LEN_2D, LEN_2D],
):
    for j in range(LEN_2D):
        for i in range(j * VLEN, LEN_2D):
            aa[i, j] = bb[i, j] + cc[i, j]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1', 'e': 'F1'}, params={}, regime='1d')
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


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1'}, params={}, regime='1d')
@dace.program
def s1244_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
):
    for i in range(LEN_1D - 1):
        a[i] = b[i] + c[i] * c[i] + b[i] * b[i] + c[i]
        d[i] = a[i] + a[i + 1]


@tsvc_kernel(args={'aa': 'F2', 'bb': 'F2', 'cc': 'F2', 'flat_2d_array': 'FL2'}, params={}, regime='2d')
@dace.program
def s125_d_single(
    flat_2d_array: dace.float64[LEN_2D * LEN_2D],
    aa: dace.float64[LEN_2D, LEN_2D],
    bb: dace.float64[LEN_2D, LEN_2D],
    cc: dace.float64[LEN_2D, LEN_2D],
):
    k = -1
    for i in range(LEN_2D):
        for j in range(LEN_2D):
            k = k + 1
            flat_2d_array[k] = aa[i, j] + bb[i, j] * cc[i, j]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1', 'e': 'F1'}, params={}, regime='1d')
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


@tsvc_kernel(args={'bb': 'F2', 'cc': 'F2', 'flat_2d_array': 'FL2'}, params={}, regime='2d')
@dace.program
def s126_d_single(
    bb: dace.float64[LEN_2D, LEN_2D],
    flat_2d_array: dace.float64[LEN_2D * LEN_2D],
    cc: dace.float64[LEN_2D, LEN_2D],
):
    k = 1
    for i in range(LEN_2D):
        for j in range(1, LEN_2D):
            bb[j, i] = bb[j - 1, i] + flat_2d_array[k - 1] * cc[j, i]
            k = k + 1
        k = k + 1


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1', 'e': 'F1'}, params={}, regime='1d')
@dace.program
def s127_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    for i in dace.map[0:LEN_1D // 2]:
        a[2 * i] = b[i] + c[i] * d[i]
        a[2 * i + 1] = b[i] + d[i] * e[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1', 'e': 'F1'}, params={}, regime='1d')
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


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1'}, params={}, regime='1d')
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


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1', 'e': 'F1'}, params={}, regime='1d')
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


@tsvc_kernel(args={'a': 'F1', 'b': 'F1'}, params={}, regime='1d')
@dace.program
def s131_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(LEN_1D - 1):
        a[i] = a[i + 1] + b[i]


@tsvc_kernel(args={'aa': 'F2', 'bb': 'F22'}, params={}, regime='2d')
@dace.program
def s13110_d_single(aa: dace.float64[LEN_2D, LEN_2D], bb: dace.float64[2, 2]):
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


@tsvc_kernel(args={'aa': 'F2', 'b': 'F2v', 'c': 'F2v'}, params={}, regime='2d')
@dace.program
def s132_d_single(aa: dace.float64[LEN_2D, LEN_2D], b: dace.float64[LEN_2D], c: dace.float64[LEN_2D]):
    for i in range(1, LEN_2D):
        aa[0, i] = aa[1, i - 1] + b[i] * c[1]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1'}, params={}, regime='1d')
@dace.program
def s1351_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    for i in range(LEN_1D):
        a[i] = b[i] + c[i]


@tsvc_kernel(args={'bb': 'F2', 'flat_2d_array': 'FL2'}, params={}, regime='2d')
@dace.program
def s141_d_single(bb: dace.float64[LEN_2D, LEN_2D], flat_2d_array: dace.float64[LEN_2D * LEN_2D]):
    for i in range(LEN_2D):
        k = (i + 1) * i // 2 + i
        for j in range(i, LEN_2D):
            flat_2d_array[k] = flat_2d_array[k] + bb[j, i]
            k = k + j + 1


@tsvc_kernel(args={'a': 'F1', 'b': 'F1'}, params={}, regime='1d')
@dace.program
def s1421_d_single(b: dace.float64[LEN_1D], a: dace.float64[LEN_1D]):
    half = LEN_1D // 2
    for i in range(half):
        b[i] = b[half + i] + a[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1'}, params={}, regime='1d')
@dace.program
def s151_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(LEN_1D - 1):
        a[i] = a[i + 1] + b[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1', 'e': 'F1'}, params={}, regime='1d')
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


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1', 'e': 'F1'}, params={}, regime='1d')
@dace.program
def s161_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    for i in range(LEN_1D - 1):
        if b[i] < 0.0:
            c[i + 1] = a[i] + d[i] * d[i]
        else:
            a[i] = c[i] + d[i] * e[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1'}, params={'k': 3}, regime='1d')
@dace.program
def s162_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    k: dace.int64,
):
    if k > 0:
        for i in range(0, LEN_1D - k):
            a[i] = a[i + k] + b[i] * c[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1'}, params={'inc': 1}, regime='1d')
@dace.program
def s171_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], inc: dace.int64):
    for i in range(LEN_1D):
        a[i * inc] = a[i * inc] + b[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1'}, params={'n1': 1, 'n3': 2}, regime='1d')
@dace.program
def s172_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], n1: dace.int64, n3: dace.int64):
    for i in range(n1 - 1, LEN_1D, n3):
        a[i] = a[i] + b[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1'}, params={}, regime='1d')
@dace.program
def s173_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(LEN_1D // 2):
        a[i + (LEN_1D // 2)] = a[i] + b[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1'}, params={'M': 'N4'}, regime='1d')
@dace.program
def s174_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], M: dace.int64):
    for i in range(M):
        a[i + M] = a[i] + b[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1'}, params={'inc': 2}, regime='1d')
@dace.program
def s175_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], inc: dace.int64):
    for i in range(0, LEN_1D - inc, inc):
        a[i] = a[i + inc] + b[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1'}, params={}, regime='1d')
@dace.program
def s176_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    m = LEN_1D // 2
    for j in range(LEN_1D // 2):
        for i in range(m):
            a[i] = a[i] + b[i + m - j - 1] * c[j]


@tsvc_kernel(args={'aa': 'F2', 'bb': 'F2', 'cc': 'F2'}, params={}, regime='2d')
@dace.program
def s2101_d_single(
    aa: dace.float64[LEN_2D, LEN_2D],
    bb: dace.float64[LEN_2D, LEN_2D],
    cc: dace.float64[LEN_2D, LEN_2D],
):
    for i in range(LEN_2D):
        aa[i, i] = aa[i, i] + bb[i, i] * cc[i, i]


@tsvc_kernel(args={'aa': 'F2'}, params={}, regime='2d')
@dace.program
def s2102_d_single(aa: dace.float64[LEN_2D, LEN_2D]):
    for i in range(LEN_2D):
        for j in range(LEN_2D):
            aa[j, i] = 0.0
        aa[i, i] = 1.0


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1', 'e': 'F1'}, params={}, regime='1d')
@dace.program
def s211_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    for i in range(1, LEN_1D - 1):
        a[i] = b[i - 1] + c[i] * d[i]
        b[i] = b[i + 1] - e[i] * d[i]


@tsvc_kernel(args={'aa': 'F2'}, params={}, regime='2d')
@dace.program
def s2111_d_single(aa: dace.float64[LEN_2D, LEN_2D]):
    for j in range(1, LEN_2D):
        for i in range(1, LEN_2D):
            aa[j, i] = (aa[j, i - 1] + aa[j - 1, i]) / 1.9


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1'}, params={}, regime='1d')
@dace.program
def s212_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
):
    for i in range(LEN_1D - 1):
        a[i] = a[i] * c[i]
        b[i] = b[i] + (a[i + 1] * d[i])


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1'}, params={}, regime='1d')
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


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'e': 'F1'}, params={}, regime='1d')
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


@tsvc_kernel(args={'aa': 'F2', 'bb': 'F2', 'cc': 'F2'}, params={}, regime='2d')
@dace.program
def s2233_d_single(
    aa: dace.float64[LEN_2D, LEN_2D],
    bb: dace.float64[LEN_2D, LEN_2D],
    cc: dace.float64[LEN_2D, LEN_2D],
):
    for i in range(8, LEN_2D):
        for j in range(8, LEN_2D):
            aa[j, i] = aa[j - 1, i] + cc[j, i]
        for j in range(8, LEN_2D):
            bb[i, j] = bb[i - 1, j] + cc[i, j]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'e': 'F1'}, params={}, regime='1d')
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


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1', 'e': 'F1'}, params={}, regime='1d')
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


@tsvc_kernel(args={
    'a': 'F2v',
    'aa': 'F2',
    'b': 'F2v',
    'bb': 'F2',
    'c': 'F2v',
    'cc': 'F2',
    'd': 'F2v'
},
             params={},
             regime='2d')
@dace.program
def s2275_d_single(
    a: dace.float64[LEN_2D],
    b: dace.float64[LEN_2D],
    c: dace.float64[LEN_2D],
    d: dace.float64[LEN_2D],
    aa: dace.float64[LEN_2D, LEN_2D],
    bb: dace.float64[LEN_2D, LEN_2D],
    cc: dace.float64[LEN_2D, LEN_2D],
):
    for i in range(LEN_2D):
        for j in range(LEN_2D):
            aa[j, i] = aa[j, i] + bb[j, i] * cc[j, i]
        a[i] = b[i] + c[i] * d[i]


@tsvc_kernel(args={'aa': 'F2', 'bb': 'F2'}, params={}, regime='2d')
@dace.program
def s231_d_single(aa: dace.float64[LEN_2D, LEN_2D], bb: dace.float64[LEN_2D, LEN_2D]):
    for i in range(LEN_2D):
        for j in range(1, LEN_2D):
            aa[j, i] = aa[j - 1, i] + bb[j, i]


@tsvc_kernel(args={'aa': 'F2', 'bb': 'F2'}, params={}, regime='2d')
@dace.program
def s232_d_single(aa: dace.float64[LEN_2D, LEN_2D], bb: dace.float64[LEN_2D, LEN_2D]):
    for j in range(1, LEN_2D):
        for i in range(1, j + 1):
            aa[j, i] = aa[j, i - 1] * aa[j, i - 1] + bb[j, i]


@tsvc_kernel(args={'aa': 'F2', 'bb': 'F2', 'cc': 'F2'}, params={}, regime='2d')
@dace.program
def s233_d_single(
    aa: dace.float64[LEN_2D, LEN_2D],
    bb: dace.float64[LEN_2D, LEN_2D],
    cc: dace.float64[LEN_2D, LEN_2D],
):
    for i in range(8, LEN_2D):
        for j in range(8, LEN_2D):
            aa[j, i] = aa[j - 1, i] + cc[j, i]
        for j in range(8, LEN_2D):
            bb[j, i] = bb[j, i - 1] + cc[j, i]


@tsvc_kernel(args={'a': 'F2v', 'aa': 'F2', 'b': 'F2v', 'bb': 'F2', 'c': 'F2v'}, params={}, regime='2d')
@dace.program
def s235_d_single(
    a: dace.float64[LEN_2D],
    b: dace.float64[LEN_2D],
    c: dace.float64[LEN_2D],
    aa: dace.float64[LEN_2D, LEN_2D],
    bb: dace.float64[LEN_2D, LEN_2D],
):
    for i in range(LEN_2D):
        a[i] = a[i] + b[i] * c[i]
        for j in range(1, LEN_2D):
            aa[j, i] = aa[j - 1, i] + bb[j, i] * a[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1'}, params={}, regime='1d')
@dace.program
def s241_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
):
    for i in range(LEN_1D - 1):
        a[i] = b[i] * c[i] * d[i]
        b[i] = a[i] * a[i + 1] * d[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1'}, params={}, regime='1d')
@dace.program
def s242_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
):
    for i in range(1, LEN_1D):
        a[i] = a[i - 1] + 0.5 + 1.0 + b[i] + c[i] + d[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1', 'e': 'F1'}, params={}, regime='1d')
@dace.program
def s243_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    for i in range(LEN_1D - 1):
        a[i] = b[i] + c[i] * d[i]
        b[i] = a[i] + d[i] * e[i]
        a[i] = b[i] + a[i + 1] * d[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1'}, params={}, regime='1d')
@dace.program
def s244_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
):
    for i in range(LEN_1D - 1):
        a[i] = b[i] + c[i] * d[i]
        b[i] = c[i] + b[i]
        a[i + 1] = b[i] + a[i + 1] * d[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1'}, params={}, regime='1d')
@dace.program
def s251_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
):
    for i in range(LEN_1D):
        s = b[i] + c[i] * d[i]
        a[i] = s * s


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1'}, params={}, regime='1d')
@dace.program
def s252_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    t = 0.0
    for i in range(LEN_1D):
        s = b[i] * c[i]
        a[i] = s + t
        t = s


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1'}, params={}, regime='1d')
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


@tsvc_kernel(args={'a': 'F1', 'b': 'F1'}, params={}, regime='1d')
@dace.program
def s254_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    x = b[LEN_1D - 1]
    for i in range(LEN_1D):
        a[i] = (b[i] + x) * 0.5
        x = b[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1'}, params={}, regime='1d')
@dace.program
def s255_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    x = b[LEN_1D - 1]
    y = b[LEN_1D - 2]
    for i in range(LEN_1D):
        a[i] = (b[i] + x + y) * 0.333
        y = x
        x = b[i]


@tsvc_kernel(args={'a': 'F2v', 'aa': 'F2', 'bb': 'F2', 'd': 'F2v'}, params={}, regime='2d')
@dace.program
def s256_d_single(
    a: dace.float64[LEN_2D],
    aa: dace.float64[LEN_2D, LEN_2D],
    bb: dace.float64[LEN_2D, LEN_2D],
    d: dace.float64[LEN_2D],
):
    for i in range(LEN_2D):
        for j in range(1, LEN_2D):
            a[j] = 1.0 - a[j - 1]
            aa[j, i] = a[j] + bb[j, i] * d[j]


@tsvc_kernel(args={'a': 'F2v', 'aa': 'F2', 'bb': 'F2'}, params={}, regime='2d')
@dace.program
def s257_d_single(
    a: dace.float64[LEN_2D],
    aa: dace.float64[LEN_2D, LEN_2D],
    bb: dace.float64[LEN_2D, LEN_2D],
):
    for i in range(8, LEN_2D):
        for j in range(LEN_2D):
            a[i] = aa[j, i] - a[i - 1]
            aa[j, i] = a[i] + bb[j, i]


@tsvc_kernel(args={'a': 'F2v', 'aa': 'F1L2', 'b': 'F2v', 'c': 'F2v', 'd': 'F2v', 'e': 'F2v'}, params={}, regime='2d')
@dace.program
def s258_d_single(
    a: dace.float64[LEN_2D],
    b: dace.float64[LEN_2D],
    c: dace.float64[LEN_2D],
    d: dace.float64[LEN_2D],
    e: dace.float64[LEN_2D],
    aa: dace.float64[1, LEN_2D],
):
    s = 0.0
    for i in range(LEN_2D):
        if a[i] > 0.0:
            s = d[i] * d[i]
        b[i] = s * c[i] + d[i]
        e[i] = (s + 1.0) * aa[0, i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1'}, params={}, regime='1d')
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


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1'}, params={}, regime='1d')
@dace.program
def s271_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    for i in range(LEN_1D):
        if b[i] > 0.0:
            a[i] = a[i] + b[i] * c[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1', 'e': 'F1', 'x': 'F1'}, params={}, regime='1d')
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


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1'}, params={}, regime='1d')
@dace.program
def s2711_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    for i in range(LEN_1D):
        if b[i] != 0.0:
            a[i] = a[i] + b[i] * c[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1'}, params={}, regime='1d')
@dace.program
def s2712_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    for i in range(LEN_1D):
        if a[i] > b[i]:
            a[i] = a[i] + b[i] * c[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1', 'e': 'F1'}, params={'threshold': 0}, regime='1d')
@dace.program
def s272_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
    threshold: dace.int64,
):
    for i in range(LEN_1D):
        if e[i] >= threshold:
            a[i] = a[i] + c[i] * d[i]
            b[i] = b[i] + c[i] * c[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1', 'e': 'F1'}, params={}, regime='1d')
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


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1', 'e': 'F1'}, params={}, regime='1d')
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


@tsvc_kernel(args={'aa': 'F2', 'bb': 'F2', 'cc': 'F2'}, params={}, regime='2d')
@dace.program
def s275_d_single(
    aa: dace.float64[LEN_2D, LEN_2D],
    bb: dace.float64[LEN_2D, LEN_2D],
    cc: dace.float64[LEN_2D, LEN_2D],
):
    for i in range(LEN_2D):
        if aa[0, i] > 0.0:
            for j in range(1, LEN_2D):
                aa[j, i] = aa[j - 1, i] + bb[j, i] * cc[j, i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1'}, params={}, regime='1d')
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


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1', 'e': 'F1'}, params={}, regime='1d')
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


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1', 'e': 'F1'}, params={}, regime='1d')
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


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1', 'e': 'F1'}, params={}, regime='1d')
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


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1'}, params={}, regime='1d')
@dace.program
def s281_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    for i in range(LEN_1D):
        x = a[LEN_1D - i - 1] + b[i] * c[i]
        a[i] = x - 1.0
        b[i] = x


@tsvc_kernel(args={'a': 'F1', 'b': 'F1'}, params={}, regime='1d')
@dace.program
def s291_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    a[0] = (b[0] + b[LEN_1D - 1]) * 0.5
    for i in range(1, LEN_1D):
        a[i] = (b[i] + b[i - 1]) * 0.5


@tsvc_kernel(args={'a': 'F1', 'b': 'F1'}, params={}, regime='1d')
@dace.program
def s292_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    a[0] = (b[0] + b[LEN_1D - 1] + b[LEN_1D - 2]) * 0.333
    a[1] = (b[1] + b[0] + b[LEN_1D - 1]) * 0.333
    for i in range(2, LEN_1D):
        a[i] = (b[i] + b[i - 1] + b[i - 2]) * 0.333


@tsvc_kernel(args={'a': 'F1'}, params={}, regime='1d')
@dace.program
def s293_d_single(a: dace.float64[LEN_1D]):
    a0 = a[0]
    for i in range(LEN_1D):
        a[i] = a0


@tsvc_kernel(args={'a': 'F1', 'sum_out': 'F1'}, params={}, regime='1d')
@dace.program
def s311_d_single(a: dace.float64[LEN_1D], sum_out: dace.float64[LEN_1D]):
    sum_out[0] = 0.0
    for i in range(LEN_1D):
        sum_out[0] = sum_out[0] + a[i]


@tsvc_kernel(args={'aa': 'F2', 'bb': 'F22'}, params={}, regime='2d')
@dace.program
def s3110_d_single(aa: dace.float64[LEN_2D, LEN_2D], bb: dace.float64[2, 2]):
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


@tsvc_kernel(args={'a': 'F1', 'b': 'F2c'}, params={}, regime='1d')
@dace.program
def s3111_d_single(a: dace.float64[LEN_1D], b: dace.float64[2]):
    sum_val = 0.0
    for i in range(LEN_1D):
        if a[i] > 0.0:
            sum_val = sum_val + a[i]
    b[0] = sum_val


@tsvc_kernel(args={'a': 'F1', 'b': 'F2c'}, params={}, regime='1d')
@dace.program
def s31111_d_single(a: dace.float64[LEN_1D], b: dace.float64[2]):
    sum_val = 0.0
    for base in range(0, LEN_1D - 3, 4):
        partial = 0.0
        partial = partial + a[base + 0]
        partial = partial + a[base + 1]
        partial = partial + a[base + 2]
        partial = partial + a[base + 3]
        sum_val = sum_val + partial
    b[0] = sum_val


@tsvc_kernel(args={'a': 'F1', 'b': 'F1'}, params={}, regime='1d')
@dace.program
def s3112_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    sum = 0.0
    for i in range(LEN_1D):
        sum = sum + a[i]
        b[i] = sum


@tsvc_kernel(args={'a': 'F1', 'b': 'F2c'}, params={}, regime='1d')
@dace.program
def s3113_d_single(a: dace.float64[LEN_1D], b: dace.float64[2]):
    maxv = dace.float64(0)
    maxv = abs(a[0])
    for i in range(LEN_1D):
        av = abs(a[i])
        if av > maxv:
            maxv = av
    b[0] = maxv


@tsvc_kernel(args={'a': 'F1', 'result': 'R1'}, params={}, regime='1d')
@dace.program
def s312_d_single(a: dace.float64[LEN_1D], result: dace.float64[1]):
    prod = 1.0
    for i in range(LEN_1D):
        prod = prod * a[i]
    result[0] = prod


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'dot': 'R1'}, params={}, regime='1d')
@dace.program
def s313_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], dot: dace.float64[1]):
    dot[0] = 0.0
    for i in range(LEN_1D):
        dot[0] = dot[0] + a[i] * b[i]


@tsvc_kernel(args={'a': 'F1', 'result': 'R1'}, params={}, regime='1d')
@dace.program
def s314_d_single(a: dace.float64[LEN_1D], result: dace.float64[1]):
    x = a[0]
    for i in range(1, LEN_1D):
        if a[i] > x:
            x = a[i]
    result[0] = x


@tsvc_kernel(args={'a': 'F1', 'result': 'R1'}, params={}, regime='1d')
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


@tsvc_kernel(args={'a': 'F1', 'result': 'R1'}, params={}, regime='1d')
@dace.program
def s316_d_single(a: dace.float64[LEN_1D], result: dace.float64[1]):
    x = a[0]
    for i in range(1, LEN_1D):
        if a[i] < x:
            x = a[i]
    result[0] = x


@tsvc_kernel(args={'q': 'F1'}, params={}, regime='1d')
@dace.program
def s317_d_single(q: dace.float64[LEN_1D]):
    q[0] = 1.0
    for i in range(LEN_1D // 2):
        q[0] = q[0] * 0.99


@tsvc_kernel(args={'a': 'F1', 'result': 'R1'}, params={'inc': 1}, regime='1d')
@dace.program
def s318_d_single(a: dace.float64[LEN_1D], result: dace.float64[1], inc: dace.int32):
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


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1', 'e': 'F1'}, params={}, regime='1d')
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


@tsvc_kernel(args={'a': 'F1', 'b': 'F1'}, params={}, regime='1d')
@dace.program
def s321_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(1, LEN_1D):
        a[i] = a[i] + a[i - 1] * b[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1'}, params={}, regime='1d')
@dace.program
def s322_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    for i in range(2, LEN_1D):
        a[i] = a[i] + a[i - 1] * b[i] + a[i - 2] * c[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1', 'e': 'F1'}, params={}, regime='1d')
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


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1', 'e': 'F1'}, params={}, regime='1d')
@dace.program
def s3251_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
):
    for i in range(LEN_1D - 1):
        a[i + 1] = b[i] + c[i]
        b[i] = c[i] * e[i]
        d[i] = a[i] * e[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F2c'}, params={}, regime='1d')
@dace.program
def s331_d_single(a: dace.float64[LEN_1D], b: dace.float64[2]):
    j = -1
    j = -1
    for i in range(LEN_1D):
        if a[i] < 0.0:
            j = i
    b[0] = j


@tsvc_kernel(args={'a': 'F1', 'result': 'R1'}, params={'threshold': 0}, regime='1d')
@dace.program
def s332_d_single(a: dace.float64[LEN_1D], result: dace.float64[1], threshold: dace.int64):
    index = -2
    value = -1.0
    for i in range(LEN_1D):
        if a[i] > threshold:
            index = i
            value = a[i]
            break
    result[0] = value + float(index)


@tsvc_kernel(args={'a': 'F1', 'b': 'F1'}, params={}, regime='1d')
@dace.program
def s341_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    j = -1
    for i in range(LEN_1D):
        if b[i] > 0.0:
            j = j + 1
            a[j] = b[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1'}, params={}, regime='1d')
@dace.program
def s342_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    j = -1
    for i in range(LEN_1D):
        if a[i] > 0.0:
            j = j + 1
            a[i] = b[j]


@tsvc_kernel(args={'aa': 'F2', 'bb': 'F2', 'flat_2d_array': 'FL2'}, params={}, regime='2d')
@dace.program
def s343_d_single(
    aa: dace.float64[LEN_2D, LEN_2D],
    bb: dace.float64[LEN_2D, LEN_2D],
    flat_2d_array: dace.float64[LEN_2D * LEN_2D],
):
    k = -1
    for i in range(LEN_2D):
        for j in range(LEN_2D):
            if bb[j, i] > 0.0:
                k = k + 1
                flat_2d_array[k] = aa[j, i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1'}, params={}, regime='1d')
@dace.program
def s351_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    alpha = c[0]
    for i in range(0, LEN_1D - 3, 4):
        a[i] = a[i] + alpha * b[i]
        a[i + 1] = a[i + 1] + alpha * b[i + 1]
        a[i + 2] = a[i + 2] + alpha * b[i + 2]
        a[i + 3] = a[i + 3] + alpha * b[i + 3]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F2c'}, params={}, regime='1d')
@dace.program
def s352_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[2]):
    dot = 0.0
    dot = 0.0
    for i in range(0, LEN_1D - 4, 5):
        dot = dot + (a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3] +
                     a[i + 4] * b[i + 4])
    c[0] = dot


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'ip': 'I1'}, params={}, regime='1d')
@dace.program
def s353_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    ip: dace.int32[LEN_1D],
):
    alpha = c[0]
    for i in range(0, LEN_1D - 3, 4):
        a[i] = a[i] + alpha * b[ip[i]]
        a[i + 1] = a[i + 1] + alpha * b[ip[i + 1]]
        a[i + 2] = a[i + 2] + alpha * b[ip[i + 2]]
        a[i + 3] = a[i + 3] + alpha * b[ip[i + 3]]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'ip': 'I1'}, params={}, regime='1d')
@dace.program
def s4112_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], ip: dace.int32[LEN_1D]):
    for i in range(LEN_1D):
        a[i] = a[i] + b[ip[i]] * 2.0


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'ip': 'I1'}, params={}, regime='1d')
@dace.program
def s4113_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    ip: dace.int32[LEN_1D],
):
    for i in range(LEN_1D):
        a[ip[i]] = b[ip[i]] + c[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd_': 'F1', 'ip': 'I1'}, params={'n1': 1}, regime='1d')
@dace.program
def s4114_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d_: dace.float64[LEN_1D],
    ip: dace.int32[LEN_1D],
    n1: dace.int32,
):
    for i in range(n1 - 1, LEN_1D):
        k = ip[i]
        a[i] = b[i] + c[LEN_1D - k - 1] * d_[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'ip': 'I1', 'sum_out': 'R1'}, params={}, regime='1d')
@dace.program
def s4115_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    ip: dace.int32[LEN_1D],
    sum_out: dace.float64[1],
):
    sum_val = 0.0
    sum_val = 0.0
    for i in range(LEN_1D):
        sum_val = sum_val + a[i] * b[ip[i]]
    sum_out[0] = sum_val


@tsvc_kernel(args={'a': 'F1', 'aa': 'F2', 'ip': 'I2', 'sum_out': 'R1'}, params={'j': 1, 'inc': 1}, regime='2d')
@dace.program
def s4116_d_single(
    a: dace.float64[LEN_1D],
    aa: dace.float64[LEN_2D, LEN_2D],
    ip: dace.int32[LEN_2D],
    j: dace.int32,
    inc: dace.int32,
    sum_out: dace.float64[1],
):
    sum_val = 0.0
    sum_val = 0.0
    for i in range(LEN_2D - 1):
        off = inc + i
        sum_val = sum_val + a[off] * aa[j - 1, ip[i]]
    sum_out[0] = sum_val


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1'}, params={}, regime='1d')
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


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1'}, params={}, regime='1d')
@dace.program
def s4121_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    for i in range(LEN_1D):
        a[i] = a[i] + b[i] * c[i]


@tsvc_kernel(args={'a': 'F1', 'flat_2d_array': 'F1'}, params={}, regime='1d')
@dace.program
def s421_d_single(a: dace.float64[LEN_1D], flat_2d_array: dace.float64[LEN_1D]):
    for i in range(LEN_1D - 1):
        flat_2d_array[i] = flat_2d_array[i + 1] + a[i]


@tsvc_kernel(args={'a': 'F1', 'flat_2d_array': 'FL2'}, params={}, regime='1d')
@dace.program
def s422_d_single(a: dace.float64[LEN_1D], flat_2d_array: dace.float64[LEN_1D * LEN_1D]):
    for i in range(LEN_1D):
        flat_2d_array[4 + i] = flat_2d_array[8 + i] + a[i]


@tsvc_kernel(args={'a': 'F1', 'flat_2d_array': 'FL2'}, params={}, regime='1d')
@dace.program
def s423_d_single(a: dace.float64[LEN_1D], flat_2d_array: dace.float64[LEN_1D * LEN_1D]):
    vl = 64
    for i in range(LEN_1D - 1):
        flat_2d_array[i + 1] = flat_2d_array[vl + i] + a[i]


@tsvc_kernel(args={'a': 'F1', 'flat': 'F1', 'xx': 'F1'}, params={}, regime='1d')
@dace.program
def s424_d_single(a: dace.float64[LEN_1D], xx: dace.float64[LEN_1D], flat: dace.float64[LEN_1D]):
    for i in range(LEN_1D - 1):
        xx[i + 1] = flat[i] + a[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1'}, params={}, regime='1d')
@dace.program
def s431_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(LEN_1D):
        a[i] = a[i] + b[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1'}, params={}, regime='1d')
@dace.program
def s441_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
):
    for i in range(LEN_1D):
        if d[i] < 0.0:
            a[i] = a[i] + b[i] * c[i]
        elif d[i] == 0.0:
            a[i] = a[i] + b[i] * b[i]
        else:
            a[i] = a[i] + c[i] * c[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1', 'e': 'F1', 'indx': 'I1'}, params={}, regime='1d')
@dace.program
def s442_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    e: dace.float64[LEN_1D],
    indx: dace.int32[LEN_1D],
):
    for i in range(LEN_1D):
        if indx[i] == 1:
            a[i] = a[i] + (b[i] * b[i])
        elif indx[i] == 2:
            a[i] = a[i] + (c[i] * c[i])
        elif indx[i] == 3:
            a[i] = a[i] + (d[i] * d[i])
        elif indx[i] == 4:
            a[i] = a[i] + (e[i] * e[i])


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1'}, params={}, regime='1d')
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


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1'}, params={}, regime='1d')
@dace.program
def s451_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    for i in range(LEN_1D):
        a[i] = sin(b[i]) + cos(c[i])


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1'}, params={}, regime='1d')
@dace.program
def s452_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    for i in range(LEN_1D):
        a[i] = b[i] + c[i] * (i + 1)


@tsvc_kernel(args={'a': 'F1', 'b': 'F1'}, params={}, regime='1d')
@dace.program
def s453_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    s = 0.0
    for i in range(LEN_1D):
        s = s + 2.0
        a[i] = s * b[i]


@tsvc_kernel(args={'b': 'F1', 'c': 'F1', 'd': 'F1', 'e': 'F1', 'x': 'F1'}, params={}, regime='1d')
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


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1'}, params={}, regime='1d')
@dace.program
def s481_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
):
    for i in range(LEN_1D):
        if d[i] < 0.0:
            break
        a[i] = a[i] + b[i] * c[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1'}, params={}, regime='1d')
@dace.program
def s482_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    for i in range(LEN_1D):
        a[i] = a[i] + b[i] * c[i]
        if c[i] > b[i]:
            break


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1', 'd': 'F1', 'ip': 'I1'}, params={}, regime='1d')
@dace.program
def s491_d_single(
    a: dace.float64[LEN_1D],
    b: dace.float64[LEN_1D],
    c: dace.float64[LEN_1D],
    d: dace.float64[LEN_1D],
    ip: dace.int32[LEN_1D],
):
    for i in range(LEN_1D):
        a[ip[i]] = b[i] + c[i] * d[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1'}, params={}, regime='1d')
@dace.program
def va_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(LEN_1D):
        a[i] = b[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'ip': 'I1'}, params={}, regime='1d')
@dace.program
def vag_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], ip: dace.int32[LEN_1D]):
    for i in range(LEN_1D):
        a[i] = b[ip[i]]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'ip': 'I1'}, params={}, regime='1d')
@dace.program
def vas_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], ip: dace.int32[LEN_1D]):
    for i in range(LEN_1D):
        a[ip[i]] = b[i]


@tsvc_kernel(args={'a': 'F2v', 'b': 'F2v', 'c': 'F2v', 'd': 'F2v', 'e': 'F2v', 'x': 'F2v'}, params={}, regime='2d')
@dace.program
def vbor_d_single(
    a: dace.float64[LEN_2D],
    b: dace.float64[LEN_2D],
    c: dace.float64[LEN_2D],
    d: dace.float64[LEN_2D],
    e: dace.float64[LEN_2D],
    x: dace.float64[LEN_2D],
):
    for i in range(LEN_2D):
        a1 = a[i]
        b1 = b[i]
        c1 = c[i]
        d1 = d[i]
        e1 = e[i]
        f1 = a[i]
        a1 = (a1 * b1 * c1 + a1 * b1 * d1 + a1 * b1 * e1 + a1 * b1 * f1 + a1 * c1 * d1 + a1 * c1 * e1 + a1 * c1 * f1 +
              a1 * d1 * e1 + a1 * d1 * f1 + a1 * e1 * f1)
        b1 = (b1 * c1 * d1 + b1 * c1 * e1 + b1 * c1 * f1 + b1 * d1 * e1 + b1 * d1 * f1 + b1 * e1 * f1)
        c1 = c1 * d1 * e1 + c1 * d1 * f1 + c1 * e1 * f1
        d1 = d1 * e1 * f1
        x[i] = a1 * b1 * c1 * d1


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'dot_out': 'F1'}, params={}, regime='1d')
@dace.program
def vdotr_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], dot_out: dace.float64[LEN_1D]):
    dot_out[0] = 0.0
    dot_out[0] = 0.0
    for i in range(LEN_1D):
        dot_out[0] = dot_out[0] + a[i] * b[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1'}, params={}, regime='1d')
@dace.program
def vif_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(LEN_1D):
        if b[i] > 0.0:
            a[i] = b[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1'}, params={}, regime='1d')
@dace.program
def vpv_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(LEN_1D):
        a[i] = a[i] + b[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1'}, params={}, regime='1d')
@dace.program
def vpvpv_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    for i in range(LEN_1D):
        a[i] = a[i] + b[i] + c[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1'}, params={}, regime='1d')
@dace.program
def vpvts_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(LEN_1D):
        a[i] = a[i] + b[i] * S


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1'}, params={}, regime='1d')
@dace.program
def vpvtv_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    for i in range(LEN_1D):
        a[i] = a[i] + b[i] * c[i]


@tsvc_kernel(args={'a': 'F1', 'sum_out': 'R1'}, params={}, regime='1d')
@dace.program
def vsumr_d_single(a: dace.float64[LEN_1D], sum_out: dace.float64[1]):
    s = 0.0
    s = 0.0
    for i in range(LEN_1D):
        s = s + a[i]
    sum_out[0] = s


@tsvc_kernel(args={'a': 'F1', 'b': 'F1'}, params={}, regime='1d')
@dace.program
def vtv_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D]):
    for i in range(LEN_1D):
        a[i] = a[i] * b[i]


@tsvc_kernel(args={'a': 'F1', 'b': 'F1', 'c': 'F1'}, params={}, regime='1d')
@dace.program
def vtvtv_d_single(a: dace.float64[LEN_1D], b: dace.float64[LEN_1D], c: dace.float64[LEN_1D]):
    for i in range(LEN_1D):
        a[i] = a[i] * b[i] * c[i]


KERNELS: List[TSVCKernel] = list(_REGISTRY)
KERNELS_BY_NAME: Dict[str, TSVCKernel] = {k.name: k for k in KERNELS}


def lengths(kernel: TSVCKernel) -> Tuple[int, int]:
    """The ``(LEN_1D, LEN_2D)`` at which to instantiate ``kernel``."""
    return (64, LEN_2D_FIXED) if kernel.regime == "1d" else (LEN_2D_FIXED, 16)


def _concrete_shape(desc, l1: int, l2: int) -> Tuple[int, ...]:
    """Concrete int shape of a data descriptor at ``LEN_1D=l1, LEN_2D=l2``."""
    subs = {LEN_1D: l1, LEN_2D: l2}
    return tuple(int(dace.symbolic.evaluate(s, subs)) for s in desc.shape)


def allocate(kernel: TSVCKernel, l1: int, l2: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Allocate one in-bounds ndarray per kernel *array* argument.

    Shapes and dtypes are read from the kernel's own SDFG ``arglist()`` -- the
    ground truth of what the compiled function expects -- so a 2D ``aa`` is sized
    ``LEN_2D x LEN_2D`` and an integer gather index is a valid ``arange`` (the
    ``args`` codes are only documentation/tags, never the allocation authority).
    Scalar parameters and ``LEN_*`` symbols are not arrays and are excluded here
    (see :func:`scalar_params` / :func:`symbols`).

    :param kernel: the corpus kernel.
    :param l1: the ``LEN_1D`` value.
    :param l2: the ``LEN_2D`` value.
    :param rng: seeded generator (a reference and a candidate run share data).
    :returns: array-arg name -> freshly allocated array.
    """
    arrays = {}
    for name, desc in kernel.program.to_sdfg(simplify=False).arglist().items():
        if not isinstance(desc, dace.data.Array):
            continue  # scalar parameter or symbol, not an input buffer
        shape = _concrete_shape(desc, l1, l2)
        np_dtype = desc.dtype.as_numpy_dtype()
        if np.issubdtype(np_dtype, np.integer):
            # Gather index: 0..n-1 in registration order is a valid in-bounds set.
            # ``.reshape`` returns a view (``arr.base is not None``); DaCe rejects
            # numpy view inputs at call time, so force a fresh standalone array
            # via ``np.array(..., copy=True)`` (``ascontiguousarray`` is a no-op
            # on an already-contiguous view, leaving ``.base`` set).
            arrays[name] = np.array((np.arange(int(np.prod(shape))) % max(shape)).astype(np_dtype).reshape(shape),
                                    copy=True)
        else:
            arrays[name] = np.array(rng.random(shape).astype(np_dtype), copy=True)
    return arrays


def scalar_params(kernel: TSVCKernel, l1: int) -> Dict[str, int]:
    """Resolve scalar parameter values (``"N4"`` -> ``l1 // 4`` to keep shifted
    writes in bounds)."""
    return {p: (l1 // 4 if v == "N4" else v) for p, v in kernel.params.items()}


def symbols(kernel: TSVCKernel, l1: int, l2: int) -> Dict[str, int]:
    """The ``LEN_*`` symbols the kernel needs (passing an unused symbol errors).

    Read from the SDFG's free symbols so it matches the compiled signature exactly.
    """
    free = {str(s) for s in kernel.program.to_sdfg(simplify=False).free_symbols}
    return {s: v for s, v in (("LEN_1D", l1), ("LEN_2D", l2), ("S", S_VALUE)) if s in free}


def _unique_name(base: str, tag: str) -> str:
    """A valid, collision-free SDFG name from ``base`` + a test ``tag``.

    Embeds the ``pytest-xdist`` worker id (when set) so concurrent workers never
    share a ``.dacecache`` build directory, and sanitizes to an identifier.
    """
    worker = os.environ.get("PYTEST_XDIST_WORKER", "")
    raw = f"{base}_{tag}_{worker}" if worker else f"{base}_{tag}"
    return re.sub(r"\W+", "_", raw).strip("_")


def to_sdfg(kernel: TSVCKernel, tag: str, *, simplify: bool = False) -> "dace.SDFG":
    """Build a fresh, uniquely-named SDFG for one test variant.

    The cached ``DaceProgram`` SDFG is deep-copied (a prior variant may have
    mutated it in place) and renamed to ``<kernel>_<tag>[_<worker>]`` so that,
    under ``pytest -n`` (xdist), no two concurrently-building SDFGs collide on a
    ``.dacecache`` directory. Pass the pytest node/param id as ``tag``.

    :param kernel: the corpus kernel.
    :param tag: per-variant suffix for uniqueness (e.g. ``request.node.name``).
    :param simplify: whether to simplify during ``to_sdfg``.
    :returns: a private, uniquely-named SDFG.
    """
    sdfg = copy.deepcopy(kernel.program.to_sdfg(simplify=simplify))
    sdfg.name = _unique_name(kernel.name, tag)
    return sdfg


def make_inputs(kernel: TSVCKernel, seed: int = 1234) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Build ``(arrays, call_kwargs)`` for one run at the kernel's default lengths.

    ``call_kwargs`` already merges the scalar parameters and the ``LEN_*``
    symbols, so a consumer runs the SDFG with ``sdfg(**arrays, **call_kwargs)``.
    A reference and a candidate must each receive a private copy of ``arrays``.

    :param kernel: the corpus kernel.
    :param seed: RNG seed (a reference and candidate must share one seed).
    :returns: ``(arrays, call_kwargs)``.
    """
    l1, l2 = lengths(kernel)
    arrays = allocate(kernel, l1, l2, np.random.default_rng(seed))
    call_kwargs = {**scalar_params(kernel, l1), **symbols(kernel, l1, l2)}
    return arrays, call_kwargs
