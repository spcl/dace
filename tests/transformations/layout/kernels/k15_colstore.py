# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""k15 row-store vs column-store -- aggregate columns of an (N, C) table.

    out[c] = sum_i T[i, c]     (a filter+sum scan, the canonical analytics access)

Row-store (AoS, ``T`` as ``[N, C]``) scans each column strided by C; column-store (SoA) makes each
column contiguous. The sweep expresses that choice by permuting ``T`` ([N,C] row-major vs [C,N]
column-major) -- transparent, so every candidate reproduces the oracle.

Source: Copeland & Khoshafian, SIGMOD'85 (DSM); Stonebraker et al., C-Store VLDB'05; Ailamaki et
al., PAX VLDB'01.
"""
import numpy
import dace

from dace.transformation.layout.brute_force import permutation_candidates

N, C = dace.symbol("N"), dace.symbol("C")


@dace.program
def colsum(T: dace.float64[N, C], out: dace.float64[C]):
    for i, c in dace.map[0:N, 0:C] @ dace.ScheduleType.Sequential:
        out[c] += T[i, c]


def oracle(T):
    return {"out": T.sum(axis=0)}


def make_inputs(n, c, seed=0):
    rng = numpy.random.default_rng(seed)
    return {"T": rng.random((n, c))}


def candidates():
    return dict(permutation_candidates("T", 2))


def run_closure(inputs, n, c):

    def run(sdfg):
        out = numpy.zeros(c)
        sdfg(T=inputs["T"].copy(), out=out, N=n, C=c)
        return {"out": out}

    return run
