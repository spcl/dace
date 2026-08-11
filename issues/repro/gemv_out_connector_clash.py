"""Repro: BLAS init tasklets use a bare `out` connector; inlining the expansion collides with a user array named `out`.

Exits 0 when every case still behaves as gemv_out_connector_clash.md documents, 1 when something moved.
No C++ compiler needed: expansion, simplification and validation alone reach the failure.
"""
import sys

import dace as dc

M, N, K = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N', 'K'))


@dc.program
def gemv_out(A: dc.float64[M, N], x: dc.float64[N], out: dc.float64[M]):
    out[:] = A @ x


@dc.program
def gemv_res(A: dc.float64[M, N], x: dc.float64[N], res: dc.float64[M]):
    res[:] = A @ x


@dc.program
def gemm_out(A: dc.float64[M, N], B: dc.float64[N, K], out: dc.float64[M, K]):
    out[:] = A @ B


@dc.program
def gemm_res(A: dc.float64[M, N], B: dc.float64[N, K], res: dc.float64[M, K]):
    res[:] = A @ B


def run(prog):
    try:
        sdfg = prog.to_sdfg(simplify=False)
        sdfg.expand_library_nodes()
        sdfg.simplify()
        sdfg.validate()
    except Exception as ex:
        return type(ex).__name__ + ': ' + str(ex).splitlines()[0]
    return None


CASES = (
    ('gemv, output named `out`', gemv_out, True),
    ('gemv, output named `res`', gemv_res, False),
    ('gemm, output named `out`', gemm_out, True),
    ('gemm, output named `res`', gemm_res, False),
)

MARKER = "Connector name 'out' is already used as a symbol, constant, or array name"

failures = 0
for label, prog, expect_error in CASES:
    err = run(prog)
    ok = (err is not None and MARKER in err) if expect_error else (err is None)
    failures += not ok
    print(('OK      ' if ok else 'CHANGED ') + label)
    print('        ' + (err if err else 'expanded, simplified and validated'))

print('\n%d/%d cases as documented' % (len(CASES) - failures, len(CASES)))
sys.exit(1 if failures else 0)
