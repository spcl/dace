#!/usr/bin/env python3
"""Chain-bisect phase A for the cloudsc ``zqe_5`` scope miscompile (COMPUTE NODE).

Re-runs the exact ``_chain()`` the cache builder uses, but saves a compressed
snapshot after the raw parse and after EVERY stage into
``cloudsc_variants/cache/bisect/``. Pure transform work -- no compilation -- but
the chain alone is ~55 min, so this must run as a batch job (normal partition,
1.5 h), never on the login node.

The snapshots are then checked WITHOUT any re-transform work (login-node OK) by
``cloudsc_bisect_check.py``, which finds the first stage whose sequentialized
legacy codegen emits C++ that no longer compiles (the ``use of undeclared
identifier 'zqe_5'`` def/use-across-sibling-scopes shape).

    sbatch cloudsc_variants/slurm_cloudsc_bisect.sh          # this file, phase A
    python3 cloudsc_variants/cloudsc_bisect_check.py         # afterwards, login
"""
import contextlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import engine  # noqa: E402  (configures MPI anti-hang env before dace import)

BISECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache', 'bisect')


def main():
    engine.configure_dace_process()
    from tests.corpus.cloudsc.generate_data_for_cloudsc import build_cloudsc_sdfg
    from tests.corpus.cloudsc.cloudsc_parallelize_chain_test import _chain

    os.makedirs(BISECT_DIR, exist_ok=True)
    t_total = time.perf_counter()

    print('bisect phase A: building the CloudSC SDFG (simplify=False parse -- minutes)...', flush=True)
    t0 = time.perf_counter()
    sdfg = build_cloudsc_sdfg(simplify=False)
    print(f'  build_cloudsc_sdfg: {time.perf_counter() - t0:.1f}s', flush=True)

    def snapshot(index, label):
        path = os.path.join(BISECT_DIR, f'{index:02d}_{label}.sdfgz')
        t0 = time.perf_counter()
        sdfg.save(path, compress=True)
        print(f'  snapshot {os.path.basename(path)} ({os.path.getsize(path) / 1e6:.1f} MB, '
              f'{time.perf_counter() - t0:.1f}s)', flush=True)

    snapshot(0, 'parse')
    for i, (label, apply_fn) in enumerate(_chain(), start=1):
        t0 = time.perf_counter()
        # The loop transforms log every refused loop; keep the job log readable.
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            apply_fn(sdfg)
        sdfg.validate()
        print(f'  {label}: {time.perf_counter() - t0:.1f}s', flush=True)
        snapshot(i, label)

    print(f'bisect phase A: {time.perf_counter() - t_total:.1f}s total, snapshots in {BISECT_DIR}', flush=True)
    print('BISECT_SNAPSHOTS_DONE', flush=True)


if __name__ == '__main__':
    main()
