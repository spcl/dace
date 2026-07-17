#!/usr/bin/env python3
"""Chain-bisect phase B for the cloudsc ``zqe_5`` scope miscompile (login-node OK).

Walks the per-stage snapshots produced by ``cloudsc_bisect_chain.py`` (batch
job) and finds the FIRST stage that introduces the defect. No transform work
happens here -- snapshots are only ever loaded -- so this is safe on the login
node.

Two checks per snapshot; the STRUCTURAL one is the primary attribution:

* ``structural`` (default): find tasklets whose code references a container of
  the surrounding SDFG that is NOT among the tasklet's connectors -- the
  malformed shape behind the miscompile (the ``__min2`` tasklet reads ``zqe_5``
  as free code text with no in-connector/memlet/AccessNode). This catches the
  producing pass even at stages where the dangling read still COMPILES by
  luck (e.g. a pre-rename ``zqe`` whose declaration happens to be hoisted).
* ``compile``: load -> ``make_sequential`` -> legacy codegen ->
  ``g++ -fsyntax-only``; finds where the breakage becomes compile-visible.

Binary-searches by default (the defect is monotonic once introduced);
``--linear`` sweeps every snapshot to double-check that assumption.

    python3 cloudsc_variants/cloudsc_bisect_check.py [--mode structural|compile|both]
                                                     [--linear] [--no-sequential]
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import engine  # noqa: E402

BISECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache', 'bisect')
_ERROR_RE = re.compile(r"error: use of undeclared identifier '(\w+)'|error: '(\w+)' was not declared")


def structural_check(path):
    """(ok, detail): scan every tasklet for code that references a container of its
    SDFG without a matching connector -- the malformed dangling-read shape. Loads
    only; no codegen, no compiler."""
    import dace
    from dace.sdfg import nodes as sdnodes

    t0 = time.perf_counter()
    sdfg = dace.SDFG.from_file(path)
    offenders = []
    for nested in sdfg.all_sdfgs_recursive():
        array_names = nested.arrays.keys()
        for state in nested.states():
            for node in state.nodes():
                if not isinstance(node, sdnodes.Tasklet):
                    continue
                dangling = (node.free_symbols & array_names)
                if dangling:
                    offenders.append(f'{nested.label}/{state.label}/{node.label}: {sorted(dangling)}')
    elapsed = time.perf_counter() - t0
    if not offenders:
        return True, f'CLEAN ({elapsed:.0f}s)'
    return False, (f'DANGLING ({elapsed:.0f}s) {len(offenders)} tasklet(s), first 3: ' + ' | '.join(offenders[:3]))


def check_snapshot(path, sequential=True):
    """(ok, detail) for one snapshot: emit legacy C++ and g++ -fsyntax-only it."""
    import dace
    from dace.codegen.codegen import generate_code
    from dace.config import set_temporary
    from tests.corpus.cloudsc.generate_data_for_cloudsc import make_sequential

    t0 = time.perf_counter()
    sdfg = dace.SDFG.from_file(path)
    if sequential:
        make_sequential(sdfg)
    with set_temporary('compiler', 'cpu', 'implementation', value='legacy'):
        objects = generate_code(sdfg)
    src = '\n'.join(obj.clean_code for obj in objects if obj.language in ('cpp', 'cu'))

    rtinc = os.path.join(os.path.dirname(os.path.abspath(dace.__file__)), 'runtime', 'include')
    with tempfile.NamedTemporaryFile('w', suffix='.cpp', delete=False) as f:
        f.write(src)
        cpp = f.name
    try:
        proc = subprocess.run(['g++', '-fsyntax-only', '-fopenmp', f'-I{rtinc}', cpp],
                              capture_output=True, text=True, timeout=600)
    finally:
        os.unlink(cpp)
    elapsed = time.perf_counter() - t0
    if proc.returncode == 0:
        return True, f'OK ({elapsed:.0f}s, {len(src)} B)'
    undeclared = sorted({a or b for a, b in _ERROR_RE.findall(proc.stderr)})
    first_err = next((l for l in proc.stderr.splitlines() if ' error: ' in l), proc.stderr[:200])
    return False, f'FAIL ({elapsed:.0f}s) undeclared={undeclared[:8]} | {first_err.strip()[:160]}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=('structural', 'compile', 'both'), default='structural',
                    help='structural: dangling code-text container reads (primary attribution); '
                         'compile: legacy codegen + g++ -fsyntax-only (compile-visible breakage)')
    ap.add_argument('--linear', action='store_true', help='sweep every snapshot instead of binary search')
    ap.add_argument('--no-sequential', action='store_true', help='skip make_sequential (test the parallel form)')
    args = ap.parse_args()

    engine.configure_dace_process()
    snaps = sorted(glob.glob(os.path.join(BISECT_DIR, '*.sdfgz')))
    if not snaps:
        raise SystemExit(f'no snapshots under {BISECT_DIR} -- run the sbatch bisect job first '
                         '(cloudsc_variants/slurm_cloudsc_bisect.sh)')
    print(f'{len(snaps)} snapshots, mode={args.mode}, sequential={not args.no_sequential}', flush=True)

    results = {}

    def probe(i):
        if i not in results:
            if args.mode == 'structural':
                ok, detail = structural_check(snaps[i])
            elif args.mode == 'compile':
                ok, detail = check_snapshot(snaps[i], sequential=not args.no_sequential)
            else:
                ok_s, det_s = structural_check(snaps[i])
                ok_c, det_c = check_snapshot(snaps[i], sequential=not args.no_sequential)
                ok, detail = ok_s and ok_c, f'{det_s} || {det_c}'
            results[i] = ok
            print(f'  [{i:2d}] {os.path.basename(snaps[i]):45s} {detail}', flush=True)
        return results[i]

    if args.linear:
        for i in range(len(snaps)):
            probe(i)
        bad = [i for i in sorted(results) if not results[i]]
        first_bad = bad[0] if bad else None
    else:
        lo, hi = 0, len(snaps) - 1
        if probe(hi):
            print('final snapshot compiles -- nothing to bisect', flush=True)
            return
        if not probe(lo):
            print('FIRST snapshot already fails -- breakage precedes the chain (parse/simplify)', flush=True)
            first_bad = lo
        else:
            while hi - lo > 1:  # invariant: lo OK, hi FAIL
                mid = (lo + hi) // 2
                if probe(mid):
                    lo = mid
                else:
                    hi = mid
            first_bad = hi

    if first_bad is not None:
        print(f'\nFIRST FAILING STAGE: {os.path.basename(snaps[first_bad])}', flush=True)
        if first_bad > 0:
            print(f'last good stage:     {os.path.basename(snaps[first_bad - 1])}', flush=True)
    print('BISECT_CHECK_DONE', flush=True)


if __name__ == '__main__':
    main()
