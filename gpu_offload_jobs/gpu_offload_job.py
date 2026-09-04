# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""One rank of the GPU-offloading sweep. ``srun`` execs THIS file directly.

The rank binds itself to its module's GPU, takes a private build folder, and hands the rest of the
command line to ``tests.corpus.measure_gpu_arms``. It exists as a python entry rather than a
shell wrapper for the SIGCHLD reason below, which is the same one
``canon_perf_jobs/corpus_perf_job.py`` carries.

Run one rank by hand::

    python gpu_offload_jobs/gpu_offload_job.py --stage codegen --preset S --suites poly
"""
import signal

# ``slurmstepd`` starts every task with SIGCHLD BLOCKED. The mask survives fork+exec and
# ``subprocess`` does not reset it, so it reaches cmake, whose KWSys learns its helpers exited by
# RECEIVING SIGCHLD and otherwise waits in ``select()`` forever -- the "stuck configure" that is
# really a lost wakeup. ``pthread_sigmask`` is PER-THREAD, so this only works in a module srun execs
# directly, before any thread pool exists. A wrapper shell, setsid, nohup or timeout in between
# brings the hang back.
SIGCHLD_WAS_BLOCKED = signal.SIGCHLD in signal.pthread_sigmask(signal.SIG_BLOCK, [])
signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGCHLD})

import os
import sys
from pathlib import Path


def repo_root() -> Path:
    """The dace checkout this file sits in."""
    for parent in Path(__file__).resolve().parents:
        if (parent / 'dace' / '__init__.py').is_file():
            return parent
    raise SystemExit(f'FATAL: no dace checkout above {__file__}')


def main() -> int:
    root = repo_root()
    sys.path.insert(0, str(root))

    rank = int(os.environ.get('SLURM_PROCID', '0'))
    local = int(os.environ.get('SLURM_LOCALID', '0'))
    total = int(os.environ.get('SLURM_NTASKS', '1'))

    # A Grace-Hopper node is four modules, each a Grace CPU with its OWN H100. Bind the rank to the
    # GPU of its module. Without this every rank drives device 0, and four ranks timing kernels on
    # one card measure each other rather than the code -- the numbers still come out, they are just
    # not about the compiler.
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(local))
    # Ranks compile DIFFERENT SDFGs and .dacecache is not written atomically, so a shared build
    # folder lets one rank load a library another is halfway through writing.
    scratch = os.environ.get('SCRATCH', '/tmp')
    os.environ.setdefault('DACE_default_build_folder', f'{scratch}/dace_gpu_offload/rank{rank}')

    from tests.corpus import measure_gpu_arms as driver

    # ``--out`` is a directory the submitter names; the driver writes ONE csv, so give each rank its
    # own file inside it. Aggregating is then a concatenation, and a rank that dies leaves the rows
    # it did finish.
    argv, out_dir = [], None
    rest = list(sys.argv[1:])
    while rest:
        item = rest.pop(0)
        if item == '--out':
            out_dir = rest.pop(0)
        else:
            argv.append(item)
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        argv += ['--csv', os.path.join(out_dir, f'shard_{rank}.csv')]

    sys.argv = [sys.argv[0], '--shard', f'{rank}/{total}'] + argv
    return driver.main()


if __name__ == '__main__':
    raise SystemExit(main())
