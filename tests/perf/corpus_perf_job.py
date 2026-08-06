#!/usr/bin/env python3
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Rank-side entry point of the ONE distributed job over the four in-repo corpora --
polybench, npbench, tsvc, tsvc_2_5 -- at the ``paper`` preset.

Launched by ``tests/perf/submit_corpus_perf.sh`` as ``X`` nodes x 4 ranks. This script only
decides *which* corpus (and which slice of it) the rank owns, hands the rank a private build
folder, and drives the two measurement drivers that already exist:

  * ``tests.corpus.measure_parallelization`` -- residual sequential loops per kernel, plus
    optional ``--check`` compile+run value-preservation. Covers all four corpora today.
  * ``tests.passes.canonicalize.canonicalize_perf_corpus_test`` -- wall-clock canon vs
    auto-opt, resumable through one JSON per kernel. Covers whatever ``corpus_suite`` exposes,
    so tsvc/tsvc_2_5 light up here the moment that adapter carries them.

``--aggregate`` is the post-step once every rank has exited: it folds the per-rank CSVs into
one parallelism report and one speedup CSV + Markdown table. Run bare (no launcher) it is rank
0 of 1, i.e. every corpus, whole::

    python tests/perf/corpus_perf_job.py --preset S --limit 2
"""
import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path

#: The four in-repo corpora, in the order the rank->work rule cuts them. Deliberately not
#: ``measure_parallelization``'s ``all``, which also selects the out-of-tree ``hpcagent`` corpus.
SUITES = ('poly', 'np', 'tsvc', 'tsvc25')
#: Ranks per node the submit script requests; one corpus per rank at the default single node.
RANKS_PER_NODE = 4
#: Free space one rank's build scratch is assumed to need. A corpus shard's objects and shared
#: libraries run to a few hundred MB; 4 GiB a rank leaves headroom and still fits in a compute
#: node's /dev/shm. Below it the in-memory root is REFUSED -- a filled tmpfs fakes corpus failures.
SCRATCH_NEED_GIB = 4

PARALLELISM_DRIVER = 'tests.corpus.measure_parallelization'
PERF_DRIVER = 'tests.passes.canonicalize.canonicalize_perf_corpus_test'


def rank_and_size() -> tuple[int, int]:
    """This process's ``(rank, size)`` as reported by the launcher; ``(0, 1)`` when run bare."""
    for rank_var, size_var in (('SLURM_PROCID', 'SLURM_NTASKS'), ('OMPI_COMM_WORLD_RANK', 'OMPI_COMM_WORLD_SIZE')):
        if os.environ.get(rank_var) and os.environ.get(size_var):
            return int(os.environ[rank_var]), int(os.environ[size_var])
    return 0, 1


def work_slots(rank: int, size: int, suites: tuple[str, ...] = SUITES) -> list[tuple[str, int, int]]:
    """The ``(suite, shard_index, shard_count)`` slots this rank owns -- ONE rule for any node count.

    Each suite is cut into ``size // len(suites)`` shards (at least one); the resulting
    ``len(suites) * shards`` slots are laid out suite-minor, so slot ``s`` is suite ``s % len(suites)``;
    rank ``r`` then takes ``slots[r::size]``. The interesting node counts all fall out of that one
    expression: 1 node (4 ranks) gives every rank exactly one whole corpus, 2 nodes (8 ranks) has
    rank ``r`` and rank ``r+4`` split corpus ``r``'s kernel list round-robin, and bare (1 rank)
    gives rank 0 all four corpora whole.

    ``shard_index/shard_count`` is passed straight through as the drivers' own ``--shard i/n``,
    which is per-corpus -- hence a slot names both the corpus and the slice.
    """
    per_suite = max(1, size // len(suites))
    slots = [(suite, shard, per_suite) for shard in range(per_suite) for suite in suites]
    return slots[rank::size]


def local_ranks(size: int) -> int:
    """Ranks sharing this node -- what the scratch root has to have room for, all of them at once."""
    for var in ('SLURM_NTASKS_PER_NODE', 'OMPI_COMM_WORLD_LOCAL_SIZE'):
        if os.environ.get(var):
            return int(os.environ[var])
    return min(size, RANKS_PER_NODE)


def free_gib(path: Path) -> float:
    """Free space on the filesystem holding ``path``, in GiB; 0 when the path does not exist."""
    if not path.is_dir():
        return 0.0
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize / 1024**3


def scratch_root(ranks_here: int) -> tuple[Path, str]:
    """``(root, why)`` for the per-rank build scratch: in memory when it demonstrably fits.

    tmpfs makes the compiles RAM-bound rather than disk-bound, which is the right answer on a
    compute node with hundreds of GB. It is the WRONG answer when the tmpfs is small -- filling one
    has already produced fake corpus failures here -- so ``/dev/shm`` is taken only when it has room
    for every rank on this node, and ``$HOME/.cache`` (on disk) is the fallback. ``DACE_JOB_SCRATCH``
    overrides the choice outright.
    """
    override = os.environ.get('DACE_JOB_SCRATCH')
    if override:
        return Path(override), 'DACE_JOB_SCRATCH'
    need = SCRATCH_NEED_GIB * ranks_here
    free = free_gib(Path('/dev/shm'))
    if free >= need:
        return Path('/dev/shm/dace-corpus-perf'), f'in memory, /dev/shm has {free:.1f} GiB >= {need} GiB needed'
    return Path.home() / '.cache' / 'dace-corpus-perf', f'on disk, /dev/shm has {free:.1f} of {need} GiB needed'


def isolate_rank(rank: int, build_root: Path) -> None:
    """Point this rank at its OWN build folder and TMPDIR, exported to the driver processes.

    Ranks sharing one ``.dacecache`` load libraries other ranks are still writing, and under
    ``srun``/``mpirun`` cmake additionally wedges with SIGCHLD blocked unless DaCe is told the
    run is distributed. ``/tmp`` is refused whatever its free space says: it is the small tmpfs
    everything else on the box shares, and a compile sweep that fills it fakes corpus failures.
    """
    if build_root.is_relative_to('/tmp'):
        raise SystemExit(f'FATAL: build root {build_root} is under /tmp; set DACE_JOB_SCRATCH elsewhere')
    mine = build_root / f'rank{rank:04d}'
    (mine / 'tmp').mkdir(parents=True, exist_ok=True)
    os.environ['DACE_default_build_folder'] = str(mine / 'dacecache')
    os.environ['TMPDIR'] = str(mine / 'tmp')
    os.environ['DACE_cache_distaware'] = '1'


def drive(cmd: list[str], repo: Path) -> int:
    """Run one driver in the repo root (its modules import ``tests.*``); return its exit code."""
    print('+ ' + ' '.join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(repo))


def measure(rank: int, slots: list[tuple[str, int, int]], args: argparse.Namespace, repo: Path, out: Path) -> int:
    """Run the selected facets over this rank's slots; return the worst driver exit code."""
    worst = 0
    limit = ['--limit', str(args.limit)] if args.limit else []
    for suite, shard, shards in slots:
        if args.facet in ('parallelism', 'both'):
            cmd = [
                sys.executable, '-m', PARALLELISM_DRIVER, suite, '--shard', f'{shard}/{shards}', '--csv',
                str(out / f'parallelism_rank{rank:04d}.csv')
            ] + limit
            worst = max(worst, drive(cmd + (['--check'] if args.check else []), repo))
        if args.facet in ('perf', 'both'):
            # Per-rank Markdown: every rank renders the shared result dir, so one shared path
            # would have the ranks overwriting each other's table mid-write.
            cmd = [
                sys.executable, '-m', PERF_DRIVER, '--suite', suite, '--shard', f'{shard}/{shards}', '--markdown',
                str(out / 'ranks' / f'speedup_rank{rank:04d}.md')
            ] + limit
            worst = max(worst, drive(cmd + (['--force'] if args.force else []), repo))
    return worst


def aggregate(repo: Path, out: Path, perf_dir: Path) -> int:
    """Fold the per-rank outputs into one report. Non-zero when a kernel errored or miscompiled."""
    shards = sorted(str(p) for p in out.glob('parallelism_rank*.csv'))
    worst = 0
    if shards:
        worst = max(worst, drive([sys.executable, '-m', PARALLELISM_DRIVER, '--summarize'] + shards, repo))
    if perf_dir.is_dir():
        # ``--no-run`` re-exports the per-kernel JSON the ranks wrote; no second gather path.
        worst = max(
            worst,
            drive([
                sys.executable, '-m', PERF_DRIVER, '--no-run', '--csv',
                str(out / 'speedup.csv'), '--markdown',
                str(out / 'speedup_table.md')
            ], repo))
    print(f'=== aggregated into {out}', flush=True)
    return worst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--facet', choices=('parallelism', 'perf', 'both'), default='both', help='which measurement to run')
    ap.add_argument('--preset', default='paper', help="corpus_suite dataset preset to time (default paper)")
    ap.add_argument('--out', default=None, help='results directory (default tests/perf/corpus_perf_results)')
    ap.add_argument('--limit', type=int, default=None, help='only the first N kernels of each corpus (smoke runs)')
    ap.add_argument('--check', action='store_true', help='parallelism facet also compiles+runs and checks values')
    ap.add_argument('--force', action='store_true', help='re-time kernels that already have a result file')
    ap.add_argument('--aggregate', action='store_true', help='post-step: fold the per-rank outputs into one report')
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    out = Path(args.out or os.environ.get('CORPUS_PERF_OUT') or repo / 'tests' / 'perf' / 'corpus_perf_results')
    perf_dir = out / 'perf_json'
    (out / 'ranks').mkdir(parents=True, exist_ok=True)
    perf_dir.mkdir(parents=True, exist_ok=True)
    # The perf harness reads both at import time; the ranks share the result dir (their kernels are
    # disjoint, so their JSON file names are too) which is what makes the aggregation a plain re-export.
    os.environ['CANON_PERF_DIR'] = str(perf_dir)
    os.environ['CANON_PERF_PRESETS'] = args.preset

    if args.aggregate:
        return aggregate(repo, out, perf_dir)

    rank, size = rank_and_size()
    root, why = scratch_root(local_ranks(size))
    isolate_rank(rank, root)
    slots = work_slots(rank, size)
    print(
        f"[rank {rank}/{size}] host={socket.gethostname()} preset={args.preset} "
        f"omp={os.environ.get('OMP_NUM_THREADS', '')} slots={slots} "
        f"scratch={why} build={os.environ['DACE_default_build_folder']}",
        flush=True)
    return measure(rank, slots, args, repo, out)


if __name__ == '__main__':
    sys.exit(main())
