#!/usr/bin/env python3
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Rank-side entry point of the ONE distributed job over the four in-repo corpora --
polybench, npbench, tsvc, tsvc_2_5, 277 kernels -- at the ``paper`` preset.

Launched by ``submit_corpus_perf.sh`` beside it as ``X`` nodes x 4 ranks. A rank works out which
kernels it owns, hands itself a private build folder, and drives the two measurement drivers that
already exist:

  * ``tests.passes.canonicalize.canonicalize_perf_corpus_test`` -- the eight timed comparison arms
    (``dace-autoopt`` and ``dace-canon`` on each of g++/clang++, plus ``dace-simplify+gcc-autopar``
    and ``dace-simplify+llvm-autopar`` on sequential post-simplify C++, each of those two both
    forced and on the auto-parallelizer's own cost model), each correctness-gated against the corpus
    reference and each required to PROVE its pass engaged, plus ``seq-cpp`` as the tsvc/tsvc_2_5
    speedup denominator -- nine labels in all. Resumable through one JSON per kernel.
  * ``tests.corpus.measure_parallelization`` -- residual sequential loops per kernel, plus
    optional ``--check`` compile+run value-preservation.

Rank -> work is round-robin over the POOLED kernel list: ``corpus_suite.kernels()`` in its
fixed order, rank ``r`` of ``n`` takes ``kernels[r::n]``, and that rank runs EVERY arm of every
kernel it owns. Two reasons, both load-bearing:

  * load balance -- one corpus per rank leaves the tsvc rank with 151 kernels while the npbench
    rank does 24, so the job is as long as its slowest corpus.
  * measurement validity -- all arms of a kernel MUST be timed on the same node. Arms split
    across nodes differ by thermal/DVFS state and memory locality as well as by pipeline, and
    the speedup stops being attributable to the pipeline.

The perf driver applies exactly that rule itself when handed ``--shard r/n`` with no ``--suite``,
so the slicing lives in ONE place and cannot drift into double-measuring or silently skipping
kernels. The order is fixed, so a re-submission lines up with the per-kernel JSONs already on
disk and resumes.

``--aggregate`` is the post-step once every rank has exited: it folds the per-rank CSVs into one
parallelism report and one speedup CSV + Markdown table, then hands the same per-kernel JSONs to
``canon_perf_jobs/plot_corpus_perf.py`` for the figure. Run bare (no launcher) this script is rank 0 of
1, i.e. the whole pooled list::

    python canon_perf_jobs/corpus_perf_job.py --preset S --limit 2
"""
import signal

# Slurm's ``slurmstepd`` starts every task with SIGCHLD BLOCKED (regression from 2025-04, fixed
# upstream 2025-12 and never backported, so all of 25.05 and 25.11.0 carry it). A signal mask
# survives fork+exec and ``subprocess`` does NOT reset it (measured), so the block reaches cmake,
# whose KWSys learns its helpers exited by RECEIVING SIGCHLD and otherwise waits in ``select()``
# forever -- the "stuck configure" that is really a lost wakeup.
#
# This module is the only place a fix works: ``pthread_sigmask`` is PER-THREAD, and the rank entry
# point is exec'd directly by srun/mpirun with no shell in between, before any thread pool exists,
# so unblocking here clears the mask for the rank and for every process it forks. Wrappers do not
# help -- ``setsid``/``nohup``/``unshare``/``timeout``/``setpriv``/``env --ignore-signal=CHLD`` all
# still hang here, and the uutils ``env`` ignores ``--default-signal`` while reporting success.
# DaCe already fixes its own compile forks in-process (``dace.codegen.compiler``
# ``build_subprocess_sigmask``, PR #2466); this is belt and braces around everything else.
SIGCHLD_WAS_BLOCKED = signal.SIGCHLD in signal.pthread_sigmask(signal.SIG_BLOCK, [])
signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGCHLD})

import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path

#: The four in-repo corpora. Deliberately not ``measure_parallelization``'s ``all``, which also
#: selects the out-of-tree ``hpcagent`` corpus.
SUITES = ('poly', 'np', 'tsvc', 'tsvc25')
#: Named corpus groups for ``--suites``. The two halves answer different questions and cost very
#: different amounts, so they are worth running as separate jobs: ``array`` divides by numpy and is
#: a few hundred large kernels, ``loops`` divides by ``seq-cpp`` and is ~200 small ones. Split, each
#: gets a time limit and a results directory that fit it, and a slow polybench sweep can no longer
#: consume the wall clock the tsvc kernels needed.
SUITE_GROUPS = {
    'array': ('poly', 'np'),
    'loops': ('tsvc', 'tsvc25'),
    'all': SUITES,
}
#: Ranks per node the submit script requests.
RANKS_PER_NODE = 4
#: Free space one rank's build scratch is assumed to need. A shard's objects and shared libraries
#: run to a few hundred MB; 4 GiB a rank leaves headroom and still fits in a compute node's
#: /dev/shm. Below it the in-memory root is REFUSED -- a filled tmpfs fakes corpus failures.
SCRATCH_NEED_GIB = 4

PARALLELISM_DRIVER = 'tests.corpus.measure_parallelization'
PERF_DRIVER = 'tests.passes.canonicalize.canonicalize_perf_corpus_test'
#: The plotter, run as a SCRIPT by ``--aggregate`` rather than imported: it must stay readable
#: without dace, and importing the perf driver here would re-probe nine toolchains to draw a plot.
#: Resolved as a SIBLING of this file, so the three scripts travel together whatever the folder is
#: called -- this one has already been at two paths in this tree.
PLOT_SCRIPT = Path(__file__).resolve().with_name('plot_corpus_perf.py')
#: Results land NEXT TO the scripts: one JSON per kernel, a figure and two tables, all in the folder
#: that produced them, so a finished run is one directory to copy off the cluster. Only the build
#: scratch goes to memory-backed storage, and that is throwaway. ``--out`` / ``OUT_DIR`` override,
#: and the submit script defaults to the same path so both agree.
DEFAULT_OUT = Path(__file__).resolve().with_name('corpus_perf_results')


def repo_root() -> Path:
    """The dace checkout above this script, found by walking up rather than by a fixed depth.

    The drivers are run as ``python -m tests....`` from the repo root, so getting this wrong makes
    every rank fail at import. A hardcoded ``parents[N]`` encodes how deep the job folder happens to
    sit, which is exactly the thing that changes when the folder is renamed or moved.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / 'dace' / '__init__.py').is_file() and (parent / 'tests').is_dir():
            return parent
    raise SystemExit(f'FATAL: no dace checkout above {__file__}; run this script from inside the repo')


def sigchld_state() -> str:
    """Assert SIGCHLD is deliverable in this rank; return the line describing how it got that way.

    Runs at rank start rather than trusting the unblock above: a mask that is still set here means
    every compile in this job is about to hang, and finding that out in the first second beats
    finding it out after an hour of apparently-running cmake.
    """
    if signal.SIGCHLD in signal.pthread_sigmask(signal.SIG_BLOCK, []):
        raise SystemExit('FATAL: SIGCHLD is still blocked after the entry-point unblock; cmake will '
                         'hang in select() instead of compiling. Refusing to start.')
    return 'blocked by the launcher, cleared at entry' if SIGCHLD_WAS_BLOCKED else 'already clear at entry'


def rank_and_size() -> tuple[int, int]:
    """This process's ``(rank, size)`` as reported by the launcher; ``(0, 1)`` when run bare."""
    for rank_var, size_var in (('SLURM_PROCID', 'SLURM_NTASKS'), ('OMPI_COMM_WORLD_RANK', 'OMPI_COMM_WORLD_SIZE')):
        if os.environ.get(rank_var) and os.environ.get(size_var):
            return int(os.environ[rank_var]), int(os.environ[size_var])
    return 0, 1


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
    return Path.home() / '.cache' / 'dace-corpus-perf', f'/dev/shm has {free:.1f} GiB of the {need} GiB needed'


def isolate_rank(rank: int, build_root: Path) -> None:
    """Point this rank at its OWN build folder and TMPDIR, exported to the driver processes.

    Ranks sharing one ``.dacecache`` load libraries other ranks are still writing, which is why
    ``DACE_cache_distaware`` is forced rather than merely defaulted. ``/tmp`` is refused whatever
    its free space says: it is the small tmpfs everything else on the box shares, and a compile
    sweep that fills it fakes corpus failures.
    """
    if build_root.is_relative_to('/tmp'):
        raise SystemExit(f'FATAL: build root {build_root} is under /tmp; set DACE_JOB_SCRATCH elsewhere')
    mine = build_root / f'rank{rank:04d}'
    (mine / 'tmp').mkdir(parents=True, exist_ok=True)
    os.environ['DACE_default_build_folder'] = str(mine / 'dacecache')
    os.environ['TMPDIR'] = str(mine / 'tmp')
    prior = os.environ.get('DACE_cache_distaware')
    if prior not in (None, '1'):
        print(f'[rank {rank}] WARNING: DACE_cache_distaware={prior} overridden to 1; ranks share this node', flush=True)
    os.environ['DACE_cache_distaware'] = '1'


def announce(rank: int, size: int, preset: str, arms: str, why: str) -> None:
    """Print, and where it matters assert, the rank's whole measurement context in its first lines."""
    tag = f'[rank {rank}/{size}]'
    root = os.environ['DACE_default_build_folder']
    print(
        f'{tag} host={socket.gethostname()} preset={preset} CANON_PERF_ARMS={arms} '
        f"omp={os.environ.get('OMP_NUM_THREADS', '')} kernels=pooled shard {rank}/{size}",
        flush=True)
    print(f'{tag} SIGCHLD {sigchld_state()}', flush=True)
    if root.startswith('/dev/shm'):
        print(f'{tag} scratch={root} ({why})', flush=True)
    else:
        print(
            f'{tag} WARNING: build scratch is ON DISK, not memory-backed -- {why}. Compiles are '
            f'disk-bound and the sweep is slower; point DACE_JOB_SCRATCH at a tmpfs with room '
            f'to restore it. scratch={root}',
            flush=True)
    if os.environ.get('DACE_cache_distaware') != '1':
        raise SystemExit('FATAL: DACE_cache_distaware is not 1; ranks on this node would load each '
                         "other's half-written libraries")
    print(f"{tag} DACE_cache_distaware=1 TMPDIR={os.environ['TMPDIR']}", flush=True)


def drive(cmd: list[str], repo: Path) -> int:
    """Run one driver in the repo root (its modules import ``tests.*``); return its exit code."""
    print('+ ' + ' '.join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(repo))


def selected_suites(spec: str) -> tuple[str, ...]:
    """Corpus tags for ``--suites``: a group name (``array``/``loops``/``all``) or an explicit list.

    Order follows :data:`SUITES` regardless of how the spec was written, so the pooled kernel list a
    rank shards is identical for ``poly,np`` and ``np,poly`` -- the shard is positional, so a spec
    that reordered the pool would hand every rank a different set and silently break resumption
    against results an earlier spelling produced.
    """
    if spec in SUITE_GROUPS:
        return SUITE_GROUPS[spec]
    want = {t.strip() for t in spec.split(',') if t.strip()}
    unknown = want - set(SUITES)
    if unknown:
        raise SystemExit(f'unknown corpus {sorted(unknown)}; pick from {list(SUITES)} '
                         f'or a group {list(SUITE_GROUPS)}')
    return tuple(s for s in SUITES if s in want) or SUITES


def measure(rank: int, size: int, args: argparse.Namespace, repo: Path, out: Path) -> int:
    """Run the selected facets over this rank's shard; return the worst driver exit code."""
    worst = 0
    # Both drivers take ``--limit`` as "the first N of each corpus, BEFORE sharding", so a bare N
    # would leave each rank with N/size kernels -- at ``--limit 1`` on 4 ranks, three ranks with
    # nothing to do and a smoke run that proves nothing about the fan-out. Scaling by the rank count
    # makes it N per corpus PER RANK, which is what a smoke run is asking for, and the shards stay
    # disjoint either way.
    limit = ['--limit', str(args.limit * size)] if args.limit else []
    shard = ['--shard', f'{rank}/{size}']
    if args.facet in ('parallelism', 'both'):
        # This driver has no pooled mode, so the same shard rule is applied to each corpus in turn.
        # It counts loops and maps statically rather than timing anything, so splitting a kernel's
        # measurements across nodes costs nothing here -- unlike the perf facet below.
        for suite in selected_suites(args.suites):
            cmd = [
                sys.executable, '-m', PARALLELISM_DRIVER, suite, *shard, '--csv',
                str(out / f'parallelism_rank{rank:04d}.csv'), *limit
            ]
            worst = max(worst, drive(cmd + (['--check'] if args.check else []), repo))
    if args.facet in ('perf', 'both'):
        # ONE invocation for ALL selected corpora, not one per corpus: the driver pools them and
        # keeps ``kernels[rank::size]``, so every arm of a kernel is timed by this rank, on this
        # node, and the shard stays balanced across the whole selection. Sharding each corpus
        # separately would give every rank a slice of every corpus and defeat the point of
        # splitting them into separate jobs. Per-rank Markdown because every rank renders the
        # shared result dir, and one shared path would have the ranks overwriting each other's
        # table mid-write.
        cmd = [
            sys.executable, '-m', PERF_DRIVER, *shard, '--suite', ','.join(selected_suites(args.suites)), '--markdown',
            str(out / 'ranks' / f'speedup_rank{rank:04d}.md'), *limit
        ]
        cmd += ['--force'] if args.force else []
        worst = max(worst, drive(cmd, repo))
    return worst


def plot(repo: Path, out: Path, preset: str) -> None:
    """Draw the figure from the JSONs the ranks wrote. Deliberately cannot fail the job.

    A sweep that measured everything and could not draw a picture has still produced its numbers,
    and the plotter exits non-zero precisely when it REFUSES to draw (no kernel with a verifiable
    denominator) -- an honest outcome that must not be reported as a failed measurement job. Its own
    message says which of the two it was.
    """
    code = drive([sys.executable, str(PLOT_SCRIPT), '--results', str(out), '--preset', preset], repo)
    if code:
        print(f'=== NO FIGURE (plotter exit {code}) -- the measurements above stand; see its message above', flush=True)


def aggregate(repo: Path, out: Path, perf_dir: Path, preset: str) -> int:
    """Fold the per-rank outputs into one report. Non-zero when a kernel errored or miscompiled."""
    shards = sorted(str(p) for p in out.glob('parallelism_rank*.csv'))
    worst = 0
    if shards:
        worst = max(worst, drive([sys.executable, '-m', PARALLELISM_DRIVER, '--summarize'] + shards, repo))
    if perf_dir.is_dir():
        # ``--no-run`` re-exports the per-kernel JSON the ranks wrote; no second gather path. The
        # columns are read back out of that JSON, so this step needs no toolchain -- but it INHERITS
        # whatever ``CANON_PERF_ARMS`` the sweep exported, and a 1 re-probes all nine arms at import,
        # so a probe that fails afterwards costs a finished run its outputs. The submit script
        # therefore runs the aggregate step with ``CANON_PERF_ARMS=0``.
        worst = max(
            worst,
            drive([
                sys.executable, '-m', PERF_DRIVER, '--no-run', '--csv',
                str(out / 'speedup.csv'), '--markdown',
                str(out / 'speedup_table.md')
            ], repo))
        plot(repo, out, preset)  # same JSONs, one read-only pass; never re-measures
    print(f'=== aggregated into {out}', flush=True)
    return worst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--facet', choices=('parallelism', 'perf', 'both'), default='both', help='which measurement to run')
    ap.add_argument('--preset', default='paper', help='corpus_suite dataset preset to time (default paper)')
    ap.add_argument('--suites',
                    default='all',
                    help="corpora to measure: a group (array=poly+np, loops=tsvc+tsvc25, all) or an "
                    "explicit comma-separated list. Use with --out to keep each half's results "
                    "in its own directory (default all)")
    ap.add_argument('--out', default=None, help=f'results directory (default {DEFAULT_OUT})')
    ap.add_argument('--limit',
                    type=int,
                    default=None,
                    help='smoke runs: measure N kernels of EACH corpus PER RANK (both facets)')
    ap.add_argument('--check', action='store_true', help='parallelism facet also compiles+runs and checks values')
    ap.add_argument('--force', action='store_true', help='re-time kernels that already have a result file')
    ap.add_argument('--aggregate', action='store_true', help='post-step: fold the per-rank outputs into one report')
    args = ap.parse_args()

    repo = repo_root()
    out = Path(args.out or os.environ.get('CORPUS_PERF_OUT') or DEFAULT_OUT)
    perf_dir = out / 'perf_json'
    (out / 'ranks').mkdir(parents=True, exist_ok=True)
    perf_dir.mkdir(parents=True, exist_ok=True)
    # The perf harness reads both at import time; the ranks share the result dir (their kernels are
    # disjoint, so their JSON file names are too) which is what makes the aggregation a re-export.
    os.environ['CANON_PERF_DIR'] = str(perf_dir)
    os.environ['CANON_PERF_PRESETS'] = args.preset

    if args.aggregate:
        return aggregate(repo, out, perf_dir, args.preset)

    # The nine arms ARE this job, so the perf driver's full table is on unless the environment
    # already said otherwise (``CANON_PERF_ARMS=0`` drops it to the plain g++ pair, for a smoke run
    # on a box without a polyhedral clang). The driver reads this at import, which is what lets it
    # probe every arm and abort before a node allocation is spent on a silently-disabled one.
    arms = os.environ.setdefault('CANON_PERF_ARMS', '1')

    rank, size = rank_and_size()
    root, why = scratch_root(local_ranks(size))
    isolate_rank(rank, root)
    announce(rank, size, args.preset, arms, why)
    return measure(rank, size, args, repo, out)


if __name__ == '__main__':
    sys.exit(main())
