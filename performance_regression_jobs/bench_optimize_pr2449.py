# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Time the DaCe optimizer over the npbench + polybench corpus, for one dace tree.

Measures ONLY ``simplify -> LoopToMap -> StateFusion -> MapFusion``. Nothing is compiled and
nothing is run, so the numbers are pure Python transformation time -- which is what a
getattr/hasattr sweep is expected to move.

Kernels are sliced round-robin across ``SLURM_NTASKS`` so each rank measures a disjoint
subset; the CSV rows carry the variant, so all ranks append to one file and ``--summarize``
merges them. Results go to ``--out``, never stdout: the frontend prints its own diagnostics
there when a kernel fails to parse.
"""
import argparse
import copy
import csv
import math
import os
import statistics
import sys
import time
from typing import List, Tuple

PIPELINE = 'simplify -> LoopToMap -> StateFusion -> MapFusion'


def kernels() -> List[Tuple[str, str]]:
    from tests.corpus import corpus_suite
    all_kernels = corpus_suite.kernels()
    rank, ntasks = int(os.environ.get('SLURM_PROCID', 0)), int(os.environ.get('SLURM_NTASKS', 1))
    return all_kernels[rank::ntasks]


def base_sdfg(suite: str, name: str):
    """The unoptimized SDFG for one kernel, straight from its ``@dace.program``."""
    from tests.corpus.npbench import npbench
    from tests.corpus.polybench import polybench
    if suite == 'poly':
        return polybench.fresh_sdfg(polybench.collect(name)[0], simplify=False)
    return npbench.fresh_sdfg(npbench.collect(name)[0], simplify=False)


def timed(sdfg) -> Tuple[float, Tuple[int, int, int]]:
    """One timed pipeline run on a private copy; the copy is NOT part of the timing.

    The applied-counts are returned with the time because they are what makes the comparison
    honest: a variant that is quicker only because it refused a transformation is not faster.
    """
    from dace.transformation.dataflow import MapFusion
    from dace.transformation.interstate import LoopToMap, StateFusion
    work = copy.deepcopy(sdfg)
    start = time.perf_counter()
    work.simplify()
    counts = (work.apply_transformations_repeated(LoopToMap), work.apply_transformations_repeated(StateFusion),
              work.apply_transformations_repeated(MapFusion))
    return time.perf_counter() - start, counts


def measure(args) -> None:
    with open(args.out, 'a', newline='') as fp:
        out = csv.writer(fp)
        todo = [tuple(args.kernel.split(':', 1))] * args.iters if args.mode == 'heavy' else kernels()
        cached = {}
        for i, (suite, name) in enumerate(todo):
            try:
                if (suite, name) not in cached:
                    cached[(suite, name)] = base_sdfg(suite, name)
                seconds, counts = timed(cached[(suite, name)])
            except Exception as ex:  # a kernel that will not parse is not a timing result
                print(f'SKIP {suite}:{name}: {type(ex).__name__}: {ex}', file=sys.stderr)
                continue
            out.writerow([args.variant, args.rep if args.mode == 'corpus' else i, suite, name, f'{seconds:.6f}',
                          *counts])
            fp.flush()


def load(path: str):
    times, counts = {}, {}
    with open(path) as fp:
        for variant, _, suite, name, seconds, *applied in csv.reader(fp):
            times.setdefault((suite, name), {}).setdefault(variant, []).append(float(seconds))
            counts.setdefault((suite, name), {})[variant] = tuple(applied)
    return times, counts


def summarize(corpus_csv: str, heavy_csv: str) -> None:
    times, counts = load(corpus_csv)
    print(f'\nOptimize time per kernel: {PIPELINE}')
    print('median over reps; speedup = base / pr  (>1 means the PR branch is faster)\n')
    print(f'{"kernel":<32}{"base ms":>10}{"pr ms":>10}{"speedup":>9}   applied l2m/sf/mf')
    print('-' * 88)
    speedups, differing = [], []
    for key, by_variant in sorted(times.items()):
        if {'pr', 'base'} - by_variant.keys():
            continue
        pr_ms, base_ms = (statistics.median(by_variant[v]) * 1e3 for v in ('pr', 'base'))
        speedups.append(base_ms / pr_ms)
        applied, same = counts[key]['pr'], counts[key]['pr'] == counts[key]['base']
        if not same:
            differing.append((key, counts[key]['base'], applied))
        print(f'{":".join(key):<32}{base_ms:>10.1f}{pr_ms:>10.1f}{base_ms / pr_ms:>9.3f}   '
              f'{"/".join(applied)}{"" if same else "  <== DIFFERS"}')
    print('-' * 88)
    geomean = math.exp(sum(math.log(s) for s in speedups) / len(speedups)) if speedups else float('nan')
    print(f'{"geomean speedup":<32}{"":>10}{"":>10}{geomean:>9.3f}   over {len(speedups)} kernels')
    for key, base, pr in differing:
        print(f'!! {":".join(key)} applied different transformations: base={base} pr={pr} -- not like-for-like')

    if not os.path.exists(heavy_csv):
        return
    heavy, _ = load(heavy_csv)
    print(f'\n\nRepeat optimization of a single kernel\n')
    print(f'{"kernel":<24}{"iters":>7}{"base total s":>14}{"pr total s":>12}{"base ms/it":>12}'
          f'{"pr ms/it":>11}{"speedup":>9}')
    print('-' * 88)
    for key, by_variant in sorted(heavy.items()):
        if {'pr', 'base'} - by_variant.keys():
            continue
        pr_all, base_all = by_variant['pr'], by_variant['base']
        pr_med, base_med = statistics.median(pr_all) * 1e3, statistics.median(base_all) * 1e3
        print(f'{":".join(key):<24}{len(pr_all):>7}{sum(base_all):>14.2f}{sum(pr_all):>12.2f}'
              f'{base_med:>12.2f}{pr_med:>11.2f}{base_med / pr_med:>9.3f}')
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--summarize', nargs=2, metavar=('CORPUS_CSV', 'HEAVY_CSV'))
    ap.add_argument('--variant', choices=('pr', 'base'))
    ap.add_argument('--out')
    ap.add_argument('--rep', type=int, default=0)
    ap.add_argument('--mode', choices=('corpus', 'heavy'), default='corpus')
    ap.add_argument('--kernel', default='poly:gemm', help='suite:name, for --mode heavy')
    ap.add_argument('--iters', type=int, default=100)
    args = ap.parse_args()
    if args.summarize:
        summarize(*args.summarize)
    else:
        measure(args)


main()
