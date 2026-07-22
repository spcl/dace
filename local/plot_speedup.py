#!/usr/bin/env python3
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Plot the cmake-newcpu vs cmake-oldcpu speedup from local_compare.py's TSV.

    speedup = median(cmake-oldcpu) / median(cmake-newcpu)     (>1 => the new codegen is faster)

One bar per kernel (sorted), a 1.0 reference line, and the geometric mean. Rows that errored or
have no median are skipped (an in-progress TSV plots fine).

    python3 plot_speedup.py --tsv local_compare.tsv --out-dir plots
"""
import argparse
import csv
import math
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

BASELINE, CANDIDATE = 'cmake-oldcpu', 'cmake-newcpu'


def read_speedups(tsv):
    """{kernel: speedup} from the candidate rows; recomputed from the medians when the speedup cell
    is empty (e.g. a partially-written TSV), so a live file still plots."""
    medians, speedups = {}, {}
    with open(tsv) as fh:
        for row in csv.DictReader(fh, delimiter='\t'):
            if row.get('status') != 'ok':
                continue
            try:
                median = float(row['median_ms'])
            except (TypeError, ValueError):
                continue
            medians.setdefault(row['kernel'], {})[row['variant']] = median
            if row['variant'] == CANDIDATE and row.get('speedup'):
                try:
                    speedups[row['kernel']] = float(row['speedup'])
                except ValueError:
                    pass
    for kernel, by_variant in medians.items():
        if kernel in speedups:
            continue
        base, cand = by_variant.get(BASELINE), by_variant.get(CANDIDATE)
        if base and cand:
            speedups[kernel] = base / cand
    return speedups


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tsv', default='local_compare.tsv')
    ap.add_argument('--out-dir', default='plots')
    ap.add_argument('--title', default=None)
    args = ap.parse_args()

    if not os.path.isfile(args.tsv):
        print(f'no TSV at {args.tsv}', file=sys.stderr)
        return 1
    speedups = read_speedups(args.tsv)
    if not speedups:
        print(f'{args.tsv}: no complete {BASELINE}/{CANDIDATE} pair yet -- nothing to plot')
        return 0

    items = sorted(speedups.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    values = [v for _, v in items]
    geomean = math.exp(sum(math.log(v) for v in values) / len(values))

    fig, ax = plt.subplots(figsize=(max(6.0, 0.32 * len(names) + 2.0), 5.0))
    colors = ['tab:green' if v >= 1.0 else 'tab:red' for v in values]
    ax.bar(range(len(names)), values, color=colors)
    ax.axhline(1.0, color='black', linewidth=1.0, label='parity (1.0x)')
    ax.axhline(geomean, color='tab:blue', linestyle='--', linewidth=1.2, label=f'geomean {geomean:.3f}x')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel(f'speedup  =  median({BASELINE}) / median({CANDIDATE})')
    ax.set_title(args.title or f'{CANDIDATE} vs {BASELINE}  ({len(names)} kernels)')
    ax.legend()
    fig.tight_layout()

    os.makedirs(args.out_dir, exist_ok=True)
    for ext in ('png', 'pdf'):
        path = os.path.join(args.out_dir, f'speedup_cmake_newcpu_vs_oldcpu.{ext}')
        fig.savefig(path, dpi=150)
        print(f'wrote {path}')

    faster = sum(1 for v in values if v > 1.0)
    print(f'geomean {geomean:.3f}x  |  faster: {faster}/{len(values)}  |  '
          f'range {min(values):.3f}x .. {max(values):.3f}x')
    return 0


if __name__ == '__main__':
    sys.exit(main())
