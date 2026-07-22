# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Speedup tables and a plot from two bench_codegen_scopes.py CSVs.

    python tests/perf/plot_codegen_scopes.py --base main.csv --new pr.csv \
        --out-md speedup.md --out-plot speedup.png

Speedup is base/new, so >1 means the new checkout is faster. Per-workload aggregates use the
GEOMETRIC mean, which is the right average for ratios: the arithmetic mean of speedups is biased by
whichever kernel happened to improve most, and is not symmetric under swapping base and new.
"""

import argparse
import csv
import math
from collections import defaultdict
from typing import Dict, List, Tuple


def read(path: str) -> Dict[Tuple[str, str], float]:
    """(workload, kernel) -> median seconds."""
    out = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            out[(row['workload'], row['kernel'])] = float(row['median_s'])
    return out


def geomean(values: List[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else float('nan')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, help='CSV from the baseline checkout')
    parser.add_argument('--new', required=True, help='CSV from the checkout under test')
    parser.add_argument('--out-md', default='speedup.md')
    parser.add_argument('--out-plot', default='speedup.png')
    parser.add_argument('--top', type=int, default=25, help='rows per per-kernel table')
    args = parser.parse_args()

    base, new = read(args.base), read(args.new)
    shared = sorted(set(base) & set(new))
    if not shared:
        raise SystemExit('no (workload, kernel) pairs in common between the two CSVs')

    by_workload: Dict[str, List[Tuple[str, float, float, float]]] = defaultdict(list)
    for key in shared:
        workload, kernel = key
        b, n = base[key], new[key]
        if b <= 0 or n <= 0:
            continue  # an unresolved timer ranks nowhere
        by_workload[workload].append((kernel, b, n, b / n))

    lines: List[str] = ['# Codegen scope passes: speedup', '']
    only_base, only_new = sorted(set(base) - set(new)), sorted(set(new) - set(base))
    if only_base or only_new:
        lines += [f'_Compared {len(shared)} pairs; {len(only_base)} only in base, {len(only_new)} only in new._', '']

    lines += [
        '## Summary', '', '| Workload | Kernels | Base median (ms) | New median (ms) | Geomean speedup |',
        '|---|---:|---:|---:|---:|'
    ]
    for workload in sorted(by_workload):
        entries = by_workload[workload]
        gm = geomean([e[3] for e in entries])
        lines.append(f'| {workload} | {len(entries)} | {sum(e[1] for e in entries) * 1000:.1f} | '
                     f'{sum(e[2] for e in entries) * 1000:.1f} | {gm:.3f}x |')

    overall = geomean([e[3] for entries in by_workload.values() for e in entries])
    lines += ['', f'**Overall geometric mean speedup: {overall:.3f}x**', '']

    for workload in sorted(by_workload):
        entries = sorted(by_workload[workload], key=lambda e: e[3])
        lines += [
            f'## {workload} (worst {args.top} by speedup)', '', '| Kernel | Base (ms) | New (ms) | Speedup |',
            '|---|---:|---:|---:|'
        ]
        for kernel, b, n, s in entries[:args.top]:
            lines.append(f'| {kernel} | {b * 1000:.2f} | {n * 1000:.2f} | {s:.3f}x |')
        lines.append('')

    with open(args.out_md, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'wrote {args.out_md}')
    print(f'overall geomean speedup: {overall:.3f}x')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed; skipped the plot')
        return

    workloads = sorted(by_workload)
    fig, axes = plt.subplots(1, len(workloads), figsize=(6 * len(workloads), 5), squeeze=False)
    for ax, workload in zip(axes[0], workloads):
        speedups = sorted(e[3] for e in by_workload[workload])
        ax.barh(range(len(speedups)), speedups, color=['#c0392b' if s < 1 else '#27ae60' for s in speedups])
        ax.axvline(1.0, color='black', linewidth=1, linestyle='--')
        ax.set_title(f'{workload}\ngeomean {geomean(speedups):.3f}x')
        ax.set_xlabel('speedup (base / new)')
        ax.set_ylabel('kernel (sorted)')
    fig.tight_layout()
    fig.savefig(args.out_plot, dpi=150)
    print(f'wrote {args.out_plot}')


if __name__ == '__main__':
    main()
