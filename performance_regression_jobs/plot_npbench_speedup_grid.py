#!/usr/bin/env python3
"""Grid plot: canon's speedup vs. baseline / auto-opt / numpy, one small cell
per NPBench+PolyBench kernel. Reads results/npbench_polybench/<kernel>/paper/
(results.csv + status.csv, written by npbench_polybench_perf.py) -- run that
script first.

    python3 plot_npbench_speedup_grid.py --results-dir results --out speedup_grid.png
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import engine
from npbench_polybench_perf import CORPUS, PRESET, SPEEDUP_VS

# Categorical palette (dataviz skill's validated default), fixed hue order.
_COLORS = {'baseline': '#2a78d6', 'auto-opt': '#1baf7a', 'numpy': '#eda100'}


def collect_speedups(results_dir):
    """kernel -> {baseline_label: speedup or None} for canon vs. each of SPEEDUP_VS.
    Globs every '*_<preset>' folder (engine.result_tag namespaces each by
    compiler+hostname), merging speedups into the same kernel cell."""
    corpus_dir = os.path.join(results_dir, CORPUS)
    out = {}
    if not os.path.isdir(corpus_dir):
        return out
    for kernel in sorted(os.listdir(corpus_dir)):
        for kdir in sorted(glob.glob(os.path.join(corpus_dir, kernel, f'*_{PRESET}'))):
            if not os.path.isdir(kdir):
                continue
            entries = engine._read_kernel(kdir)
            canon = entries.get('canon')
            if not canon or not canon.get('correct') or not canon.get('median_ms'):
                continue
            for base_label in SPEEDUP_VS:
                base = entries.get(base_label)
                if base and base.get('correct') and base.get('median_ms'):
                    out.setdefault(kernel, {})[base_label] = base['median_ms'] / canon['median_ms']
    return out


def plot_grid(speedups, out_path):
    kernels = sorted(speedups)
    n = len(kernels)
    if n == 0:
        print('nothing to plot (no measured kernels found)')
        return
    ncols = min(6, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.0 * nrows), squeeze=False)

    for i, kernel in enumerate(kernels):
        ax = axes[i // ncols][i % ncols]
        vals = speedups[kernel]
        labels = [l for l in SPEEDUP_VS if l in vals]
        heights = [vals[l] for l in labels]
        colors = [_COLORS[l] for l in labels]
        ax.bar(range(len(labels)), heights, color=colors, width=0.7)
        ax.axhline(1.0, color='#8a8a86', linewidth=1, linestyle='--')
        ax.set_title(kernel, fontsize=8)
        ax.set_xticks([])
        for x, h in enumerate(heights):
            ax.text(x, h, f'{h:.2f}x', ha='center', va='bottom' if h >= 1 else 'top', fontsize=6)

    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].axis('off')

    handles = [plt.Rectangle((0, 0), 1, 1, color=_COLORS[l]) for l in SPEEDUP_VS]
    fig.legend(handles, [f'canon vs {l}' for l in SPEEDUP_VS], loc='upper center', ncol=len(SPEEDUP_VS),
              bbox_to_anchor=(0.5, 1.02), fontsize=9, frameon=False)
    fig.suptitle(f'canonicalize speedup, NPBench+PolyBench ({PRESET} preset)', y=1.06, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'wrote {out_path} ({n} kernels)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results-dir', default='results')
    ap.add_argument('--out', default='npbench_speedup_grid.png')
    args = ap.parse_args()
    plot_grid(collect_speedups(args.results_dir), args.out)


if __name__ == '__main__':
    main()
