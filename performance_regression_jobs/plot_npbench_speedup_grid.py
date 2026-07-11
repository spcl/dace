#!/usr/bin/env python3
"""Grid plot: parallel and canon (and fast-canon) speedup vs. the
auto_opt baseline, one small cell per NPBench+PolyBench kernel, split by
device (cpu / gpu). Reads results/npbench_polybench/<kernel>/*_paper-<device>/
(results.csv + status.csv, written by npbench_polybench_perf.py) -- run that
script first.

    python3 plot_npbench_speedup_grid.py --results-dir results --out speedup_grid.png
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MPI4PY_RC_INITIALIZE', '0')
os.environ.setdefault('OMPI_MCA_pml', 'ob1')
os.environ.setdefault('OMPI_MCA_btl', 'self,vader')
os.environ.setdefault('UCX_VFS_ENABLE', 'n')

import argparse
import glob
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import engine
from npbench_polybench_perf import CORPUS, PRESET, CANDIDATES, DEVICES, BASELINE_LANE, preset_tag

# Categorical palette (dataviz skill's validated default), fixed hue per candidate.
_COLORS = {'parallel': '#2a78d6', 'canon': '#1baf7a', 'fast-canon': '#eda100'}
#: Device is encoded by hatch so the candidate hue stays readable.
_HATCH = {'cpu': '', 'gpu': '///'}


def collect_speedups(results_dir):
    """kernel -> {(device, candidate): speedup vs auto_opt}. Globs every
    '*_paper-<device>' folder (engine.result_tag namespaces each by
    compiler+hostname), computing the speedup within each folder so a
    per-compiler baseline is never mixed with another compiler's candidate."""
    corpus_dir = os.path.join(results_dir, CORPUS)
    out = {}
    if not os.path.isdir(corpus_dir):
        return out
    for kernel in sorted(os.listdir(corpus_dir)):
        for device in DEVICES:
            for kdir in sorted(glob.glob(os.path.join(corpus_dir, kernel, f'*_{preset_tag(device)}'))):
                if not os.path.isdir(kdir):
                    continue
                entries = engine._read_kernel(kdir)
                base = entries.get(BASELINE_LANE)
                if not base or not base.get('correct') or not base.get('min_ms'):
                    continue
                for cand in CANDIDATES:
                    e = entries.get(cand)
                    if e and e.get('correct') and e.get('min_ms'):
                        out.setdefault(kernel, {})[(device, cand)] = base['min_ms'] / e['min_ms']
    return out


def plot_grid(speedups, out_path):
    kernels = sorted(speedups)
    n = len(kernels)
    if n == 0:
        print('nothing to plot (no measured kernels found)')
        return
    devices = [d for d in DEVICES if any((d, c) in v for v in speedups.values() for c in CANDIDATES)]
    ncols = min(6, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.1 * nrows), squeeze=False)

    for i, kernel in enumerate(kernels):
        ax = axes[i // ncols][i % ncols]
        vals = speedups[kernel]
        bars = [(d, c) for c in CANDIDATES for d in devices if (d, c) in vals]
        heights = [vals[k] for k in bars]
        colors = [_COLORS[c] for (_d, c) in bars]
        hatches = [_HATCH[d] for (d, _c) in bars]
        rects = ax.bar(range(len(bars)), heights, color=colors, width=0.8)
        for r, h in zip(rects, hatches):
            r.set_hatch(h)
        ax.axhline(1.0, color='#8a8a86', linewidth=1, linestyle='--')
        ax.set_title(kernel, fontsize=8)
        ax.set_xticks([])
        for x, h in enumerate(heights):
            ax.text(x, h, f'{h:.2f}x', ha='center', va='bottom' if h >= 1 else 'top', fontsize=5)

    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].axis('off')

    handles = [plt.Rectangle((0, 0), 1, 1, color=_COLORS[c]) for c in CANDIDATES]
    handles += [plt.Rectangle((0, 0), 1, 1, facecolor='#cccccc', hatch=_HATCH[d], edgecolor='#555') for d in devices]
    labels = [f'{c} vs {BASELINE_LANE}' for c in CANDIDATES] + [f'device: {d}' for d in devices]
    fig.legend(handles, labels, loc='upper center', ncol=len(handles), bbox_to_anchor=(0.5, 1.03),
               fontsize=8, frameon=False)
    fig.suptitle(f'speedup vs {BASELINE_LANE}, NPBench+PolyBench ({PRESET} preset)', y=1.07, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'wrote {out_path} ({n} kernels, devices={devices})')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results-dir', default='results')
    ap.add_argument('--out', default='npbench_speedup_grid.png')
    args = ap.parse_args()
    plot_grid(collect_speedups(args.results_dir), args.out)


if __name__ == '__main__':
    main()
