#!/usr/bin/env python3
"""Boxplot: canonicalize's speedup vs. the loop2map+mapfusion baseline,
distributed over every TSVC2 and TSVC2.5 kernel. Reads results/tsvc2/ and
results/tsvc2_5/ (written by tsvc2_perf.py / tsvc2_5_perf.py) -- run those
scripts first.

    python3 plot_tsvc_boxplot.py --results-dir results --out tsvc_boxplot.png
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

CORPORA = ('tsvc2', 'tsvc2_5')
BASELINE_LANE = 'baseline-par'
_COLOR = '#2a78d6'  # single-series -> sequential/categorical slot 1


def collect_speedups(results_dir, corpus):
    """[canon-par speedup vs baseline-par, one value per correctly-measured kernel].
    Globs every '*_default' folder (engine.result_tag namespaces each by
    compiler+hostname) rather than assuming exactly one per kernel."""
    corpus_dir = os.path.join(results_dir, corpus)
    speedups = []
    if not os.path.isdir(corpus_dir):
        return speedups
    for kernel in sorted(os.listdir(corpus_dir)):
        for kdir in sorted(glob.glob(os.path.join(corpus_dir, kernel, '*_default'))):
            if not os.path.isdir(kdir):
                continue
            entries = engine._read_kernel(kdir)
            base = entries.get(BASELINE_LANE)
            canon = entries.get('canon-par')
            if (base and base.get('correct') and base.get('median_ms') and canon and canon.get('correct')
                    and canon.get('median_ms')):
                speedups.append(base['median_ms'] / canon['median_ms'])
    return speedups


def plot_boxplot(data_by_corpus, out_path):
    labels = [c for c in CORPORA if data_by_corpus.get(c)]
    if not labels:
        print('nothing to plot (no measured kernels found)')
        return
    data = [data_by_corpus[c] for c in labels]

    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 2, 5))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5, showmeans=True)
    for patch in bp['boxes']:
        patch.set_facecolor(_COLOR)
        patch.set_alpha(0.35)
    for element in ('whiskers', 'caps', 'medians'):
        for line in bp[element]:
            line.set_color(_COLOR)
    ax.axhline(1.0, color='#8a8a86', linewidth=1, linestyle='--', label='no speedup')
    ax.set_ylabel('canon-par speedup vs. baseline-par (median time ratio)')
    ax.set_title('canonicalize speedup over TSVC2 / TSVC2.5')
    for i, vals in enumerate(data, start=1):
        ax.annotate(f'n={len(vals)}', (i, ax.get_ylim()[0]), textcoords='offset points', xytext=(0, -20),
                    ha='center', fontsize=8, color='#52514e')
    ax.legend(frameon=False, loc='upper right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'wrote {out_path} (' + ', '.join(f'{l}: n={len(d)}' for l, d in zip(labels, data)) + ')')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results-dir', default='results')
    ap.add_argument('--out', default='tsvc_boxplot.png')
    args = ap.parse_args()
    data_by_corpus = {c: collect_speedups(args.results_dir, c) for c in CORPORA}
    plot_boxplot(data_by_corpus, args.out)


if __name__ == '__main__':
    main()
