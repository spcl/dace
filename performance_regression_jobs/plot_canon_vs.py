#!/usr/bin/env python3
"""canon_vs figure: one panel per corpus (npbench, polybench, tsvc2, tsvc2_5)
showing the distribution of each lane's speedup over that corpus's baseline
(numpy for npbench/polybench, compiler-seq for tsvc2/tsvc2_5), across the
corpus's kernels. Reads results/canon_vs/<corpus>/<kernel>/<tag>/ (results.csv +
status.csv, written by run_perf.py) -- run that first.

    python3 plot_canon_vs.py --results-dir results --out canon_vs.png
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MPI4PY_RC_INITIALIZE', '0')
os.environ.setdefault('OMPI_MCA_pml', 'ob1')
os.environ.setdefault('OMPI_MCA_btl', 'self,vader')
os.environ.setdefault('UCX_VFS_ENABLE', 'n')

import argparse
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import engine

EXPERIMENT = 'canon_vs'
CORPORA = ('npbench', 'polybench', 'tsvc2', 'tsvc2_5')

#: (lane, colour) shown per panel -- the baseline lane is dropped per corpus.
#: Categorical hues from the dataviz skill's validated default palette.
LANE_COLORS = [('dace-canon', '#1baf7a'), ('dace-parallel', '#2a78d6'), ('compiler-autopar', '#eda100'),
               ('compiler-seq', '#d9534f')]


def baseline_lane(corpus):
    return 'numpy' if corpus in ('npbench', 'polybench') else 'compiler-seq'


def merge_kernel(kernel_dir):
    """All lanes for one kernel merged across its tag folders (dace-tag +
    native-tag). Lane names never collide across the two, so one flat dict
    is unambiguous."""
    merged = {}
    for tag in sorted(os.listdir(kernel_dir)):
        p = os.path.join(kernel_dir, tag)
        if os.path.isdir(p):
            merged.update(engine._read_kernel(p))
    return merged


def collect(results_dir, corpus, base_lane, cand_lanes):
    corpus_dir = os.path.join(results_dir, EXPERIMENT, corpus)
    out = {lane: [] for lane in cand_lanes}
    if not os.path.isdir(corpus_dir):
        return out
    for kernel in sorted(os.listdir(corpus_dir)):
        kdir = os.path.join(corpus_dir, kernel)
        if not os.path.isdir(kdir) or kernel == 'native_build':
            continue
        entries = merge_kernel(kdir)
        base = entries.get(base_lane)
        if not base or not base.get('correct') or not base.get('min_ms'):
            continue
        for lane in cand_lanes:
            e = entries.get(lane)
            if e and e.get('correct') and e.get('min_ms'):
                out[lane].append(base['min_ms'] / e['min_ms'])
    return out


def plot(results_dir, out_path):
    per_corpus = {}
    for corpus in CORPORA:
        base = baseline_lane(corpus)
        cands = [(lane, c) for lane, c in LANE_COLORS if lane != base]
        per_corpus[corpus] = (base, cands, collect(results_dir, corpus, base, [lane for lane, _ in cands]))
    corpora = [c for c in CORPORA if any(per_corpus[c][2].values())]
    if not corpora:
        print('nothing to plot (no measured kernels found under results/canon_vs)')
        return

    fig, axes = plt.subplots(1, len(corpora), figsize=(4.2 * len(corpora) + 1, 5.2), squeeze=False)
    for ax, corpus in zip(axes[0], corpora):
        base, cands, data = per_corpus[corpus]
        positions, box_data, colors, ticks = [], [], [], []
        for lane, color in cands:
            vals = data.get(lane, [])
            if not vals:
                continue
            positions.append(len(positions) + 1)
            box_data.append(vals)
            colors.append(color)
            ticks.append(f'{lane}\nn={len(vals)}')
        if not box_data:
            ax.axis('off')
            continue
        bp = ax.boxplot(box_data, positions=positions, patch_artist=True, widths=0.6, showmeans=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
        for elem in ('whiskers', 'caps', 'medians'):
            for line in bp[elem]:
                line.set_color('#52514e')
        ax.axhline(1.0, color='#8a8a86', linewidth=1, linestyle='--')
        ax.set_xticks(positions)
        ax.set_xticklabels(ticks, fontsize=7)
        ax.set_title(f'{corpus}\n(baseline: {base})', fontsize=10)
        ax.set_ylabel('speedup = baseline_min / lane_min  (>1 = faster)')
    fig.suptitle('canon_vs: dace-canon / dace-parallel / native lanes vs per-corpus baseline', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'wrote {out_path} (' + ', '.join(corpora) + ')')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results-dir', default='results', help='results root (contains canon_vs/)')
    ap.add_argument('--out', default='canon_vs.png')
    args = ap.parse_args()
    plot(args.results_dir, args.out)


if __name__ == '__main__':
    main()
