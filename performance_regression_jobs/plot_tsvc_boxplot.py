#!/usr/bin/env python3
"""Boxplot: the DaCe pipelines' speedup over the native C baselines, distributed
over every TSVC2 / TSVC2.5 kernel. For each corpus the distribution of
    native_baseline_min / dace_min   (best-of-N; >1 == DaCe faster than that native)
is drawn for the two DaCe candidates (dace_parallel, canon) against BOTH native
baselines: single-core (native-clang) and multi-core compiler auto-parallelization
(native-clang-polly-autopar, else native-gcc-autopar). Reads results/tsvc2/ and
results/tsvc2_5/ (written by tsvc2_perf.py / tsvc2_5_perf.py) -- run those first.

    python3 plot_tsvc_boxplot.py --results-dir results --out tsvc_boxplot.png
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
import native_harness as nh

CORPORA = ('tsvc2', 'tsvc2_5')
#: (dace lane, palette colour). The par schedule is the one to pit against the
#: multi-core native lanes; -seq is the fair single-core comparison but the
#: distribution headline uses -par (multi-core DaCe vs multi-core native).
_CANDIDATES = (('dace_parallel-par', '#2a78d6'), ('canon-par', '#1baf7a'))
#: (native baseline lane-or-role, short label used in the legend).
_SINGLE = nh.SINGLE_CORE_LANE
_MULTI = 'multicore'  # resolved per-kernel to whichever MULTICORE_LANES has data


def _merge_kernel(corpus_dir, kernel):
    """All lanes for one kernel merged across its tag folders (dace-tag +
    native-tag, cpu). Lane names never collide across the two, so one flat
    {lane: {'correct','min_ms',...}} is unambiguous."""
    merged = {}
    for kdir in sorted(glob.glob(os.path.join(corpus_dir, kernel, '*_default'))):
        if os.path.isdir(kdir):
            merged.update(engine._read_kernel(kdir))
    return merged


def _best_ms(entry):
    if entry and entry.get('correct') and entry.get('min_ms'):
        return entry['min_ms']
    return None


def _multicore_ms(entries):
    for lane in nh.MULTICORE_LANES:
        ms = _best_ms(entries.get(lane))
        if ms is not None:
            return ms
    return None


def collect(results_dir, corpus):
    """{(candidate, native_role): [speedup, ...]} over the corpus's kernels."""
    corpus_dir = os.path.join(results_dir, corpus)
    out = {(cand, role): [] for cand, _ in _CANDIDATES for role in (_SINGLE, _MULTI)}
    if not os.path.isdir(corpus_dir):
        return out
    for kernel in sorted(os.listdir(corpus_dir)):
        if not os.path.isdir(os.path.join(corpus_dir, kernel)):
            continue
        entries = _merge_kernel(corpus_dir, kernel)
        single = _best_ms(entries.get(_SINGLE))
        multi = _multicore_ms(entries)
        for cand, _color in _CANDIDATES:
            cand_ms = _best_ms(entries.get(cand))
            if cand_ms is None:
                continue
            if single:
                out[(cand, _SINGLE)].append(single / cand_ms)
            if multi:
                out[(cand, _MULTI)].append(multi / cand_ms)
    return out


def plot_boxplot(data_by_corpus, out_path):
    corpora = [c for c in CORPORA if any(data_by_corpus.get(c, {}).values())]
    if not corpora:
        print('nothing to plot (no measured kernels found)')
        return
    series = [(cand, color, role) for cand, color in _CANDIDATES for role in (_SINGLE, _MULTI)]
    role_label = {_SINGLE: 'single-core', _MULTI: 'multicore-autopar'}

    fig, axes = plt.subplots(1, len(corpora), figsize=(4.5 * len(corpora) + 1, 5.5), squeeze=False)
    for ax, corpus in zip(axes[0], corpora):
        data = data_by_corpus[corpus]
        positions, box_data, colors, ticks = [], [], [], []
        for i, (cand, color, role) in enumerate(series):
            vals = data.get((cand, role), [])
            if not vals:
                continue
            positions.append(len(positions) + 1)
            box_data.append(vals)
            colors.append(color)
            ticks.append(f'{cand.split("-")[0]}\nvs {role_label[role]}\nn={len(vals)}')
        if not box_data:
            ax.axis('off')
            continue
        bp = ax.boxplot(box_data, positions=positions, patch_artist=True, widths=0.6, showmeans=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.4)
        for elem in ('whiskers', 'caps', 'medians'):
            for line in bp[elem]:
                line.set_color('#52514e')
        ax.axhline(1.0, color='#8a8a86', linewidth=1, linestyle='--', label='parity (native = DaCe)')
        ax.set_xticks(positions)
        ax.set_xticklabels(ticks, fontsize=7)
        ax.set_title(corpus)
        ax.set_ylabel('speedup = native_min / dace_min  (best-of-N; >1 = DaCe faster)')
        ax.legend(frameon=False, loc='upper right', fontsize=8)
    fig.suptitle('DaCe pipelines vs. native single-core & multi-core-autopar baselines', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'wrote {out_path} (' + ', '.join(corpora) + ')')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results-dir', default='results')
    ap.add_argument('--out', default='tsvc_boxplot.png')
    args = ap.parse_args()
    data_by_corpus = {c: collect(args.results_dir, c) for c in CORPORA}
    plot_boxplot(data_by_corpus, args.out)


if __name__ == '__main__':
    main()
