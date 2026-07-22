#!/usr/bin/env python3
"""Normalized median-speedup scatter, npbench-style, for the canon_vs run.

Two figures, each merging a pair of corpora and normalizing every framework
lane to that pair's baseline (>1 = faster than baseline):

    * npbench + polybench   -> baseline = numpy
      (the only lanes that run: numpy + the three dace lanes; the native
       compiler lanes have no C source here and are always N/A)
    * tsvc2   + tsvc2_5      -> baseline = compiler-seq
      (no numpy reference exists for these C-loop kernels, so the sequential
       native compile is the baseline)

For each kernel x lane the representative time is the *median across tags*
(compiler x node) of that lane's per-run median_ms, so a lane with several
tag folders collapses to one robust number. speedup = median(baseline) /
median(lane), computed only where both are correct and timed.

Each figure is a scatter: one column per kernel (sorted by the reference
lane's speedup), median speedup on a log y-axis, one coloured marker series
per lane, dashed line at y=1 (baseline). A companion markdown file gives the
per-lane geomean summary plus the full per-kernel speedup table.

    python3 plot_speedup_scatter.py --results-dir results --out-prefix speedup_scatter

Reads results/canon_vs/<corpus>/summary.csv (written by run_perf.py) -- run
that first.
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MPI4PY_RC_INITIALIZE', '0')
os.environ.setdefault('OMPI_MCA_pml', 'ob1')
os.environ.setdefault('OMPI_MCA_btl', 'self,vader')
os.environ.setdefault('UCX_VFS_ENABLE', 'n')

import argparse
import csv
import math
import statistics
import sys

# A failed kernel's traceback in status.csv can exceed csv's 128 KB default field cap; lift it
# so any raw csv read here stays robust (matches engine.py).
csv.field_size_limit(min(2 ** 31 - 1, sys.maxsize))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPERIMENT = 'canon_vs'

#: The two merged figures: (slug, corpora, baseline lane).
GROUPS = [
    ('npbench_polybench', ('npbench', 'polybench'), 'numpy'),
    ('tsvc', ('tsvc2', 'tsvc2_5'), 'compiler-seq'),
]

#: Candidate lanes shown per figure (baseline is dropped), fixed order + hue.
#: dace hues match plot_canon_vs; numpy/native carry their own steady colours.
#: Categorical palette from the dataviz skill's validated default.
LANE_COLORS = [
    ('dace-autoopt', '#e07a2b'),
    ('dace-canon', '#1baf7a'),
    ('dace-parallel', '#2a78d6'),
    ('compiler-autopar', '#eda100'),
    ('compiler-seq', '#d9534f'),
    ('numpy', '#7a7a7a'),
]
#: Marker per lane so series stay separable even where hues sit close.
LANE_MARKERS = {
    'dace-autoopt': 'o', 'dace-canon': 's', 'dace-parallel': '^',
    'compiler-autopar': 'D', 'compiler-seq': 'v', 'numpy': 'x',
}
#: Lane the scatter is sorted by (first present wins) -- dace-canon leads so the
#: kernels read left-to-right by canonicalized-lane speedup.
SORT_PREF = ('dace-canon', 'dace-parallel', 'dace-autoopt', 'compiler-autopar')


def geomean(vals):
    vals = [v for v in vals if v and v > 0 and math.isfinite(v)]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def load_lane_times(results_dir, corpus):
    """{kernel: {'ok': {lane: median_ms}, 'ran': set(lanes)}} for one corpus.

    'ok' holds only lanes that are correct AND timed -- one number per
    (kernel, lane), the median across that lane's tag rows in summary.csv.
    'ran' is every lane that produced a row (correct or not), so callers can
    report how many results were excluded for failing correctness."""
    path = os.path.join(results_dir, EXPERIMENT, corpus, 'summary.csv')
    if not os.path.isfile(path):
        return {}
    ok_buckets, ran = {}, {}
    with open(path, newline='') as fh:
        for r in csv.DictReader(fh):
            kernel, lane = r['kernel'], r['pipeline']
            ran.setdefault(kernel, set()).add(lane)
            if r['correct'].strip().lower() != 'true' or not r['median_ms'].strip():
                continue
            try:
                ms = float(r['median_ms'])
            except ValueError:
                continue
            if ms <= 0:
                continue
            ok_buckets.setdefault(kernel, {}).setdefault(lane, []).append(ms)
    return {k: {'ok': {lane: statistics.median(v) for lane, v in ok_buckets.get(k, {}).items()},
                'ran': ran[k]}
            for k in ran}


def collect_group(results_dir, corpora, base_lane, cand_lanes):
    """Merge the group's corpora into a flat kernel list. Kernel keys are
    'corpus/kernel' so same-named kernels across corpora never collide.

    A speedup is emitted ONLY when both the lane and the baseline are correct
    vs the reference (i.e. present in 'ok'); a lane that ran but failed
    correctness, or a kernel whose baseline itself is incorrect, is dropped and
    counted. Returns ({kernel_key: {lane: speedup}}, exclusion-stats dict)."""
    out = {}
    excl = {'no_baseline': 0, 'baseline_incorrect': 0,
            'per_lane_incorrect': {lane: 0 for lane in cand_lanes}}
    for corpus in corpora:
        for kernel, rec in load_lane_times(results_dir, corpus).items():
            ok, ran = rec['ok'], rec['ran']
            base = ok.get(base_lane)
            if not base:  # no correct baseline -> speedup is undefined, skip kernel
                excl['no_baseline'] += 1
                if base_lane in ran:
                    excl['baseline_incorrect'] += 1
                continue
            speeds = {}
            for lane in cand_lanes:
                if ok.get(lane):
                    speeds[lane] = base / ok[lane]
                elif lane in ran:  # lane ran but was incorrect -> excluded
                    excl['per_lane_incorrect'][lane] += 1
            if speeds:
                out[f'{corpus}/{kernel}'] = speeds
    return out, excl


def sort_key(speeds):
    for lane in SORT_PREF:
        if lane in speeds:
            return speeds[lane]
    return max(speeds.values()) if speeds else 0.0


def plot_group(slug, data, base_lane, cand_lanes, out_path):
    """Scatter for one merged group; returns the per-lane geomean summary."""
    kernels = sorted(data, key=lambda k: sort_key(data[k]), reverse=True)
    present = [(lane, color) for lane, color in cand_lanes
               if any(lane in data[k] for k in kernels)]
    summary = []  # (lane, geomean, n, n_faster)
    if not kernels or not present:
        print(f'[{slug}] nothing to plot')
        return summary

    fig_w = max(7.0, min(28.0, len(kernels) * 0.11))
    fig, ax = plt.subplots(figsize=(fig_w, 6.0))
    x = range(len(kernels))
    for lane, color in present:
        xs, ys = [], []
        for i, k in enumerate(kernels):
            if lane in data[k]:
                xs.append(i)
                ys.append(data[k][lane])
        vals = ys
        summary.append((lane, geomean(vals), len(vals), sum(1 for v in vals if v > 1.0)))
        gm = geomean(vals)
        ax.scatter(xs, ys, s=22, c=color, marker=LANE_MARKERS.get(lane, 'o'),
                   alpha=0.75, edgecolors='none',
                   label=f'{lane}  (gm {gm:.2f}x, n={len(vals)})' if gm else lane)

    ax.axhline(1.0, color='#8a8a86', linewidth=1, linestyle='--')
    ax.set_yscale('log')
    ax.set_ylabel('median speedup vs baseline  (log, >1 = faster)')
    ax.set_title(f'{slug}: median speedup over {base_lane}  (sorted by {_ref_lane(present)})')
    # kernel names on the x-axis, rotated 90deg; font shrinks as kernels grow.
    ax.set_xticks(list(x))
    fs = max(3.0, min(7.0, 620.0 / max(1, len(kernels))))
    ax.set_xticklabels(kernels, rotation=90, fontsize=fs)
    ax.set_xlabel(f'kernel (n={len(kernels)})')
    ax.margins(x=0.005)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out_pdf = os.path.splitext(out_path)[0] + '.pdf'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')  # high-dpi raster
    fig.savefig(out_pdf, bbox_inches='tight')             # vector (scales losslessly)
    plt.close(fig)
    print(f'wrote {out_path} + {out_pdf} ({len(kernels)} kernels, lanes={[l for l, _ in present]})')
    return summary


def _ref_lane(present):
    present_lanes = [l for l, _ in present]
    for lane in SORT_PREF:
        if lane in present_lanes:
            return lane
    return present_lanes[0] if present_lanes else '?'


def excl_report(base_lane, excl):
    """One-line-per-fact list of what correctness gating dropped."""
    lines = [f'Correctness gating (speedup shown only where correct vs {base_lane}):',
             f'  - kernels dropped (no correct {base_lane} baseline): {excl["no_baseline"]}'
             f' (of which {base_lane} ran but was incorrect: {excl["baseline_incorrect"]})']
    for lane, n in excl['per_lane_incorrect'].items():
        if n:
            lines.append(f'  - {lane}: {n} kernel result(s) excluded (ran but incorrect)')
    return lines


def write_markdown(groups_out, out_path):
    lines = ['# Normalized median speedup (canon_vs)', '']
    for slug, base_lane, corpora, data, summary, cand_lanes, excl in groups_out:
        lines.append(f'## {slug}  ({" + ".join(corpora)}, baseline = {base_lane})')
        lines.append('')
        for ln in excl_report(base_lane, excl):
            lines.append(ln)
        lines.append('')
        if not summary:
            lines.append('_no measured kernels_\n')
            continue
        lines.append('| lane | geomean speedup | kernels | % faster than baseline |')
        lines.append('|------|----------------:|--------:|-----------------------:|')
        for lane, gm, n, nf in summary:
            gm_s = f'{gm:.2f}x' if gm else 'n/a'
            pct = f'{100.0 * nf / n:.0f}%' if n else 'n/a'
            lines.append(f'| {lane} | {gm_s} | {n} | {pct} |')
        lines.append('')
        lane_order = [l for l, _ in cand_lanes if any(l in data[k] for k in data)]
        header = '| kernel | ' + ' | '.join(lane_order) + ' |'
        sep = '|' + '---|' * (len(lane_order) + 1)
        lines.append(header)
        lines.append(sep)
        for k in sorted(data, key=lambda k: sort_key(data[k]), reverse=True):
            cells = [f'{data[k][l]:.2f}x' if l in data[k] else '' for l in lane_order]
            lines.append(f'| {k} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    with open(out_path, 'w') as fh:
        fh.write('\n'.join(lines))
    print(f'wrote {out_path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results-dir', default='results', help='results root (contains canon_vs/)')
    ap.add_argument('--out-prefix', default='speedup_scatter',
                    help='output prefix; writes <prefix>_<group>.png and <prefix>.md')
    args = ap.parse_args()

    groups_out = []
    for slug, corpora, base_lane in GROUPS:
        cand_lanes = [(l, c) for l, c in LANE_COLORS if l != base_lane]
        data, excl = collect_group(args.results_dir, corpora, base_lane, [l for l, _ in cand_lanes])
        out_png = f'{args.out_prefix}_{slug}.png'
        summary = plot_group(slug, data, base_lane, cand_lanes, out_png)
        for ln in excl_report(base_lane, excl):
            print(f'[{slug}] {ln}')
        groups_out.append((slug, base_lane, corpora, data, summary, cand_lanes, excl))
    write_markdown(groups_out, f'{args.out_prefix}.md')


if __name__ == '__main__':
    main()
