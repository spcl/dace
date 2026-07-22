#!/usr/bin/env python3
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Plot the external-TU sweep produced by ``run_external_tu_perf.py``.

    python3 plot_external_tu.py exttu_results.tsv [--outdir plots]

Emits (PNG, plus one combined figure):
  P1  build cost, OLD (legacy) codegen, external-TU: native vs cmake  (codegen + compile time)
  P2  build cost, OLD (legacy) codegen, single-TU:   native vs cmake  (codegen + compile time)
  P3  runtime, cmake builder, default flags: NEW (experimental) vs OLD (legacy) codegen
  P4  runtime, cmake builder: the four codegen x split variants

New CPU codegen is at its default (intended-optimal) flags -- see the runner. Build cost is the mean
over kernels; runtime plots show per-kernel speedups so a few dominant kernels do not hide the spread.
"""
import argparse
import csv
import os
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

CMAKE_C, NATIVE_C = '#4C72B0', '#DD8452'
OLD_C, NEW_C = '#8172B3', '#55A868'


def load(path):
    with open(path, newline='') as f:
        rows = [r for r in csv.DictReader(f, delimiter='\t') if not r.get('error')]
    for r in rows:
        for k in ('codegen_ms', 'compile_ms', 'runtime_ms'):
            r[k] = float(r[k]) if r[k] not in ('', None) else None
    return rows


def pick(rows, **eq):
    return [r for r in rows if all(r.get(k) == v for k, v in eq.items())]


def mean(values):
    vals = [v for v in values if v is not None]
    return statistics.fmean(vals) if vals else 0.0


def cell(rows, kernel, field, **eq):
    match = pick(rows, kernel=kernel, **eq)
    return match[0][field] if match and match[0][field] is not None else None


# --------------------------------------------------------------------------- #
def build_cost_plot(ax, rows, split, title):
    """Mean codegen + compile time, cmake vs native, at legacy codegen for the given split."""
    base = dict(codegen='legacy', split=split)
    groups = ['codegen', 'compile']
    cmake = [mean(r['codegen_ms'] for r in pick(rows, build_mode='cmake', **base)),
             mean(r['compile_ms'] for r in pick(rows, build_mode='cmake', **base))]
    native = [mean(r['codegen_ms'] for r in pick(rows, build_mode='native', **base)),
              mean(r['compile_ms'] for r in pick(rows, build_mode='native', **base))]
    x = range(len(groups))
    w = 0.38
    b1 = ax.bar([i - w / 2 for i in x], cmake, w, label='cmake', color=CMAKE_C)
    b2 = ax.bar([i + w / 2 for i in x], native, w, label='native', color=NATIVE_C)
    ax.bar_label(b1, fmt='%.0f', fontsize=8)
    ax.bar_label(b2, fmt='%.0f', fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(['codegen', 'compile (build)'])
    ax.set_ylabel('mean time [ms]')
    ax.set_title(title)
    ax.legend()


def runtime_new_vs_old_plot(ax, rows):
    """Per-kernel runtime speedup old/new (experimental vs legacy), cmake builder, single-TU (defaults)."""
    base = dict(build_mode='cmake', split='off')
    kernels = sorted({r['kernel'] for r in rows})
    speedups = []
    for k in kernels:
        old = cell(rows, k, 'runtime_ms', codegen='legacy', **base)
        new = cell(rows, k, 'runtime_ms', codegen='experimental', **base)
        if old and new:
            speedups.append((k, old / new))
    speedups.sort(key=lambda kv: kv[1])
    if not speedups:
        ax.text(0.5, 0.5, 'no legacy+experimental runtime pairs', ha='center', va='center')
        ax.set_title('P3 runtime: new vs old (cmake, defaults)')
        return
    names = [k for k, _ in speedups]
    vals = [v for _, v in speedups]
    colors = [NEW_C if v >= 1 else OLD_C for v in vals]
    ax.barh(range(len(names)), vals, color=colors)
    ax.axvline(1.0, color='k', lw=0.8, ls='--')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('speedup  (legacy_runtime / experimental_runtime)  >1 = new faster')
    gm = statistics.fmean(vals)
    ax.set_title(f'P3 runtime: new vs old (cmake, defaults) -- geo-ish mean {gm:.2f}x')


def four_variant_runtime_plot(ax, rows):
    """Mean runtime of the four codegen x split variants (cmake builder)."""
    variants = [('legacy', 'off', 'old, single'), ('legacy', 'on', 'old, external'),
                ('experimental', 'off', 'new, single'), ('experimental', 'on', 'new, external')]
    labels, vals = [], []
    for cg, sp, lab in variants:
        vals.append(mean(r['runtime_ms'] for r in pick(rows, codegen=cg, split=sp, build_mode='cmake')))
        labels.append(lab)
    colors = [OLD_C, OLD_C, NEW_C, NEW_C]
    bars = ax.bar(labels, vals, color=colors)
    ax.bar_label(bars, fmt='%.2f', fontsize=8)
    ax.set_ylabel('mean runtime [ms]')
    ax.set_title('P4 runtime: 4 variants (cmake)')
    ax.tick_params(axis='x', labelrotation=15)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('tsv')
    ap.add_argument('--outdir', default=None, help='directory for PNGs (default: alongside the TSV)')
    args = ap.parse_args()
    rows = load(args.tsv)
    if not rows:
        print('no successful rows in TSV', file=sys.stderr)
        sys.exit(2)
    outdir = args.outdir or os.path.dirname(os.path.abspath(args.tsv))
    os.makedirs(outdir, exist_ok=True)

    # Individual figures.
    specs = [
        ('p1_build_old_external', lambda ax: build_cost_plot(ax, rows, 'on',
                                                             'P1 build: OLD codegen, external-TU (native vs cmake)')),
        ('p2_build_old_single', lambda ax: build_cost_plot(ax, rows, 'off',
                                                           'P2 build: OLD codegen, single-TU (native vs cmake)')),
        ('p3_runtime_new_vs_old', lambda ax: runtime_new_vs_old_plot(ax, rows)),
        ('p4_runtime_four_variants', lambda ax: four_variant_runtime_plot(ax, rows)),
    ]
    for stem, draw in specs:
        fig, ax = plt.subplots(figsize=(7, 5))
        draw(ax)
        fig.tight_layout()
        path = os.path.join(outdir, stem + '.png')
        fig.savefig(path, dpi=140)
        plt.close(fig)
        print('wrote', path)

    # Combined 2x2 overview.
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    build_cost_plot(axes[0][0], rows, 'on', 'P1 OLD codegen, external-TU')
    build_cost_plot(axes[0][1], rows, 'off', 'P2 OLD codegen, single-TU')
    runtime_new_vs_old_plot(axes[1][0], rows)
    four_variant_runtime_plot(axes[1][1], rows)
    fig.suptitle('External-TU codegen/build/runtime sweep', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    combined = os.path.join(outdir, 'external_tu_overview.png')
    fig.savefig(combined, dpi=140)
    plt.close(fig)
    print('wrote', combined)


if __name__ == '__main__':
    main()
