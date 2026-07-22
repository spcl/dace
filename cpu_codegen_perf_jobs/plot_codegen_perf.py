#!/usr/bin/env python3
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Plot the readable-vs-legacy codegen sweep (run_readable_perf.py / the sbatch runners).

Reads one or more TSV files whose header is (tab-separated):

    kernel  corpus  codegen  preset  threads  cxx  phase  codegen_ms  compile_ms  runtime_ms
    speedup  correctness  error

``codegen`` is ``legacy`` or ``experimental``; ``corpus`` is npbench|polybench|tsvc2|
tsvc2_5. ``codegen_ms`` (DaCe code generation) and ``compile_ms`` (the C++ compiler) are
SEPARATE, not summed -- both are blank on error rows, and ``codegen_ms`` is blank for GPU
rows. ``speedup`` is legacy_runtime / this_runtime. ``cxx`` (g++ | clang++) is the host
compiler the row was built with -- on a gpu row the device compiler is always nvcc, so it
stays the host one -- and ``phase`` is single_core | multi_core | gpu: submit_daint_readable.sbatch
merges all of its phases into ONE TSV, so these are what separate the lanes again. Both are
read by column name, so their position in the header does not matter; older TSVs that predate
them still work -- without ``phase`` the mode is derived from ``preset`` (S = single core,
paper = multi-core) and every row is assumed to be CPU, with a printed note. Rows carrying a
non-empty ``error`` are skipped and counted.

Figures (each written to --out-dir as both PNG and PDF):

  FIGURE A  runtime_<phase>       total-runtime comparison, legacy vs experimental, as
                                  speedup = legacy / experimental (runtimes span orders of
                                  magnitude across kernels, so the ratio is the readable
                                  axis). One figure per phase -- single_core / multi_core /
                                  gpu -- with one panel per `cxx` group inside it, so
                                  "g++ vs clang++ CPU, before/after" and "GPU" each read
                                  off directly. Dashed parity line at 1.0, geomean
                                  annotated per group.
  FIGURE B  build_and_quality     multi-panel: (i) STACKED build time per kernel --
                                  compile_ms on the BOTTOM, codegen_ms stacked ON TOP,
                                  legacy vs experimental grouped; (ii) generated LoC,
                                  legacy vs experimental; (iii) readability panel
                                  (max_nesting + nloc headline, tokens_per_stmt
                                  normalized, max_ccn as an unchanged control).

Panels (ii)/(iii) need generated sources: pass --srcdir (walked recursively; any of the
per-variant layouts readability_metrics.infer_kernel_codegen understands) or --metrics-csv
(precomputed, from `readability_metrics.py --csv`). Without either, those panels are
skipped with a printed note. Missing columns / presets / files never crash a run.

    python3 plot_codegen_perf.py --tsv results.tsv --out-dir plots/
    python3 plot_codegen_perf.py --tsv s.tsv paper.tsv --out-dir plots/ --srcdir generated/
    python3 plot_codegen_perf.py --tsv r.tsv --out-dir plots/ --metrics-csv metrics.csv
"""
import argparse
import csv
import math
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from readability_metrics import (CONTROL_METRICS, METRIC_LABEL, SIZE_SENSITIVE_METRICS, find_sources,
                                     infer_kernel_codegen, readability_score)
    METRICS_IMPORT_ERROR = None
except ImportError as exc:  # the perf figures must still render without the metrics helper
    METRICS_IMPORT_ERROR = str(exc)
    CONTROL_METRICS = ('max_ccn', )
    SIZE_SENSITIVE_METRICS = ('tokens', )
    METRIC_LABEL = {}

CODEGENS = ('legacy', 'experimental')

#: Build-time bar segments: C++ compile on the bottom, DaCe codegen stacked on top.
#: Categorical hues from the dataviz skill's validated default palette.
SEGMENT_COLORS = {'compile': '#2a78d6', 'codegen': '#e07a2b'}
SEGMENT_LABEL = {'compile': 'C++ compile', 'codegen': 'DaCe codegen'}
#: The codegen lane is encoded by hatch so the segment hue stays readable.
LANE_HATCH = {'legacy': '', 'experimental': '///'}
#: Lane hues for the non-stacked (LoC / metric) panels.
LANE_COLOR = {'legacy': '#2a78d6', 'experimental': '#e07a2b'}
#: Speedup bars: green where experimental is at least as fast as legacy, red where it
#: regressed (readability is meant to be perf-neutral, so red flags a guard hit).
FASTER_COLOR = '#1baf7a'
SLOWER_COLOR = '#d9534f'
NEUTRAL_COLOR = '#8a8a86'

#: phase -> (display label, sort key). Anything else sorts last under its own name.
PHASE_LABEL = {'single_core': 'single core', 'multi_core': 'multi-core', 'gpu': 'GPU'}
PHASE_ORDER = ('single_core', 'multi_core', 'gpu')
#: preset -> phase, for TSVs written before the phase column existed.
PRESET_PHASE = {'S': 'single_core', 'paper': 'multi_core'}
#: Stand-in group name when the TSV has no cxx column (or the cell is blank).
CXX_UNKNOWN = 'cxx n/a'

#: The readability panel's metrics, in reading order, with how each is to be read.
PANEL_METRICS = ('nloc', 'max_nesting', 'tokens_per_stmt', 'tokens', 'max_ccn')
PANEL_METRIC_TAG = {
    'nloc': 'headline',
    'max_nesting': 'headline',
    'tokens_per_stmt': 'normalized',
    'tokens': 'size-sensitive',
    'max_ccn': 'control',
}


def as_float(val):
    """Parse a TSV cell to float, mapping empty / 'n/a' / non-numeric to None."""
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in ('n/a', 'na', 'nan', 'none', '-'):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def geomean(vals):
    vals = [v for v in vals if v and v > 0 and math.isfinite(v)]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def mean(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def load_rows(paths):
    """Read every TSV, dropping error rows. Returns (rows, columns, stats). ``columns`` is
    the union of headers seen (so a figure can check a needed column is present); ``stats``
    counts files read, data rows and rows skipped for a non-empty error."""
    rows, columns = [], set()
    total = skipped_error = files = 0
    for path in paths:
        if not os.path.isfile(path):
            print(f'warning: {path} not found -- skipping')
            continue
        files += 1
        with open(path, newline='') as fh:
            reader = csv.DictReader(fh, delimiter='\t')
            if reader.fieldnames:
                columns.update(fn for fn in reader.fieldnames if fn)
            for r in reader:
                total += 1
                if (r.get('error') or '').strip():
                    skipped_error += 1
                    continue
                rows.append(r)
    stats = dict(files=files, total=total, skipped_error=skipped_error, kept=len(rows))
    return rows, columns, stats


def row_phase(row):
    """single_core | multi_core | gpu, from the phase column if the sibling runner wrote
    one, else derived from preset (S = single core, paper = multi-core)."""
    phase = (row.get('phase') or '').strip().lower()
    if phase:
        return phase
    preset = (row.get('preset') or '').strip()
    return PRESET_PHASE.get(preset, preset.lower() or 'unknown')


def row_cxx(row):
    return (row.get('cxx') or '').strip() or CXX_UNKNOWN


def index_rows(rows):
    """{(phase, cxx, kernel): {codegen: row}}. Rank-split TSVs cover disjoint kernels, so a
    late duplicate simply overwrites (a re-run of the same cell keeps the latest)."""
    idx = {}
    for r in rows:
        key = (row_phase(r), row_cxx(r), (r.get('kernel') or '').strip())
        idx.setdefault(key, {})[(r.get('codegen') or '').strip()] = r
    return idx


def phase_sort_key(phase):
    return (PHASE_ORDER.index(phase), phase) if phase in PHASE_ORDER else (len(PHASE_ORDER), phase)


def cxx_sort_key(cxx):
    return (cxx == CXX_UNKNOWN, cxx)


def phases_present(idx):
    return sorted({p for (p, _c, _k) in idx if p}, key=phase_sort_key)


def cxx_present(idx, phase=None):
    return sorted({c for (p, c, _k) in idx if phase is None or p == phase}, key=cxx_sort_key)


def phase_label(phase):
    return PHASE_LABEL.get(phase, phase.replace('_', ' '))


def group_label(phase, cxx):
    return phase_label(phase) if cxx == CXX_UNKNOWN else f'{cxx}, {phase_label(phase)}'


def save(fig, out_dir, base):
    """Write <base>.png (raster) + <base>.pdf (vector) under out_dir."""
    png = os.path.join(out_dir, base + '.png')
    pdf = os.path.join(out_dir, base + '.pdf')
    fig.savefig(png, dpi=150, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    return png, pdf


def tick_fontsize(n):
    return max(3.0, min(8.0, 620.0 / max(1, n)))


# --------------------------------------------------------------------------
# Figure A -- runtime: experimental-vs-legacy speedup, grouped by cxx / phase.
# --------------------------------------------------------------------------
def pair_speedup(lanes):
    """legacy / experimental runtime for one kernel (>1 = experimental faster).

    Computed from the two runtimes when both lanes are present; falls back to the
    experimental row's own ``speedup`` column (same definition) when the legacy runtime is
    missing from this TSV."""
    legacy, experimental = lanes.get('legacy'), lanes.get('experimental')
    if not experimental:
        return None
    lr = as_float(legacy.get('runtime_ms')) if legacy else None
    er = as_float(experimental.get('runtime_ms'))
    if lr and er and lr > 0 and er > 0:
        return lr / er
    s = as_float(experimental.get('speedup'))
    return s if s and s > 0 else None


def speedup_pairs(idx, phase, cxx):
    pairs = []
    for (p, c, kernel), lanes in idx.items():
        if p != phase or c != cxx:
            continue
        s = pair_speedup(lanes)
        if s:
            pairs.append((kernel, s))
    pairs.sort(key=lambda kv: kv[1], reverse=True)
    return pairs


def draw_speedup_panel(ax, pairs, title):
    kernels = [k for k, _ in pairs]
    speedups = [s for _, s in pairs]
    n = len(kernels)
    gm = geomean(speedups)
    gm_s = f'{gm:.2f}x' if gm else 'n/a'

    ax.bar(range(n), speedups, color=[FASTER_COLOR if s >= 1.0 else SLOWER_COLOR for s in speedups], width=0.8)
    ax.axhline(1.0, color=NEUTRAL_COLOR, linewidth=1, linestyle='--', label='parity (legacy = experimental)')
    if gm:
        ax.axhline(gm, color='#3c3c3a', linewidth=1, linestyle=':', label=f'geomean {gm_s}')
    if n <= 45:
        afs = max(4.0, min(7.0, 300.0 / max(1, n)))
        for x, s in enumerate(speedups):
            ax.text(x, s, f'{s:.2f}', ha='center', va='bottom' if s >= 1 else 'top', fontsize=afs)

    ax.set_xticks(range(n))
    ax.set_xticklabels(kernels, rotation=90, fontsize=tick_fontsize(n))
    ax.set_xlabel(f'kernel (n={n})')
    ax.set_ylabel('speedup = legacy / experimental\n(>1 = experimental faster)', fontsize=9)
    #: The geomean rides in the title, not a text box: at 40+ kernels a box either covers
    #: bars or fights the legend, and the title is where the reader looks for the summary.
    ax.set_title(f'{title}  --  geomean {gm_s} (n={n})', fontsize=10)
    ax.margins(x=0.01)
    ax.set_ylim(0, max(speedups + [1.0]) * 1.3)  # headroom so the legend clears the bars
    ax.legend(loc='upper right', fontsize=7, framealpha=0.9)
    return gm, n


def plot_runtime(idx, columns, out_dir):
    """FIGURE A: one figure per phase, one panel per cxx group inside it."""
    if 'runtime_ms' not in columns and 'speedup' not in columns:
        print('[runtime] skipped: TSV lacks both a runtime_ms and a speedup column')
        return
    phases = phases_present(idx)
    if not phases:
        print('[runtime] skipped: no rows to plot')
        return

    for phase in phases:
        groups = []
        for cxx in cxx_present(idx, phase):
            pairs = speedup_pairs(idx, phase, cxx)
            if pairs:
                groups.append((cxx, pairs))
        base = f'runtime_{phase}'
        if not groups:
            print(f'[{base}] skipped: no kernel has a legacy/experimental runtime pair for phase {phase!r}')
            continue

        widest = max(len(p) for _c, p in groups)
        fig_w = max(7.0, min(34.0, widest * 0.34))
        fig, axes = plt.subplots(len(groups), 1, figsize=(fig_w, 4.6 * len(groups)), squeeze=False)
        summary = []
        for ax, (cxx, pairs) in zip(axes[:, 0], groups):
            gm, n = draw_speedup_panel(ax, pairs, f'Experimental vs legacy runtime -- {group_label(phase, cxx)}')
            summary.append(f'{cxx}: geomean {gm:.2f}x (n={n})' if gm else f'{cxx}: n/a')
        fig.suptitle(f'Runtime, legacy vs experimental codegen -- {phase_label(phase)}', fontsize=12, y=1.0)
        fig.tight_layout()
        png, pdf = save(fig, out_dir, base)
        print(f'wrote {png} + {pdf}  [{"; ".join(summary)}]')


# --------------------------------------------------------------------------
# Figure B -- build time (stacked) + generated-code quality.
# --------------------------------------------------------------------------
def build_kernels(idx, phase, cxx):
    """Kernels with at least one compile/codegen timing, for one (phase, cxx) cell."""
    lanes_by_kernel = {k: lanes for (p, c, k), lanes in idx.items() if p == phase and c == cxx}
    kernels = []
    for k in sorted(lanes_by_kernel):
        lanes = lanes_by_kernel[k]
        if any(
                lanes.get(cg) and
            (as_float(lanes[cg].get('compile_ms')) is not None or as_float(lanes[cg].get('codegen_ms')) is not None)
                for cg in CODEGENS):
            kernels.append(k)
    return lanes_by_kernel, kernels


def draw_build_panel(ax, lanes_by_kernel, kernels, title):
    """Stacked build time: compile_ms on the bottom, codegen_ms stacked on top, with the
    two codegen lanes as adjacent grouped bars (lane = hatch, segment = hue)."""
    group_w = 0.8
    bar_w = group_w / len(CODEGENS)
    tallest = 0.0
    for j, cg in enumerate(CODEGENS):
        offset = (j - (len(CODEGENS) - 1) / 2.0) * bar_w
        for i, k in enumerate(kernels):
            row = lanes_by_kernel[k].get(cg)
            if not row:
                continue
            compile_v = as_float(row.get('compile_ms')) or 0.0
            codegen_v = as_float(row.get('codegen_ms')) or 0.0
            tallest = max(tallest, compile_v + codegen_v)
            x = i + offset
            ax.bar(x,
                   compile_v,
                   bar_w,
                   bottom=0.0,
                   color=SEGMENT_COLORS['compile'],
                   hatch=LANE_HATCH[cg],
                   edgecolor='white',
                   linewidth=0.4)
            ax.bar(x,
                   codegen_v,
                   bar_w,
                   bottom=compile_v,
                   color=SEGMENT_COLORS['codegen'],
                   hatch=LANE_HATCH[cg],
                   edgecolor='white',
                   linewidth=0.4)

    ax.set_xticks(range(len(kernels)))
    ax.set_xticklabels(kernels, rotation=90, fontsize=tick_fontsize(len(kernels)))
    ax.set_xlabel(f'kernel (n={len(kernels)})')
    ax.set_ylabel('build time (ms)', fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.margins(x=0.01)
    if tallest > 0:
        ax.set_ylim(0, tallest * 1.22)  # headroom so the legend clears the bars

    handles = [plt.Rectangle((0, 0), 1, 1, color=SEGMENT_COLORS[s]) for s in ('compile', 'codegen')]
    labels = [SEGMENT_LABEL[s] for s in ('compile', 'codegen')]
    handles += [
        plt.Rectangle((0, 0), 1, 1, facecolor='#cccccc', hatch=LANE_HATCH[cg], edgecolor='#555') for cg in CODEGENS
    ]
    labels += list(CODEGENS)
    ax.legend(handles, labels, loc='upper right', fontsize=7, framealpha=0.9)


def draw_loc_panel(ax, metrics):
    """Generated LoC per kernel, legacy vs experimental grouped bars."""
    kernels = [
        k for k in metrics if metrics[k].get('legacy', {}).get('nloc') or metrics[k].get('experimental', {}).get('nloc')
    ]
    kernels.sort(key=lambda k: -(metrics[k].get('legacy', {}).get('nloc') or 0))
    if not kernels:
        return False, 'no kernel has an nloc value'

    group_w = 0.8
    bar_w = group_w / len(CODEGENS)
    for j, cg in enumerate(CODEGENS):
        offset = (j - (len(CODEGENS) - 1) / 2.0) * bar_w
        xs, ys = [], []
        for i, k in enumerate(kernels):
            v = metrics[k].get(cg, {}).get('nloc')
            if v:
                xs.append(i + offset)
                ys.append(v)
        ax.bar(xs, ys, bar_w, color=LANE_COLOR[cg], edgecolor='white', linewidth=0.4, label=cg)

    ratios = [
        metrics[k]['experimental']['nloc'] / metrics[k]['legacy']['nloc'] for k in kernels
        if metrics[k].get('legacy', {}).get('nloc') and metrics[k].get('experimental', {}).get('nloc')
    ]
    gm = geomean(ratios)
    ax.set_xticks(range(len(kernels)))
    ax.set_xticklabels(kernels, rotation=90, fontsize=tick_fontsize(len(kernels)))
    ax.set_xlabel(f'kernel (n={len(kernels)})')
    ax.set_ylabel('generated lines of code (nloc)', fontsize=9)
    title = 'Generated code size: legacy vs experimental'
    if gm:
        title += f'  (geomean experimental/legacy = {gm:.2f}x)'
    ax.set_title(title, fontsize=10)
    ax.margins(x=0.01)
    tallest = max((v or 0) for k in kernels for v in (metrics[k].get(cg, {}).get('nloc') for cg in CODEGENS))
    if tallest > 0:
        ax.set_ylim(0, tallest * 1.22)  # headroom so the legend clears the bars
    ax.legend(loc='upper right', fontsize=7, framealpha=0.9)
    return True, None


def metric_ratio_color(metric, ratio):
    if metric in CONTROL_METRICS or metric in SIZE_SENSITIVE_METRICS:
        return NEUTRAL_COLOR
    return FASTER_COLOR if ratio <= 1.0 else SLOWER_COLOR


def draw_readability_panel(ax, metrics):
    """Per-metric geomean of experimental/legacy, so metrics on wildly different scales
    (nloc in the hundreds, nesting in the single digits) share one axis. Lower = more
    readable for every metric here; max_ccn is the control and is expected to sit on 1.0."""
    shown, labels, ratios, colors, absolutes = [], [], [], [], []
    for metric in PANEL_METRICS:
        per_kernel = []
        legacy_vals, experimental_vals = [], []
        for lanes in metrics.values():
            lv = lanes.get('legacy', {}).get(metric)
            ev = lanes.get('experimental', {}).get(metric)
            if lv and ev and lv > 0 and ev > 0:
                per_kernel.append(ev / lv)
                legacy_vals.append(lv)
                experimental_vals.append(ev)
        gm = geomean(per_kernel)
        if not gm:
            continue
        shown.append(metric)
        ratios.append(gm)
        colors.append(metric_ratio_color(metric, gm))
        labels.append(f'{metric}\n({PANEL_METRIC_TAG.get(metric, "")}, n={len(per_kernel)})')
        absolutes.append((mean(legacy_vals), mean(experimental_vals)))
    if not shown:
        return False, 'no metric has a legacy/experimental pair'

    x = range(len(shown))
    ax.bar(x, ratios, 0.6, color=colors, edgecolor='white', linewidth=0.5)
    ax.axhline(1.0, color=NEUTRAL_COLOR, linewidth=1, linestyle='--', label='parity (legacy = experimental)')
    for i, (ratio, (leg, exp)) in enumerate(zip(ratios, absolutes)):
        note = f'{ratio:.2f}x'
        if leg is not None and exp is not None:
            note += f'\n{leg:.1f} -> {exp:.1f}'
        ax.text(i, ratio, note, ha='center', va='bottom', fontsize=7)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('experimental / legacy (geomean)\n<1 = experimental more readable', fontsize=9)
    ax.set_title('Generated-code readability (mean absolute values under each bar)', fontsize=10)
    ax.set_ylim(0, max(1.35, max(ratios) * 1.35))
    ax.legend(loc='upper left', fontsize=7, framealpha=0.9)

    print('readability panel -- how to read each metric:')
    for metric, ratio in zip(shown, ratios):
        print(f'  {metric:<18} {ratio:>5.2f}x   {METRIC_LABEL.get(metric, PANEL_METRIC_TAG.get(metric, ""))}')
    return True, None


def pick_build_cell(idx, preferred_phase, preferred_cxx):
    """(phase, cxx) for the build-time panel. Build time is per-compiler, so the panel shows
    exactly one (phase, cxx) cell -- the requested one, else the cell carrying the MOST
    kernels (ties broken by phase/cxx order), with a printed note. (None, None) when nothing
    has build timings. Use --phase / --cxx to pin a different cell."""
    cells = sorted({(p, c) for (p, c, _k) in idx}, key=lambda pc: (phase_sort_key(pc[0]), cxx_sort_key(pc[1])))
    counted = [(p, c, len(build_kernels(idx, p, c)[1])) for p, c in cells]
    counted = [(p, c, n) for p, c, n in counted if n]
    if not counted:
        return None, None
    wanted = [(p, c, n) for p, c, n in counted
              if (preferred_phase is None or p == preferred_phase) and (preferred_cxx is None or c == preferred_cxx)]
    pool = wanted or counted
    best = max(pool, key=lambda pcn: pcn[2])  # richest cell; max() keeps the first on a tie
    if not wanted:
        print(f'[build_and_quality] no build timings for phase={preferred_phase!r} cxx={preferred_cxx!r}; '
              f'using phase={best[0]!r} cxx={best[1]!r} instead')
    elif len(pool) > 1:
        others = ', '.join(f'{c} (n={n})' for _p, c, n in pool if (_p, c) != (best[0], best[1]))
        print(f'[build_and_quality] build panel shows phase={best[0]!r} cxx={best[1]!r} (n={best[2]}); '
              f'also available: {others} -- pin with --cxx/--phase')
    return best[0], best[1]


def collect_metrics(srcdir, metrics_csv):
    """{kernel: {codegen: {metric: value}}} from a precomputed CSV and/or by walking a
    directory of generated sources. Returns ({}, note) when neither is usable."""
    if METRICS_IMPORT_ERROR and (srcdir or metrics_csv):
        return {}, f'readability_metrics is not importable ({METRICS_IMPORT_ERROR})'
    out = {}
    if metrics_csv:
        if not os.path.isfile(metrics_csv):
            return {}, f'--metrics-csv {metrics_csv} not found'
        with open(metrics_csv, newline='') as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                kernel = (r.get('kernel') or '').strip()
                codegen = (r.get('codegen') or '').strip()
                if (not kernel or not codegen) and (r.get('path') or '').strip():
                    kernel, codegen = infer_kernel_codegen(r['path'].strip())
                    codegen = codegen or ''
                if not kernel or codegen not in CODEGENS:
                    continue
                vals = {m: as_float(r.get(m)) for m in PANEL_METRICS if r.get(m) is not None}
                out.setdefault(kernel, {})[codegen] = {k: v for k, v in vals.items() if v is not None}
        if not out:
            return {}, f'--metrics-csv {metrics_csv} carried no usable kernel/codegen rows'
        print(f'metrics: read {len(out)} kernel(s) from {metrics_csv}')
        return out, None

    if not os.path.isdir(srcdir):
        return {}, f'--srcdir {srcdir} is not a directory'
    sources = find_sources(srcdir)
    if not sources:
        return {}, f'--srcdir {srcdir} holds no .cpp/.cu sources'
    unlabeled, notes = 0, set()
    for path, kernel, codegen in sources:
        if codegen not in CODEGENS:
            unlabeled += 1
            continue
        score = readability_score(path)
        if not score['ok']:
            print(f'metrics: {path}: {score["error"]}')
            continue
        out.setdefault(kernel, {})[codegen] = {m: score.get(m) for m in PANEL_METRICS}
        for note in score['notes']:
            notes.add(note)
    for note in sorted(notes):
        print(f'metrics note: {note}')
    if unlabeled:
        print(f'metrics: {unlabeled} source(s) under {srcdir} carry no legacy/experimental marker -- ignored')
    if not out:
        return {}, f'--srcdir {srcdir} yielded no legacy/experimental source pairs'
    print(f'metrics: scored {len(out)} kernel(s) from {srcdir}')
    return out, None


def plot_build_and_quality(idx,
                           columns,
                           out_dir,
                           build_phase,
                           build_cxx,
                           metrics,
                           metrics_note,
                           base='build_and_quality'):
    """FIGURE B: stacked build time + LoC + readability, as one multi-panel figure."""
    panels = []

    have_build_cols = 'codegen_ms' in columns and 'compile_ms' in columns
    phase = cxx = None
    lanes_by_kernel = kernels = None
    if not have_build_cols:
        print(f'[{base}] build-time panel skipped: TSV lacks a codegen_ms and/or compile_ms column')
    else:
        phase, cxx = pick_build_cell(idx, build_phase, build_cxx)
        if phase is None:
            print(f'[{base}] build-time panel skipped: no kernel has compile/codegen timings')
        else:
            lanes_by_kernel, kernels = build_kernels(idx, phase, cxx)
            panels.append('build')

    if metrics:
        panels.extend(['loc', 'readability'])
    else:
        print(f'[{base}] LoC + readability panels skipped: {metrics_note}')

    if not panels:
        print(f'[{base}] skipped: no panel has data')
        return

    widest = max([len(kernels) if kernels else 0] + [len(metrics)])
    fig_w = max(8.0, min(34.0, widest * 0.55))
    heights = {'build': 5.2, 'loc': 5.2, 'readability': 4.4}
    #: Constrained layout, not tight_layout: the panels have very different heights and
    #: rotated tick labels, and tight_layout does not handle a gridspec + suptitle here.
    fig = plt.figure(figsize=(fig_w, sum(heights[p] for p in panels)), layout='constrained')
    gs = fig.add_gridspec(len(panels), 1, height_ratios=[heights[p] for p in panels])

    drawn = []
    for i, panel in enumerate(panels):
        ax = fig.add_subplot(gs[i, 0])
        if panel == 'build':
            draw_build_panel(ax, lanes_by_kernel, kernels,
                             f'Build time (C++ compile + DaCe codegen) -- {group_label(phase, cxx)}')
            drawn.append(f'build[{len(kernels)} kernels, {group_label(phase, cxx)}]')
        elif panel == 'loc':
            ok, note = draw_loc_panel(ax, metrics)
            if ok:
                drawn.append(f'loc[{len(metrics)} kernels]')
            else:
                ax.set_axis_off()
                print(f'[{base}] LoC panel skipped: {note}')
        else:
            ok, note = draw_readability_panel(ax, metrics)
            if ok:
                drawn.append('readability')
            else:
                ax.set_axis_off()
                print(f'[{base}] readability panel skipped: {note}')

    fig.suptitle('Build cost and generated-code quality: legacy vs experimental codegen', fontsize=12)
    png, pdf = save(fig, out_dir, base)
    print(f'wrote {png} + {pdf}  [panels: {", ".join(drawn) or "none"}]')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tsv', nargs='+', required=True, help='one or more result TSV files from the perf runners')
    ap.add_argument('--out-dir', default='plots', help='directory for the output figures (default: plots)')
    ap.add_argument('--phase',
                    help='phase for the build-time panel when several are present '
                    '(single_core | multi_core | gpu; default: multi_core if present)')
    ap.add_argument('--cxx', help='cxx group for the build-time panel when several are present (e.g. g++, clang++)')
    ap.add_argument('--srcdir', help='root of generated .cpp/.cu sources (walked) for the LoC/readability panels')
    ap.add_argument('--metrics-csv', help='precomputed metrics CSV (readability_metrics.py --csv); wins over --srcdir')
    args = ap.parse_args()

    rows, columns, stats = load_rows(args.tsv)
    print(f'read {stats["files"]} file(s): {stats["total"]} row(s), kept {stats["kept"]}, '
          f'skipped {stats["skipped_error"]} error row(s)')
    if not rows:
        print('nothing to plot (no usable rows)')
        return

    os.makedirs(args.out_dir, exist_ok=True)
    idx = index_rows(rows)
    if 'phase' not in columns:
        print('note: TSV has no phase column -- deriving single_core/multi_core from preset, assuming CPU rows')
    if 'cxx' not in columns:
        print(f'note: TSV has no cxx column -- all rows grouped as {CXX_UNKNOWN!r}')
    print(f'phases present: {phases_present(idx) or "none"}   cxx present: {cxx_present(idx) or "none"}')

    metrics, metrics_note = ({}, 'pass --srcdir or --metrics-csv to draw them')
    if args.srcdir or args.metrics_csv:
        metrics, metrics_note = collect_metrics(args.srcdir, args.metrics_csv)

    plot_runtime(idx, columns, args.out_dir)
    build_phase = args.phase or ('multi_core' if 'multi_core' in phases_present(idx) else None)
    plot_build_and_quality(idx, columns, args.out_dir, build_phase, args.cxx, metrics, metrics_note)


if __name__ == '__main__':
    main()
