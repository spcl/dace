#!/usr/bin/env python3
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Plot the codegen_buildperf_jobs sweep (run_buildperf*.py / submit_codegen_buildperf.sbatch).

Reads one or more unified TSV files whose header is (tab-separated):

    kernel  corpus  codegen  preset  threads  cxx  phase  codegen_ms  compile_ms  runtime_ms
    code_bytes  nloc  max_nesting  tokens  max_ccn  speedup  correctness  error

``codegen`` is ``legacy`` or ``experimental``; ``corpus`` is npbench|polybench|tsvc2|tsvc2_5;
``cxx`` (g++ | clang++) is the host compiler the row was built with; ``phase`` is
single_core | multi_core. The build/size/readability metrics (code_bytes, nloc, max_nesting,
tokens, max_ccn) are recorded inline per row -- so BOTH figures come from the merged TSV
alone; no separate metrics file is needed. Rows carrying a non-empty ``error`` are skipped.

Two figures (each written to --out-dir as PNG and PDF), both aggregated PER CORPUS:

  FIGURE build_and_quality   four panels comparing generated-code build cost and quality,
                             legacy vs experimental, per corpus, at ONE phase (default
                             multi_core, override with --phase):
                               (i)   DaCe codegen time (geomean ms) -- legacy vs experimental
                                     (compiler-independent, so aggregated over cxx);
                               (ii)  C++ compile time (geomean ms) -- g++ vs clang++ x
                                     legacy vs experimental (this one DOES depend on cxx);
                               (iii) generated code size (geomean nloc / LoC) -- legacy vs
                                     experimental (compiler-independent);
                               (iv)  readability -- per-corpus geomean of the experimental/
                                     legacy ratio for the two discriminators (nloc,
                                     max_nesting); <1 means experimental is smaller/flatter.

  FIGURE performance         two panels -- single-core (S) and multi-core (paper) -- of the
                             experimental-vs-legacy runtime speedup (geomean legacy/
                             experimental per corpus), g++ vs clang++. Absolute runtimes span
                             orders of magnitude across kernels, so the ratio is the readable
                             axis (same convention the sibling harness uses); a dashed parity
                             line at 1.0 flags any regression, since readability is meant to be
                             perf-neutral.

Missing columns / corpora / phases never crash a run -- a panel with no data is annotated and
left blank, and a note is printed.

    python3 plot_buildperf.py --tsv perfresults.tsv --out-dir plots/
    python3 plot_buildperf.py --tsv perfresults.tsv --out-dir plots/ --phase single_core
"""
import argparse
import csv
import math
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

CODEGENS = ('legacy', 'experimental')
#: Corpus x-axis order; anything else is appended in first-seen order after these.
CORPUS_ORDER = ('npbench', 'polybench', 'tsvc2', 'tsvc2_5')
#: phase -> display label / sort order.
PHASE_LABEL = {'single_core': 'single core (S preset)', 'multi_core': 'multi-core (paper preset)'}
PHASE_ORDER = ('single_core', 'multi_core')
#: Stand-in group name when the TSV has no cxx value.
CXX_UNKNOWN = 'cxx n/a'

#: Lane hues (legacy vs experimental) + the cxx x lane series, from the dataviz skill's
#: validated categorical palette.
LANE_COLOR = {'legacy': '#2a78d6', 'experimental': '#e07a2b'}
CXX_LANE_COLOR = {
    ('g++', 'legacy'): '#2a78d6',
    ('g++', 'experimental'): '#7fb0e6',
    ('clang++', 'legacy'): '#e07a2b',
    ('clang++', 'experimental'): '#f0b787',
}
FASTER_COLOR = '#1baf7a'   # experimental at least as good (faster / smaller / flatter)
SLOWER_COLOR = '#d9534f'   # experimental regressed
NEUTRAL_COLOR = '#8a8a86'
#: The two size-independent readability discriminators shown in panel (iv).
READABILITY_METRICS = ('nloc', 'max_nesting')


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


def load_rows(paths):
    """Read every TSV, dropping error rows. Returns (rows, columns, stats)."""
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
    """single_core | multi_core, from the phase column, else derived from preset."""
    phase = (row.get('phase') or '').strip().lower()
    if phase:
        return phase
    preset = (row.get('preset') or '').strip()
    return {'S': 'single_core', 'paper': 'multi_core'}.get(preset, preset.lower() or 'unknown')


def row_cxx(row):
    return (row.get('cxx') or '').strip() or CXX_UNKNOWN


def row_corpus(row):
    return (row.get('corpus') or '').strip()


def corpora_present(rows):
    seen = {row_corpus(r) for r in rows if row_corpus(r)}
    ordered = [c for c in CORPUS_ORDER if c in seen]
    ordered += sorted(c for c in seen if c not in CORPUS_ORDER)
    return ordered


def cxx_present(rows, phase=None):
    seen = {row_cxx(r) for r in rows if phase is None or row_phase(r) == phase}
    return sorted(seen, key=lambda c: (c == CXX_UNKNOWN, c))


def phases_present(rows):
    seen = {row_phase(r) for r in rows if row_phase(r)}
    ordered = [p for p in PHASE_ORDER if p in seen]
    ordered += sorted(p for p in seen if p not in PHASE_ORDER)
    return ordered


def index_by_kernel(rows, phase, cxx=None):
    """{(corpus, kernel): {codegen: row}} at `phase` (and `cxx` if given). A later duplicate
    overwrites -- rank-split TSVs cover disjoint kernels, and cxx-independent metrics (nloc,
    codegen_ms) agree across cxx, so which duplicate wins does not matter for those."""
    idx = {}
    for r in rows:
        if row_phase(r) != phase:
            continue
        if cxx is not None and row_cxx(r) != cxx:
            continue
        key = (row_corpus(r), (r.get('kernel') or '').strip())
        idx.setdefault(key, {})[(r.get('codegen') or '').strip()] = r
    return idx


def save(fig, out_dir, base):
    png = os.path.join(out_dir, base + '.png')
    pdf = os.path.join(out_dir, base + '.pdf')
    fig.savefig(png, dpi=150, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    return png, pdf


# --------------------------------------------------------------------------
# Grouped-bar primitive: x = corpus, one bar per series.
# --------------------------------------------------------------------------
def grouped_bars(ax, corpora, series, value_of, color_of, hatch_of=None):
    """Draw grouped bars. `series` is the ordered series labels; value_of(series, corpus)
    returns a float or None (a None/absent value simply leaves a gap). Returns the tallest
    drawn value (for y-limit headroom), or 0.0 if nothing was drawn."""
    n = max(1, len(series))
    group_w = 0.82
    bar_w = group_w / n
    tallest = 0.0
    for j, s in enumerate(series):
        offset = (j - (n - 1) / 2.0) * bar_w
        xs, ys = [], []
        for i, corpus in enumerate(corpora):
            v = value_of(s, corpus)
            if v is None or not math.isfinite(v):
                continue
            xs.append(i + offset)
            ys.append(v)
            tallest = max(tallest, v)
        ax.bar(xs,
               ys,
               bar_w,
               color=color_of(s),
               hatch=(hatch_of(s) if hatch_of else ''),
               edgecolor='white',
               linewidth=0.5,
               label=str(s))
    ax.set_xticks(range(len(corpora)))
    ax.set_xticklabels(corpora, fontsize=9)
    ax.margins(x=0.02)
    return tallest


# --------------------------------------------------------------------------
# Figure build_and_quality.
# --------------------------------------------------------------------------
def geomean_metric(rows, corpus, codegen, phase, field, cxx=None):
    """Geomean of `field` over every kernel row in (corpus, codegen, phase[, cxx])."""
    vals = []
    for r in rows:
        if row_corpus(r) != corpus or (r.get('codegen') or '').strip() != codegen or row_phase(r) != phase:
            continue
        if cxx is not None and row_cxx(r) != cxx:
            continue
        v = as_float(r.get(field))
        if v is not None:
            vals.append(v)
    return geomean(vals)


def readability_ratio(idx, corpus, metric):
    """Per-corpus geomean of experimental/legacy for `metric` (paired per kernel). <1 means
    experimental generates smaller/flatter code. None when no kernel has both lanes."""
    ratios = []
    for (c, _kernel), lanes in idx.items():
        if c != corpus:
            continue
        leg = lanes.get('legacy')
        exp = lanes.get('experimental')
        if not leg or not exp:
            continue
        lv = as_float(leg.get(metric))
        ev = as_float(exp.get(metric))
        if lv and ev and lv > 0 and ev > 0:
            ratios.append(ev / lv)
    return geomean(ratios)


def draw_codegen_panel(ax, rows, corpora, phase):
    tallest = grouped_bars(ax, corpora, CODEGENS,
                           lambda cg, c: geomean_metric(rows, c, cg, phase, 'codegen_ms'),
                           lambda cg: LANE_COLOR[cg])
    ax.set_ylabel('DaCe codegen time (ms, geomean)', fontsize=9)
    ax.set_title('(i) SDFG -> C++ codegen time -- legacy vs experimental (compiler-independent)', fontsize=10)
    if tallest > 0:
        ax.set_ylim(0, tallest * 1.2)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)


def draw_compile_panel(ax, rows, corpora, phase, cxxs):
    series = [(cx, cg) for cx in cxxs for cg in CODEGENS]
    color_of = lambda s: CXX_LANE_COLOR.get(s, LANE_COLOR[s[1]])
    tallest = grouped_bars(ax, corpora, series,
                           lambda s, c: geomean_metric(rows, c, s[1], phase, 'compile_ms', cxx=s[0]), color_of)
    ax.set_ylabel('C++ compile time (ms, geomean)', fontsize=9)
    ax.set_title('(ii) C++ -> binary compile time -- g++ vs clang++, legacy vs experimental', fontsize=10)
    if tallest > 0:
        ax.set_ylim(0, tallest * 1.2)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [f'{cx}, {cg}' for (cx, cg) in series], loc='upper right', fontsize=7, framealpha=0.9)


def draw_size_panel(ax, rows, corpora, phase):
    tallest = grouped_bars(ax, corpora, CODEGENS,
                           lambda cg, c: geomean_metric(rows, c, cg, phase, 'nloc'),
                           lambda cg: LANE_COLOR[cg])
    ax.set_ylabel('generated LoC (nloc, geomean)', fontsize=9)
    ax.set_title('(iii) Generated code size -- legacy vs experimental (compiler-independent)', fontsize=10)
    if tallest > 0:
        ax.set_ylim(0, tallest * 1.2)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)


def draw_readability_panel(ax, idx, corpora):
    ratios = {(m, c): readability_ratio(idx, c, m) for m in READABILITY_METRICS for c in corpora}

    def color_of(metric):
        return lambda c: (FASTER_COLOR if (ratios[(metric, c)] or 1.0) <= 1.0 else SLOWER_COLOR)

    # One series per metric; color per bar reflects improvement/regression, so pass a
    # per-corpus color via a metric-keyed closure through a light wrapper.
    n = len(READABILITY_METRICS)
    group_w = 0.82
    bar_w = group_w / max(1, n)
    tallest = 0.0
    for j, metric in enumerate(READABILITY_METRICS):
        offset = (j - (n - 1) / 2.0) * bar_w
        xs, ys, cols = [], [], []
        for i, c in enumerate(corpora):
            v = ratios[(metric, c)]
            if v is None:
                continue
            xs.append(i + offset)
            ys.append(v)
            cols.append(FASTER_COLOR if v <= 1.0 else SLOWER_COLOR)
            tallest = max(tallest, v)
        ax.bar(xs, ys, bar_w, color=cols, edgecolor='white', linewidth=0.5,
               hatch='' if metric == 'nloc' else '///', label=metric)
        for x, y in zip(xs, ys):
            ax.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=7)
    ax.axhline(1.0, color=NEUTRAL_COLOR, linewidth=1, linestyle='--', label='parity (legacy = experimental)')
    ax.set_xticks(range(len(corpora)))
    ax.set_xticklabels(corpora, fontsize=9)
    ax.set_ylabel('experimental / legacy (geomean)\n<1 = experimental more readable', fontsize=9)
    ax.set_title('(iv) Readability discriminators -- nloc (size) & max_nesting (shape), per corpus', fontsize=10)
    ax.set_ylim(0, max(1.35, tallest * 1.3))
    ax.margins(x=0.02)
    ax.legend(loc='upper right', fontsize=7, framealpha=0.9)


def plot_build_and_quality(rows, columns, out_dir, phase):
    corpora = corpora_present(rows)
    if not corpora:
        print('[build_and_quality] skipped: no rows to plot')
        return
    cxxs = [c for c in cxx_present(rows, phase) if c != CXX_UNKNOWN] or cxx_present(rows, phase)
    idx = index_by_kernel(rows, phase)

    fig = plt.figure(figsize=(max(8.0, 2.2 * len(corpora) + 4.0), 18.5), layout='constrained')
    gs = fig.add_gridspec(4, 1)
    draw_codegen_panel(fig.add_subplot(gs[0, 0]), rows, corpora, phase)
    draw_compile_panel(fig.add_subplot(gs[1, 0]), rows, corpora, phase, cxxs)
    draw_size_panel(fig.add_subplot(gs[2, 0]), rows, corpora, phase)
    draw_readability_panel(fig.add_subplot(gs[3, 0]), idx, corpora)
    fig.suptitle(f'Build cost and generated-code quality: legacy vs experimental codegen '
                 f'-- {PHASE_LABEL.get(phase, phase)}', fontsize=13)
    png, pdf = save(fig, out_dir, 'build_and_quality')
    print(f'wrote {png} + {pdf}  [phase={phase}, corpora={corpora}, cxx={cxxs}]')


# --------------------------------------------------------------------------
# Figure performance.
# --------------------------------------------------------------------------
def speedup_geomean(rows, corpus, cxx, phase):
    """Per-corpus geomean of legacy_runtime / experimental_runtime (paired per kernel).
    >1 means experimental is faster. Falls back to the experimental row's own speedup
    column when a legacy runtime is not in this TSV."""
    idx = index_by_kernel(rows, phase, cxx)
    ratios = []
    for (c, _kernel), lanes in idx.items():
        if c != corpus:
            continue
        exp = lanes.get('experimental')
        if not exp:
            continue
        leg = lanes.get('legacy')
        lr = as_float(leg.get('runtime_ms')) if leg else None
        er = as_float(exp.get('runtime_ms'))
        if lr and er and lr > 0 and er > 0:
            ratios.append(lr / er)
        else:
            s = as_float(exp.get('speedup'))
            if s and s > 0:
                ratios.append(s)
    return geomean(ratios)


def cxx_color(cx):
    """A stable hue per host compiler. 'clang' is checked first because 'clang++' contains
    'g++' as a substring -- keying on 'g++' would colour clang++ like g++."""
    return LANE_COLOR['experimental'] if 'clang' in cx else LANE_COLOR['legacy']


def draw_perf_panel(ax, rows, corpora, phase, cxxs):
    tallest = grouped_bars(ax, corpora, cxxs, lambda cx, c: speedup_geomean(rows, c, cx, phase), cxx_color)
    # Annotate each bar with its geomean value.
    for cont in ax.containers:
        ax.bar_label(cont, fmt='%.2f', fontsize=7, padding=2)
    ax.axhline(1.0, color=NEUTRAL_COLOR, linewidth=1, linestyle='--', label='parity (legacy = experimental)')
    ax.set_ylabel('speedup = legacy / experimental\n(>1 = experimental faster; ~1 = perf-neutral)', fontsize=9)
    ax.set_title(PHASE_LABEL.get(phase, phase), fontsize=10)
    ax.set_ylim(0, max(1.35, (tallest or 1.0) * 1.3))
    ax.legend(loc='upper right', fontsize=7, framealpha=0.9)


def plot_performance(rows, columns, out_dir):
    if 'runtime_ms' not in columns and 'speedup' not in columns:
        print('[performance] skipped: TSV lacks both runtime_ms and speedup columns')
        return
    corpora = corpora_present(rows)
    phases = [p for p in phases_present(rows) if p in PHASE_ORDER] or phases_present(rows)
    if not corpora or not phases:
        print('[performance] skipped: no rows to plot')
        return
    fig, axes = plt.subplots(len(phases), 1, figsize=(max(8.0, 2.2 * len(corpora) + 4.0), 5.0 * len(phases)),
                             squeeze=False)
    for ax, phase in zip(axes[:, 0], phases):
        cxxs = [c for c in cxx_present(rows, phase) if c != CXX_UNKNOWN] or cxx_present(rows, phase)
        draw_perf_panel(ax, rows, corpora, phase, cxxs)
    fig.suptitle('Runtime: legacy vs experimental codegen (geomean speedup per corpus)', fontsize=13, y=1.0)
    fig.tight_layout()
    png, pdf = save(fig, out_dir, 'performance')
    print(f'wrote {png} + {pdf}  [phases={phases}, corpora={corpora}]')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tsv', nargs='+', required=True, help='one or more unified TSV files from the perf runners')
    ap.add_argument('--out-dir', default='plots', help='directory for the output figures (default: plots)')
    ap.add_argument('--phase',
                    help='phase for the build/quality figure (single_core | multi_core; '
                    'default: multi_core if present, else the first phase seen)')
    args = ap.parse_args()

    rows, columns, stats = load_rows(args.tsv)
    print(f'read {stats["files"]} file(s): {stats["total"]} row(s), kept {stats["kept"]}, '
          f'skipped {stats["skipped_error"]} error row(s)')
    if not rows:
        print('nothing to plot (no usable rows)')
        return

    os.makedirs(args.out_dir, exist_ok=True)
    phases = phases_present(rows)
    print(f'corpora: {corpora_present(rows) or "none"}   phases: {phases or "none"}   '
          f'cxx: {cxx_present(rows) or "none"}')

    build_phase = args.phase or ('multi_core' if 'multi_core' in phases else (phases[0] if phases else 'multi_core'))
    plot_build_and_quality(rows, columns, args.out_dir, build_phase)
    plot_performance(rows, columns, args.out_dir)


if __name__ == '__main__':
    main()
