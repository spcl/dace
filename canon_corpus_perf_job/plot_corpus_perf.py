#!/usr/bin/env python3
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Plot and tabulate what the corpus perf job already measured. READ-ONLY: nothing is timed here.

Input is the per-kernel JSON the timing driver writes -- ``<out>/perf_json/<suite>_<kernel>.json``
for a job run, ``tests/passes/canonicalize/perf_results/`` for a bare driver run. This script never
builds, compiles or re-times a kernel: a plot that could trigger a measurement would silently mix a
fresh number into an old sweep, and the two would no longer be the same machine state.

    python plot_corpus_perf.py                       # the job's default results directory
    python canon_corpus_perf_job/plot_corpus_perf.py --results DIR --suite tsvc --arm dace-canon-gcc

TWO figures, never one. The corpora split into two groups with DIFFERENT denominators:

* ``np`` + ``poly`` divide by the timed numpy reference. That reference is PARALLEL -- numpy
  dispatches gemm/2mm/3mm/syrk/cholesky into a threaded OpenBLAS -- so the claim is "x times the
  parallel numpy baseline".
* ``tsvc`` + ``tsvc25`` divide by the ``seq-cpp`` arm: single-threaded C++ from the same
  post-simplify SDFG. The claim is "x times optimized SEQUENTIAL C++".

3x over threaded numpy and 3x over single-core C++ are different claims. One shared axis invites a
comparison that is not valid, so each group gets its own image file, its own geomean table and a
caption naming its denominator in words. There is deliberately NO combined geomean anywhere.

Other rules the figures obey:

* y is the SIGNED symmetric speedup ``s``: 0 parity, +1 = 2x faster, -1 = 2x slower. Every tick
  also carries its raw ratio (``+2 (3.00x)``) so ``+2`` cannot be read as "2x".
* Every geomean -- legend and table alike -- is taken over the RAW ratio, then converted for
  display. The signed scale spans zero and negatives, where a geometric mean is undefined.
* A ``python-scalar`` reference (the tsvc oracles, and polybench kernels whose numpy form is a
  scalar loop, e.g. seidel_2d) is NEVER a denominator; such kernels are dropped and counted.
* Arms whose value check failed, arms that errored, and kernels with no verifiable denominator are
  DROPPED and counted in the report. A missing bar is a fact; a substituted 1.0 would be a lie.
* Result files written by the pre-six-arm labels are reported as STALE, not mixed in.

It deliberately does not import the timing harness: that module pulls in dace and the corpora, and
probes every compiler arm at import time when ``CANON_PERF_ARMS`` is set. Reading JSON must not
depend on a working toolchain. The corpus rules below therefore mirror the harness's ``DENOMINATOR``
/ ``REFERENCE_KIND`` tables, and are the only thing that has to stay in step with it.
"""
import argparse
import json
import math
import statistics
import sys
import textwrap
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use('Agg')  # a cluster node has no display; MUST happen before pyplot is imported

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter

#: Where the job writes its results, i.e. ``corpus_perf_job.DEFAULT_OUT`` -- outside the repo,
#: because a sweep produces one JSON per kernel and the job folder holds only its three scripts.
#: Not imported from there: this script must stay readable on a box with no dace and no corpora.
DEFAULT_RESULTS = Path.home() / '.cache' / 'dace-corpus-perf-results'
#: Corpus tags in figure order.
SUITES = ('poly', 'np', 'tsvc', 'tsvc25')
SUITE_TITLE = {'poly': 'polybench', 'np': 'npbench', 'tsvc': 'tsvc', 'tsvc25': 'tsvc_2_5'}
#: Kernels each corpus contributes (``tests/corpus/corpus_suite.py``), used ONLY to report how much
#: of the sweep is on disk. Hardcoded rather than imported so the plot needs no dace import.
CORPUS_SIZE = {'poly': 30, 'np': 24, 'tsvc': 151, 'tsvc25': 72}
#: The speedup denominator per corpus; mirrors ``DENOMINATOR`` in the timing harness. ``reference``
#: is the timed corpus oracle, ``seq-cpp`` the sequential-C++ arm built from the same SDFG.
CORPUS_DENOMINATOR = {'poly': 'reference', 'np': 'reference', 'tsvc': 'seq-cpp', 'tsvc25': 'seq-cpp'}
#: Reference kinds that may never be a timing denominator. The tsvc oracles are scalar Python loops,
#: and so is polybench's seidel_2d; dividing by CPython would yield speedups in the hundreds and
#: silently invalidate the figure.
REFUSED_KINDS = ('python-scalar', 'unknown', '')
#: How each denominator kind reads in a panel title -- the claim the panel's numbers make.
DENOM_PHRASE = {
    'numpy': 'the timed parallel numpy reference',
    'dace-untransformed': 'the timed untransformed-SDFG reference',
    'sequential-c++': 'seq-cpp (sequential C++, same SDFG)',
}
#: Arm plotting order: the six comparison arms, then the sequential denominator arm, then the
#: labels older sweeps on disk used. Unknown labels are appended alphabetically, so a renamed or
#: added arm still plots -- the series come from the DATA, this list only fixes order and colour.
ARM_ORDER = ('dace-autoopt-gcc', 'dace-autoopt-llvm', 'dace-canon-gcc', 'dace-canon-llvm', 'dace-simplify+gcc-autopar',
             'dace-simplify+llvm-autopar', 'seq-cpp', 'auto-opt', 'auto-opt+llvm', 'canon', 'canon+gcc', 'canon+llvm',
             'gcc-autopar', 'llvm-autopar', 'canon-serial+gcc-autopar', 'canon-serial+llvm-autopar', 'gcc-graphite',
             'clang-polly')
#: The arm labels the current job writes. A record carrying none of them predates the six-arm table:
#: its numbers were taken under different flags and are reported as stale rather than plotted next
#: to fresh ones.
CURRENT_ARMS = ARM_ORDER[:7]
#: The harness's own "too fast to time reliably" floor. Not applied by default (dropping data is the
#: reader's call, not the plotter's) but kernels below it are counted, since a ratio between two
#: sub-millisecond timings is per-call overhead, not a speedup.
MIN_TIMEABLE_MS = 0.5
TAB10 = matplotlib.colormaps['tab10']
PALETTE = tuple(TAB10(i) for i in range(TAB10.N))
LINESTYLES = ('-', '--', ':', '-.')
MARKERS = ('o', 's', '^', 'D', 'v', 'P', 'X', '*')
#: Above this many kernels a panel drops the connecting lines (six polylines over 151 kernels is
#: spaghetti) and its tick labels; colour + marker shape still separate the arms, and each arm's
#: geomean is drawn as a thin horizontal line so the panel still reads at 151x6 points.
DENSE_PANEL = 40

#: Exclusion categories, so the report can say WHY in three numbers before it says it in prose.
CAT_PY_SCALAR = 'python-scalar reference'
CAT_NO_DENOM = 'denominator not measured'
CAT_FOREIGN = "denominator belongs to the other figure"
CAT_STALE = 'pre-six-arm result file'
CAT_MISCOMPILE = 'value check FAILED'
CAT_UNTIMED = 'arm not timed'
CAT_TOO_FAST = 'denominator below --min-ms'
CAT_UNGROUPED = 'corpus has no denominator group'


class Group(NamedTuple):
    """One figure: the corpora that share ONE denominator, and what that denominator is in words."""
    slug: str  # goes into the file name, so the two images are told apart on disk
    suites: tuple[str, ...]
    kind: str  # the only denominator kind this figure may draw; anything else is excluded
    title: str
    caption: str


#: The two figures. A corpus appears in exactly one of them, and no number crosses between them.
GROUPS = (
    Group(
        'numpy-reference', ('np', 'poly'), 'numpy', 'npbench + polybench, speedup over the timed parallel numpy reference',
        'Denominator: each kernel\'s own numpy reference, timed like any other arm at the machine default thread '
        'count. numpy dispatches gemm / 2mm / 3mm / syrk / cholesky into a THREADED OpenBLAS, so this axis is a '
        'comparison against parallel numpy, not against one core. Kernels whose numpy form is a scalar Python loop '
        '(seidel_2d) carry a python-scalar reference, are never divided by it, and are absent by construction. '
        'These numbers are NOT comparable with the seq-cpp figure: the two divide by different things.'),
    Group(
        'seq-cpp', ('tsvc', 'tsvc25'), 'sequential-c++', 'tsvc + tsvc_2_5, speedup over sequential C++ (the seq-cpp arm)',
        'Denominator: the seq-cpp arm -- single-threaded C++ generated from the same post-simplify SDFG, -O3, no '
        'autopar, no polyhedral flags, timed on the same rank as every other arm of the same kernel. The tsvc and '
        'tsvc_2_5 python oracles are scalar loops used for the value check only; they are never timed and never '
        'divided by. These numbers are NOT comparable with the numpy-reference figure: the baseline here is one '
        'core, there it is threaded numpy.'),
)


class Point(NamedTuple):
    """One plotted number: this arm on this kernel, against this kernel's own denominator."""
    suite: str
    kernel: str
    arm: str
    kind: str  # what the ratio divides by; part of the facet key, never pooled across kinds
    ratio: float  # RAW denominator_min_ms / arm_min_ms; > 1 means the arm is faster
    den_ms: float


class Excluded(NamedTuple):
    """One thing deliberately not drawn, and the reason it is not drawn."""
    suite: str
    kernel: str
    arm: str
    category: str
    why: str


class Denominator(NamedTuple):
    """A verified speedup denominator: the time, what it is, and the arm label it came from."""
    min_ms: float
    kind: str
    label: str  # the arm that IS the denominator (it is the axis, so it is not drawn as a series)


class Report(NamedTuple):
    """Everything the figures and the tables are rendered from."""
    preset: str
    mode: str
    sources: list[Path]
    kernels_on_disk: dict[str, int]
    points: list[Point]
    excluded: list[Excluded]
    notes: list[str]


def geomean(ratios: Iterable[float]) -> tuple[float, int]:
    """Geometric mean of RAW ratios, and the count actually behind it.

    Non-finite and non-positive entries contribute nothing at all -- substituting 1.0 for a kernel
    that errored would drag the mean toward parity and inflate ``n`` with numbers nobody measured.
    """
    good = [r for r in ratios if math.isfinite(r) and r > 0.0]
    if not good:
        return float('nan'), 0
    return math.exp(sum(math.log(r) for r in good) / len(good)), len(good)


def signed(ratio: float) -> float:
    """Raw ratio -> the signed symmetric display scale: parity 0, 2x faster +1, 2x slower -1."""
    if not (math.isfinite(ratio) and ratio > 0.0):
        return float('nan')
    return ratio - 1.0 if ratio >= 1.0 else -(1.0 / ratio - 1.0)


def ratio_of(value: float) -> float:
    """Inverse of :func:`signed` -- the raw ratio a signed-scale position means."""
    return 1.0 + value if value >= 0.0 else 1.0 / (1.0 - value)


def tick_label(value: float, _pos: float = 0.0) -> str:
    """A y tick carrying BOTH scales, so ``+2`` can never be read as "2x".

    Enough decimals for the tick to be distinguishable at whatever range the panel spans: a corpus
    that lands inside +-0.05 would otherwise print a column of ``+0.0``.
    """
    if abs(value) < 1e-9:
        return '0 (parity)'
    magnitude = abs(value)
    if magnitude < 0.1:
        signed_s = f'{value:+.3f}'
    elif magnitude < 1.0:
        signed_s = f'{value:+.2f}'
    elif magnitude < 10.0:
        signed_s = f'{value:+.1f}'
    else:
        signed_s = f'{value:+.0f}'
    return f'{signed_s} ({ratio_of(value):.3g}x)'


def result_files(roots: list[Path]) -> list[Path]:
    """Every per-kernel result JSON under ``roots``; a ``perf_json/`` subdir wins over the root.

    Accepts a job ``--out`` directory, a raw results directory, or single files, so the same command
    reads a cluster run and a laptop run. ``*.json.tmp`` (a half-written file from a killed sweep)
    does not match and is therefore never read.
    """
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
            continue
        if not root.is_dir():
            continue
        found = sorted((root / 'perf_json').glob('*.json')) or sorted(root.glob('*.json'))
        files += found
    seen: dict[Path, None] = {}
    for path in files:
        seen.setdefault(path.resolve(), None)
    return list(seen)


def load_records(files: list[Path]) -> tuple[dict[tuple[str, str], dict], list[str]]:
    """``(suite, kernel) -> newest result record``, plus notes on anything unreadable or duplicated.

    A killed sweep leaves partial output, and two result roots can carry the same kernel; the newer
    ``timestamp`` wins and the collision is reported rather than silently resolved.
    """
    records: dict[tuple[str, str], dict] = {}
    notes: list[str] = []
    duplicates: list[str] = []
    for path in files:
        try:
            with open(path) as f:
                record = json.load(f)
        except (OSError, ValueError) as e:
            notes.append(f'unreadable {path}: {type(e).__name__}')
            continue
        suite, kernel = record.get('suite'), record.get('kernel')
        if not isinstance(suite, str) or not isinstance(kernel, str):
            notes.append(f'{path}: no suite/kernel field, not a result file')
            continue
        old = records.get((suite, kernel))
        if old is not None:
            duplicates.append(f'{suite}:{kernel}')
            if str(old.get('timestamp', '')) >= str(record.get('timestamp', '')):
                continue
        records[(suite, kernel)] = record
    if duplicates:
        shown = ' '.join(duplicates[:6])
        more = f' ... and {len(duplicates) - 6} more' if len(duplicates) > 6 else ''
        notes.append(f'{len(duplicates)} kernels are in more than one results root; kept the newest '
                     f'timestamp of each ({shown}{more})')
    return records, notes


def preset_of(record: dict, preset: str) -> dict:
    """The record's entry for one preset, or an empty dict; a partial sweep has records without it."""
    return (record.get('presets') or {}).get(preset) or {}


def stale_record(record: dict, preset: str) -> bool:
    """True if this record predates the six-arm table, i.e. carries none of the current arm labels.

    Such a sweep was taken under different flags and a different arm set. Its numbers are real, but
    putting them on the same axis as fresh ones would compare two machines states, so they are
    reported instead. ``--denominator baseline`` is the mode that can still read them.
    """
    pipelines = preset_of(record, preset).get('pipelines') or {}
    return bool(pipelines) and not any(arm in CURRENT_ARMS for arm in pipelines)


def reference_kind(record: dict, preset: str) -> str:
    """What this kernel's own reference IS, per the harness's ``reference_kind()``, or ``''``."""
    pres = preset_of(record, preset)
    ref = pres.get('reference') or {}
    recorded = pres.get('denominator') or {}
    if str(recorded.get('source') or 'reference') != 'reference':
        return str(ref.get('kind') or '')
    return str(ref.get('kind') or recorded.get('kind') or '')


def denominator(record: dict, preset: str, mode: str) -> tuple[Denominator | None, str, str]:
    """The verified denominator for one kernel/preset, or ``(None, category, why not)``.

    ``mode='corpus'`` is the paper rule: the timed corpus reference for polybench/npbench, the
    ``seq-cpp`` arm for tsvc/tsvc_2_5. ``mode='baseline'`` divides by the record's own ``baseline``
    arm instead -- the only thing a pre-arms sweep on disk can support, and a different claim, which
    is why it is a separate mode and ends up in the figure caption.
    """
    pres = preset_of(record, preset)
    pipes = pres.get('pipelines') or {}
    if mode == 'baseline':
        label = str(record.get('baseline') or '')
        entry = pipes.get(label) or {}
        if entry.get('correct') is True and (entry.get('min_ms') or 0.0) > 0.0:
            return Denominator(float(entry['min_ms']), f'arm {label}', label), '', ''
        return None, CAT_NO_DENOM, f'baseline arm {label!r} was not measured'
    source = CORPUS_DENOMINATOR.get(str(record.get('suite')), 'reference')
    if source == 'reference':
        kind = reference_kind(record, preset)
        # Kind first: a python-scalar oracle is refused whatever it timed at, and that is a
        # different fact from "the reference was never run".
        if kind in REFUSED_KINDS:
            return None, CAT_PY_SCALAR, f'reference kind {kind or "missing"!r} is not a valid timing denominator'
        min_ms = (pres.get('reference') or {}).get('min_ms') or (pres.get('denominator') or {}).get('min_ms')
        if not min_ms:
            return None, CAT_NO_DENOM, 'no timed corpus reference in the result file'
        return Denominator(float(min_ms), kind, ''), '', ''
    entry = pipes.get(source) or {}
    if entry.get('correct') is True and (entry.get('min_ms') or 0.0) > 0.0:
        return Denominator(float(entry['min_ms']), 'sequential-c++', source), '', ''
    return None, CAT_NO_DENOM, f'{source} arm (this corpus denominator) was not measured'


def group_of(suite: str) -> Group | None:
    """The figure a corpus belongs to. A corpus in no group is reported, never folded into one."""
    for group in GROUPS:
        if suite in group.suites:
            return group
    return None


def collect(records: dict[tuple[str, str], dict], preset: str, mode: str, suites: list[str], arms: list[str],
            min_ms: float) -> tuple[list[Point], list[Excluded]]:
    """Turn result records into plottable points, and everything else into a stated exclusion."""
    points: list[Point] = []
    excluded: list[Excluded] = []
    for (suite, kernel), record in sorted(records.items()):
        if suites and suite not in suites:
            continue
        group = group_of(suite)
        if group is None:
            excluded.append(
                Excluded(suite, kernel, '', CAT_UNGROUPED, 'corpus ' + repr(suite) + ' is in neither denominator group'))
            continue
        if mode == 'corpus' and stale_record(record, preset):
            arm_names = ' '.join(sorted((preset_of(record, preset).get('pipelines') or {})))
            excluded.append(
                Excluded(suite, kernel, '', CAT_STALE, f'stale result file: arms {arm_names} predate the six-arm table'))
            continue
        den, category, why = denominator(record, preset, mode)
        if den is None:
            excluded.append(Excluded(suite, kernel, '', category, why))
            continue
        if mode == 'corpus' and den.kind != group.kind:
            excluded.append(
                Excluded(suite, kernel, '', CAT_FOREIGN,
                         f'denominator kind {den.kind!r} is not the {group.kind!r} this figure divides by'))
            continue
        if den.min_ms < min_ms:
            excluded.append(
                Excluded(suite, kernel, '', CAT_TOO_FAST, f'denominator {den.min_ms:.4f} ms below --min-ms'))
            continue
        pipelines = preset_of(record, preset).get('pipelines') or {}
        for arm, entry in pipelines.items():
            if arms and arm not in arms:
                continue
            if arm == den.label:
                continue  # the denominator is the axis of its own panel, not a flat series on it
            if entry.get('correct') is False:
                excluded.append(Excluded(suite, kernel, arm, CAT_MISCOMPILE, 'value check FAILED (miscompile)'))
                continue
            value = entry.get('min_ms')
            if not value or value <= 0.0:
                excluded.append(
                    Excluded(suite, kernel, arm, CAT_UNTIMED, f"not timed: {entry.get('error') or 'no min_ms'}"))
                continue
            points.append(Point(suite, kernel, arm, den.kind, den.min_ms / float(value), den.min_ms))
    return points, excluded


def order_arms(labels: Iterable[str]) -> list[str]:
    """Arms in plotting order: the known table first, then anything new, alphabetically."""
    present = set(labels)
    known = [arm for arm in ARM_ORDER if arm in present]
    return known + sorted(present - set(known))


def facet_keys(points: list[Point]) -> list[tuple[str, str]]:
    """``(suite, denominator kind)`` panels in corpus order. A suite measured against two different
    denominators gets two panels rather than one mixed axis."""
    keys = {(p.suite, p.kind) for p in points}
    return sorted(keys, key=lambda k: (SUITES.index(k[0]) if k[0] in SUITES else len(SUITES), k[0], k[1]))


def short_path(path: Path, keep: int = 3) -> str:
    """A results root, short enough for a figure title; the full path stays in the table."""
    parts = path.parts
    return str(path) if len(parts) <= keep else '.../' + '/'.join(parts[-keep:])


def short_name(kernel: str, keep: int = 26) -> str:
    """A tick-sized kernel name. tsvc_2_5 kernels carry their module path
    (``tests_corpus_tsvc_2_5_tsvc_2_5_ext_gather_load``); the TAIL is what distinguishes them."""
    return kernel if len(kernel) <= keep else '...' + kernel[-keep:]


def median_signed(kernel: str, by_arm: dict[str, dict[str, float]]) -> float:
    """Sort key for a kernel: the median signed speedup over the arms that measured it."""
    vals = [signed(per_kernel[kernel]) for per_kernel in by_arm.values() if kernel in per_kernel]
    good = [v for v in vals if math.isfinite(v)]
    return statistics.median(good) if good else float('-inf')


def draw_facet(ax: Axes, suite: str, kind: str, points: list[Point], arms: list[str], sort_by: str,
               symlog: bool) -> None:
    """One corpus panel: every arm as a series over that corpus's kernels, one shared denominator."""
    by_arm: dict[str, dict[str, float]] = {}
    for p in points:
        by_arm.setdefault(p.arm, {})[p.kernel] = p.ratio
    kernels = sorted({p.kernel for p in points})
    if sort_by == 'speedup':
        kernels.sort(key=lambda k: median_signed(k, by_arm))
    xs = list(range(len(kernels)))
    dense = len(kernels) > DENSE_PANEL
    for arm in arms:
        per_kernel = by_arm.get(arm)
        if not per_kernel:
            continue
        index = ARM_ORDER.index(arm) if arm in ARM_ORDER else len(ARM_ORDER) + arms.index(arm)
        colour = PALETTE[index % len(PALETTE)]
        ys = [signed(per_kernel[k]) if k in per_kernel else float('nan') for k in kernels]
        gm, n = geomean(per_kernel.values())
        gm_s = f'{gm:.2f}x' if math.isfinite(gm) else '--'
        ax.plot(xs,
                ys,
                marker=MARKERS[index % len(MARKERS)],
                markersize=3.2,
                linewidth=0.0 if dense else 0.9,
                linestyle='none' if dense else LINESTYLES[(index // len(PALETTE)) % len(LINESTYLES)],
                alpha=0.85,
                color=colour,
                label=f'{arm}   geomean {gm_s} (n={n})')
        if math.isfinite(gm):
            # The geomean as a thin rule: at 151 kernels x 6 arms the cloud is the distribution and
            # this line is the number the table reports, on the same axis.
            ax.axhline(signed(gm), color=colour, linewidth=0.8, alpha=0.45, zorder=0)
    ax.axhline(0.0, color='0.3', linewidth=1.0, linestyle='--')
    if symlog:
        # linthresh 1 keeps the +-1 band (parity to 2x either way) linear; the subs put labelled
        # minor ticks between the decades, which a bare symlog axis leaves empty.
        ax.set_yscale('symlog', linthresh=1.0, subs=[2, 5])
        ax.yaxis.set_minor_formatter(FuncFormatter(tick_label))
        ax.tick_params(axis='y', which='minor', labelsize=6)
    ax.yaxis.set_major_formatter(FuncFormatter(tick_label))
    ax.grid(axis='y', alpha=0.25, linewidth=0.5)
    ax.set_ylabel('signed speedup s\n0 = parity, +1 = 2x faster, -1 = 2x slower', fontsize=7.5)
    ax.tick_params(labelsize=7)
    on_disk = CORPUS_SIZE.get(suite)
    coverage = f'{len(kernels)}/{on_disk}' if on_disk else str(len(kernels))
    ax.set_title(f'{SUITE_TITLE.get(suite, suite)}  -  speedup vs {DENOM_PHRASE.get(kind, kind)}'
                 f'  -  {coverage} kernels plotted',
                 fontsize=9,
                 loc='left')
    if not dense:
        ax.set_xticks(xs, [short_name(k) for k in kernels], rotation=90, fontsize=5.5)
    else:
        # 151 tsvc kernels cannot carry readable tick labels; the job's speedup.csv keeps the names.
        ax.set_xticks([])
        ax.set_xlabel(f'{len(kernels)} kernels, sorted by median speedup (names in the job\'s speedup.csv)', fontsize=7)
    # Headroom for the legend: it carries the geomeans, so it must not sit on top of the points.
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + 0.30 * (high - low))
    # Two columns, not three: the arm labels carry their geomean, and three of them overflow the
    # axes at the 9 in width a small corpus gets.
    ax.legend(loc='upper left',
              fontsize=6.0,
              ncols=min(2, len(by_arm)),
              framealpha=0.9,
              title='thin horizontal rule = that arm\'s geomean of the raw ratio',
              title_fontsize=6.0)


def render_group(group: Group, report: Report, points: list[Point], out_path: Path, sort_by: str, symlog: bool,
                 dpi: int) -> Path:
    """Draw ONE group's panels into ONE figure and write it. Returns the path written."""
    facets = facet_keys(points)
    arms = order_arms(p.arm for p in points)
    widest = max(len({p.kernel for p in points if (p.suite, p.kind) == key}) for key in facets)
    width = min(24.0, max(9.0, 0.09 * widest + 6.0))
    fig, axes = plt.subplots(len(facets),
                             1,
                             figsize=(width, 3.2 * len(facets) + 2.2),
                             squeeze=False,
                             constrained_layout=True)
    for ax, (suite, kind) in zip(axes[:, 0], facets, strict=True):
        panel = [p for p in points if p.suite == suite and p.kind == kind]
        draw_facet(ax, suite, kind, panel, arms, sort_by, symlog)
    sources = ', '.join(short_path(s) for s in report.sources)
    kernels = len({(p.suite, p.kernel) for p in points})
    mode = '' if report.mode == 'corpus' else f' - denominator mode `{report.mode}`'
    fig.suptitle(
        f'{group.title}\n'
        f'preset `{report.preset}`{mode} - {len(points)} measurements over {kernels} kernels - '
        f'geomeans are over the RAW ratio - {sources}',
        fontsize=10)
    fig.supxlabel(textwrap.fill(group.caption, width=int(width * 13)), fontsize=7.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def coverage_lines(group: Group, report: Report, points: list[Point]) -> list[str]:
    """Per corpus: what is plotted, what is on disk, what the corpus holds. Absence is a result."""
    plotted: dict[str, set[str]] = {}
    for p in points:
        plotted.setdefault(p.suite, set()).add(p.kernel)
    out = [
        '', '## Coverage', '', '| suite | kernels plotted | result files on disk | in corpus | no result file |',
        '|:--|--:|--:|--:|--:|'
    ]
    for suite in group.suites:
        have = len(plotted.get(suite, ()))
        on_disk = report.kernels_on_disk.get(suite, 0)
        total = CORPUS_SIZE.get(suite)
        absent = str(total - on_disk) if total else '?'
        out.append(f'| {suite} | {have} | {on_disk} | {total or "?"} | {absent} |')
    out += ['', 'A killed or partial sweep shows up as `no result file`; those kernels are absent, never imputed.']
    return out


def geomean_lines(group: Group, points: list[Point]) -> list[str]:
    """Geomean per (corpus, arm) and per arm within THIS figure's single denominator.

    There is no cross-figure row and no all-corpora row anywhere: the other figure divides by
    something else, so a pooled geomean would blend two denominators into one meaningless number.
    """
    per_corpus: dict[tuple[str, str, str], list[float]] = {}
    per_kind: dict[tuple[str, str], list[float]] = {}
    corpora: dict[tuple[str, str], set[str]] = {}
    for p in points:
        per_corpus.setdefault((p.suite, p.kind, p.arm), []).append(p.ratio)
        per_kind.setdefault((p.kind, p.arm), []).append(p.ratio)
        corpora.setdefault((p.kind, p.arm), set()).add(p.suite)
    out = [
        '', '## Geomean per corpus and arm', '',
        'Geomean of the RAW ratio `denominator_min / arm_min`; the signed column is that same geomean '
        'converted for reading (0 parity, `+1` = 2x faster, `-1` = 2x slower). `n` is the number of '
        'kernels actually behind it -- kernels that errored or miscompiled are excluded, not counted as 1.0.', '',
        '| suite | denominator | arm | geomean | signed | n |', '|:--|:--|:--|--:|--:|--:|'
    ]
    for (suite, kind, arm), ratios in sorted(per_corpus.items()):
        gm, n = geomean(ratios)
        out.append(f'| {suite} | {kind} | {arm} | {gm:.3f}x | {signed(gm):+.3f} | {n} |')
    out += [
        '', f'## Geomean per arm over this figure ({" + ".join(group.suites)})', '',
        'Pooled over the corpora on THIS figure only, which share one denominator. There is deliberately '
        'no row pooling these with the other figure: the denominators differ.', '',
        '| denominator | arm | geomean | signed | n | corpora |', '|:--|:--|--:|--:|--:|:--|'
    ]
    for (kind, arm), ratios in sorted(per_kind.items()):
        gm, n = geomean(ratios)
        where = ' '.join(sorted(corpora[(kind, arm)]))
        out.append(f'| {kind} | {arm} | {gm:.3f}x | {signed(gm):+.3f} | {n} | {where} |')
    return out


def excluded_lines(excluded: list[Excluded], points: list[Point], limit: int) -> list[str]:
    """What was measured but not drawn: the category counts first, then the reasons in detail."""
    if not excluded:
        return ['', '## Excluded from this figure', '', 'Nothing: every result file in this group contributed.']
    by_cat: dict[str, list[Excluded]] = {}
    for e in excluded:
        by_cat.setdefault(e.category, []).append(e)
    out = [
        '', '## Excluded from this figure', '', '| category | kernels | arm entries |', '|:--|--:|--:|'
    ]
    for cat, items in sorted(by_cat.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        kernels = len({(e.suite, e.kernel) for e in items if not e.arm})
        out.append(f'| {cat} | {kernels} | {sum(1 for e in items if e.arm)} |')
    by_why: dict[str, list[Excluded]] = {}
    for e in excluded:
        by_why.setdefault(e.why, []).append(e)
    out += ['', '| reason | count | examples |', '|:--|--:|:--|']
    for why, items in sorted(by_why.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        shown = ', '.join(f'{e.suite}:{e.kernel}' + (f'/{e.arm}' if e.arm else '') for e in items[:limit])
        more = f' ... and {len(items) - limit} more' if len(items) > limit else ''
        out.append(f'| {why} | {len(items)} | {shown}{more} |')
    slow = {(p.suite, p.kernel) for p in points if p.den_ms < MIN_TIMEABLE_MS}
    if slow:
        out += [
            '', f"{len(slow)} plotted kernels have a denominator below the harness's {MIN_TIMEABLE_MS} ms timeable "
            'floor -- their ratio is per-call overhead as much as speedup. `--min-ms 0.5` drops them.'
        ]
    return out


def python_scalar_line(group: Group, records: dict[tuple[str, str], dict], preset: str) -> str:
    """How many of this figure's kernels have a scalar-Python reference, and what happened to them."""
    scalar = [k for (s, k), r in records.items() if s in group.suites and reference_kind(r, preset) == 'python-scalar']
    if not scalar:
        return 'No kernel in this group carries a python-scalar reference.'
    if group.kind == 'sequential-c++':
        return (f'{len(scalar)} of this group\'s kernels carry a python-scalar reference (the tsvc oracles). '
                'It is a correctness oracle only: it is never timed, and every bar here divides by seq-cpp.')
    return (f'{len(scalar)} of this group\'s kernels carry a python-scalar reference and are therefore ABSENT '
            'from the figure: dividing by a CPython scalar loop would read as a speedup in the hundreds.')


def table_lines(group: Group, report: Report, records: dict[tuple[str, str], dict], points: list[Point],
                excluded: list[Excluded], figure: Path | None, limit: int) -> list[str]:
    """One figure's whole Markdown report: provenance, caption, coverage, geomeans, exclusions."""
    sources = ', '.join(f'`{s}`' for s in report.sources)
    provenance = f'sources {sources} - denominator mode `{report.mode}` - {len(points)} measurements'
    if figure is not None:
        provenance += f' - figure `{figure}`'
    out = [
        f'# {group.title} -- preset `{report.preset}`', '', provenance, '', group.caption, '',
        python_scalar_line(group, records, report.preset), '',
        'Speedup = `denominator_min_ms / arm_min_ms` (best-of-N), so `>1x` means the arm is faster.'
    ]
    out += coverage_lines(group, report, points)
    out += geomean_lines(group, points)
    out += excluded_lines(excluded, points, limit)
    for note in report.notes:
        out += ['', f'note: {note}']
    return out + ['']


def split_csv(values: list[str] | None) -> list[str]:
    """``--suite a,b --suite c`` and ``--suite a --suite b`` mean the same thing."""
    return [item for value in (values or []) for item in value.split(',') if item]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=(__doc__ or '').splitlines()[0])
    ap.add_argument('--results',
                    action='append',
                    metavar='PATH',
                    help=f'job --out dir, results dir or single result JSON; repeatable '
                    f'(default {DEFAULT_RESULTS})')
    ap.add_argument('--preset', metavar='NAME', default='', help='dataset preset to plot (default paper, if present)')
    ap.add_argument('--suite', action='append', metavar='TAG', help='only these corpora (repeatable or comma-list)')
    ap.add_argument('--arm', action='append', metavar='LABEL', help='only these arms (repeatable or comma-list)')
    ap.add_argument('--denominator',
                    choices=('corpus', 'baseline'),
                    default='corpus',
                    help='corpus: the paper rule (timed numpy reference for np/poly, seq-cpp for tsvc*). '
                    'baseline: divide by the record\'s own baseline arm, the only denominator a '
                    'pre-six-arm sweep carries. Named in the figure either way')
    ap.add_argument('--min-ms',
                    type=float,
                    default=0.0,
                    metavar='MS',
                    help=f'drop kernels whose denominator is faster than this ({MIN_TIMEABLE_MS} is the '
                    'harness\'s timeable floor). Default 0: plot everything, report the fast ones')
    ap.add_argument('--sort', choices=('speedup', 'name'), default='speedup', help='kernel order within a panel')
    ap.add_argument('--symlog', action='store_true', help='symmetric-log y axis, for corpora with huge outliers')
    ap.add_argument('--out',
                    metavar='PREFIX',
                    default='',
                    help='path PREFIX for the two figures; each group appends its own slug and '
                    'extension (default <results>/corpus_speedup_<preset>)')
    ap.add_argument('--examples', type=int, default=8, metavar='N', help='exclusions listed per reason in the table')
    ap.add_argument('--dpi', type=int, default=160)
    return ap.parse_args(argv)


def choose_preset(records: dict[tuple[str, str], dict], requested: str) -> tuple[str, str]:
    """``(preset, why not)``: the requested one, else ``paper``, else the only one present."""
    present: list[str] = []
    for record in records.values():
        for preset in record.get('presets') or {}:
            if preset not in present:
                present.append(preset)
    if requested:
        if requested in present:
            return requested, ''
        return '', f'preset {requested!r} is in no result file; present: {present or "none"}'
    if 'paper' in present:
        return 'paper', ''
    if len(present) == 1:
        return present[0], ''
    return '', f'several presets present {present}; pick one with --preset'


def no_figure_reason(group: Group, excluded: list[Excluded], considered: int) -> str:
    """Why this group drew nothing. Refusing to draw is a correct outcome, but the causes -- no
    results at all, no verifiable denominator, an arm nobody measured -- need different fixes."""
    if considered == 0:
        return f'{group.slug}: no result file for {" ".join(group.suites)} under --results/--suite.'
    if sum(1 for e in excluded if not e.arm) >= considered:
        stale = sum(1 for e in excluded if e.category == CAT_STALE)
        extra = ' All of them are pre-six-arm files; --denominator baseline reads those.' if stale else ''
        return (f'{group.slug}: none of the {considered} kernels had a verifiable denominator. '
                f'The table lists why.{extra}')
    return f'{group.slug}: the --arm selection matched no measured arm. The table lists what is there.'


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    roots = [Path(r).expanduser() for r in (args.results or [str(DEFAULT_RESULTS)])]
    files = result_files(roots)
    if not files:
        print(f'no result JSON under {[str(r) for r in roots]}; run the job first', file=sys.stderr)
        return 1
    records, notes = load_records(files)
    preset, why = choose_preset(records, args.preset)
    if not preset:
        print(why, file=sys.stderr)
        return 1

    suites = split_csv(args.suite)
    points, excluded = collect(records, preset, args.denominator, suites, split_csv(args.arm), args.min_ms)
    on_disk: dict[str, int] = {}
    for suite, _kernel in records:
        on_disk[suite] = on_disk.get(suite, 0) + 1
    report = Report(preset, args.denominator, roots, on_disk, points, excluded, notes)

    prefix = Path(args.out) if args.out else roots[0] / f'corpus_speedup_{preset}'
    for note in notes:
        print(f'note: {note}')
    print(f'preset {preset} - denominator {args.denominator} - {len(records)} result files - '
          f'{len(points)} measurements - {len(excluded)} excluded')

    drawn = 0
    for group in GROUPS:
        if suites and not set(group.suites) & set(suites):
            continue
        group_points = [p for p in points if p.suite in group.suites]
        group_excluded = [e for e in excluded if e.suite in group.suites]
        considered = sum(1 for suite, _kernel in records if suite in group.suites and (not suites or suite in suites))
        figure: Path | None = None
        if group_points:
            figure = render_group(group, report, group_points, Path(f'{prefix}_{group.slug}.png'), args.sort,
                                  args.symlog, args.dpi)
            drawn += 1
        table_path = Path(f'{prefix}_{group.slug}.md')
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table_path.write_text('\n'.join(
            table_lines(group, report, records, group_points, group_excluded, figure, args.examples)))
        kernels = len({(p.suite, p.kernel) for p in group_points})
        print(f'[{group.slug}] {len(group_points)} measurements over {kernels} kernels - '
              f'{len(group_excluded)} excluded - table {table_path}')
        if figure is None:
            print(no_figure_reason(group, group_excluded, considered), file=sys.stderr)
        else:
            print(f'[{group.slug}] figure {figure}')
    if drawn == 0:
        print('NO FIGURE: neither group had a plottable measurement; see the tables above.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
