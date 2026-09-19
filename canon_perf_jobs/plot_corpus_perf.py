#!/usr/bin/env python3
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Plot and tabulate what the corpus perf job already measured. READ-ONLY: nothing is timed here.

Input is the per-kernel JSON the timing driver writes -- ``<out>/perf_json/<suite>_<kernel>.json``
for a job run, ``tests/passes/canonicalize/perf_results/`` for a bare driver run. This script never
builds, compiles or re-times a kernel: a plot that could trigger a measurement would silently mix a
fresh number into an old sweep, and the two would no longer be the same machine state.

    python plot_corpus_perf.py                       # the job's default results directory
    python canon_perf_jobs/plot_corpus_perf.py --results DIR --suite tsvc --arm dace-canon-gcc

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

* The form is SMALL MULTIPLES OF SCATTER: one panel per arm, one dot per kernel, and never a line
  joining kernels -- kernels are a categorical axis, so a polyline across them would draw a trend
  that does not exist. All panels share one kernel order and one y axis, so a kernel sits at the
  same x in every panel and two arms are compared by looking straight up the column.
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
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

#: Where the job writes its results, i.e. ``corpus_perf_job.DEFAULT_OUT``: beside the scripts, so a
#: finished run is one directory. Resolved from THIS file rather than imported -- the plotter must
#: stay readable on a box with no dace and no corpora, and the two defaults have to agree.
DEFAULT_RESULTS = Path(__file__).resolve().with_name('corpus_perf_results')
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
#: added arm still plots -- the series come from the DATA, this list only fixes panel order.
#: Each autopar arm is immediately followed by its ``-default`` twin (same flags minus the forcing
#: knob), so the forced/unforced pair reads side by side in every figure and table.
ARM_ORDER = ('dace-autoopt-gcc', 'dace-autoopt-llvm', 'dace-canon-gcc', 'dace-canon-llvm', 'dace-simplify+gcc-autopar',
             'dace-simplify+gcc-autopar-default', 'dace-simplify+llvm-autopar', 'dace-simplify+llvm-autopar-default',
             'seq-cpp', 'auto-opt', 'auto-opt+llvm', 'canon', 'canon+gcc', 'canon+llvm', 'gcc-autopar', 'llvm-autopar',
             'canon-serial+gcc-autopar', 'canon-serial+llvm-autopar', 'gcc-graphite', 'clang-polly')
#: The arm labels the current job writes. A record carrying none of them predates the eight-arm
#: table: its numbers were taken under different flags and are reported as stale rather than plotted
#: next to fresh ones.
CURRENT_ARMS = ARM_ORDER[:9]
#: The harness's own "too fast to time reliably" floor. Not applied by default (dropping data is the
#: reader's call, not the plotter's) but kernels below it are counted, since a ratio between two
#: sub-millisecond timings is per-call overhead, not a speedup.
MIN_TIMEABLE_MS = 0.5
#: Slots 1-3 of the documented categorical palette, which are the only three that clear the
#: all-pairs colourblind gates a scatter/small-multiples figure is held to. Hue therefore encodes
#: the arm's COMPILER BACKEND, not its identity -- nine arms cannot get nine safe hues, and the
#: backend is the split the figure is actually read for. Identity is carried by the panel title.
BACKEND_COLOUR = {'gcc': '#2a78d6', 'llvm': '#eb6834', 'other': '#1baf7a'}
SURFACE, INK, INK_SOFT, INK_MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#898781'
GRID_INK, AXIS_INK, BAND_INK = '#e1e0d9', '#c3c2b7', '#f2f1ec'
#: Marker area in points^2, interpolated between a 54-kernel and a 223-kernel panel.
DOT_LARGE, DOT_SMALL = 13.0, 4.5
#: The raw ratios that get a y tick. A ladder rather than symlog's own decades: the decades snap the
#: axis out to +-100x for one outlier and leave two thirds of every panel empty, and symlog's minor
#: subs put a second row of labels between them.
SPEEDUP_TICKS = (0.01, 0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0)

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
        'numpy-reference', ('np', 'poly'), 'numpy',
        'npbench + polybench, speedup over the timed parallel numpy reference',
        'Denominator: each kernel\'s own numpy reference, timed like any other arm at the machine default thread '
        'count. numpy dispatches gemm / 2mm / 3mm / syrk / cholesky into a THREADED OpenBLAS, so this axis is a '
        'comparison against parallel numpy, not against one core. Kernels whose numpy form is a scalar Python loop '
        '(seidel_2d) carry a python-scalar reference, are never divided by it, and are absent by construction. '
        'These numbers are NOT comparable with the seq-cpp figure: the two divide by different things.'),
    Group(
        'seq-cpp', ('tsvc', 'tsvc25'), 'sequential-c++',
        'tsvc + tsvc_2_5, speedup over sequential C++ (the seq-cpp arm)',
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
                Excluded(suite, kernel, '', CAT_UNGROUPED,
                         'corpus ' + repr(suite) + ' is in neither denominator group'))
            continue
        if mode == 'corpus' and stale_record(record, preset):
            arm_names = ' '.join(sorted((preset_of(record, preset).get('pipelines') or {})))
            excluded.append(
                Excluded(suite, kernel, '', CAT_STALE,
                         f'stale result file: arms {arm_names} predate the six-arm table'))
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
            excluded.append(Excluded(suite, kernel, '', CAT_TOO_FAST,
                                     f'denominator {den.min_ms:.4f} ms below --min-ms'))
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


def panel_keys(points: list[Point], arms: list[str]) -> list[tuple[str, str]]:
    """``(denominator kind, arm)`` panels: ONE arm per panel, in arm order, kinds never mixed.

    Faceting by arm rather than drawing nine series into one axes is what makes 223x9 points
    readable, and it is also the only encoding the colour rules allow: a scatter is held to the
    all-pairs colourblind gates, which no ordering of nine hues can pass.
    """
    keys = {(p.kind, p.arm) for p in points}
    return sorted(keys, key=lambda k: (k[0], arms.index(k[1])))


def short_path(path: Path, keep: int = 3) -> str:
    """A results root, short enough for a figure title; the full path stays in the table."""
    parts = path.parts
    return str(path) if len(parts) <= keep else '.../' + '/'.join(parts[-keep:])


def median_signed(ratios: Iterable[float]) -> float:
    """Sort key for a kernel: the median signed speedup over the arms that measured it."""
    good = [v for v in (signed(r) for r in ratios) if math.isfinite(v)]
    return statistics.median(good) if good else float('-inf')


def backend_of(arm: str) -> str:
    """Which compiler backend an arm's hue stands for; anything else shares the third slot."""
    if 'llvm' in arm or 'clang' in arm or 'polly' in arm:
        return 'llvm'
    if 'gcc' in arm or 'graphite' in arm:
        return 'gcc'
    return 'other'


class Layout(NamedTuple):
    """The kernel x axis, computed ONCE and reused by every panel of a figure.

    Sharing it is what lets the reader compare arms: kernel ``k`` is at the same x in all panels,
    so looking straight up a column is a like-for-like comparison of the arms on one kernel.
    """
    slot: dict[tuple[str, str], int]  # (suite, kernel) -> x
    blocks: list[tuple[str, int, int]]  # suite, first x, one past the last x
    width: int


def kernel_layout(points: list[Point], sort_by: str) -> Layout:
    """Lay the kernels out along x: one contiguous, separately sorted block per corpus."""
    ratios: dict[tuple[str, str], list[float]] = {}
    for p in points:
        ratios.setdefault((p.suite, p.kernel), []).append(p.ratio)
    suites = sorted({s for s, _ in ratios}, key=lambda s: (SUITES.index(s) if s in SUITES else len(SUITES), s))
    gap = max(2, len(ratios) // 30)
    slot: dict[tuple[str, str], int] = {}
    blocks: list[tuple[str, int, int]] = []
    x = 0
    for suite in suites:
        kernels = sorted(k for s, k in ratios if s == suite)
        if sort_by == 'speedup':
            kernels.sort(key=lambda k: median_signed(ratios[(suite, k)]))
        start = x
        for kernel in kernels:
            slot[(suite, kernel)] = x
            x += 1
        blocks.append((suite, start, x))
        x += gap
    return Layout(slot, blocks, max(1, x - gap))


def sym(value: float) -> float:
    """The symlog(linthresh=1) transform of a signed speedup, and :func:`unsym` its inverse.

    Only used to pad the shared y limits by a constant fraction of the axis, which is otherwise
    either wrong (padding in data units on a log tail) or left to symlog's decade snapping.
    """
    return value if abs(value) <= 1.0 else math.copysign(1.0 + math.log10(abs(value)), value)


def unsym(value: float) -> float:
    return value if abs(value) <= 1.0 else math.copysign(10.0**(abs(value) - 1.0), value)


def y_bounds(values: list[float], symlog: bool) -> tuple[float, float]:
    """Shared y limits: the data, padded by 6% of the axis it is actually drawn on."""
    low, high = min(values), max(values)
    if not symlog:
        pad = 0.06 * (high - low) or 1.0
        return low - pad, high + pad
    pad = 0.06 * (sym(high) - sym(low)) or 0.1
    return unsym(sym(low) - pad), unsym(sym(high) + pad)


def block_ticks(layout: Layout, plotted: dict[str, int]) -> tuple[list[float], list[str]]:
    """One x tick per corpus block, centred, carrying that corpus's coverage."""
    positions, labels = [], []
    for suite, start, stop in layout.blocks:
        on_disk = CORPUS_SIZE.get(suite)
        coverage = f'{plotted.get(suite, stop - start)}/{on_disk}' if on_disk else str(stop - start)
        positions.append((start + stop - 1) / 2.0)
        labels.append(f'{SUITE_TITLE.get(suite, suite)}\n{coverage} kernels')
    return positions, labels


def draw_arm_panel(ax: Axes, arm: str, points: list[Point], layout: Layout, symlog: bool, paired: bool) -> None:
    """ONE arm: a dot per kernel, its geomean as a horizontal rule, and nothing joining the dots."""
    colour = BACKEND_COLOUR[backend_of(arm)]
    forced = not arm.endswith('-default')
    size = DOT_SMALL + (DOT_LARGE - DOT_SMALL) * max(0.0, min(1.0, (223 - layout.width) / 169.0))
    for index, (_suite, start, stop) in enumerate(layout.blocks):
        if index % 2:
            ax.axvspan(start - 0.5, stop - 0.5, color=BAND_INK, linewidth=0.0, zorder=0)
    ax.scatter([layout.slot[(p.suite, p.kernel)] for p in points], [signed(p.ratio) for p in points],
               s=size if forced else size * 1.9,
               marker='o',
               facecolors=colour if forced else 'none',
               edgecolors=colour,
               linewidths=0.0 if forced else 0.7,
               alpha=0.85 if forced else 1.0,
               zorder=4)
    ax.axhline(0.0, color=AXIS_INK, linewidth=1.1, zorder=3)
    gm, n = geomean(p.ratio for p in points)
    if math.isfinite(gm):
        ax.axhline(signed(gm), color=colour, linewidth=1.4, linestyle=(0, (5, 3)), zorder=5)
    if symlog:
        # linthresh 1 keeps the +-1 band (parity to 2x either way) linear, so the interesting
        # near-parity kernels are not squeezed onto one row of pixels.
        ax.set_yscale('symlog', linthresh=1.0)
    ax.set_yticks([signed(r) for r in SPEEDUP_TICKS])
    ax.yaxis.set_major_formatter(FuncFormatter(tick_label))
    ax.minorticks_off()
    ax.grid(axis='y', color=GRID_INK, linewidth=0.6, zorder=1)
    ax.set_axisbelow(False)
    ax.set_xlim(-1.0, layout.width)
    ax.set_facecolor(SURFACE)
    # labelleft on EVERY panel, not just the shared left column: the columns sit a figure width
    # apart, and tracing a gridline that far is how a reader misreads a value.
    ax.tick_params(labelsize=7, colors=INK_SOFT, length=2.5, labelleft=True)
    for side in ('left', 'bottom'):
        ax.spines[side].set(color=AXIS_INK, linewidth=0.8)
    for side in ('right', 'top'):
        ax.spines[side].set_visible(False)
    variant = ('  -  forcing knob ON' if forced else '  -  compiler cost model decides') if paired else ''
    gm_s = f'{gm:.2f}x' if math.isfinite(gm) else '--'
    ax.set_title(f'{arm}\ngeomean {gm_s}  -  n={n}{variant}',
                 fontsize=8.2,
                 loc='left',
                 color=INK,
                 linespacing=1.4,
                 pad=4.0)


def draw_summary(ax: Axes, panels: list[tuple[str, str]], points: list[Point], symlog: bool) -> None:
    """The ranking the small multiples cannot show at a glance: one geomean dot per arm, one axis.

    A dot and a stem, never a polyline: these are eight independent numbers, not a trajectory.
    """
    labels, spread = [], [0.0]
    for row, (kind, arm) in enumerate(reversed(panels)):
        labels.append(arm)
        gm, _n = geomean(p.ratio for p in points if p.arm == arm and p.kind == kind)
        if not math.isfinite(gm):
            continue
        colour = BACKEND_COLOUR[backend_of(arm)]
        forced = not arm.endswith('-default')
        spread.append(signed(gm))
        ax.plot([0.0, signed(gm)], [row, row], color=colour, linewidth=1.2, alpha=0.5, zorder=2)
        ax.plot([signed(gm)], [row],
                marker='o',
                markersize=7.0,
                color=colour,
                markerfacecolor=colour if forced else SURFACE,
                markeredgewidth=1.3,
                zorder=3)
        ax.annotate(f'{gm:.2f}x', (signed(gm), row),
                    textcoords='offset points',
                    xytext=(10, -2.5),
                    fontsize=7.2,
                    color=INK_SOFT)
    ax.axvline(0.0, color=AXIS_INK, linewidth=1.1, zorder=1)
    if symlog:
        ax.set_xscale('symlog', linthresh=1.0)
    ax.set_xticks([signed(r) for r in SPEEDUP_TICKS])
    # Its own limits, not the panels': every geomean lands within a few x of parity, so borrowing
    # the panels' 0.01x..100x span would squeeze the whole ranking into a tenth of the axis. Set
    # AFTER set_xticks, which widens the view interval to span whatever ticks it is given.
    low, high = y_bounds(spread, symlog)
    ax.set_xlim(low, unsym(sym(high) + 0.15 * (sym(high) - sym(low))))
    ax.xaxis.set_major_formatter(FuncFormatter(tick_label))
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(-0.8, len(labels) - 0.2)
    ax.minorticks_off()
    ax.grid(axis='x', color=GRID_INK, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)
    ax.tick_params(labelsize=7, colors=INK_SOFT, length=2.5)
    for side in ('left', 'bottom'):
        ax.spines[side].set(color=AXIS_INK, linewidth=0.8)
    for side in ('right', 'top'):
        ax.spines[side].set_visible(False)
    ax.set_title('geomean per arm  -  all kernels pooled', fontsize=8.2, loc='left', color=INK, pad=4.0)


def legend_handles(layout: Layout, arms: list[str]) -> list[Line2D]:
    """Proxies for the encoding itself: what a dot, a hollow dot and each rule mean."""
    grey = dict(color='none', markeredgecolor=INK_MUTED, markerfacecolor=INK_MUTED)
    suites = ' then '.join(SUITE_TITLE.get(s, s) for s, _, _ in layout.blocks)
    backends = {backend_of(arm) for arm in arms}
    hue_keys = [
        Line2D([], [],
               marker='o',
               markersize=5,
               color='none',
               markerfacecolor=BACKEND_COLOUR[backend],
               markeredgecolor='none',
               label=f'dot colour: {backend} backend' if backend != 'other' else 'dot colour: neither')
        for backend in ('gcc', 'llvm', 'other') if backend in backends
    ]
    return [
        Line2D([], [], marker='o', markersize=5, label='one kernel x this arm', **grey),
        Line2D([], [],
               marker='o',
               markersize=6.5,
               color='none',
               markeredgecolor=INK_MUTED,
               markerfacecolor='none',
               label='hollow: the `-default` twin of the arm above it'),
        Line2D([], [],
               color=INK_MUTED,
               linestyle=(0, (5, 3)),
               linewidth=1.4,
               label="this arm's geomean of the raw "
               'ratio (the number the table reports)'),
        Line2D([], [], color=AXIS_INK, linewidth=1.4, label='parity, 1.00x'),
        *hue_keys,
        Line2D([], [], color='none', label=f'x: kernels, {suites}, each sorted by median speedup'),
        Line2D([], [], color='none', label='no line joins kernels -- the x axis is categorical'),
    ]


def render_group(group: Group, report: Report, points: list[Point], out_path: Path, sort_by: str, symlog: bool,
                 dpi: int) -> Path:
    """Draw ONE group's panels into ONE figure and write it. Returns the path written."""
    arms = order_arms(p.arm for p in points)
    panels = panel_keys(points, arms)
    layout = kernel_layout(points, sort_by)
    plotted: dict[str, int] = {}
    for suite, kernel in {(p.suite, p.kernel) for p in points}:
        plotted[suite] = plotted.get(suite, 0) + 1
    ncols = 1 if len(panels) < 2 else 2
    # Reserved cells: the legend always, the per-arm ranking only when there is more than one arm to
    # rank -- a lone lollipop restates the panel title in a cell-sized empty box.
    reserved = 2 if len(panels) > 1 else 1
    nrows = math.ceil((len(panels) + reserved) / ncols)
    panel_w = min(11.0, max(4.2, 0.021 * layout.width + 3.6))
    width = ncols * panel_w + 1.1
    fig, axes = plt.subplots(nrows,
                             ncols,
                             figsize=(width, 2.35 * nrows + 2.4),
                             squeeze=False,
                             sharex=True,
                             sharey=True,
                             constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    cells = [axes[r][c] for r in range(nrows) for c in range(ncols)]
    positions, labels = block_ticks(layout, plotted)
    kinds = {kind for kind, _ in panels}
    twins = {arm.removesuffix('-default') for arm in arms if arm.endswith('-default')}
    for cell, (kind, arm) in zip(cells, panels, strict=False):
        draw_arm_panel(cell, arm, [p for p in points if p.arm == arm and p.kind == kind], layout, symlog,
                       arm.removesuffix('-default') in twins)
        if len(kinds) > 1:
            cell.set_xlabel(f'vs {DENOM_PHRASE.get(kind, kind)}', fontsize=6.5, color=INK_SOFT)
    cells[0].set_ylim(*y_bounds([signed(p.ratio) for p in points], symlog))
    for column in range(ncols):
        bottom = max((i for i in range(len(panels)) if i % ncols == column), default=None)
        if bottom is not None:
            cells[bottom].set_xticks(positions, labels)
            cells[bottom].tick_params(axis='x', labelbottom=True, labelsize=7, colors=INK_SOFT)
    for spare in cells[len(panels):]:
        spare.set_axis_off()
    if reserved > 1:
        # The summary carries speedup on x, so it cannot stay in the shared-axis grid: it is dropped
        # and rebuilt on its own cell.
        spec = cells[len(panels)].get_subplotspec()
        cells[len(panels)].remove()
        draw_summary(fig.add_subplot(spec), panels, points, symlog)
    cells[len(panels) + reserved - 1].legend(handles=legend_handles(layout, arms),
                                             loc='upper left',
                                             fontsize=7,
                                             frameon=False,
                                             labelcolor=INK_SOFT,
                                             handletextpad=0.9,
                                             title=f'one panel = one arm, {len(plotted)} '
                                             f'{"corpus" if len(plotted) == 1 else "corpora"}, shared y axis',
                                             title_fontsize=7.5,
                                             alignment='left')
    for index in range(0, len(panels), ncols):
        cells[index].set_ylabel('signed speedup s\n0 = parity, +1 = 2x, -1 = half', fontsize=6.8, color=INK_SOFT)
    sources = ', '.join(short_path(s) for s in report.sources)
    kernels = len({(p.suite, p.kernel) for p in points})
    mode = '' if report.mode == 'corpus' else f' - denominator mode `{report.mode}`'
    subtitle = (f'preset `{report.preset}`{mode} - {len(points)} measurements over {kernels} kernels in '
                f'{len(panels)} arm panels - geomeans are over the RAW ratio - {sources}')
    # Wrapped against the figure width: an unwrapped provenance line is simply clipped off the page.
    fig.suptitle(f'{group.title}\n' + textwrap.fill(subtitle, width=int(width * 13)), fontsize=10, color=INK)
    fig.supxlabel(textwrap.fill(group.caption, width=int(width * 13)), fontsize=7.5, color=INK_SOFT)
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
    out = ['', '## Excluded from this figure', '', '| category | kernels | arm entries |', '|:--|--:|--:|']
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
    ap.add_argument('--yscale',
                    choices=('symlog', 'linear'),
                    default='symlog',
                    help='symlog (default): one arm 1000x slower than the reference otherwise flattens '
                    'the whole panel onto the parity line. The +-1 band stays linear either way')
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
                                  args.yscale == 'symlog', args.dpi)
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
