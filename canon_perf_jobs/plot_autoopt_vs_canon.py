#!/usr/bin/env python3
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Bar-plot summary of dace-autoopt vs dace-canon from a corpus_perf_job run.

Reads per-kernel JSONs written by tests.passes.canonicalize.canonicalize_perf_corpus_test
and produces two CPU figures:

* array_corpora_speedup.png  - polybench + npbench speedup over the timed numpy reference
* loop_corpora_speedup.png   - tsvc + tsvc_2_5 speedup over dace-autoopt-gcc

Only numerically-correct arms are included; the geomean is taken over the raw ratios.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

AUTOOPT = 'dace-autoopt-gcc'
CANON = 'dace-canon-gcc'

ARRAY_SUITES = ('poly', 'np')
LOOP_SUITES = ('tsvc', 'tsvc25')

SUITE_TITLES = {
    'poly': 'polybench',
    'np': 'npbench',
    'tsvc': 'tsvc',
    'tsvc25': 'tsvc_2_5',
}


def geomean(ratios: Iterable[float]) -> tuple[float, int]:
    good = [r for r in ratios if math.isfinite(r) and r > 0.0]
    if not good:
        return float('nan'), 0
    return math.exp(sum(math.log(r) for r in good) / len(good)), len(good)


def load_records(results_dir: Path) -> dict[tuple[str, str], dict]:
    records: dict[tuple[str, str], dict] = {}
    perf_json = results_dir / 'perf_json'
    if not perf_json.is_dir():
        return records
    for path in sorted(perf_json.glob('*.json')):
        with open(path) as f:
            rec = json.load(f)
        records[(rec.get('suite'), rec.get('kernel'))] = rec
    return records


def ratios_for_suites(records: dict, suites: tuple[str, ...], key: str, arm: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {s: [] for s in suites}
    for (suite, _kernel), rec in records.items():
        if suite not in suites:
            continue
        for pres in (rec.get('presets') or {}).values():
            val = (pres.get(key) or {}).get(arm)
            entry = (pres.get('pipelines') or {}).get(arm, {})
            if entry.get('correct') and val is not None and math.isfinite(val) and val > 0.0:
                out[suite].append(float(val))
    return out


def plot_group(records: dict,
               suites: tuple[str, ...],
               key: str,
               ylabel: str,
               title: str,
               out_path: Path,
               show_autoopt: bool = False) -> None:
    """Plot grouped bars for the two arms. When show_autoopt is True the autoopt arm is also drawn."""
    arms = [AUTOOPT, CANON] if show_autoopt else [CANON]
    data = {arm: ratios_for_suites(records, suites, key, arm) for arm in arms}

    fig, ax = plt.subplots(figsize=(max(7, 2.2 * len(suites) + 3), 5.5))
    x = np.arange(len(suites))
    width = 0.35 if len(arms) == 2 else 0.5

    for offset, arm in enumerate(arms):
        means = []
        counts = []
        for suite in suites:
            vals = data[arm].get(suite, [])
            gm, n = geomean(vals)
            means.append(gm)
            counts.append(n)
        pos = x + (offset - (len(arms) - 1) / 2) * width
        bars = ax.bar(pos, means, width, label=arm, color=['#2a78d6', '#eb6834'][offset])
        for bar, gm, n in zip(bars, means, counts):
            height = bar.get_height()
            if math.isfinite(height):
                ax.annotate(f'{height:.2f}×\n(n={n})',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords='offset points',
                            ha='center',
                            va='bottom',
                            fontsize=8)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([SUITE_TITLES[s] for s in suites])
    ax.axhline(y=1.0, color='k', linestyle='--', linewidth=0.8, label='parity' if len(arms) == 1 else None)
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_path}')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--array-dir', required=True, help='results directory for the array (poly+np) sweep')
    ap.add_argument('--loops-dir', required=True, help='results directory for the loops (tsvc+tsvc25) sweep')
    ap.add_argument('--out-dir', default='.', help='where to write the PNG files')
    args = ap.parse_args()

    array_records = load_records(Path(args.array_dir))
    loops_records = load_records(Path(args.loops_dir))
    out_dir = Path(args.out_dir)

    if array_records:
        plot_group(array_records,
                   ARRAY_SUITES,
                   'speedup_vs_reference',
                   'geomean speedup over parallel numpy',
                   'CPU: dace-autoopt vs dace-canon\n(polybench + npbench, preset S)',
                   out_dir / 'array_corpora_speedup.png',
                   show_autoopt=True)
    else:
        print(f'warning: no array records in {args.array_dir}')

    if loops_records:
        plot_group(loops_records,
                   LOOP_SUITES,
                   'speedup_vs_baseline',
                   'geomean speedup over dace-autoopt-gcc',
                   'CPU: dace-canon vs dace-autoopt\n(tsvc + tsvc_2_5, preset S)',
                   out_dir / 'loop_corpora_speedup.png',
                   show_autoopt=False)
    else:
        print(f'warning: no loops records in {args.loops_dir}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
