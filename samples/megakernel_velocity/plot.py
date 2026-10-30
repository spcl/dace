# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Plots the ``times.csv`` of one or more runs side by side: a row per device, a column per run, a group of bars per
namelist variant. Rows share their axis, so a column compares with its neighbor by height. Bars are the median time per
call and whiskers the quartiles; on a device the time is the compute region's, without the host-device copies. Each
megakernel bar is labeled with its change from the base of its group. The same numbers go to the terminal. Needs
matplotlib."""

import argparse
import collections
import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

KINDS = {"base": "#2a78d6", "megakernel": "#eb6834", "tuned": "#1baf7a"}
LABELS = {"base": "base", "megakernel": "megakernel, all syncs", "tuned": "megakernel, dependent syncs"}
DEVICES = ("cpu", "gpu")
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
BAR_WIDTH = 0.26


def read(paths: list[Path], schedule: str) -> tuple[dict, list[str]]:
    """The milliseconds of every repetition but a timestep's first, by run, device, variant and kind, and the runs in
    the order given. The CPU keeps one schedule, so that a bar is one configuration."""
    samples = collections.defaultdict(list)
    runs: list[str] = []
    for path in paths:
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        names = sorted({row["machine"] for row in rows})
        if clash := set(names) & set(runs):
            raise SystemExit(f"{path}: {sorted(clash)} is already a column; set MACHINE differently for one run")
        runs += names
        for row in rows:
            if int(row["rep"]) > 1 and (row["device"] != "cpu" or row["schedule"] == schedule):
                variant = f"lvn {row['lvn_only']}, istep {row['istep']}"
                key = (row["machine"], row["device"], variant, row["kind"])
                samples[key].append(1e3 * float(row["compute_s"] or row["call_s"]))
    return samples, runs


def change(samples: dict, run: str, device: str, variant: str, kind: str) -> float | None:
    """The percent difference of the median of ``kind`` from the base of its group, if both ran."""
    base = samples.get((run, device, variant, "base"))
    values = samples.get((run, device, variant, kind))
    if kind == "base" or not base or not values:
        return None
    return 100 * (statistics.median(values) / statistics.median(base) - 1)


def draw(ax, samples: dict, run: str, device: str, variants: list[str]) -> None:
    for index, kind in enumerate(KINDS):
        for x, variant in enumerate(variants):
            values = samples.get((run, device, variant, kind))
            if not values:
                continue
            quarters = statistics.quantiles(values, n=4)
            median = statistics.median(values)
            spot = x + (index - 1) * (BAR_WIDTH + 0.02)
            ax.bar(spot, median, BAR_WIDTH, color=KINDS[kind], label=LABELS[kind])
            ax.errorbar(spot, median, [[median - quarters[0]], [quarters[2] - median]], color=INK, lw=1, capsize=2)
            delta = change(samples, run, device, variant, kind)
            ax.annotate(
                f"{median:.2f}" if delta is None else f"{delta:+.0f}%",
                (spot, quarters[2]),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color=INK if delta is None else MUTED,
            )
    ax.set_xticks(range(len(variants)), variants, fontsize=8)
    ax.set_xlim(-0.6, len(variants) - 0.4)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, colors=MUTED, labelsize=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def table(samples: dict, schedule: str) -> None:
    print(f"{'run':<22}{'device':<7}{'variant':<16}{'kind':<12}{'ms':>7}{'vs base':>9}{'reps':>6}   (CPU: {schedule})")
    for key in sorted(samples, key=lambda k: (k[0], k[1], k[2], list(KINDS).index(k[3]))):
        run, device, variant, kind = key
        delta = change(samples, run, device, variant, kind)
        shown = "" if delta is None else f"{delta:+.1f}%"
        print(f"{run:<22}{device:<7}{variant:<16}{kind:<12}{statistics.median(samples[key]):>7.2f}{shown:>9}", end="")
        print(f"{len(samples[key]):>6}")


def main() -> None:
    argp = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    argp.add_argument("csv", type=Path, nargs="+", help="One times.csv per column, in the order to draw them")
    argp.add_argument("--out", type=Path, default=Path("megakernel_velocity.png"))
    argp.add_argument("--schedule", default="static0", help="CPU schedule to draw; static0 is plain schedule(static)")
    args = argp.parse_args()
    samples, runs = read(args.csv, args.schedule)
    if not samples:
        raise SystemExit("no repetitions to draw: is --schedule one of the CSV's schedules?")
    devices = [d for d in DEVICES if any(k[1] == d for k in samples)]
    variants = sorted({k[2] for k in samples})
    fig, axes = plt.subplots(
        len(devices),
        len(runs),
        figsize=(max(5.2, 3.6 * len(runs)), 2.9 * len(devices) + 0.5),
        squeeze=False,
        sharey="row",
        facecolor=SURFACE,
    )
    for row, device in enumerate(devices):
        for column, run in enumerate(runs):
            draw(axes[row, column], samples, run, device, variants)
        unit = "compute ms per call" if device == "gpu" else "ms per call"
        axes[row, 0].set_ylabel(f"{device.upper()}: {unit}", fontsize=8, color=MUTED)
        axes[row, 0].margins(y=0.14)
    for column, run in enumerate(runs):
        axes[0, column].set_title(run, fontsize=9, color=INK, loc="left")
    handles = {}
    for ax in axes.flat:
        handles |= {label: handle for handle, label in zip(*ax.get_legend_handles_labels())}
    fig.legend(handles.values(), handles.keys(), loc="upper center", ncols=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / fig.get_figheight()))
    fig.savefig(args.out, dpi=200)
    table(samples, args.schedule)


if __name__ == "__main__":
    main()
