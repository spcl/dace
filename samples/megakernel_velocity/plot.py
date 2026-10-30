# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Plots the ``times.csv`` of ``run_experiment.sh``: median time per call and quartiles of every kind, one panel per
device and namelist variant. On a device the time is the compute region's, without the host-device copies. Needs
matplotlib."""

import argparse
import collections
import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

COLORS = {"base": "#2a78d6", "megakernel": "#eb6834", "tuned": "#1baf7a"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
BAR_WIDTH = 0.26


def read(paths: list[Path]) -> dict:
    """The milliseconds of every repetition but a timestep's first, by machine, device, variant, schedule and kind."""
    samples = collections.defaultdict(list)
    for path in paths:
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if int(row["rep"]) > 1:
                    variant = f"lvn_only={row['lvn_only']}, istep={row['istep']}"
                    key = (row["machine"], row["device"], variant, row["schedule"], row["kind"])
                    samples[key].append(1e3 * float(row["compute_s"] or row["call_s"]))
    return samples


def draw(ax, samples: dict, panel: tuple) -> None:
    schedules = sorted({k[3] for k in samples if k[:3] == panel}, key=lambda name: (len(name), name))
    kinds = sorted({k[4] for k in samples if k[:3] == panel}, key=list(COLORS).index)
    top = 0.0
    for index, kind in enumerate(kinds):
        for x, schedule in enumerate(schedules):
            values = samples.get((*panel, schedule, kind))
            if not values:
                continue
            quarters = statistics.quantiles(values, n=4)
            median = statistics.median(values)
            spot = x + (index - (len(kinds) - 1) / 2) * (BAR_WIDTH + 0.02)
            ax.bar(spot, median, BAR_WIDTH, color=COLORS[kind], label=kind)
            ax.errorbar(spot, median, [[median - quarters[0]], [quarters[2] - median]], color=INK, lw=1, capsize=2)
            ax.annotate(
                f"{median:.1f}",
                (spot, quarters[2]),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=6,
                color=MUTED,
            )
            top = max(top, quarters[2])
    ax.set_xticks(range(len(schedules)), schedules, color=MUTED, fontsize=8)
    ax.set_xlim(-1, len(schedules))
    ax.set_ylim(0, top * 1.15)
    ax.set_title(f"{panel[0]} {panel[1].upper()}, {panel[2]}", fontsize=8, color=INK, loc="left")
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, colors=MUTED, labelsize=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def main() -> None:
    argp = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    argp.add_argument("csv", type=Path, nargs="+")
    argp.add_argument("--out", type=Path, default=Path("megakernel_velocity.png"))
    args = argp.parse_args()
    samples = read(args.csv)
    panels = sorted({k[:3] for k in samples})
    columns = len({p[2] for p in panels})
    rows = (len(panels) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3.6 * columns, 2.8 * rows), squeeze=False, facecolor=SURFACE)
    for ax, panel in zip(axes.flat, panels):
        draw(ax, samples, panel)
    for ax in axes.flat[len(panels) :]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("ms per call", fontsize=8, color=MUTED)
    handles = {}
    for ax in axes.flat:
        handles |= {label: handle for handle, label in zip(*ax.get_legend_handles_labels())}
    fig.legend(handles.values(), handles.keys(), loc="upper right", ncols=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, dpi=200)


if __name__ == "__main__":
    main()
