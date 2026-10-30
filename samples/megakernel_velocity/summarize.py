# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Summarizes the logs of ``run_experiment.sh``: the median call time of every configuration and the numerics check,
and with ``--csv`` every repetition, one row each, which ``plot.py`` reads.

A log is ``<device>_<base|megakernel|tuned>_l<lvn_only>i<istep>[_<schedule>].log``. The first repetition of a timestep is
dropped from the summary: it allocates and, on a device, launches the first kernels. The CSV keeps it.
"""

import argparse
import collections
import csv
import re
import statistics
from pathlib import Path

TIME = re.compile(r"^TIME dace timestep=(\d+) rep=(\d+) seconds=\s*([\d.]+)")
LOG_NAME = re.compile(r"^(cpu|gpu)_(base|megakernel|tuned)_l(\d)i(\d)(?:_(.+))?$")
CSV_COLUMNS = ("machine", "device", "kind", "lvn_only", "istep", "schedule", "timestep", "rep", "call_s", "compute_s")
COMPUTE = re.compile(r"^COMPUTE seconds=([\d.]+)")
DIFF = re.compile(r"^DIFF dace_vs_t1 (\S+) max_abs=\s*\S+ max_rel=\s*(\S+) n_differ=(\d+)")


def read_log(path: Path) -> dict:
    calls, computes, worst, differing = [], [], 0.0, 0
    compute = None  # Printed by the kernel, before the driver prints the time of the call
    for line in path.read_text().splitlines():
        if match := COMPUTE.match(line):
            compute = float(match[1])
        elif match := TIME.match(line):
            if int(match[2]) > 1:
                calls.append(float(match[3]))
                if compute is not None:
                    computes.append(compute)
        elif match := DIFF.match(line):
            worst = max(worst, float(match[2]))
            differing += int(match[3])
    return {"calls": calls, "computes": computes, "worst": worst, "differing": differing}


def repetitions(path: Path) -> list[dict]:
    """One row per repetition of the log ``path``. ``compute_s``, the time without host-device copies, is empty on the
    CPU, which has none."""
    device, kind, lvn_only, istep, schedule = LOG_NAME.match(path.stem).groups()
    config = dict(device=device, kind=kind, lvn_only=lvn_only, istep=istep, schedule=schedule or "default")
    rows, compute = [], ""
    for line in path.read_text().splitlines():
        if match := COMPUTE.match(line):
            compute = match[1]
        elif match := TIME.match(line):
            rows.append(config | dict(timestep=match[1], rep=match[2], call_s=match[3], compute_s=compute))
            compute = ""
    return rows


def write_csv(logs: Path, machine: str, target: Path) -> None:
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, CSV_COLUMNS)
        writer.writeheader()
        for path in sorted(logs.glob("*.log")):
            writer.writerows(row | {"machine": machine} for row in repetitions(path))


def main() -> None:
    argp = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    argp.add_argument("logs", type=Path)
    argp.add_argument("--csv", type=Path, help="Write every repetition to this file")
    argp.add_argument("--machine", default="", help="The machine column of the CSV")
    args = argp.parse_args()
    if args.csv:
        write_csv(args.logs, args.machine, args.csv)
    rows = collections.defaultdict(dict)
    for path in sorted(args.logs.glob("*.log")):
        config, variant, *schedule = path.stem.replace("cpu_", "cpu-").replace("gpu_", "gpu-").split("_")
        rows[(variant, "_".join(schedule))][config] = read_log(path)
    print(
        f"{'variant':8} {'schedule':12} {'config':18} {'calls':>5} {'call [ms]':>10} {'compute [ms]':>13} {'max rel':>9} {'differ':>7}"
    )
    for (variant, schedule), configs in sorted(rows.items()):
        for config, row in sorted(configs.items()):
            call = 1e3 * statistics.median(row["calls"]) if row["calls"] else float("nan")
            compute = 1e3 * statistics.median(row["computes"]) if row["computes"] else float("nan")
            print(
                f"{variant:8} {schedule:12} {config:18} {len(row['calls']):5} {call:10.2f} {compute:13.2f} "
                f"{row['worst']:9.1e} {row['differing']:7}"
            )


if __name__ == "__main__":
    main()
