from argparse import ArgumentParser
import numpy as np
import dace
import cupy as cp
from utils.general import optimize_sdfg
from subprocess import run
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.ncu import get_all_actions_filtered, get_achieved_performance, get_peak_performance, get_achieved_bytes
from utils.paths import get_thesis_playground_root_dir
from utils.general import enable_debug_flags, use_cache

# NBLOCKS = int(7e4)
KLEV = 137
NBLOCKS = dace.symbol('NBLOCKS')
nblocks_range = [4e4, 5e4, 6e4, 7e4, 8e4, 1e5, 2e5, 3e5, 4e4, 5e5]

dtype = dace.float64
ntype = np.float64


@dace.program
def kernel1(inp1: dtype[KLEV, NBLOCKS], out1: dtype[KLEV, NBLOCKS]):
    tmp = np.zeros([2, NBLOCKS], dtype=ntype)
    for i in dace.map[0:NBLOCKS]:
        for j in range(1, KLEV):
            tmp[j % 2, i] = (inp1[j, i] + inp1[j - 1, i])
            out1[j, i] = (tmp[j % 2, i] + tmp[(j-1) % 2, i])


def plot(df: pd.DataFrame, plot_filename: str):
    plt.rcParams.update({'figure.figsize': (19, 10)})
    plt.rcParams.update({'font.size': 12})
    sns.lineplot(data=df, x='Q', y='bandwidth_pct', marker='o', label='Q')
    sns.lineplot(data=df, x='D', y='bandwidth_pct', marker='o', label='D')
    print(df)
    plt.savefig(plot_filename)


def main():
    parser = ArgumentParser()
    parser.add_argument('--plot-only', action='store_true', default=False)
    parser.add_argument('--nblocks', type=int, default=None)
    args = parser.parse_args()

    if args.nblocks is not None:
        print(f"Run for NBLOCKS {args.nblocks}")

        # enable_debug_flags()
        # use_cache(dacecache_folder='kernel1')
        rng = np.random.default_rng(42)

        inputs = {}
        for i in range(1):
            name = f"inp{i+1}"
            inputs[name] = rng.random((KLEV, args.nblocks), dtype=ntype)

        out = (np.zeros((KLEV, args.nblocks), dtype=ntype))
        out_dev = cp.asarray(out.copy())

        sdfg = kernel1.to_sdfg(simplity=True)
        optimize_sdfg(sdfg, device=dace.DeviceType.GPU)

        this_inputs_dev = {}
        this_inputs = {}
        for i in range(1):
            name = f"inp{i+1}"
            this_inputs_dev[name] = cp.asarray(inputs[name])
            this_inputs[name] = inputs[name].copy()

        csdfg = sdfg.compile()
        # kernel1.f(**this_inputs, out1=out)
        csdfg(**this_inputs_dev, out1=out_dev, NBLOCKS=args.nblocks)

    elif not args.plot_only:
        print("Run different NBLOCKS sizes")
        data = []
        for nblocks in nblocks_range:
            nblocks = int(nblocks)
            run(['ncu', '--export', '/tmp/report.ncu-rep', '-f',  '--set', 'full',
                 'python3', __file__, '--nblocks', str(nblocks)], capture_output=False)
            actions = get_all_actions_filtered('/tmp/report.ncu-rep', '_numpy_full__map')
            if len(actions) > 1:
                print(f"WARNING: More than one action found, taking first")
            action = actions[0]
            if action is not None:
                D = get_achieved_bytes(action)
                bw_pct = get_achieved_performance(action)[1] / get_peak_performance(action)[1]
            else:
                D = 1
                bw_pct = 0
            Q = (KLEV * nblocks * 2 + 2 * nblocks) * 8
            data.append([f"Kernel{1}", bw_pct, Q, D, nblocks, Q / D])

        file = os.path.join(get_thesis_playground_root_dir(), "kernel_example_data.csv")
        this_df = pd.DataFrame(data, columns=['program', 'bandwidth_pct', 'Q', 'D', 'NBLOCKS', 'I/0 efficiency'])
        this_df.set_index(['program', 'NBLOCKS'], inplace=True)
        if os.path.exists(file):
            df = pd.read_csv(file, index_col=['program', 'NBLOCKS'])
            df = pd.concat([df, this_df])
        else:
            df = this_df
        print(df)
        df.to_csv(file)
    elif args.plot_only:
        print("Plot")
        file = os.path.join(get_thesis_playground_root_dir(), "kernel_example_data.csv")
        df = pd.read_csv(file, index_col=['program', 'NBLOCKS'])
        plot(df, os.path.join(get_thesis_playground_root_dir(), "kernel_example_plot.png"))


if __name__ == '__main__':
    main()
