from argparse import ArgumentParser
from tabulate import tabulate

from ncu_utils import get_action


def main():
    parser = ArgumentParser(description="Print bandwidths from ncu report")
    parser.add_argument('ncureport')

    args = parser.parse_args()

    action = get_action(args.ncureport)

    cycles = action.metric_by_name('gpc__cycles_elapsed.max').as_double()
    peak_beta = action.metric_by_name('dram__bytes.sum.peak_sustained').as_double()
    beta_pct = action.metric_by_name('dram__bytes_read.sum.pct_of_peak_sustained_elapsed').as_double()
    beta_pct += action.metric_by_name('dram__bytes_write.sum.pct_of_peak_sustained_elapsed').as_double()
    Q = action.metric_by_name('dram__bytes_write.sum').as_double()
    Q += action.metric_by_name('dram__bytes_read.sum').as_double()

    print(tabulate(
        [
            ['Cycles', int(cycles), 'gpc__cycles_elapsed.max'],
            ['Q [byte]', int(Q), 'dram__bytes_write.sum + dram__bytes_read.sum'],
            ['beta [byte/cycle]', Q / cycles, 'Q / cycles'],
            ['beta / peak_beta [%]', Q / cycles / peak_beta * 100, 'beta / dram__bytes.sum.peak_sustained'],
            ['beta_pct [%]', beta_pct, 'dram__bytes_read.sum.pct_of_peak_sustained_elapsed + dram__bytes_write.sum.pct_of_peak_sustained_elapsed']
        ], 
        headers=['Name', 'Value', 'formula'],
        floatfmt='.1f'))


if __name__ == '__main__':
    main()
