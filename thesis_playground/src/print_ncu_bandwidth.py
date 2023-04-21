from argparse import ArgumentParser
from tabulate import tabulate

from ncu_utils import get_action


def main():
    parser = ArgumentParser(description="Print bandwidths from ncu report")
    parser.add_argument('ncureport')

    args = parser.parse_args()

    action = get_action(args.ncureport)

    cycles = action.metric_by_name('gpc__cycles_elapsed.max').as_double()
    # assume here that the unit is always nseconds
    runtime = action.metric_by_name('gpu__time_duration.sum').as_double() / 1e9
    peak_beta = action.metric_by_name('dram__bytes.sum.peak_sustained').as_double()
    beta_pct = action.metric_by_name('dram__bytes_read.sum.pct_of_peak_sustained_elapsed').as_double()
    beta_pct += action.metric_by_name('dram__bytes_write.sum.pct_of_peak_sustained_elapsed').as_double()
    Q = action.metric_by_name('dram__bytes_write.sum').as_double()
    Q += action.metric_by_name('dram__bytes_read.sum').as_double()
    cycles_memory_second = action.metric_by_name('dram__cycles_elapsed.avg.per_second').as_double()
    cycles_memory = cycles_memory_second * runtime
    achieved_traffic = action.metric_by_name('dram__bytes.sum.per_second').as_double()

    print(tabulate(
        [
            ['Cycles', int(cycles), 'gpc__cycles_elapsed.max'],
            ['Cycles memory [cycles/s]', cycles_memory_second, 'dram__cycles_elapsed.avg.per_second'],
            ['Cycles memory [cycles]', int(cycles_memory), 'cycles_memory * runtime'],
            ['Q [byte]', int(Q), 'dram__bytes_write.sum + dram__bytes_read.sum'],
            ['beta [byte/cycle]', Q / cycles_memory, 'Q / cycles'],
            ['beta / peak_beta [%]', Q / cycles_memory / peak_beta * 100, 'beta / dram__bytes.sum.peak_sustained'],
            ['beta_pct [%]', beta_pct, 'dram__bytes_read.sum.pct_of_peak_sustained_elapsed +' +
                                       ' dram__bytes_write.sum.pct_of_peak_sustained_elapsed'],
            ['peak traffic (ncu roofline) [byte/s]', peak_beta * cycles_memory_second, 'dram__bytes.sum.peak_sustained *' +
                                                                                'Cycles memory'],
            ['achieved traffic (ncu roofline) [byte/s]', achieved_traffic, 'dram__bytes.sum.per_second'],
            ['achieved / peak traffic [%]', achieved_traffic / (peak_beta * cycles_memory_second) * 100,
             'achieved traffic / peak traffic'],

        ],
        headers=['Name', 'Value', 'formula'],
        floatfmt='.1f'))


if __name__ == '__main__':
    main()
