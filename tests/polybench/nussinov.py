# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import math
import dace
import polybench

N = dace.symbol('N')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.int32
base = dace.int8

# Dataset sizes
sizes = [{N: 60}, {N: 180}, {N: 500}, {N: 2500}, {N: 5500}]

args = [([N], datatype), ([N, N], datatype)]


def init_array(seq, table, n):
    for i in range(0, n):
        seq[i] = datatype((i + 1) % 4)
    table[:] = datatype(0)


@dace.program
def nussinov(seq: datatype[N], table: datatype[N, N]):
    for i in range(N - 1, -1, -1):
        for j in range(i + 1, N, 1):
            if j - 1 >= 0:

                @dace.tasklet
                def set_table_1():
                    center << table[i, j]
                    west << table[i, j - 1]
                    out >> table[i, j]
                    out = max(center, west)

            if i + 1 < N:

                @dace.tasklet
                def set_table_2():
                    center << table[i, j]
                    south << table[i + 1, j]
                    out >> table[i, j]
                    out = max(center, south)

            if j - 1 >= 0 and i + 1 < N:
                if i < j - 1:

                    @dace.tasklet
                    def set_table_3():
                        center << table[i, j]
                        swest << table[i + 1, j - 1]
                        seq_i << seq[i]
                        seq_j << seq[j]
                        out >> table[i, j]
                        out = max(center, swest + int(seq_i + seq_j == 3))
                else:

                    @dace.tasklet
                    def set_table_4():
                        center << table[i, j]
                        swest << table[i + 1, j - 1]
                        out >> table[i, j]
                        out = max(center, swest)

            for k in range(i + 1, j, 1):

                @dace.tasklet
                def set_table_5():
                    center << table[i, j]
                    k_center << table[i, k]
                    k_south << table[k + 1, j]
                    out >> table[i, j]
                    out = max(center, k_center + k_south)


def print_result(filename, *args, n=None, **kwargs):
    with open(filename, 'w') as fp:
        fp.write("==BEGIN DUMP_ARRAYS==\n")
        fp.write("begin dump: %s\n" % 'table')
        for i in range(0, n):
            for j in range(i, n):
                fp.write("{} ".format(args[1][i, j]))
            fp.write("\n")
        fp.write("\nend   dump: %s\n" % 'table')
        fp.write("==END   DUMP_ARRAYS==\n")


if __name__ == '__main__':
    polybench.main(sizes, args, print_result, init_array, nussinov)
