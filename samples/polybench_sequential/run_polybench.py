# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import subprocess

names = [
    '2mm',
    '3mm',
    'adi',
    'atax',
    'bicg',
    'cholesky',
    'correlation',
    'covariance',
    'deriche',
    'doitgen',
    'durbin',
    'fdtd-2d',
    'floyd-warshall',
    'gemm',
    'gemver',
    'gesummv',
    'gramschmidt',
    'heat-3d',
    'jacobi-1d',
    'jacobi-2d',
    'lu',
    'ludcmp',
    'mvt',
    'nussinov',
    'seidel-2d',
    'symm',
    'syr2k',
    'syrk',
    'trisolv',
    'trmm'
]


def run_test(name, size):
    out = subprocess.Popen(['python',
                            '{}.py'.format(name),
                            '--size={}'.format(size)],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    stdout, stderr = out.communicate()
    assert stderr is None
    try:
        time_ms = float(stdout.splitlines()[-2][5:-3])
        return time_ms
    except:
        for line in stdout.splitlines():
            print(line)
        return 0.0


sizes = ['mini', 'small', 'medium', 'large', 'extralarge']


def main():
    size = sizes[0]
    for name in names:
        time = run_test(name, size)
        print(name, time)


if __name__ == '__main__':
    # name = 'doitgen'
    # size = 'mini'
    # print(name, run_test(name, size))
    main()
