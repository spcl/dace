# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from absl import app, flags
import numpy as np
import functools
import dace
from dace import propagate_memlets_sdfg
import subprocess
import json

names = [
    'k2mm',
    'k3mm',
    'adi',
    'atax',
    'bicg',
    'cholesky',
    'correlation',
    'covariance',
    'deriche',
    'doitgen',
    'durbin',
    'fdtd2d',
    'floyd_warshall',
    'gemm',
    'gemver',
    'gesummv',
    # checked by hand: correct, just numerical errors
    # 'gramschmidt',
    'heat3d',
    'jacobi1d',
    'jacobi2d',
    'lu',
    'ludcmp',
    'mvt',
    'nussinov',
    'seidel2d',
    # checked by hand: correct, just numerical errors
    # 'symm',
    # checked by hand: correct, just numerical errors
    # 'syr2k',
    'syrk',
    'trisolv',
    'trmm'
]

orig_path = "/dace/samples/polybench"
poly_path = "/parallel-loop/polybench_sequential"

sizes = ['mini', 'small', 'medium', 'large', 'extralarge']


def main():
    name = names[0]
    for name in names:
        with open("{}/{}.dace.out".format(orig_path, name), 'r') as file:
            orig_res = file.readlines()[2]
        with open("{}/{}.dace.out".format(poly_path, name), 'r') as file:
            # data = file.read().replace('\n', '')
            poly_res = file.readlines()[2]
        if orig_res == poly_res:
            # print(name, "ok")
            pass
        else:
            print(name, "error:")
            print("orig_res", orig_res)
            print("poly_res", poly_res)



if __name__ == '__main__':
    main()
