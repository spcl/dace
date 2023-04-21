from argparse import ArgumentParser
import copy
import cupy as cp
import cupyx as cpx
import dace
import numpy as np

from cupyx.profiler import benchmark
from dace.data import _prod
from dace.transformation.auto.auto_optimize import auto_optimize
from typing import Any, Dict, Set, Tuple


NBLOCKS = 65536
NPROMA = 1
KLEV = 137

dtype = dace.float64
ntype = np.float64

@dace.program
def map_loop(inp: dtype[NBLOCKS, KLEV], out: dtype[NBLOCKS, KLEV]):
    tmp = np.zeros(([NBLOCKS, KLEV]), dtype=ntype)
    # tmp = dace.define_local([NBLOCKS, KLEV], dtype)
    for i in dace.map[0:NBLOCKS]:
        # tmp[i, 0] = inp[i, 0]
        # tmp[i, 1] = (inp[i, 0] + inp[i, 1]) * 2
        for j in range(2, KLEV):
            tmp[i, j] = (inp[i, j] + inp[i, j - 1] + inp[i, j - 2]) * 3
            out[i, j] = (tmp[i, j] + tmp[i, j - 1] + tmp[i, j - 2]) * 3


@dace.program
def map_loop2(inp: dtype[KLEV, NBLOCKS], out: dtype[KLEV, NBLOCKS]):
    tmp = np.zeros(([3, NBLOCKS]), dtype=ntype)
    # tmp = dace.define_local([KLEV, NBLOCKS], dtype)
    for i in dace.map[0:NBLOCKS]:
        # tmp[0, i] = inp[0, i]
        # tmp[1, i] = (inp[0, i] + inp[1, i]) * 2
        for j in range(2, KLEV):
            tmp[j % 3, i] = (inp[j, i] + inp[j - 1, i] + inp[j - 2, i]) * 3
            out[j, i] = (tmp[j % 3, i] + tmp[(j - 1) % 3, i] + tmp[(j - 2) % 3, i]) * 3


def change_strides(sdfg: dace.SDFG, klev_vals: Tuple[int], syms_to_add: Set[str] = None) -> Dict[str, int]:

    permutation = dict()
    syms_to_add = syms_to_add or set()

    for name, desc in sdfg.arrays.items():

        # We target arrays that have KLEV or KLEV + 1 in their shape
        shape_str = [str(s) for s in desc.shape]
        klev_idx = None
        divisor = None
        for v in klev_vals:
            if str(v) in shape_str:
                klev_idx = shape_str.index(str(v))
                divisor = v
                break
        if klev_idx is None:
            continue

        permutation[name] = klev_idx

        is_fortran = (desc.strides[0] == 1)

        # Update the strides
        new_strides = list(desc.strides)
        if is_fortran:
            for idx in range(klev_idx + 1, len(desc.shape)):
                new_strides[idx] /= divisor
        else:
            for idx in range(klev_idx):
                new_strides[idx] /= divisor
        new_strides[klev_idx] = _prod(desc.shape) / divisor
        desc.strides = tuple(new_strides)

        # Go to nested SDFGs
        # Assuming only 1 level of nested SDFG
        for sd in sdfg.all_sdfgs_recursive():

            if sd is sdfg:
                continue

            assert sd.parent_sdfg is sdfg

            for s in syms_to_add:
                if s not in sd.parent_nsdfg_node.symbol_mapping:
                        sd.parent_nsdfg_node.symbol_mapping[s] = s
                        sd.add_symbol(s, dace.int32)

            for nname, ndesc in sd.arrays.items():

                if isinstance(ndesc, dace.data.Scalar):
                    continue
                if ndesc.transient:
                    continue

                nsdfg_node = sd.parent_nsdfg_node
                is_input = True
                edges = list(sd.parent.in_edges_by_connector(nsdfg_node, nname))
                if len(edges) == 0:
                    is_input = False
                    edges = list(sd.parent.out_edges_by_connector(nsdfg_node, nname))
                    if len(edges) == 0:
                        raise ValueError
                edge = edges[0]
                if is_input:
                    src = sd.parent.memlet_path(edge)[0].src
                else:
                    src = sd.parent.memlet_path(edge)[-1].dst
                assert isinstance(src, dace.nodes.AccessNode)
                if src.data not in sdfg.arrays:
                    continue

                subset = edge.data.subset
                squeezed = copy.deepcopy(subset)
                rem_idx = squeezed.squeeze()
                assert len(squeezed) == len(ndesc.shape)
                # inv_sqz_idx = [i for i in range(len(desc.shape)) if i not in sqz_idx]
                nnew_strides = [new_strides[i] for i in rem_idx]
                ndesc.strides = tuple(nnew_strides)

    return permutation


def permute(input: Dict[str, Any], permutation: Dict[str, int]) -> Dict[str, Any]:
    permuted = dict()
    for name, arr in input.items():
        if name in permutation:
            klev_idx = permutation[name]
            is_fortran = (arr.strides[0] == 1)
            if is_fortran:
                order = list(range(klev_idx)) + list(range(klev_idx + 1, len(arr.shape))) + [klev_idx]
            else:
                order = [klev_idx] + list(range(klev_idx)) + list(range(klev_idx + 1, len(arr.shape)))
            permuted[name] = arr.transpose(order).copy()
        else:
            permuted[name] = arr
    return permuted


def unpermute(output: Dict[str, Any], permutation: Dict[str, int]) -> Dict[str, Any]:
    unpermuted = dict()
    for name, arr in output.items():
        if name in permutation:
            klev_idx = permutation[name]
            is_fortran = (arr.strides[0] == 1)
            if is_fortran:
                order = list(range(klev_idx)) + [len(arr.shape) - 1] + list(range(klev_idx + 1, len(arr.shape) - 1))
            else:
                order = list(range(1, klev_idx + 1)) + [0] + list(range(klev_idx + 1, len(arr.shape)))
            unpermuted[name] = arr.transpose(order).copy()
        else:
            unpermuted[name] = arr
    return unpermuted


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--loop', choices=['1', '2', '3', 'all'], default='all')
    parser.add_argument('--skip-test', action='store_true', default=False)
    args = parser.parse_args()

    rng = np.random.default_rng(42)

    if issubclass(ntype, np.integer):
        inp = rng.integers(-5, 5, (NBLOCKS, KLEV), dtype=ntype)
    else:
        inp = rng.random((NBLOCKS, KLEV), dtype=ntype)
    ref = np.zeros((NBLOCKS, KLEV), ntype)
    val = np.zeros((NBLOCKS, KLEV), ntype)

    inp2 = np.transpose(inp).copy()
    if not args.skip_test:
        map_loop.f(inp, ref)
        map_loop(inp=inp, out=val)

        assert np.allclose(ref, val)

        val2 = np.zeros((KLEV, NBLOCKS), ntype)
        map_loop2.f(inp2, val2)

        assert np.allclose(np.transpose(ref), val2)

    sdfg = map_loop.to_sdfg(simplify=True)
    sdfg.arrays['inp'].storage = dace.StorageType.GPU_Global
    sdfg.arrays['tmp'].storage = dace.StorageType.GPU_Global
    sdfg.arrays['out'].storage = dace.StorageType.GPU_Global
    # auto_optimize(sdfg, dace.DeviceType.GPU)
    sdfg.apply_gpu_transformations()

    inp_dev = cp.asarray(inp)
    ref_dev = cp.asarray(ref)
    val_dev = cp.zeros_like(inp_dev)

    if not args.skip_test:
        sdfg(inp=inp_dev, out=val_dev)

        assert cp.allclose(ref_dev, val_dev)

    sdfg2 = map_loop2.to_sdfg(simplify=True)
    sdfg2.arrays['inp'].storage = dace.StorageType.GPU_Global
    sdfg2.arrays['tmp'].storage = dace.StorageType.GPU_Global
    sdfg2.arrays['out'].storage = dace.StorageType.GPU_Global
    # auto_optimize(sdfg2, dace.DeviceType.GPU)
    sdfg2.apply_gpu_transformations()

    inp2_dev = cp.asarray(inp2)
    val2_dev = cp.zeros_like(inp2_dev)

    if not args.skip_test:
        sdfg2(inp=inp2_dev, out=val2_dev)

        assert cp.allclose(cp.transpose(ref_dev), val2_dev)

    sdfg3 = map_loop.to_sdfg(simplify=True)
    sdfg3.name = 'map_loop3'
    perm = change_strides(sdfg3, (KLEV, KLEV+1))
    sdfg3.arrays['inp'].storage = dace.StorageType.GPU_Global
    sdfg3.arrays['tmp'].storage = dace.StorageType.GPU_Global
    sdfg3.arrays['out'].storage = dace.StorageType.GPU_Global
    # auto_optimize(sdfg3, dace.DeviceType.GPU)
    sdfg3.apply_gpu_transformations()

    input3 = {'inp': inp}
    inp3 = permute(input3, perm)['inp']
    inp_dev = cp.asarray(inp3)
    val_dev = cp.zeros_like(inp_dev)

    if not args.skip_test:
        sdfg3(inp=inp_dev, out=val_dev)

        value3 = {'out': val_dev}
        val3 = unpermute(value3, perm)['out']

        # print(f"Reference input: \n{inp}\n")
        # print(f"Transposed input: \n{inp2}\n")
        # print(f"Result input: \n{inp3}\n")

        # print(f"Reference: \n{ref_dev}\n")
        # print(f"Transposed result: \n{val2_dev}\n")
        # print(f"Result: \n{val_dev}\n")

        assert cp.allclose(ref_dev, val3)

    # Benchmark
    def func(*args):
        inp_dev, val_dev = args
        csdfg(inp=inp_dev, out=val_dev)

    if args.loop in ['1', 'all']:
        csdfg = sdfg.compile()
        print(benchmark(func, (inp_dev, val_dev), n_repeat=10, n_warmup=10))

    if args.loop in ['2', 'all']:
        csdfg = sdfg2.compile()
        print(benchmark(func, (inp2_dev, val2_dev), n_repeat=10, n_warmup=10))

    if args.loop in ['3', 'all']:
        csdfg = sdfg3.compile()
        print(benchmark(func, (inp_dev, val_dev), n_repeat=10, n_warmup=10))
