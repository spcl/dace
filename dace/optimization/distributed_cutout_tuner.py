# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import os

from collections import OrderedDict
import dace
import json
import itertools

from typing import Dict

from dace.optimization import cutout_tuner as ct
from dace.sdfg.analysis import cutout as cutter
from dace.codegen.instrumentation.data import data_report

try:
    from tqdm import tqdm
except (ImportError, ModuleNotFoundError):
    tqdm = lambda x, **kwargs: x


class DistributedCutoutTuner:
    """Distributed wrapper for cutout tuning that distributes the cutouts across ranks."""

    def __init__(self, tuner: ct.CutoutTuner) -> None:
        self._tuner = tuner

    def optimize(self, measurements: int = 30, **kwargs) -> Dict:
        cutouts = OrderedDict()
        existing_files = set()
        for cutout, cutout_hash in self._tuner.cutouts():
            cutouts[cutout_hash] = cutout

            file_name = self._tuner.file_name(cutout_hash)
            result = self._tuner.try_load(file_name)
            if result is not None:
                existing_files.add(cutout_hash)

        # Filter cutouts
        new_cutouts = []
        for hash in cutouts:
            if hash not in existing_files:
                new_cutouts.append(hash)

        # Split work
        rank = get_world_rank()
        num_ranks = get_world_size()
        chunk_size = len(new_cutouts) // max(num_ranks, 1)
        chunks = list(partition(new_cutouts, chunk_size))

        if rank >= len(chunks):
            return

        self._tuner.rank = rank
        self._tuner.num_ranks = num_ranks
        # Tune new cutouts
        chunk = chunks[rank]
        for hash in chunk:
            cutout = cutouts[hash]
            results = self._tuner.search(cutout=cutout, measurements=measurements, **kwargs)

            file_name = self._tuner.file_name(hash)
            with open(file_name, 'w') as fp:
                json.dump(results, fp)


class DistributedSpaceTuner:
    """Distributed wrapper for cutout tuning that distributes search space of each cutout across ranks."""

    def __init__(self, tuner: ct.CutoutTuner) -> None:
        self._tuner = tuner

    def optimize(self, measurements: int = 30, **kwargs) -> Dict:
        rank = get_world_rank()
        num_ranks = get_world_size()

        cutouts = OrderedDict()
        existing_files = set()
        for cutout, cutout_hash in self._tuner.cutouts():
            cutout_hash = f'{cutout_hash}_{rank}'

            cutouts[cutout_hash] = cutout
            file_name = self._tuner.file_name(cutout_hash)
            result = self._tuner.try_load(file_name)
            if result is not None:
                existing_files.add(cutout_hash)

        # Filter cutouts
        new_cutouts = []
        for hash in cutouts:
            if hash not in existing_files:
                new_cutouts.append(hash)

        self._tuner.rank = rank
        self._tuner.num_ranks = num_ranks

        for cutout_hash in new_cutouts:
            cutout = cutouts[cutout_hash]
            evaluate_kwargs = self._tuner.pre_evaluate(cutout=cutout,
                                                       measurements=measurements,
                                                       **kwargs)

            configs = list(self._tuner.space(**(evaluate_kwargs["space_kwargs"])))

            # Split work
            chunk_size = len(configs) // max(num_ranks, 1)
            chunk_start = rank * chunk_size
            chunk_end = None if rank == (num_ranks - 1) else ((rank + 1) * chunk_size)

            label = f'{rank + 1}/{num_ranks}: {cutout_hash}'
            results = {}
            key = evaluate_kwargs["key"]
            for config in tqdm(list(itertools.islice(configs, chunk_start, chunk_end)), desc=label):
                evaluate_kwargs["config"] = config
                runtime = self._tuner.evaluate(**evaluate_kwargs)
                results[key(config)] = runtime

                file_name = self._tuner.file_name(cutout_hash)
                with open(file_name, 'w') as fp:
                    json.dump(results, fp)


def partition(it, size):
    it = iter(it)
    return iter(lambda: tuple(itertools.islice(it, size)), ())


def get_world_rank():
    if 'MV2_COMM_WORLD_RANK' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_RANK'])
    elif 'OMPI_COMM_WORLD_RANK' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        return int(os.environ['SLURM_PROCID'])
    else:
        print('Cannot get world rank, running in sequential mode')
        return 0


def get_world_size():
    if 'MV2_COMM_WORLD_SIZE' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_SIZE'])
    elif 'OMPI_COMM_WORLD_SIZE' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_SIZE'])
    elif 'SLURM_NTASKS' in os.environ:
        return int(os.environ['SLURM_NTASKS'])
    else:
        print('Cannot get world size, running in sequential mode')
        return 1
