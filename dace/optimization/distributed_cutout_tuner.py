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
        hash_groups = OrderedDict()
        existing_files = {}
        for (state_id, node_id), (state, node) in self._tuner.cutouts():
            label = node.label
            if isinstance(node, (dace.nodes.LibraryNode, dace.nodes.Tasklet)):
                node_hash = label.split("_")[-1]
            else:
                node_hash = (state_id, node_id)

            # Group nodes by hashes
            if node_hash not in hash_groups:
                hash_groups[node_hash] = []

            hash_groups[node_hash].append((state_id, node_id, label))

            # Keep track of existing files
            file_name = self._tuner.file_name(state_id, node_id, node.label)
            result = self._tuner.try_load(file_name)
            if result is not None:
                if node_hash not in existing_files:
                    existing_files[node_hash] = set()

                existing_files[node_hash].add(file_name)

        # Filter cutouts
        new_cutouts = []
        copy_cutouts = []
        for node_hash in hash_groups:
            if node_hash not in existing_files:
                new_cutouts.append(node_hash)
            elif len(hash_groups[node_hash]) < len(existing_files[node_hash]):
                copy_cutouts.append(node_hash)

        # Split work
        rank = get_world_rank()
        num_ranks = get_world_size()
        chunk_size = len(new_cutouts) // max(num_ranks, 1)
        chunks = list(partition(new_cutouts, chunk_size))

        if rank >= len(chunks):
            return

        dreport: data_report.InstrumentedDataReport = self._tuner._sdfg.get_instrumented_data()
        self._tuner.rank = rank
        self._tuner.num_ranks = num_ranks
        # Tune new cutouts
        chunk = chunks[rank]
        for node_hash in chunk:
            state_id, node_id, label = hash_groups[node_hash][0]
            state = self._tuner._sdfg.node(state_id)
            node = state.node(node_id)

            results = self._tuner.evaluate(state=state, node=node, dreport=dreport, measurements=measurements, **kwargs)

            # Write out for all identical cutouts
            for (state_id, node_id, label) in hash_groups[node_hash]:
                file_name = self._tuner.file_name(state_id, node_id, label)
                with open(file_name, 'w') as fp:
                    json.dump(results, fp)

        # Finish incomplete groups
        if rank == 0:
            for node_hash in copy_cutouts:
                for (state_id, node_id, label) in hash_groups[node_hash]:
                    file_name = self._tuner.file_name(state_id, node_id, label)
                    if file_name not in existing_files[node_hash]:
                        with open(file_name, 'w') as fp:
                            json.dump(results, fp)


class DistributedSpaceTuner:
    """Distributed wrapper for cutout tuning that distributes search space of each cutout across ranks."""

    def __init__(self, tuner: ct.CutoutTuner) -> None:
        self._tuner = tuner

    def optimize(self, measurements: int = 30, **kwargs) -> Dict:
        rank = get_world_rank()
        num_ranks = get_world_size()

        hash_groups = OrderedDict()
        existing_files = {}
        for (state_id, node_id), (state, node) in self._tuner.cutouts():
            label = f'{rank}_{node.label}'
            if isinstance(node, (dace.nodes.LibraryNode, dace.nodes.Tasklet)):
                node_hash = label.split("_")[-1]
            else:
                node_hash = (state_id, node_id)

            # Group nodes by hashes
            if node_hash not in hash_groups:
                hash_groups[node_hash] = []

            hash_groups[node_hash].append((state_id, node_id, label))

            # Keep track of existing files
            file_name = self._tuner.file_name(state_id, node_id, label)
            result = self._tuner.try_load(file_name)
            if result is not None:
                if node_hash not in existing_files:
                    existing_files[node_hash] = set()

                existing_files[node_hash].add(file_name)

        # Filter cutouts
        new_cutouts = []
        copy_cutouts = []
        for node_hash in hash_groups:
            if node_hash not in existing_files:
                new_cutouts.append(node_hash)
            elif len(hash_groups[node_hash]) < len(existing_files[node_hash]):
                copy_cutouts.append(node_hash)

        dreport: data_report.InstrumentedDataReport = self._tuner._sdfg.get_instrumented_data()
        self._tuner.rank = rank
        self._tuner.num_ranks = num_ranks

        # Tune each cutout
        for node_hash in new_cutouts:
            state_id, node_id, label = hash_groups[node_hash][0]
            state = self._tuner._sdfg.node(state_id)
            node = state.node(node_id)

            cutout = cutter.cutout_state(state, node, make_copy=False)
            cutout.instrument = self._tuner.instrument

            # Setup arguments once
            arguments = {}
            for cstate in cutout.nodes():
                for dnode in cstate.data_nodes():
                    if cutout.arrays[dnode.data].transient:
                        continue

                    arguments[dnode.data] = dreport.get_first_version(dnode.data)

            configs = list(self._tuner.space(node))

            # Split work
            chunk_size = len(configs) // max(num_ranks, 1)
            chunk_start = rank * chunk_size
            chunk_end = None if rank == (num_ranks - 1) else ((rank + 1) * chunk_size)

            label = f'{rank + 1}/{num_ranks}: {node.label}'
            results = {}
            for config in tqdm(list(itertools.islice(configs, chunk_start, chunk_end)), desc=label):
                key, value = self._tuner.evaluate_single(config, cutout, arguments, state, node, dreport, measurements,
                                                         **kwargs)
                results[key] = value

            # Write out for all identical cutouts
            for (state_id, node_id, label) in hash_groups[node_hash]:
                file_name = self._tuner.file_name(state_id, node_id, label)
                with open(file_name, 'w') as fp:
                    json.dump(results, fp)

        # Finish incomplete groups
        if rank == 0:
            for node_hash in copy_cutouts:
                for (state_id, node_id, label) in hash_groups[node_hash]:
                    file_name = self._tuner.file_name(state_id, node_id, label)
                    if file_name not in existing_files[node_hash]:
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
