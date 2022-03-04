# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import json
import os
import os.path
import itertools

from typing import Dict

from dace.optimization import cutout_tuner as ct

class DistributedCutoutTuner():
    """Distrubuted wrapper for cutout tuning."""

    def __init__(self, tuner: ct.CutoutTuner) -> None:
        self._tuner = tuner

    def optimize(self, **kwargs) -> Dict:
        cutouts = []
        hash_groups = {}
        existing_files = {}
        for state, node in self._tuner.cutouts():
            state_id = self._tuner._sdfg.node_id(state)
            node_id = state.node_id(node)
            label = node.label

            # TODO: How to get the hash?
            node_hash = hash(node)

            if node_hash not in hash_groups:
                hash_groups[node_hash] = []
                cutouts.append((node_hash, (state, node)))

            hash_groups[node_hash].append((state_id, node_id, label))

            file_name = self._tuner.file_name(state_id, node_id, node.label)
            if file_name is not None:
                if node_hash not in existing_files:
                    existing_files[node_hash] = set()

                existing_files[node_hash].add(file_name)
        
        num_ranks = get_world_size()
        chunk_size = len(cutouts) // (num_ranks - 1)
        chunks = list(chunk(cutouts, chunk_size))
        
        chunk = chunks[get_local_rank()]
        for node_hash, (state, node) in chunk:
            state_id, node_id, label = hash_groups[node_hash][0]

            if node_hash not in existing_files:
                results = self._tuner.evaluate(state=state, node=node, **kwargs)
                
                file_name = self._tuner.file_name(state_id, node_id, label)
                with open(file_name, 'w') as fp:
                    json.dump(results, fp)

                existing_files[node_hash] = set([file_name])

            with open(next(iter(existing_files[node_hash])), "r") as handle:
                results = json.load(handle)

            for (state_id, node_id, label) in hash_groups[node_hash][1:]:
                file_name = self._tuner.file_name(state_id, node_id, label)
                if file_name not in existing_files[node_hash]:
                    with open(file_name, 'w') as fp:
                        json.dump(results, fp)

  
def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(itertools.islice(it, size)), ())

def get_local_rank():
    if 'MV2_COMM_WORLD_LOCAL_RANK' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_LOCAL_RANK'])
    elif 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    elif 'SLURM_LOCALID' in os.environ:
        return int(os.environ['SLURM_LOCALID'])
    else:
        raise RuntimeError('Cannot get local rank')


def get_local_size():
    if 'MV2_COMM_WORLD_LOCAL_SIZE' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_LOCAL_SIZE'])
    elif 'OMPI_COMM_WORLD_LOCAL_SIZE' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
    elif 'SLURM_NTASKS_PER_NODE' in os.environ:
        return int(os.environ['SLURM_NTASKS_PER_NODE'])
    else:
        raise RuntimeError('Cannot get local comm size')


def get_world_rank():
    if 'MV2_COMM_WORLD_RANK' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_RANK'])
    elif 'OMPI_COMM_WORLD_RANK' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        return int(os.environ['SLURM_PROCID'])
    else:
        raise RuntimeError('Cannot get world rank')


def get_world_size():
    if 'MV2_COMM_WORLD_SIZE' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_SIZE'])
    elif 'OMPI_COMM_WORLD_SIZE' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_SIZE'])
    elif 'SLURM_NTASKS' in os.environ:
        return int(os.environ['SLURM_NTASKS'])
    else:
        raise RuntimeError('Cannot get world size')
