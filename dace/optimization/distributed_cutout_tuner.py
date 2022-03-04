# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import os

from simplejson import OrderedDict
import dace
import json
import itertools

from typing import Dict

from dace.optimization import cutout_tuner as ct

class DistributedCutoutTuner():
    """Distrubuted wrapper for cutout tuning."""

    def __init__(self, tuner: ct.CutoutTuner) -> None:
        self._tuner = tuner

    def optimize(self, **kwargs) -> Dict:
        cutouts = []
        hash_groups = OrderedDict()
        existing_files = {}
        for state, node in self._tuner.cutouts():
            state_id = self._tuner._sdfg.node_id(state)
            node_id = state.node_id(node)
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
            if file_name is not None:
                if node_hash not in existing_files:
                    existing_files[node_hash] = set()
                
                existing_files[node_hash].add(file_name)

            cutouts.append((node_hash, (state, node)))

        # Filter cutouts
        new_cutouts = []
        copy_cutouts = []
        for node_hash in hash_groups:
            if node_hash not in existing_files:
                new_cutouts.append(node_hash)
            elif len(hash_groups[node_hash]) < len(existing_files[node_hash]):
                copy_cutouts.append(copy_cutouts)


        # Split work
        rank = get_local_rank()
        num_ranks = get_world_size()
        chunk_size = len(new_cutouts) // (num_ranks - 1)
        chunks = list(chunk(new_cutouts, chunk_size))
        
        # Tune new cutouts
        chunk = chunks[get_local_rank()]
        for node_hash in chunk:
            state_id, node_id, label = hash_groups[node_hash][0]
            state = self._tuner._sdfg.node(state_id)
            node = state.node(node_id)

            results = self._tuner.evaluate(state=state, node=node, **kwargs)

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
