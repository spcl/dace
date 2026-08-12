# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
"""
Root pytest configuration file.
"""
import os
import subprocess

import pytest


def parse_worker_index(worker_id: str) -> int:
    """Parse the digits in a worker id or MPI rank string (e.g. 'gw3' -> 3, '12' -> 12).

    Falls back to 0 for text with no digits (e.g. the non-xdist 'master' worker id).
    """
    digits = ''.join(char for char in worker_id if char.isdigit())
    return int(digits) if digits else 0


def list_cuda_devices() -> list:
    """Enumerate CUDA device indices via `nvidia-smi -L`, without touching CUDA in this process."""
    try:
        result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    device_lines = [line for line in result.stdout.splitlines() if line.startswith('GPU ')]
    return [str(index) for index in range(len(device_lines))]


def pick_gpu_worker_device(worker_id: str, device_pool: list) -> str:
    """Round-robin a worker id (xdist worker or MPI rank) onto one device of device_pool."""
    index = parse_worker_index(worker_id)
    return device_pool[index % len(device_pool)]


def resolve_worker_id() -> str:
    """Id that places this process in a device rotation.

    The pytest-xdist worker id if present, else the first set MPI/job-launcher rank env var
    (dace.sdfg.sdfg.LAUNCHER_RANK_VARS, the same list dace itself reads for per-rank build
    folders), else '' if this process is neither an xdist worker nor an MPI rank.
    """
    xdist_worker = os.environ.get('PYTEST_XDIST_WORKER')
    if xdist_worker is not None:
        return xdist_worker
    from dace.sdfg.sdfg import LAUNCHER_RANK_VARS
    for var in LAUNCHER_RANK_VARS:
        rank = os.environ.get(var)
        if rank:
            return rank
    return ''


def pin_worker_to_gpu() -> None:
    """Pin this worker process (pytest-xdist or an MPI-launched rank) to a single GPU.

    Without this, every worker/rank sees all GPUs and piles its CUDA context onto device 0, which
    is the root cause of flaky 'invalid device ordinal' failures under high worker/rank counts.
    """
    worker_id = resolve_worker_id()
    if not worker_id:
        return
    preset = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    device_pool = [entry.strip() for entry in preset.split(',') if entry.strip()] if preset.strip() \
        else list_cuda_devices()
    if len(device_pool) <= 1:
        return
    os.environ['CUDA_VISIBLE_DEVICES'] = pick_gpu_worker_device(worker_id, device_pool)


def pytest_configure(config: pytest.Config) -> None:
    pin_worker_to_gpu()
