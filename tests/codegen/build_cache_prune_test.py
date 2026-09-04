# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The machine-global build cache must stay bounded.

Its default root is ``/dev/shm``, i.e. RAM, and a PCH entry is ~110 MB. Without eviction a
long-lived session that keeps producing fresh signatures -- a corpus sweep, a flag search --
fills it and the machine runs out of memory. These pin that :func:`~dace.codegen.build_cache.prune`
bounds it and evicts by LAST USE, not by creation order.
"""
import os

import pytest

from dace.codegen import build_cache


def make_entry(root, name, kilobytes, atime):
    """A cache entry directory of a known size, last used at ``atime``."""
    path = os.path.join(root, name)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'payload'), 'wb') as fp:
        fp.write(b'\0' * (kilobytes * 1024))
    os.utime(path, (atime, atime))
    return path


def test_budget_honours_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv('DACE_BUILD_CACHE_MAX_MB', '7')
    assert build_cache.budget(str(tmp_path)) == 7 * 1024**2


def test_budget_is_a_fraction_of_the_filesystem(tmp_path, monkeypatch):
    """With no override the budget tracks the backing filesystem, so /dev/shm on a small machine
    gets a small cache rather than the same constant a large node would get."""
    monkeypatch.delenv('DACE_BUILD_CACHE_MAX_MB', raising=False)
    stat = os.statvfs(str(tmp_path))
    assert build_cache.budget(str(tmp_path)) == int(stat.f_blocks * stat.f_frsize * build_cache.CACHE_FRACTION)


def test_prune_evicts_until_under_budget(tmp_path, monkeypatch):
    monkeypatch.setenv('DACE_BUILD_CACHE_MAX_MB', '1')
    root = str(tmp_path)
    for index in range(4):  # 4 x 512 KiB = 2 MiB against a 1 MiB budget
        make_entry(root, f'entry{index}', 512, atime=1000 + index)

    build_cache.prune(root)

    surviving = sorted(os.listdir(root))
    assert surviving == ['entry2', 'entry3'], 'oldest-used entries must go first'
    assert sum(build_cache.entry_size(os.path.join(root, e)) for e in surviving) <= 1024**2


def test_prune_evicts_by_last_use_not_creation(tmp_path, monkeypatch):
    """A frequently reused entry must outlive a newer one that is never touched again -- evicting
    by write time would drop exactly the entries earning their keep."""
    monkeypatch.setenv('DACE_BUILD_CACHE_MAX_MB', '1')
    root = str(tmp_path)
    old_but_hot = make_entry(root, 'hot', 512, atime=1000)
    make_entry(root, 'cold', 512, atime=2000)
    make_entry(root, 'coldest', 512, atime=3000)
    build_cache.touch(old_but_hot)  # a cache HIT on the oldest entry

    build_cache.prune(root)

    assert 'hot' in os.listdir(root)
    assert 'cold' not in os.listdir(root)


def test_prune_handles_file_entries(tmp_path, monkeypatch):
    """The ``commands`` cache stores one JSON FILE per key, not a directory."""
    monkeypatch.setenv('DACE_BUILD_CACHE_MAX_MB', '1')
    root = str(tmp_path)
    for index in range(3):
        path = os.path.join(root, f'cmd{index}.json')
        with open(path, 'wb') as fp:
            fp.write(b'\0' * (512 * 1024))
        os.utime(path, (1000 + index, 1000 + index))

    build_cache.prune(root)

    assert sorted(os.listdir(root)) == ['cmd1.json', 'cmd2.json']


def test_prune_keeps_everything_under_budget(tmp_path, monkeypatch):
    monkeypatch.setenv('DACE_BUILD_CACHE_MAX_MB', '64')
    root = str(tmp_path)
    for index in range(3):
        make_entry(root, f'entry{index}', 512, atime=1000 + index)

    build_cache.prune(root)

    assert len(os.listdir(root)) == 3


def test_prune_on_missing_root_is_silent(tmp_path):
    """A cache kind that has never been written yet must not raise."""
    build_cache.prune(os.path.join(str(tmp_path), 'never-created'))


if __name__ == '__main__':
    pytest.main([__file__])
