# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import os
import pytest
import tempfile

from dace.sdfg import sdfg as sdfg_module
from dace.sdfg import utils as sdfg_utils


@dace.program
def customprog(A: dace.float64[20]):
    return A + 1


def test_custom_build_folder():
    with tempfile.TemporaryDirectory() as tmpdir:
        with dace.config.set_temporary('default_build_folder', value=tmpdir):
            # Ensure build folder matches
            sdfg = customprog.to_sdfg()
            assert tmpdir in sdfg.build_folder
            csdfg = sdfg.compile()

            # Ensure files were generated in the right folder
            assert os.path.isfile(os.path.join(sdfg.build_folder, 'program.sdfgz'))

            # Ensure file is closed so it can be deleted
            del csdfg


@pytest.fixture
def unlaunched(monkeypatch):
    """Drop the rank and cache settings the surrounding environment exports, which override config."""
    for var in sdfg_module.LAUNCHER_RANK_VARS:
        monkeypatch.delenv(var, raising=False)
    for var in ('DACE_cache', 'DACE_cache_distaware', 'DACE_default_build_folder'):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


@pytest.mark.parametrize('rank_var', sdfg_module.LAUNCHER_RANK_VARS)
def test_distaware_gives_each_rank_its_own_cache_root(unlaunched, rank_var):
    """Ranks that each compile would otherwise build into one folder and load a half-written .so."""
    unlaunched.setenv('DACE_cache_distaware', '1')

    unlaunched.setenv(rank_var, '0')
    rank0 = sdfg_module.build_folder_root()
    unlaunched.setenv(rank_var, '1')

    assert sdfg_module.build_folder_root() != rank0


@pytest.mark.parametrize('cache_mode', ['name', 'hash', 'unique', 'single'])
def test_every_cache_mode_builds_under_the_rank_root(unlaunched, cache_mode):
    """Splitting the root rather than the SDFG name separates the ranks in every mode."""
    sdfg = dace.SDFG('rankprobe')
    unlaunched.setenv('SLURM_PROCID', '3')

    with dace.config.set_temporary('cache', value=cache_mode):
        unlaunched.setenv('DACE_cache_distaware', '1')
        ranked = sdfg.build_folder
        unlaunched.delenv('DACE_cache_distaware')

        assert sdfg.build_folder != ranked


def test_ranks_share_a_build_folder_unless_asked_otherwise(unlaunched):
    """The default has to stay: distributed_compile has rank 0 build where every other rank looks."""
    sdfg = dace.SDFG('rankprobe')

    unlaunched.setenv('OMPI_COMM_WORLD_RANK', '0')
    rank0 = sdfg.build_folder
    unlaunched.setenv('OMPI_COMM_WORLD_RANK', '1')

    assert sdfg.build_folder == rank0


def test_a_process_no_launcher_started_keeps_its_folder(unlaunched):
    """No launcher is not rank 0: a lone process keeps the folder it always had, distaware or not."""
    unlaunched.setenv('DACE_cache_distaware', '1')

    assert sdfg_module.build_folder_root() == dace.Config.get('default_build_folder')


class OneRankOfAJob:
    """Stands in for an mpi4py communicator, with the ranks taking their turn in this one process."""

    def __init__(self, rank: int):
        self.rank = rank
        self.broadcast = None

    def Get_rank(self) -> int:
        return self.rank

    def bcast(self, value, root: int = 0):
        if self.rank == root:
            self.broadcast = value
        return self.broadcast

    def Barrier(self):
        pass


def test_distributed_compile_puts_every_rank_in_rank_0_folder(unlaunched, tmp_path):
    """Only rank 0 builds, so the others must look where it built and not where they would have."""
    unlaunched.setenv('DACE_default_build_folder', str(tmp_path))
    unlaunched.setenv('DACE_cache_distaware', '1')

    unlaunched.setenv('OMPI_COMM_WORLD_RANK', '0')
    builder = OneRankOfAJob(0)
    csdfg = sdfg_utils.distributed_compile(customprog.to_sdfg(), builder)
    del csdfg  # Close the library, so the loading rank below opens it fresh

    unlaunched.setenv('OMPI_COMM_WORLD_RANK', '1')
    loader = OneRankOfAJob(1)
    loader.broadcast = builder.broadcast
    sdfg = customprog.to_sdfg()
    assert sdfg.build_folder != builder.broadcast, "rank 1 was looking in rank 0's folder regardless"

    csdfg = sdfg_utils.distributed_compile(sdfg, loader)

    assert sdfg.build_folder == builder.broadcast
    del csdfg

    # A rank that only loads is free to hold no SDFG at all, as tests/library/mpi does.
    csdfg = sdfg_utils.distributed_compile(None, loader)
    del csdfg


if __name__ == '__main__':
    test_custom_build_folder()
