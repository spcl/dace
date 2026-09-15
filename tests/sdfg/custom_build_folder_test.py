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


@dace.program
def customprog2(A: dace.float64[20]):
    return A * 2


@pytest.fixture
def unlaunched(monkeypatch):
    """Drop the rank and cache settings the surrounding environment exports, which override config."""
    for var in sdfg_module.LAUNCHER_RANK_VARS:
        monkeypatch.delenv(var, raising=False)
    for var in ('DACE_cache', 'DACE_cache_distaware', 'DACE_default_build_folder'):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_default_build_folder(unlaunched):
    """Tests if the `default_build_folder` configuration key is respected.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with dace.config.set_temporary('default_build_folder', value=tmpdir), dace.config.set_temporary('cache',
                                                                                                        value='single'):
            # Ensure build folder matches
            sdfg = customprog.to_sdfg()
            assert str(sdfg.build_folder).startswith(tmpdir)
            assert str(sdfg.build_folder).endswith("/single_cache")

            # Ensure that `build_folder` is not serialized if it was not specified.
            json_dump = sdfg.to_json()
            assert 'build_folder' not in json_dump['attributes']

            sdfg_restore = dace.SDFG.from_json(json_dump)
            assert sdfg_restore._build_folder is None

            csdfg = sdfg.compile()
            assert sdfg._build_folder is None

            # Ensure files were generated in the right folder
            sdfg_dump_path = os.path.join(sdfg.build_folder, 'program.sdfgz')
            assert os.path.isfile(sdfg_dump_path)

            # Because the build folder was explicitly set during compilation, it should be dumped.
            assert csdfg._sdfg._build_folder is not None
            assert csdfg._sdfg._build_folder == sdfg.build_folder

            # Also test if it is was stored in the dump.
            prog_sdfg = dace.SDFG.from_file(sdfg_dump_path)
            assert prog_sdfg._build_folder == sdfg.build_folder

            # Ensure file is closed so it can be deleted
            del csdfg


def test_explicitly_set_build_folder(unlaunched):
    with tempfile.TemporaryDirectory() as tmpdir_def, tempfile.TemporaryDirectory() as tmpdir_used:
        with dace.config.set_temporary('default_build_folder',
                                       value=tmpdir_def), dace.config.set_temporary('cache', value='single'):

            # Ensure build folder matches
            sdfg = customprog.to_sdfg()
            sdfg.build_folder = tmpdir_used

            # Because the build folder was set explicitly it is used exactly, i.e. the `cache` mode is ignored.
            assert str(sdfg.build_folder) == tmpdir_used
            assert str(sdfg._build_folder) == tmpdir_used

            # Ensure that `build_folder` is serialized because it was set explicitly.
            json_dump = sdfg.to_json()
            assert json_dump['attributes']['build_folder'] == tmpdir_used

            sdfg_restore = dace.SDFG.from_json(json_dump)
            assert sdfg_restore._build_folder == tmpdir_used


@pytest.mark.parametrize('rank_var', sdfg_module.LAUNCHER_RANK_VARS)
def test_distaware_gives_each_rank_its_own_cache_root(unlaunched, rank_var):
    """Ranks that each compile would otherwise build into one folder and load a half-written .so."""
    unlaunched.setenv('DACE_cache_distaware', '1')

    unlaunched.setenv(rank_var, '0')
    rank0 = sdfg_module.build_folder_root()
    unlaunched.setenv(rank_var, '1')

    assert sdfg_module.build_folder_root() != rank0


@pytest.mark.parametrize('cache_mode', ['name', 'hash', 'unique', 'single'])
def test_every_cache_mode_builds_under_the_rank_root(unlaunched, cache_mode, tmp_path):
    """Splitting the root rather than the SDFG name separates the ranks in every mode."""
    unlaunched.setenv('DACE_default_build_folder', str(tmp_path))
    sdfg = dace.SDFG('rankprobe')
    unlaunched.setenv('SLURM_PROCID', '3')

    with dace.config.set_temporary('cache', value=cache_mode):
        # distaware defaults on: every mode's root gets the rank suffix.
        unlaunched.setenv('DACE_cache_distaware', '1')
        ranked = sdfg.build_folder
        assert os.path.dirname(ranked) == f'{tmp_path}_rank3'
        unlaunched.delenv('DACE_cache_distaware')

        # Turning distaware off is how a caller opts back into the old shared root.
        with dace.config.set_temporary('cache_distaware', value=False):
            assert sdfg.build_folder != ranked
            assert os.path.dirname(sdfg.build_folder) == str(tmp_path)
            assert os.path.basename(sdfg.build_folder) == os.path.basename(ranked)


def test_ranks_share_a_build_folder_when_distaware_is_off(unlaunched):
    """Turning distaware off restores the old default: distributed_compile has rank 0 build
    where every other rank looks."""
    with dace.config.set_temporary('cache_distaware', value=False):
        sdfg = dace.SDFG('rankprobe')

        unlaunched.setenv('OMPI_COMM_WORLD_RANK', '0')
        rank0 = sdfg.build_folder
        unlaunched.setenv('OMPI_COMM_WORLD_RANK', '1')

        assert sdfg.build_folder == rank0


def test_ranks_do_not_share_a_build_folder_by_default(unlaunched, tmp_path):
    """distaware defaults on: ranks that each compile must not land in one folder."""
    unlaunched.setenv('DACE_default_build_folder', str(tmp_path))
    unlaunched.setenv('DACE_cache', 'name')  # pin the leaf naming policy so the exact path holds
    sdfg = dace.SDFG('rankprobe')

    unlaunched.setenv('OMPI_COMM_WORLD_RANK', '0')
    rank0 = sdfg.build_folder
    unlaunched.setenv('OMPI_COMM_WORLD_RANK', '1')
    rank1 = sdfg.build_folder

    assert rank0 == os.path.join(f'{tmp_path}_rank0', 'rankprobe')
    assert rank1 == os.path.join(f'{tmp_path}_rank1', 'rankprobe')
    assert rank0 != rank1


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


def test_hash_cache_is_stable_across_identical_builds(unlaunched, tmp_path):
    """`cache: hash` names the build folder from the SDFG's contents, so a rebuild of the SAME
    program must land in the SAME folder -- otherwise every run is a cache miss and old build
    folders pile up forever.

    `to_json()` embeds a fresh `uuid4` `guid` on every SDFG/state/node/edge construction
    (`generate_element_id`, dace/sdfg/graph.py), so two parses of one unchanged program produce
    two different JSON strings even in the same process with no hash randomization involved --
    the guids are random regardless of `PYTHONHASHSEED`. `hash_sdfg()` already strips `guid`
    (and other derived/non-identity keys) before hashing.
    """
    unlaunched.setenv('DACE_default_build_folder', str(tmp_path))
    with dace.config.set_temporary('cache', value='hash'):
        sdfg1 = customprog.to_sdfg(simplify=False)
        sdfg2 = customprog.to_sdfg(simplify=False)

        # Confirms the reproducer is actually exercising the instability, not a vacuous check.
        assert sdfg1.to_json() != sdfg2.to_json(), 'guids no longer vary between builds -- update this reproducer'
        assert sdfg1.hash_sdfg() == sdfg2.hash_sdfg()
        assert sdfg1.build_folder == sdfg2.build_folder


def test_hash_cache_differs_for_a_different_program(unlaunched, tmp_path):
    """The flip side of the stability check: a cache key that collided across genuinely different
    SDFGs would serve a stale binary, which is worse than a miss. Names are pinned equal so the
    only thing left to tell the folders apart is the hash component itself.
    """
    unlaunched.setenv('DACE_default_build_folder', str(tmp_path))
    with dace.config.set_temporary('cache', value='hash'):
        sdfg1 = customprog.to_sdfg(simplify=False)
        sdfg2 = customprog2.to_sdfg(simplify=False)
        sdfg1.name = sdfg2.name = 'probe'

        assert sdfg1.hash_sdfg() != sdfg2.hash_sdfg()
        assert sdfg1.build_folder != sdfg2.build_folder


if __name__ == '__main__':
    print("Must be called using `pytest`.")
