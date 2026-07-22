# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The CMake configure/build subprocesses must not inherit this process's MPI-rank
identity: when DaCe compiles from inside an srun/mpirun-launched process, a child that
inherits PMI_RANK / PMIX_* / OMPI_COMM_WORLD_* and touches a PMI/PMIx client blocks
forever in its init, hanging ``cmake`` with defunct children. ``_build_subprocess_env``
drops exactly those variables and preserves everything else."""
import os

from dace.codegen import compiler


def test_build_subprocess_env_strips_mpi_rank_vars(monkeypatch):
    # A representative rank-identity var from every launcher family we handle.
    stripped = {
        'PMI_RANK': '3',
        'PMI_SIZE': '8',
        'PMI_FD': '7',
        'PMIX_RANK': '3',
        'PMIX_NAMESPACE': 'slurm.pmix.123.0',
        'PMIX_SERVER_URI21': 'pmix-server:...',
        'OMPI_COMM_WORLD_RANK': '3',
        'OMPI_COMM_WORLD_LOCAL_RANK': '1',
        'OMPI_UNIVERSE_SIZE': '8',
        'MV2_COMM_WORLD_RANK': '3',
        'MPI_LOCALRANKID': '1',
        'MPI_LOCALNRANKS': '4',
        'SLURM_PROCID': '3',
        'SLURM_LOCALID': '1',
    }
    # Things a build genuinely needs, plus a lookalike that must NOT be stripped.
    preserved = {
        'PATH': os.environ.get('PATH', '/usr/bin'),
        'OMPI_MCA_pml': 'ob1',  # MCA tuning, not a rank id -- keep
        'SLURM_JOB_ID': '12345',  # job-level, not per-rank -- keep
        'PMIXER_HOME': '/opt/pmixer',  # starts with 'PMIX' letters but not the 'PMIX_' prefix -- keep
    }
    for k, v in {**stripped, **preserved}.items():
        monkeypatch.setenv(k, v)

    env = compiler._build_subprocess_env()

    for k in stripped:
        assert k not in env, f'{k} must be stripped from the build environment'
    for k, v in preserved.items():
        assert env.get(k) == v, f'{k} must be preserved in the build environment'


if __name__ == '__main__':
    import pytest
    raise SystemExit(pytest.main([__file__, '-v']))
