# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The PBLAS nodes must refuse a state that makes ScaLAPACK answer with wrong numbers.

Two of those states cost a CI day each. ``descinit_`` reports a rejected descriptor in ``info``,
which every expansion here used to drop on the floor -- so a bad descriptor surfaced as a plausible
wrong product rather than as an error. And MPI brought up by a bare ``MPI_Init`` runs at
``MPI_THREAD_SINGLE``, a promise no DaCe process keeps (OpenMP maps, a threaded BLAS); below
``MPI_THREAD_FUNNELED`` Open MPI drops the locking around its shared-memory transport and
ScaLAPACK's panel broadcasts come back corrupted, differently on every call.

These assert the emitted C++, not a runtime outcome: the guards are only worth anything if they
reach the generated program, and reproducing the corruption needs a specific MPI build.
"""
import dace
import pytest

from dace.libraries.pblas.environments import (intel_mkl_mpich, intel_mkl_openmpi, ref_mpich, ref_openmpi)

#: Every ScaLAPACK environment, whichever MPI and BLAS it links.
ENVIRONMENTS = (intel_mkl_mpich.IntelMKLScaLAPACKMPICH, intel_mkl_openmpi.IntelMKLScaLAPACKOpenMPI,
                ref_mpich.ScaLAPACKMPICH, ref_openmpi.ScaLAPACKOpenMPI)


@pytest.mark.parametrize('environment', ENVIRONMENTS, ids=lambda env: env.__name__)
def test_environment_brings_mpi_up_at_a_level_it_can_keep(environment):
    """Left to BLACS, an unlaunched rank's MPI comes up under a bare ``MPI_Init``."""
    init = environment.init_code

    assert 'MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED' in init
    # Ahead of the BLACS calls, or BLACS gets there first with MPI_Init.
    assert init.index('MPI_Init_thread') < init.index('blacs_pinfo')


@pytest.mark.parametrize('environment', ENVIRONMENTS, ids=lambda env: env.__name__)
def test_environment_refuses_a_thread_level_below_funneled(environment):
    """Someone else's bring-up is not ours to fix; refusing to run is all that is left."""
    init = environment.init_code

    assert 'MPI_Query_thread' in init
    assert 'MPI_THREAD_FUNNELED' in init
    assert 'MPI_Abort' in init
    # Before the grid, not after it: a corrupted broadcast is what the check exists to prevent.
    assert init.index('MPI_Query_thread') < init.index('gridinit')


def expanded_tasklet_code(program, implementation):
    """The C++ of every tasklet ``program``'s library nodes expand to under ``implementation``."""
    with dace.config.set_temporary('library', 'pblas', 'default_implementation', value=implementation):
        sdfg = program.to_sdfg(simplify=True)
        sdfg.expand_library_nodes()
    return '\n'.join(node.code.as_string for node, _ in sdfg.all_nodes_recursive()
                     if isinstance(node, dace.sdfg.nodes.Tasklet))


@pytest.mark.parametrize('implementation', ('MKLMPICH', 'MKLOpenMPI', 'ReferenceMPICH', 'ReferenceOpenMPI'))
def test_pgemm_checks_every_descinit_info(implementation):
    """A descriptor ``descinit_`` rejected must stop the multiply, not feed it."""
    LMx, LNy, LKx, LKy, Px, Py, GK = (dace.symbol(s, positive=True)
                                      for s in ('LMx', 'LNy', 'LKx', 'LKy', 'Px', 'Py', 'GK'))

    @dace.program
    def pdgemm(A: dace.float64[LMx, LKy], B: dace.float64[LKx, LNy]):
        return dace.distr.MatMult(A, B, (LMx * Px, LNy * Py, GK))

    code = expanded_tasklet_code(pdgemm, implementation)

    for descriptor in ('c', 'a', 'b'):
        assert f'&info_{descriptor}' in code, f'descinit_ writes info_{descriptor} nowhere'
        assert f'info_{descriptor} != 0' in code, f'info_{descriptor} is never read'
    assert 'MPI_Abort' in code


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
