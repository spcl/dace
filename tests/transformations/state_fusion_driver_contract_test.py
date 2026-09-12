# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The contract ``sdfg.apply_transformations_repeated([StateFusion*])`` relies on: a match that
``can_be_applied`` accepted must be APPLIED, so the driver's fixpoint loop terminates.

The driver counts an application whenever it calls ``apply``, whether or not the graph changed.
An ``apply`` that mutates nothing therefore does not merely waste a step -- the same pair is
re-matched on the next sweep, forever. A hanging optimizer is what that looks like from outside,
so every assertion here is on ONE step (progress) plus a CAPPED fixpoint, never on the unbounded
driver: an unfixed tree must fail these, not hang in them.

None of these SDFGs is compiled; the whole file is transformation-level.
"""
import pytest

import dace
from dace import Memlet
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import SDFGState
from dace.transformation.interstate import StateFusion, StateFusionExtended

XFORMS = [StateFusion, StateFusionExtended]
XFORM_IDS = ['plain', 'extended']
#: Well past any of these SDFGs' block count, so reaching it means "did not converge".
STEP_CAP = 50


def block_count(sdfg: dace.SDFG) -> int:
    """Blocks across the whole region tree -- what a fusion has to strictly reduce."""
    return sum(region.number_of_nodes() for region in sdfg.all_control_flow_regions(recursive=True))


def fuse_to_fixpoint(sdfg: dace.SDFG, xform, permissive: bool) -> int:
    """Apply ``xform`` one match at a time until none is left, asserting the driver contract.

    Every applied step must strictly reduce the block count (that is what "the match was
    applied" means for state fusion, which always removes one of the two states), and the
    fixpoint must be reached within ``STEP_CAP`` steps.
    """
    applied = 0
    for _ in range(STEP_CAP):
        before = block_count(sdfg)
        step = sdfg.apply_transformations([xform], permissive=permissive, validate=False, validate_all=False)
        if step == 0:
            return applied
        applied += step
        after = block_count(sdfg)
        assert after < before, (f'{xform.__name__} reported {step} application(s) but left the block count at '
                                f'{after}; apply_transformations_repeated re-matches this pair forever')
    pytest.fail(f'{xform.__name__} did not converge within {STEP_CAP} steps')


def write(state: SDFGState, name: str, index: str = '0', value: str = '1.0'):
    """A trivial ``tasklet -> AccessNode`` write, so the state is not empty."""
    an = state.add_access(name)
    t = state.add_tasklet(f't_{name}_{state.label}_{value[0]}', {}, {'o': None}, f'o = {value}')
    state.add_edge(t, 'o', an, None, Memlet(f'{name}[{index}]'))
    return an


def base(name: str) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [8], dace.float64)
    sdfg.add_array('B', [8], dace.float64)
    sdfg.add_array('C', [8], dace.float64)
    sdfg.add_scalar('s', dace.float64, transient=True)
    sdfg.add_symbol('p', dace.int32)
    return sdfg


# --------------------------------------------------------------------------- shapes that only
# --------------------------------------------------------------------------- match permissively
def make_join() -> dace.SDFG:
    """A diamond: the join block has two incoming edges, which only permissive matching allows."""
    sdfg = base('permissive_join')
    s0 = sdfg.add_state('s0', is_start_block=True)
    s1 = sdfg.add_state('s1')
    s2 = sdfg.add_state('s2')
    s3 = sdfg.add_state('s3')
    sdfg.add_edge(s0, s1, InterstateEdge(condition='p < 4'))
    sdfg.add_edge(s0, s2, InterstateEdge(condition='p >= 4'))
    sdfg.add_edge(s1, s3, InterstateEdge())
    sdfg.add_edge(s2, s3, InterstateEdge())
    write(s1, 'A')
    write(s2, 'B')
    return sdfg


def make_back_edge() -> dace.SDFG:
    """A state that loops back to itself: the second state keeps two incoming edges."""
    sdfg = base('permissive_self_loop')
    s0 = sdfg.add_state('s0', is_start_block=True)
    s1 = sdfg.add_state('s1')
    sdfg.add_edge(s0, s1, InterstateEdge())
    sdfg.add_edge(s1, s1, InterstateEdge(condition='p < 4'))
    write(s0, 'A')
    write(s1, 'B')
    return sdfg


def make_wcr_after_init() -> dace.SDFG:
    """First state initializes ``A``, second WCR-accumulates into it -- refused in strict mode
    because the seed read is a real RAW dependency, matched under permissive."""
    sdfg = base('permissive_wcr')
    s1 = sdfg.add_state('s1', is_start_block=True)
    s2 = sdfg.add_state('s2')
    sdfg.add_edge(s1, s2, InterstateEdge())

    me, mx = s1.add_map('init', dict(i='0:8'))
    t = s1.add_tasklet('zero', {}, {'o': None}, 'o = 0.0')
    s1.add_nedge(me, t, Memlet())
    s1.add_memlet_path(t, mx, s1.add_access('A'), src_conn='o', memlet=Memlet('A[i]'))

    me2, mx2 = s2.add_map('acc', dict(i='0:8'))
    t2 = s2.add_tasklet('add', {'b': None}, {'o': None}, 'o = b')
    s2.add_memlet_path(s2.add_access('B'), me2, t2, dst_conn='b', memlet=Memlet('B[i]'))
    s2.add_memlet_path(t2, mx2, s2.add_access('A'), src_conn='o', memlet=Memlet('A[0]', wcr='lambda a, b: a + b'))
    return sdfg


def make_reused_scalar() -> dace.SDFG:
    """Two top-level producers of one transient scalar in the first state, and the second state
    writes it again -- the reused-transient ambiguity guard refuses this in strict mode only."""
    sdfg = base('permissive_scalar')
    s1 = sdfg.add_state('s1', is_start_block=True)
    s2 = sdfg.add_state('s2')
    sdfg.add_edge(s1, s2, InterstateEdge())
    write(s1, 's', '0', '1.0')
    write(s1, 's', '0', '2.0')
    write(s2, 's', '0', '3.0')
    src, dst = s2.add_access('s'), s2.add_access('A')
    s2.add_edge(src, None, dst, None, Memlet('s[0]'))
    return sdfg


PERMISSIVE_ONLY = [make_join, make_back_edge, make_wcr_after_init, make_reused_scalar]
PERMISSIVE_IDS = ['join', 'back_edge', 'wcr_after_init', 'reused_scalar']


@pytest.mark.parametrize('xform', XFORMS, ids=XFORM_IDS)
@pytest.mark.parametrize('permissive', [False, True], ids=['strict', 'permissive'])
@pytest.mark.parametrize('factory', PERMISSIVE_ONLY, ids=PERMISSIVE_IDS)
def test_every_application_makes_progress(xform, permissive, factory):
    """The contract itself: an accepted match is applied, so the fixpoint is reached.

    ``StateFusionExtended.apply`` used to replay its ``can_be_applied`` recorder in the STRICT
    mode regardless of the mode the match was made in, so a permissive-only match declined
    inside ``apply``, mutated nothing, and spun the driver.
    """
    sdfg = factory()
    sdfg.validate()
    fuse_to_fixpoint(sdfg, xform, permissive)
    sdfg.validate()


@pytest.mark.parametrize('xform', XFORMS, ids=XFORM_IDS)
@pytest.mark.parametrize('factory', PERMISSIVE_ONLY, ids=PERMISSIVE_IDS)
def test_accepted_match_is_not_declined_by_apply(xform, factory):
    """Same contract, stated on ONE step: what the matcher accepts, ``apply`` must carry out."""
    sdfg = factory()
    before = block_count(sdfg)
    applied = sdfg.apply_transformations([xform], permissive=True, validate=False, validate_all=False)
    if applied:
        assert block_count(sdfg) < before


# --------------------------------------------------------------------------- driver-level sweep
@pytest.mark.parametrize('xform', XFORMS, ids=XFORM_IDS)
@pytest.mark.parametrize('permissive', [False, True], ids=['strict', 'permissive'])
def test_repeated_driver_terminates_and_validates(xform, permissive):
    """The real entry point, on a shape mixing every construct the two states can hold.

    Safe to call unbounded only because ``test_every_application_makes_progress`` above already
    pins the progress invariant -- this one is the end-to-end check that the driver returns and
    hands back a valid SDFG.
    """
    sdfg = base('driver_sweep')
    sdfg.add_transient('T', [8], dace.float64)
    s0 = sdfg.add_state('s0', is_start_block=True)
    s1 = sdfg.add_state('s1')
    s2 = sdfg.add_state('s2')
    s3 = sdfg.add_state('s3')
    sdfg.add_edge(s0, s1, InterstateEdge())
    sdfg.add_edge(s1, s2, InterstateEdge())
    sdfg.add_edge(s2, s3, InterstateEdge())

    me, mx = s0.add_map('produce', dict(i='0:8'))
    t = s0.add_tasklet('p', {'a': None}, {'o': None}, 'o = a + 1.0')
    s0.add_memlet_path(s0.add_access('A'), me, t, dst_conn='a', memlet=Memlet('A[i]'))
    s0.add_memlet_path(t, mx, s0.add_access('T'), src_conn='o', memlet=Memlet('T[i]'))

    s1.add_edge(s1.add_access('T'), None, s1.add_access('B'), None, Memlet('T[0:8]'))
    write(s2, 'C', '0')
    write(s2, 'A', '1')
    s3.add_edge(s3.add_access('C'), None, s3.add_access('A'), None, Memlet('C[0:8]'))

    sdfg.validate()
    sdfg.apply_transformations_repeated([xform], permissive=permissive, validate=False, validate_all=False)
    sdfg.validate()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
