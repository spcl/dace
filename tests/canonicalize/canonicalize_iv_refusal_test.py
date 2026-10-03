# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shapes that LOOK like induction variables and must NOT be substituted.

``k = k + 1`` and ``sum = sum + a[i]`` are the same shape to a structural matcher. Only the STEP
tells them apart, and getting it wrong is a silent miscompile rather than a missed optimization:
a closed form ``k0 + step * trip`` is simply false when ``step`` reads memory that changes every
iteration. ``induction_variable_substitution.step_is_loop_invariant`` is the discriminator; these
tests hold it to its contract from the outside.

Two families are covered:

* DATA-DEPENDENT steps -- reductions, which belong to ``LoopToReduce`` / ``LoopToScan``. In the
  dataflow they arrive as ``__out = __in1 + __in2`` with a second input CONNECTOR, and a connector
  name is never an SDFG symbol.
* CONDITIONAL steps -- ``if c: j += 1`` makes ``j`` a COUNT of true predicates, which is not an
  affine function of the loop indices at all, so no closed form exists.

``s124`` is deliberately NOT in the conditional list: its increment happens identically in EVERY
branch, so it advances unconditionally and IS affine. ``_hoist_branch_uniform_iv`` closes it and
``canonicalize_iv_chain_test`` asserts it does.
"""

import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import induction_variable_substitution as ivs
from dace.transformation.passes.canonicalize.induction_variable_substitution import extract_tasklet_iv
from dace.transformation.passes.canonicalize.pipeline import canonicalize

from tests.corpus.tsvc import tsvc
from tests.corpus.tsvc.tsvc_numpy import REFERENCES

#: Every entry point the IVS fixed point can substitute through.
IVS_ENTRY_POINTS = ('_try_substitute', 'try_substitute_use_site_iv', '_try_substitute_iedge_iv',
                    '_try_substitute_derived_symbol', '_hoist_branch_uniform_iv')


def canonicalize_recording_ivs(name, monkeypatch):
    """Canonicalize corpus kernel ``name``, recording which IVS entry points substituted.

    :returns: ``(fired, values_match)`` -- the entry points that reported a substitution, and
              whether the compiled kernel still matches its numpy reference.
    """
    fired = []

    def recording(entry_point, base):

        def wrapper(parent, loop, sdfg, sdfg_free_symbols):
            out = base(parent, loop, sdfg, sdfg_free_symbols)
            if out:
                fired.append(entry_point)
            return out

        return wrapper

    for entry_point in IVS_ENTRY_POINTS:
        monkeypatch.setattr(ivs, entry_point, recording(entry_point, vars(ivs)[entry_point]))

    kernel = [k for k in tsvc.collect() if k.name == name][0]
    sdfg = tsvc.to_sdfg(kernel, 'refuse_' + name, simplify=True)
    canonicalize(sdfg, validate=True, peel_limit=4)

    arrays, call_kwargs = tsvc.make_inputs(kernel)
    ref = {n: a.copy() for n, a in arrays.items()}
    got = {n: a.copy() for n, a in arrays.items()}
    REFERENCES[name](**ref, **call_kwargs)
    sdfg.compile()(**got, **call_kwargs)
    values_match = all(
        np.allclose(ref[n], got[n], equal_nan=True) for n, arr in arrays.items()
        if not np.issubdtype(arr.dtype, np.integer))
    return fired, values_match


#: (kernel, the step expression that makes it a reduction rather than an IV).
DATA_DEPENDENT_STEPS = [
    ('s3112_d_single', 'a[i]'),
    ('s312_d_single', 'a[i]'),
    ('vsumr_d_single', 'a[i]'),
    ('s352_d_single', 'a[i] * b[i]'),
    ('s4115_d_single', 'a[i] * b[ip[i]]'),
    ('s4116_d_single', 'a[off] * aa[j - 1, ip[i]]'),
]

#: Guarded increments: the accumulator counts true predicates, so it is not affine in the indices.
CONDITIONAL_STEPS = ['s123_d_single', 's3111_d_single', 's341_d_single', 's342_d_single', 's343_d_single']


@pytest.mark.parametrize('name,step', DATA_DEPENDENT_STEPS, ids=[n.split('_')[0] for n, _ in DATA_DEPENDENT_STEPS])
def test_data_dependent_step_is_not_an_induction_variable(name, step, monkeypatch):
    """A step that READS an array has no closed form; IVS must leave it to reduction lifting."""
    fired, values_match = canonicalize_recording_ivs(name, monkeypatch)
    assert not fired, (f"{name}: step is `{step}` (data-dependent), so it is a REDUCTION, not an "
                       f"induction variable -- IVS must not substitute it, but {sorted(set(fired))} did")
    assert values_match, f"{name}: values diverged from the numpy reference"


@pytest.mark.parametrize('name', CONDITIONAL_STEPS, ids=[n.split('_')[0] for n in CONDITIONAL_STEPS])
def test_conditional_step_is_not_an_induction_variable(name, monkeypatch):
    """A guarded increment counts true predicates, so no affine closed form exists."""
    fired, values_match = canonicalize_recording_ivs(name, monkeypatch)
    assert not fired, (f"{name}: the increment is CONDITIONAL, so the accumulator is a count of true "
                       f"predicates and has no closed form -- but {sorted(set(fired))} substituted one")
    assert values_match, f"{name}: values diverged from the numpy reference"


def build_accumulator_loop(data_dependent_step: bool, n: int = 8):
    """``for i in range(n): acc[0] = acc[0] + <step>`` with ``<step>`` a literal or ``a[i]``.

    The two SDFGs are identical but for where the step comes from, which is exactly the
    distinction :func:`step_is_loop_invariant` exists to draw.

    :returns: ``(sdfg, loop, body, tasklet)``.
    """
    sdfg = dace.SDFG('iv_step_' + ('data' if data_dependent_step else 'const'))
    sdfg.add_array('acc', [1], dace.float64)
    sdfg.add_array('a', [n], dace.float64)
    init = sdfg.add_state('init', is_start_block=True)
    loop = LoopRegion('loop', condition_expr=f'i < {n}', loop_var='i', initialize_expr='i = 0', update_expr='i = i + 1')
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge())
    body = loop.add_state('body', is_start_block=True)
    read, write = body.add_access('acc'), body.add_access('acc')
    # Connectors as dicts, not sets: a set makes DaCe warn about nondeterministic ordering.
    if data_dependent_step:
        tasklet = body.add_tasklet('step', {'__in1': None, '__in2': None}, {'__out': None}, '__out = __in1 + __in2')
        body.add_edge(body.add_access('a'), None, tasklet, '__in2', dace.Memlet('a[i]'))
    else:
        tasklet = body.add_tasklet('step', {'__in1': None}, {'__out': None}, '__out = __in1 + 2.0')
    body.add_edge(read, None, tasklet, '__in1', dace.Memlet('acc[0]'))
    body.add_edge(tasklet, '__out', write, None, dace.Memlet('acc[0]'))
    return sdfg, loop, body, tasklet


@pytest.mark.parametrize('data_dependent_step,expect_match', [(False, True), (True, False)],
                         ids=['constant-step-is-an-IV', 'data-step-is-a-reduction'])
def test_extract_tasklet_iv_separates_a_counter_from_a_reduction(data_dependent_step, expect_match):
    """The dataflow half of the discriminator, exercised directly.

    ``acc += 2.0`` and ``acc += a[i]`` differ only in a second input CONNECTOR. The literal one
    is an induction variable with the closed form ``acc0 + 2.0 * trip``; the other is a reduction
    with no closed form. Driven through :func:`extract_tasklet_iv` rather than end-to-end,
    because relaxing this check is what a future edit would plausibly get wrong.
    """
    sdfg, loop, body, tasklet = build_accumulator_loop(data_dependent_step)
    matched = extract_tasklet_iv(tasklet, body, loop, sdfg, sdfg.free_symbols) is not None
    assert matched is expect_match, ('a data-dependent step must NOT match the IV shape'
                                     if data_dependent_step else 'a literal step must match the IV shape')


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
