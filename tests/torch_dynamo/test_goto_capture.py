# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Bytecode-level capture of plain Python ``if``/``while`` on tensor data (``cfg/goto_capture``).

The experimental ``CfgDaceBackend`` patches Dynamo's conditional-jump handlers so that data-dependent branches become
``torch.cond`` / ``torch.while_loop`` HOPs instead of graph breaks; the existing importer lowers those to
``ConditionalBlock`` / ``LoopRegion``. Tests assert compile-once behaviour (``compile_count == 1``) across shapes.

The capture consumes whatever bytecode the interpreter produces, so each loop test targets one *instruction pattern*
and uses a source shape that produces it on every supported CPython version (3.10-3.14); ``_assert_patterns`` checks
the pattern with ``dis`` so that a compiler change fails loudly instead of silently testing something else. For
example, CPython 3.10, 3.12 and 3.14 replace a jump to a small exit block (at most 4 instructions ending in a return)
with a copy of that block, so a ``break`` followed only by ``return acc + 1`` is an early return from inside the loop
body in their bytecode; the ``break`` tests therefore keep a longer post-loop tail.
"""
import dis
from typing import Callable, Iterable, List, Set, Tuple

import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")

import torch  # noqa: E402

from dace.frontend.ml.torch.dynamo.cfg import CfgDaceBackend  # noqa: E402
from dace.sdfg.state import ConditionalBlock, LoopRegion  # noqa: E402


@pytest.fixture
def cfg_backend():
    return CfgDaceBackend()


def _check(backend, fn, inputs_per_shape, expected_compiles=1, **compile_kwargs):
    compile_kwargs.setdefault('dynamic', True)
    compile_kwargs.setdefault('fullgraph', True)
    compiled = torch.compile(fn, backend=backend, **compile_kwargs)
    with torch.no_grad():
        for inputs in inputs_per_shape:
            ref = fn(*inputs)
            out = compiled(*inputs)
            torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)
    assert backend.compile_count == expected_compiles, \
        f'expected {expected_compiles} compilation(s), got {backend.compile_count}'
    return compiled


def _regions(backend, kind):
    return [r for r in backend.last_sdfg.all_control_flow_regions(recursive=True) if isinstance(r, kind)]


def _events(backend):
    return backend.capture.kinds()


# ---------------------------------------------------------------------- bytecode instruction patterns
_UNCONDITIONAL_JUMPS = frozenset(
    {'JUMP_FORWARD', 'JUMP_ABSOLUTE', 'JUMP_BACKWARD', 'JUMP_BACKWARD_NO_INTERRUPT', 'JUMP', 'JUMP_NO_INTERRUPT'})
_SCOPE_EXITS = frozenset({'RETURN_VALUE', 'RETURN_CONST', 'RAISE_VARARGS', 'RERAISE'})
_FILLERS = frozenset({'NOP', 'NOT_TAKEN', 'EXTENDED_ARG', 'CACHE'})
#: Opcodes with a jump target on this interpreter (3.10 has absolute and relative jumps, 3.12+ only relative ones).
#: The 3.10 ``SETUP_*`` opcodes only register a handler and fall through.
_JUMPS = frozenset(dis.opname[op] for op in dis.hasjrel + dis.hasjabs if not dis.opname[op].startswith('SETUP_'))


def _bytecode_patterns(fn: Callable) -> Set[str]:
    """
    Classifies the loop-exit instruction patterns of ``fn``'s bytecode, as ``dis`` reports it on the running
    interpreter. The natural loops of all back-edges are merged into one region, so ``fn`` must contain exactly one
    source-level loop.

    - ``'break-jump'``: an edge leaving the loop body lands on an unconditional forward jump to the loop's exit (the
      target of a conditional exit), i.e., a ``break`` that CPython did not replace with a copy of the exit block.
    - ``'early-return'``: the loop exits to at least two different places (after following unconditional jumps) and
      one of them is straight-line code ending in a return (an explicit ``return`` or a copied exit block).
    - ``'multiple-back-edges'``: the loop has more than one back-edge (e.g., ``continue``).
    - ``'single-exit-edge'``: exactly one edge leaves the loop body.

    :param fn: The function to inspect.
    :return: The names of the patterns present.
    """
    insts = list(dis.get_instructions(fn))
    index = {inst.offset: k for k, inst in enumerate(insts)}
    succ: List[List[int]] = []
    for k, inst in enumerate(insts):
        fallthrough = [k + 1] if k + 1 < len(insts) else []
        if inst.opname in _SCOPE_EXITS:
            succ.append([])
        elif inst.opname in _UNCONDITIONAL_JUMPS:
            succ.append([index[inst.argval]])
        elif inst.opname in _JUMPS:
            succ.append(fallthrough + [index[inst.argval]])
        else:
            succ.append(fallthrough)
    preds: List[List[int]] = [[] for _ in insts]
    for u, targets in enumerate(succ):
        for v in targets:
            preds[v].append(u)

    # Natural loop of each back-edge u -> v: v plus every instruction that reaches u without passing through v
    back_edges = [(u, v) for u, targets in enumerate(succ) for v in targets if v <= u]
    body: Set[int] = set()
    for u, v in back_edges:
        loop, stack = {v}, [u]
        while stack:
            x = stack.pop()
            if x not in loop:
                loop.add(x)
                stack.extend(preds[x])
        body |= loop
    exits: List[Tuple[int, int]] = [(u, w) for u in sorted(body) for w in succ[u] if w not in body]

    def skip_fillers(k: int) -> int:
        while insts[k].opname in _FILLERS:
            k += 1
        return k

    def resolve(k: int) -> int:
        for _ in range(len(insts)):
            k = skip_fillers(k)
            if insts[k].opname not in _UNCONDITIONAL_JUMPS or index[insts[k].argval] <= k:
                break
            k = index[insts[k].argval]
        return k

    def ends_in_return(k: int) -> bool:
        while insts[k].opname not in _SCOPE_EXITS:
            if insts[k].opname in _JUMPS or insts[k].opname in _UNCONDITIONAL_JUMPS or k + 1 >= len(insts):
                return False
            k += 1
        return True

    patterns: Set[str] = set()
    conditional_exits = {w for u, w in exits if insts[u].opname in _JUMPS - _UNCONDITIONAL_JUMPS}
    for _, w in exits:
        k = skip_fillers(w)
        if insts[k].opname in _UNCONDITIONAL_JUMPS:
            target = index[insts[k].argval]
            if target > k and target in conditional_exits:
                patterns.add('break-jump')
    destinations = {resolve(w) for _, w in exits}
    if len(destinations) > 1 and any(ends_in_return(d) for d in destinations):
        patterns.add('early-return')
    if len(back_edges) > 1:
        patterns.add('multiple-back-edges')
    if len(exits) == 1:
        patterns.add('single-exit-edge')
    return patterns


def _assert_patterns(fn: Callable, present: Iterable[str] = (), absent: Iterable[str] = ()) -> None:
    """
    Asserts that ``fn``'s bytecode on this interpreter has the instruction patterns the test relies on (see
    :func:`_bytecode_patterns`).

    :param fn: The function under test.
    :param present: Pattern names that must be present.
    :param absent: Pattern names that must not be present.
    """
    patterns = _bytecode_patterns(fn)
    missing = set(present) - patterns
    unexpected = set(absent) & patterns
    assert not missing and not unexpected, (
        f'{fn.__qualname__}: the bytecode no longer has the intended instruction pattern (missing {sorted(missing)}, '
        f'unexpected {sorted(unexpected)}; found {sorted(patterns)}). Change the source shape, not the test '
        f'expectation:\n{dis.Bytecode(fn).dis()}')


_POS = [(torch.rand(3, 4) + 0.5, ), (torch.rand(4, 5) + 0.5, ), (torch.rand(5, 2) * 0.1 + 0.01, )]
_MIXED = [(torch.rand(4, 6) + 0.5, ), (-(torch.rand(5, 7) + 0.5), ), (torch.randn(3, 2), )]


# ---------------------------------------------------------------------- if / else
def _if_else(x):
    y = x * 2
    if y.sum() > 0:
        y = y.sin()
    else:
        y = y.cos()
    return y + 1


@pytest.mark.torch
def test_if_else_both_continuations(cfg_backend):
    _check(cfg_backend, _if_else, _MIXED)
    assert _events(cfg_backend) == ['if']
    assert len(_regions(cfg_backend, ConditionalBlock)) == 1


@pytest.mark.torch
def test_if_without_else(cfg_backend):

    def f(x):
        y = x * 2
        if y.sum() > 0:
            y = y.sin()
        return y + 1

    _check(cfg_backend, f, _MIXED)
    assert len(_regions(cfg_backend, ConditionalBlock)) == 1


@pytest.mark.torch
def test_if_with_int_and_symint_locals(cfg_backend):
    # Live locals that are Python ints / SymInts travel through the cond operands

    def f(x, k):
        y = x * k
        n = x.shape[0]
        if y.sum() > 0:
            y = y.sin() + n
        else:
            y = y.cos() * k
        return y + 1

    _check(cfg_backend, f, [(torch.rand(4, 6) + 0.5, 3), (-(torch.rand(5, 7) + 0.5), 3)])


@pytest.mark.torch
def test_if_elif_else_nested(cfg_backend):

    def f(x):
        y = x * 2
        if y.sum() > 100:
            y = y.sin()
        elif y.sum() > 0:
            if y.mean() > 1:
                y = y.cos()
            else:
                y = y.tan()
        else:
            y = -y
        return y + 1

    _check(cfg_backend, f, [(torch.rand(4, 6) * 20, ), (torch.rand(5, 7) * 0.1, ), (-(torch.rand(3, 2) + 0.5), )])
    assert _events(cfg_backend).count('if') >= 3  # tail duplication traces nested ifs per continuation


@pytest.mark.torch
def test_ternary_and_multiple_returns(cfg_backend):

    def f(x):
        y = x * 2
        z = y.sin() if y.sum() > 0 else y.cos()
        if z.mean() > 10:
            return z
        return z + 1

    _check(cfg_backend, f, _MIXED)


# ---------------------------------------------------------------------- while
def _while(x):
    acc = x
    while acc.sum() < 100:
        acc = acc * 2
    return acc + 1


@pytest.mark.torch
def test_while_tensor_predicate(cfg_backend):
    _assert_patterns(_while, absent=('break-jump', 'early-return'))
    _check(cfg_backend, _while, _POS)
    assert _events(cfg_backend) == ['while', 'backedge']
    assert len(_regions(cfg_backend, LoopRegion)) == 1


@pytest.mark.torch
def test_while_with_if_in_body(cfg_backend):

    def f(x):
        acc = x
        while acc.sum() < 100:
            if acc.mean() > 1:
                acc = acc * 1.5
            else:
                acc = acc * 2
        return acc + 1

    _check(cfg_backend, f, _POS)
    assert len(_regions(cfg_backend, LoopRegion)) == 1 and len(_regions(cfg_backend, ConditionalBlock)) == 1


@pytest.mark.torch
def test_while_break(cfg_backend):
    # Pattern: unconditional jump from inside the loop body to the loop exit. The post-loop tail is longer than 4
    # instructions, so no CPython version replaces the jump with a copy of the tail.

    def f(x):
        acc = x
        while acc.sum() < 100:
            acc = acc * 2
            if acc.max() > 20:
                break
            acc = acc + 1
        acc = acc + 1
        return acc * 2

    _assert_patterns(f, present=('break-jump', ), absent=('early-return', ))
    _check(cfg_backend, f, _POS)
    assert 'exit' in _events(cfg_backend)
    assert len(_regions(cfg_backend, LoopRegion)) == 1


@pytest.mark.torch
def test_while_continue(cfg_backend):
    # Pattern: a second back-edge (``continue``) inside the loop body; it re-reaches the loop test

    def f(x):
        acc = x
        while acc.sum() < 100:
            acc = acc * 2
            if acc.max() > 30:
                continue
            acc = acc + 1
        return acc + 1

    _assert_patterns(f, present=('multiple-back-edges', ), absent=('break-jump', 'early-return'))
    _check(cfg_backend, f, _POS)
    assert 'exit' not in _events(cfg_backend)
    assert len(_regions(cfg_backend, LoopRegion)) == 1


@pytest.mark.torch
def test_while_break_and_continue(cfg_backend):
    # Both patterns above in one loop body (long post-loop tail, see test_while_break)

    def f(x):
        acc = x
        while acc.sum() < 100:
            acc = acc * 2
            if acc.max() > 1e9:
                continue
            if acc.max() > 20:
                break
            acc = acc + 1
        acc = acc + 1
        return acc * 2

    _assert_patterns(f, present=('break-jump', 'multiple-back-edges'), absent=('early-return', ))
    _check(cfg_backend, f, _POS)
    assert 'exit' in _events(cfg_backend)
    assert len(_regions(cfg_backend, LoopRegion)) == 1


@pytest.mark.torch
def test_while_true_break(cfg_backend):
    # Pattern: a bottom-tested loop whose only exit is the conditional jump of the ``break`` test. The exit block may
    # be laid out inline or out of line (3.14), but no version emits a jump-based ``break`` or a second exit here.

    def f(x):
        acc = x
        while True:
            acc = acc * 2
            if acc.sum() > 100:
                break
        return acc + 1

    _assert_patterns(f, present=('single-exit-edge', ), absent=('break-jump', 'early-return'))
    _check(cfg_backend, f, _POS)
    assert len(_regions(cfg_backend, LoopRegion)) == 1


@pytest.mark.torch
def test_nested_while(cfg_backend):
    # No break/return: every loop exit is a loop test. The statement after the inner loop keeps the inner exit from
    # being jump-threaded to the outer loop's test.

    def f(x):
        acc = x
        while acc.sum() < 1000:
            inner = acc
            while inner.sum() < 100:
                inner = inner * 2
            acc = inner + acc
        return acc

    _check(cfg_backend, f, _POS)
    assert len(_regions(cfg_backend, LoopRegion)) == 2


@pytest.mark.torch
def test_if_then_while_tail_duplicates(cfg_backend):
    # The loop after the if is traced into both cond branches (tail duplication); still one SDFG

    def f(x):
        y = x * 2
        if y.sum() > 0:
            y = y.sin()
        else:
            y = y.cos()
        y = y.abs() + 0.1
        while y.sum() < 100:
            y = y * 2
        return y

    _check(cfg_backend, f, _MIXED)
    assert len(_regions(cfg_backend, LoopRegion)) == 2 and len(_regions(cfg_backend, ConditionalBlock)) == 1


# ---------------------------------------------------------------------- fallbacks (stock Dynamo behaviour)
@pytest.mark.torch
def test_fallback_return_inside_loop(cfg_backend):
    # Pattern: ``RETURN_VALUE`` reached from inside the loop body (an explicit early return; on 3.14 the return block
    # is laid out after the loop). Not supported yet: the capture fails (``cond`` branches of different arity), the
    # jumps are blacklisted and Dynamo restarts the analysis with its stock behaviour at the loop guard (a graph break;
    # for a data-dependent loop, torch 2.14 then runs the frame eagerly), so the frame still runs correctly.

    def f(x):
        acc = x
        while acc.sum() < 100:
            acc = acc * 2
            if acc.max() > 20:
                return acc
        return acc + 1

    _assert_patterns(f, present=('early-return', ), absent=('break-jump', ))
    compiled = torch.compile(f, backend=cfg_backend, dynamic=True, fullgraph=False)
    with torch.no_grad():
        for (x, ) in _POS:
            torch.testing.assert_close(compiled(x), f(x))
    events = _events(cfg_backend)
    assert 'error' in events and 'fallback' in events, events


@pytest.mark.torch
def test_fallback_non_empty_stack(cfg_backend):
    # A tensor-dependent while inside a Python-unrolled for: the range iterator sits on the value stack

    def f(x):
        acc = x
        for i in range(2):
            while acc.sum() < 100:
                acc = acc * 2
            acc = acc * 0.25
        return acc

    compiled = torch.compile(f, backend=cfg_backend, dynamic=True, fullgraph=False)
    with torch.no_grad():
        for (x, ) in _POS:
            torch.testing.assert_close(compiled(x), f(x))
    assert 'fallback' in _events(cfg_backend)


@pytest.mark.torch
@pytest.mark.xfail(strict=False, reason='for over range(SymInt) is not captured yet (FOR_ITER specialises the range)')
def test_for_range_symint_compiles_once(cfg_backend):

    def f(x):
        acc = torch.zeros_like(x[0])
        for i in range(x.shape[0]):
            acc = acc + x[i]
        return acc

    _check(cfg_backend, f, [(torch.randn(3, 4), ), (torch.randn(5, 4), )])


@pytest.mark.torch
def test_production_backend_unaffected():
    # The patch is gated on the backend object: the stock DaceBackend still graph-breaks (fullgraph=True errors)
    from dace.frontend.ml.torch.dynamo import DaceBackend
    CfgDaceBackend()  # ensure the patch is installed
    compiled = torch.compile(_if_else, backend=DaceBackend(), dynamic=True, fullgraph=True)
    with pytest.raises(Exception, match='Data-dependent branching'):
        compiled(torch.rand(3, 4))


if __name__ == '__main__':
    for test in (test_if_else_both_continuations, test_if_without_else, test_if_with_int_and_symint_locals,
                 test_if_elif_else_nested, test_ternary_and_multiple_returns, test_while_tensor_predicate,
                 test_while_with_if_in_body, test_while_break, test_while_continue, test_while_break_and_continue,
                 test_while_true_break, test_nested_while, test_if_then_while_tail_duplicates,
                 test_fallback_return_inside_loop, test_fallback_non_empty_stack):
        torch._dynamo.reset()
        test(CfgDaceBackend())
        print(test.__name__, 'ok')
    torch._dynamo.reset()
    test_production_backend_unaffected()
