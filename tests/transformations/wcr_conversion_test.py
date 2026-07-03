import dace

from dace.transformation.dataflow import AugAssignToWCR


def test_aug_assign_tasklet_lhs():

    @dace.program
    def sdfg_aug_assign_tasklet_lhs(A: dace.float64[32], B: dace.float64[32]):
        for i in range(32):
            with dace.tasklet:
                a << A[i]
                k << B[i]
                b >> A[i]
                b = a + k

    sdfg = sdfg_aug_assign_tasklet_lhs.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(AugAssignToWCR)
    assert applied == 1


def test_aug_assign_tasklet_lhs_brackets():

    @dace.program
    def sdfg_aug_assign_tasklet_lhs_brackets(A: dace.float64[32], B: dace.float64[32]):
        for i in range(32):
            with dace.tasklet:
                a << A[i]
                k << B[i]
                b >> A[i]
                b = a + (k + 1)

    sdfg = sdfg_aug_assign_tasklet_lhs_brackets.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(AugAssignToWCR)
    assert applied == 1


def test_aug_assign_tasklet_rhs():

    @dace.program
    def sdfg_aug_assign_tasklet_rhs(A: dace.float64[32], B: dace.float64[32]):
        for i in range(32):
            with dace.tasklet:
                a << A[i]
                k << B[i]
                b >> A[i]
                b = k + a

    sdfg = sdfg_aug_assign_tasklet_rhs.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(AugAssignToWCR)
    assert applied == 1


def test_aug_assign_tasklet_rhs_brackets():

    @dace.program
    def sdfg_aug_assign_tasklet_rhs_brackets(A: dace.float64[32], B: dace.float64[32]):
        for i in range(32):
            with dace.tasklet:
                a << A[i]
                k << B[i]
                b >> A[i]
                b = (k + 1) + a

    sdfg = sdfg_aug_assign_tasklet_rhs_brackets.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(AugAssignToWCR)
    assert applied == 1


def test_aug_assign_tasklet_lhs_cpp():

    @dace.program
    def sdfg_aug_assign_tasklet_lhs_cpp(A: dace.float64[32], B: dace.float64[32]):
        for i in range(32):
            with dace.tasklet(language=dace.Language.CPP):
                a << A[i]
                k << B[i]
                b >> A[i]
                """
                b = a + k;
                """

    sdfg = sdfg_aug_assign_tasklet_lhs_cpp.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(AugAssignToWCR)
    assert applied == 1


def test_aug_assign_tasklet_lhs_brackets_cpp():

    @dace.program
    def sdfg_aug_assign_tasklet_lhs_brackets_cpp(A: dace.float64[32], B: dace.float64[32]):
        for i in range(32):
            with dace.tasklet(language=dace.Language.CPP):
                a << A[i]
                k << B[i]
                b >> A[i]
                """
                b = a + (k + 1);
                """

    sdfg = sdfg_aug_assign_tasklet_lhs_brackets_cpp.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(AugAssignToWCR)
    assert applied == 1


def test_aug_assign_tasklet_rhs_brackets_cpp():

    @dace.program
    def sdfg_aug_assign_tasklet_rhs_brackets_cpp(A: dace.float64[32], B: dace.float64[32]):
        for i in range(32):
            with dace.tasklet(language=dace.Language.CPP):
                a << A[i]
                k << B[i]
                b >> A[i]
                """
                b = (k + 1) + a;
                """

    sdfg = sdfg_aug_assign_tasklet_rhs_brackets_cpp.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(AugAssignToWCR)
    assert applied == 1


def test_aug_assign_tasklet_func_lhs_cpp():

    @dace.program
    def sdfg_aug_assign_tasklet_func_lhs_cpp(A: dace.float64[32], B: dace.float64[32]):
        for i in range(32):
            with dace.tasklet(language=dace.Language.CPP):
                a << A[i]
                c << B[i]
                b >> A[i]
                """
                b = min(a, c);
                """

    sdfg = sdfg_aug_assign_tasklet_func_lhs_cpp.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(AugAssignToWCR)
    assert applied == 1


def test_aug_assign_tasklet_func_rhs_cpp():

    @dace.program
    def sdfg_aug_assign_tasklet_func_rhs_cpp(A: dace.float64[32], B: dace.float64[32]):
        for i in range(32):
            with dace.tasklet(language=dace.Language.CPP):
                a << A[i]
                c << B[i]
                b >> A[i]
                """
                b = min(c, a);
                """

    sdfg = sdfg_aug_assign_tasklet_func_rhs_cpp.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(AugAssignToWCR)
    assert applied == 1


def test_aug_assign_free_map():

    @dace.program
    def sdfg_aug_assign_free_map(A: dace.float64[32], B: dace.float64[32]):
        for i in dace.map[0:32]:
            with dace.tasklet(language=dace.Language.CPP):
                a << A[0]
                k << B[i]
                b >> A[0]
                """
                b = k * a;
                """

    sdfg = sdfg_aug_assign_free_map.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(AugAssignToWCR)
    assert applied == 1


def test_aug_assign_state_fission_map():

    @dace.program
    def sdfg_aug_assign_state_fission(A: dace.float64[32], B: dace.float64[32]):
        for i in dace.map[0:32]:
            with dace.tasklet:
                a << B[i]
                b >> A[i]
                b = a

        for i in dace.map[0:32]:
            with dace.tasklet:
                a << A[0]
                b >> A[0]
                b = a * 2

        for i in dace.map[0:32]:
            with dace.tasklet:
                a << A[0]
                b >> A[0]
                b = a * 2

    sdfg = sdfg_aug_assign_state_fission.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(AugAssignToWCR)
    assert applied == 2


def test_free_map_permissive():

    @dace.program
    def sdfg_free_map_permissive(A: dace.float64[32], B: dace.float64[32]):
        for i in dace.map[0:32]:
            with dace.tasklet(language=dace.Language.CPP):
                a << A[i]
                k << B[i]
                b >> A[i]
                """
                b = k * a;
                """

    sdfg = sdfg_free_map_permissive.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(AugAssignToWCR, permissive=False)
    assert applied == 0

    applied = sdfg.apply_transformations_repeated(AugAssignToWCR, permissive=True)
    assert applied == 1


def test_aug_assign_same_inconns():

    @dace.program
    def sdfg_aug_assign_same_inconns(A: dace.float64[32]):
        for i in dace.map[0:31]:
            with dace.tasklet(language=dace.Language.Python):
                a << A[i]
                b << A[i + 1]
                c >> A[i]

                c = a * b

    sdfg = sdfg_aug_assign_same_inconns.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(AugAssignToWCR, permissive=True)
    assert applied == 1


def _build_copy_wrapped_rmw(op_code: str, op_wcr: str, n: int = 6):
    """Build an SDFG whose loop body is the *copy-wrapped* RMW the array
    frontends emit: the accumulator slice ``A[0]`` is materialized into a scalar
    transient, combined with the per-iteration increment ``B[j]`` in a tasklet,
    and copied back into ``A[0]`` -- ``A[0] -> a_in -> tasklet -> a_sum ->
    A[0]``. ``op_code`` is the tasklet RHS (``__in1 <op> __in2``); ``op_wcr`` is
    the numpy reduction used to build the oracle."""
    sdfg = dace.SDFG(f'copy_wrapped_rmw_{op_wcr}')
    sdfg.add_array('A', [2], dace.float64)
    sdfg.add_array('B', [n], dace.float64)
    sdfg.add_scalar('a_in', dace.float64, transient=True)
    sdfg.add_scalar('b_in', dace.float64, transient=True)
    sdfg.add_scalar('a_sum', dace.float64, transient=True)

    body = sdfg.add_state('body')
    a_r = body.add_read('A')
    a_in = body.add_access('a_in')
    b_r = body.add_read('B')
    b_in = body.add_access('b_in')
    tasklet = body.add_tasklet('combine', {'__in1', '__in2'}, {'__out'}, f'__out = {op_code}')
    a_sum = body.add_access('a_sum')
    a_w = body.add_write('A')

    body.add_edge(a_r, None, a_in, None, dace.Memlet('A[0]'))  # accumulator load copy
    body.add_edge(a_in, None, tasklet, '__in1', dace.Memlet('a_in[0]'))
    body.add_edge(b_r, None, b_in, None, dace.Memlet('B[j]'))
    body.add_edge(b_in, None, tasklet, '__in2', dace.Memlet('b_in[0]'))
    body.add_edge(tasklet, '__out', a_sum, None, dace.Memlet('a_sum[0]'))
    body.add_edge(a_sum, None, a_w, None, dace.Memlet('A[0]'))  # accumulator store copy

    before = sdfg.add_state('before', is_start_block=True)
    after = sdfg.add_state('after')
    sdfg.add_loop(before, body, after, 'j', '0', 'j < %d' % n, 'j + 1')
    sdfg.reset_cfg_list()
    return sdfg


def test_aug_assign_copy_wrapped_rmw_match():
    """The copy-wrapped RMW is recognised and rewritten to a WCR write: the
    accumulator load is dropped, the tasklet emits only the increment, and the
    write into ``A[0]`` carries the reduction WCR."""
    from dace.sdfg import nodes
    sdfg = _build_copy_wrapped_rmw('__in1 + __in2', 'sum')

    applied = sdfg.apply_transformations_repeated(AugAssignToWCR)
    assert applied == 1
    sdfg.validate()

    body = next(s for s in sdfg.all_states() if s.label == 'body')
    wcr_writes = [
        e for e in body.edges() if isinstance(e.dst, nodes.AccessNode) and e.dst.data == 'A' and e.data.wcr is not None
    ]
    assert len(wcr_writes) == 1
    assert 'a + b' in wcr_writes[0].data.wcr
    # The accumulator is no longer loaded inside the body.
    assert not any(isinstance(n, nodes.AccessNode) and n.data == 'A' and body.out_degree(n) > 0 for n in body.nodes())


def test_aug_assign_copy_wrapped_rmw_value_and_parallelize():
    """The rewrite is value-preserving and the now-WCR loop parallelizes via
    LoopToMap (the accumulator write is no longer iteration-indexed but is
    conflict-resolved)."""
    import numpy as np
    from dace.transformation.interstate import LoopToMap
    from dace.sdfg.state import LoopRegion

    rng = np.random.default_rng(0)
    n = 6

    def run(sdfg):
        A = np.array([3.0, 99.0], dtype=np.float64)
        B = rng.random(n)
        sdfg(A=A, B=B.copy())
        return A, B

    ref_sdfg = _build_copy_wrapped_rmw('__in1 + __in2', 'sum')
    A_ref, B = run(ref_sdfg)
    assert np.allclose(A_ref[0], 3.0 + B.sum(), rtol=1e-15, atol=1e-15)

    cand = _build_copy_wrapped_rmw('__in1 + __in2', 'sum')
    assert cand.apply_transformations_repeated(AugAssignToWCR) == 1
    n_l2m = cand.apply_transformations_repeated(LoopToMap)
    assert n_l2m == 1, 'WCR accumulator loop should parallelize'
    assert not [r for r in cand.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]

    A = np.array([3.0, 99.0], dtype=np.float64)
    cand(A=A, B=B.copy())
    assert np.allclose(A, A_ref, rtol=1e-12, atol=1e-12)


def test_aug_assign_copy_wrapped_rmw_max():
    """max-reduction copy-wrapped RMW lifts to a ``max`` WCR."""
    import numpy as np
    sdfg = _build_copy_wrapped_rmw('max(__in1, __in2)', 'max')
    assert sdfg.apply_transformations_repeated(AugAssignToWCR) == 1
    sdfg.validate()
    A = np.array([0.5, 0.0], dtype=np.float64)
    B = np.array([0.1, 0.9, 0.3, 0.2, 0.7, 0.4], dtype=np.float64)
    sdfg(A=A, B=B.copy())
    assert np.allclose(A[0], max(0.5, B.max()))


def test_aug_assign_copy_wrapped_rmw_subtract_left_only():
    """Subtraction lifts only with the accumulator on the left (``a - b``);
    ``b - a`` is not an order-independent reduction and must be refused."""
    import numpy as np
    sdfg = _build_copy_wrapped_rmw('__in1 - __in2', 'sub')  # acc on left -> OK
    assert sdfg.apply_transformations_repeated(AugAssignToWCR) == 1
    A = np.array([10.0, 0.0], dtype=np.float64)
    B = np.array([1.0, 2.0, 0.5, 1.5, 0.0, 1.0], dtype=np.float64)
    sdfg(A=A, B=B.copy())
    assert np.allclose(A[0], 10.0 - B.sum())

    refused = _build_copy_wrapped_rmw('__in2 - __in1', 'rsub')  # acc on right -> refuse
    assert refused.apply_transformations_repeated(AugAssignToWCR) == 0


def test_aug_assign_refuses_cross_element_operand():
    """A map whose tasklet is ``A[i,j] = A[i,k] * A[k,j]`` must NOT be rewritten to a
    WCR: the operand ``A[i,k]`` shares the array NAME with the output ``A[i,j]`` but
    reads a DIFFERENT element, so it is not the accumulator of a read-modify-write.
    Treating it as one would rewrite the reduction with the delta's operator (``*``)
    instead of the real accumulation, corrupting the result (polybench ``lu``). The
    Python branch must apply the same element/subset check the CPP branch already has.
    """
    sdfg = dace.SDFG('aug_cross_element')
    sdfg.add_array('A', [8, 8], dace.float64)
    st = sdfg.add_state()
    a_in = st.add_access('A')
    me, mx = st.add_map('m', dict(i='0:8', j='0:8', k='0:8'))
    tlet = st.add_tasklet('prod', {'aik', 'akj'}, {'out'}, 'out = aik * akj')
    st.add_memlet_path(a_in, me, tlet, dst_conn='aik', memlet=dace.Memlet('A[i, k]'))
    st.add_memlet_path(a_in, me, tlet, dst_conn='akj', memlet=dace.Memlet('A[k, j]'))
    a_out = st.add_access('A')
    st.add_memlet_path(tlet, mx, a_out, src_conn='out', memlet=dace.Memlet('A[i, j]'))
    sdfg.validate()
    # Cross-element operand -> not a read-modify-write -> refused.
    assert sdfg.apply_transformations_repeated(AugAssignToWCR) == 0


def test_aug_assign_cross_element_operand_trisolv_shape():
    """Same cross-element guard, in the shape that regressed polybench ``trisolv``:
    tasklet ``x[i] = L[i,j] * x[j]`` -- the operand ``x[j]`` shares the array NAME with
    the output ``x[i]`` but reads a different element, so it is not the accumulator of a
    read-modify-write. It must not be rewritten to a WCR (which previously misread the
    reduction operator and then tried to fission a scope it cannot legally split). The
    value-preserving corpus test (``poly:trisolv``) is the end-to-end check; this pins
    the transform-level refusal.
    """
    sdfg = dace.SDFG('aug_cross_element_trisolv')
    sdfg.add_array('x', [8], dace.float64)
    sdfg.add_array('L', [8, 8], dace.float64)
    st = sdfg.add_state()
    x_in = st.add_access('x')
    ell = st.add_access('L')
    me, mx = st.add_map('m', dict(i='0:8', j='0:8'))
    tlet = st.add_tasklet('prod', {'lij', 'xj'}, {'out'}, 'out = lij * xj')
    st.add_memlet_path(ell, me, tlet, dst_conn='lij', memlet=dace.Memlet('L[i, j]'))
    st.add_memlet_path(x_in, me, tlet, dst_conn='xj', memlet=dace.Memlet('x[j]'))
    x_out = st.add_access('x')
    st.add_memlet_path(tlet, mx, x_out, src_conn='out', memlet=dace.Memlet('x[i]'))
    sdfg.validate()
    # ``x[j]`` operand vs ``x[i]`` output: cross-element, not an accumulator -> refused.
    assert sdfg.apply_transformations_repeated(AugAssignToWCR) == 0
