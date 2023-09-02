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


def test_aug_assign_dependent_map():

    @dace.program
    def sdfg_aug_assign_dependent_map(A: dace.float64[32], B: dace.float64[32]):
        for i in dace.map[0:32]:
            with dace.tasklet:
                a << B[i]
                b >> A[i]
                b = a

        for i in dace.map[0:32]:
            with dace.tasklet:
                a << A[i]
                b >> A[i]
                b = a * 2

        for i in dace.map[0:32]:
            with dace.tasklet:
                a << A[i]
                b >> B[i]
                b = a

    sdfg = sdfg_aug_assign_dependent_map.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(AugAssignToWCR)
    assert applied == 1


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
