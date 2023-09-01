import dace

from dace.transformation.dataflow import AugAssignToWCR


def test_aug_assign_tasklet_lhs():

    @dace.program
    def sdfg_aug_assign_tasklet_lhs(A: dace.float64[32]):
        for i in range(32):
            with dace.tasklet:
                a << A[i]
                b >> A[i]
                b = a + 1

    sdfg = sdfg_aug_assign_tasklet_lhs.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(AugAssignToWCR)
    assert applied == 1


def test_aug_assign_tasklet_lhs_brackets():

    @dace.program
    def sdfg_aug_assign_tasklet_lhs_brackets(A: dace.float64[32]):
        for i in range(32):
            with dace.tasklet:
                a << A[i]
                b >> A[i]
                b = a + (1 + 1)

    sdfg = sdfg_aug_assign_tasklet_lhs_brackets.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(AugAssignToWCR)
    assert applied == 1


def test_aug_assign_tasklet_rhs():

    @dace.program
    def sdfg_aug_assign_tasklet_rhs(A: dace.float64[32]):
        for i in range(32):
            with dace.tasklet:
                a << A[i]
                b >> A[i]
                b = 1 + a

    sdfg = sdfg_aug_assign_tasklet_rhs.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(AugAssignToWCR)
    assert applied == 1


def test_aug_assign_tasklet_rhs_brackets():

    @dace.program
    def sdfg_aug_assign_tasklet_rhs_brackets(A: dace.float64[32]):
        for i in range(32):
            with dace.tasklet:
                a << A[i]
                b >> A[i]
                b = (1 + 1) + a

    sdfg = sdfg_aug_assign_tasklet_rhs_brackets.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(AugAssignToWCR)
    assert applied == 1


def test_aug_assign_tasklet_lhs_cpp():

    @dace.program
    def sdfg_aug_assign_tasklet_lhs_cpp(A: dace.float64[32]):
        for i in range(32):
            with dace.tasklet(language=dace.Language.CPP):
                a << A[i]
                b >> A[i]
                """
                b = a + 1;
                """

    sdfg = sdfg_aug_assign_tasklet_lhs_cpp.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(AugAssignToWCR)
    assert applied == 1


def test_aug_assign_tasklet_lhs_brackets_cpp():

    @dace.program
    def sdfg_aug_assign_tasklet_lhs_brackets_cpp(A: dace.float64[32]):
        for i in range(32):
            with dace.tasklet(language=dace.Language.CPP):
                a << A[i]
                b >> A[i]
                """
                b = a + (1 + 1);
                """

    sdfg = sdfg_aug_assign_tasklet_lhs_brackets_cpp.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(AugAssignToWCR)
    assert applied == 1


def test_aug_assign_tasklet_rhs_brackets_cpp():

    @dace.program
    def sdfg_aug_assign_tasklet_rhs_brackets_cpp(A: dace.float64[32]):
        for i in range(32):
            with dace.tasklet(language=dace.Language.CPP):
                a << A[i]
                b >> A[i]
                """
                b = (1 + 1) + a;
                """

    sdfg = sdfg_aug_assign_tasklet_rhs_brackets_cpp.to_sdfg()
    sdfg.simplify()

    applied = sdfg.apply_transformations_repeated(AugAssignToWCR)
    assert applied == 1


if __name__ == "__main__":
    test_aug_assign_tasklet_lhs_cpp()
    test_aug_assign_tasklet_lhs_brackets_cpp()
    test_aug_assign_tasklet_rhs_brackets_cpp()
