"""
Test our pytest fixtures using pytester
"""


def test_gpu(pytester):
    pytester.copy_example("tests/daceml/conftest.py")
    pytester.makepyfile("""
    def test_gpu(gpu):
        pass
    """)
    pytester.runpytest().assert_outcomes(passed=1)
    pytester.runpytest("--gpu").assert_outcomes(passed=2)
    pytester.runpytest("--gpu-only").assert_outcomes(passed=1)
    pytester.runpytest("--skip-cpu-blas").assert_outcomes(passed=1)


def test_gpu_only_skips(pytester):
    pytester.copy_example("tests/daceml/conftest.py")
    pytester.makepyfile("""
    def test_gpu(gpu):
        assert gpu
    """)
    pytester.runpytest("--gpu").assert_outcomes(passed=1, failed=1)
    pytester.runpytest("--gpu-only").assert_outcomes(passed=1)


def test_gpu_marker(pytester):
    pytester.copy_example("tests/daceml/conftest.py")
    pytester.makepyfile("""
    import pytest
    @pytest.mark.gpu
    def test_gpu():
        pass
    """)
    pytester.runpytest().assert_outcomes(skipped=1)
    pytester.runpytest("--gpu").assert_outcomes(passed=1)
    pytester.runpytest("--gpu-only").assert_outcomes(passed=1)
    pytester.runpytest("--skip-cpu-blas").assert_outcomes(skipped=1)


def test_cpublas_marker(pytester):
    pytester.copy_example("tests/daceml/conftest.py")
    pytester.makepyfile("""
    import pytest
    @pytest.mark.cpublas
    def test_cpublas():
        pass
    """)
    pytester.runpytest().assert_outcomes(passed=1)
    pytester.runpytest("--gpu").assert_outcomes(passed=1)
    pytester.runpytest("--gpu-only").assert_outcomes(skipped=1)
    pytester.runpytest("--skip-cpu-blas").assert_outcomes(skipped=1)


def test_cpublas_marker_with_gpu_fixture(pytester):
    pytester.copy_example("tests/daceml/conftest.py")
    pytester.makepyfile("""
    import pytest
    @pytest.mark.cpublas
    def test_cpublas(gpu):
        pass
    """)
    pytester.runpytest().assert_outcomes(passed=1)
    pytester.runpytest("--gpu").assert_outcomes(passed=2)
    pytester.runpytest("--gpu-only").assert_outcomes(passed=1)
    pytester.runpytest("--skip-cpu-blas").assert_outcomes(skipped=1)
    pytester.runpytest("--gpu", "--skip-cpu-blas").assert_outcomes(passed=1)
    pytester.runpytest("--gpu-only",
                       "--skip-cpu-blas").assert_outcomes(passed=1)
