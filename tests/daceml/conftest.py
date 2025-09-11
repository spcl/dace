import pytest
# import dace.libraries.onnx as donnx
import dace.libraries.onnx as donnx

import sys

pytest_plugins = "pytester"

# the bert encoder is very nested, and exceeds the recursion limit while serializing
sys.setrecursionlimit(2000)

getname = lambda x: x.name


def pytest_addoption(parser):
    parser.addoption("--skip-cpu-blas",
                     action="store_true",
                     help="Skip tests that are slow wthout cpu blas.")
    parser.addoption("--gpu", action="store_true", help="Run tests using gpu.")
    parser.addoption("--gpu-only",
                     action="store_true",
                     help="Run tests using gpu, and skip CPU tests.")


def pytest_runtest_setup(item):
    """
    This method handles skipping tests depending on command line flags
    """
    # if @pytest.mark.gpu is applied skip the test on CPU
    if "gpu" in map(getname, item.iter_markers()):
        if not (item.config.getoption("--gpu")
                or item.config.getoption("--gpu-only")):
            pytest.skip(
                'Skipping test since --gpu or --gpu-only were not passed')
    # else: if the gpu fixture is used, the test is parameterized. Skipping
    #       will be handled in pytest_generate_tests
    elif "gpu" in item.fixturenames:
        pass
    else:
        # this test is not marked with @pytest.mark.gpu, and doesn't have the gpu fixture
        # skip it if --gpu-only is passed
        if item.config.getoption("--gpu-only"):
            pytest.skip('Skipping test since --gpu-only was passed')
        # skip it if --skip-cpu-blas is passed an it has the cpublas marker
        if (item.config.getoption("--skip-cpu-blas")
                and "cpublas" in (m.name for m in item.iter_markers())):
            pytest.skip('Skipping test since --skip-cpu-blas was passed')


def pytest_generate_tests(metafunc):
    """
    This method sets up the parametrizations for the custom fixtures
    """
    if "gpu" in metafunc.fixturenames:
        if metafunc.config.getoption("--gpu"):
            runs = [
                pytest.param(True, id="use_gpu"),
            ]

            if metafunc.config.getoption(
                    "--skip-cpu-blas") and 'cpublas' in map(
                        getname, metafunc.definition.iter_markers()):
                # don't run the test on cpu
                pass
            else:
                runs.append(pytest.param(False, id="use_cpu"))
            metafunc.parametrize("gpu", runs)
        elif metafunc.config.getoption("--gpu-only"):
            metafunc.parametrize("gpu", [pytest.param(True, id="use_gpu")])
        else:
            if metafunc.config.getoption(
                    "--skip-cpu-blas") and 'cpublas' in map(
                        getname, metafunc.definition.iter_markers()):
                # skip cpublas tests
                runs = []
            else:
                runs = [pytest.param(False, id="use_cpu")]
            metafunc.parametrize("gpu", runs)

    if "default_implementation" in metafunc.fixturenames:
        implementations = [
            pytest.param("pure", marks=pytest.mark.pure),
        ]
        metafunc.parametrize("default_implementation", implementations)

    if "use_cpp_dispatcher" in metafunc.fixturenames:
        metafunc.parametrize("use_cpp_dispatcher", [
            pytest.param(True, id="use_cpp_dispatcher"),
            pytest.param(False, id="no_use_cpp_dispatcher"),
        ])


@pytest.fixture
def sdfg_name(request):
    return request.node.name.replace("[", "-").replace("]",
                                                       "").replace("-", "_")


@pytest.fixture(autouse=True)
def setup_default_implementation(request):
    # this fixture is used for all tests (autouse)

    old_default = donnx.default_implementation

    pure_marker = request.node.get_closest_marker("pure")

    if pure_marker is not None:
        donnx.default_implementation = "pure"
        yield

    if pure_marker is None:
        yield

    donnx.default_implementation = old_default
