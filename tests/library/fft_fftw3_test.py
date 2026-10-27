# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit test for the FFTW3 lowering of :class:`dace.libraries.fft.nodes.FFT`.

Drives a forward + inverse round-trip through the FFTW3 expansion and
compares against numpy's reference. The marker ``fftw`` opts the test
into the CI step that installs ``libfftw3-dev``.
"""

import numpy as np
import pytest

import dace
import dace.libraries.fft as fftlib


@pytest.mark.fftw
def test_fft_fftw3_lowering():
    """One test exercising the FFTW3 backend for both FFT and IFFT."""
    N = 256

    @dace.program
    def fft_then_ifft(x: dace.complex128[N]):
        y = np.fft.fft(x)
        z = np.fft.ifft(y, norm="forward")
        return y, z

    sdfg = fft_then_ifft.to_sdfg()
    fftlib.FFT.default_implementation = "FFTW3"
    fftlib.IFFT.default_implementation = "FFTW3"
    try:
        sdfg.expand_library_nodes()
    finally:
        fftlib.FFT.default_implementation = "pure"
        fftlib.IFFT.default_implementation = "pure"

    rng = np.random.default_rng(42)
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    y, z = sdfg(x.copy())
    np.testing.assert_allclose(y, np.fft.fft(x), rtol=1e-12, atol=1e-12)
    # ``norm='forward'`` leaves the IFFT un-normalised, matching FFTW's
    # backward convention. The forward+backward round trip therefore
    # scales by ``N``, so divide out before comparing.
    np.testing.assert_allclose(z / N, x, rtol=1e-12, atol=1e-12)


def test_a_threaded_companion_library_is_linked_ahead_of_its_base(tmp_path):
    """FFTW's threaded planner ships as ``libfftw3_omp`` / ``libfftw3_threads`` beside ``libfftw3``; without it the
    transform ran on one thread against numba's threaded pocketfft (fft_1d, 10x slower)."""
    from dace.libraries.fft.environments import fftw3

    (tmp_path / "include").mkdir()
    (tmp_path / "include" / "fftw3.h").touch()
    (tmp_path / "lib").mkdir()
    for lib in ("libfftw3.so", "libfftw3_threads.so", "libfftw3_omp.so", "libfftw3f.so"):
        (tmp_path / "lib" / lib).touch()
    names = [p.rsplit("/", 1)[1] for p in fftw3.prefix_libraries(str(tmp_path))]
    assert names == ["libfftw3_omp.so", "libfftw3.so", "libfftw3f.so"]


@pytest.mark.parametrize("threaded", [True, False])
def test_a_threaded_fftw_plans_for_the_team(monkeypatch, threaded):
    from dace import data
    from dace.libraries.fft.environments import fftw3
    from dace.libraries.fft.nodes.fft import Transform, fftw3_call

    monkeypatch.setattr(fftw3, "threaded", lambda double: threaded)
    desc = data.Array(dace.complex128, [64])
    code = fftw3_call(Transform(desc, desc, [0], inverse=False, half_spectrum=False)).code.as_string
    assert ("fftw_plan_with_nthreads(__threads)" in code) == threaded
    assert ("__threads = omp_in_parallel() ? 1 : omp_get_max_threads()" in code) == threaded


if __name__ == "__main__":
    test_fft_fftw3_lowering()
    print("FFTW3 FFT lowering test PASS")
    import pathlib
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_a_threaded_companion_library_is_linked_ahead_of_its_base(pathlib.Path(tmp))
    for threaded in (True, False):
        with pytest.MonkeyPatch.context() as mp:
            test_a_threaded_fftw_plans_for_the_team(mp, threaded)
