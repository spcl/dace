# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The FFT / IFFT library node against numpy, through every implementation that can run here.

``pure`` is the separable DFT every other expansion falls back to. ``FFTW3`` is the CPU vendor
call and ``cuFFT`` / ``hipFFT`` the GPU one, all three wrapped in the same nested SDFG: a staged copy
for an input of another type or layout, the unnormalized transform with a plan cached in the SDFG
state, then the ``norm`` scale. Each case pins the numpy semantics that wrapper has to reproduce --
the axes, the three norms, both precisions, a real input, a strided (leading-axis) transform and the
in-place ``x[:] = fft(x)`` -- or the plan and the layout the vendor is given.
"""

import warnings

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.codegen.common import get_gpu_backend
from dace.libraries.fft.environments.fftw3 import FFTW3
from dace.libraries.fft.nodes import FFT, IFFT
from dace.libraries.fft.environments import fftw3
from dace.libraries.fft.nodes.fft import GpuFftLayout, Transform, fftw3_call, gpu_fft_layout

N = dace.symbol("N")

CPU_IMPLEMENTATIONS = ["pure", pytest.param("FFTW3", marks=pytest.mark.fftw)]


def gpu_implementation() -> str:
    """The vendor FFT of this build's GPU backend."""
    return "hipFFT" if get_gpu_backend() == "hip" else "cuFFT"


def compile_with(program, implementation: str, gpu: bool = False) -> dace.SDFG:
    """``program`` as an SDFG whose every FFT / IFFT node takes ``implementation``."""
    sdfg = program.to_sdfg(simplify=True)
    # One build folder per implementation: parallel workers compiling one name race on it
    sdfg.name = f"{sdfg.name}_{implementation}"
    if gpu:
        sdfg.apply_gpu_transformations()
    nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, (FFT, IFFT))]
    assert nodes, "the program lost its FFT library nodes"
    for node in nodes:
        node.implementation = implementation
    return sdfg


def run_without_fallback(sdfg: dace.SDFG, **arguments):
    """Run ``sdfg``; a vendor expansion falling back to ``pure`` warns, and that fails the case."""
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".* cannot transform axes")
        return sdfg(**arguments)


@dace.program
def poisson_round_trip(x: dace.float64[N, N, N]):
    """ls3df's reciprocal-space solve: a real ``fftn``, a scale and a normalized ``ifftn``, whose real part the
    caller keeps."""
    return np.fft.ifftn(4.0 * np.fft.fftn(x))


@dace.program
def fft_1d_round_trip(x: dace.complex128[N], y: dace.complex128[N], z: dace.complex128[N]):
    """fft_1d: the forward transform and the round trip back, which must recover ``x``."""
    y[:] = np.fft.fft(x)
    z[:] = np.fft.ifft(y)


def rng_complex(shape, dtype=np.complex128) -> np.ndarray:
    rng = np.random.default_rng(sum(shape))
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)


def check_poisson_round_trip(implementation: str, gpu: bool = False):
    x = np.random.default_rng(1).standard_normal((12, 12, 12))
    got = run_without_fallback(compile_with(poisson_round_trip, implementation, gpu), x=x.copy(), N=12)
    np.testing.assert_allclose(got.real, np.fft.ifftn(4.0 * np.fft.fftn(x)).real, rtol=1e-12, atol=1e-12)


def check_fft_1d_round_trip(implementation: str, gpu: bool = False):
    x = rng_complex((1000,))
    y, z = np.zeros_like(x), np.zeros_like(x)
    run_without_fallback(compile_with(fft_1d_round_trip, implementation, gpu), x=x.copy(), y=y, z=z, N=1000)
    np.testing.assert_allclose(y, np.fft.fft(x), rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(z, x, rtol=1e-12, atol=1e-12)


def forward_and_inverse(axes, norm):
    """``fftn`` and ``ifftn`` of a rank-3 array over ``axes`` under ``norm``, as one program."""

    @dace.program
    def transforms_forward_and_inverse(x: dace.complex128[6, 10, 8]):
        return np.fft.fftn(x, axes=axes, norm=norm), np.fft.ifftn(x, axes=axes, norm=norm)

    return transforms_forward_and_inverse


def check_norms_and_axes(implementation: str, axes, gpu: bool = False):
    """``fftn`` / ``ifftn`` over ``axes`` of a rank-3 array under each of numpy's three norms."""
    for norm in ("backward", "ortho", "forward"):
        x = rng_complex((6, 10, 8))
        program = forward_and_inverse(axes, norm)
        sdfg = compile_with(program, implementation, gpu)
        sdfg.name = f"{sdfg.name}_{'_'.join(map(str, axes or ()))}_{norm}"
        forward, inverse = run_without_fallback(sdfg, x=x.copy())
        np.testing.assert_allclose(forward, np.fft.fftn(x, axes=axes, norm=norm), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(inverse, np.fft.ifftn(x, axes=axes, norm=norm), rtol=1e-12, atol=1e-12)


def check_single_precision(implementation: str, gpu: bool = False):
    """complex64 stays complex64 through the transform and its normalisation."""

    @dace.program
    def transforms_check_single_precision(x: dace.complex64[N]):
        return np.fft.fft(x), np.fft.ifft(x)

    x = rng_complex((256,), np.complex64)
    forward, inverse = run_without_fallback(
        compile_with(transforms_check_single_precision, implementation, gpu), x=x.copy(), N=256
    )
    assert forward.dtype == np.complex64 and inverse.dtype == np.complex64
    np.testing.assert_allclose(forward, np.fft.fft(x), rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(inverse, np.fft.ifft(x), rtol=1e-4, atol=1e-5)


def check_leading_axis(implementation: str, gpu: bool = False):
    """``fft(x, axis=0)`` of a matrix: each transform steps by the row length, the batch by one."""

    @dace.program
    def transform_check_leading_axis(x: dace.complex128[16, 12]):
        return np.fft.fft(x, axis=0)

    x = rng_complex((16, 12))
    got = run_without_fallback(compile_with(transform_check_leading_axis, implementation, gpu), x=x.copy())
    np.testing.assert_allclose(got, np.fft.fft(x, axis=0), rtol=1e-12, atol=1e-12)


def check_in_place(implementation: str, gpu: bool = False):
    """``x[:] = fft(x)`` overwrites the operand it reads."""

    @dace.program
    def transform_check_in_place(x: dace.complex128[N]):
        x[:] = np.fft.fft(x)

    x = rng_complex((300,))
    want = np.fft.fft(x)
    run_without_fallback(compile_with(transform_check_in_place, implementation, gpu), x=x, N=300)
    np.testing.assert_allclose(x, want, rtol=1e-12, atol=1e-10)


@pytest.mark.parametrize("implementation", CPU_IMPLEMENTATIONS)
def test_cpu_poisson_round_trip_of_a_real_grid(implementation):
    check_poisson_round_trip(implementation)


@pytest.mark.parametrize("implementation", CPU_IMPLEMENTATIONS)
def test_cpu_1d_round_trip_over_a_symbolic_extent(implementation):
    check_fft_1d_round_trip(implementation)


@pytest.mark.parametrize("axes", [None, (1, 2), (0, 2), (2, 0)])
@pytest.mark.parametrize("implementation", CPU_IMPLEMENTATIONS)
def test_cpu_norms_over_every_axis_set(implementation, axes):
    check_norms_and_axes(implementation, axes)


@pytest.mark.parametrize("implementation", CPU_IMPLEMENTATIONS)
def test_cpu_single_precision(implementation):
    check_single_precision(implementation)


@pytest.mark.parametrize("implementation", CPU_IMPLEMENTATIONS)
def test_cpu_leading_axis(implementation):
    check_leading_axis(implementation)


@pytest.mark.parametrize("implementation", CPU_IMPLEMENTATIONS)
def test_cpu_in_place(implementation):
    check_in_place(implementation)


@pytest.mark.fftw
def test_fftw3_falls_back_to_pure_for_a_repeated_axis():
    """numpy transforms a repeated axis twice; one FFTW plan cannot, so the expansion hands it to ``pure``."""

    @dace.program
    def transform_fftw3_falls_back_to_pure_for_a_repeated_axis(x: dace.complex128[8, 6]):
        return np.fft.fftn(x, axes=(1, 1))

    x = rng_complex((8, 6))
    sdfg = compile_with(transform_fftw3_falls_back_to_pure_for_a_repeated_axis, "FFTW3")
    with pytest.warns(UserWarning, match="FFTW3 cannot transform axes"):
        sdfg.expand_library_nodes()
    np.testing.assert_allclose(sdfg(x=x.copy()), np.fft.fftn(x, axes=(1, 1)), rtol=1e-12, atol=1e-12)


@pytest.mark.gpu
def test_gpu_poisson_round_trip_of_a_real_grid():
    check_poisson_round_trip(gpu_implementation(), gpu=True)


@pytest.mark.gpu
def test_gpu_1d_round_trip_over_a_symbolic_extent():
    check_fft_1d_round_trip(gpu_implementation(), gpu=True)


@pytest.mark.gpu
@pytest.mark.parametrize("axes", [None, (1, 2), (0, 1), (2,)])
def test_gpu_norms_over_every_batched_axis_block(axes):
    check_norms_and_axes(gpu_implementation(), axes, gpu=True)


@pytest.mark.gpu
def test_gpu_single_precision():
    check_single_precision(gpu_implementation(), gpu=True)


@pytest.mark.gpu
def test_gpu_leading_axis():
    check_leading_axis(gpu_implementation(), gpu=True)


@pytest.mark.gpu
def test_gpu_in_place():
    check_in_place(gpu_implementation(), gpu=True)


@pytest.mark.gpu
def test_gpu_fft_falls_back_to_pure_for_a_middle_axis():
    """A middle axis leaves two batch strides, which no ``MakePlanMany`` layout expresses."""

    @dace.program
    def transform_gpu_fft_falls_back_to_pure_for_a_middle_axis(x: dace.complex128[4, 6, 8]):
        return np.fft.fft(x, axis=1)

    x = rng_complex((4, 6, 8))
    sdfg = compile_with(transform_gpu_fft_falls_back_to_pure_for_a_middle_axis, gpu_implementation(), gpu=True)
    with pytest.warns(UserWarning, match="cannot transform axes"):
        sdfg.expand_library_nodes()
    np.testing.assert_allclose(sdfg(x=x.copy()), np.fft.fft(x, axis=1), rtol=1e-12, atol=1e-12)


def strided_array(shape, strides, storage=dtypes.StorageType.GPU_Global) -> dace.data.Array:
    """A complex128 descriptor of ``shape`` with explicit ``strides``."""
    return dace.data.Array(dace.complex128, shape, storage=storage, strides=strides)


#: (4, 5, 6, 3) laid out column-major, and as cegterg's ``psic.reshape(n1, n2, n3, m, order='F')``
#: of a C-order ``(n1*n2*n3, m)`` array: the transformed axes column-major, the batch innermost.
FORTRAN = [1, 4, 20, 120]
BATCH_INNERMOST = [3, 12, 60, 1]
C_ORDER = [90, 18, 3, 1]


def test_a_gpu_fft_plans_each_side_s_own_dense_block():
    """One plan reads and writes every layout whose transformed axes form one dense block.

    cegterg's ``ifftn`` over axes (0, 1, 2) reads a batch-innermost view and writes a Fortran-order
    result. Both were refused, and the separable DFT that ran instead had host maps over device
    memory, which failed validation.
    """
    shape = [4, 5, 6, 3]
    fortran, innermost = strided_array(shape, FORTRAN), strided_array(shape, BATCH_INNERMOST)
    c_order = strided_array(shape, C_ORDER)

    def layout(src, out):
        return gpu_fft_layout(Transform(src, out, [0, 1, 2], inverse=False, half_spectrum=False))

    assert layout(fortran, fortran) == GpuFftLayout([6, 5, 4], [6, 5, 4], 1, 120, [6, 5, 4], 1, 120, 3)
    assert layout(innermost, fortran) == GpuFftLayout([6, 5, 4], [6, 5, 4], 3, 1, [6, 5, 4], 1, 120, 3)
    assert layout(c_order, c_order) == GpuFftLayout([4, 5, 6], [4, 5, 6], 3, 1, [4, 5, 6], 3, 1, 3)
    # The two sides order the transformed axes oppositely: no shared plan, so the source is staged.
    assert layout(fortran, c_order) is None


def test_a_half_spectrum_gpu_fft_embeds_each_side_s_own_extents():
    """``rfftn`` over a C-order (8, 6) grid: the plan lengths are the real extents, the complex side's embed keeps
    the halved last axis, and both sides step their batch by their own block size."""
    real = dace.data.Array(dace.float64, [2, 8, 6], storage=dtypes.StorageType.GPU_Global)
    spectrum = strided_array([2, 8, 4], [32, 4, 1])
    got = gpu_fft_layout(Transform(real, spectrum, [1, 2], inverse=False, half_spectrum=True))
    assert got == GpuFftLayout([8, 6], [8, 6], 1, 48, [8, 4], 1, 32, 2)


def run_strided_fftn(name: str, src_strides, out_strides):
    """``fftn`` over axes (0, 1, 2) of a (4, 5, 6, 3) operand, each side in its own layout, on the GPU."""
    shape = (4, 5, 6, 3)
    sdfg = dace.SDFG(name)
    # Device buffers in the host arrays' own layouts, so both transfers are plain copies.
    for name, strides in (("x", src_strides), ("y", out_strides)):
        sdfg.add_datadesc(name, strided_array(shape, strides, dtypes.StorageType.Default))
        device = strided_array(shape, strides)
        device.transient = True
        sdfg.add_datadesc(f"g{name}", device)
    state = sdfg.add_state()
    node = FFT("fft", axes=[0, 1, 2])
    node.implementation = gpu_implementation()
    gx, gy = state.add_access("gx"), state.add_access("gy")
    state.add_nedge(state.add_read("x"), gx, dace.Memlet.from_array("x", sdfg.arrays["x"]))
    state.add_edge(gx, None, node, "_inp", dace.Memlet.from_array("gx", sdfg.arrays["gx"]))
    state.add_edge(node, "_out", gy, None, dace.Memlet.from_array("gy", sdfg.arrays["gy"]))
    state.add_nedge(gy, state.add_write("y"), dace.Memlet.from_array("gy", sdfg.arrays["gy"]))
    itemsize = np.dtype(np.complex128).itemsize
    buffer = rng_complex((4 * 5 * 6 * 3,))
    x = np.lib.stride_tricks.as_strided(buffer, shape, [s * itemsize for s in src_strides])
    y_buffer = np.zeros(4 * 5 * 6 * 3, dtype=np.complex128)
    y = np.lib.stride_tricks.as_strided(y_buffer, shape, [s * itemsize for s in out_strides])
    # The operands ARE strided views; that layout is the point of the case.
    with dace.config.set_temporary("compiler", "allow_view_arguments", value=True):
        run_without_fallback(sdfg, x=x, y=y)
    np.testing.assert_allclose(y, np.fft.fftn(x, axes=(0, 1, 2)), rtol=1e-12, atol=1e-10)


@pytest.mark.gpu
def test_gpu_batch_innermost_view_into_a_fortran_result_takes_one_plan():
    """cegterg's layout pair end to end: the vendor plan, no fallback, numpy's numbers."""
    run_strided_fftn("batch_innermost_into_fortran_fftn", BATCH_INNERMOST, FORTRAN)


@pytest.mark.gpu
def test_gpu_oppositely_ordered_sides_stage_the_source():
    """No shared plan: the source is copied into the output's layout on the device, then planned."""
    run_strided_fftn("fortran_into_c_order_fftn", FORTRAN, C_ORDER)


@pytest.mark.gpu
def test_gpu_fft_over_an_empty_batch_is_a_no_op():
    """A batch that is empty at run time transforms nothing, as numpy returns an empty array.

    The vendor planner raises SIGFPE on a zero batch; cegterg's canon GPU run died there once no
    unconverged vector was left (``__sym_notcnv_iter == 0``).
    """
    batch = dace.symbol("batch")
    shape = (4, 5, 6, batch)
    sdfg = dace.SDFG("empty_batch_fftn")
    for name in ("x", "y"):
        sdfg.add_array(name, shape, dace.complex128, strides=FORTRAN)
        sdfg.add_array(
            f"g{name}", shape, dace.complex128, strides=FORTRAN, storage=dtypes.StorageType.GPU_Global, transient=True
        )
    state = sdfg.add_state()
    node = FFT("fft", axes=[0, 1, 2])
    node.implementation = gpu_implementation()
    gx, gy = state.add_access("gx"), state.add_access("gy")
    state.add_nedge(state.add_read("x"), gx, dace.Memlet.from_array("x", sdfg.arrays["x"]))
    state.add_edge(gx, None, node, "_inp", dace.Memlet.from_array("gx", sdfg.arrays["gx"]))
    state.add_edge(node, "_out", gy, None, dace.Memlet.from_array("gy", sdfg.arrays["gy"]))
    state.add_nedge(gy, state.add_write("y"), dace.Memlet.from_array("gy", sdfg.arrays["gy"]))
    compiled = sdfg.compile()
    for extent in (3, 0):
        x = np.asfortranarray(rng_complex((4, 5, 6, extent)))
        y = np.zeros_like(x, order="F")
        compiled(x=x, y=y, batch=extent)
        np.testing.assert_allclose(y, np.fft.fftn(x, axes=(0, 1, 2)), rtol=1e-12, atol=1e-10)




@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(32, 48), (8, 12, 16)])
def test_gpu_fftn_over_every_axis(shape):
    """One plan over every axis of a rank 2 or 3 array, the case the vendor's own ``MakePlan{2,3}d`` covers."""

    @dace.program
    def transform_gpu_fftn_over_every_axis(x: dace.complex128[shape]):
        return np.fft.fftn(x)

    x = rng_complex(shape)
    sdfg = compile_with(transform_gpu_fftn_over_every_axis, gpu_implementation(), gpu=True)
    sdfg.name = f"{sdfg.name}_{len(shape)}d"
    got = run_without_fallback(sdfg, x=x.copy())
    np.testing.assert_allclose(got, np.fft.fftn(x), rtol=1e-12, atol=1e-10)


@pytest.mark.fftw
def test_a_batch_of_two_dense_axes_goes_to_fftw3_as_one_dimension():
    """MKL's FFTW3 symbols, once an MKL program loaded them, plan one batch dimension at most."""

    @dace.program
    def transform_a_batch_of_two_dense_axes(x: dace.complex128[4, 6, 8]):
        return np.fft.fft(x, axis=-1)

    sdfg = compile_with(transform_a_batch_of_two_dense_axes, "FFTW3")
    sdfg.expand_library_nodes()
    code = "".join(n.code.as_string for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet))
    assert "fftw_iodim64 __howmany[1] = {{24, 8, 8}};" in code
    x = rng_complex((4, 6, 8))
    np.testing.assert_allclose(sdfg(x=x.copy()), np.fft.fft(x, axis=-1), rtol=1e-12, atol=1e-12)


@pytest.mark.fftw
@pytest.mark.parametrize("shape,axes", [((3, 4, 5, 2), (1, 2, 3)), ((4, 5, 6), (0, 2)), ((4, 5, 6), (-1, -3)),
                                        ((3, 4, 5, 2), (0, 1, 2, 3))])
@pytest.mark.parametrize("dtype,dace_type", [(np.complex128, dace.complex128), (np.complex64, dace.complex64)])
def test_fftn_and_ifftn_over_any_axes_take_one_fftw3_plan(shape, axes, dtype, dace_type):
    """Any axis set is one ``fftw_plan_guru64_dft``, which steps each transformed and batch axis by the strides."""

    @dace.program
    def transforms_over_any_axes(x: dace_type[tuple(shape)]):
        return np.fft.fftn(x, axes=axes), np.fft.ifftn(x, axes=axes, norm="forward")

    sdfg = compile_with(transforms_over_any_axes, "FFTW3")
    sdfg.name = f"fftn_axes_{'_'.join(str(a % len(shape)) for a in axes)}_of_{len(shape)}d_{dtype.__name__}"
    x = rng_complex(shape, dtype)
    forward, inverse = run_without_fallback(sdfg, x=x.copy())
    tolerance = 1e-4 if dtype == np.complex64 else 1e-12
    np.testing.assert_allclose(forward, np.fft.fftn(x, axes=axes), rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(inverse, np.fft.ifftn(x, axes=axes, norm="forward"), rtol=tolerance, atol=tolerance)


def test_a_threaded_companion_library_is_linked_ahead_of_its_base(tmp_path):
    """FFTW's threaded planner ships as ``libfftw3_omp`` / ``libfftw3_threads`` beside ``libfftw3``; without it the
    transform ran on one thread against numba's threaded pocketfft (fft_1d, 10x slower)."""
    (tmp_path / "include").mkdir()
    (tmp_path / "include" / "fftw3.h").touch()
    (tmp_path / "lib").mkdir()
    for lib in ("libfftw3.so", "libfftw3_threads.so", "libfftw3_omp.so", "libfftw3f.so"):
        (tmp_path / "lib" / lib).touch()
    names = [p.rsplit("/", 1)[1] for p in fftw3.prefix_libraries(str(tmp_path))]
    assert names == ["libfftw3_omp.so", "libfftw3.so", "libfftw3f.so"]


@pytest.mark.parametrize("threaded", [True, False])
def test_a_threaded_fftw_plans_for_the_team(monkeypatch, threaded):
    monkeypatch.setattr(fftw3, "threaded", lambda: threaded)
    desc = dace.data.Array(dace.complex128, [64])
    code = fftw3_call(Transform(desc, desc, [0], inverse=False, half_spectrum=False)).code.as_string
    assert ("fftw_plan_with_nthreads(__threads)" in code) == threaded


def test_an_fftw3_plan_lives_in_the_sdfg_state():
    """Tal's point on the PR: a plan made and destroyed on every call costs more than a small transform. The plan is
    a state field, made on the first call and kept while its key -- layout, aliasing, threads, alignment -- holds."""
    desc = dace.data.Array(dace.complex128, [64])
    tasklet = fftw3_call(Transform(desc, desc, [0], inverse=False, half_spectrum=False))
    plan = tasklet.state_fields[0].split()[1].rstrip(";")
    assert tasklet.state_fields[0] == f"fftw_plan {plan};"
    assert f"__state->{plan}_made = false;" in tasklet.code_init.as_string
    assert f"fftw_destroy_plan(__state->{plan})" in tasklet.code_exit.as_string
    code = tasklet.code.as_string
    assert f"__plan = __state->{plan};" in code
    assert code.count("fftw_plan_guru64_dft(") == 1


@pytest.mark.fftw
def test_fftw3_transforms_inside_a_parallel_map_match_numpy():
    """Every thread of a parallel map runs the one tasklet: the cached plan is shared, and only planning is
    serialized."""

    @dace.program
    def transforms_inside_a_parallel_map(x: dace.complex128[8, 64], y: dace.complex128[8, 64]):
        for i in dace.map[0:8]:
            y[i] = np.fft.fft(x[i])

    x = rng_complex((8, 64))
    y = np.zeros_like(x)
    sdfg = compile_with(transforms_inside_a_parallel_map, "FFTW3")
    run_without_fallback(sdfg, x=x.copy(), y=y)
    np.testing.assert_allclose(y, np.fft.fft(x, axis=-1), rtol=1e-12, atol=1e-10)


@pytest.mark.fftw
def test_fftw3_reuses_its_plan_across_calls_and_replans_for_new_extents():
    """One compiled program called three times, the third with other extents: the key mismatch replans."""

    @dace.program
    def transform_reuses_its_plan(x: dace.complex128[N]):
        return np.fft.fft(x)

    compiled = compile_with(transform_reuses_its_plan, "FFTW3").compile()
    for length in (64, 64, 48):
        x = rng_complex((length, ))
        np.testing.assert_allclose(compiled(x=x.copy(), N=length), np.fft.fft(x), rtol=1e-12, atol=1e-10)


def expansion_tasklets(program, implementation: str) -> list[str]:
    """The tasklet labels of the nested SDFG each FFT / IFFT node of ``program`` expands to, in state order."""
    sdfg = compile_with(program, implementation)
    sdfg.expand_library_nodes(recursive=False)
    nested = next(node.sdfg for node, _ in sdfg.all_nodes_recursive() if isinstance(node, dace.nodes.NestedSDFG))
    return [
        node.label for state in nested.bfs_nodes(nested.start_block) for node in state.nodes()
        if isinstance(node, dace.nodes.Tasklet)
    ]


@pytest.mark.fftw
def test_the_normalization_folds_into_the_staging_copy():
    """A real input is staged into the complex type anyway; the ``1/N`` of ``ifft`` scales that copy rather than
    adding a pass over the output. A complex input has no staging copy, so it keeps the separate pass."""

    @dace.program
    def inverse_of_a_real_input(x: dace.float64[N]):
        return np.fft.ifft(x)

    @dace.program
    def inverse_of_a_complex_input(x: dace.complex128[N]):
        return np.fft.ifft(x)

    assert expansion_tasklets(inverse_of_a_real_input, "FFTW3") == ["stage", "fftw3_ifft"]
    assert expansion_tasklets(inverse_of_a_complex_input, "FFTW3") == ["fftw3_ifft", "normalize"]
    x = np.random.default_rng(3).standard_normal(40)
    got = run_without_fallback(compile_with(inverse_of_a_real_input, "FFTW3"), x=x.copy(), N=40)
    np.testing.assert_allclose(got, np.fft.ifft(x), rtol=1e-12, atol=1e-12)
