import importlib
import pkgutil
import pytest

import dace
import numpy as np

from dace import dtypes
from dace.frontend.common import op_repository as oprepo
import dace.frontend.python.replacements as replacements_pkg

_KNOWN_OPTIONAL_DEPENDENCIES = {'torch', 'onnx'}
_REPLACEMENTS_IMPORTED = False


def _import_replacement_modules() -> None:
    global _REPLACEMENTS_IMPORTED
    if _REPLACEMENTS_IMPORTED:
        return

    for module in pkgutil.iter_modules(replacements_pkg.__path__):
        try:
            importlib.import_module(f'{replacements_pkg.__name__}.{module.name}')
        except ModuleNotFoundError as exc:
            missing = (exc.name or '').split('.')[0]
            if missing not in _KNOWN_OPTIONAL_DEPENDENCIES:
                raise
    _REPLACEMENTS_IMPORTED = True


def _is_numpy_ufunc_name(name: str) -> bool:
    parts = name.split('.')
    if len(parts) < 2 or parts[0] != 'numpy':
        return False

    value = np
    for part in parts[1:]:
        value = getattr(value, part, None)
        if value is None:
            return False
    return isinstance(value, np.ufunc)


def _function_has_inference_coverage(name: str) -> bool:
    if name in oprepo.Replacements._dtype_rep:
        return True
    if oprepo.Replacements.get_ufunc_descriptor_inference('ufunc') is None:
        return False
    return _is_numpy_ufunc_name(name)


def _method_has_inference_coverage(key) -> bool:
    return key in oprepo.Replacements._dtype_method_rep


def _attribute_has_inference_coverage(key) -> bool:
    return key in oprepo.Replacements._dtype_attr_rep


def _ufunc_has_inference_coverage(name: str) -> bool:
    return name in oprepo.Replacements._dtype_ufunc_rep


def _operator_has_inference_coverage(key) -> bool:
    left_class, right_class, optype = key
    return (left_class, right_class,
            optype) in oprepo.Replacements._dtype_op_rep or (None, None, optype) in oprepo.Replacements._dtype_op_rep


def test_ufunc_descriptor_registry_parity():
    _import_replacement_modules()
    assert set(oprepo.Replacements._ufunc_rep) == set(oprepo.Replacements._dtype_ufunc_rep)


def test_ufunc_descriptor_inference_shapes():
    _import_replacement_modules()
    infer_ufunc = oprepo.Replacements.get_ufunc_descriptor_inference('ufunc')
    infer_reduce = oprepo.Replacements.get_ufunc_descriptor_inference('reduce')
    infer_accumulate = oprepo.Replacements.get_ufunc_descriptor_inference('accumulate')
    infer_outer = oprepo.Replacements.get_ufunc_descriptor_inference('outer')

    left = dace.data.Array(dace.float32, [4, 1], transient=True)
    right = dace.data.Array(dace.float32, [1, 5], transient=True)
    vector = dace.data.Array(dace.float32, [4, 5], transient=True)

    add_result = infer_ufunc({'A': left, 'B': right}, 'add', 'A', 'B')
    assert isinstance(add_result, dace.data.Array)
    assert tuple(add_result.shape) == (4, 5)
    assert add_result.dtype == dace.float32

    divmod_result = infer_ufunc({'A': vector, 'B': vector}, 'divmod', 'A', 'B')
    assert isinstance(divmod_result, tuple)
    assert len(divmod_result) == 2
    assert all(isinstance(result, dace.data.Array) for result in divmod_result)
    assert all(tuple(result.shape) == (4, 5) for result in divmod_result)

    reduce_result = infer_reduce({'A': vector}, 'add', 'A')
    assert isinstance(reduce_result, dace.data.Array)
    assert tuple(reduce_result.shape) == (5, )

    accumulate_result = infer_accumulate({'A': vector}, 'add', 'A')
    assert isinstance(accumulate_result, dace.data.Array)
    assert tuple(accumulate_result.shape) == (4, 5)

    outer_result = infer_outer({'A': left, 'B': right}, 'add', 'A', 'B')
    assert isinstance(outer_result, dace.data.Array)
    assert tuple(outer_result.shape) == (4, 1, 1, 5)


def test_operator_descriptor_dispatch_uses_operand_categories():
    _import_replacement_modules()
    generic_matmul = oprepo.Replacements.get_operator_descriptor_inference(
        'MatMult', dace.data.Array(dace.float32, [4, 3], transient=True),
        dace.data.Array(dace.float32, [3, 2], transient=True))
    storage_cast = oprepo.Replacements.get_operator_descriptor_inference(
        'MatMult', dace.data.Array(dace.float32, [4], transient=True), dtypes.StorageType.GPU_Global)

    assert generic_matmul is not None
    assert storage_cast is not None
    assert generic_matmul is not storage_cast

    source = dace.data.Array(dace.float32, [4], transient=True, storage=dtypes.StorageType.Default)
    result = storage_cast(source, dtypes.StorageType.GPU_Global)
    assert isinstance(result, dace.data.Array)
    assert result.storage == dtypes.StorageType.GPU_Global
    assert tuple(result.shape) == (4, )


def test_recent_alias_and_method_inference_regressions():
    _import_replacement_modules()

    infer_conj = oprepo.Replacements.get_descriptor_inference('numpy.conj')
    infer_exp = oprepo.Replacements.get_descriptor_inference('exp')
    infer_floor = oprepo.Replacements.get_descriptor_inference('math.floor')
    infer_max = oprepo.Replacements.get_descriptor_inference('max')
    infer_min = oprepo.Replacements.get_descriptor_inference('min')
    infer_float32 = oprepo.Replacements.get_descriptor_inference('float32')
    infer_numpy_int16 = oprepo.Replacements.get_descriptor_inference('numpy.int16')
    infer_dace_bool = oprepo.Replacements.get_descriptor_inference('dace.bool')
    infer_cart_create = oprepo.Replacements.get_descriptor_inference('dace.comm.Cart_create')
    infer_cart_sub = oprepo.Replacements.get_descriptor_inference('dace.comm.Cart_sub')
    infer_clip = oprepo.Replacements.get_descriptor_inference('numpy.clip')
    infer_bcast = oprepo.Replacements.get_descriptor_inference('dace.comm.Bcast')
    infer_isend = oprepo.Replacements.get_descriptor_inference('dace.comm.Isend')
    infer_irecv = oprepo.Replacements.get_descriptor_inference('dace.comm.Irecv')
    infer_subarray = oprepo.Replacements.get_descriptor_inference('dace.comm.Subarray')
    infer_bcscatter = oprepo.Replacements.get_descriptor_inference('dace.comm.BCScatter')
    infer_distr_matmult = oprepo.Replacements.get_descriptor_inference('dace.distr.MatMult')
    infer_fft = oprepo.Replacements.get_descriptor_inference('numpy.fft.fft')
    infer_ifft = oprepo.Replacements.get_descriptor_inference('numpy.fft.ifft')
    infer_dot = oprepo.Replacements.get_descriptor_inference('numpy.dot')
    infer_einsum = oprepo.Replacements.get_descriptor_inference('numpy.einsum')
    infer_inv = oprepo.Replacements.get_descriptor_inference('numpy.linalg.inv')
    infer_rot90 = oprepo.Replacements.get_descriptor_inference('numpy.rot90')
    infer_solve = oprepo.Replacements.get_descriptor_inference('numpy.linalg.solve')
    infer_tensordot = oprepo.Replacements.get_descriptor_inference('numpy.tensordot')
    infer_cholesky = oprepo.Replacements.get_descriptor_inference('numpy.linalg.cholesky')
    infer_real = oprepo.Replacements.get_descriptor_inference('numpy.real')
    infer_full_like = oprepo.Replacements.get_descriptor_inference('numpy.full_like')
    infer_identity = oprepo.Replacements.get_descriptor_inference('numpy.identity')
    infer_select = oprepo.Replacements.get_descriptor_inference('numpy.select')
    infer_transpose = oprepo.Replacements.get_descriptor_inference('transpose')
    infer_where = oprepo.Replacements.get_descriptor_inference('numpy.where')
    infer_sum = oprepo.Replacements.get_descriptor_inference('sum')
    infer_intracomm_create_cart = oprepo.Replacements.get_method_descriptor_inference('Intracomm', 'Create_cart')
    infer_intracomm_allreduce = oprepo.Replacements.get_method_descriptor_inference('Intracomm', 'Allreduce')
    infer_processgrid_sub = oprepo.Replacements.get_method_descriptor_inference('ProcessGrid', 'Sub')
    infer_processgrid_isend = oprepo.Replacements.get_method_descriptor_inference('ProcessGrid', 'Isend')
    infer_slice = oprepo.Replacements.get_descriptor_inference('slice')
    infer_define_stream = oprepo.Replacements.get_descriptor_inference('dace.define_stream')
    infer_define_streamarray = oprepo.Replacements.get_descriptor_inference('dace.define_streamarray')
    infer_elementwise = oprepo.Replacements.get_descriptor_inference('dace.elementwise')
    infer_reduce = oprepo.Replacements.get_descriptor_inference('dace.reduce')
    infer_cupy_full = oprepo.Replacements.get_descriptor_inference('cupy.full')
    infer_cupy_empty_like = oprepo.Replacements.get_descriptor_inference('cupy.empty_like')
    infer_fill = oprepo.Replacements.get_method_descriptor_inference('Array', 'fill')
    infer_view = oprepo.Replacements.get_method_descriptor_inference('Array', 'view')

    complex_vector = dace.data.Array(dace.complex64, [4], transient=True)
    cond = dace.data.Array(dace.bool_, [2, 1], transient=True)
    matrix = dace.data.Array(dace.float32, [2, 3], transient=True)
    square = dace.data.Array(dace.float64, [4, 4], transient=True)
    rhs = dace.data.Array(dace.float64, [4], transient=True)

    conj_result = infer_conj({'A': complex_vector}, 'A')
    assert isinstance(conj_result, dace.data.Array)
    assert conj_result.dtype == dace.complex64
    assert tuple(conj_result.shape) == (4, )

    exp_result = infer_exp({'A': matrix}, 'A')
    assert isinstance(exp_result, dace.data.Array)
    assert exp_result.dtype == dace.float32
    assert tuple(exp_result.shape) == (2, 3)

    floor_result = infer_floor({'A': matrix}, 'A')
    assert isinstance(floor_result, dace.data.Array)
    assert floor_result.dtype == dtypes.typeclass(int)
    assert tuple(floor_result.shape) == (2, 3)

    float32_result = infer_float32({'A': matrix}, 'A')
    assert isinstance(float32_result, dace.data.Array)
    assert float32_result.dtype == dace.float32
    assert tuple(float32_result.shape) == (2, 3)

    numpy_int16_result = infer_numpy_int16({'A': matrix}, 'A')
    assert isinstance(numpy_int16_result, dace.data.Array)
    assert numpy_int16_result.dtype == dace.int16
    assert tuple(numpy_int16_result.shape) == (2, 3)

    cart_create_result = infer_cart_create({}, [2, 2])
    assert isinstance(cart_create_result, dace.data.Scalar)
    assert isinstance(cart_create_result.dtype, dtypes.pyobject)

    cart_sub_result = infer_cart_sub({}, 'pgrid', [True, False])
    assert isinstance(cart_sub_result, dace.data.Scalar)
    assert isinstance(cart_sub_result.dtype, dtypes.pyobject)

    dace_bool_result = infer_dace_bool({'A': matrix}, 'A')
    assert isinstance(dace_bool_result, dace.data.Array)
    assert dace_bool_result.dtype == dace.bool_
    assert tuple(dace_bool_result.shape) == (2, 3)

    assert infer_bcast({'A': matrix}, 'A') == ()

    isend_result = infer_isend({'A': matrix}, 'A', 0, 0)
    assert isinstance(isend_result, dace.data.Array)
    assert tuple(isend_result.shape) == (1, )
    assert isinstance(isend_result.dtype, dtypes.opaque)

    assert infer_isend({'A': matrix, 'req': isend_result}, 'A', 0, 0, request='req') == ()

    irecv_result = infer_irecv({'A': matrix}, 'A', 0, 0)
    assert isinstance(irecv_result, dace.data.Array)
    assert tuple(irecv_result.shape) == (1, )
    assert isinstance(irecv_result.dtype, dtypes.opaque)

    subarray_result = infer_subarray({'A': square}, 'A', [2, 2])
    assert isinstance(subarray_result, dace.data.Scalar)
    assert isinstance(subarray_result.dtype, dtypes.pyobject)

    bcscatter_result = infer_bcscatter({'A': square, 'B': square}, 'A', 'B', [2, 2])
    assert isinstance(bcscatter_result, tuple)
    assert len(bcscatter_result) == 2
    assert all(isinstance(result, dace.data.Array) for result in bcscatter_result)
    assert all(result.dtype == dace.int32 for result in bcscatter_result)
    assert all(tuple(result.shape) == (9, ) for result in bcscatter_result)

    distr_matmult_result = infer_distr_matmult({'A': square, 'B': square}, 'A', 'B', (4, 4, 4))
    assert isinstance(distr_matmult_result, dace.data.Array)
    assert distr_matmult_result.dtype == dace.float64
    assert tuple(distr_matmult_result.shape) == (4, 4)

    clip_result = infer_clip({'A': matrix}, 'A', 1.0, 3.0)
    assert isinstance(clip_result, dace.data.Array)
    assert clip_result.dtype == dace.float32
    assert tuple(clip_result.shape) == (2, 3)

    clip_max_only_result = infer_clip({'A': matrix}, 'A', None, 3.0)
    assert isinstance(clip_max_only_result, dace.data.Array)
    assert clip_max_only_result.dtype == dace.float32
    assert tuple(clip_max_only_result.shape) == (2, 3)

    fft_result = infer_fft({'A': matrix}, 'A')
    assert isinstance(fft_result, dace.data.Array)
    assert fft_result.dtype == dace.complex64
    assert tuple(fft_result.shape) == (2, 3)

    dot_result = infer_dot({'A': square, 'B': square}, 'A', 'B')
    assert isinstance(dot_result, dace.data.Array)
    assert dot_result.dtype == dace.float64
    assert tuple(dot_result.shape) == (4, 4)

    einsum_result = infer_einsum({'A': square, 'B': square}, 'ik,kj->ij', 'A', 'B')
    assert isinstance(einsum_result, dace.data.Array)
    assert einsum_result.dtype == dace.float64
    assert tuple(einsum_result.shape) == (4, 4)

    dim_a, dim_b, dim_c, dim_d, dim_e = (dace.symbol(name) for name in ('dim_a', 'dim_b', 'dim_c', 'dim_d', 'dim_e'))
    multi_contract_left = dace.data.Array(dace.float64, [dim_a, dim_b, dim_c, dim_d], transient=True)
    multi_contract_right = dace.data.Array(dace.float64, [dim_b, dim_d, dim_c, dim_e], transient=True)
    multi_contract_result = infer_einsum({
        'A': multi_contract_left,
        'B': multi_contract_right
    }, 'abcd,bdce->ae', 'A', 'B')
    assert isinstance(multi_contract_result, dace.data.Array)
    assert multi_contract_result.dtype == dace.float64
    assert tuple(multi_contract_result.shape) == (dim_a, dim_e)

    vec_extent = dace.symbol('vec_extent')
    symbolic_vector = dace.data.Array(dace.float64, [vec_extent], transient=True)
    repeated_index_result = infer_einsum({'A': symbolic_vector}, 'i->ii', 'A')
    assert isinstance(repeated_index_result, dace.data.Array)
    assert repeated_index_result.dtype == dace.float64
    assert tuple(repeated_index_result.shape) == (vec_extent, vec_extent)

    reduced_extent = dace.symbol('reduced_extent')
    kept_extent = dace.symbol('kept_extent')
    contracted_input_vector = dace.data.Array(dace.float64, [reduced_extent], transient=True)
    retained_input_vector = dace.data.Array(dace.float64, [kept_extent], transient=True)
    contracted_input_result = infer_einsum({
        'A': contracted_input_vector,
        'B': retained_input_vector
    }, 'j,k->k', 'A', 'B')
    assert isinstance(contracted_input_result, dace.data.Array)
    assert contracted_input_result.dtype == dace.float64
    assert tuple(contracted_input_result.shape) == (kept_extent, )

    complex_matrix = dace.data.Array(dace.complex64, [2, 3], transient=True)
    ifft_result = infer_ifft({'A': complex_matrix}, 'A')
    assert isinstance(ifft_result, dace.data.Array)
    assert ifft_result.dtype == dace.complex64
    assert tuple(ifft_result.shape) == (2, 3)

    rot90_result = infer_rot90({'A': matrix}, 'A')
    assert isinstance(rot90_result, dace.data.Array)
    assert rot90_result.dtype == dace.float32
    assert tuple(rot90_result.shape) == (3, 2)

    inv_result = infer_inv({'A': square}, 'A')
    assert isinstance(inv_result, dace.data.Array)
    assert inv_result.dtype == dace.float64
    assert tuple(inv_result.shape) == (4, 4)

    solve_result = infer_solve({'A': square, 'B': rhs}, 'A', 'B')
    assert isinstance(solve_result, dace.data.Array)
    assert solve_result.dtype == dace.float64
    assert tuple(solve_result.shape) == (4, )

    tensordot_result = infer_tensordot({'A': square, 'B': square}, 'A', 'B', axes=1)
    assert isinstance(tensordot_result, dace.data.Array)
    assert tensordot_result.dtype == dace.float64
    assert tuple(tensordot_result.shape) == (4, 4)

    cholesky_result = infer_cholesky({'A': square}, 'A')
    assert isinstance(cholesky_result, dace.data.Array)
    assert cholesky_result.dtype == dace.float64
    assert tuple(cholesky_result.shape) == (4, 4)

    real_result = infer_real({'A': complex_vector}, 'A')
    assert isinstance(real_result, dace.data.Array)
    assert real_result.dtype == dace.float32
    assert tuple(real_result.shape) == (4, )

    full_like_result = infer_full_like({'A': matrix}, 'A', 1.0)
    assert isinstance(full_like_result, dace.data.Array)
    assert full_like_result.dtype == dace.float32
    assert tuple(full_like_result.shape) == (2, 3)

    identity_result = infer_identity({}, 5, dtype=np.float64)
    assert isinstance(identity_result, dace.data.Array)
    assert identity_result.dtype == dace.float64
    assert tuple(identity_result.shape) == (5, 5)

    transpose_result = infer_transpose({'A': matrix}, 'A')
    assert isinstance(transpose_result, dace.data.Array)
    assert tuple(transpose_result.shape) == (3, 2)

    where_result = infer_where({'cond': cond, 'A': matrix}, 'cond', 'A', 1.0)
    assert isinstance(where_result, dace.data.Array)
    assert where_result.dtype == dace.float32
    assert tuple(where_result.shape) == (2, 3)

    assert infer_where({'cond': dace.data.Scalar(dace.bool_, transient=True)}, 'cond', 1, 2) is None

    select_result = infer_select({'cond': cond, 'A': matrix}, ['cond'], ['A'], default=1.0)
    assert isinstance(select_result, dace.data.Array)
    assert select_result.dtype == dace.float32
    assert tuple(select_result.shape) == (2, 3)

    pyobject_self = dace.data.Scalar(dtypes.pyobject(), transient=True)
    intracomm_create_result = infer_intracomm_create_cart(pyobject_self, [2, 2])
    assert isinstance(intracomm_create_result, dace.data.Scalar)
    assert isinstance(intracomm_create_result.dtype, dtypes.pyobject)

    assert infer_intracomm_allreduce(pyobject_self, None, 'A', 'MPI_SUM') == ()

    processgrid_sub_result = infer_processgrid_sub(pyobject_self, [True, False])
    assert isinstance(processgrid_sub_result, dace.data.Scalar)
    assert isinstance(processgrid_sub_result.dtype, dtypes.pyobject)

    processgrid_isend_result = infer_processgrid_isend(pyobject_self, 'A', 0, 0)
    assert isinstance(processgrid_isend_result, dace.data.Array)
    assert tuple(processgrid_isend_result.shape) == (1, )
    assert isinstance(processgrid_isend_result.dtype, dtypes.opaque)

    slice_result = infer_slice({}, 0, 5, 2)
    assert isinstance(slice_result, tuple)
    assert len(slice_result) == 1
    assert isinstance(slice_result[0], dace.data.Scalar)
    assert isinstance(slice_result[0].dtype, dtypes.pyobject)

    define_stream_result = infer_define_stream({}, dace.float32, buffer_size=4)
    assert isinstance(define_stream_result, dace.data.Stream)
    assert define_stream_result.dtype == dace.float32
    assert tuple(define_stream_result.shape) == (1, )
    assert define_stream_result.buffer_size == 4

    define_streamarray_result = infer_define_streamarray({}, [2, 3], dace.float64, buffer_size=8)
    assert isinstance(define_streamarray_result, dace.data.Stream)
    assert define_streamarray_result.dtype == dace.float64
    assert tuple(define_streamarray_result.shape) == (2, 3)
    assert define_streamarray_result.buffer_size == 8

    elementwise_result = infer_elementwise({'A': matrix}, 'lambda x: x + 1', 'A')
    assert isinstance(elementwise_result, dace.data.Array)
    assert elementwise_result.dtype == dace.float32
    assert tuple(elementwise_result.shape) == (2, 3)

    reduce_result = infer_reduce({'A': matrix}, 'lambda x, y: x + y', 'A', axis=1)
    assert isinstance(reduce_result, dace.data.Array)
    assert reduce_result.dtype == dace.float32
    assert tuple(reduce_result.shape) == (2, )

    assert infer_reduce({
        'A': matrix,
        'B': dace.data.Array(dace.float32, [3], transient=True)
    },
                        'lambda x, y: x + y',
                        'A',
                        out_array='B',
                        axis=0) == ()

    cupy_full_result = infer_cupy_full({}, [4, 2], 3.0)
    assert isinstance(cupy_full_result, dace.data.Array)
    assert cupy_full_result.dtype == dace.float64
    assert tuple(cupy_full_result.shape) == (4, 2)
    assert cupy_full_result.storage == dtypes.StorageType.GPU_Global

    cupy_empty_like_result = infer_cupy_empty_like({'A': matrix}, 'A', dtype=dace.float16, shape=[3, 4])
    assert isinstance(cupy_empty_like_result, dace.data.Array)
    assert cupy_empty_like_result.dtype == dace.float16
    assert tuple(cupy_empty_like_result.shape) == (3, 4)
    assert cupy_empty_like_result.storage == dtypes.StorageType.GPU_Global

    sum_result = infer_sum({'A': matrix}, 'A')
    assert isinstance(sum_result, dace.data.Array)
    assert tuple(sum_result.shape) == (3, )

    max_result = infer_max({}, 1, 2.0, np.float32(3.0))
    assert isinstance(max_result, dace.data.Scalar)
    assert max_result.dtype == dace.float32

    min_result = infer_min({'x': dace.data.Scalar(dace.int32, transient=True)}, 'x', 5)
    assert isinstance(min_result, dace.data.Scalar)
    assert min_result.dtype == dace.int32

    assert infer_fill(matrix, 7) == ()

    view_result = infer_view(dace.data.Array(dace.float32, [4], transient=True), np.float16)
    assert isinstance(view_result, dace.data.View)
    assert view_result.dtype == dace.float16
    assert tuple(view_result.shape) == (8, )


def test_runtime_registry_entries_have_inference_coverage_or_allowlisted_gap():
    _import_replacement_modules()

    missing_functions = [name for name in oprepo.Replacements._rep if not _function_has_inference_coverage(name)]
    missing_methods = [key for key in oprepo.Replacements._method_rep if not _method_has_inference_coverage(key)]
    missing_attributes = [key for key in oprepo.Replacements._attr_rep if not _attribute_has_inference_coverage(key)]
    missing_ufuncs = [name for name in oprepo.Replacements._ufunc_rep if not _ufunc_has_inference_coverage(name)]
    missing_operators = [key for key in oprepo.Replacements._oprep if not _operator_has_inference_coverage(key)]

    assert missing_functions == [], f'uncovered function inference registrations: {missing_functions}'
    assert missing_methods == [], f'uncovered method inference registrations: {missing_methods}'
    assert missing_attributes == [], f'uncovered attribute inference registrations: {missing_attributes}'
    assert missing_ufuncs == [], f'uncovered ufunc inference registrations: {missing_ufuncs}'
    assert missing_operators == [], f'uncovered operator inference registrations: {missing_operators}'


if __name__ == '__main__':
    pytest.main([__file__])
