from typing import TYPE_CHECKING

from dace import data, subsets, symbolic

if TYPE_CHECKING:
    from dace.codegen.py.framecode import DaCePythonCodeGenerator
    from dace.codegen.py.target import PythonTargetCodeGenerator


def python_symbolic(expr) -> str:
    return symbolic.symstr(expr, cpp_mode=False)


def _python_slice_component(start, end, step) -> str:
    try:
        start_int = int(start)
        end_int = int(end)
        step_int = int(step)
    except (TypeError, ValueError):
        start_int = end_int = step_int = None

    if step_int is not None:
        if step_int == 1 and start_int == end_int:
            return str(start_int)
        stop_expr = str(end_int + 1) if step_int > 0 else str(end_int - 1)
        if step_int == 1:
            return f'{start_int}:{stop_expr}'
        return f'{start_int}:{stop_expr}:{step_int}'

    start_expr = python_symbolic(start)
    end_expr = python_symbolic(end)
    step_expr = python_symbolic(step)
    if step_expr == '1' and start_expr == end_expr:
        return start_expr
    stop_expr = f'(({end_expr}) + (1 if ({step_expr}) > 0 else -1))'
    if step_expr == '1':
        return f'{start_expr}:{stop_expr}'
    return f'{start_expr}:{stop_expr}:{step_expr}'


def subset_to_python_indices(desc: data.Data, subset: subsets.Subset | None) -> str:
    if subset is None:
        return ''

    if hasattr(desc, 'offset') and any(str(offset) != '0' for offset in desc.offset):
        subset = subset.offset_new(desc.offset, False)

    if isinstance(subset, subsets.Range):
        parts = [_python_slice_component(start, end, step) for start, end, step in subset.ranges]
    else:
        raise NotImplementedError(f'Unsupported subset type for Python backend: {type(subset).__name__}')

    if len(parts) == 1:
        return parts[0]
    return ', '.join(parts)


def data_access_expression(name: str,
                           desc: data.Data,
                           subset: subsets.Subset | None = None,
                           scalar_buffer: bool = False,
                           is_write: bool = False) -> str:
    if isinstance(desc, data.Scalar):
        if scalar_buffer:
            # Scalar buffers are 0-d numpy arrays. Reads use ``[()]`` so the
            # value is a numpy scalar (usable in cupy expressions); write
            # targets use ``[...]`` so assignment mutates the buffer in place.
            return f'{name}[...]' if is_write else f'{name}[()]'
        return name

    indices = subset_to_python_indices(desc, subset)
    if not indices:
        return name
    return f'{name}[{indices}]'


def numpy_array_expression(sdfg,
                           memlet,
                           with_brackets=True,
                           offset=None,
                           relative_offset=True,
                           packed_veclen=1,
                           use_other_subset=False,
                           indices=None,
                           referenced_array=None,
                           codegen: 'PythonTargetCodeGenerator | None' = None,
                           framecode: 'DaCePythonCodeGenerator | None' = None):
    del offset, relative_offset, packed_veclen, indices, framecode
    subset = memlet.other_subset if use_other_subset else memlet.subset
    desc = sdfg.arrays[memlet.data] if referenced_array is None else referenced_array
    name = codegen.ptr(memlet.data, desc, sdfg, subset=subset) if codegen is not None else memlet.data
    if not with_brackets:
        return subset_to_python_indices(desc, subset)
    return data_access_expression(name, desc, subset)


def numpy_offset_expression(d: data.Data, subset_in: subsets.Subset) -> str:
    return subset_to_python_indices(d, subset_in)