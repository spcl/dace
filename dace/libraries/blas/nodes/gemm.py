# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from copy import deepcopy as dc
from typing import Any, Dict, Optional, Tuple
from dace import dtypes, memlet as mm, properties, data as dt
from dace.symbolic import symstr, equal, equal_valued
import dace.library
from dace import SDFG, SDFGState
from dace.frontend.common import op_repository as oprepo
import dace.sdfg.nodes
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.blas.blas_helpers import to_blastype, check_access, dtype_to_cudadatatype, to_cublas_computetype
from dace.libraries.blas.nodes.matmul import (_get_matmul_operands, _get_codegen_gemm_opts)
from .. import environments
import numpy as np
import warnings


def _is_complex(dtype):
    if hasattr(dtype, "is_complex") and callable(dtype.is_complex):
        return dtype.is_complex()
    else:
        return dtype in [np.complex64, np.complex128]


def _cast_to_dtype_str(value, dtype: dace.dtypes.typeclass) -> str:
    if _is_complex(dtype) and _is_complex(type(value)):
        raise ValueError("Cannot use complex beta with non-complex array")

    if _is_complex(dtype):
        cast_value = complex(value)

        return "dace.{type}({real}, {imag})".format(
            type=dace.dtype_to_typeclass(dtype).to_string(),
            real=cast_value.real,
            imag=cast_value.imag,
        )
    else:
        return "dace.{}({})".format(dace.dtype_to_typeclass(dtype).to_string(), value)


def _coeff_conn_descs(node: "Gemm", state: SDFGState, sdfg: SDFG) -> Dict[str, dt.Data]:
    """Descriptors of the wired runtime coefficient connectors (``_alpha`` / ``_beta``), keyed by
    connector name.

    A coefficient supplied at runtime (a symbol or a compile-time constant is *not* a connector)
    arrives as a scalar the caller has placed either on the host (a CPU scalar, used with cuBLAS host
    pointer mode) or on the device (a length-1 GPU array, used with device pointer mode). Mirrors
    ``symm._scalar_conn_descs``.

    :param node: The ``Gemm`` library node being expanded.
    :param state: The state that contains ``node``.
    :param sdfg: The SDFG that contains ``state``.
    :return: A mapping from each wired coefficient connector name to its data descriptor.
    """
    return {e.dst_conn: sdfg.arrays[e.data.data] for e in state.in_edges(node) if e.dst_conn in ("_alpha", "_beta")}


def _host_value_expr(conn: str) -> str:
    """C expression that reads the value of a host coefficient connector.

    A runtime coefficient is wired as a single-element access (``_alpha[0]``) of a host ``Scalar`` or
    a host length-1 array. Either way the connector reaches the expansion by value (a scalar
    reference, ``const T&``), so the connector name is the value and is used directly -- a ``[0]``
    subscript would be applied to a non-array reference and fails to compile.

    :param conn: The tasklet-visible connector name of the coefficient.
    :return: A C expression evaluating to the scalar coefficient value.
    """
    return conn


def _host_coeff(var: str, conn: str, prop: Any, desc: Optional[dt.Data], arr_prefix: str, dtype: dtypes.typeclass,
                cdtype: str) -> Tuple[str, str]:
    """Render a host-pointer-mode coefficient for the GPU BLAS call.

    A wired runtime host scalar is read by value and composed multiplicatively with the compile-time
    ``prop`` (the identity in the usual runtime case); otherwise ``prop`` is inlined (the existing
    behavior). cuBLAS/rocBLAS always dereference the coefficient through a pointer, so the value is
    materialized into a host local and its address is passed.

    :param var: Name of the host local to declare (for example ``__alpha``).
    :param conn: The coefficient connector name (for example ``_alpha``), without the copy prefix.
    :param prop: The compile-time coefficient property (number, complex, or symbol).
    :param desc: The wired runtime scalar descriptor, or ``None`` when the coefficient is compile-time.
    :param arr_prefix: Connector-name prefix used when A/B/C are staged to the GPU (``_conn`` or ``''``).
    :param dtype: The element type of the coefficient.
    :param cdtype: The C type name the BLAS call expects the pointer cast to.
    :return: A ``(declaration, pointer_argument)`` pair for the emitted code.
    """
    if desc is not None:
        val = _host_value_expr(f"{arr_prefix}{conn}")
        if equal_valued(1, prop):
            rhs = val
        elif _is_complex(dtype):
            raise NotImplementedError(f"Gemm GPU host pointer mode: a non-unit complex coefficient "
                                      f"composed with a runtime {conn} scalar is unsupported.")
        else:
            rhs = f"{dtype.ctype}({prop}) * {val}"
        decl = f"{dtype.ctype} {var} = {rhs};\n"
    elif isinstance(prop, complex):
        decl = f"{dtype.ctype} {var} = {dtype.ctype}({prop.real}, {prop.imag});\n"
    else:
        decl = f"{dtype.ctype} {var} = {dtype.ctype}({prop});\n"
    return decl, f"({cdtype} *)&{var}"


def _device_coeff(conn: str, prop: Any, desc: Optional[dt.Data], arr_prefix: str, cdtype: str,
                  constants: Dict[float, str]) -> str:
    """Render a device-pointer-mode coefficient for the GPU BLAS call.

    A wired runtime GPU length-1 array is passed straight through as a device pointer (never copied
    host to device). A compile-time ``0``/``1`` uses the preallocated device constant so it can
    coexist with a device-scalar partner under the single handle-wide device pointer mode. Scaling a
    device scalar host-side, or materializing an arbitrary compile-time constant as a device scalar,
    is out of scope for this prototype.

    :param conn: The coefficient connector name (for example ``_alpha``), without the copy prefix.
    :param prop: The compile-time coefficient property (number, complex, or symbol).
    :param desc: The wired runtime scalar descriptor, or ``None`` when the coefficient is compile-time.
    :param arr_prefix: Connector-name prefix used when A/B/C are staged to the GPU (``_conn`` or ``''``).
    :param cdtype: The C type name the BLAS call expects the pointer cast to.
    :param constants: Mapping from ``0.0``/``1.0`` to the corresponding preallocated device constant.
    :return: The device-pointer argument for the emitted code.
    """
    if desc is not None:
        if not equal_valued(1, prop):
            raise NotImplementedError(f"Gemm GPU device pointer mode: a runtime {conn} device scalar "
                                      f"cannot be scaled host-side by a non-unit coefficient ({prop}).")
        return f"({cdtype} *){arr_prefix}{conn}"
    if prop in constants:
        return constants[prop]
    raise NotImplementedError(f"Gemm GPU device pointer mode: compile-time {conn}={prop} would need a "
                              f"device scalar; only 0/1 are supported next to a device-scalar coefficient.")


def _cblas_coeff(var: str, conn: str, prop: Any, desc: Optional[dt.Data], dtype: dtypes.typeclass) -> str:
    """Render a host declaration of an effective CBLAS coefficient.

    The declared local holds the compile-time ``prop``, times the wired runtime host scalar when one
    is present (composed multiplicatively, mirroring ``symm._coeff_decl``).

    :param var: Name of the host local to declare (for example ``__alpha``).
    :param conn: The coefficient connector name (for example ``_alpha``).
    :param prop: The compile-time coefficient property (number, complex, or symbol).
    :param desc: The wired runtime scalar descriptor, or ``None`` when the coefficient is compile-time.
    :param dtype: The element type of the coefficient.
    :return: A C declaration statement for the effective coefficient local.
    """
    if isinstance(prop, complex):
        lit = f"{dtype.ctype}({prop.real}, {prop.imag})"
    else:
        lit = f"{dtype.ctype}({prop})"
    if desc is None:
        return f"{dtype.ctype} {var} = {lit};\n"
    val = _host_value_expr(conn)
    if equal_valued(1, prop):
        return f"{dtype.ctype} {var} = {val};\n"
    return f"{dtype.ctype} {var} = {lit} * {val};\n"


@dace.library.expansion
class ExpandGemmPure(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg, rowwise=False):
        sdfg = dace.SDFG(node.label + "_sdfg")

        ((edge_a, outer_array_a, shape_a, strides_a, _, _), (edge_b, outer_array_b, shape_b, strides_b, _, _),
         cdata) = _get_matmul_operands(node, parent_state, parent_sdfg)

        dtype_a = outer_array_a.dtype.type
        dtype_b = outer_array_b.dtype.type
        dtype_c = dace.dtype_to_typeclass(np.result_type(dtype_a, dtype_b).type)

        if node.transA:
            trans_shape_a = list(reversed(shape_a))
        else:
            trans_shape_a = shape_a

        if node.transB:
            trans_shape_b = list(reversed(shape_b))
        else:
            trans_shape_b = shape_b

        if len(trans_shape_a) != 2 or len(trans_shape_b) != 2:
            raise SyntaxError("Matrix sizes must match")
        res = equal(trans_shape_a[1], trans_shape_b[0])
        if res is None:
            warnings.warn(
                f"First matrix columns {trans_shape_a[1]} may not match "
                f"second matrix rows {trans_shape_b[0]}", UserWarning)
        elif not res:
            raise SyntaxError("Matrix sizes must match")
        M, K, N = trans_shape_a[0], trans_shape_a[1], trans_shape_b[1]
        shape_c = (M, N)

        storage = outer_array_a.storage

        _, array_a = sdfg.add_array("_a", shape_a, dtype_a, strides=strides_a, storage=outer_array_a.storage)
        _, array_b = sdfg.add_array("_b", shape_b, dtype_b, strides=strides_b, storage=outer_array_b.storage)
        _, array_c = sdfg.add_array("_c", shape_c, dtype_c, strides=cdata[-3], storage=cdata[1].storage)

        # Runtime coefficients: a wired ``_alpha`` / ``_beta`` scalar connector is added as a [1]
        # input array and folded multiplicatively into the scaling tasklets (mirroring Symm). This
        # keeps the reference lowering correct when alpha/beta arrive as runtime values rather than
        # compile-time properties.
        scalars = _coeff_conn_descs(node, parent_state, parent_sdfg)
        for conn, desc in scalars.items():
            sdfg.add_array(conn, [1], desc.dtype.base_type, storage=desc.storage)
        rt_alpha = "_alpha" in scalars
        rt_beta = "_beta" in scalars

        if rt_alpha:
            if equal_valued(1, node.alpha):
                mul_program = "__out = __alpha * __a * __b"
            else:
                mul_program = "__out = {} * __alpha * __a * __b".format(_cast_to_dtype_str(node.alpha, dtype_a))
        elif equal_valued(1, node.alpha):
            mul_program = "__out = __a * __b"
        else:
            mul_program = "__out = {} * __a * __b".format(_cast_to_dtype_str(node.alpha, dtype_a))

        if equal_valued(1, node.beta) and not rt_beta:
            state = sdfg.add_state(node.label + "_state")
        else:
            init_state = sdfg.add_state(node.label + "_initstate")
            state = sdfg.add_state_after(init_state, node.label + "_state")

        mul_out, mul_out_array = "_c", array_c
        output_nodes = None

        # Initialization / beta map
        if rt_beta:
            # Runtime beta: always take the scale path (its value is unknown at build time),
            # composing the ``_beta`` connector with any compile-time ``beta`` property.
            if equal_valued(1, node.beta):
                add_program = "__y = (__beta * __c)"
            else:
                add_program = "__y = ({} * __beta * __c)".format(_cast_to_dtype_str(node.beta, dtype_a))
            if list(shape_c) == [M, N]:
                memlet_idx = '__i0, __i1'
            elif list(shape_c) == [1, N]:
                memlet_idx = '0, __i1'
            elif list(shape_c) == [M, 1]:
                memlet_idx = '__i0, 0'
            elif list(shape_c) == [N]:
                memlet_idx = '__i1'
            else:
                raise ValueError("Could not broadcast input _c to ({}, {})".format(M, N))
            init_state.add_mapped_tasklet("gemm_init", {
                "__i%d" % i: "0:%s" % s
                for i, s in enumerate([M, N])
            }, {
                "__c": dace.Memlet.simple("_c", memlet_idx),
                "__beta": dace.Memlet.simple("_beta", "0"),
            },
                                          add_program, {"__y": dace.Memlet.simple("_c", "__i0, __i1")},
                                          external_edges=True)
        elif equal_valued(0, node.beta):
            init_state.add_mapped_tasklet(
                'gemm_init', {
                    '_o%d' % i: '0:%s' % symstr(d)
                    for i, d in enumerate(shape_c)
                }, {},
                'out = 0', {'out': dace.Memlet.simple(mul_out, ','.join(['_o%d' % i for i in range(len(shape_c))]))},
                external_edges=True)
        elif equal_valued(1, node.beta):
            # Do nothing for initialization, only update the values
            pass
        else:
            # Beta map
            add_program = "__y = ({} * __c)".format(_cast_to_dtype_str(node.beta, dtype_a))

            # manually broadcasting C to [M, N]
            if list(shape_c) == [M, N]:
                memlet_idx = '__i0, __i1'
            elif list(shape_c) == [1, N]:
                memlet_idx = '0, __i1'
            elif list(shape_c) == [M, 1]:
                memlet_idx = '__i0, 0'
            elif list(shape_c) == [N]:
                memlet_idx = '__i1'
            else:
                raise ValueError("Could not broadcast input _c to ({}, {})".format(M, N))

            init_state.add_mapped_tasklet("gemm_init", {
                "__i%d" % i: "0:%s" % s
                for i, s in enumerate([M, N])
            }, {
                "__c": dace.Memlet.simple("_c", memlet_idx),
            },
                                          add_program, {"__y": dace.Memlet.simple("_c", "__i0, __i1")},
                                          external_edges=True)

        # Multiplication map. The default (``rowwise=False``) form is a single 3D
        # ``(M, N, K)`` map whose K axis is the per-element WCR contraction. The row-wise
        # (ikj) form instead puts K as the MIDDLE param ``(M, K, N)`` and then peels the map
        # apart with MapExpansion, making the inner (k, j) maps Sequential: this yields
        # ``for i (parallel): for k (seq): for j (vector)`` -- i.e. ``C[i,:] += A[i,k]*B[k,:]``,
        # a vectorizable row update with a sequential K accumulation -- instead of a per-element
        # inner-K reduction. Used by canonicalize for known-small GEMMs, where a loop of tiny
        # BLAS calls is pure overhead and a fusible/vectorizable nest is faster.
        if rowwise:
            _, mult_entry, _ = state.add_mapped_tasklet(
                "gemm", {
                    "__i%d" % i: "0:%s" % s
                    for i, s in enumerate([M, K, N])
                }, {
                    "__a": dace.Memlet.simple("_a", "__i1, __i0" if node.transA else "__i0, __i1"),
                    "__b": dace.Memlet.simple("_b", "__i2, __i1" if node.transB else "__i1, __i2"),
                    **({
                        "__alpha": dace.Memlet.simple("_alpha", "0")
                    } if rt_alpha else {}),
                },
                mul_program, {"__out": dace.Memlet.simple(mul_out, "__i0, __i2", wcr_str="lambda x, y: x + y")},
                external_edges=True,
                output_nodes=output_nodes)
            # Peel into i / k / j maps; inner (k, j) become Sequential (the MapExpansion
            # default), leaving the outer i-map parallel.
            from dace.transformation.dataflow.map_expansion import MapExpansion
            MapExpansion.apply_to(sdfg, verify=False, map_entry=mult_entry)
        else:
            state.add_mapped_tasklet("gemm", {
                "__i%d" % i: "0:%s" % s
                for i, s in enumerate([M, N, K])
            }, {
                "__a": dace.Memlet.simple("_a", "__i2, __i0" if node.transA else "__i0, __i2"),
                "__b": dace.Memlet.simple("_b", "__i1, __i2" if node.transB else "__i2, __i1"),
                **({
                    "__alpha": dace.Memlet.simple("_alpha", "0")
                } if rt_alpha else {}),
            },
                                     mul_program,
                                     {"__out": dace.Memlet.simple(mul_out, "__i0, __i1", wcr_str="lambda x, y: x + y")},
                                     external_edges=True,
                                     output_nodes=output_nodes)

        return sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        return ExpandGemmPure.make_sdfg(node, state, sdfg)


@dace.library.expansion
class ExpandGemmPureRowWise(ExpandTransformation):
    """Row-wise (ikj) pure expansion: ``for i (parallel): for k (seq): for j (vector)``.

    A semantically-identical alternative to :class:`ExpandGemmPure` that lowers to a
    vectorizable row update (``C[i,:] += A[i,k]*B[k,:]``) with a sequential K accumulation,
    rather than a 3D map with a per-element inner-K WCR. Selected by canonicalization for
    known-small GEMMs, where a loop of tiny BLAS calls is overhead and an inner-K WCR
    serializes on the reduction; here j (unit-stride) vectorizes and k stays a clean
    sequential accumulation.
    """

    environments = []

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        return ExpandGemmPure.make_sdfg(node, state, sdfg, rowwise=True)


@dace.library.expansion
class ExpandGemmOpenBLAS(ExpandTransformation):

    environments = [environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        (_, adesc, _, _, _, _), (_, bdesc, _, _, _, _), _ = _get_matmul_operands(node, state, sdfg)
        dtype = adesc.dtype.base_type
        func = to_blastype(dtype.type).lower() + 'gemm'
        alpha = f'{dtype.ctype}({node.alpha})'
        beta = f'{dtype.ctype}({node.beta})'

        # Deal with complex input constants
        if isinstance(node.alpha, complex):
            alpha = f'{dtype.ctype}({node.alpha.real}, {node.alpha.imag})'
        if isinstance(node.beta, complex):
            beta = f'{dtype.ctype}({node.beta.real}, {node.beta.imag})'

        cdesc = sdfg.arrays[state.out_edges(node)[0].data.data]

        check_access(dtypes.ScheduleType.CPU_Multicore, adesc, bdesc, cdesc)

        opt = _get_codegen_gemm_opts(node, state, sdfg, adesc, bdesc, cdesc, alpha, beta, dtype.ctype, func)

        # Adaptations for BLAS API
        opt['ta'] = 'CblasNoTrans' if opt['ta'] == 'N' else 'CblasTrans'
        opt['tb'] = 'CblasNoTrans' if opt['tb'] == 'N' else 'CblasTrans'

        # Runtime coefficients: a wired ``_alpha`` / ``_beta`` host scalar is composed with the
        # compile-time property into a host local. Complex CBLAS takes coefficient pointers, real
        # takes values -- so the effective locals are declared whenever a coefficient is complex or
        # supplied at runtime, and passed accordingly.
        scalars = _coeff_conn_descs(node, state, sdfg)
        is_complex = dtype in (dace.complex64, dace.complex128)
        code = ''
        if is_complex or scalars:
            code = (_cblas_coeff('__alpha', '_alpha', node.alpha, scalars.get('_alpha'), dtype) +
                    _cblas_coeff('__beta', '_beta', node.beta, scalars.get('_beta'), dtype))
            opt['alpha'] = '&__alpha' if is_complex else '__alpha'
            opt['beta'] = '&__beta' if is_complex else '__beta'

        code += ("cblas_{func}(CblasColMajor, {ta}, {tb}, "
                 "{M}, {N}, {K}, {alpha}, {x}, {lda}, {y}, {ldb}, {beta}, "
                 "_c, {ldc});").format_map(opt)

        tasklet = dace.sdfg.nodes.Tasklet(
            node.name,
            node.in_connectors,
            node.out_connectors,
            code,
            language=dace.dtypes.Language.CPP,
        )
        return tasklet


@dace.library.expansion
class ExpandGemmMKL(ExpandTransformation):
    environments = [environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(*args, **kwargs):
        return ExpandGemmOpenBLAS.expansion(*args, **kwargs)


@dace.library.expansion
class ExpandGemmGPUBLAS(ExpandTransformation):

    environments = []

    @classmethod
    def expansion(cls, node, state, sdfg):
        node.validate(sdfg, state)

        # Find inputs and output
        adesc, bdesc, cdesc = None, None, None
        for e in state.in_edges(node):
            if e.dst_conn == '_a':
                anode = state.memlet_path(e)[0].src
                if isinstance(anode, dace.sdfg.nodes.AccessNode):
                    adesc: dt.Array = sdfg.arrays[anode.data]
            elif e.dst_conn == '_b':
                bnode = state.memlet_path(e)[0].src
                if isinstance(bnode, dace.sdfg.nodes.AccessNode):
                    bdesc: dt.Array = sdfg.arrays[bnode.data]
        for e in state.out_edges(node):
            if e.src_conn == '_c':
                cnode = state.memlet_path(e)[-1].dst
                if isinstance(cnode, dace.sdfg.nodes.AccessNode):
                    cdesc: dt.Array = sdfg.arrays[cnode.data]
        if not adesc or not bdesc or not cdesc:
            raise ValueError('Unsupported input/output arrays')

        # If buffers are not on the GPU, copy them. Note: a wired ``_alpha`` / ``_beta`` scalar is
        # never staged to the GPU -- it is passed through wherever it already lives (host or device),
        # and the pointer mode is chosen accordingly. ``arr_prefix`` (``_conn`` when staging A/B/C)
        # therefore also prefixes the scalar connector names inside the emitted call.
        needs_copy = any(desc.storage not in (dace.StorageType.GPU_Global, dace.StorageType.CPU_Pinned)
                         for desc in (adesc, bdesc, cdesc))
        # The expansion is always wrapped in a nested SDFG whose connectors live in the `_conn`
        # namespace (see below), so the emitted call always addresses `_conn`-prefixed names --
        # whether or not A/B/C are staged to the device.
        arr_prefix = '_conn'

        dtype = adesc.dtype.base_type
        func = cls.funcname(to_blastype(dtype.type))
        if dtype == dace.float16:
            cdtype = '__half'
            factort = 'Half'
        elif dtype == dace.float32:
            cdtype = 'float'
            factort = 'Float'
        elif dtype == dace.float64:
            cdtype = 'double'
            factort = 'Double'
        elif dtype == dace.complex64:
            cdtype = f'{cls.dtype_backend}Complex'
            factort = 'Complex64'
        elif dtype == dace.complex128:
            cdtype = f'{cls.dtype_backend}DoubleComplex'
            factort = 'Complex128'
        else:
            raise ValueError("Unsupported type: " + str(dtype))

        call_prefix = cls.environments[0].handle_setup_code(node)
        call_suffix = ''

        # Handle alpha / beta
        constants = {
            1.0: f"__state->{cls.backend}blas_handle.Constants(__dace_cuda_device).{factort}Pone()",
            #-1.0: f"__state->cublas_handle.Constants(__dace_cuda_device).{factort}Mone()",
            0.0: f"__state->{cls.backend}blas_handle.Constants(__dace_cuda_device).{factort}Zero()",
        }
        handle = f"__dace_{cls.backend}blas_handle"
        scalars = _coeff_conn_descs(node, state, sdfg)
        if scalars:
            # Runtime coefficient(s) wired as scalar connectors. The pointer mode is selected by
            # WHERE the scalar lives -- a host CPU scalar uses host pointer mode (its value is read
            # and a host address is passed), a GPU length-1 array uses device pointer mode (the
            # device pointer is passed straight through). ``cublasSetPointerMode`` is handle-wide,
            # so alpha and beta must share a memory space; a compile-time 0/1 partner rides along as
            # the preallocated device constant under device mode, or as an inlined host value under
            # host mode. A CPU scalar is never promoted to the GPU.
            alpha_desc = scalars.get('_alpha')
            beta_desc = scalars.get('_beta')
            device_mode = any(d is not None and d.storage == dace.StorageType.GPU_Global
                              for d in (alpha_desc, beta_desc))
            host_mode = any(d is not None and d.storage != dace.StorageType.GPU_Global for d in (alpha_desc, beta_desc))
            if device_mode and host_mode:
                raise NotImplementedError(
                    "Gemm GPU: alpha and beta runtime scalars must share a memory space (both host or "
                    "both device); the cuBLAS/rocBLAS pointer mode is handle-wide.")
            if device_mode:
                call_prefix += f'{cls.set_pointer_mode}({handle}, {cls.pointer_device});\n'
                alpha = _device_coeff('_alpha', node.alpha, alpha_desc, arr_prefix, cdtype, constants)
                beta = _device_coeff('_beta', node.beta, beta_desc, arr_prefix, cdtype, constants)
            else:
                call_prefix += f'{cls.set_pointer_mode}({handle}, {cls.pointer_host});\n'
                adecl, alpha = _host_coeff('__alpha', '_alpha', node.alpha, alpha_desc, arr_prefix, dtype, cdtype)
                bdecl, beta = _host_coeff('__beta', '_beta', node.beta, beta_desc, arr_prefix, dtype, cdtype)
                call_prefix += adecl + bdecl
                call_suffix += f'{cls.set_pointer_mode}({handle}, {cls.pointer_device});'
        elif node.alpha not in constants or node.beta not in constants:
            # Deal with complex input constants
            if isinstance(node.alpha, complex):
                alpha = f'{dtype.ctype}({node.alpha.real}, {node.alpha.imag})'
            else:
                alpha = f'{dtype.ctype}({node.alpha})'
            if isinstance(node.beta, complex):
                beta = f'{dtype.ctype}({node.beta.real}, {node.beta.imag})'
            else:
                beta = f'{dtype.ctype}({node.beta})'

            # Set pointer mode to host
            call_prefix += f'''{cls.set_pointer_mode}(__dace_{cls.backend}blas_handle, {cls.pointer_host});
            {dtype.ctype} __alpha = {alpha};
            {dtype.ctype} __beta = {beta};
            '''
            call_suffix += f'''{cls.set_pointer_mode}(__dace_{cls.backend}blas_handle, {cls.pointer_device});'''
            alpha = f'({cdtype} *)&__alpha'
            beta = f'({cdtype} *)&__beta'
        else:
            alpha = constants[node.alpha]
            beta = constants[node.beta]

        # Set up options for code formatting
        opt = _get_codegen_gemm_opts(node, state, sdfg, adesc, bdesc, cdesc, alpha, beta, cdtype, func)
        opt['arr_prefix'] = arr_prefix

        # Matrix multiplication
        if (node.compute_type is None and node.accumulator_type is None and node.algorithm is None):
            opt['backend'] = cls.backend
            opt['backend_op_ta'] = cls.backend_op(opt['ta'])
            opt['backend_op_tb'] = cls.backend_op(opt['tb'])

            call = '''{backend}blas{func}(__dace_{backend}blas_handle,
                {backend_op_ta}, {backend_op_tb},
                {M}, {N}, {K},
                {alpha},
                ({dtype}*){arr_prefix}{x}, {lda},
                ({dtype}*){arr_prefix}{y}, {ldb},
                {beta},
                ({dtype}*){arr_prefix}_c, {ldc});'''.format_map(opt)
        else:
            if node.compute_type is not None:
                acctype = node.compute_type
            elif node.accumulator_type is not None:
                acc_dtype: dtypes.typeclass = node.accumulator_type
                acctype = f'{cls.backend.upper()}BLAS_COMPUTE_{to_cublas_computetype(acc_dtype)}'
            else:
                acctype = f'{cls.backend.upper()}BLAS_COMPUTE_{to_cublas_computetype(dtype)}'

            algorithm = f'{cls.backend.upper()}BLAS_GEMM_DEFAULT_TENSOR_OP'
            if node.algorithm is not None:
                algorithm = node.algorithm

            call = f'''
            {cls.backend}blas{cls.ex_suffix}(__dace_{cls.backend}blas_handle,
                {cls.backend_op(opt['ta'])},
                {cls.backend_op(opt['tb'])},
                {opt['M']}, {opt['N']}, {opt['K']},
                {alpha},
                {arr_prefix}{opt['x']},
                {dtype_to_cudadatatype(opt['xdtype'])},
                {opt['lda']},
                {arr_prefix}{opt['y']},
                {dtype_to_cudadatatype(opt['ydtype'])},
                {opt['ldb']},
                {beta},
                {arr_prefix}_c,
                {dtype_to_cudadatatype(opt['cdtype'])},
                {opt['ldc']},
                {acctype},
                {algorithm});
            '''

        code = (call_prefix + call + call_suffix)
        tasklet = dace.sdfg.nodes.Tasklet(
            node.name,
            node.in_connectors,
            node.out_connectors,
            code,
            language=dace.dtypes.Language.CPP,
        )

        # cuBLAS/rocBLAS read and write C in place through a single pointer, so the expansion is
        # ALWAYS wrapped in a nested SDFG: a bare tasklet cannot carry `_c` as both an in- and an
        # out-connector (a duplicate-connector error -- which surfaces once the operands are already
        # on the GPU, e.g. after apply_gpu_transformations, so no host staging renames it away).
        # Inside the wrapper the connectors live in the `_conn` namespace so they never collide with
        # the nested `_a`/`_b`/`_c`/`_alpha`/`_beta` arrays, and the C read (`_conn_cin`, only when
        # beta != 0) is a distinct connector from the C write (`_conn_c`). A/B/C are additionally
        # staged through GPU_Global transients when they are not already resident on the device
        # (`needs_copy`); when they already are, the nested arrays pass straight through.
        nsdfg = dace.SDFG('nested_gemm')
        for name, desc in [('_a', adesc), ('_b', bdesc), ('_c', cdesc)]:
            if isinstance(desc, dt.View):
                dcopy = desc.as_array()
            else:
                dcopy = dc(desc)
            dcopy.transient = False
            nsdfg.add_datadesc(name, dcopy)
            if needs_copy:
                dcopy_gpu = dc(dcopy)
                dcopy_gpu.transient = True
                dcopy_gpu.storage = dace.StorageType.GPU_Global
                dcopy_gpu.lifetime = dtypes.AllocationLifetime.Scope
                nsdfg.add_datadesc(name + '_gpu', dcopy_gpu)
        nstate = nsdfg.add_state()

        # Rename the tasklet connectors into the `_conn` namespace and drop the `_conn_c` INPUT; the
        # C read is re-added below as the distinct `_conn_cin` connector so the single-pointer
        # in-place update never needs `_c` as both an in- and an out-connector.
        tasklet.in_connectors = {"_conn" + k: None for k in tasklet.in_connectors}
        tasklet.out_connectors = {"_conn" + k: None for k in tasklet.out_connectors}
        if "_conn_c" in tasklet.in_connectors:
            tasklet.remove_in_connector("_conn_c")
        nstate.add_node(tasklet)

        # A and B inputs: staged through a GPU_Global transient when a copy is needed, else passed
        # straight through to the tasklet.
        for name, desc in (('_a', adesc), ('_b', bdesc)):
            src = nstate.add_read(name)
            if needs_copy:
                gpu = nstate.add_access(name + '_gpu')
                nstate.add_nedge(src, gpu, dace.Memlet.from_array(name, desc))
                nstate.add_edge(gpu, None, tasklet, '_conn' + name,
                                dace.Memlet.from_array(name + '_gpu', nsdfg.arrays[name + '_gpu']))
            else:
                nstate.add_edge(src, None, tasklet, '_conn' + name, dace.Memlet.from_array(name, desc))

        # C output, with a device-to-host copyback when C was staged.
        cout = nstate.add_write('_c')
        if needs_copy:
            gc = nstate.add_access('_c_gpu')
            nstate.add_edge(tasklet, '_conn_c', gc, None, dace.Memlet.from_array('_c_gpu', nsdfg.arrays['_c_gpu']))
            nstate.add_nedge(gc, cout, dace.Memlet.from_array('_c', cdesc))
        else:
            nstate.add_edge(tasklet, '_conn_c', cout, None, dace.Memlet.from_array('_c', cdesc))

        # Runtime coefficient scalars pass straight to the tasklet -- no host<->device staging,
        # so a host scalar stays on the host and a device scalar stays on the device.
        for conn, desc in scalars.items():
            sdesc = dc(desc)
            sdesc.transient = False
            nsdfg.add_datadesc(conn, sdesc)
            nstate.add_edge(nstate.add_read(conn), None, tasklet, '_conn' + conn, dace.Memlet.from_array(conn, sdesc))
            if desc.storage == dace.StorageType.GPU_Global:
                # Device pointer mode: the length-1 coefficient is handed to cuBLAS as a raw device
                # pointer (dereferenced on the GPU). Type the connector as a pointer so it is a
                # device-pointer pass-through, not a size-1 host dereference of GPU memory.
                tasklet.in_connectors['_conn' + conn] = dtypes.pointer(sdesc.dtype)

        # C read for beta != 0, through the distinct `_conn_cin` connector (staged when needed).
        if not equal_valued(0, node.beta):
            tasklet.add_in_connector('_conn_cin')
            rc = nstate.add_read('_c')
            if needs_copy:
                rgc = nstate.add_access('_c_gpu')
                nstate.add_nedge(rc, rgc, dace.Memlet.from_array('_c', cdesc))
                nstate.add_edge(rgc, None, tasklet, '_conn_cin',
                                dace.Memlet.from_array('_c_gpu', nsdfg.arrays['_c_gpu']))
            else:
                nstate.add_edge(rc, None, tasklet, '_conn_cin', dace.Memlet.from_array('_c', cdesc))

        return nsdfg


@dace.library.expansion
class ExpandGemmCuBLAS(ExpandGemmGPUBLAS):
    environments = [environments.cublas.cuBLAS]
    backend = 'cu'
    dtype_backend = 'cu'
    set_pointer_mode = 'cublasSetPointerMode'
    pointer_host = 'CUBLAS_POINTER_MODE_HOST'
    pointer_device = 'CUBLAS_POINTER_MODE_DEVICE'
    ex_suffix = 'GemmEx'

    @classmethod
    def backend_op(cls, mode: str) -> str:
        return f'CUBLAS_OP_{mode}'

    @classmethod
    def funcname(cls, dtype: str) -> str:
        return f'{dtype}gemm'


@dace.library.expansion
class ExpandGemmRocBLAS(ExpandGemmGPUBLAS):
    environments = [environments.rocblas.rocBLAS]
    backend = 'roc'
    dtype_backend = 'hip'
    set_pointer_mode = 'rocblas_set_pointer_mode'
    pointer_host = 'rocblas_pointer_mode_host'
    pointer_device = 'rocblas_pointer_mode_device'
    ex_suffix = '_gemm_ex'

    @classmethod
    def backend_op(cls, mode: str) -> str:
        if mode == 'N':
            return 'rocblas_operation_none'
        elif mode == 'T':
            return 'rocblas_operation_transpose'
        raise ValueError(f'Invalid gemm matrix operation {mode}')

    @classmethod
    def funcname(cls, dtype: str) -> str:
        return f'_{dtype.lower()}gemm'


@dace.library.expansion
class ExpandGemmPBLAS(ExpandTransformation):

    environments = []

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        (_, adesc, ashape, _, _, _), (_, bdesc, bshape, _, _, _), _ = _get_matmul_operands(node, state, sdfg)
        dtype = adesc.dtype.base_type

        if not equal_valued(0, node.beta):
            raise NotImplementedError

        M = ashape[0]
        K = ashape[1]
        N = bshape[1]
        Px = dace.symbol('Px', dtype=dace.int32, integer=True, positive=True)
        Py = dace.symbol('Py', dtype=dace.int32, integer=True, positive=True)
        try:
            sdfg.add_symbol('Px', dace.int32)
            sdfg.add_symbol('Py', dace.int32)
        except FileExistsError:
            pass

        @dace.program
        def _gemm_pblas(_a: dtype[M, K], _b: dtype[K, N], _c: dtype[M, N]):
            lA = np.empty((M // Px, K // Py), dtype=_a.dtype)
            lB = np.empty((K // Px, N // Py), dtype=_b.dtype)
            dace.comm.BCScatter(_a, lA, (M // Px, K // Py))
            dace.comm.BCScatter(_b, lB, (K // Px, N // Py))
            lC = distr.MatMult(lA, lB, (M, N, K))
            dace.comm.BCGather(lC, _c, (M // Px, N // Py))

        return _gemm_pblas.to_sdfg()


@dace.library.node
class Gemm(dace.sdfg.nodes.LibraryNode):
    """Executes alpha * (A @ B) + beta * C. C should be unidirectionally
       broadcastable (ONNX terminology) to A @ B.
    """

    # Global properties
    implementations = {
        "pure": ExpandGemmPure,
        "rowwise": ExpandGemmPureRowWise,
        "MKL": ExpandGemmMKL,
        "OpenBLAS": ExpandGemmOpenBLAS,
        "cuBLAS": ExpandGemmCuBLAS,
        "rocBLAS": ExpandGemmRocBLAS,
        "PBLAS": ExpandGemmPBLAS,
    }
    default_implementation = None

    # Object fields
    transA = properties.Property(dtype=bool, desc="Whether to transpose A before multiplying")
    transB = properties.Property(dtype=bool, desc="Whether to transpose B before multiplying")
    alpha = properties.Property(allow_none=False,
                                default=1,
                                desc="A scalar which will be multiplied with A @ B before adding C")
    beta = properties.Property(allow_none=False,
                               default=0,
                               desc="A scalar which will be multiplied with C before adding C")
    cin = properties.Property(dtype=bool, default=True, desc="Whether to have a _c in connector when beta != 0")
    alpha_input = properties.Property(dtype=bool,
                                      default=False,
                                      desc="Whether alpha is supplied at runtime through an '_alpha' scalar "
                                      "connector (composed multiplicatively with the 'alpha' property). The GPU "
                                      "expansion selects the cuBLAS pointer mode by the connector's storage: a "
                                      "host CPU scalar -> host pointer mode, a GPU length-1 array -> device "
                                      "pointer mode. A host scalar is never promoted to the GPU.")
    beta_input = properties.Property(dtype=bool,
                                     default=False,
                                     desc="Whether beta is supplied at runtime through a '_beta' scalar connector "
                                     "(composed multiplicatively with the 'beta' property); forces C to be read. "
                                     "Same host/device pointer-mode selection as alpha_input.")
    algorithm = properties.Property(dtype=str,
                                    allow_none=True,
                                    default=None,
                                    desc="If applicable, chooses the vendor-provided implementation "
                                    "(algorithm) for the multiplication")
    accumulator_type = properties.TypeClassProperty(
        default=None, allow_none=True, desc="Accumulator or intermediate storage type used in multiplication")
    compute_type = properties.Property(default=None,
                                       dtype=str,
                                       allow_none=True,
                                       desc="If applicable, overrides computation type (CUBLAS-specific, see "
                                       "``cublasComputeType_t``)")

    def __init__(self,
                 name,
                 location=None,
                 transA=False,
                 transB=False,
                 alpha=1,
                 beta=0,
                 cin=True,
                 alpha_input=False,
                 beta_input=False):
        # C is read when a nonzero compile-time beta is added in place, or whenever beta is a
        # runtime input (its value is unknown at build time, so C must be available).
        reads_c = (not equal_valued(0, beta) and cin) or beta_input
        inputs = {"_a", "_b"}
        if reads_c:
            inputs.add("_c")
        if alpha_input:
            inputs.add("_alpha")
        if beta_input:
            inputs.add("_beta")
        super().__init__(name, location=location, inputs=inputs, outputs={"_c"})
        self.transA = True if transA else False
        self.transB = True if transB else False
        self.alpha = alpha
        self.beta = beta
        self.cin = cin
        self.alpha_input = alpha_input
        self.beta_input = beta_input

    def validate(self, sdfg, state):
        in_edges = state.in_edges(self)
        # ``_alpha`` / ``_beta`` runtime coefficient connectors are not matrix operands.
        matrix_in = [e for e in in_edges if e.dst_conn in ('_a', '_b', '_c')]
        if len(matrix_in) not in [2, 3]:
            raise ValueError("Expected 2 or 3 matrix inputs to gemm")
        size2 = None
        for _, _, _, dst_conn, memlet in state.in_edges(self):
            if dst_conn == '_a':
                size0 = memlet.subset.size()
            if dst_conn == '_b':
                size1 = memlet.subset.size()
            if dst_conn == '_c':
                size2 = memlet.subset.size()

        if self.transA:
            size0 = list(reversed(size0))
        if self.transB:
            size1 = list(reversed(size1))

        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one output from matrix-matrix product")
        out_memlet = out_edges[0].data
        # Function is symmetric, edge order does not matter
        if len(size0) != 2 or len(size1) != 2:
            raise ValueError("matrix-matrix product only supported on matrices")
        res = equal(size0[1], size1[0])
        if res is None:
            warnings.warn(f'First matrix columns {size0[1]} and second matrix rows {size1[0]} may not match',
                          UserWarning)
        elif not res:
            raise ValueError("Inputs to matrix-matrix product must agree in the k-dimension")
        size3 = out_memlet.subset.size()
        if size2 is not None:
            res = [equal(s0, s1) for s0, s1 in zip(size2, size3)]
            fail = any([r is False for r in res])
            success = all([r is True for r in res])
            if fail:
                raise ValueError("Input C matrix must match output matrix.")
            elif not success:
                warnings.warn(f"Size of input C matrix {size2} may not match output matrix size {size3}", UserWarning)
        if len(size3) != 2:
            raise ValueError("matrix-matrix product only supported on matrices")
        if len(size3) == 2:
            res = [equal(s0, s1) for s0, s1 in zip(size3, [size0[-2], size1[-1]])]
            fail = any([r is False for r in res])
            success = all([r is True for r in res])
            if fail:
                raise ValueError("Output to matrix-matrix product must agree in the m and n dimensions")
            elif not success:
                warnings.warn(f'Size of output {size3} may not match input {size0} @ {size1}', UserWarning)


# Numpy replacement
@oprepo.replaces('dace.libraries.blas.gemm')
@oprepo.replaces('dace.libraries.blas.Gemm')
def gemm_libnode(pv: 'ProgramVisitor',
                 sdfg: SDFG,
                 state: SDFGState,
                 A,
                 B,
                 C,
                 alpha,
                 beta,
                 trans_a=False,
                 trans_b=False):
    # ``alpha`` / ``beta`` may be numbers/symbols (compile-time coefficients) or the name of a scalar
    # array already in the SDFG (a runtime coefficient wired via ``_alpha`` / ``_beta``, with the GPU
    # pointer mode chosen by the array's storage). Mirrors ``symm.symm_libnode``.
    alpha_input = isinstance(alpha, str) and alpha in sdfg.arrays
    beta_input = isinstance(beta, str) and beta in sdfg.arrays
    reads_c = beta_input or (not isinstance(beta, str) and not equal_valued(0, beta))

    # Add nodes
    A_in, B_in = (state.add_read(name) for name in (A, B))
    C_out = state.add_write(C)

    libnode = Gemm('gemm',
                   transA=trans_a,
                   transB=trans_b,
                   alpha=1 if alpha_input else alpha,
                   beta=1 if beta_input else beta,
                   alpha_input=alpha_input,
                   beta_input=beta_input)
    state.add_node(libnode)

    # Connect nodes
    state.add_edge(A_in, None, libnode, '_a', mm.Memlet(A))
    state.add_edge(B_in, None, libnode, '_b', mm.Memlet(B))
    state.add_edge(libnode, '_c', C_out, None, mm.Memlet(C))

    if reads_c:
        C_in = state.add_read(C)
        state.add_edge(C_in, None, libnode, '_c', mm.Memlet(C))
    if alpha_input:
        state.add_edge(state.add_read(alpha), None, libnode, '_alpha', mm.Memlet(alpha))
    if beta_input:
        state.add_edge(state.add_read(beta), None, libnode, '_beta', mm.Memlet(beta))

    return []
