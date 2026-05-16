# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Core Dialect compliance check.

The Core Dialect is the subset of the SDFG IR that downstream passes (the experimental CUDA
codegen, layout-permutation transformations, etc.) consume. It disallows ``ConsumeEntry`` scopes,
``Stream`` descriptors, conditional interstate edges, WCR / ``other_subset`` memlets, implicit
AccessNode-to-AccessNode copies, views, and ``GPU_ThreadBlock_Dynamic`` / ``GPU_Persistent`` maps.
"""
from typing import List, Tuple

from dace import data as dt, dtypes
from dace.sdfg import SDFG, nodes


class CoreDialectCompliant:
    """Per-feature Core Dialect compliance checks.

    Every ``check_*`` method returns ``True`` when the SDFG contains none of the
    corresponding forbidden construct; ``offenders_*`` returns human-readable
    locators for the concrete offenders.
    """

    @staticmethod
    def offenders_consume_scopes(sdfg: SDFG) -> List[str]:
        out: List[str] = []
        for node, _parent in sdfg.all_nodes_recursive():
            if isinstance(node, nodes.ConsumeEntry):
                out.append(f'consume scope "{node.label}"')
        return out

    @classmethod
    def check_no_consume_scopes(cls, sdfg: SDFG) -> bool:
        return not cls.offenders_consume_scopes(sdfg)

    @staticmethod
    def offenders_sdfg_streams(sdfg: SDFG) -> List[str]:
        out: List[str] = []
        for sub_sdfg in sdfg.all_sdfgs_recursive():
            stream_names = {name for name, desc in sub_sdfg.arrays.items() if isinstance(desc, dt.Stream)}
            for name in stream_names:
                out.append(f'SDFG stream "{name}" in "{sub_sdfg.label}"')
            for state in sub_sdfg.states():
                for node in state.nodes():
                    if isinstance(node, nodes.AccessNode) and node.data in stream_names:
                        out.append(f'stream AccessNode "{node.data}" in state "{state.label}"')
        return out

    @classmethod
    def check_no_sdfg_streams(cls, sdfg: SDFG) -> bool:
        return not cls.offenders_sdfg_streams(sdfg)

    @staticmethod
    def offenders_conditional_interstate_edges(sdfg: SDFG) -> List[str]:
        out: List[str] = []
        for sub_sdfg in sdfg.all_sdfgs_recursive():
            for edge in sub_sdfg.edges():
                cond = getattr(edge.data, 'condition', None)
                if cond is None:
                    continue
                cond_str = cond.as_string.strip() if hasattr(cond, 'as_string') else str(cond).strip()
                # Unconditional edges carry an empty string or a literal "1" / "True".
                if cond_str and cond_str not in ('1', 'True'):
                    out.append(f'conditional interstate edge {edge.src.label} -> {edge.dst.label} if {cond_str}')
        return out

    @classmethod
    def check_no_conditional_interstate_edges(cls, sdfg: SDFG) -> bool:
        return not cls.offenders_conditional_interstate_edges(sdfg)

    @staticmethod
    def offenders_wcr_edges(sdfg: SDFG) -> List[str]:
        out: List[str] = []
        for edge, _parent in sdfg.all_edges_recursive():
            memlet = getattr(edge, 'data', None)
            if memlet is not None and getattr(memlet, 'wcr', None) is not None:
                out.append(f'WCR memlet "{memlet}"')
        return out

    @classmethod
    def check_no_wcr_edges(cls, sdfg: SDFG) -> bool:
        return not cls.offenders_wcr_edges(sdfg)

    @staticmethod
    def offenders_other_subsets(sdfg: SDFG) -> List[str]:
        out: List[str] = []
        for edge, _parent in sdfg.all_edges_recursive():
            memlet = getattr(edge, 'data', None)
            if memlet is not None and getattr(memlet, 'other_subset', None) is not None:
                out.append(f'memlet with other_subset "{memlet}"')
        return out

    @classmethod
    def check_no_other_subsets(cls, sdfg: SDFG) -> bool:
        return not cls.offenders_other_subsets(sdfg)

    @staticmethod
    def offenders_implicit_copies(sdfg: SDFG) -> List[str]:
        out: List[str] = []
        for sub_sdfg in sdfg.all_sdfgs_recursive():
            for state in sub_sdfg.states():
                for edge in state.edges():
                    if isinstance(edge.src, nodes.AccessNode) and isinstance(edge.dst, nodes.AccessNode):
                        out.append(f'implicit copy {edge.src.data} -> {edge.dst.data} in state "{state.label}"')
        return out

    @classmethod
    def check_no_implicit_copies(cls, sdfg: SDFG) -> bool:
        return not cls.offenders_implicit_copies(sdfg)

    @staticmethod
    def offenders_implicit_gpu_copies(sdfg: SDFG) -> List[str]:
        """Implicit AccessNode->AccessNode copies with at least one GPU-global endpoint and
        neither endpoint device-level. ``InsertExplicitGPUGlobalMemoryCopies`` lowers these;
        leftovers after the pipeline are a bug or unsupported pattern."""
        from dace.sdfg.scope import is_devicelevel_gpu
        from dace import dtypes as _dtypes
        out: List[str] = []
        for sub_sdfg in sdfg.all_sdfgs_recursive():
            for state in sub_sdfg.states():
                for edge in state.edges():
                    if not (isinstance(edge.src, nodes.AccessNode) and isinstance(edge.dst, nodes.AccessNode)):
                        continue
                    src_desc = sub_sdfg.arrays[edge.src.data]
                    dst_desc = sub_sdfg.arrays[edge.dst.data]
                    # An Array<->View edge is a reference link, not a memcpy (codegen emits the
                    # View as a pointer offset). InsertExplicitCopies skips these; the strict
                    # check must agree or it flags every ``np.reshape(GPU_array)`` slice.
                    if isinstance(src_desc, dt.View) or isinstance(dst_desc, dt.View):
                        continue
                    src_storage = src_desc.storage
                    dst_storage = dst_desc.storage
                    touches_gpu = (src_storage == _dtypes.StorageType.GPU_Global
                                   or dst_storage == _dtypes.StorageType.GPU_Global)
                    if not touches_gpu:
                        continue
                    if (is_devicelevel_gpu(sub_sdfg, state, edge.src) or is_devicelevel_gpu(sub_sdfg, state, edge.dst)):
                        # cudaMemcpyAsync cannot be issued from device code; the
                        # codegen handles intra-kernel cross-storage AccessNode
                        # edges via its register/local copy paths.
                        continue
                    out.append(f'implicit GPU-memory copy {edge.src.data} ({src_storage.name}) -> '
                               f'{edge.dst.data} ({dst_storage.name}) in state "{state.label}"')
        return out

    @classmethod
    def check_no_implicit_gpu_copies(cls, sdfg: SDFG) -> bool:
        return not cls.offenders_implicit_gpu_copies(sdfg)

    @staticmethod
    def offenders_views(sdfg: SDFG) -> List[str]:
        out: List[str] = []
        for sub_sdfg in sdfg.all_sdfgs_recursive():
            view_names = {name for name, desc in sub_sdfg.arrays.items() if isinstance(desc, dt.View)}
            for name in view_names:
                out.append(f'view data descriptor "{name}" in "{sub_sdfg.label}"')
            for state in sub_sdfg.states():
                for node in state.nodes():
                    if isinstance(node, nodes.AccessNode) and node.data in view_names:
                        out.append(f'view AccessNode "{node.data}" in state "{state.label}"')
        return out

    @classmethod
    def check_no_views(cls, sdfg: SDFG) -> bool:
        return not cls.offenders_views(sdfg)

    @staticmethod
    def offenders_dynamic_threadblock_maps(sdfg: SDFG) -> List[str]:
        out: List[str] = []
        for node, _parent in sdfg.all_nodes_recursive():
            if isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock_Dynamic:
                out.append(f'dynamic thread-block map "{node.map.label}"')
        return out

    @classmethod
    def check_no_dynamic_threadblock_maps(cls, sdfg: SDFG) -> bool:
        return not cls.offenders_dynamic_threadblock_maps(sdfg)

    @staticmethod
    def offenders_persistent_gpu_device_maps(sdfg: SDFG) -> List[str]:
        out: List[str] = []
        for node, _parent in sdfg.all_nodes_recursive():
            if isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.GPU_Persistent:
                out.append(f'persistent GPU device map "{node.map.label}"')
        return out

    @classmethod
    def check_no_persistent_gpu_device_maps(cls, sdfg: SDFG) -> bool:
        return not cls.offenders_persistent_gpu_device_maps(sdfg)

    _CHECKS = (
        ('consume scopes', offenders_consume_scopes),
        ('SDFG streams', offenders_sdfg_streams),
        ('conditional interstate edges', offenders_conditional_interstate_edges),
        ('WCR memlets', offenders_wcr_edges),
        ('memlets with other_subset', offenders_other_subsets),
        ('implicit copies', offenders_implicit_copies),
        ('views', offenders_views),
        ('dynamic thread-block maps', offenders_dynamic_threadblock_maps),
        ('persistent GPU device maps', offenders_persistent_gpu_device_maps),
    )

    @classmethod
    def collect(cls, sdfg: SDFG) -> List[Tuple[str, List[str]]]:
        """Return ``(feature_label, offenders)`` pairs for every failing feature, in report order.

        :param sdfg: the SDFG to inspect.
        :returns: a list of ``(label, offenders)`` tuples; empty if ``sdfg`` is compliant.
        """
        out: List[Tuple[str, List[str]]] = []
        for label, getter in cls._CHECKS:
            offenders = getter.__func__(sdfg) if isinstance(getter, staticmethod) else getter(sdfg)
            if offenders:
                out.append((label, offenders))
        return out

    @classmethod
    def is_compliant(cls, sdfg: SDFG) -> bool:
        """Return ``True`` iff the SDFG is core-dialect-compliant."""
        return not cls.collect(sdfg)


def warn_if_not_core_dialect(sdfg: SDFG, source: str = 'pass') -> None:
    """Emit a ``UserWarning`` if ``sdfg`` violates Core Dialect.

    The warning enumerates each offending feature together with up to five concrete locators.
    It never raises; the caller proceeds best-effort.

    :param sdfg: the SDFG to check.
    :param source: short tag identifying the caller, included in the warning header.
    """
    import warnings

    offenders_by_feature = CoreDialectCompliant.collect(sdfg)
    if not offenders_by_feature:
        return

    max_per_feature = 5
    lines: List[str] = []
    for label, offenders in offenders_by_feature:
        shown = offenders[:max_per_feature]
        extra = len(offenders) - len(shown)
        bullet = '\n    * ' + '\n    * '.join(shown)
        if extra > 0:
            bullet += f'\n    * ... and {extra} more'
        lines.append(f'  - {label}:{bullet}')
    banner = '=' * 72
    body = '\n'.join(lines)
    warnings.warn(
        f'\n{banner}\n'
        f'{source}: SDFG is NOT core-dialect-compliant.\n'
        f'Generated code may be incorrect. Offending feature(s):\n'
        f'{body}\n'
        f'{banner}',
        stacklevel=2,
    )


def require_core_dialect(sdfg: SDFG, source: str = 'pass') -> None:
    """Raise ``ValueError`` if ``sdfg`` violates Core Dialect. Strict counterpart to ``warn_if_not_core_dialect``."""
    offenders_by_feature = CoreDialectCompliant.collect(sdfg)
    if not offenders_by_feature:
        return
    lines = []
    for label, offenders in offenders_by_feature:
        shown = offenders[:5]
        extra = len(offenders) - len(shown)
        suffix = f' ... and {extra} more' if extra > 0 else ''
        lines.append(f'{label}: {", ".join(shown)}{suffix}')
    raise ValueError(f'{source} requires core-dialect-compliant SDFG. Offenders: ' + '; '.join(lines))
