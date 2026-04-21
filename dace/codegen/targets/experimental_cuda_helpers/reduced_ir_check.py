# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Simplified-SDFG-Dialect compliance check for the experimental CUDA codegen.

The experimental codegen assumes a reduced form of the SDFG IR (the
"Simplified SDFG Dialect").  Constructs disallowed by this form include:

* Control flow:
    - ``ConsumeEntry`` scopes ("consume maps")
    - ``Stream`` data descriptors (SDFG streams) and the access nodes that
      reference them
    - conditional interstate edges (inter-state edges with non-trivial
      ``condition``)
* Data movement:
    - memlets with WCR
    - memlets with ``other_subset``
    - implicit AccessNode -> AccessNode copies
    - view data descriptors (and view access nodes)
* GPU-specific:
    - dynamic thread-block maps (``GPU_ThreadBlock_Dynamic``)
    - persistent GPU device maps (``GPU_Persistent``)

``SimplifiedDialectCompliant`` exposes one method per feature, each returning
``True`` when the SDFG is compliant with respect to that feature (i.e. the
forbidden construct is absent).  ``is_compliant(sdfg)`` aggregates them; if
any check fails the experimental codegen emits a prominent warning via
``warn_if_not_simplified_dialect``.
"""
from typing import List, Tuple

from dace import data as dt, dtypes
from dace.sdfg import SDFG, nodes


class SimplifiedDialectCompliant:
    """Per-feature compliance checks for the Simplified SDFG Dialect.

    Each ``check_*`` method returns ``True`` when the SDFG is compliant with
    respect to that feature (i.e. contains none of the forbidden construct).
    ``offenders_*`` returns a list of human-readable locator strings for the
    offending constructs (used for the aggregated warning message).
    """

    # ---- Control flow -----------------------------------------------------

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
            # Access nodes referencing those streams.
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

    # ---- Data movement ----------------------------------------------------

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

    # ---- GPU-specific -----------------------------------------------------

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

    # ---- Aggregator -------------------------------------------------------

    # Ordered list of (feature label, offenders-getter) pairs.  The order is
    # the order shown in the warning message.
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
        """Return a list of (feature_label, offenders) for every feature that fails its check."""
        out: List[Tuple[str, List[str]]] = []
        for label, getter in cls._CHECKS:
            offenders = getter.__func__(sdfg) if isinstance(getter, staticmethod) else getter(sdfg)
            if offenders:
                out.append((label, offenders))
        return out

    @classmethod
    def is_compliant(cls, sdfg: SDFG) -> bool:
        """``True`` iff ``sdfg`` is compliant with every feature of the Simplified Dialect."""
        return not cls.collect(sdfg)


def warn_if_not_simplified_dialect(sdfg: SDFG) -> None:
    """Emit a prominent warning if ``sdfg`` violates the Simplified SDFG Dialect.

    The warning lists each offending feature together with up to a handful of
    concrete locators.  It does not raise -- the experimental codegen then
    proceeds on a best-effort basis.
    """
    import warnings

    offenders_by_feature = SimplifiedDialectCompliant.collect(sdfg)
    if not offenders_by_feature:
        return

    max_per_feature = 5
    lines: List[str] = []
    for label, offenders in offenders_by_feature:
        shown = offenders[:max_per_feature]
        extra = len(offenders) - len(shown)
        bullet = f'\n    * ' + '\n    * '.join(shown)
        if extra > 0:
            bullet += f'\n    * ... and {extra} more'
        lines.append(f'  - {label}:{bullet}')
    banner = '=' * 72
    body = '\n'.join(lines)
    warnings.warn(
        f'\n{banner}\n'
        f'ExperimentalCUDACodeGen: SDFG is NOT compliant with the Simplified SDFG Dialect.\n'
        f'Generated code may be incorrect. Offending feature(s):\n'
        f'{body}\n'
        f'{banner}',
        stacklevel=2,
    )


# Backwards-compatible re-export: older callers used ``warn_if_not_reduced_ir``.
warn_if_not_reduced_ir = warn_if_not_simplified_dialect
