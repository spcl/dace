# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Backend-independent surface of a compiled SDFG.

    :class:`CompiledSDFGProtocol` captures the interface both the ctypes and the
    nanobind backends provide. The concrete ctypes implementation was renamed to
    :class:`~dace.codegen.ctypes_compiled_sdfg.CtypesCompiledSDFG` (and moved to
    ``dace.codegen.ctypes_compiled_sdfg``); the ``CompiledSDFG`` name is kept here
    as a deprecated alias for backward compatibility.
"""
import warnings
from typing import Any, Callable, Dict, Optional, Protocol, runtime_checkable

from dace import dtypes

# Re-exported for backward compatibility (they used to be defined in this module).
from dace.codegen.ctypes_compiled_sdfg import CtypesCompiledSDFG, ReloadableDLL  # noqa: F401 [unused-import]  # Compatibility reexport.


@runtime_checkable
class CompiledSDFGProtocol(Protocol):
    """The interface a compiled SDFG object provides, regardless of backend.

    Both the ctypes :class:`~dace.codegen.ctypes_compiled_sdfg.CtypesCompiledSDFG`
    and the nanobind ``NanobindCompiledSDFG`` satisfy this protocol structurally
    (neither inherits it); interface-agnostic code - ``SDFG.__call__``,
    ``DaceProgram``, hooks, the profiler - should annotate against it and
    use only these members. Backend internals (ctypes: ``_cfunc``,
    ``_libhandle``, ``fast_call``, ...; nanobind: ``_handle``) are
    deliberately absent: code that needs them is backend-specific and must
    branch explicitly.

    Where the concrete signatures diverge, the protocol declares the
    narrower ctypes form (``get_workspace_sizes`` takes no arguments there;
    the nanobind version accepting per-call symbols is a compatible
    superset) or the loosest common return type (``initialize`` returns the
    state handle on ctypes and ``None`` on nanobind; ``get_state_struct``
    returns a ``ctypes.Structure`` vs an address).

    :note: ``isinstance`` checks (via ``runtime_checkable``) verify member
           *presence* only - the behavioral contract is carried by the test
           suites that run against both backends.
    """

    #: When True, calls skip the program execution. The profiler toggles
    #: this around its replay loop, so it must be writable.
    do_not_execute: bool

    @property
    def sdfg(self) -> Any:  # dace.SDFG
        ...

    @property
    def filename(self) -> str:
        ...

    @property
    def has_gpu_code(self) -> bool:
        ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def safe_call(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def initialize(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def finalize(self) -> None:
        ...

    def get_exported_function(self, name: str, restype=None) -> Optional[Callable[..., Any]]:
        ...

    def get_state_struct(self) -> Any:
        ...

    def get_workspace_sizes(self) -> Dict[dtypes.StorageType, int]:
        ...

    def set_workspace(self, storage: dtypes.StorageType, workspace: Any) -> None:
        ...


class CompiledSDFG(CtypesCompiledSDFG):
    """Deprecated compatibility alias for :class:`~dace.codegen.ctypes_compiled_sdfg.CtypesCompiledSDFG`.

    The ctypes-based compiled-SDFG implementation was renamed to
    ``CtypesCompiledSDFG`` (and moved to ``dace.codegen.ctypes_compiled_sdfg``)
    when the nanobind interface was added. This deprecated alias is kept for a
    smooth transition.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Deprecated: use :class:`~dace.codegen.ctypes_compiled_sdfg.CtypesCompiledSDFG` instead."""
        super().__init__(*args, **kwargs)
        warnings.warn('CompiledSDFG is deprecated, please use CtypesCompiledSDFG instead.',
                      DeprecationWarning,
                      stacklevel=2)
