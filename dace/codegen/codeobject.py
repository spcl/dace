import ctypes
import numpy as np

from dace import symbolic, types
from dace.config import Config
from dace.frontend import operations
from dace.properties import Property, make_properties
from dace.codegen.targets.target import TargetCodeGenerator

from dace.codegen.instrumentation.perfsettings import PerfMetaInfo


@make_properties
class CodeObject(object):
    name = Property(dtype=str, desc="Filename to use")
    code = Property(dtype=str, desc="The code attached to this object")
    perf_meta_info = Property(
        dtype=PerfMetaInfo, desc="Meta information used to map nodes to LOC")
    language = Property(
        dtype=str,
        desc="Language used for this code (same " +
        "as its file extension)")  # dtype=types.Language?
    target = Property(dtype=type, desc="Target to use for compilation")
    title = Property(dtype=str, desc="Title of code for GUI")
    extra_compiler_kwargs = Property(
        dtype=dict,
        desc="Additional compiler argument "
        "variables to add to template")
    linkable = Property(
        dtype=bool, desc='Should this file participate in '
        'overall linkage?')

    def __init__(self,
                 name,
                 code,
                 language,
                 target,
                 title,
                 additional_compiler_kwargs={},
                 linkable=True,
                 meta_info=PerfMetaInfo()):
        super(CodeObject, self).__init__()

        self.name = name
        self.code = code
        self.language = language
        self.target = target
        self.title = title
        self.extra_compiler_kwargs = additional_compiler_kwargs
        self.linkable = linkable
        self.perf_meta_info = meta_info
