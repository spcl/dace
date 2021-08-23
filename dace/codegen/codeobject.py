# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import re
from dace import sourcemap
from dace.properties import (Property, DictProperty, SetProperty,
                             make_properties)


@make_properties
class CodeObject(object):
    name = Property(dtype=str, desc="Filename to use")
    code = Property(dtype=str, desc="The code attached to this object")
    language = Property(dtype=str,
                        desc="Language used for this code (same " +
                        "as its file extension)")
    target = Property(dtype=type,
                      desc="Target to use for compilation",
                      allow_none=True)
    target_type = Property(
        dtype=str,
        desc="Sub-target within target (e.g., host or device code)",
        default="")
    title = Property(dtype=str, desc="Title of code for GUI")
    extra_compiler_kwargs = DictProperty(key_type=str,
                                         value_type=str,
                                         desc="Additional compiler argument "
                                         "variables to add to template")
    linkable = Property(dtype=bool,
                        desc='Should this file participate in '
                        'overall linkage?')
    environments = SetProperty(
        str,
        desc="Environments required by CMake to build and run this code node.",
        default=set())

    def __init__(self,
                 name,
                 code,
                 language,
                 target,
                 title,
                 target_type="",
                 additional_compiler_kwargs=None,
                 linkable=True,
                 environments=None,
                 sdfg=None):
        super(CodeObject, self).__init__()

        self.name = name
        self.code = code
        self.language = language
        self.target = target
        self.target_type = target_type
        self.title = title
        self.extra_compiler_kwargs = additional_compiler_kwargs or {}
        self.linkable = linkable
        self.environments = environments or set()

        if language == 'cpp' and title == 'Frame' and sdfg:
            sourcemap.create_maps(sdfg, code, self.target.target_name)

    @property
    def clean_code(self):
        return re.sub(r'[ \t]*////__(DACE:|CODEGEN;)[^\n]*', '', self.code)
