from dace.properties import Property, make_properties


@make_properties
class CodeObject(object):
    name = Property(dtype=str, desc="Filename to use")
    code = Property(dtype=str, desc="The code attached to this object")
    language = Property(
        dtype=str,
        desc="Language used for this code (same " +
        "as its file extension)")  # dtype=dtypes.Language?
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
                 additional_compiler_kwargs=None,
                 linkable=True):
        super(CodeObject, self).__init__()

        self.name = name
        self.code = code
        self.language = language
        self.target = target
        self.title = title
        self.extra_compiler_kwargs = additional_compiler_kwargs or {}
        self.linkable = linkable
