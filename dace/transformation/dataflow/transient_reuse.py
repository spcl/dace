from dace import registry
from dace.transformation import pattern_matching
from dace.properties import make_properties

@registry.autoregister
@make_properties
class TransientReuse(pattern_matching.Transformation):

