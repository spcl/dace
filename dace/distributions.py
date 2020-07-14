from typing import List, Tuple, Union
from dace.properties import make_properties, ShapeProperty
from dace.symbolic import symbol

Integer = Union[int, symbol]
ShapeTuple = Tuple[Integer]
ShapeList = List[Integer]
Shape = Union[ShapeTuple, ShapeList]

@make_properties
class ProcessGrid:
    """ Describes a process grid for distributing data and computation.
    """

    grid = ShapeProperty(default=[])

    def __init__(self,
                 pgrid: Shape):
        self.grid = pgrid
