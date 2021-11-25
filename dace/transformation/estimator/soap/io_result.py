import sympy as sp
from dace.transformation.estimator.soap.sdg import SDG

@dataclass
class io_result():
    name : int
    Q: sp.core.Expr
    sdg: SDG
    subgraphs : list[io_result_subgraph]


@dataclass
class io_result_subgraph():
    name : int
    Q: sp.core.Expr
    rho : sp.core.Expr
    varsOpt : list[sp.core.Expr]
    inner_tile : list[sp.core.Expr]
    outer_tile : list[sp.core.Expr]