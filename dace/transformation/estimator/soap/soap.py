import ast
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from collections import OrderedDict
import numbers


import astunparse
from dace.sdfg.graph import MultiConnectorEdge
import dace
from dace.sdfg.nodes import *
from dace.subsets import Range

from dace.symbolic import pystr_to_symbolic
from dace import subsets, Config

# from dace.libraries.blas import MatMul, Transpose
import sympy
import sys
from typing import Any, DefaultDict, Dict, List, Union
import sympy as sp
import numpy as np
from sympy.solvers import solve
from sympy.parsing.sympy_parser import parse_expr
from sympy import oo, symbols, Poly, poly
import json
from datetime import datetime
import time
import copy
from typing import Optional
import concurrent.futures
import dace.symbolic as ds
import networkx as nx
import math

# from ordered_set import OrderedSet
from dace.transformation.estimator.soap.utils import *
from warnings import warn
from numbers import Number
from dace.transformation.estimator.soap.solver import Solver


# ------------------------------------
# Code generation
# ------------------------------------
@dataclass(frozen=True)
class LoopHeader:
    loopvar: str
    start: str
    end: str
    step: int = 1
    pragmas: str = ""
    indent: int = 0

    def __str__(self):
        if len(self.pragmas) > 0:
            pragma_str = f"{' ' * self.indent}{self.pragmas}\n"
        else:
            pragma_str = ""
        return f"{pragma_str}{' ' * self.indent}for {self.loopvar} in range({self.start}, {self.end}, {self.step}):"

    def __repr__(self):
        return str(self)


@dataclass
class LoopBody:
    statements: list = field(default_factory=list)    



@dataclass(frozen=True)
class Loop:
    header: LoopHeader
    body: LoopBody
    

# ------------------------------------
# --------- SOAP analysis ------------
# ------------------------------------

relax_index_merging_requirements = True

"""
Container class  storing parameters for access functions in SOAP statements.
It contains information about the ssa (reduction) dimensions in this statement,
as well as access offsets (required for stencil/overlapping accesses)
"""


@dataclass
class AccessParams:
    ssa_dim: List[dace.symbol] = field(default_factory=list)
    offsets: set() = field(default_factory=set)


"""
Container class storing parameters for non-input arrays in the entire SDFG.
It is used to determine whethere an array is updated "somewhere" in the SDFG,
or is the current SOAP statement updating it. Needed to correctly append the
SSA dimension.
"""


@dataclass
class OutputArrayParams:
    baseAccess: str
    wcr: bool
    ssa_dim: List[dace.symbol] = field(default_factory=list)
    offsets: set() = field(default_factory=set)


def strip_range_name_from_ranges(iters_, ranges_, sep="_"):
    ranges = copy.deepcopy(ranges_)
    iters = copy.deepcopy(iters_)
    if isinstance(ranges, list):
        ranges = {str(rng[0]): (rng[1], rng[2], 0) for rng in ranges}
    if iters:
        if isinstance(iters, dict):
            newIters = {}
            for i, range in iters.items():
                # oldIterVar = range[0].free_symbols.pop()
                iterRange = str(range[1].free_symbols.pop())
                newIterVar = str(i).replace(sep + iterRange, "")
                newIters[newIterVar] = range
            iters = newIters
        else:
            # check if we are operating on state ranges (e.g., [i, 1, N]) or memlet ranges (e.g., [i, i, 1])
            if iters[0][0] == iters[0][1]:
                for i, range in enumerate(iters):
                    oldIterVar = range[0].free_symbols.pop()
                    iterRange = str(ranges[str(oldIterVar)][1].free_symbols.pop())
                    newIterVar = sp.sympify(
                        str(oldIterVar).replace(sep + iterRange, "")
                    )
                    iters[i] = [
                        range[0].subs(oldIterVar, newIterVar),
                        range[1].subs(oldIterVar, newIterVar),
                        range[2],
                    ]
            else:
                for i, range in enumerate(iters):
                    oldIterVar = range[0].free_symbols.pop()
                    iterRange = str(ranges[str(oldIterVar)][1].free_symbols.pop())
                    newIterVar = sp.sympify(
                        str(oldIterVar).replace(sep + iterRange, "")
                    )
                    iters[i] = [
                        range[0].subs(oldIterVar, newIterVar),
                        range[1],
                        range[2],
                    ]
        a = 1
        return iters


@dataclass
class SoapStatement:
    variables: List = field(default_factory=list)
    ranges: Dict = field(default_factory=defaultdict)
    daceRanges: Dict = field(default_factory=dict)
    reductionRanges: List = field(default_factory=list)
    phis: Dict = field(default_factory=dict)
    output_accesses: Dict = field(default_factory=dict)
    Q: dace.symbol = sp.sympify(1)
    name: str = ""
    numExecutions: sp.Expr = sp.sympify(1)
    appended_versions: bool = False
    SSAed: bool = False
    parent_subgraph: "SoapStatement" = None
    subgraph: Set = field(default_factory=set)
    subgraph_inputs: Set = field(default_factory=set)
    in_transients: Set = field(default_factory=set)
    out_transients: Set = field(default_factory=set)
    inner_tile: List = field(default_factory=list)
    outer_tile: List = field(default_factory=list)
    loc_domain_dims: List = field(default_factory=list)
    p_grid: List = field(default_factory=list)
    domain_grid: List = field(default_factory=list)
    wcr_edges: List = field(default_factory=list)
    prefix_trivial_kernels: List = field(default_factory=list)
    suffix_trivial_kernels: List = field(default_factory=list)
    trivial: bool = True
    later_reuse_dim = None

    def __post_init__(self):
        self.ranges = defaultdict(list)

    def get_phis(self):
        flat_phis = {}
        for arrName, baseAccesses in self.phis.items():
            for baseAccess, offsets in baseAccess.items():
                flat_phis[arrName + "[" + baseAccess + "]"] = offsets

        return flat_phis

    def solve(self, solver: Solver) -> None:
        self.update_ranges()
        self.calculate_dominator_size()
        self.calculate_H_size()
        self.count_V()

        self.reductionRanges = [
            red_dim[0]
            for sublist in [
                arr.ssa_dim
                for arr in self.output_accesses.values()
                if arr.ssa_dim is not None
            ]
            for red_dim in sublist
        ]

        if not self.Dom_size or self.Dom_size == 0 or not self.Dom_size.free_symbols:
            self.rhoOpts = oo
            self.Q = self.V / self.rhoOpts
            if "j" in str(self.rhoOpts):
                a = 1
            return

        # if the input size is of the same order as the output size,
        # then the computational intensity is constant
        if sum(sp.degree_list(sp.LT(self.Dom_size))) == sum(
            sp.degree_list(sp.LT(self.H_size))
        ):
            self.rhoOpts = sp.LT(self.H_size) / sp.LT(self.Dom_size)
            if not self.rhoOpts.is_number:
                self.rhoOpts = 1
            self.rhoOpts = sp.sympify(self.rhoOpts)
            self.Q = self.V / self.rhoOpts
            if "j" in str(self.rhoOpts):
                a = 1
            return
        if len(self.Dom_size.free_symbols) != len(self.H_size.free_symbols):
            # we have a scenario like the dominator is `i`, but the computation size is `i*k`,
            # so we have a constant I/O for some free parameter (in this case `k`). What it means
            # is that rho = k
            rho = sp.LT(self.H_size) / sp.LT(self.Dom_size)
            # now we need to resolve the range of that free iteration variable
            all_ranges = [
                [item[0], [item[1], item[2]]]
                for sublist in list(self.ranges.values())
                for item in sublist
            ]
            rangesDict = dict(all_ranges)
            rhoRanges = [
                (d2sp(it), d2sp(r[0]), d2sp(r[1]))
                for (it, r) in rangesDict.items()
                if d2sp(it) in rho.free_symbols
            ]
            for rng in rhoRanges:
                rho = rho / rng[0]
                rho = sp.Sum(rho, rng).doit()

            self.rhoOpts = rho
            if "tmp_index" in str(rho):
                a = 1
            self.Q = self.V / self.rhoOpts
            if "j" in str(self.rhoOpts):
                a = 1
            return

        # TODO: CONTROVERSIAL!
        # self.Dom_size = sp.sympify(str(self.Dom_size).replace('_', ''))
        self.H_size = np.prod(list(self.Dom_size.free_symbols))
        # [self.rhoOpts, self.Xopts, self.varsOpt,
        #  self.inner_tile, self.outer_tile, self.xpart_dims] = \
        #     solver.sympy_solve(self.Dom_size, self.H_size)
        input = str(self.Dom_size).replace("_", "x").replace("**", "^")
        output = str(self.H_size).replace("_", "x").replace("**", "^")
        print("input: " + input + ",   output: " + output)

        self.parse_solution(solver.send_command(input + ";" + output))
        if self.rhoOpts == 0:
            self.rhoOpts = oo
        else:
            self.trivial = False
        if "J" in str(self.rhoOpts) or "I" in str(self.rhoOpts):
            a = 1
        self.Q = self.V / self.rhoOpts

        # if we analyze memory-independent bounds (infinite private memory), we use an x-partition size
        # (varsOpt) as a function of X, and not S. Then, we divide the whole iteration domain by this to
        # solve for X
        if (
            Config.get("soap", "decomposition", "chosen_par_setup")
            == "memory_independent"
        ):
            X = sp.symbols("X")
            p = sp.symbols("p")
            loc_domain_size = sp.prod(self.varsOpt[0])
            X_opt = solve(self.V - (p * loc_domain_size), X)[0]
            # absolutely horrible hack, but sp.sympify is broken - we cannot specify assumptions that
            # all the symbols are positive and the solution also has to be positive. So we need to manually
            # flip the sign if needed.
            if str(X_opt)[0] == "-":
                X_opt = X_opt * (-1)
            self.outer_tile = [
                tile_size.subs(X, X_opt) for tile_size in self.varsOpt[0]
            ]
            self.Q = X_opt

    def parse_solution(self, input: str):
        data = input
        data = data.replace("Inf", "oo")
        data = data.replace("^", "**")
        data = data.replace("S", "Ss")
        data = json.loads(data)
        if data["rhoOpts"][0] == "-1" or data["rhoOpts"][0] == "-2":
            print("Matlab server error")
            print("Error description: " + str(data["varsOpt"][0]))
            print(
                "Parameters: "
                + str(data["Xopts"][0])
                + ", "
                + str(data["variables"][0])
            )
            print("Error code: " + str(data["rhoOpts"][0]))
            quit()
        self.variables = sp.symbols(data["variables"], positive=True)
        try:
            self.rhoOpts = [sp.sympify(expr) for expr in data["rhoOpts"]][0]
            self.varsOpt = [sp.sympify(expr) for expr in data["varsOpt"]]
            self.varsOpt = np.reshape(
                self.varsOpt,
                (len(self.varsOpt) // len(self.variables), len(self.variables)),
            )
            self.Xopts = [sp.sympify(expr) for expr in data["Xopts"]]
            self.inner_tile = [sp.sympify(expr) for expr in data["inner_tile"]]
            self.outer_tile = [sp.sympify(expr) for expr in data["outer_tile"]]
            self.xpart_dims = copy.deepcopy(self.outer_tile)

            self.create_schedule()
        except:
            a = 1

        if "j" in str(self.rhoOpts):
            a = 1

    def create_schedule(self):
        # Get the streaming dimension. If we do have reduction ranges, we pick
        # the direction of the largest tile dimension in it.
        # If we don't have any reduction, we pick just the largest tile dimension among all dimensions
        Ss = sp.symbols("Ss")
        X = sp.symbols("X")
        p = sp.symbols("p")
        streaming = sp.symbols("streaming")
        if not self.reductionRanges:
            self.stream_dim = self.variables[
                np.argmax([t.subs(Ss, 100) for t in self.inner_tile])
            ]
        else:
            # filtered_tile sets to 1 all tile dimenstions that are not in self.reductionRanges
            filtered_tile = [
                v if str(k) in [str(r) for r in self.reductionRanges] else sp.sympify(1)
                for k, v in zip(self.variables, self.inner_tile)
            ]
            self.stream_dim = self.variables[
                np.argmax([t.subs(Ss, 100) for t in filtered_tile])
            ]

        # calculate the outer tile (the parallel local domain) as a function of S, P, and total iteration domain
        loc_domain_size = sp.prod(self.variables).subs(self.stream_dim, X)
        loc_domain_size = loc_domain_size.subs(zip(self.variables, self.outer_tile))
        X_size = solve(self.V - (p * loc_domain_size), X)[0]
        self.outer_tile[self.variables.index(self.stream_dim)] = X_size

        # we EXPLICITLY fix the inner_tile dimension in the streaming dimension to "streaming"
        self.inner_tile[self.variables.index(self.stream_dim)] = streaming


    def init_decomposition(self, subs_list) -> None:
        dimensions = {
            str(d[0]).replace("_", "x"): d[2] - d[1] + 1
            for d in list(self.ranges.values())[0]
        }
        self.dimensions_ordered = [
            sp.sympify(str(dimensions[str(i)])).subs(subs_list) for i in self.variables
        ]
        stream_dim_number = (
            self.variables.index(self.stream_dim) if hasattr(self, "stream_dim") else -1
        )
        self.param_vals = [(sp.symbols(p), val) for (p, val) in subs_list]
        Ss = sp.symbols("Ss")
        X = sp.symbols("X")
        p = sp.symbols("P")
        comm_world = [x[1] for x in self.param_vals if x[0] == p][0]
        S_val = [x[1] for x in self.param_vals if x[0] == Ss][0]

        loc_domain_found = False

        if self.trivial:
            return

        self.trivial = False
        # xpart_dims is the size of the local domain as a function of X
        xpart_dims = copy.deepcopy(self.varsOpt[0])
        # xpart_opt_dims is the size of the local domain as a function of S
        xpart_opt_dims = copy.deepcopy(self.xpart_dims)
        while not loc_domain_found:
            loc_domain_found = True
            loc_domain_dims = [sp.floor(dim) for dim in xpart_opt_dims]

            if any(not isinstance(n, Number) for n in loc_domain_dims):
                if (
                    Config.get("soap", "decomposition", "chosen_par_setup")
                    == "memory_dependent"
                ):
                    loc_domain_vol = sp.prod(self.variables).subs(self.stream_dim, X)
                    loc_domain_vol = loc_domain_vol.subs(
                        zip(self.variables, loc_domain_dims)
                    )
                    solutions = solve(self.V - (p * loc_domain_vol), X)
                    idx = 0
                    if len(solutions) > 0:
                        for i, sol in enumerate(solutions):
                            if sp.sign(sol).args[0] != -1:
                                idx = i
                                break
                    X_size = solutions[idx]
                    # X_size = solve(self.V - (p * loc_domain_vol), X)[0]
                    loc_domain_dims[self.variables.index(self.stream_dim)] = X_size
                else:
                    loc_domain_vol = sp.prod(loc_domain_dims).subs(Ss, X)
                    try:
                        solutions = solve(self.V - (p * loc_domain_vol), X)
                        idx = 0
                        if len(solutions) > 0:
                            for i, sol in enumerate(solutions):
                                if sp.sign(sol).args[0] != -1:
                                    idx = i
                                    break
                        X_size = solutions[idx]
                        # X_size = solve(self.V - (p * loc_domain_vol), X)[0]
                    except:
                        loc_domain_dims = [dim for dim in xpart_opt_dims]
                        loc_domain_vol = sp.prod(loc_domain_dims).subs(Ss, X)
                        print(loc_domain_dims)
                        print(self.V - (p * loc_domain_vol))
                        idx = 0
                        solutions = solve(self.V - (p * loc_domain_vol), X)
                        if len(solutions) > 0:
                            for i, sol in enumerate(solutions):
                                if sp.sign(sol).args[0] != -1:
                                    idx = i
                                    break
                        X_size = solutions[idx]
                        # X_size = solve(self.V - (p * loc_domain_vol), X)[0]
                    loc_domain_dims = [dim.subs(Ss, X_size) for dim in loc_domain_dims]

            self.loc_domain_dims = [
                min(sp.ceiling(loc_dim.subs(self.param_vals)), glob_dim) 
                for (loc_dim, glob_dim) in zip(loc_domain_dims, self.dimensions_ordered)
                ]
            for dim_num, (loc_dim, glob_dim) in enumerate(
                zip(self.loc_domain_dims, self.dimensions_ordered)
            ):
                if loc_dim > glob_dim:
                    xpart_dims[dim_num] = self.dimensions_ordered[dim_num]
                    xpart_opt_dims[dim_num] = self.dimensions_ordered[dim_num]

                    # we do not need to reevaluate it for memory dependent bound,
                    # as there is only one degree of freedom (in the streaming dimension).
                    if (
                        Config.get("soap", "decomposition", "chosen_par_setup")
                        == "memory_independent"
                    ):
                        loc_domain_found = False

        self.p_grid = [
            sp.ceiling((v / t).subs(self.param_vals))
            for v, t in zip(self.dimensions_ordered, self.loc_domain_dims)
        ]
        
        
        # Set the number of local domains per processor. By default, it is 1, that is,
        # we have eonugh processors to have one local domain per processor and interpolate
        # between 2D decomposition (just enough memory, lower bound on P) 
        # and 3D decomposition (strong scaling). If P is too low, we have a "multicore"
        # scenario, where a single processor will have to handle multiple local domains.
        self.num_domains_per_proc = 1
        
        # if self.num_domains_per_proc == 1, then the outer tile is the same as the local domain size. If we increase the number of local domains per processor, we need to increase the outer tile size (while the local domain size remains the same).
        self.outer_tile = copy.deepcopy(self.loc_domain_dims)        
        self.domain_grid = [1] * len(self.variables)
        
        subs_list = {str(k): v for k, v in subs_list}
        alignment = 1
        if "aligment" in subs_list:
            alignment = subs_list["alignment"]
            # if alignment is specified, we round down the values of hte outer tile to the nearest multiple of the alignment. Rembember that the original values of the outer tiles are floats, so we cannot use the % operator
            if alignment > 1:
                self.loc_domain_dims = [max(1, math.floor(x / alignment) * alignment) for x in self.loc_domain_dims]
                self.outer_tile = [max(1, math.floor(x / alignment) * alignment) if isinstance(x, numbers.Number) else x for x in self.outer_tile]
        

        strategy = "squeeze_dims"
        strategy = "increaseX"
        strategy = "narrowest_dim_first"
        while sp.prod(self.p_grid) > comm_world:
            if (
                Config.get("soap", "decomposition", "chosen_par_setup")
                == "memory_dependent"
                and stream_dim_number >= 0
            ):
                if self.p_grid[stream_dim_number] == 1:
                    # Now we are in the multi-core case. What it means:
                    # Given the local domain size self.loc_domain_dims, and the number of local domains 
                    # sp.prod(self.p_grid), we have less processors (comm_world) than needed.
                    # It means that a single processor will have to handle multiple local domains.
                    
                    self.num_domains_per_proc = math.ceil(sp.prod(self.p_grid) / comm_world)
                    
                    if self.later_reuse_dim is not None:
                        # one processor will handle multiple local domains in the later_reuse_dim            
                        # find out which dimension is the later_reuse_dim (in order) from self.variables
                        reuse_dim_number = list(map(str, self.variables)).index(self.later_reuse_dim)
                        
                        if self.p_grid[reuse_dim_number] == 1:
                            # if the later_reuse_dim is already squeezed, we need to squeeze another dimension
                            raise NotImplementedError("Later reuse dimension is already squeezed. Need to implement another strategy.")
                        else:
                            if self.num_domains_per_proc > self.p_grid[reuse_dim_number]:
                                # if the number of local domains per processor is greater than the number of local domains in the later_reuse_dim, we need to decrease the number of local domains in the later_reuse_dim
                                remaining_domains_to_squeze = self.num_domains_per_proc / self.p_grid[reuse_dim_number]
                                self.outer_tile[reuse_dim_number] = \
                                    self.loc_domain_dims[reuse_dim_number]*self.p_grid[reuse_dim_number]
                                self.p_grid[reuse_dim_number] = 1
                                
                                while remaining_domains_to_squeze > 1:
                                    # squeeze another dimension
                                    dim = np.argmax(self.p_grid)
                                    if self.p_grid[dim] == 1:
                                        raise NotImplementedError("All dimensions are squeezed.")
                                    if self.p_grid[dim] >= remaining_domains_to_squeze:
                                        self.outer_tile[dim] = self.loc_domain_dims[dim]*remaining_domains_to_squeze
                                        self.p_grid[dim] /= remaining_domains_to_squeze
                                        remaining_domains_to_squeze = 1
                                        break
                                    remaining_domains_to_squeze /= self.p_grid[dim]
                                    self.outer_tile[dim] = self.loc_domain_dims[dim]*self.p_grid[dim]
                                    self.p_grid[dim] = 1
                            else:
                                raise NotImplementedError("TODO: do something.")
                    else:
                        remaining_domains_to_squeze = self.num_domains_per_proc
                        while remaining_domains_to_squeze > 1:
                            # squeeze another dimension
                            dim = np.argmax(self.p_grid)
                            if self.p_grid[dim] == 1:
                                raise NotImplementedError("All dimensions are squeezed.")
                            if self.p_grid[dim] >= remaining_domains_to_squeze:
                                self.outer_tile[dim] = self.loc_domain_dims[dim]*remaining_domains_to_squeze
                                self.p_grid[dim] /= remaining_domains_to_squeze
                                remaining_domains_to_squeze = 1
                                break
                            remaining_domains_to_squeze /= self.p_grid[dim]
                            self.outer_tile[dim] = self.loc_domain_dims[dim]*self.p_grid[dim]
                            self.p_grid[dim] = 1
                                
                        
                        
            
                    # comm_world *= self.num_domains_per_proc
                    print(
                        "\n\nWARNING!!!\nMemory-dependent bound. For S={}, the minimum number of ranks is p_min={}. \
                        \nHowever, only {} ranks are given.\nTherefore,a single rank will sequentially execute {} local domains.\n\n\n".format(
                            S_val, sp.prod(self.p_grid), comm_world, self.num_domains_per_proc
                        )
                    )
                else:
                    self.p_grid[stream_dim_number] -= 1
            else:
                correct_decomp = False
                X_val = X_size.subs(self.param_vals)
                while not correct_decomp:
                    if strategy == "increaseX":
                        # gradually increase X
                        X_val = (X_val * 1.02).evalf()
                        self.loc_domain_dims = [
                            sp.ceiling(var.subs(X, X_val)) for var in xpart_dims
                        ]
                        self.p_grid = [
                            sp.ceiling((v / t).subs(self.param_vals))
                            for v, t in zip(
                                self.dimensions_ordered, self.loc_domain_dims
                            )
                        ]

                    if strategy == "squeeze_dims":
                        # check in which dimension the squeezing will maintain the shape the best
                        max_ratio = 0
                        best_dim = -1
                        for i in range(len(self.p_grid)):
                            cur_domain_size = (
                                self.dimensions_ordered[i] / self.p_grid[i]
                            )
                            projected_dom_size = self.dimensions_ordered[i] / (
                                self.p_grid[i] - 1
                            )
                            proj_ratio = cur_domain_size / projected_dom_size
                            if proj_ratio > max_ratio:
                                best_dim = i
                                max_ratio = proj_ratio
                        self.p_grid[best_dim] -= 1

                    if strategy == "narrowest_dim_first":
                        # decrease the number of processes in the least populated dimension
                        dim = np.argmax(self.p_grid)
                        self.p_grid[dim]
                        self.p_grid[dim] -= 1

                    # recalculate the constraint
                    if sp.prod(self.p_grid) <= comm_world:
                        correct_decomp = True

        self.outer_tile = [
            np.ceil(d / p) for (d, p) in zip(self.dimensions_ordered, self.p_grid)
        ]

        if sp.prod(self.p_grid) < comm_world:
            print(
                "\n\nWarning!!!\nUsing {} out of {} processors\n\n".format(
                    sp.prod(self.p_grid), comm_world
                )
            )

        else:
            print("\nUsing all {} processors.\n".format(comm_world))


    def get_data_decomposition(self, pp):
        # p_pos is an N -> N^d mapping, translating a global rank to a cooridnate rank in
        # the d-dimensional iteration space  (p  -> [p_x, p_y, p_z, ....])
        if sp.prod(self.p_grid) <= pp:
            print(
                "\n\nDecomposition uses only {} processors. Given rank {} will be idle!\n\n".format(
                    sp.prod(self.p_grid), pp
                )
            )
            return {}

        div = lambda x, y: (x - x % y) / y
        p_pos = []
        for dim_num, dim_size in enumerate(self.p_grid):
            p_dim = div(pp, sp.prod(self.p_grid[dim_num + 1 :]))
            p_pos.append(p_dim)
            pp -= p_dim * sp.prod(self.p_grid[dim_num + 1 :])

        iter_ranges = {
            str(var): (
                p_pos[i] * self.loc_domain_dims[i],
                min((p_pos[i] + 1) * self.loc_domain_dims[i], dim_size),
            )
            for i, (var, dim_size) in enumerate(
                zip(self.variables, self.dimensions_ordered)
            )
        }

        par_distribution = {}
        for arr, accesses in self.phis.items():
            for access, pars in accesses.items():
                rngs = {}
                for it in access.split("*"):
                    rngs[it] = iter_ranges[it.replace("_", "x")]
            par_distribution[arr] = rngs

        return par_distribution


    def update_ranges(self):
        # converting ranges from "Range" type to list type
        for Sname, Sranges in self.daceRanges.items():
            for daceVar in Sranges:
                daceRange = Sranges[daceVar]
                if isinstance(daceRange, Range):
                    daceRange = daceRange[0]
                Sranges[daceVar] = daceRange

            iterVars = [dace.symbol(k) for k in list(Sranges.keys())]
            iterVarsStr = [str(x) for x in iterVars]
            while Sranges:
                daceRangesTmp = copy.deepcopy(Sranges)
                for daceVar in daceRangesTmp:
                    iterVar = dace.symbol(daceVar)
                    daceRange = Sranges[daceVar]
                    rangeVars = np.sum(list(daceRange)[:2]).free_symbols
                    rangeVarsStr = [str(x) for x in rangeVars]
                    if set(iterVarsStr).intersection(set(rangeVarsStr)):
                        continue
                    iterStep = daceRange[2]
                    if iterStep == 1 or iterStep == 0:
                        iterStart = daceRange[0]
                        iterEnd = daceRange[1]
                    elif iterStep == -1:
                        iterStart = daceRange[1]
                        iterEnd = daceRange[2]
                    else:
                        exit("incorrect step in iteration range!")

                    self.ranges[Sname].append((iterVar, iterStart, iterEnd))
                    iterVarsStr.remove(daceVar)
                    del Sranges[daceVar]

    def calculate_H_size(self):
        # SDG subgraph may span different scopes with different ranges
        # (e.g., containing one tasklet from a 3D iteration space and and
        # another one from a 2D iteration space). In such cases, we take the
        # single scope with maximal dimension.
        max_dom_size = 0
        max_rngs_state = []
        for st, rngs in self.ranges.items():
            if len(rngs) > max_dom_size:
                max_dom_size = len(rngs)
                max_rngs = rngs
        for st, rngs in self.ranges.items():
            if len(rngs) == max_dom_size:
                max_rngs_state.append(rngs)

        # if we entered this branch, instead of taking one maximal scope, we would
        # add their sizes (e.g., instead of a single I*J*K, we would have I*J*K + I*J)
        if len(max_rngs_state) < 0:
            self.H_size = sum(
                sp.prod(v[0] for v in Sranges) for Sranges in self.ranges.values()
            )
        else:
            for st, rngs in self.ranges.items():
                if len(rngs) > max_dom_size:
                    max_dom_size = len(rngs)
                    max_rngs = rngs
            self.H_size = sp.prod(v[0] for v in max_rngs)

        simplify_H_size = True
        if simplify_H_size:
            # Keep only highest degree terms in the Vh
            # e.g., for Vh = IJ + IK + Kj - I - j - K, we keep only IJ + IK + KJ
            H_Pol = sp.Poly(self.H_size)
            H_vars = H_Pol.gens
            maxDeg = sum(sp.degree_list(sp.LT(self.H_size)))
            simp_H = sp.sympify(0)
            for k, c in zip(H_Pol.monoms(), H_Pol.coeffs()):
                if sum(k) >= maxDeg:
                    monom = c * sp.prod(x**k1 for x, k1 in zip(H_vars, k))
                    simp_H += monom
            self.H_size = d2sp(simp_H)

        if "**" in str(self.H_size):
            raise Exception(
                "Incorrect subcomputation volume H_size = {}. It cannot contain powers".format(
                    self.H_size
                )
            )

    def simplify_dom_size(self):
        # another simplification term. Keep only highest degree terms in the denominator
        # e.g., for Dom = IJ + IK + Kj - I - j - K, we keep only IJ + IK + KJ
        DomPol = sp.Poly(self.Dom_size)
        Dom_sizears = DomPol.gens
        # NEW_VER
        maxDeg = min(sum(sp.degree_list(sp.LT(self.Dom_size))), 3)
        simpDom = sp.sympify(0)
        for k, c in zip(DomPol.monoms(), DomPol.coeffs()):
            if sum(k) >= maxDeg - 1:
                # we filter out the negative terms (e.g., -i -j)
                if c > 0:
                    monom = c * sp.prod(x**k1 for x, k1 in zip(Dom_sizears, k))
                    simpDom += monom
        self.Dom_size = d2sp(simpDom)

    def calculate_dominator_size(self, strip_array_versions: bool = False):
        if strip_array_versions:
            # remove the version numbers from array names
            stripped_phis = {}
            for arr_name, base_accesses in self.phis.items():
                strip_name = "_".join(arr_name.split("_")[:-1])
                stripped_phis[strip_name] = base_accesses

            stripped_outputs = {}
            for arr_name, base_accesses in self.output_accesses.items():
                strip_name = "_".join(arr_name.split("_")[:-1])
                stripped_outputs[strip_name] = base_accesses
        else:
            stripped_phis = self.phis
            stripped_outputs = self.output_accesses

        self.Dom_size = sp.simplify(0)

        # Quick check if the reduction of symbolic size is present. If yes, then don't add the SSA dim.
        # There is a symbolic reduction if the highest polynomial degree among all inputs
        # is smaller than the polynomial degree of the output
        max_inp_degree = max(
            [
                len(next(iter(access_params.offsets)))
                for array_accesses in list(stripped_phis.values())
                for access_params in array_accesses.values()
            ]
        )

        out_degree = len(
            sp.sympify(
                "+".join(
                    [
                        base_access
                        for array_accesses in list(stripped_phis.values())
                        for base_access in array_accesses.keys()
                    ]
                )
            ).free_symbols
        )

        # second necessary condition for the symbolic reduction: for a given input array, either
        # it does not have any ssa_dim assigned, or all its ssa_dims (temporal dimensions) are part of
        # the baseAccess of some of the output arrays (spatial diemnsions).
        # Example 1:
        # stencils like fdtd2d, input hx[i], ssa_dim = [t].  Output, hz[i,j], ssa_dim = [t].
        # the temporal dimension t is NOT a spatial dimension of any of the outputs. Therefore there is no symbolic reduction.
        # Example 2:
        # chain MMM, where C is the output of one GEMM and the input of the second GEMM. Then,
        # input C[i,j], ssa_dim = [k]. Output, D[i,k], ssa_dim = [j]. The temporal dimension k is the spatial dimension of the output.
        # therefore, the symbolic reduction is present.
        list_of_lists = [
            params.baseAccess.split("*") for params in self.output_accesses.values()
        ]
        out_spat_dims = set([var for vars in list_of_lists for var in vars])

        # iterate over input accesses
        for array_name, array_accesses in stripped_phis.items():

            if max_inp_degree < out_degree and all(
                [
                    (
                        params.ssa_dim == []
                        or all([str(ssa[0]) in out_spat_dims for ssa in params.ssa_dim])
                    )
                    for params in array_accesses.values()
                ]
            ):
                sym_reduction = True
            else:
                sym_reduction = False

            # iterate over different base accesses to the same array
            for base_access, access_params in array_accesses.items():
                dim = len(next(iter(access_params.offsets)))

                vars = sp.sympify(base_access.split("*"))
                t = []
                access_size = sp.simplify(1)

                # iterate over different offsets in the same base access. Creating access offset set t
                for offset in access_params.offsets:
                    for i in range(dim):
                        if len(t) <= i:
                            t.append(set())
                        t[i].add(offset[i])

                # check if we need to add the SSA dimension
                SSAing = False

                # if the symbolic reduction is present, we don't add the SSA dimension
                if not sym_reduction:
                    # we need to check two things to add the SSA dimension to the dominator set:
                    # 1. this array (array_name) is ever updated SOMEWHERE in the SDFG
                    # 2. it has to have the same input base access as the update base access
                    # TODO: new, to be checked
                    if array_name in self.output_arrays.keys():

                        # if array_name in self.output_arrays.keys() and \
                        #         eq_accesses(base_access, self.output_arrays[array_name][0]):
                        # We only checked so far that the current array is updated SOMEWHERE in our SDFG scope.
                        # Now we need to check if THIS statement updates it:
                        if array_name in stripped_outputs.keys() and eq_accesses(
                            base_access, stripped_outputs[array_name].baseAccess
                        ):
                            SSAing = True

                        # looping over all SSA dimensions
                        for c, ssa_dim in enumerate(access_params.ssa_dim):

                            # TODO: Is the following necessary? Why?
                            # for i in range(dim):
                            #     # check if this output array is transient
                            #     if self.output_arrays[array_name][1] == []:
                            #         # if so, at least it should be in the output accesses, properly unrolled
                            #         t[i].add(self.output_accesses[array_name][1][i])
                            #     else:
                            #         t[i].add(self.output_arrays[array_name][1][i])  #stripped_outputs[array_name][1][i])

                            # We add this offset only to the first ssa_dim. This is necessary if more than one ssa_dim
                            # is present. Without this, we will have an incorrect (diagonal) offset.
                            if c == 0:
                                t.append({0, 1})
                            else:
                                # TODO: controversial. By putting a pass here, we take only one reduction dimension.
                                # Confront, MTTKRP: "ijk,jl,kl->il". Temporary partial result - should it be two-dimensional (il:
                                # then, we keep this continue), or three-dimensional (e.g., ilk: then, we remove this continue)
                                continue
                                t.append({0})
                            dim += 1
                            vars.append(sp.sympify(str(ssa_dim[0])))
                            base_access += "*" + str(ssa_dim[0])
                            # # if ssa_dim is empty, then every element is updated only once (not a parametric number of times).
                            # # Then, we don't add additional parametric dimension.
                            # if len(access_params.ssa_dim) == 1:
                            #     vars.append(sp.sympify(str(access_params.ssa_dim[0][0])))
                            #     base_access += "*" + str(access_params.ssa_dim[0][0])
                            # elif SSAing:
                            #     ssa_dim = sp.sympify("temp")
                            #     vars.append(ssa_dim)
                            #     base_access += "*" + str(ssa_dim)
                            # if not (len(access_params.ssa_dim) > 0 or SSAing):
                            #     dim -= 1

                for i in range(dim):
                    access_size *= vars[i] - len(t[i]) + 1
                    # accessSize *= (vars[i] - len(t[i] - set({0})))
                access_size = sp.simplify(2 * sp.sympify(base_access) - access_size)

                if SSAing:
                    access_size -= sp.sympify(base_access)

                self.Dom_size += sp.simplify(access_size)

        if len(self.Dom_size.free_symbols) > 0:
            self.simplify_dom_size()

        self.Dom_size = d2sp(self.Dom_size)

    def count_V(self):
        if "hz_guard_1;ex_guard_1" in self.name:
            a = 1

        self.V = 0

        # TODO: experimental. Instead of summing the sizes for each of the scopes,
        # we take only the largest one
        max_V = 0
        for Sname, Sranges in self.ranges.items():
            # Sranges = StripRangeNameFromRanges(Sranges_, Sranges_)
            Visize = sp.prod(srange[0] for srange in Sranges)
            # reorder the loops so that dependent ranges are inntermost
            dependentSymbols = [str(x) for x in Visize.free_symbols]
            ordered_ranges = []
            unordered_ranges = copy.deepcopy(Sranges)
            while unordered_ranges:
                for loop in unordered_ranges:
                    if not (
                        any([str(x) in dependentSymbols for x in loop[2].free_symbols])
                    ) and not (
                        any([str(x) in dependentSymbols for x in loop[1].free_symbols])
                    ):
                        ordered_ranges.append(loop)
                        if str(loop[0]) in dependentSymbols:
                            dependentSymbols.remove(str(loop[0]))
                        unordered_ranges.remove(loop)

            Sranges = ordered_ranges

            # check if the dominator set is degenerated
            if Visize == sp.prod(self.Dom_size.free_symbols):
                degeneratedDom = True
            else:
                degeneratedDom = False

            Vsize = 1
            self.W = 1
            for loop in reversed(Sranges):
                self.W = sp.Sum(self.W, loop).doit()
                if str(loop[0]) in [str(x) for x in list(Visize.free_symbols)]:
                    Vsize = sp.Sum(Vsize, loop).doit()
                else:
                    if degeneratedDom:
                        a = 1
                        Vsize = sp.Sum(Vsize, loop).doit()

            if (
                str(max_V)
                == "dace_m_0*dace_n_0 + dace_m_0*(dace_n_0**2/2 - dace_n_0/2)"
            ):
                a = 1
            if compare_Q(Vsize, max_V):
                self.V = d2sp(Vsize)
                max_V = Vsize
            # self.V += d2sp(Vsize)

        if "j" in str(self.V):
            a = 1

    def concatenate_sdg_statements(self, pred, in_S):
        # update iteration vars used by the statements. They are stored in self.all_iter_vars
        self.find_all_iter_vars()
        in_S.find_all_iter_vars()

        self.calculate_H_size()
        initial_iteration_space_degree = sum(
            [poly(self.H_size).degree(v) for v in poly(self.H_size).free_symbols]
        )
        inS_iteration_space_degree = sum(
            [poly(in_S.H_size).degree(v) for v in poly(in_S.H_size).free_symbols]
        )

        # find matching iteration variables
        iters_to_merge = self.match_iter_vars(in_S, pred)
        if len(set(iters_to_merge.values())) != len(list(iters_to_merge.values())):
            return -1

        in_S.swap_iter_vars(iters_to_merge, {})

        # self.swap_iter_vars(self, swaplist, inv_swaplist, solver)
        # self.name = ';'.join(list(OrderedSet(self.name.split(';')).union(OrderedSet(in_S.name.split(';')))))
        # In Python versions >= 3.7, dict is guaranteed to preserve order
        self.name = ";".join(
            list(
                dict.fromkeys(self.name.split(";"))
                | dict.fromkeys(in_S.name.split(";"))
            )
        )
        self.tasklet = in_S.tasklet + self.tasklet
        self.subgraph = self.subgraph.union(in_S.subgraph)
        self.subgraph_inputs = self.subgraph_inputs.union(
            in_S.subgraph_inputs
        ).difference(self.subgraph)
        self.output_arrays = {**self.output_arrays, **in_S.output_arrays}

        # # If one statement's output is the input of the other statement, then
        # # we have only one "final output" of the subgraph. In this case, we keep
        # # the ranges only from the "final" computation. If not, we keep all ranges.

        # if pred:
        #     not_consumed_prev_outputs = list(in_S.output_accesses.keys())
        #     for concatenation_array in in_S.output_accesses.keys():
        #         if concatenation_array in self.phis.keys():
        #             self.match_iter_vars(in_S, concatenation_array, True)
        #             not_consumed_prev_outputs.remove(concatenation_array)

        # # if not pred or len(not_consumed_prev_outputs) > 0:
        # #     # merge ranges of two statements
        # #     for in_state, in_ranges in in_S.ranges.items():
        # #         if in_state in self.ranges.keys():
        # #             for in_rng in in_ranges:
        # #                 if all(in_rng[0] != rng[0] for rng in self.ranges[in_state]):
        # #                     self.ranges[in_state].append(in_rng)
        # #         else:
        # #             self.ranges[in_state] = in_ranges

        # merge ranges of two statements
        for in_state, in_ranges in in_S.ranges.items():
            if in_state in self.ranges.keys():
                for in_rng in in_ranges:
                    if all(in_rng[0] != rng[0] for rng in self.ranges[in_state]):
                        self.ranges[in_state].append(in_rng)
            else:
                self.ranges[in_state] = in_ranges

        if not self.has_correct_ranges():
            a = 1

        # merge inputs
        for array, phi in in_S.phis.items():
            if array in self.phis:
                for baseAccess in phi.keys():
                    # check for an equivalent existing access (e.g. i*j and j*i)
                    equiv_base_access = [
                        other
                        for other in self.phis[array].keys()
                        if sp.sympify(baseAccess) == sp.sympify(other)
                    ]

                    if baseAccess in self.phis[array].keys():
                        self.phis[array][baseAccess].offsets |= copy.deepcopy(
                            phi[baseAccess].offsets
                        )
                    elif equiv_base_access:
                        self.phis[array][equiv_base_access[0]].offsets |= copy.deepcopy(
                            phi[baseAccess].offsets
                        )
                    else:
                        self.phis[array][baseAccess] = copy.deepcopy(phi[baseAccess])
            else:
                self.phis[array] = copy.deepcopy(phi)

        for array, access in in_S.output_accesses.items():
            # TODO: check this condition
            # if array not in self.phis:
            self.output_accesses[array] = copy.deepcopy(access)

        # if exists A such that self.phis[A].base_access == in_S.output_accesses[A].base_access then the output of in_S
        # is the input of self, so we remove it from phis
        if pred:
            array = "_".join(pred.split("_")[:-1])
            if array not in in_S.phis.keys():
                out_access_params = in_S.output_accesses[array]
                in_s_out_access = out_access_params.baseAccess
                if relax_index_merging_requirements:
                    in_s_out_access = in_s_out_access.replace("_", "")
                if array in self.phis.keys():
                    for access, access_params in copy.copy(self.phis[array]).items():
                        cur_access = access
                        if relax_index_merging_requirements:
                            cur_access = cur_access.replace("_", "")
                        if cur_access == in_s_out_access:
                            del self.phis[array][access]
                    if len(self.phis[array]) == 0:
                        del self.phis[array]

        self.calculate_H_size()
        post_iteration_space_degree = sum(
            [poly(self.H_size).degree(v) for v in poly(self.H_size).free_symbols]
        )
        if post_iteration_space_degree > max(
            initial_iteration_space_degree, inS_iteration_space_degree
        ):
            return -1

        return 0

    def add_edge_to_statement(self, memlet: dace.Memlet, ssa_dim: List[dace.symbol]):
        if memlet.subset is None or memlet.is_empty():
            return

        # all iteration variables known to this statement
        itervars = set(
            map(dace.symbol, [v.keys() for v in self.daceRanges.values()][0])
        )
        (arrayName, baseAccess, offsets) = get_access_from_memlet(memlet, itervars)
        if arrayName not in self.phis.keys():
            self.phis[arrayName] = defaultdict(set)
        # check if the access is SOAP: that is, if the same array is accessed by
        # DIFFERENT baseAccess, we need to check if they overlap

        if len(self.phis[arrayName]) > 0:
            existingPhi = list(self.phis[arrayName])[0].split("*")
            baseAccessVec = baseAccess.split("*")
            if baseAccessVec != existingPhi:
                if Config.get("soap", "analysis", "all_injective"):
                    warn(
                        "Assuming that the access "
                        + arrayName
                        + str(existingPhi)
                        + " never overlaps with "
                        + arrayName
                        + str(baseAccessVec)
                        + "\n"
                    )
                else:
                    warn(
                        "Assuming that the access "
                        + arrayName
                        + str(existingPhi)
                        + " perfectly overlaps with "
                        + arrayName
                        + str(baseAccessVec)
                        + "], "
                        + "thus, the latter one does not generate any additional inputs.\n"
                    )
                    return

        # check for an equivalent existing access (e.g. i*j and j*i)
        equiv_base_access = [
            other
            for other in self.phis[arrayName].keys()
            if sp.sympify(baseAccess) == sp.sympify(other)
        ]
        if equiv_base_access:
            baseAccess = equiv_base_access[0]

        if baseAccess not in self.phis[arrayName].keys():
            # self.phis[arrayName][baseAccess] = set()
            access_params = (
                AccessParams()
            )  # namedtuple('params', ['ssa_dim', 'offsets'])
            self.phis[arrayName][baseAccess] = access_params
            self.phis[arrayName][baseAccess].offsets = set()

        self.phis[arrayName][baseAccess].offsets.add(tuple(offsets))
        # will this work?
        self.phis[arrayName][baseAccess].ssa_dim = ssa_dim

    def add_output_edge_to_statement(
        self, memlet: dace.Memlet, ssa_dim: List[dace.symbol]
    ):
        # all iteration variables known to this statement
        itervars = set(
            map(dace.symbol, [v.keys() for v in self.daceRanges.values()][0])
        )
        (arrayName, baseAccess, offsets) = get_access_from_memlet(memlet, itervars)

        self.output_accesses[arrayName] = OutputArrayParams(
            baseAccess, memlet.wcr is not None, offsets=tuple(offsets), ssa_dim=ssa_dim
        )

    def match_iter_vars(self, in_S, pred) -> Dict:
        """ "
        here we can also resolve matching iteration variables. E.g.,
        if self.phis[concatenation_array] == A[tmp_for_3] and in_S.output_accessses == A[tmp_for_4],
        then we know that tmp_for_3 == tmp_for_4.
        BUT! Their ranges also must match, e.g., temp[i] and temp[j],
        if i in range(N) and j in range(i, N) SHOULD NOT be merged
        """
        iters_to_merge = {}
        for concatenation_array in list(in_S.phis.keys()) + list(
            in_S.output_accesses.keys()
        ):
            if concatenation_array in self.phis.keys():
                if concatenation_array in in_S.output_accesses.keys():
                    join_base_accesses = [
                        in_S.output_accesses[concatenation_array].baseAccess
                    ]
                else:
                    join_base_accesses = list(in_S.phis[concatenation_array].keys())

                for join_access in join_base_accesses:
                    for base_access in self.phis[concatenation_array].keys():
                        # TODO: controversial commenting out this check: if eq_accesses(base_access, join_access):
                        # it doesn't make sense to merge identical accesses (e.g., A[i] -> A[i])
                        # it makes sense only for non-trivial merges, e.g., A[tmp_1] -> A[tmp_2]
                        if base_access != join_access:
                            its = dict(
                                zip(join_access.split("*"), base_access.split("*"))
                            )

                            # we also don't merge if the iteration variable is also present in another access.
                            # E.g., S1: A[i,k] can be merged with S2: A[i,j] only if k is not used in S2 and
                            # j is not used in S1:
                            for it1, it2 in copy.copy(its).items():
                                if it2 in in_S.all_iter_vars or (
                                    it1 in self.all_iter_vars and pred != None
                                ):
                                    del its[it1]

                            iters_to_merge = {**iters_to_merge, **its}

        return iters_to_merge

    # def subs_itervars_names(self, iters_to_merge):
    #     for old,new in iters_to_merge.items():
    #         # renaming ranges
    #         for scope, ranges in self.ranges.items():
    #             rng_dict = rng_list2dict(ranges)
    #             for i, rng in enumerate(ranges):
    #                 if str(rng[0]) in iters_to_merge.keys():
    #                     new_it = dace.symbol(iters_to_merge[str(rng[0])])
    #                     # check if iteration variable (new_it) is present in our scope to swap it
    #                     if new_it not in rng_dict.keys():
    #                     #     # and if it is, the ranges must match
    #                     #     if rng_dict[new_it][0].free_symbols == rng[1].free_symbols \
    #                     #             and rng_dict[new_it][1].free_symbols == rng[2].free_symbols:
    #                     #         ranges[i] = (dace.symbol(iters_to_merge[str(rng[0])]), rng[1], rng[2])
    #                     # else:
    #                         ranges[i] = (dace.symbol(iters_to_merge[str(rng[0])]), rng[1], rng[2])

    #         # renaming phis
    #         for accesses in copy.copy(self.phis.values()):
    #             for access, params in copy.copy(accesses.items()):
    #                 merged_access = access
    #                 for k,v in iters_to_merge.items():
    #                     merged_access = merged_access.replace(v, k)
    #                 if merged_access != access:

    #                 # merge input reuse access variables
    #                 in_S.phis[concatenation_array][merged_access] = in_S.phis[concatenation_array][join_access]
    #                 del in_S.phis[concatenation_array][join_access]

    #         if out_join:
    #             # TODO: experimental
    #             del self.phis[concatenation_array]

    def find_all_iter_vars(self):
        self.all_iter_vars = set()
        for accesses in self.phis.values():
            for access, params in accesses.items():
                self.all_iter_vars = self.all_iter_vars.union(set(access.split("*")))
                self.all_iter_vars = self.all_iter_vars.union(
                    set([str(i) for [i, j, k] in params.ssa_dim])
                )

    def print_schedule(self):
        print("Kernel:              " + self.name)
        print("Iteration variables: " + str(self.variables))
        print("Inner tile:          " + str(self.inner_tile))
        print("Outer tile:          " + str(self.outer_tile))
        print("Streaming direction: " + str(self.stream_dim))
        print("Comp. intensity:     " + str(self.rhoOpts))

    # TODO:
    # below is the work in progress for proper depth counting

    #    # do the same to update numExecutions
    #    # dependentSymbols = [str(x) for x in self.numExecutions.free_symbols]
    #     ordered_iteration_ranges = []
    #     dependentSymbols = [str(var[0]) for var in self.ranges
    #                                 if str(var[0]) in [str(x) for x in sp.sympify(self.numExecutions).free_symbols]]
    #     unordered_ranges = [rng for rng in self.ranges if str(rng[0]) in dependentSymbols]
    #     while unordered_ranges:
    #         for loop in unordered_ranges:
    #             if not (any([str(x) in dependentSymbols for x in loop[2].free_symbols])) and \
    #                     not (any([str(x) in dependentSymbols for x in loop[1].free_symbols])):
    #                 ordered_iteration_ranges.append(loop)
    #                 if str(loop[0]) in dependentSymbols:
    #                     dependentSymbols.remove(str(loop[0]))
    #                 unordered_ranges.remove(loop)

    #     # # we fist calculate number of executions which are iteration-variable-dependent
    #     # if "i" in str(self.numExecutions):
    #     #     a = 1
    #     #     # i = dace.symbol("i")
    #     #     # N = dace.symbol("N")
    #     #     # if "j" in str(self.numExecutions):
    #     #     #     j = dace.symbol("j")
    #     #     #     self.numExecutions = self.numExecutions.subs([[i, N], [j, N]])/6
    #     #     # else:
    #     #     #     self.numExecutions = self.numExecutions.subs(i, N)/2
    #     # dependentExec = sp.sympify(1)
    #     # for loop in reversed(ordered_iteration_ranges):
    #     #     dependentExec = sp.Sum(dependentExec, loop).doit()

    #     # # and finally we multiply by the independent part. To get the independent part, we substitute iteration
    #     # # variables with 1
    #     # subsList = []
    #     # dependentSymbols = [(var[0]) for var in self.ranges
    #     #                             if str(var[0]) in [str(x) for x in sp.sympify(self.numExecutions).free_symbols]]
    #     # for depVar in dependentSymbols:
    #     #     subsList.append([depVar, 1])
    #     # independentExec = self.numExecutions.subs(subsList)

    #     # for loop in reversed(ordered_iteration_ranges):
    #     #     independentExec = sp.Sum(independentExec, loop).doit()

    #     # self.numExecutions = dependentExec * independentExec

    def count_D(self):
        self.D = sp.sympify(1)
        for redRange in self.reductionRanges:
            redVar = redRange[0]
            thisD = sp.log(sp.Sum(1, redRange)).doit()
            self.D += thisD

        cur_ranges = next(iter(self.ranges.values()))
        cur_ranges_dict = {str(x[0]): (x[1], x[2], 1) for x in cur_ranges}

        while True:
            dependentVars = [
                var[0]
                for var in cur_ranges
                if str(var[0]) in [str(x) for x in self.D.free_symbols]
            ]
            if not dependentVars:
                break
            for var in dependentVars:
                varRange = [rng for rng in cur_ranges if rng[0] == var][0]
                dfdx = sp.diff(self.D, var)
                if dfdx.free_symbols:
                    subsList = []
                    for symbol in dfdx.free_symbols:
                        subsList.append([symbol, 10 if symbol == var else 1000])
                    dfdx = dfdx.subs(subsList)
                if dfdx > 0:
                    self.D = self.D.subs(var, varRange[2])
                else:
                    self.D = self.D.subs(var, varRange[1])

        if self.name == "T83":
            a = 1

        # while (any(str(x) in self.SOAPranges for x in self.numExecutions.free_symbols)):
        #     x = [x for x in self.numExecutions.free_symbols if str(x) in self.SOAPranges][0]
        #     cur_range = self.SOAPranges[str(x)]
        #     self.numExecutions = sp.Sum(self.numExecutions, (x, cur_range[0], cur_range[1])).doit()

        # reductionRange may be dynamic (dependent on outer iteration variables)
        # if so, use self.ranges to resolve it

        # self.ranges is a dict of ranges indexed by the state name (to encompass SDG statements
        # spanning multiple states). However, counting D is only relevant for atomic (single-tasklet)
        # statements, so we can just take the first element of the dictionary
        if len(self.ranges) > 1:
            raise Exception(
                "Counting statement depth on the composite subgraph statement"
            )

        d = 1
        for cur_depth in reversed(self.numExecutions):
            x = sp.sympify(str(cur_depth[0]))
            d = sp.Sum(d, (x, 1, cur_depth[1])).doit()
        #     dependent_iter_vars = [x for x in cur_depth.free_symbols if str(x) in cur_ranges_dict.keys()]
        #     if len(dependent_iter_vars) == 0:
        #         x = sp.sympify('aux_iter_var')
        #         relevant_range = (0, cur_depth-1)
        #     else:
        #         x = dependent_iter_vars[0]
        #         relevant_range = cur_ranges_dict[str(x)]
        #     d = sp.Sum(d, (x, relevant_range[0], relevant_range[1])).doit()
        # d /= sp.simplify(self.numExecutions[0])
        if "-" in str(d):
            a = 1

        self.D *= d

        if "i" in str(self.D):
            a = 1

    def max_rho(self):
        solverTimeout = 15
        inputs = self.Dom_size
        outputs = sp.prod(inputs.free_symbols)

        vars = list(inputs.free_symbols.union(outputs.free_symbols))
        self.variables = vars
        u = sp.symbols(["u{}".format(i) for i in range(len(vars))])
        A, X, M = sp.symbols("A, X, M", positive=True)
        ineqconstr = [a * b for a, b in zip(u, np.array(list(vars)) - 1)]

        # form the Lagrangian from the KKT multipliers
        L = outputs - A * (inputs - X) - sum(ineqconstr)
        grad = [L.diff(x) for x in vars + [A]]
        sys = grad + ineqconstr
        varss = vars + [A] + u
        #  print("\nsys:\n" + str(sys))
        #  print("\nvarss:\n" + str(varss))
        #  print("\nsolve(sys,vars):\n")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(solve, sys, varss)
            try:
                sol = future.result(timeout=solverTimeout)
            except:
                a = 1
        # print(return_value)

        # sol = solve(sys, varss)

        Xoptstmp = []
        rhoOptstmp = []
        varsOpttmp = []
        Xopts = []
        rhoOpts = []
        varsOpt = []
        maxRho = sp.sympify(0)
        if self.name == "comp_iout":
            a = 1
        for i in range(len(sol)):
            varOpt = sol[i][: len(vars)]
            rho = (outputs / (X - M)).subs(tuple(zip(vars, varOpt)))
            # Xopt = solve(sp.diff(rho, X), X)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(solve, sp.diff(rho, X), X)
                try:
                    Xopt = future.result(timeout=solverTimeout)
                    a = 1
                except:
                    Xopt = []
                    a = 1
            if not Xopt:
                curRho = sp.limit(rho, X, oo)
                Xopt = oo
            else:
                if len(Xopt) > 1:
                    # TODO: again, something fishy... We have suspiciously many solutions to optimal rho.
                    oneGoodRho = False
                    for XoptSol in Xopt:
                        if M in XoptSol.free_symbols:
                            curRho = sp.simplify(rho.subs(X, XoptSol[0]))
                            if not oneGoodRho:
                                oneGoodRho = True
                            else:
                                # now it means that more than one solution contains M
                                sys.exit("Too many solutions in dRho == 0")
                    if not oneGoodRho:
                        sys.exit("Many solutions, but none with M in dRho == 0")
                else:
                    curRho = sp.simplify(rho.subs(X, Xopt[0]))

            # TODO: Do we want only the best solution, or all of them?
            if curRho.subs(M, 1000) > maxRho.subs(M, 1000):
                maxRho = curRho
                rhoOpts = maxRho
                varsOpt = varOpt
                Xopts = Xopt

        self.rhoOpts = rhoOpts
        self.varsOpt = varsOpt
        self.Xopts = Xopts
        
        self.trivial = False
        return [rhoOpts, varsOpt, Xopts]

    def swap_iter_vars(self, swaplist, inv_swaplist, solver=[]):
        # used for a loop swap transformation, e.g., interchanging i<->j

        # swap inputs (phis)
        temp_phis = defaultdict(set)
        for arr, base_accesses in self.phis.items():
            temp_phis[arr] = defaultdict(set)
            for base_access in base_accesses:
                new_base = swap_in_string(base_access, swaplist, inv_swaplist)
                temp_phis[arr][new_base] = copy.deepcopy(self.phis[arr][base_access])
        self.phis = temp_phis

        # swap outputs
        for arr, access in self.output_accesses.items():
            self.output_accesses[arr].baseAccess = swap_in_string(
                access.baseAccess, swaplist, inv_swaplist
            )

        # swap ranges
        for st, rngs in self.ranges.items():
            new_rngs = []
            for rng in rngs:
                new_rngs += [
                    (
                        dace.symbol(
                            swap_in_string(str(rng[0]), swaplist, inv_swaplist)
                        ),
                        rng[1],
                        rng[2],
                    )
                ]
            self.ranges[st] = new_rngs

        # recalculate
        if solver != []:
            self.solve(solver)
            
            
# --------------------------------------------
# ---------- code generation  ----------------
# --------------------------------------------
        
    def generate_omp_code(self, decomp_params: dict, alignment:int=32) -> LoopBody:
        """
        Create a nested loop structure for the given statement given
        the outer_tile for parallelization and the inner_tile for streaming (sequential schedule for a single parallel rank).

        The output is a dictionary with keys being loop objects and values being the loop body (possibly nested loops)
        """

        """
        Example genarated code for tiled MMM:

# 3D parallelism
@pragma omp parallel for
for i_outer in range(0, M, tile_i_outer):
    @pragma omp parallel for
    for j_outer in range(0, N, tile_j_outer):
        @pragma omp parallel for
        for k_outer in range(0, K, tile_k_outer):
        # sequential schedule within the tile (streaming dimension)
            for k_inner in range(k_outer, min(K, k_outer + tile_k_outer)):
            # two innermost loops are for SIMD parallelism (vectorization or tensor cores)
                @pragma omp simd
                for i_inner in range(i_outer, min(M, i_outer + tile_i_outer)):
                    @pragma omp simd
                    for j_inner in range(j_outer, min(N, j_outer + tile_j_outer)):
                        C[i_inner, j_inner] += A[i_inner, k_inner] * B[k_inner, j_inner]

                    """
        code = OrderedDict()
        # code = []

        # if the inner tile size has the Ss parameter,
        # which correspnods to the
        # memory size, substtite it with S_private symbolic variable.
        # This corresponds to e.g., the warp size in the GPU or the
        # L1 cache size in the CPU
        if "Ss" in str(self.inner_tile):
            self.inner_tile = [x.subs("Ss", "S_private") for x in self.inner_tile]

        # for the outer_tile, we subs with S_shared, which corresponds to the shared memory size in the GPU or the L3 cache in CPU
        if "Ss" in str(self.outer_tile):
            self.outer_tile = [x.subs("Ss", "S_shared") for x in self.outer_tile]
            
        # substitute the tile size with the actual value
        self.inner_tile = [x.subs(decomp_params) for x in self.inner_tile]
        # self.outer_tile = [x.subs(decomp_params) for x in self.outer_tile]
        
        if "aligment" in decomp_params:
            alignment = decomp_params["alignment"]
            # if alignment is specified, we round down the values of hte outer tile to the nearest multiple of the alignment. Rembember that the original values of the outer tiles are floats, so we cannot use the % operator
            if alignment > 1:
                self.loc_domain_dims = [max(1, math.floor(x / alignment) * alignment) for x in self.outer_tile]
                self.inner_tile = [max(1, math.floor(x / alignment) * alignment) if isinstance(x, numbers.Number) else x for x in self.inner_tile]
        
        # prepare the trivial (no-data-reuse) kernels. Since for them the order of execution does not matter,
        # we place them only according to the depth of the loop nest.
        # Create a dictionary of the trivial kernels, where the key is the depth of the loop nest
        prefix_trivial_kernels = OrderedDict()
        for kernel in self.prefix_trivial_kernels:
            depth = len(kernel.variables)
            if depth not in prefix_trivial_kernels:
                prefix_trivial_kernels[depth] = []
            prefix_trivial_kernels[depth].append(kernel)
            
        suffix_trivial_kernels = OrderedDict()
        for kernel in self.suffix_trivial_kernels:
            depth = len(kernel.variables)
            if depth not in suffix_trivial_kernels:
                suffix_trivial_kernels[depth] = []
            suffix_trivial_kernels[depth].append(kernel)

        # Because the program is tiled, each loop has to be tiled first,
        # corresponding to the sequential schedule within a single rank
        # inside the tile.
        #
        # Then, we iterate again over all loops, adding the parallelization
        # pragmas to the outer loops.

        # 1. Generate a sequential schedule within the tile
        # Starting from the innermost dimension,
        # iterate over inner_tile field, such that the "streaming" dimension
        # is the last one in the inner_tile list.
        inner_loop_body = LoopBody()
        cur_indent = 2*len(self.variables)+1

        loops = list(enumerate(self.inner_tile))
        # order by putting the streaming dimension last
        loops_ordered = sorted(loops, key=lambda x: str(x[1]) == "streaming")
        
        cur_depth = len(loops_ordered) + 1
        for j, (i, inner_tile_size) in enumerate(loops_ordered):
            # inner_loop_body = []
            cur_indent -= 1
            cur_depth -= 1
            # get iteration variable:
            var = str(self.variables[i])

            # get the corresponing outer loop iteration variable
            outer_var = str(self.variables[i]) + "_outer"
            
            # check if we have any trivial kernels at this depth
            if cur_depth in prefix_trivial_kernels:
                for trivial_kernel in prefix_trivial_kernels[cur_depth]:
                    inner_loop_body.statements = [
                        f"{' ' * cur_indent}{x}\n" for x in trivial_kernel.tasklet] \
                        + inner_loop_body.statements

            # create the loop object
            loop_start = outer_var
            loop_end = f"min({str(self.loc_domain_dims[i])} + {outer_var}, {self.dimensions_ordered[i]})"
            if str(inner_tile_size) != "streaming":
                pragma = "# simd exec (vectorization or tensor cores)"
            else:
                pragma = ""
                
            if j == 0:
                inner_loop_body.statements += \
                        [f"{' ' * cur_indent}{x}\n" for x in self.tasklet]
                    
            # check for the suffix trivial kernels
            if cur_depth in suffix_trivial_kernels:
                for trivial_kernel in suffix_trivial_kernels[cur_depth]:
                    inner_loop_body.statements += \
                        [f"{' ' * cur_indent}{x}\n" for x in trivial_kernel.tasklet]
                
            header = LoopHeader(
                    loopvar=var,
                    start=loop_start,
                    end=loop_end,
                    step=1,
                    pragmas=pragma,
                    indent=cur_indent
                )
            loop = Loop(header, inner_loop_body)

            inner_loop_body = LoopBody()
            inner_loop_body.statements.append(loop)



        # Starting from the innermost dimension,
        # iterate over variables and create the loop structure
        for i, var in enumerate(self.variables):
            cur_indent -= 1
            # get tile size:
            tile_size = self.loc_domain_dims[i]

            # substitute the tile size with the actual value
            # tile_size = tile_size.subs(decomp_params)
            # create the loop object

            header = LoopHeader(
                loopvar=str(var) + "_outer",
                start=0,
                end=self.dimensions_ordered[i],
                step=tile_size,
                pragmas="@pragma omp parallel for", 
                indent=cur_indent          
                )
            
            loop = Loop(header, inner_loop_body)
            inner_loop_body = LoopBody()
            inner_loop_body.statements.append(loop)

        code = inner_loop_body #OrderedDict(inner_loop_body)
        return code      

    def generate_CUDA_code(self) -> str:
        """
        Create a nested loop structure for the given statement given.
        the outer_tile for parallelization and the inner_tile for streaming (sequential schedule for a single parallel rank).
        Parallelization is done in a kernel call. 
        
        WARNING: For now, we generate Python code, not CUDA C code. CUDA is just an inspiration.
        As an input argument, we pass the parallel rank p, which corresponds to CUDA thread block.
        Given p, we calculate offsets from global indices to local indices.

        Example genarated code for tiled MMM:

def sequential_kernal(A, B, C, p):
    # we are threadblock p, simiarly to the CUDA counterpart:
    # p = blockIdx.x
    # get the 3D mapping of p to process grid
    p_i = p // (self.p_grid[1] * self.p_grid[2])
    p_j = (p // self.p_grid[2]) % self.p_grid[1]
    p_k = p % self.p_grid[2]
    #
    # calculate the offsets given the self.outer_tile
    i_outer = p_i * self.tile_outer[0]
    j_outer = p_j * self.tile_outer[1]
    k_outer = p_k * self.tile_outer[2]

    # each threadblock computes self.num_domains_per_proc local domains
    for domain_nr in range(self.num_domains_per_proc):
        # calculate the offsets given the self.loc_domain_dims
        i_inner = domain_nr // (self.loc_domain_dims[0] * self.loc_domain_dims[1])
        j_inner = (domain_nr // self.loc_domain_dims[0]) % self.loc_domain_dims[1]
        k_inner = domain_nr % self.loc_domain_dims[2]

        # calculate the global indices
        i_start = i_outer + i_inner
        j_start = j_outer + j_inner
        k_start = k_outer + k_inner
        
        # sequential schedule within the tile (streaming dimension)
        for k in range(k_start, min(K, k_start + loc_domain_dims[2])):
        # two innermost loops are for SIMD parallelism (vectorization or tensor cores)
            for i in range(i_start, min(M, i_start + loc_domain_dims[0])):
                for j in range(j_start, min(N, j_start + loc_domain_dims[1])):            
                    C[i, j] += A[i, k] * B[k, j]

        """
        
        # function header will be applied elsewhere
        code = f"""
    # we are threadblock p, simiarly to the CUDA counterpart:
    # p = blockIdx.x
    # get the n-D mapping of p to process grid
    p_remain = p
"""
        grid_remain = np.prod(self.p_grid[1:])
        for i, var in enumerate(self.variables):
            code += f"    p_{var} = p_remain // {grid_remain}\n"
            code += f"    p_remain = p_remain % {grid_remain}\n"
            if i == len(self.variables) - 1:
                break
            grid_size = self.p_grid[i + 1]
            grid_remain = math.ceil(grid_remain/ grid_size)
        
        code += """

    # calculate the offsets given the self.outer_tile
"""
        for i, (var, tile_size) in enumerate(zip(self.variables, self.outer_tile)):
            code += f"    {var}_outer = p_{var} * {tile_size}\n"
            
        # calculate the self.domain_grid based on the self.outer_tile_size and self.loc_domain_dims
        self.domain_grid = [math.ceil(x / y) for x, y in zip(self.outer_tile, self.loc_domain_dims)]
        code += f"""
    # each threadblock computes self.num_domains_per_proc local domains
    for domain_nr in range({self.num_domains_per_proc}):
        # calculate the offsets given the loc_domain_dims
        domain_remain = domain_nr
"""
        grid_remain = np.prod(self.domain_grid[1:])
        for i, var in enumerate(self.variables):
            code += f"        {var}_inner = domain_remain // {grid_remain}\n"
            code += f"        domain_remain = domain_remain % {grid_remain}\n"
            if i == len(self.variables) - 1:
                break
            grid_size = self.domain_grid[i + 1]
            grid_remain = math.ceil(grid_remain / grid_size)
            
        code += """
        # calculate the global indices
"""
        for i, var in enumerate(self.variables):
            code += f"        {var}_start = {var}_outer + {var}_inner*{self.loc_domain_dims[i]}\n"
        
        code += "\n        # sequential schedule within the tile (streaming dimension)\n"
        cur_indent = "    "

        loops = list(enumerate(self.inner_tile))
        # order by putting the streaming dimension last
        loops_ordered = sorted(loops, key=lambda x: str(x[1]) == "streaming", reverse=True)        
        cur_depth = len(loops_ordered) + 1
        for j, (i, _) in enumerate(loops_ordered):
            inner_tile_size = self.loc_domain_dims[i]
            cur_indent += "    "
            code += f"{cur_indent}for {self.variables[i]} in range({self.variables[i]}_start, min({self.dimensions_ordered[i]}, {self.variables[i]}_start + {inner_tile_size})):\n"
        
        for t in self.tasklet:
            code += f"{cur_indent}    {t}\n"
        
        return code      
            
    # Fusing trivial statements, that is the one with no data reuse (rho is constant).
    # Such statements can be executed in any loop nest in any order, since the order does not affect the
    # reuse.
    def append_trivial_kernel_start(self, prev_kernel: 'SoapStatement'):
        '''
        append_trivial_kernel_start is used when prev_kernel.output_arrays and self.phis
        are not disjoint.
        '''
        # find the common arrays
        common_arrays = set(prev_kernel.output_arrays.keys()).intersection(set(self.phis.keys()))
        assert(len(common_arrays) > 0)
        
        # pick the first one
        common_array = common_arrays.pop()
        
        # check if they have the same base access
        com_ar_out = prev_kernel.output_accesses[common_array].baseAccess
        com_ar_in = list(self.phis[common_array].keys())[0]
        if com_ar_out != com_ar_in:
            # TODO: implement the swap_iter_vars
            raise Exception('Base accesses are not the same')
        self.prefix_trivial_kernels.append(prev_kernel)
        
        
    def append_trivial_kernel_end(self, next_kernel: 'SoapStatement'):
        '''
        append_trivial_kernel_end is used when nex_kernel.phis and self.output_arrays
        are not disjoint.
        '''
        # find the common arrays
        common_arrays = set(self.output_arrays.keys()).intersection(set(next_kernel.phis.keys()))
        assert(len(common_arrays) > 0)
        
        # pick the first one
        common_array = common_arrays.pop()
        
        # check if they have the same base access
        com_ar_out = self.output_accesses[common_array].baseAccess
        com_ar_in = list(next_kernel.phis[common_array].keys())[0]
        # if com_ar_out != com_ar_in:
        #     # get the swaplist and the inv_swaplist
        #     swaplist = {x:y for (x,y) in zip(com_ar_out.split('*'), com_ar_in.split('*'))}
        #     inv_swaplist = {y:x for (x,y) in swaplist.items()}
        #     next_kernel.swap_iter_vars(swaplist, inv_swaplist)
        #     raise Warning('Base accesses are not the same')
        self.suffix_trivial_kernels.append(next_kernel)

    # --------------------------------------------
    # ---------- various helper functions --------
    # --------------------------------------------

    def has_correct_ranges(self) -> bool:
        for state, rngs in self.ranges.items():
            vars = [x for (x, y, z) in rngs]
            if len(vars) != len(set(vars)):
                return False

        return True

    # removes version numbers from iteration variables generated by DaCe. E.g.,
    # npbench -> polybench -> gemver, we have both __i0 and __i0_0 iteration variables,
    # which causes wrong dominator sets when we do SDG fusion.
    def clean_iter_vars(self) -> None:
        # this changes [xxx]_[0-9] to [xxx]
        fixed = re.compile(r"(?P<iter>[a-zA-Z0-9])_[0-9]+")
        for arr, accesses in self.phis.items():
            fixed_accesses = defaultdict()
            for access, pars in accesses.items():
                fixed_access = fixed.sub(r"\g<iter>", access)
                fixed_ssas = []
                for ssa in pars.ssa_dim:
                    fixed_iter = dace.symbol(fixed.sub(r"\g<iter>", str(ssa[0])))
                    fixed_ssa = [fixed_iter, fixed_iter, 1]
                    fixed_ssas.append(fixed_ssa)
                pars.ssa_dim = fixed_ssas
                fixed_accesses[fixed_access] = pars

            self.phis[arr] = fixed_accesses

        for arr, accesses in self.output_accesses.items():
            accesses.baseAccess = fixed.sub(r"\g<iter>", accesses.baseAccess)

        # this changes [xxx][0-9] to [xxx]
        # fixed = re.compile(r'(?P<iter>[a-zA-Z0-9])[0-9]+')
        # for arr, accesses in self.phis.items():
        #     fixed_accesses = defaultdict()
        #     for access, pars in accesses.items():
        #         fixed_access = fixed.sub(r'\g<iter>', access)
        #         fixed_ssas = []
        #         for ssa in pars.ssa_dim:
        #             fixed_iter = dace.symbol(fixed.sub(r'\g<iter>', str(ssa[0])))
        #             fixed_ssa = [fixed_iter, fixed_iter, 1]
        #             fixed_ssas.append(fixed_ssa)
        #         pars.ssa_dim = fixed_ssas
        #         fixed_accesses[fixed_access] = pars

        #     self.phis[arr] = fixed_accesses

        # for arr, accesses in self.output_accesses.items():
        #     accesses.baseAccess = fixed.sub(r'\g<iter>', accesses.baseAccess)


# ------------------------------------
# W/D result container
# ------------------------------------


class WDresult:
    W = sp.sympify(0)
    D_manual = sp.sympify(0)
    D_auto = sp.sympify(0)
    avpar_manual = sp.sympify(0)
    avpar_auto = sp.sympify(0)
    Wstr = ""
    D_manual_str = ""
    D_auto_str = ""
    avpar_manual_str = ""
    avpar_auto_str = ""

    def __init__(self):
        self.W = sp.sympify(0)
        self.D_manual = sp.sympify(0)
        self.D_auto = sp.sympify(0)
        self.avpar_manual = sp.sympify(0)
        self.avpar_auto = sp.sympify(0)
        self.Wstr = ""
        self.D_manual_str = ""
        self.D_auto_str = ""
        self.avpar_manual_str = ""
        self.avpar_auto_str = ""


