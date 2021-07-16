# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
polyhedral loop transformation and functions to interface with ISL
"""

import copy
import islpy as isl
import sympy as sp
import dace.symbolic as symbolic
from collections import defaultdict
from dace.sdfg import nodes
from dace import sdfg as sd, SDFG, Memlet, subsets, dtypes
from dace.subsets import Range, Subset
from dace.codegen import control_flow as cflow
from dace.frontend.python.astutils import negate_expr
from dace.symbolic import pystr_to_symbolic
from dace.transformation.polyhedral.sympy_isl_conversion import SympyToPwAff, \
    extract_end_cond, extract_step_cond, to_sympy, sympy_to_pystr
from typing import List, Set, Dict, Tuple, Union

from dace.transformation.polyhedral.sympy_isl_conversion import parse_isl_set
import dace.serialize
from functools import reduce
from dace.subsets import _expr, _tuple_to_symexpr


def create_access_map(
        ctx: isl.Context,
        params: Set[str],
        variables: List[Union[str, symbolic.symbol]],
        constants: Dict[str, sp.Number],
        accesses: List[Union[str, symbolic.symbol]],
        stmt_name: str = None,
        access_name: str = None) -> isl.Map:
    """
    :param ctx: The ISL Context to use
    :param params: The parameters of the map
    :param variables: The variables of the input (statement) dimensions
    :param constants: Dict of constant variables and its values
    :param accesses: The variables of the output (access) dimensions
    :param stmt_name: The name of the statement
    :param access_name: The name of the access array
    :return: created ISL map
    """
    space = isl.Space.create_from_names(ctx=ctx, set=variables, params=params)
    if stmt_name:
        space = space.set_tuple_name(isl.dim_type.set, stmt_name)
    sympy_to_pwaff = SympyToPwAff(space, constants)
    empty = isl.Map.empty(space)
    empty = empty.add_dims(isl.dim_type.in_, len(accesses)).reverse()
    mpa = isl.MultiPwAff.from_pw_multi_aff(isl.PwMultiAff.from_map(empty))
    for i, idx_expr in enumerate(accesses):
        idx_pwaff = sympy_to_pwaff.visit(idx_expr)
        mpa = mpa.set_pw_aff(i, idx_pwaff)
    access_map = isl.Map.from_multi_pw_aff(mpa)
    if access_name:
        access_map = access_map.set_tuple_name(isl.dim_type.set, access_name)
    return access_map


def create_constrained_set(
        ctx: isl.Context,
        params: List[str],
        constants: Dict[str, sp.Number],
        constr_ranges: List[Tuple[str, Range]],
        constraints: List[dtypes.typeclass] = None,
        set_name: str = None,
        extra_var: List[str] = None) -> isl.Set:
    """
    :param ctx: The ISL Context to use
    :param params: The parameters of the set
    :param constants: Dict of constant variables and its values
    :param constr_ranges: List of tuples (variable,  'Ranges')
    :param constraints: List of 'Sympy' constraints
    :param constraints: List of 'Sympy' constraints
    :param set_name: The name of the set
    :param extra_var: a set dimension that is not in constr_ranges
    :return: created ISL set
    """
    if not constraints:
        constraints = []
    loop_vars = [v for v, r in constr_ranges]
    params = [p for p in params if p not in loop_vars]
    if extra_var:
        for var in extra_var:
            if var not in loop_vars:
                loop_vars.append(var)
    space = isl.Space.create_from_names(ctx=ctx, set=loop_vars, params=params)
    pw_affs = isl.affs_from_space(space)
    domain = isl.Set.universe(space)
    for p, rng in constr_ranges:
        if isinstance(rng, Subset):
            start, end, step = rng.ranges[0]
        elif isinstance(rng, tuple) and len(rng) == 3:
            start, end, step = rng
        else:
            raise NotImplementedError
        lb = SympyToPwAff(space, constants=constants).visit(start)
        ub = SympyToPwAff(space, constants=constants).visit(end)
        assert (isinstance(step, int) or (step.is_Integer and step.is_Integer))
        # start <= loop_dim <= end
        loop_condition = lb.le_set(pw_affs[p]) & pw_affs[p].le_set(ub)
        if step > 1 or step < -1:
            # create the stride condition
            init = lb.copy()
            dim = init.dim(isl.dim_type.in_)
            # add a temporary dimension for the stride
            init = isl.PwAff.add_dims(init, isl.dim_type.in_, 1)
            stride_space = init.get_domain_space()
            loop_dim = isl.affs_from_space(stride_space)[p]
            ls = isl.LocalSpace.from_space(stride_space)
            stride_dim = isl.PwAff.var_on_domain(ls, isl.dim_type.set, dim)
            scaled_stride_dim = stride_dim.scale_val(isl.Val(int(step)))
            # loop_dim = start + step * stride_dim
            stride = loop_dim.eq_set(init + scaled_stride_dim)
            # stride_dim >= 0
            stride = stride.lower_bound_val(isl.dim_type.set, dim, 0)
            # (start + loop_dim) mod step = 0
            stride = stride.project_out(isl.dim_type.set, dim, 1)
            loop_condition = loop_condition & stride
        domain = domain & loop_condition
    # add extra constrains (e.g. from if/else) to domain
    for cond_expr in constraints:
        condition = SympyToPwAff(space, constants=constants).visit(cond_expr)
        domain = domain & condition
    if set_name:
        domain = domain.set_tuple_name(set_name)
    domain = domain.coalesce()
    return domain


def add_loop_dim(schedule: isl.Schedule, n: int) -> isl.Schedule:
    """
    Add a the nth domain dimension to the time-stamp
    :param schedule: The ISL Schedule
    :param n: the number of the dimension
    :return: modified schedule
    """
    sched_dom = schedule.get_domain()
    loop_dim = isl.UnionPwMultiAff.empty(sched_dom.get_space())
    dom_list = [
        sched_dom.get_set_list().get_at(i)
        for i in range(0, sched_dom.get_set_list().n_set())
    ]

    # map from all domains to its nth parameter. e.g. {A[i,j] -> [j]} for n=2
    for dom in dom_list:
        dim = dom.dim(isl.dim_type.set)
        pma = isl.PwMultiAff.project_out_map(dom.get_space(), isl.dim_type.set,
                                             n, dim - n)
        if n > 1:
            pma = pma.drop_dims(isl.dim_type.out, 0, n - 1)
        loop_dim = loop_dim.add_pw_multi_aff(pma)
    loop_sched = isl.MultiUnionPwAff.from_union_pw_multi_aff(loop_dim)
    schedule = schedule.insert_partial_schedule(loop_sched)
    return schedule


def get_set_dict(uset: isl.UnionSet) -> Dict[str, isl.Set]:
    """
    :param uset: The ISL UnionSet
    :return: a dict from set name to isl.Set
    """
    set_dict = dict()
    set_list = uset.get_set_list()
    for i in range(set_list.n_set()):
        s = set_list.get_at(i)
        name = s.get_tuple_name()
        set_dict[name] = s
    return set_dict


class PolyhedralRepresentation:
    """
    The polyhedral representation P = (domain, read, write, schedule)
    """
    def __init__(self):
        self.domain = None
        self.read = None
        self.write = None
        self.schedule = None

    def _add_other(self, other):
        if self.read and other.read:
            self.read = self.read.union(other.read)
        elif not self.read:
            self.read = other.read
        if self.write and other.write:
            self.write = self.write.union(other.write)
        elif not self.write:
            self.write = other.write
        if self.domain and other.domain:
            self.domain = self.domain.union(other.domain)
        elif not self.domain:
            self.domain = other.domain

    def union(self, other):
        """
        The union of the polyhedral representation of `self` and `other`
        """
        if self.schedule and other.schedule:
            # combines two schedules in an arbitrary order
            self.schedule = self.schedule.set(other.schedule)
        elif not self.schedule:
            self.schedule = other.schedule
        self._add_other(other)

    def sequence(self, other):
        """
        The sequence of the polyhedral representation of `self` and `other`
        """
        if self.schedule and other.schedule:
            # combines two schedules in the given order (self before other)
            self.schedule = self.schedule.sequence(other.schedule)
        elif not self.schedule:
            self.schedule = other.schedule
        self._add_other(other)


def isl_set_to_ranges(isl_set: isl.Set):
    """
    convert a ISL set into Ranges
    :param uset: The ISL Set
    :return loop_ranges: A list of Tuples (variable, Range)
    :return repl_dict: A replacement dictionary for the variables
    """
    ctx = isl_set.get_ctx()
    set_var_names = isl_set.get_var_names(isl.dim_type.set)

    # build a schedule for n_dim dimensions
    schedule = isl.Schedule.from_domain(isl_set)
    for i in range(isl_set.n_dim()):
        schedule = add_loop_dim(schedule, i + 1)

    def ast_build_options(node):
        # separate | atomic | unroll | default
        if node.get_type() != isl.schedule_node_type.band:
            return node
        # a domain space only appears in a single loop at the specified level
        option = isl.UnionSet.read_from_str(node.get_ctx(),
                                            "{ atomic[x] }")
        node = node.band_set_ast_build_options(option)
        return node

    build = isl.AstBuild.alloc(ctx)
    length = isl_set.n_dim()
    iterators = isl.IdList.alloc(ctx, length)
    for i in range(length):
        if isl_set.has_dim_name(isl.dim_type.set, i):
            name = isl_set.get_dim_name(isl.dim_type.set, i)
        else:
            name = "i{}".format(i)
        id = isl.Id.alloc(ctx, name, None)
        iterators = iterators.add(id)
    build = build.set_iterators(iterators)

    root = schedule.get_root()
    root = root.map_descendant_bottom_up(ast_build_options)
    schedule = root.get_schedule()
    ast = build.node_from_schedule(schedule)

    loop_ranges = []

    def print_for(pr, opts, ast_node):
        iter_sympy = to_sympy(ast_node.for_get_iterator())
        init_sympy = to_sympy(ast_node.for_get_init())
        cond_sympy = to_sympy(ast_node.for_get_cond())
        end_sympy = extract_end_cond(cond_sympy, iter_sympy)
        step_sym = to_sympy(ast_node.for_get_inc())

        loop_rng = Range([(init_sympy, end_sympy, step_sym)])
        loop_ranges.append((iter_sympy, loop_rng))
        return ast_node.for_print(pr, opts)

    repl_dict = dict()

    def print_user(pr, opts, node):
        ast_expr = node.user_get_expr()
        for i, var in enumerate(set_var_names):
            old_sym = pystr_to_symbolic(var)
            new_var = ast_expr.get_op_arg(i + 1).to_C_str()
            new_sym = pystr_to_symbolic(new_var)
            repl_dict[old_sym] = new_sym
        return pr.start_line().print_ast_node(node).end_line()

    # AST build options
    ctx.set_ast_build_atomic_upper_bound(True)
    ctx.set_ast_build_group_coscheduled(True)
    ctx.set_ast_build_detect_min_max(True)
    ctx.set_ast_build_prefer_pdiv(True)

    options = isl.AstPrintOptions.alloc(ctx)
    options, _ = options.set_print_for(print_for)
    options, _ = options.set_print_user(print_user)

    printer = isl.Printer.to_str(ctx)
    printer = printer.set_output_format(isl.format.C)
    printer = ast.print_(printer, options)

    # print("Domain:", isl_set, "\n")
    # print("Schedule: ", schedule, "\n")
    # print("Code:\n", printer.get_str(), "\n")

    return loop_ranges, repl_dict


class PolyhedralBuilder:
    """
    Functionality to extract the polyhedral representation from a SDFG.
    """
    def __init__(self, sdfg):
        self._duplicate_name_cnt = defaultdict(int)
        self._name_table = dict()
        self.tasklets = dict()
        self.sdfg = copy.deepcopy(sdfg)
        self.ctx = isl.Context()
        self.param_vars = self.sdfg.free_symbols.copy()
        self.constants = {}
        self.poly_model = None
        self.transformed_poly_model = None
        self.original_isl_ast = None
        self.new_isl_ast = None
        self.deps = None

        for k, v in sdfg.constants.items():
            self.constants[k] = pystr_to_symbolic(str(v))

    def _get_stmt_name(self, *args):
        name = "_".join(args)
        dup_cnt = self._duplicate_name_cnt[name]
        self._duplicate_name_cnt[name] += 1
        if dup_cnt:
            name += "_{}".format(dup_cnt)
        return name

    def traverse(self,
                 node: cflow.ControlFlow,
                 constraints: List[dtypes.typeclass],
                 loop_ranges: List[Tuple[str, Range]],
                 replace_vars: Dict[dtypes.typeclass, dtypes.typeclass]):
        """
        Extract a partial polyhedral representation at each node starting from
        the leaves of the control flow tree and traversing bottom-up.
        A partial polyhedral representation then contains the polyhedral
        representation of a node and its subtree.

        :param node: a control flow node to traverse
        :param constraints: list of constraints that have to hold in the subtree
        :param loop_ranges: the ranges on the collected iteration variables
        :param replace_vars: replacement an expression with another one
        :return seq_poly: return a (partial) polyhedral representation
        """

        if isinstance(node, cflow.GeneralBlock):
            # visit child and make sequence
            seq_poly = PolyhedralRepresentation()
            for child in node.children:
                next_poly = self.traverse(node=child,
                                          constraints=constraints.copy(),
                                          loop_ranges=loop_ranges.copy(),
                                          replace_vars=replace_vars.copy())
                seq_poly.sequence(next_poly)
            return seq_poly

        elif isinstance(node, cflow.SingleState):
            state = node.state
            for orig_sym, transformed_sym in replace_vars.items():
                state.replace(str(orig_sym), str(transformed_sym))
            code_nodes = [
                n for n in state.nodes() if isinstance(n, nodes.CodeNode)
            ]

            loop_vars = [v for v, r in loop_ranges]

            node_poly = PolyhedralRepresentation()
            for cn in code_nodes:
                stmt_name = self._get_stmt_name(state.label, cn.label)
                cn.label = stmt_name

                stmt_poly = PolyhedralRepresentation()
                stmt_domain = create_constrained_set(ctx=self.ctx,
                                                     params=self.param_vars,
                                                     constants=self.constants,
                                                     constr_ranges=loop_ranges,
                                                     constraints=constraints,
                                                     set_name=stmt_name)
                stmt_poly.domain = isl.UnionSet.from_set(stmt_domain)
                stmt_poly.schedule = isl.Schedule.from_domain(stmt_domain)
                stmt_poly.write = isl.UnionMap.empty(stmt_domain.get_space())
                stmt_poly.read = isl.UnionMap.empty(stmt_domain.get_space())

                out_data = dict()
                read_set, write_set = state.read_and_write_sets()
                for e in state.out_edges(cn):
                    if e.data.wcr:
                        continue
                    access_name = e.data.data
                    access_str = e.data.subset.string_list()
                    access_sympy = [pystr_to_symbolic(a) for a in access_str]
                    if access_name in write_set:
                        write = create_access_map(ctx=self.ctx,
                                                  params=self.param_vars,
                                                  variables=loop_vars,
                                                  constants=self.constants,
                                                  accesses=access_sympy,
                                                  stmt_name=stmt_name,
                                                  access_name=access_name)
                        stmt_poly.write = stmt_poly.write.union(write)
                        out_data[e.src_conn] = e.data

                in_data = dict()
                for e in state.in_edges(cn):
                    if e.data.wcr:
                        continue
                    access_name = e.data.data
                    access_str = e.data.subset.string_list()
                    access_sympy = [pystr_to_symbolic(a) for a in access_str]
                    if access_name in read_set:
                        read = create_access_map(ctx=self.ctx,
                                                 params=self.param_vars,
                                                 variables=loop_vars,
                                                 constants=self.constants,
                                                 accesses=access_sympy,
                                                 stmt_name=stmt_name,
                                                 access_name=access_name)
                        stmt_poly.read = stmt_poly.read.union(read)
                        in_data[e.dst_conn] = e.data

                self.tasklets[stmt_name] = (cn, loop_vars, in_data, out_data)
                node_poly.sequence(stmt_poly)

            return node_poly

        elif isinstance(node, cflow.ForScope):
            itersym = pystr_to_symbolic(node.itervar)
            start = pystr_to_symbolic(node.init)
            step_cond = pystr_to_symbolic(node.update)
            step = extract_step_cond(step_cond, itersym)
            end_cond = pystr_to_symbolic(node.condition.as_string)
            end = extract_end_cond(end_cond, itersym)
            # Normalize the loop: start <= end and step > 0
            for orig_sym, transformed_sym in replace_vars.items():
                start = start.replace(orig_sym, transformed_sym)
                end = end.replace(orig_sym, transformed_sym)
            if step < 0:
                step = -step
                start, end = end, start
                shift = start
                start -= shift
                end -= shift
                itersym_transformed = (end - itersym) + shift
                replace_vars[itersym] = itersym_transformed
            loop_ranges.append((node.itervar, Range([(start, end, step)])))
            loop_poly = self.traverse(node=node.body,
                                      constraints=constraints,
                                      loop_ranges=loop_ranges,
                                      replace_vars=replace_vars)
            depth = len(loop_ranges)
            loop_poly.schedule = add_loop_dim(loop_poly.schedule, depth)
            return loop_poly

        elif isinstance(node, cflow.IfScope):
            if_condition = pystr_to_symbolic(node.condition.as_string)
            for orig_sym, transformed_sym in replace_vars.items():
                if_condition = if_condition.replace(orig_sym, transformed_sym)
            if_constraints = constraints.copy()
            if_constraints.append(if_condition)
            if_poly = self.traverse(node=node.body,
                                    constraints=if_constraints,
                                    loop_ranges=loop_ranges.copy(),
                                    replace_vars=replace_vars.copy())
            if node.orelse:
                else_condition = negate_expr(if_condition)
                else_constraints = constraints.copy()
                else_constraints.append(else_condition)
                else_poly = self.traverse(node=node.orelse,
                                          constraints=else_constraints,
                                          loop_ranges=loop_ranges.copy(),
                                          replace_vars=replace_vars.copy())
                if_poly.union(else_poly)
            return if_poly

        elif isinstance(node, cflow.IfElseChain):
            raise NotImplementedError

        elif isinstance(node, cflow.WhileScope):
            raise NotImplementedError

        elif isinstance(node, cflow.DoWhileScope):
            raise NotImplementedError

        elif isinstance(node, cflow.SwitchCaseScope):
            raise NotImplementedError

    @staticmethod
    def code_from_ast(ast):
        """
        Print c-like code for an ISL AST extracted from the polyhedral
        representation. Mark if loops can be parallelized.
        """

        def print_for(pr, opts, node):
            info = node.get_annotation().user
            parallel_info = "[is_parallel: {}]".format(info.is_parallel)
            pr = pr.start_line().print_str(parallel_info).end_line()
            return node.for_print(pr, opts)

        options = isl.AstPrintOptions.alloc(ast.get_ctx())
        options, _ = options.set_print_for(print_for)
        printer = isl.Printer.to_str(ast.get_ctx())
        printer = printer.set_output_format(isl.format.C)
        printer = ast.print_(printer, options)
        return printer.get_str()

    @staticmethod
    def dependency_analysis(domain: isl.UnionSet,
                            write: isl.UnionMap,
                            read: isl.UnionMap,
                            init_schedule: isl.UnionMap,
                            exact_dependency_analysis: bool = True,
                            print_info: bool = True) -> isl.UnionMap:
        """
        Uses ISL to compute the dependencies between the polyhedral statements
        of a polyhedral representation.

        :param domain: The iteration domain of polyhedral representation
        :param write: Maps polyhedral statements to their write accesses
        :param read: Maps polyhedral statements to their read accesses
        :param init_schedule: Initial schedule of the polyhedral representation
        :param exact_dependency_analysis: If True perform exact value-based
            dependency analysis, if False use approximating memory-based
            dependency analysis
        :param print_info: Print the results
        :return dependencies: The computed dependencies between the polyhedral
            statements
        """

        if write:
            write = write.intersect_domain(domain)
        else:
            write = isl.UnionMap.empty(domain.get_space())
        if read:
            read = read.intersect_domain(domain)
        else:
            read = isl.UnionMap.empty(domain.get_space())
        init_schedule = init_schedule.intersect_domain(domain)
        empty = isl.UnionMap.empty(domain.get_space())

        # ISL Dataflow Analysis
        # For details see: "Presburger Formulas and polyhedral Compilation"

        # value-based exact dependencies without transitive dependencies:
        # a read statement depends on the last statement that performed a
        # write to the same data element
        if exact_dependency_analysis:
            # RAW dependencies from the last write to a read
            (_, raw, _, _, _) = read.compute_flow(write, empty,
                                                  init_schedule)
            # WAW dependencies from the last write to a write
            (_, waw, _, _, _) = write.compute_flow(write, empty,
                                                   init_schedule)

            # WAR dependencies from the last read to a write
            flow_info = isl.UnionAccessInfo.from_sink(write)
            flow_info = flow_info.set_may_source(read)
            flow_info = flow_info.set_kill(write)
            flow_info = flow_info.set_schedule_map(init_schedule)
            flow = flow_info.compute_flow()
            war = flow.get_may_dependence()
        else:
            # memory-based transitive dependencies (over-approximation):
            # a statement depends on any previous statement that accesses the
            # same data element and at least one of the accesses is a write

            # RAW dependencies from any previous write to a read
            (_, _, raw, _, _) = read.compute_flow(empty, write, init_schedule)
            raw = raw.range_factor_domain()

            # WAR dependencies from any previous read to a write
            (_, _, war, _, _) = write.compute_flow(empty, read, init_schedule)
            war = war.range_factor_domain()

            # WAW dependencies from any previous write to a write
            (_, _, waw, _, _) = write.compute_flow(empty, write, init_schedule)
            waw = waw.range_factor_domain()

        # coalescing: replace pairs of disjunct sets by single disjunct sets
        # without changing its meaning
        raw = raw.coalesce()
        war = war.coalesce()
        waw = waw.coalesce()

        dependencies = waw.union(war).union(raw)
        dependencies = dependencies.coalesce()

        # simplify the relation representation by detecting implicit equalities
        dependencies = dependencies.detect_equalities()

        if print_info:
            print("Domain:", domain, "\n")
            print("Read: ", read, "\n")
            print("Write: ", write, "\n")
            print("WAR: ", war, "\n")
            print("WAW: ", waw, "\n")
            print("RAW: ", raw, "\n")
            print("Dependency: ", dependencies, "\n")
            print("Init Schedule: ", init_schedule, "\n")

        return dependencies

    @staticmethod
    def optimize_schedule(domain: isl.UnionSet,
                          init_schedule: isl.UnionMap,
                          dependencies: isl.UnionMap,
                          params_constraints: isl.Set,
                          use_pluto: bool = True,
                          tile_size: int = 0,
                          print_info: bool = True):
        """
        Computes an optimized schedule using a PLUTO-like Scheduler
        or a FEAUTRIER Scheduler. The new schedule is optimized for
        data-locality and parallelism.

        :param domain: The iteration domain of polyhedral representation
        :param init_schedule: Initial schedule of the polyhedral representation
        :param dependencies: The dependencies between the polyhedral statements
        :param params_constraints: (optional) constraints on the parameters
        :param use_pluto: If True use PLUTO scheduler, else FEAUTRIER
        :param print_info: Print the results
        :return new_schedule: The new optimized schedule
        :return new_ast_root: The AST of the new polyhedral representation
        :return init_ast_root: The AST of the init polyhedral representation
        """

        init_schedule = init_schedule.intersect_domain(domain)

        # The generated schedule respects all validity dependencies. That is,
        # all dependence distances over these dependencies in the scheduled
        # space are lexicographically positive. Mapping domain elements i to
        # domain elements that should schedule after i
        validity = dependencies.copy()

        # coincidence dependencies: mapping domain elements i to domain elements
        # that should be scheduled together with I, if possible
        coincidence = dependencies.copy()

        # minimize the dependence distances over proximity dependencies
        # mapping domain elements i to domain elements that should be
        # scheduled either before I or as early as possible after i
        proximity = dependencies.copy()

        ctx = domain.get_ctx()
        # default: ISL (PLUTO-like Scheduler), alternative: FEAUTRIER
        if use_pluto:
            ctx.set_schedule_algorithm(isl.schedule_algorithm.ISL)
        else:
            ctx.set_schedule_algorithm(isl.schedule_algorithm.FEAUTRIER)

        # Scheduler Options:
        ctx.set_schedule_serialize_sccs(False)
        ctx.set_schedule_maximize_band_depth(True)
        ctx.set_schedule_outer_coincidence(True)
        ctx.set_schedule_maximize_coincidence(True)
        ctx.set_schedule_whole_component(False)
        ctx.set_tile_scale_tile_loops(False)

        # AST Build Options:
        ctx.set_ast_build_atomic_upper_bound(True)
        ctx.set_ast_build_detect_min_max(True)
        ctx.set_ast_build_prefer_pdiv(True)

        sc = isl.ScheduleConstraints.on_domain(domain.copy())

        # constraints on parameters to hold during construction of the schedule
        sc.set_context(params_constraints)

        try:
            # Try to simplify validity, coincidence and proximity
            # gist: simplify domain or range with respect to known constraints
            validity = validity.gist_domain(domain)
            validity = validity.gist_range(domain)
            validity = validity.coalesce()
            sc = sc.set_validity(validity)
            coincidence = coincidence.gist_domain(domain)
            coincidence = coincidence.gist_range(domain)
            coincidence = coincidence.coalesce()
            sc = sc.set_coincidence(coincidence)
            proximity = proximity.gist_domain(domain)
            proximity = proximity.gist_range(domain)
            proximity = proximity.coalesce()
            sc = sc.set_proximity(proximity)
            new_schedule = sc.compute_schedule()
        except:
            # simplification failed: continue without simplification
            sc = sc.set_validity(dependencies.copy())
            sc = sc.set_coincidence(dependencies.copy())
            sc = sc.set_proximity(dependencies.copy())
            new_schedule = sc.compute_schedule()

        # The generated schedule represents a schedule tree: turn into map
        schedule_map = new_schedule.get_map()
        schedule_map = schedule_map.intersect_domain(domain)

        init_ast_root = BuildFns.get_ast_from_schedule_map(dependencies,
                                                           init_schedule)
        init_code = PolyhedralBuilder.code_from_ast(init_ast_root)

        new_ast_root = BuildFns.get_ast_from_schedule(dependencies,
                                                      new_schedule,
                                                      tile_size=tile_size)
        new_code = PolyhedralBuilder.code_from_ast(new_ast_root)

        if print_info:
            print("Context:", params_constraints, "\n")
            print("Domain:", domain, "\n")
            print("Init Schedule: ", init_schedule, "\n")
            print("New Schedule: ", schedule_map, "\n")
            print("Init Code:", init_code, "\n")
            print("New Code:", new_code, "\n")

        return new_schedule, new_ast_root, init_ast_root

    def get_polyhedral_representation(self) -> \
            Tuple[isl.UnionSet, isl.UnionMap, isl.UnionMap, isl.UnionMap]:
        """
        Returns the Polyhderal Representation:
        P = (domain, read, write, schedule)
        """
        if self.poly_model:
            rep = self.poly_model
        else:
            cft = cflow.structured_control_flow_tree(
                self.sdfg, lambda _: "")
            rep = self.traverse(cft, [], [], {})
            self.poly_model = rep
        return rep.domain, rep.read, rep.write, rep.schedule

    def transform(self,
                  exact_dependency_analysis: bool = True,
                  use_pluto: bool = True,
                  tile_size: int = 0) -> PolyhedralRepresentation:
        """
        Computes the dependencies of the polyhedral representation.
        Uses the dependencies to optimize the polyhedral representation
        using a schedule optimizer.

        :param exact_dependency_analysis: True: compute exact dependencies
        :param use_pluto: Use Pluto optimizer, else use Feautrier optimizer
        :param tile_size: If zero: do not tile, else tile all with tile_size
        :return: the optimized polyhedral representation
        """
        if self.transformed_poly_model:
            return self.transformed_poly_model
        else:
            poly_rep = self.get_polyhedral_representation()
            domain, read, write, schedule_tree = poly_rep
            schedule = schedule_tree.get_map()
            parameter_constraints = isl.Set.empty(domain.get_space())

            dependencies = self.dependency_analysis(
                domain=domain,
                write=write,
                read=read,
                init_schedule=schedule,
                exact_dependency_analysis=exact_dependency_analysis)

            poly_res = self.optimize_schedule(
                domain=domain,
                init_schedule=schedule,
                dependencies=dependencies,
                params_constraints=parameter_constraints,
                use_pluto=use_pluto,
                tile_size=tile_size)

            new_schedule, new_ast, orig_ast = poly_res
            new_model = PolyhedralRepresentation()
            new_model.domain = domain
            new_model.read = read
            new_model.write = write
            new_model.schedule = new_schedule
            self.deps = dependencies
            self.transformed_poly_model = new_model
            self.original_isl_ast = orig_ast
            self.new_isl_ast = new_ast
            return self.transformed_poly_model

    def get_original_isl_ast(self):
        if not self.original_isl_ast:
            self.transform()
        return self.original_isl_ast

    def get_new_isl_ast(self):
        if not self.new_isl_ast:
            self.transform()
        return self.new_isl_ast

    def rebuild_original_sdfg(self,
                              sdfg: SDFG,
                              input_arrays: Set,
                              output_arrays: Set,
                              parallelize_loops: bool,
                              use_polytope: bool):
        """
        Rebuild an SDFG from the original polyhedral representation

        :param sdfg: Rebuild inplace overwriting this SDFG
        :param input_arrays: The input array of the SDFG
        :param output_arrays: The output array of the SDFG
        :param parallelize_loops: Rebuild parallel loops as maps
        :param use_polytope: Rebuild SDFG using Polytopes as subsets
        :return: the rebuilt sdfg
        """
        _, new_poly_rep = IslAstVisitor.parse_isl_ast(
            sdfg=sdfg,
            ast=self.get_original_isl_ast(),
            poly_builder=self,
            input_arrays=input_arrays,
            output_arrays=output_arrays,
            parallelize=parallelize_loops,
            use_polytope=use_polytope)
        return sdfg

    def rebuild_optimized_sdfg(self,
                               sdfg: SDFG,
                               input_arrays: Set,
                               output_arrays: Set,
                               parallelize_loops: bool,
                               use_polytope: bool):
        """
        Rebuild an SDFG from the optimized polyhedral representation

        :param sdfg: Rebuild inplace overwriting this SDFG
        :param input_arrays: The input array of the SDFG
        :param output_arrays: The output array of the SDFG
        :param parallelize_loops: Rebuild parallel loops as maps
        :param use_polytope: Rebuild SDFG using Polytopes as subsets
        :return: the rebuilt sdfg
        """
        _, new_poly_rep = IslAstVisitor.parse_isl_ast(
            sdfg=sdfg,
            ast=self.get_new_isl_ast(),
            poly_builder=self,
            input_arrays=input_arrays,
            output_arrays=output_arrays,
            parallelize=parallelize_loops,
            use_polytope=use_polytope)
        return sdfg


class IslAstVisitor:
    """
    Convert a synthetic ISL AST to an SDFG
    """
    def __init__(self, sdfg, poly_builder, input_arrays, output_arrays,
                 parallelize, use_polytope):
        self.sdfg = sdfg
        self.poly_builder = poly_builder
        self.inputs = {}
        self.outputs = {}
        self.input_arrays = input_arrays
        self.output_arrays = output_arrays
        self.ctx = poly_builder.ctx
        self.parallelize = parallelize
        self.use_polytope = use_polytope

    @staticmethod
    def parse_isl_ast(sdfg: SDFG,
                      ast: isl.AstNode,
                      poly_builder: PolyhedralBuilder,
                      input_arrays: Set[dtypes.typeclass],
                      output_arrays: Set[dtypes.typeclass],
                      parallelize: bool = True,
                      use_polytope: bool = False):
        """
        Generates a new SDFG from the ISL AST

        :param sdfg: use this SDFG to generate inplace
        :param ast: The root of the ISL AST
        :param poly_builder A PolyhedralBuilder with polyhedral representation
        :param input_arrays: The input arrays of the SDFG
        :param output_arrays: The output arrays of the SDFG
        :param parallelize: if True: turn parallel loops into maps
        :param use_polytope: if True: use Polytope subsets, else Range subsets
        :return sdfg: The generated SDFG
        :return poly: The polyhedral representation of the SDFG
        """
        pv = IslAstVisitor(sdfg=sdfg,
                           poly_builder=poly_builder,
                           input_arrays=input_arrays,
                           output_arrays=output_arrays,
                           parallelize=parallelize,
                           use_polytope=use_polytope)

        init = sdfg.add_state("init", is_start_state=True)
        first, last, poly = pv._visit(ast, [], [])
        sdfg.add_edge(init, first, sd.InterstateEdge())
        final = sdfg.add_state("end_state", is_start_state=False)
        sdfg.add_edge(last, final, sd.InterstateEdge())
        return sdfg, poly

    def _visit(self, ast_node, loop_ranges, constraints):
        """
        Visit a AST node.
        """
        if ast_node.get_type() == isl.ast_node_type.block:
            first, last, poly = self._visit_block(ast_node, loop_ranges,
                                                  constraints)
        elif ast_node.get_type() == isl.ast_node_type.for_:
            first, last, poly = self._visit_for(ast_node, loop_ranges,
                                                constraints)
        elif ast_node.get_type() == isl.ast_node_type.if_:
            first, last, poly = self._visit_if(ast_node, loop_ranges,
                                               constraints)
        elif ast_node.get_type() == isl.ast_node_type.user:
            first, last, poly = self._visit_user(ast_node, loop_ranges,
                                                 constraints)
        else:
            raise NotImplementedError
        return first, last, poly

    def _visit_block(self, ast_node, loop_ranges, constraints):
        """
        Visit a AST block node.
        """
        node_list = ast_node.block_get_children()
        n_children = node_list.n_ast_node()
        states = []
        seq_poly = PolyhedralRepresentation()
        for child_node in [node_list.get_at(i) for i in range(n_children)]:
            ret_val = self._visit(child_node, loop_ranges.copy(), constraints)
            s1, s2, next_poly = ret_val

            seq_poly.sequence(next_poly)
            states.append((s1, s2))
        for (_, s1), (s2, _) in zip(states[:-1], states[1:]):
            self.sdfg.add_edge(s1, s2, sd.InterstateEdge())
        return states[0][0], states[-1][1], seq_poly

    def _visit_for(self, ast_node, loop_ranges, constraints):
        """
        Visit a AST for node.
        """

        iter_sympy = to_sympy(ast_node.for_get_iterator())
        iterator_var = sympy_to_pystr(iter_sympy)

        init_sympy = to_sympy(ast_node.for_get_init())
        init_str = sympy_to_pystr(init_sympy)

        cond_sympy = to_sympy(ast_node.for_get_cond())
        end_sympy = extract_end_cond(cond_sympy, iter_sympy)
        condition_str = sympy_to_pystr(cond_sympy)

        step_sym = to_sympy(ast_node.for_get_inc())
        incr_str = sympy_to_pystr(sp.Add(iter_sympy, step_sym))

        loop_rng = subsets.Range([(init_sympy, end_sympy, step_sym)])
        loop_ranges.append((iterator_var, loop_rng))

        stmt_domain = create_constrained_set(
            ctx=self.ctx,
            params=self.poly_builder.param_vars,
            constants=self.poly_builder.constants,
            constr_ranges=loop_ranges,
            constraints=constraints)
        stmt_domain = stmt_domain.coalesce()

        while stmt_domain.n_dim() > 1:
            stmt_domain = stmt_domain.move_dims(isl.dim_type.param, 0,
                                                isl.dim_type.set, 0, 1)

        is_parallel = False
        if self.parallelize:
            build = ast_node.get_annotation().user.build
            part_schedule = build.get_schedule()
            deps = self.poly_builder.deps
            is_parallel = BuildFns.is_parallel(part_schedule, deps)
            # is_parallel = ast_node.get_annotation().user.is_parallel
        if is_parallel:
            state = self.sdfg.add_state('MapState')
            if self.use_polytope:
                subset = Polytope([(init_sympy, end_sympy, step_sym)],
                                            [iterator_var])
                # subset = Polytope.from_isl_set(stmt_domain)
                pass
            else:
                subset = loop_rng

            map_nodes = nodes.Map(label='map',
                                  params=[iterator_var],
                                  ndrange=subset)

            entry = nodes.MapEntry(map_nodes)
            exit = nodes.MapExit(map_nodes)
            state.add_nodes_from([entry, exit])

            # create a new SDFG for the map body
            body_sdfg = SDFG("{}_body".format(entry.label))

            # add all arrays of SDFG to the body-SDFG
            # all transient become False
            for arr_label, arr in self.sdfg.arrays.items():
                arr_copy = copy.deepcopy(arr)
                if arr_label in self.input_arrays:
                    arr_copy.transient = False
                elif arr_label in self.output_arrays:
                    arr_copy.transient = False
                body_sdfg.add_datadesc(arr_label, arr_copy)

            body_sdfg.symbols.update(self.sdfg.symbols)

            # walk and add the states to the body_sdfg
            pv = IslAstVisitor(sdfg=body_sdfg,
                               poly_builder=self.poly_builder,
                               input_arrays=self.input_arrays,
                               output_arrays=self.output_arrays,
                               parallelize=self.parallelize,
                               use_polytope=self.use_polytope)

            _, _, loop_poly = pv._visit(ast_node.for_get_body(),
                                        loop_ranges.copy(), constraints)
            depth = len(loop_ranges)
            loop_poly.schedule = add_loop_dim(loop_poly.schedule, depth)

            body_inputs = {
                c: m
                for c, m in pv.inputs.items()
                if not body_sdfg.arrays[c].transient
            }
            body_outputs = {
                c: m
                for c, m in pv.outputs.items()
                if not body_sdfg.arrays[c].transient
            }

            used_arrays = {
                s.label
                for s in body_sdfg.input_arrays() + body_sdfg.output_arrays()
            }
            for arr in {a for a in body_sdfg.arrays if a not in used_arrays}:
                body_sdfg.remove_data(arr)

            body = state.add_nested_sdfg(body_sdfg, self.sdfg,
                                         body_inputs.keys(),
                                         body_outputs.keys())

            for arr_name, in_mem in body_inputs.items():
                if arr_name not in body.in_connectors:
                    continue
                read_node = state.add_read(arr_name)
                arr = body_sdfg.arrays[arr_name]
                if self.use_polytope:
                    subset = Polytope.from_array(arr)
                    # TODO: the following gives tighter bound for the memlets
                    # read_range = loop_poly.read.intersect_domain(
                    #     loop_poly.domain).range().coalesce()
                    # set_dict = get_set_dict(read_range)
                    # arr_set = set_dict[arr_name]
                    # subset = Polytope.from_isl_set(arr_set)
                else:
                    subset = Range.from_array(arr)

                memlet = Memlet(data=arr_name, subset=subset)

                state.add_memlet_path(read_node,
                                      entry,
                                      body,
                                      memlet=memlet,
                                      dst_conn=arr_name,
                                      propagate=False)
            if len(body.in_connectors) == 0:
                state.add_edge(entry, None, body, None, Memlet())

            for arr_name, out_mem in body_outputs.items():
                if arr_name not in body.out_connectors:
                    continue
                write_node = state.add_write(arr_name)
                arr = body_sdfg.arrays[arr_name]
                if self.use_polytope:
                    subset = Polytope.from_array(arr)
                    # TODO: the following gives tighter bound for the memlets
                    # write_range = loop_poly.write.intersect_domain(
                    #     loop_poly.domain).range().coalesce()
                    # set_dict = get_set_dict(write_range)
                    # arr_set = set_dict[arr_name]
                    # subset = Polytope.from_isl_set(arr_set)
                else:
                    subset = Range.from_array(arr)

                memlet = Memlet(data=arr_name, subset=subset)

                state.add_memlet_path(body,
                                      exit,
                                      write_node,
                                      memlet=memlet,
                                      src_conn=arr_name,
                                      dst_conn=None,
                                      propagate=False)
            if len(body.out_connectors) == 0:
                state.add_edge(body, None, exit, None, Memlet())

            self.inputs.update(body_inputs)
            self.outputs.update(body_outputs)
            return state, state, loop_poly
        else:
            body_begin, body_end, loop_poly = \
                self._visit(ast_node.for_get_body(), loop_ranges.copy(),
                            constraints)
            depth = len(loop_ranges)
            loop_poly.schedule = add_loop_dim(loop_poly.schedule, depth)

            if iterator_var not in self.sdfg.symbols:
                self.sdfg.add_symbol(iterator_var, dtypes.int64)

            if body_begin == body_end:
                body_end = None

            loop_result = self.sdfg.add_loop(before_state=None,
                                             loop_state=body_begin,
                                             loop_end_state=body_end,
                                             after_state=None,
                                             loop_var=iterator_var,
                                             initialize_expr=init_str,
                                             condition_expr=condition_str,
                                             increment_expr=incr_str)
            before_state, guard, after_state = loop_result
            return before_state, after_state, loop_poly

    def _visit_if(self, ast_node, loop_ranges, constraints):
        """
        Visit an AST if node.
        """
        # Add a guard state
        if_guard = self.sdfg.add_state('if_guard')
        end_if_state = self.sdfg.add_state('end_if')

        # Generate conditions
        if_cond_sym = to_sympy(ast_node.if_get_cond())
        if_cond_str = sympy_to_pystr(if_cond_sym)
        else_cond_sym = negate_expr(if_cond_sym)
        else_cond_str = sympy_to_pystr(else_cond_sym)

        then_node = ast_node.if_get_then_node()
        if_constraints = constraints.copy()
        if_constraints.append(if_cond_sym)
        first_if_state, last_if_state, if_poly = \
            self._visit(then_node, loop_ranges.copy(), if_constraints)

        # Connect the states
        self.sdfg.add_edge(if_guard, first_if_state,
                           sd.InterstateEdge(if_cond_str))
        self.sdfg.add_edge(last_if_state, end_if_state, sd.InterstateEdge())

        if ast_node.if_has_else_node():
            else_node = ast_node.if_get_else_node()
            else_constraints = constraints.copy()
            else_constraints.append(else_cond_sym)
            first_else_state, last_else_state, else_poly = \
                self._visit(else_node, loop_ranges.copy(), else_constraints)
            if_poly.union(else_poly)

            # Connect the states
            self.sdfg.add_edge(if_guard, first_else_state,
                               sd.InterstateEdge(else_cond_str))
            self.sdfg.add_edge(last_else_state, end_if_state,
                               sd.InterstateEdge())
        else:
            self.sdfg.add_edge(if_guard, end_if_state,
                               sd.InterstateEdge(else_cond_str))
        return if_guard, end_if_state, if_poly

    def _visit_user(self, ast_node, loop_ranges, constraints):
        """
        Visit an AST user node.
        """
        ast_expr = ast_node.user_get_expr()
        if ast_expr.get_op_type() == isl.ast_expr_op_type.call:
            stmt_name = ast_expr.get_op_arg(0).to_C_str()
            state = self.sdfg.add_state("state")
            (tasklet, iter_vars, in_data, out_data) = \
                self.poly_builder.tasklets[stmt_name]

            repl_dict = {}
            for i, var in enumerate(iter_vars):
                old_sym = pystr_to_symbolic(var)
                new_var = ast_expr.get_op_arg(i + 1).to_C_str()
                new_sym = pystr_to_symbolic(new_var)
                repl_dict[old_sym] = new_sym

            param_vars = self.poly_builder.param_vars
            constants = self.poly_builder.constants
            loop_vars = [v for v, r in loop_ranges]
            stmt_poly = PolyhedralRepresentation()
            stmt_domain = create_constrained_set(ctx=self.ctx,
                                                 params=param_vars,
                                                 constants=constants,
                                                 constr_ranges=loop_ranges,
                                                 constraints=constraints,
                                                 set_name=stmt_name)

            stmt_poly.domain = isl.UnionSet.from_set(stmt_domain)
            stmt_poly.schedule = isl.Schedule.from_domain(stmt_domain)
            stmt_poly.write = isl.UnionMap.empty(stmt_domain.get_space())
            stmt_poly.read = isl.UnionMap.empty(stmt_domain.get_space())

            state.add_node(tasklet)
            for conn, out_mem in out_data.items():
                arr_name = out_mem.data
                new_subset = subsets.Range(out_mem.subset)
                new_subset.replace(repl_dict)
                access = [
                    pystr_to_symbolic(a) for a in new_subset.string_list()
                ]
                write = create_access_map(ctx=self.ctx,
                                          params=param_vars,
                                          variables=loop_vars,
                                          constants=constants,
                                          accesses=access,
                                          stmt_name=stmt_name,
                                          access_name=arr_name)
                stmt_poly.write = stmt_poly.write.union(write)
                polytope = Polytope(new_subset)
                write_node = state.add_write(arr_name)
                if self.use_polytope:
                    memlet = Memlet(data=arr_name, subset=polytope)
                else:
                    memlet = Memlet(data=arr_name, subset=new_subset)
                state.add_edge(tasklet, conn, write_node, None, memlet)
                self.outputs[arr_name] = memlet
            for conn, in_mem in in_data.items():
                arr_name = in_mem.data
                new_subset = subsets.Range(in_mem.subset)
                new_subset.replace(repl_dict)
                access = [
                    pystr_to_symbolic(a) for a in new_subset.string_list()
                ]
                read = create_access_map(ctx=self.ctx,
                                         params=param_vars,
                                         variables=loop_vars,
                                         constants=constants,
                                         accesses=access,
                                         stmt_name=stmt_name,
                                         access_name=arr_name)
                stmt_poly.read = stmt_poly.read.union(read)
                polytope = Polytope(new_subset)
                read_node = state.add_read(arr_name)
                if self.use_polytope:
                    memlet = Memlet(data=arr_name, subset=polytope)
                else:
                    memlet = Memlet(data=arr_name, subset=new_subset)
                self.inputs[arr_name] = memlet
                state.add_edge(read_node, None, tasklet, conn, memlet)

            return state, state, stmt_poly
        return None, None, None


class BuildFns:
    """
    Helper Class to generate a synthetic AST from the polyhedral representation.
    Functionality for tiling and detecting parallelism in synthetic AST.
    """
    deps = [None]

    class UserInfo:
        def __init__(self):
            # Loops is parallel
            self.is_parallel = False
            self.build = None
            self.schedule = None
            self.domain = None

    @staticmethod
    def at_each_domain(node: isl.AstNode, build: isl.AstBuild):
        """
        Annotated each node in the AST with the domain and partial schedule
        """
        info = BuildFns.UserInfo()
        id = isl.Id.alloc(ctx=isl.AstBuild.get_ctx(build), name="",
                          user=info)
        info.build = isl.AstBuild.copy(build)
        info.schedule = build.get_schedule()
        info.domain = info.schedule.domain()
        node.set_annotation(id)
        return node

    @staticmethod
    def before_each_for(build: isl.AstBuild):
        """
        Detection of parallel loops.
        This function is called for each for in depth-first pre-order.
        """

        # A (partial) schedule for the domains elements for which part of
        # the AST still needs to be generated in the current build.
        # The domain elements are mapped to those iterations of the loops
        # enclosing the current point of the AST generation inside which
        # the domain elements are executed.
        part_sched = build.get_schedule()
        info = BuildFns.UserInfo()

        # Test for parallelism
        info.is_parallel = BuildFns.is_parallel(part_sched,
                                                BuildFns.deps[0])
        info.build = isl.AstBuild.copy(build)
        info.schedule = part_sched
        info.domain = part_sched.domain()

        return isl.Id.alloc(ctx=build.get_ctx(), name="", user=info)

    @staticmethod
    def is_parallel(part_sched: isl.UnionMap, stmt_deps: isl.UnionMap) -> bool:
        """
        Check if the current scheduling dimension is parallel by verifying that
        the loop does not carry any dependencies.

        :param part_sched: A partial schedule
        :param stmt_deps: The dependencies between the statements
        :return True if current the scheduling dimension is parallel, else False
        """

        # translate the dependencies into time-space, by applying part_sched
        time_deps = stmt_deps.apply_range(part_sched).apply_domain(part_sched)

        # the loop is parallel, if there are no dependencies in time-space
        if time_deps.is_empty():
            return True

        time_deps = isl.Map.from_union_map(time_deps)
        time_deps = time_deps.flatten_domain().flatten_range()

        curr_dim = time_deps.dim(isl.dim_type.set) - 1
        # set all dimension in the time-space equal, except the current one:
        # if the distance in all outer dimensions is zero, then it
        # has to be zero in the current dimension as well to be parallel
        for i in range(curr_dim):
            time_deps = time_deps.equate(isl.dim_type.in_, i,
                                         isl.dim_type.out, i)

        # computes a delta set containing the differences between image
        # elements and corresponding domain elements in the time_deps.
        time_deltas = time_deps.deltas()

        # the loop is parallel, if there are no deltas in the time-space
        if time_deltas.is_empty():
            return True

        # The loop is parallel, if the distance is zero in the current dimension
        delta = time_deltas.plain_get_val_if_fixed(isl.dim_type.set, curr_dim)
        return delta.is_zero()

    @staticmethod
    def get_annotation_build(ctx: isl.Context, deps: isl.UnionMap) \
            -> isl.AstBuild:
        """
        helper function that return an isl.AstBuild
        """
        build = isl.AstBuild.alloc(ctx)
        # callback at_each_domain will be called for each domain AST node
        build, _ = build.set_at_each_domain(BuildFns.at_each_domain)
        BuildFns.deps = [deps]
        # callback before_each_for be called in depth-first pre-order
        build, _ = build.set_before_each_for(BuildFns.before_each_for)
        return build

    @staticmethod
    def get_ast_from_schedule(deps: isl.UnionMap,
                              schedule: isl.Schedule,
                              tile_size: int = 0) -> isl.AstNode:
        """
        Compute a synthetic AST from the dependencies and a schedule tree
        :param deps: The dependencies to use
        :param schedule: The schedule to use
        :param tile_size: If tile_size>1 perform tiling on synthetic AST
        :return: The root of the generated synthetic AST
        """

        ctx = schedule.get_ctx()
        build = BuildFns.get_annotation_build(ctx, deps)

        def ast_build_options(node: isl.AstNode) -> isl.AstNode:
            """
            Sets the building options for this node in the AST
            """
            if node.get_type() != isl.schedule_node_type.band:
                return node
            # options: separate | atomic | unroll | default
            option = isl.UnionSet.read_from_str(node.get_ctx(),
                                                "{default[x]}")
            node = node.band_set_ast_build_options(option)
            return node

        def tile_band(node: isl.AstNode) -> isl.AstNode:
            """
            Tile this node of the AST if possible
            """
            # return if not tileable
            if node.get_type() != isl.schedule_node_type.band:
                return node
            elif node.n_children() != 1:
                return node
            elif not node.band_get_permutable():
                return node
            elif node.band_n_member() <= 1:
                return node
            n_dim = node.band_get_space().dim(isl.dim_type.set)
            if n_dim <= 1:
                return node

            tile_multi_size = isl.MultiVal.zero(node.band_get_space())
            for i in range(n_dim):
                tile_multi_size = tile_multi_size.set_val(
                    i, isl.Val(tile_size))

            # tile the current band with tile_size
            node = node.band_tile(tile_multi_size)
            # mark all dimensions in the band node to be "atomic"
            for i in range(node.band_n_member()):
                atomic_type = isl.ast_loop_type.atomic
                node = node.band_member_set_ast_loop_type(i, atomic_type)
            return node

        root = schedule.get_root()
        root = root.map_descendant_bottom_up(ast_build_options)
        if tile_size >= 1:
            root = root.map_descendant_bottom_up(tile_band)
        schedule = root.get_schedule()
        root = build.node_from_schedule(schedule)
        return root

    @staticmethod
    def get_ast_from_schedule_map(deps, schedule_map):
        """
        :param deps: :class:`UnionMap`
        :param schedule_map: :class:`UnionMap`
        :return: :class:`AstBuild`
        """

        ctx = schedule_map.get_ctx()
        ctx.set_ast_build_detect_min_max(True)  # True
        # options that control how AST is created from the individual schedule
        # dimensions are stored in build
        build = BuildFns.get_annotation_build(ctx, deps)
        # generates an AST
        root = build.node_from_schedule_map(schedule_map)
        return root


@dace.serialize.serializable
class Polytope(Subset):
    """ Subset defined as a polytope. """
    def __init__(self, ranges, dim_vars=None):
        self.ranges = ranges
        self.n_dims = len(ranges)
        if dim_vars:
            self.dim_vars = [str(p) for p in dim_vars]
        else:
            self.dim_vars = ["i{}".format(nr) for nr in range(self.n_dims)]

        self.range_tuples = list(zip(self.dim_vars, ranges))

        self.params = set()
        for dim in ranges:
            for d in dim:
                self.params |= symbolic.symlist(d).keys()
        self.params = {r for r in self.params if r not in self.dim_vars}

        parsed_ranges = []
        parsed_tiles = []
        for r in ranges:
            if len(r) != 3 and len(r) != 4:
                raise ValueError("Expected 3-tuple or 4-tuple")
            parsed_ranges.append(
                (_tuple_to_symexpr(r[0]), _tuple_to_symexpr(r[1]),
                 _tuple_to_symexpr(r[2])))
            if len(r) == 3:
                parsed_tiles.append(symbolic.pystr_to_symbolic(1))
            else:
                parsed_tiles.append(symbolic.pystr_to_symbolic(r[3]))

        self.ranges = parsed_ranges
        self.tile_sizes = parsed_tiles

        self.ctx = isl.Context()
        self.isl_set = create_constrained_set(ctx=self.ctx,
                                              params=list(self.params),
                                              constants=dict(),
                                              constr_ranges=self.range_tuples)

        self.isl_set = self.isl_set.coalesce().detect_equalities()
        new_tuple, new_mapping = isl_set_to_ranges(self.isl_set)
        rng_list = []
        if not new_tuple:
            range_dict = {str(k): v for (k, v) in new_mapping.items()}
            for p in self.dim_vars:
                index = range_dict[p]
                rng_list.append((index, index, 1))
        else:
            range_dict = {str(k): v for (k, v) in new_tuple}
            for p in self.dim_vars:
                rng = range_dict[p]
                rng_list.append(rng.ranges[0])

        new_range = Range(rng_list)
        # assert new_range == subsets.Range(self.ranges)

        for i in range(self.n_dims):
            self.isl_set.set_dim_name(isl.dim_type.set, i, self.dim_vars[i])

    @staticmethod
    def from_isl_set(isl_set):
        params = isl_set.get_var_names(isl.dim_type.set)
        if isl_set.is_empty():
            return Range([])
        new_tuple, new_mapping = isl_set_to_ranges(isl_set)
        rng_list = []
        if None in params:
            for i in range(len(params)):
                params[i] = "i{}".format(i)
        if not new_tuple:
            range_dict = {str(k): v for (k, v) in new_mapping.items()}
            for p in params:
                index = range_dict[p]
                rng_list.append((index, index, 1))
        else:
            range_dict = {str(k): v for (k, v) in new_tuple}
            for p in params:
                rng = range_dict[p]
                rng_list.append(rng.ranges[0])
        new_range = Range(rng_list)
        return Polytope(new_range.ranges, params)

    def to_ranges(self):
        new_tuple, new_mapping = isl_set_to_ranges(self.isl_set)
        rng_list = []
        if not new_tuple:
            range_dict = {str(k): v for (k, v) in new_mapping.items()}
            for p in self.dim_vars:
                index = range_dict[p]
                rng_list.append((index, index, 1))
        else:
            range_dict = {str(new_mapping[k]): v for (k, v) in new_tuple}
            for p in self.dim_vars:
                rng = range_dict[p]
                rng_list.append(rng.ranges[0])

        new_range = Range(rng_list)
        return new_range

    def to_json(self):
        ret = []

        for (start, end, step), tile in zip(self.ranges, self.tile_sizes):
            ret.append({
                'start': str(start) + "x",
                'end': str(end),
                'step': str(step),
            })

        domain_constraints = [str(c) for c in self.get_sympy_constraints()]

        return {
            'type': 'Polytope',
            # 'ranges': ret,
            'params': self.dim_vars,
            'domain': domain_constraints
        }

    @staticmethod
    def from_array(array: 'dace.data.Data'):
        """ Constructs a range that covers the full array given as input. """
        return Polytope([(0, s - 1, 1) for s in array.shape])

    def num_elements(self):
        return reduce(sp.Mul, self.size(), 1)

    def size(self, for_codegen=False):
        """
        Returns the number of elements in each dimension.
        """

        if for_codegen == True:
            int_ceil = sp.Function('int_ceil')
            return [
                ts * int_ceil(
                    ((iMax.approx if isinstance(iMax, symbolic.SymExpr) else
                      iMax) + 1 - (iMin.approx if isinstance(
                          iMin, symbolic.SymExpr) else iMin)),
                    (step.approx
                     if isinstance(step, symbolic.SymExpr) else step))
                for (iMin, iMax, step), ts in zip(self.ranges, self.tile_sizes)
            ]
        else:
            return [
                ts * sp.ceiling(
                    ((iMax.approx
                      if isinstance(iMax, symbolic.SymExpr) else iMax) + 1 -
                     (iMin.approx if isinstance(iMin, symbolic.SymExpr) else
                      iMin)) / (step.approx if isinstance(
                          step, symbolic.SymExpr) else step))
                for (iMin, iMax, step), ts in zip(self.ranges, self.tile_sizes)
            ]

    def min_element(self):
        return [_expr(x[0]) for x in self.ranges]

    def max_element(self):
        return [_expr(x[1]) for x in self.ranges]

    def dims(self):
        return len(self.ranges)

    @property
    def free_symbols(self) -> Set[str]:
        return self.params

    def __iter__(self):
        return iter(self.ranges)

    def __len__(self):
        return len(self.ranges)

    def get_isl_str(self):
        return self.isl_set.to_str()

    def get_sympy_constraints(self):
        _, _, _, constr = parse_isl_set(self.isl_set)
        return constr

    def offset(self, other, negative, indices=None):
        return self.to_ranges().offset(other, negative, indices)

    def at(self, i, strides):
        raise NotImplementedError

    def offset_new(self, other, negative, indices=None):
        raise NotImplementedError

    @staticmethod
    def dim_to_string(d, t=1):
        raise NotImplementedError

    @staticmethod
    def ndslice_to_string(slice, tile_sizes=None):
        raise NotImplementedError

    def __str__(self):
        return self.isl_set.to_str()

    @staticmethod
    def _range_pystr(range, param):
        raise NotImplementedError

    def pystr(self):
        raise NotImplementedError

    @staticmethod
    def from_indices(indices: 'Indices'):
        raise NotImplementedError

    @staticmethod
    def from_json(obj, context=None):
        raise NotImplementedError

    def __hash__(self):
        raise NotImplementedError

    def __add__(self, other):
        raise NotImplementedError

    def num_elements_exact(self):
        raise NotImplementedError

    def size_exact(self):
        raise NotImplementedError

    def bounding_box_size(self):
        raise NotImplementedError

    def max_element_approx(self):
        raise NotImplementedError

    def min_element_approx(self):
        raise NotImplementedError

    def coord_at(self, i):
        raise NotImplementedError

    def data_dims(self):
        raise NotImplementedError

    def absolute_strides(self, global_shape):
        raise NotImplementedError

    def strides(self):
        raise NotImplementedError

    def reorder(self, order):
        raise NotImplementedError

    @staticmethod
    def from_string(string):
        raise NotImplementedError

    @staticmethod
    def ndslice_to_string_list(slice, tile_sizes=None):
        raise NotImplementedError

    def ndrange(self):
        raise NotImplementedError

    def __getitem__(self, key):
        return self.ranges.__getitem__(key)

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __eq__(self, other):
        other_new = isl.Set(other.isl_set.to_str(),
                            context=self.isl_set.get_ctx())
        return self.isl_set.is_equal(other_new)

    def __ne__(self, other):
        return not (self == other)

    def compose(self, other):
        raise NotImplementedError

    def squeeze(self, ignore_indices=None):
        raise NotImplementedError

    def unsqueeze(self, axes):
        raise NotImplementedError

    def pop(self, dimensions):
        raise NotImplementedError

    def string_list(self):
        raise NotImplementedError

    def replace(self, repl_dict):
        raise NotImplementedError

    def intersects(self, other: 'Polytope'):
        other_new = isl.Set(other.isl_set.to_str(),
                            context=self.isl_set.get_ctx())
        intersected_set = self.isl_set.intersect(other_new)
        Polytope.from_isl_set(intersected_set)

    def union(self, other: 'Polytope'):
        other_new = isl.Set(other.isl_set.to_str(),
                            context=self.isl_set.get_ctx())
        intersected_set = self.isl_set.union(other_new)
        Polytope.from_isl_set(intersected_set)



