# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Functions for generating C++ code for control flow in SDFGs using control flow regions.
"""

import re
from typing import TYPE_CHECKING, Callable, Dict, Optional, Set
import warnings
from dace import dtypes
from dace.sdfg.analysis import cfg as cfg_analysis
from dace.sdfg.state import (AbstractControlFlowRegion, BreakBlock, ConditionalBlock, ContinueBlock, ControlFlowBlock,
                             ControlFlowRegion, LoopRegion, ReturnBlock, SDFGState, UnstructuredControlFlow)
from dace.sdfg.sdfg import SDFG, InterstateEdge
from dace.sdfg.graph import Edge
from dace.codegen.common import unparse_interstate_edge

if TYPE_CHECKING:
    from dace.codegen.targets.framecode import DaCeCodeGenerator


def _clean_loop_body(body: str) -> str:
    """ Cleans loop body from extraneous continue statements. """
    if body.endswith('continue;\n'):
        body = body[:-len('continue;\n')]
    return body


def _child_of(node: SDFGState, parent: SDFGState, ptree: Dict[SDFGState, SDFGState]) -> bool:
    curnode = node
    while curnode is not None:
        if curnode is parent:
            return True
        curnode = ptree[curnode]
    return False


def _generate_interstate_edge_code(edge: Edge[InterstateEdge],
                                   sdfg: SDFG,
                                   cfg: ControlFlowRegion,
                                   codegen: 'DaCeCodeGenerator',
                                   assignments_only: bool = False,
                                   exit_on_else: bool = False) -> str:
    """
    Generates C++ code for an interstate edge, which may include a condition and assignments.
    :param edge:             The interstate edge to generate code for.
    :param sdfg:             The SDFG containing the edge.
    :param cfg:              The control flow region containing the edge.
    :param codegen:          The code generator object, used for allocation information and defined variables in scope.
    :param assignments_only: If True, only generate code for assignments, without condition or goto.
    :param exit_on_else:     If True, generate an else branch that exits the region if the condition is not met.
    :return:                 C++ string with the generated code for the interstate edge.
    """
    expr = ''
    condition_string = unparse_interstate_edge(edge.data.condition.code[0], sdfg, codegen=codegen)

    if not edge.data.is_unconditional() and not assignments_only:
        expr += f'if ({condition_string}) {{\n'

    if len(edge.data.assignments) > 0:
        expr += ';\n'.join([
            "{} = {}".format(variable, unparse_interstate_edge(value, sdfg, codegen=codegen))
            for variable, value in edge.data.assignments.items()
        ] + [''])

    if not assignments_only:
        dst: ControlFlowBlock = edge.dst
        expr += 'goto __state_{}_{};\n'.format(cfg.cfg_id, re.sub(r'\s+', '_', dst.label))

    if not edge.data.is_unconditional() and not assignments_only:
        if exit_on_else:
            expr += '} else {\n'
            expr += f'goto __state_exit_{cfg.cfg_id};\n'
            expr += '}\n'
        else:
            expr += '}\n'
    return expr


def _loop_region_to_code(region: LoopRegion, dispatch_state: Callable[[SDFGState], str], codegen: 'DaCeCodeGenerator',
                         symbols: Dict[str, dtypes.typeclass]) -> str:
    """
    Converts a LoopRegion to C++ code with the correct control flow expressions.

    :param region:          The LoopRegion to convert.
    :param dispatch_state:  A callback to generate code for a given SDFG state.
    :param codegen:         The code generator object, used for allocation information and defined variables in scope.
    :param symbols:         A dictionary of symbol names and their types.
    :return:                C++ string with the generated code of the loop region.
    """
    sdfg = region.sdfg
    loop = region
    cond = unparse_interstate_edge(loop.loop_condition.code[0], sdfg, codegen=codegen, symbols=symbols)

    expr = ''

    if loop.update_statement and loop.init_statement and loop.loop_variable:
        lsyms = {}
        lsyms.update(symbols)
        if codegen.dispatcher.defined_vars.has(loop.loop_variable) and not loop.loop_variable in lsyms:
            lsyms[loop.loop_variable] = codegen.dispatcher.defined_vars.get(loop.loop_variable)[1]
        init = unparse_interstate_edge(loop.init_statement.code[0], sdfg, codegen=codegen, symbols=lsyms)
        init = init.strip(';')

        update = unparse_interstate_edge(loop.update_statement.code[0], sdfg, codegen=codegen, symbols=lsyms)
        update = update.strip(';')

        if loop.inverted:
            if loop.update_before_condition:
                expr += f'{init};\n'
                expr += 'do {\n'
                expr += _clean_loop_body(control_flow_region_to_code(loop, dispatch_state, codegen, symbols))
                expr += f'{update};\n'
                expr += f'}} while({cond});\n'
            else:
                expr += f'{init};\n'
                expr += 'while (1) {\n'
                expr += _clean_loop_body(control_flow_region_to_code(loop, dispatch_state, codegen, symbols))
                expr += f'if (!({cond}))\n'
                expr += 'break;\n'
                expr += f'{update};\n'
                expr += '}\n'
        else:
            if loop.unroll:
                if loop.unroll_factor >= 1:
                    expr += f'#pragma unroll {loop.unroll_factor}\n'
                else:
                    expr += f'#pragma unroll\n'
            expr += f'for ({init}; {cond}; {update}) {{\n'
            expr += _clean_loop_body(control_flow_region_to_code(loop, dispatch_state, codegen, symbols))
            expr += '\n}\n'
    else:
        if loop.inverted:
            expr += 'do {\n'
            expr += _clean_loop_body(control_flow_region_to_code(loop, dispatch_state, codegen, symbols))
            expr += f'\n}} while({cond});\n'
        else:
            expr += f'while ({cond}) {{\n'
            expr += _clean_loop_body(control_flow_region_to_code(loop, dispatch_state, codegen, symbols))
            expr += '\n}\n'

    return expr


def _conditional_block_to_code(region: ConditionalBlock, dispatch_state: Callable[[SDFGState], str],
                               codegen: 'DaCeCodeGenerator', symbols: Dict[str, dtypes.typeclass]) -> str:
    """
    Converts a ConditionalBlock to C++ code with the correct control flow expressions.

    :param region:          The ConditionalBlock to convert.
    :param dispatch_state:  A callback to generate code for a given SDFG state.
    :param codegen:         The code generator object, used for allocation information and defined variables in scope.
    :param symbols:         A dictionary of symbol names and their types.
    :return:                C++ string with the generated code of the conditional block.
    """
    sdfg = region.sdfg
    expr = ''
    for i, (cond, body) in enumerate(region.branches):
        if cond is not None:
            cond_str = unparse_interstate_edge(cond.code, sdfg, codegen=codegen, symbols=symbols)
            cond_str = cond_str.strip(';')
            if i == 0:
                expr += f'if ({cond_str}) {{\n'
            else:
                expr += f'}} else if ({cond_str}) {{\n'
        else:
            if i < len(region.branches) - 1 or i == 0:
                raise RuntimeError('Missing branch condition for non-final conditional branch')
            expr += '} else {\n'
        expr += control_flow_region_to_code(body, dispatch_state, codegen, symbols)
        if i == len(region.branches) - 1:
            expr += '}\n'
    return expr


def control_flow_region_to_code(region: AbstractControlFlowRegion,
                                dispatch_state: Callable[[SDFGState], str],
                                codegen: 'DaCeCodeGenerator',
                                symbols: Dict[str, dtypes.typeclass],
                                start: Optional[ControlFlowBlock] = None,
                                stop: Optional[ControlFlowBlock] = None,
                                generate_children_of: Optional[ControlFlowBlock] = None,
                                ptree: Optional[Dict[ControlFlowBlock, ControlFlowBlock]] = None,
                                visited: Optional[Set[ControlFlowBlock]] = None) -> str:
    """
    Converts a control flow region to C++ code with the correct control flow expressions.

    :param region:          The control flow region to convert.
    :param dispatch_state:  A callback to generate code for a given SDFG state.
    :param codegen:         The code generator object, used for allocation information and defined variables in scope.
    :param symbols:         A dictionary of symbol names and their types.
    :return:                C++ string with the generated code of the control flow region.
    """
    expr = ''

    if ptree is None:
        ptree = cfg_analysis.block_parent_tree(region, with_loops=False)
    start = start if start is not None else region.start_block
    visited = set() if visited is None else visited

    contains_irreducible = any(region.out_degree(node) > 1
                               for node in region.nodes()) or isinstance(region, UnstructuredControlFlow)

    stack = [region.start_block]
    while stack:
        node = stack.pop()
        if (generate_children_of is not None and not _child_of(node, generate_children_of, ptree)):
            continue
        if node in visited or node is stop:
            continue
        visited.add(node)

        expr += '__state_{}_{}:;\n'.format(region.cfg_id, re.sub(r'\s+', '_', node.label))
        if isinstance(node, SDFGState):
            if node.number_of_nodes() > 0:
                expr += '{\n'
                expr += dispatch_state(node)
                expr += '\n}\n'
            else:
                # Dispatch empty state in any case in order to register that the state was dispatched.
                expr += dispatch_state(node)
        elif isinstance(node, BreakBlock):
            expr += 'break;\n'
        elif isinstance(node, ContinueBlock):
            expr += 'continue;\n'
        elif isinstance(node, ReturnBlock):
            expr += 'return;\n'
        elif isinstance(node, LoopRegion):
            expr += _loop_region_to_code(node, dispatch_state, codegen, symbols)
        elif isinstance(node, ConditionalBlock):
            expr += _conditional_block_to_code(node, dispatch_state, codegen, symbols)
        elif isinstance(node, ControlFlowRegion):
            expr += control_flow_region_to_code(node, dispatch_state, codegen, symbols)
        else:
            raise NotImplementedError(f'Control flow block {type(node)} not implemented')

        out_edges = region.out_edges(node)
        if len(out_edges) == 0:
            # If no outgoing edges, this is the last block and we can exit the region.
            expr += f'goto __state_exit_{region.cfg_id};\n'
        elif len(out_edges) == 1:
            # Only one outgoing edge, continue to the next block.
            if out_edges[0].data.is_unconditional():
                # If unconditional, just continue to the next state, adding an unconditional goto.
                expr += _generate_interstate_edge_code(out_edges[0],
                                                       region.sdfg,
                                                       region,
                                                       codegen,
                                                       assignments_only=(not contains_irreducible))
            else:
                # If conditional, generate a conditional goto and exit otherwise.
                expr += _generate_interstate_edge_code(out_edges[0], region.sdfg, region, codegen, exit_on_else=True)
            stack.append(out_edges[0].dst)
        else:
            # If multiple outgoing edges, generate a conditional goto for each edge. This is the case for
            # unstructured / irreducible control flow.
            unconditional_edge = None
            for e in out_edges:
                if e.data.is_unconditional():
                    # Defer generating the unconditional edge until the end so any conditional edges are checked first
                    # before the "else" edge is taken. If there are multiple unconditional edges, i.e., there already
                    # is a deferred unconditional edge, raise a warning and generate the remaining ones as they appear.
                    if unconditional_edge is not None:
                        warnings.warn(
                            f'Unstructured control flow region {region.label} has multiple unconditional edges '
                            f'leading out of block {node.label}.')
                    else:
                        unconditional_edge = e
                        continue
                expr += _generate_interstate_edge_code(e, region.sdfg, region, codegen)
                stack.append(e.dst)
            if unconditional_edge is None:
                # If no unconditional edge, we need to exit the region if none of the conditions are met.
                expr += f'goto __state_exit_{region.cfg_id};\n'
            else:
                expr += _generate_interstate_edge_code(unconditional_edge, region.sdfg, region, codegen)
                stack.append(unconditional_edge.dst)

    expr += f'__state_exit_{region.cfg_id}:;\n'

    return expr
