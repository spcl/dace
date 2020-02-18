""" Contains classes and functions related to optimization of the stateful
    dataflow graph representation. """

import copy
import os
import re
import time

import dace
from dace.config import Config
from dace.graph import labeling
from dace.graph.graph import SubgraphView
from dace.transformation import pattern_matching

# This import is necessary since it registers all the patterns
from dace.transformation import dataflow, interstate


class Optimizer(object):
    """ Implements methods for optimizing a DaCe program stateful dataflow
        graph representation, by matching patterns and applying 
        transformations on it.
    """

    def __init__(self, sdfg, inplace=False):
        """ Constructs an SDFG optimizer.
            :param sdfg: The SDFG to transform.
            :param inplace: If True, performs transformations on the given SDFG
                            in-place. Uses a copy of the SDFG otherwise, and
                            stores it as `self.sdfg`.
        """
        if inplace == True:
            self.sdfg = sdfg
        else:
            self.sdfg = copy.deepcopy(sdfg)

        # Initialize patterns to search for
        self.patterns = set(
            k for k, v in pattern_matching.Transformation.extensions().items()
            if v.get('singlestate', False))
        self.stateflow_patterns = set(pattern_matching.Transformation.
                                      extensions().keys()) - self.patterns
        self.applied_patterns = set()

    def optimize(self):
        # Should be implemented by subclass
        raise NotImplementedError

    def get_pattern_matches(self,
                            strict=False,
                            states=None,
                            patterns=None,
                            sdfg=None):
        """ Returns all possible transformations for the current SDFG.
            :param strict: Only consider strict transformations (i.e., ones
                           that surely increase performance or enhance
                           readability)
            :param states: An iterable of SDFG states to consider when pattern
                           matching. If None, considers all.
            :param patterns: An iterable of transformation classes to consider
                             when matching. If None, considers all registered
                             transformations in `Transformation`.
            :param sdfg: If not None, searches for patterns on given SDFG.
            :return: List of matching `Transformation` objects.
            @see: Transformation
        """
        sdfg = sdfg or self.sdfg

        if states is None:
            if patterns is None:
                _patterns = self.stateflow_patterns
            else:
                _patterns = [
                    p for p in patterns if p in self.stateflow_patterns
                ]

            for pattern in _patterns:
                yield from pattern_matching.match_stateflow_pattern(
                    sdfg, pattern, strict=strict)

        state_enum = []
        if states is None:
            for state_id, state in enumerate(sdfg.nodes()):
                state_enum.append((state_id, state))
        else:
            for state in states:
                state_id = sdfg.nodes().index(state)
                state_enum.append((state_id, state))

        if patterns is None:
            _patterns = self.patterns
        else:
            _patterns = [p for p in patterns if p in self.patterns]
        for state_id, state in state_enum:
            for pattern in _patterns:
                yield from pattern_matching.match_pattern(
                    state, pattern, sdfg, strict=strict)

    def optimization_space(self):
        """ Returns the optimization space of the current SDFG """

        def get_actions(actions, graph, match):
            subgraph_node_ids = match.subgraph.values()
            subgraph_nodes = [graph.nodes()[nid] for nid in subgraph_node_ids]
            for node in subgraph_nodes:
                version = 0
                while (node, type(match).__name__, match.expr_index,
                       version) in actions.keys():
                    version += 1
                actions[(node, type(match).__name__, match.expr_index,
                         version)] = match
            subgraph = SubgraphView(graph, subgraph_nodes)
            for edge in subgraph.edges():
                version = 0
                while (edge, type(match).__name__, match.expr_index,
                       version) in actions.keys():
                    version += 1
                actions[(edge, type(match).__name__, match.expr_index,
                         version)] = match
            return actions

        def get_dataflow_actions(actions, sdfg, match):
            graph = sdfg.sdfg_list[match.sdfg_id].nodes()[match.state_id]
            return get_actions(actions, graph, match)

        def get_stateflow_actions(actions, sdfg, match):
            graph = sdfg.sdfg_list[match.sdfg_id]
            return get_actions(actions, graph, match)

        actions = dict()

        for match in self.get_pattern_matches():
            if match.state_id >= 0:
                actions = get_dataflow_actions(actions, self.sdfg, match)
            else:
                actions = get_stateflow_actions(actions, self.sdfg, match)

        return actions


def _parse_cli_input(line):
    """ Parses a command line input, which may include a transformation name
        (optional), its occurrence ID, and its parameters (optional).
        Syntax Examples:
            * 5                  - Chooses the fifth transformation
            * MapReduceFusion$0  - First occurrence of MapReduceFusion
            * 4(array='A')       - Transformation number 4 with one parameter
            * StripMining$1(param='i', tile_size=64) - Strip mining #2 with
                                                       parameters
        :param line: Input line string
        :return: A tuple with (transformation name or None if not given,
                                      occurrence or -1 if not given,
                                      parameter dictionary or {} if not given)
    """
    # First try matching explicit all-inclusive string "A$num(values)"
    match = re.findall(r'(.*)\$(\d+)\((.*)\)', line)
    if len(match) == 1:
        trans_name, occurrence, param_dict = match[0]
    else:
        # Then, try to match "num(values)"
        match = re.findall(r'(\d+)\((.*)\)', line)
        if len(match) == 1:
            trans_name = None
            occurrence, param_dict = match[0]
        else:
            # After that, try to match "A$num"
            match = re.findall(r'(.*)\$(\d+)', line)
            if len(match) == 1:
                trans_name, occurrence = match[0]
                param_dict = {}
            else:
                # Finally, try to match "num"
                match = re.findall(r'(\d+)', line)
                if len(match) == 1:
                    trans_name = None
                    occurrence = match[0]
                    param_dict = {}
                else:
                    return (None, -1, {})

    # Try to parse the results
    try:
        occurrence = int(occurrence)
    except ValueError:
        occurrence = -1
    try:
        if isinstance(param_dict, str):
            param_dict = eval('dict(' + param_dict + ')')
    except:  # Here we have to catch ANY exception since literally anything
        # can happen
        param_dict = {}

    return trans_name, occurrence, param_dict


class SDFGOptimizer(Optimizer):
    def optimize(self):
        """ A command-line UI for applying patterns on the SDFG.
            :return: An optimized SDFG object
        """
        sdfg_file = self.sdfg.name + '.sdfg'
        if os.path.isfile(sdfg_file):
            ui_input = input(
                'An SDFG with the filename "%s" was found. '
                'Would you like to use it instead? [Y/n] ' % sdfg_file)
            if len(ui_input) == 0 or ui_input[0] not in ['n', 'N']:
                return dace.SDFG.from_file(sdfg_file)

        # Visualize SDFGs during optimization process
        VISUALIZE = Config.get_bool('optimizer', 'visualize')
        VISUALIZE_SDFV = Config.get_bool('optimizer', 'visualize_sdfv')
        SAVE_DOTS = Config.get_bool('optimizer', 'savedots')

        if SAVE_DOTS:
            self.sdfg.draw_to_file('before.dot')
            self.sdfg.save(os.path.join('_dotgraphs', 'before.sdfg'))
            if VISUALIZE:
                os.system('xdot _dotgraphs/before.dot&')
            if VISUALIZE_SDFV:
                os.system('sdfv _dotgraphs/before.sdfg&')

        # Optimize until there is not pattern matching or user stops the process.
        pattern_counter = 0
        while True:
            # Print in the UI all the pattern matching options.
            ui_options = sorted(self.get_pattern_matches())
            ui_options_idx = 0
            for pattern_match in ui_options:
                sdfg = self.sdfg.sdfg_list[pattern_match.sdfg_id]
                print('%d. Transformation %s' %
                      (ui_options_idx, pattern_match.print_match(sdfg)))
                ui_options_idx += 1

            # If no pattern matchings were found, quit.
            if ui_options_idx == 0:
                print('No viable transformations found')
                break

            ui_input = input(
                'Select the pattern to apply (0 - %d or name$id): ' %
                (ui_options_idx - 1))

            pattern_name, occurrence, param_dict = _parse_cli_input(ui_input)

            pattern_match = None
            if (pattern_name is None and occurrence >= 0
                    and occurrence < ui_options_idx):
                pattern_match = ui_options[occurrence]
            elif pattern_name is not None:
                counter = 0
                for match in ui_options:
                    if type(match).__name__ == pattern_name:
                        if occurrence == counter:
                            pattern_match = match
                            break
                        counter = counter + 1

            if pattern_match is None:
                print(
                    'You did not select a valid option. Quitting optimization ...'
                )
                break

            match_id = (str(occurrence) if pattern_name is None else
                        '%s$%d' % (pattern_name, occurrence))
            sdfg = self.sdfg.sdfg_list[pattern_match.sdfg_id]
            print('You selected (%s) pattern %s with parameters %s' %
                  (match_id, pattern_match.print_match(sdfg), str(param_dict)))

            # Set each parameter of the parameter dictionary separately
            for k, v in param_dict.items():
                setattr(pattern_match, k, v)

            pattern_match.apply(sdfg)
            self.applied_patterns.add(type(pattern_match))

            if SAVE_DOTS:
                filename = 'after_%d_%s_b4lprop' % (
                    pattern_counter + 1, type(pattern_match).__name__)
                self.sdfg.save(os.path.join('_dotgraphs', filename + '.sdfg'))
                self.sdfg.draw_to_file(filename + '.dot')

            if not pattern_match.annotates_memlets():
                labeling.propagate_labels_sdfg(self.sdfg)

            if True:
                pattern_counter += 1
                if SAVE_DOTS:
                    filename = 'after_%d_%s' % (pattern_counter,
                                                type(pattern_match).__name__)
                    self.sdfg.save(
                        os.path.join('_dotgraphs', filename + '.sdfg'))
                    self.sdfg.draw_to_file(filename + '.dot')

                    if VISUALIZE:
                        time.sleep(0.7)
                        os.system(
                            'xdot _dotgraphs/after_%d_%s.dot&' %
                            (pattern_counter, type(pattern_match).__name__))

                    if VISUALIZE_SDFV:
                        os.system(
                            'sdfv _dotgraphs/after_%d_%s.sdfg&' %
                            (pattern_counter, type(pattern_match).__name__))

        return self.sdfg
