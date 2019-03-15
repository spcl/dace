""" Contains classes and functions related to optimization of the stateful
    dataflow graph representation. """

import copy
import os
import re
import time

import dace
from dace.config import Config
from dace.graph import labeling
from dace.transformation import pattern_matching

# This import is necessary since it registers all the patterns
from dace.transformation import dataflow, interstate


class SDFGOptimizer(object):
    """ Implements methods for optimizing a DaCe program stateful dataflow
        graph representation, by matching patterns and applying 
        transformations on it.
    """

    def __init__(self, sdfg, inplace=False):
        """ Constructs an SDFG optimizer.
            @param sdfg: The SDFG to transform.
            @param inplace: If True, performs transformations on the given SDFG
                            in-place. Uses a copy of the SDFG otherwise, and
                            stores it as `self.sdfg`.
        """
        if inplace == True:
            self.sdfg = sdfg
        else:
            self.sdfg = copy.deepcopy(sdfg)

        # Initialize patterns to search for
        self.patterns = pattern_matching.Transformation.patterns()
        self.stateflow_patterns = pattern_matching.Transformation.stateflow_patterns(
        )
        self.applied_patterns = set()

    def get_pattern_matches(self, strict=False, states=None, patterns=None):
        """ Returns all possible transformations for the current SDFG.
            @param strict: Only consider strict transformations (i.e., ones
                           that surely increase performance or enhance
                           readability)
            @param states: An iterable of SDFG states to consider when pattern
                           matching. If None, considers all.
            @param patterns: An iterable of transformation classes to consider
                             when matching. If None, considers all registered
                             transformations in `Transformation`.
            @return: List of matching `Transformation` objects.
            @see: Transformation
        """

        matches = []

        if states is None:
            if patterns is None:
                _patterns = self.stateflow_patterns
            else:
                _patterns = [
                    p for p in patterns if p in self.stateflow_patterns
                ]

            for pattern in _patterns:
                matches += pattern_matching.match_stateflow_pattern(
                    self.sdfg, pattern, strict=strict)

        state_enum = []
        if states is None:
            for state_id, state in enumerate(self.sdfg.nodes()):
                state_enum.append((state_id, state))
        else:
            for state in states:
                state_id = self.sdfg.nodes().index(state)
                state_enum.append((state_id, state))

        if patterns is None:
            _patterns = self.patterns
        else:
            _patterns = [p for p in patterns if p in self.patterns]
        for state_id, state in state_enum:
            for pattern in _patterns:
                matches += pattern_matching.match_pattern(
                    state_id, state, pattern, self.sdfg, strict=strict)

        return matches

    def optimize(self, debugprint=True):
        """ A command-line UI for applying patterns on the SDFG.
            @param debugprint: Whether to print verbose information to the 
                               console.
            @return: An optimized SDFG object
        """

        # Visualize SDFGs during optimization process
        VISUALIZE = Config.get_bool('optimizer', 'visualize')
        SAVE_DOTS = Config.get_bool('optimizer', 'savedots')

        if SAVE_DOTS:
            with open('before.dot', 'w') as dot_file:
                dot_file.write(self.sdfg.draw())
            if VISUALIZE:
                os.system('xdot before.dot&')

        # Optimize until there is not pattern matching or user stops the process.
        pattern_counter = 0
        while True:
            # Print in the UI all the pattern matching options.
            ui_options = self.get_pattern_matches()
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

            # Code working for both python 2.x and 3.x.
            try:
                ui_input = raw_input(
                    'Select the pattern to apply (0 - %d or name$id): ' %
                    (ui_options_idx - 1))
            except NameError:
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
                with open(
                        'after_%d_%s_b4lprop.dot' %
                    (pattern_counter + 1, type(pattern_match).__name__),
                        'w') as dot_file:
                    dot_file.write(self.sdfg.draw())

            if not pattern_match.annotates_memlets():
                labeling.propagate_labels_sdfg(self.sdfg)

            if True:
                pattern_counter += 1
                if SAVE_DOTS:
                    with open(
                            'after_%d_%s.dot' % (pattern_counter,
                                                 type(pattern_match).__name__),
                            'w') as dot_file:
                        dot_file.write(self.sdfg.draw())
                    if VISUALIZE:
                        time.sleep(0.7)
                        os.system(
                            'xdot after_%d_%s.dot&' %
                            (pattern_counter, type(pattern_match).__name__))

        return self.sdfg


def _parse_cli_input(line):
    """ Parses a command line input, which may include a transformation name
        (optional), its occurrence ID, and its parameters (optional).
        Syntax Examples:
            * 5                  - Chooses the fifth transformation
            * MapReduceFusion$0  - First occurrence of MapReduceFusion
            * 4(array='A')       - Transformation number 4 with one parameter
            * StripMining$1(param='i', tile_size=64) - Strip mining #2 with
                                                       parameters
        @param line: Input line string
        @return: A tuple with (transformation name or None if not given,
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
