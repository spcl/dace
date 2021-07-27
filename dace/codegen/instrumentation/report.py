# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Implementation of the performance instrumentation report. """

import json
import numpy as np
import re
from collections import defaultdict
from dace.properties import EnumProperty, make_properties, Property
from dace import dtypes


@make_properties
class InstrumentationReport(object):

    sortingType = EnumProperty(
        dtype=dtypes.InstrumentationReportPrintType,
        default=dtypes.InstrumentationReportPrintType.SDFG,
        desc="SDFG: Sorting: SDFG, State, Node, Location"
        "Location:  Sorting: Location, SDFG, State, Node")

    @staticmethod
    def get_event_uuid(event):
        try:
            args = event['args']
        except KeyError:
            return (-1, -1, -1, -1)

        uuid = (args.get('sdfg_id', -1), args.get('state_id', -1),
                args.get('id', -1), args.get('loc_id', -1))

        return uuid

    def __init__(self,
                 filename: str,
                 sortingType: dtypes.InstrumentationReportPrintType = dtypes.
                 InstrumentationReportPrintType.SDFG):
        self.sortingType = sortingType
        # Parse file
        match = re.match(r'.*report-(\d+)\.json', filename)
        self._name = match.groups()[0] if match is not None else 'N/A'

        self._durations = {}
        self._counters = {}

        with open(filename, 'r') as fp:
            report = json.load(fp)

            if 'traceEvents' not in report or 'sdfgHash' not in report:
                print(filename, 'is not a valid SDFG instrumentation report!')
                return

            self._sdfg_hash = report['sdfgHash']

            events = report['traceEvents']

            for event in events:
                if 'ph' in event:
                    phase = event['ph']
                    name = event['name']
                    if phase == 'X':
                        uuid = self.get_event_uuid(event)
                        if uuid not in self._durations:
                            self._durations[uuid] = {}
                        if name not in self._durations[uuid]:
                            self._durations[uuid][name] = []
                        self._durations[uuid][name].append(event['dur'] / 1000)
                    if phase == 'C':
                        if name not in self.counters:
                            self.counters[name] = 0
                        self.counters[name] += event['args'][name]

    def __repr__(self):
        return 'InstrumentationReport(name=%s)' % self._name

    def _get_runtimes_string(self,
                             label,
                             runtimes,
                             element,
                             sdfg,
                             state,
                             string,
                             row_format,
                             format_dict,
                             with_element_heading=True):
        location_label = f'Device: {element[3]}' if element[3] != -1 else 'CPU'
        indent = ''
        if len(runtimes) > 0:
            element_label = ''
            format_dict['loc'] = ''
            format_dict['min'] = ''
            format_dict['mean'] = ''
            format_dict['median'] = ''
            format_dict['max'] = ''
            if element[0] > -1 and element[1] > -1 and element[2] > -1:

                # This element is a node.
                if sdfg != element[0]:
                    # No parent SDFG row present yet.
                    format_dict['elem'] = 'SDFG (' + str(element[0]) + ')'
                sdfg = element[0]

                if state != element[1]:
                    # No parent state row present yet.
                    format_dict['elem'] = '|-State (' + str(element[1]) + ')'

                # Print
                string += row_format.format(**format_dict)
                state = element[1]
                element_label = '| |-Node (' + str(element[2]) + ')'
                indent = '| | |'
            elif element[0] > -1 and element[1] > -1:
                # This element is a state.
                if sdfg != element[0]:
                    # No parent SDFG row present yet, print it.
                    format_dict['elem'] = 'SDFG (' + str(element[0]) + ')'
                    string += row_format.format(**format_dict)

                sdfg = element[0]
                state = element[1]
                element_label = '|-State (' + str(element[1]) + ')'
                indent = '| |'
            elif element[0] > -1:
                # This element is an SDFG.
                sdfg = element[0]
                state = -1
                element_label = 'SDFG (' + str(element[0]) + ')'
                indent = '|'
            else:
                element_label = 'N/A'

            if with_element_heading:
                format_dict['elem'] = element_label
                string += row_format.format(**format_dict)

            format_dict['elem'] = indent + label + ':'
            string += row_format.format(**format_dict)

            format_dict['elem'] = indent
            format_dict['loc'] = location_label
            format_dict['min'] = '%.3f' % np.min(runtimes)
            format_dict['mean'] = '%.3f' % np.mean(runtimes)
            format_dict['median'] = '%.3f' % np.median(runtimes)
            format_dict['max'] = '%.3f' % np.max(runtimes)
            string += row_format.format(**format_dict)

        return string, sdfg, state

    def __str__(self):
        element_list = list(self._durations.keys())
        element_list.sort()

        string = 'Instrumentation report\n'
        string += 'SDFG Hash: ' + self._sdfg_hash + '\n'

        if len(self._durations) > 0:
            COLW_ELEM = 30
            COLW_LOC = 15
            COLW_RUNTIME = 15
            NUM_RUNTIME_COLS = 4

            line_string = (
                '-' *
                (COLW_RUNTIME * NUM_RUNTIME_COLS + COLW_ELEM + COLW_LOC)) + '\n'

            string += line_string

            if self.sortingType == dtypes.InstrumentationReportPrintType.Location:
                row_format = ('{loc:<{loc_width}}') + (
                    '{elem:<{elem_width}}') + ('{min:<{width}}') + (
                        '{mean:<{width}}') + ('{median:<{width}}') + (
                            '{max:<{width}}') + '\n'
                string += ('{:<{width}}').format('Location', width=COLW_LOC)
                string += ('{:<{width}}').format('Element', width=COLW_ELEM)
            else:
                row_format = ('{elem:<{elem_width}}') + (
                    '{loc:<{loc_width}}') + ('{min:<{width}}') + (
                        '{mean:<{width}}') + ('{median:<{width}}') + (
                            '{max:<{width}}') + '\n'
                string += ('{:<{width}}').format('Element', width=COLW_ELEM)
                string += ('{:<{width}}').format('Location', width=COLW_LOC)

            string += ('{:<{width}}').format('Runtime (ms)', width=COLW_RUNTIME)
            string += '\n'
            format_dict = {
                'elem_width': COLW_ELEM,
                'loc_width': COLW_LOC,
                'width': COLW_RUNTIME,
                'loc': '',
                'elem': '',
                'min': 'Min',
                'mean': 'Mean',
                'median': 'Median',
                'max': 'Max'
            }
            string += row_format.format(**format_dict)
            string += line_string

            if self.sortingType == dtypes.InstrumentationReportPrintType.Location:
                location_elements = defaultdict(list)
                for element in element_list:
                    location_elements[element[3]].append(element)
                for location in sorted(location_elements):
                    elements_list = location_elements[location]
                    sdfg = -1
                    state = -1
                    for element in elements_list:
                        events = self._durations[element]
                        if len(events) > 0:
                            with_element_heading = True
                            for event in events.keys():
                                runtimes = events[event]
                                string, sdfg, state = self._get_runtimes_string(
                                    event, runtimes, element, sdfg, state,
                                    string, row_format, format_dict,
                                    with_element_heading)
                                with_element_heading = False

            else:
                sdfg = -1
                state = -1
                for element in element_list:
                    events = self._durations[element]
                    if len(events) > 0:
                        with_element_heading = True
                        for event in events.keys():
                            runtimes = events[event]
                            string, sdfg, state = self._get_runtimes_string(
                                event, runtimes, element, sdfg, state, string,
                                row_format, format_dict, with_element_heading)
                            with_element_heading = False
                    runtimes = self._durations[element]

            string += line_string

        if len(self._counters) > 0:
            COUNTER_COLW = 39
            counter_format = ('{:<{width}}' * 2) + '\n'

            string += ('-' * (COUNTER_COLW * 2)) + '\n'
            string += ('{:<{width}}' * 2).format(
                'Counter', 'Value', width=COUNTER_COLW) + '\n'
            string += ('-' * (COUNTER_COLW * 2)) + '\n'
            for counter in self._counters:
                string += counter_format.format(counter,
                                                self._counters[counter],
                                                width=COUNTER_COLW)
            string += ('-' * (COUNTER_COLW * 2)) + '\n'

        return string
