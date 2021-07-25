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
                        if uuid not in self.durations:
                            self.durations[uuid] = {}
                        if name not in self.durations[uuid]:
                            self.durations[uuid][name] = []
                        self.durations[uuid][name].append(event['dur'] / 1000)
                    if phase == 'C':
                        if name not in self.counters:
                            self.counters[name] = 0
                        self.counters[name] += event['args'][name]

    def __repr__(self):
        return 'InstrumentationReport(name=%s)' % self._name

    def _get_runtimes_string(
        self, label, runtimes, element, sdfg, state, string, row_format, colw,
        with_element_heading=True
    ):
        indent = ''
        if len(runtimes) > 0:
            element_label = ''
            if element[0] > -1 and element[1] > -1 and element[2] > -1:
                # This element is a node.
                if sdfg != element[0]:
                    # No parent SDFG row present yet, print it.
                    string += row_format.format('SDFG (' +
                                                str(element[0]) + ')',
                                                '',
                                                '',
                                                '',
                                                '',
                                                width=colw)
                sdfg = element[0]
                if state != element[1]:
                    # No parent state row present yet, print it.
                    string += row_format.format('|-State (' +
                                                str(element[1]) + ')',
                                                '',
                                                '',
                                                '',
                                                '',
                                                width=colw)
                state = element[1]
                element_label = '| |-Node (' + str(element[2]) + ')'
                indent = '| | |'
            elif element[0] > -1 and element[1] > -1:
                # This element is a state.
                if sdfg != element[0]:
                    # No parent SDFG row present yet, print it.
                    string += row_format.format('SDFG (' +
                                                str(element[0]) + ')',
                                                '',
                                                '',
                                                '',
                                                '',
                                                width=colw)
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
                string += row_format.format(element_label,
                                            '',
                                            '',
                                            '',
                                            '',
                                            width=colw)

            string += row_format.format(indent + label + ':',
                                        '', '', '', '', width=colw)
            string += row_format.format(indent,
                                        '%.3f' % np.min(runtimes),
                                        '%.3f' % np.mean(runtimes),
                                        '%.3f' % np.median(runtimes),
                                        '%.3f' % np.max(runtimes),
                                        width=colw)

        return string, sdfg, state

    def __str__(self):
        if self.sortingType == dtypes.InstrumentationReportPrintType.Location:
            return self._LocationGroup()
        return self._SDFGGroup()

    def _LocationGroup(self):
        COLW = 15
        COUNTER_COLW = 39
        NUM_COLS = 6

        element_list = list(self._durations.keys())
        element_list.sort()

        row_format = ('{:<{width}}' * NUM_COLS) + '\n'
        counter_format = ('{:<{width}}' * 2) + '\n'

        string = 'Instrumentation report\n'
        string += 'SDFG Hash: ' + self._sdfg_hash + '\n'

        location_elements = defaultdict(list)

        for element in element_list:
            location_elements[element[3]].append(element)

        if len(self._durations) > 0:
            string += ('-' * (COLW * NUM_COLS)) + '\n'
            string += ('{:<{width}}' * 3).format(
                'Location', 'Element', 'Runtime (ms)', width=COLW) + '\n'
            string += row_format.format('',
                                        '',
                                        'Min',
                                        'Mean',
                                        'Median',
                                        'Max',
                                        width=COLW)
            string += ('-' * (COLW * 5)) + '\n'
            for location, elements_list in location_elements.items():

                sdfg = -1
                state = -1

                for element in elements_list:
                    runtimes = self._durations[element]
                    location_label = f'Device: {location}' if location != -1 else 'CPU'
                    if len(runtimes) > 0:
                        element_label = ''
                        if element[0] > -1 and element[1] > -1 and element[
                                2] > -1:
                            # This element is a node.
                            if sdfg != element[0]:
                                # No parent SDFG row present yet, print it.
                                string += row_format.format(
                                    '',
                                    'SDFG (' + str(element[0]) + ')',
                                    '',
                                    '',
                                    '',
                                    '',
                                    width=COLW)
                            sdfg = element[0]
                            if state != element[1]:
                                # No parent state row present yet, print it.
                                string += row_format.format(
                                    '',
                                    '| State (' + str(element[1]) + ')',
                                    '',
                                    '',
                                    '',
                                    '',
                                    width=COLW)
                            state = element[1]
                            element_label = '| | Node (' + str(element[2]) + ')'
                        elif element[0] > -1 and element[1] > -1:
                            # This element is a state.
                            if sdfg != element[0]:
                                # No parent SDFG row present yet, print it.
                                string += row_format.format(
                                    '',
                                    'SDFG (' + str(element[0]) + ')',
                                    '',
                                    '',
                                    '',
                                    '',
                                    width=COLW)
                            sdfg = element[0]
                            state = element[1]
                            element_label = '| State (' + str(element[1]) + ')'
                        elif element[0] > -1:
                            # This element is an SDFG.
                            sdfg = element[0]
                            state = -1
                            element_label = 'SDFG (' + str(element[0]) + ')'
                        else:
                            element_label = 'N/A'

                        string += row_format.format(location_label,
                                                    element_label,
                                                    '%.3f' % np.min(runtimes),
                                                    '%.3f' % np.mean(runtimes),
                                                    '%.3f' %
                                                    np.median(runtimes),
                                                    '%.3f' % np.max(runtimes),
                                                    width=COLW)
                string += ('-' * (COLW * NUM_COLS)) + '\n'

        if len(self._counters) > 0:
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

    def _SDFGGroup(self):
        COLW = 15
        COUNTER_COLW = 39
        NUM_COLS = 6

        element_list = list(self._durations.keys())
        element_list.sort()

        row_format = ('{:<{width}}' * NUM_COLS) + '\n'
        counter_format = ('{:<{width}}' * 2) + '\n'

        string = 'Instrumentation report\n'
        string += 'SDFG Hash: ' + self._sdfg_hash + '\n'

        if len(self._durations) > 0:
            string += ('-' * (COLW * NUM_COLS)) + '\n'
            string += ('{:<{width}}' * 3).format(
                'Element', 'Location', 'Runtime (ms)', width=COLW) + '\n'
            string += row_format.format('',
                                        '',
                                        'Min',
                                        'Mean',
                                        'Median',
                                        'Max',
                                        width=COLW)
            string += ('-' * (COLW * 5)) + '\n'

            sdfg = -1
            state = -1

            for element in element_list:
                events = self.durations[element]
                if len(events) > 0:
                    with_element_heading = True
                    for event in events.keys():
                        runtimes = events[event]
                        string, sdfg, state = self._get_runtimes_string(
                            event, runtimes, element, sdfg, state, string,
                            row_format, COLW, with_element_heading
                        )
                        with_element_heading = False
                runtimes = self._durations[element]
                location_label = f'Device: {element[3]}' if element[
                    3] != -1 else 'CPU'
                if len(runtimes) > 0:
                    element_label = ''
                    if element[0] > -1 and element[1] > -1 and element[2] > -1:
                        # This element is a node.
                        if sdfg != element[0]:
                            # No parent SDFG row present yet, print it.
                            string += row_format.format('SDFG (' +
                                                        str(element[0]) + ')',
                                                        '',
                                                        '',
                                                        '',
                                                        '',
                                                        '',
                                                        width=COLW)
                        sdfg = element[0]
                        if state != element[1]:
                            # No parent state row present yet, print it.
                            string += row_format.format('| State (' +
                                                        str(element[1]) + ')',
                                                        '',
                                                        '',
                                                        '',
                                                        '',
                                                        '',
                                                        width=COLW)
                        state = element[1]
                        element_label = '| | Node (' + str(element[2]) + ')'
                    elif element[0] > -1 and element[1] > -1:
                        # This element is a state.
                        if sdfg != element[0]:
                            # No parent SDFG row present yet, print it.
                            string += row_format.format('SDFG (' +
                                                        str(element[0]) + ')',
                                                        '',
                                                        '',
                                                        '',
                                                        '',
                                                        '',
                                                        width=COLW)
                        sdfg = element[0]
                        state = element[1]
                        element_label = '| State (' + str(element[1]) + ')'
                    elif element[0] > -1:
                        # This element is an SDFG.
                        sdfg = element[0]
                        state = -1
                        element_label = 'SDFG (' + str(element[0]) + ')'
                    else:
                        element_label = 'N/A'

            string += ('-' * (COLW * 5)) + '\n'

        if len(self._counters) > 0:
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
