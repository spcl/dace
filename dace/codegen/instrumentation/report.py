# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Implementation of the performance instrumentation report. """

import json
import numpy as np
import re


class InstrumentationReport(object):

    @staticmethod
    def get_event_uuid(event):
        uuid = (-1, -1, -1)
        if 'args' in event:
            args = event['args']
            if 'sdfg_id' in args and args['sdfg_id'] is not None:
                uuid = (args['sdfg_id'], -1, -1)
                if 'state_id' in args and args['state_id'] is not None:
                    uuid = (uuid[0], args['state_id'], -1)
                    if 'id' in args and args['id'] is not None:
                        uuid = (uuid[0], uuid[1], args['id'])
        return uuid

    def __init__(self, filename: str):
        # Parse file
        match = re.match(r'.*report-(\d+)\.json', filename)
        self.name = match.groups()[0] if match is not None else 'N/A'

        self.durations = {}
        self.counters = {}

        with open(filename, 'r') as fp:
            report = json.load(fp)

            if 'traceEvents' not in report or 'sdfgHash' not in report:
                print(filename, 'is not a valid SDFG instrumentation report!')
                return

            self.sdfg_hash = report['sdfgHash']

            events = report['traceEvents']

            for event in events:
                if 'ph' in event:
                    phase = event['ph']
                    if phase == 'X':
                        uuid = self.get_event_uuid(event)
                        if uuid not in self.durations:
                            self.durations[uuid] = []
                        self.durations[uuid].append(event['dur'] / 1000)
                    if phase == 'C':
                        name = event['name']
                        if name not in self.counters:
                            self.counters[name] = 0
                        self.counters[name] += event['args'][name]

    def __repr__(self):
        return 'InstrumentationReport(name=%s)' % self.name

    def __str__(self):
        COLW = 15
        COUNTER_COLW = 39

        element_list = list(self.durations.keys())
        element_list.sort()

        row_format = ('{:<{width}}' * 5) + '\n'
        counter_format = ('{:<{width}}' * 2) + '\n'

        string = 'Instrumentation report\n'
        string += 'SDFG Hash: ' + self.sdfg_hash + '\n'

        if len(self.durations) > 0:
            string += ('-' * (COLW * 5)) + '\n'
            string += ('{:<{width}}' * 2).format(
                'Element', 'Runtime (ms)', width=COLW
            ) + '\n'
            string += row_format.format(
                '', 'Min', 'Mean', 'Median', 'Max', width=COLW
            )
            string += ('-' * (COLW * 5)) + '\n'

            sdfg = -1
            state = -1

            for element in element_list:
                runtimes = self.durations[element]
                if len(runtimes) > 0:
                    element_label = ''
                    if element[0] > -1 and element[1] > -1 and element[2] > -1:
                        # This element is a node.
                        if sdfg != element[0]:
                            # No parent SDFG row present yet, print it.
                            string += row_format.format(
                                'SDFG (' + str(element[0]) + ')',
                                '', '', '', '', width=COLW
                            )
                        sdfg = element[0]
                        if state != element[1]:
                            # No parent state row present yet, print it.
                            string += row_format.format(
                                '| State (' + str(element[1]) + ')',
                                '', '', '', '', width=COLW
                            )
                        state = element[1]
                        element_label = '| | Node (' + str(element[2]) + ')'
                    elif element[0] > -1 and element[1] > -1:
                        # This element is a state.
                        if sdfg != element[0]:
                            # No parent SDFG row present yet, print it.
                            string += row_format.format(
                                'SDFG (' + str(element[0]) + ')',
                                '', '', '', '', width=COLW
                            )
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

                    string += row_format.format(
                        element_label,
                        np.min(runtimes),
                        np.mean(runtimes),
                        np.median(runtimes),
                        np.max(runtimes),
                        width=COLW
                    )
            string += ('-' * (COLW * 5)) + '\n'

        if len(self.counters) > 0:
            string += ('-' * (COUNTER_COLW * 2)) + '\n'
            string += ('{:<{width}}' * 2).format(
                'Counter', 'Value', width=COUNTER_COLW
            ) + '\n'
            string += ('-' * (COUNTER_COLW * 2)) + '\n'
            for counter in self.counters:
                string += counter_format.format(
                    counter,
                    self.counters[counter],
                    width=COUNTER_COLW
                )
            string += ('-' * (COUNTER_COLW * 2)) + '\n'

        return string
