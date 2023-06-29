# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Implementation of the performance instrumentation report. """

from dataclasses import dataclass
import json
import numpy as np
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from io import StringIO

from collections import defaultdict

UUIDType = Tuple[int, int, int]


def _uuid_to_dict(uuid: UUIDType) -> Dict[str, int]:
    result = {}
    if uuid[0] != -1:
        result['sdfg_id'] = uuid[0]
    if uuid[1] != -1:
        result['state_id'] = uuid[1]
    if uuid[2] != -1:
        result['id'] = uuid[2]

    return result


@dataclass
class DurationEvent:
    """
    Instrumentation report event of a duration (e.g., execution time).
    """
    name: str  #: Event name
    category: str  #: Category
    uuid: UUIDType  #: Unique locator for SDFG/state/node/edge
    timestamp: int  #: Beginning time (in microseconds)
    duration: float  #: Duration (in microseconds)
    pid: int  #: Process ID
    tid: int = -1  #: Thread ID (or -1 if not applicable)
    additional_info: Optional[Dict[str, Any]] = None  #: More arguments in the event

    def save(self) -> Dict[str, Any]:
        info = self.additional_info or {}
        args = {**info, **_uuid_to_dict(self.uuid)}
        return dict(name=self.name,
                    cat=self.category,
                    ph='X',
                    ts=self.timestamp,
                    dur=self.duration,
                    pid=self.pid,
                    tid=self.tid,
                    args=args)


@dataclass
class CounterEvent:
    """
    Instrumentation report event of a counter (e.g., performance counters).
    """
    name: str  #: Event name
    category: str  #: Category
    uuid: UUIDType  #: Unique locator for SDFG/state/node/edge
    timestamp: int  #: Event time (in microseconds)
    counters: Dict[str, float]  #: Counter names and their values
    pid: int  #: Process ID
    tid: int = -1  #: Thread ID (or -1 if not applicable)

    def save(self) -> Dict[str, Any]:
        args = {**self.counters, **_uuid_to_dict(self.uuid)}
        return dict(name=self.name, cat=self.category, ph='C', ts=self.timestamp, pid=self.pid, tid=self.tid, args=args)


class InstrumentationReport(object):
    """
    An object that represents a DaCe program instrumentation report.
    Such reports may include runtimes of all or parts of an SDFG, as well as performance counters.

    Instrumentation reports are stored as JSON files, in the Chrome Tracing format.
    """
    @staticmethod
    def get_event_uuid_and_other_info(event) -> Tuple[UUIDType, Dict[str, Any]]:
        uuid = (-1, -1, -1)
        other_info = {}
        if 'args' in event:
            args = event['args']
            if 'sdfg_id' in args and args['sdfg_id'] is not None:
                uuid = (args['sdfg_id'], -1, -1)
                if 'state_id' in args and args['state_id'] is not None:
                    uuid = (uuid[0], args['state_id'], -1)
                    if 'id' in args and args['id'] is not None:
                        uuid = (uuid[0], uuid[1], args['id'])
            other_info = {k: v for k, v in args.items() if k not in ('sdfg_id', 'state_id', 'id')}
        return uuid, other_info

    def __init__(self, filename: str):
        self.name = None

        # Raw events
        self.events: List[Union[DurationEvent, CounterEvent]] = []

        # Summarized fields:
        # UUID -> Name -> Thread ID -> Times
        self.durations: Dict[UUIDType,
                             Dict[str, Dict[int,
                                            List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        # UUID -> Name -> Counter -> Thread ID -> Values
        self.counters: Dict[UUIDType, Dict[str, Dict[str, Dict[int, List[float]]]]] = defaultdict(dict)

        self._sortcat = None
        self._sortdesc = False
        self.sdfg_hash: str = ''

        if not filename:  # Empty instrumentation report
            return

        # Parse file
        match = re.match(r'.*report-(\d+)\.json', filename)
        self.name = match.groups()[0] if match is not None else 'N/A'
        self.filepath = filename

        with open(filename, 'r') as fp:
            report = json.load(fp)

            if 'traceEvents' not in report or 'sdfgHash' not in report:
                print(filename, 'is not a valid SDFG instrumentation report!')
                return

            # Parse events from file
            self.sdfg_hash: str = report['sdfgHash']
            for event in report['traceEvents']:
                if "ph" not in event:
                    continue
                uuid, other_info = self.get_event_uuid_and_other_info(event)
                if event['ph'] == 'X':  # Duration event
                    self.events.append(
                        DurationEvent(event['name'], event['cat'], uuid, event['ts'], event['dur'], event['pid'],
                                      event['tid'], other_info))
                elif event['ph'] == 'C':  # Counter event
                    self.events.append(
                        CounterEvent(event['name'], event['cat'], uuid, event['ts'], other_info, event['pid'],
                                     event['tid']))

        # Summarize events for printouts
        self.process_events()

    def process_events(self):
        """
        Summarizes the events in the report into dictionaries.
        """
        for event in self.events:
            name = event.name
            uuid = event.uuid
            tid = event.tid

            if isinstance(event, DurationEvent):
                # Time
                if uuid not in self.durations:
                    self.durations[uuid] = {}
                if name not in self.durations[uuid]:
                    self.durations[uuid][name] = defaultdict(list)

                self.durations[uuid][name][tid].append(event.duration / 1000)

            elif isinstance(event, CounterEvent):
                # Counter
                if uuid not in self.counters:
                    self.counters[uuid] = {}
                if name not in self.counters[uuid]:
                    self.counters[uuid][name] = defaultdict(list)

                ctrs = event.counters
                for counter, value in ctrs.items():
                    if counter not in self.counters[uuid][name]:
                        self.counters[uuid][name][counter] = defaultdict(list)

                    self.counters[uuid][name][counter][tid].append(value)

    def __repr__(self):
        return 'InstrumentationReport(name=%s)' % self.name

    def sortby(self, column: str, ascending: bool = False):
        if (column and column.lower() not in ('counter', 'value', 'min', 'max', 'mean', 'median')):
            raise ValueError('Only Counter, Value, Min, Max, Mean, Median are supported')
        self._sortcat = column if column is None else column.lower()
        self._sortdesc = not ascending

    def _get_runtimes_string(self,
                             label,
                             runtimes,
                             element,
                             sdfg,
                             state,
                             string,
                             row_format,
                             colw,
                             with_element_heading=True,
                             title=''):
        indent = ''
        if len(runtimes) > 0:
            element_label = ''
            if element[0] > -1 and element[1] > -1 and element[2] > -1:
                # This element is a node.
                if sdfg != element[0]:
                    # No parent SDFG row present yet, print it.
                    string += row_format.format('SDFG (' + str(element[0]) + ')', '', '', '', '', width=colw)
                sdfg = element[0]
                if state != element[1]:
                    # No parent state row present yet, print it.
                    string += row_format.format('|-State (' + str(element[1]) + ')', '', '', '', '', width=colw)
                state = element[1]
                if title:
                    element_label = '| |-Node (' + str(element[2]) + ', ' + title + ')'
                else:
                    element_label = '| |-Node (' + str(element[2]) + ')'
                indent = '| | |'
            elif element[0] > -1 and element[1] > -1:
                # This element is a state.
                if sdfg != element[0]:
                    # No parent SDFG row present yet, print it.
                    string += row_format.format('SDFG (' + str(element[0]) + ')', '', '', '', '', width=colw)
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
                string += row_format.format(element_label, '', '', '', '', width=colw)

            string += row_format.format(indent + label + ':', '', '', '', '', width=colw)
            string += row_format.format(indent,
                                        '%.3f' % np.min(runtimes),
                                        '%.3f' % np.mean(runtimes),
                                        '%.3f' % np.median(runtimes),
                                        '%.3f' % np.max(runtimes),
                                        width=colw)

        return string, sdfg, state

    def _get_counters_string(self,
                             counter,
                             label,
                             values,
                             element,
                             sdfg,
                             state,
                             string,
                             row_format,
                             colw,
                             with_element_heading=True):
        indent = ''
        if len(values) > 0:
            element_label = ''
            if element[0] > -1 and element[1] > -1 and element[2] > -1:
                # This element is a node.
                if sdfg != element[0]:
                    # No parent SDFG row present yet, print it.
                    string += row_format.format('SDFG (' + str(element[0]) + ')', '', '', '', '', width=colw)
                sdfg = element[0]
                if state != element[1]:
                    # No parent state row present yet, print it.
                    string += row_format.format('|-State (' + str(element[1]) + ')', '', '', '', '', width=colw)
                state = element[1]
                element_label = '| |-Node (' + str(element[2]) + ')'
                indent = '| | |'
            elif element[0] > -1 and element[1] > -1:
                # This element is a state.
                if sdfg != element[0]:
                    # No parent SDFG row present yet, print it.
                    string += row_format.format('SDFG (' + str(element[0]) + ')', '', '', '', '', width=colw)
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
                string += row_format.format(element_label, '', '', '', '', width=colw)
                string += row_format.format(f"{counter}", '', '', '', '', width=colw)

            string += row_format.format(indent + "|" + label + ':', '', '', '', '', width=colw)
            string += row_format.format(indent,
                                        np.min(values),
                                        '%.2f' % np.mean(values),
                                        '%.2f' % np.median(values),
                                        np.max(values),
                                        width=colw)

        return string, sdfg, state

    def getkey(self, element):
        events = self.durations[element]
        result = []
        for event in events.keys():
            runtimes = events[event]
            result.extend(runtimes)

        result = np.array(result)
        if self._sortcat == 'min':
            return np.min(result)
        elif self._sortcat == 'max':
            return np.max(result)
        elif self._sortcat == 'mean':
            return np.mean(result)
        else:  # if self._sortcat == 'median':
            return np.median(result)

    def __str__(self):
        COLW = 15
        row_format = ('{:<{width}}' * 5) + '\n'

        string = 'Instrumentation report\n'
        string += 'SDFG Hash: ' + self.sdfg_hash + '\n'

        if len(self.durations) > 0:
            string += ('-' * (COLW * 5)) + '\n'
            string += ('{:<{width}}' * 2).format('Element', 'Runtime (ms)', width=COLW) + '\n'
            string += row_format.format('', 'Min', 'Mean', 'Median', 'Max', width=COLW)
            string += ('-' * (COLW * 5)) + '\n'

            sdfg = -1
            state = -1

            element_list = list(self.durations.keys())
            element_list.sort()
            if self._sortcat in ('min', 'mean', 'median', 'max'):
                element_list = sorted(element_list, key=self.getkey, reverse=self._sortdesc)

            for element in element_list:
                events = self.durations[element]
                for event in events:
                    times = events[event]
                    with_element_heading = True
                    for tid in times:
                        runtimes = times[tid]
                        if tid >= 0:
                            label = f"Thread {tid}"
                        else:
                            label = ""

                        string, sdfg, state = self._get_runtimes_string(label, runtimes, element, sdfg, state, string,
                                                                        row_format, COLW, with_element_heading, event)

                        with_element_heading = False

                string += ('-' * (COLW * 5)) + '\n'

        if len(self.counters) > 0:
            string += ('-' * (COLW * 5)) + '\n'
            string += ('{:<{width}}' * 2).format('Element', 'Counter', width=COLW) + '\n'
            string += row_format.format('', 'Min', 'Mean', 'Median', 'Max', width=COLW)
            string += ('-' * (COLW * 5)) + '\n'

            sdfg = -1
            state = -1

            for element, events in self.counters.items():
                for event, counters in events.items():
                    for counter, values in counters.items():
                        with_element_heading = True
                        for tid in values:
                            thread_values = values[tid]
                            if tid >= 0:
                                label = f"Thread {tid}"
                            else:
                                label = ""

                            string, sdfg, state = self._get_counters_string(counter, label, thread_values, element,
                                                                            sdfg, state, string, row_format, COLW,
                                                                            with_element_heading)

                            with_element_heading = False

                string += ('-' * (COLW * 5)) + '\n'

        return string

    def as_csv(self) -> Tuple[str, str]:
        """
        Generates a CSV version of the report.        

        :return: A tuple of two strings: (durations CSV, counters CSV).
        """

        durations_csv, counters_csv = StringIO(), StringIO()

        # Create durations CSV
        if len(self.durations) > 0:
            durations_csv.write('Name,SDFG,State,Node,Thread,Count,MinMS,MeanMS,MedianMS,MaxMS\n')

            for element, events in self.durations.items():
                for name, times in events.items():
                    for tid, runtimes in times.items():
                        sdfg, state, node = element
                        nptimes = np.array(runtimes)
                        cnt = len(runtimes)
                        mint, meant, mediant, maxt = np.min(nptimes), np.mean(nptimes), np.median(nptimes), np.max(
                            nptimes)
                        durations_csv.write(f'{name},{sdfg},{state},{node},{tid},{cnt},{mint},{meant},{mediant},{maxt}\n')

        # Create counters CSV
        if len(self.counters) > 0:
            counters_csv.write('Counter,Name,SDFG,State,Node,Thread,Count,Min,Mean,Median,Max\n')

            for element, events in self.counters.items():
                for name, counters in events.items():
                    for ctrname, ctrvalues in counters.items():
                        for tid, values in ctrvalues.items():
                            sdfg, state, node = element
                            npval = np.array(values)
                            cnt = len(values)
                            mint, meant, mediant, maxt = np.min(npval), np.mean(npval), np.median(npval), np.max(npval)
                            counters_csv.write(
                                f'{ctrname},{name},{sdfg},{state},{node},{tid},{cnt},{mint},{meant},{mediant},{maxt}\n')

        return durations_csv.getvalue(), counters_csv.getvalue()

    def save(self, filename: str) -> None:
        """
        Stores an instrumentation report to a file in the Chrome Tracing JSON format.

        :param filename: The file name to store.
        """

        report_json = {}
        report_json['sdfgHash'] = self.sdfg_hash
        report_json['traceEvents'] = [ev.save() for ev in self.events]
        with open(filename, 'w') as fp:
            json.dump(report_json, fp)
