import argparse
import json
import numpy as np
import os


COLW = 15


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('path',
                        help='Path to the file containing the report')

    args = parser.parse_args()

    path = os.path.abspath(args.path)
    if not os.path.isfile(path):
        print(path, 'does not exist or isn\'t a regular file, aborting.')
        exit(1)

    element_map = {}
    sdfg_hash = ''

    with open(path) as report_file:
        report = json.load(report_file)

        if 'traceEvents' not in report or 'sdfgHash' not in report:
            print(path, 'isn\'t a valid SDFG instrumentation report, aborting.')
            exit(1)

        events = report['traceEvents']
        sdfg_hash = report['sdfgHash']

        for event in events:
            if 'ph' in event and event['ph'] == 'X':
                if 'dur' in event:
                    uuid = get_event_uuid(event)
                    if uuid not in element_map:
                        element_map[uuid] = []

                    element_map[uuid].append(event['dur'] / 1000)

    element_list = list(element_map.keys())
    element_list.sort()

    row_format = '{:<{width}}' * 5

    print('Instrumentation summary')
    print('SDFG Hash:', sdfg_hash)

    print('-' * (COLW * 5))
    print(('{:<{width}}' * 2).format('Element', 'Runtime (ms)', width=COLW))
    print(row_format.format('', 'Min', 'Mean', 'Median', 'Max', width=COLW))
    print('-' * (COLW * 5))

    sdfg = -1
    state = -1

    for element in element_list:
        runtimes = element_map[element]
        if len(runtimes) > 0:
            element_label = ''
            if element[0] > -1 and element[1] > -1 and element[2] > -1:
                # This element is a node.
                if sdfg != element[0]:
                    # No parent SDFG row present yet, print it.
                    print(row_format.format('SDFG (' + str(element[0]) + ')',
                        '', '', '', '', width=COLW))
                sdfg = element[0]
                if state != element[1]:
                    # No parent state row present yet, print it.
                    print(row_format.format('| State (' + str(element[1]) + ')',
                        '', '', '', '', width=COLW))
                state = element[1]
                element_label = '| | Node (' + str(element[2]) + ')'
            elif element[0] > -1 and element[1] > -1:
                # This element is a state.
                if sdfg != element[0]:
                    # No parent SDFG row present yet, print it.
                    print(row_format.format('SDFG (' + str(element[0]) + ')',
                        '', '', '', '', width=COLW))
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

            print(row_format.format(
                element_label,
                np.min(runtimes),
                np.mean(runtimes),
                np.median(runtimes),
                np.max(runtimes),
                width=COLW
            ))
    print('-' * (COLW * 5))