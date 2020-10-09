# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Implementation of the performance instrumentation report. """

import json
import re
from typing import Dict, List


class InstrumentationReport(object):
    def __init__(self, filename: str):
        # Parse file
        match = re.match(r'.*report-(\d+)\.json', filename)
        self.name = match.groups()[0] if match is not None else 'N/A'
        with open(filename, 'r') as fp:
            self.entries: Dict[str, List[float]] = json.load(fp)

    def __repr__(self):
        return 'InstrumentationReport(name=%s)' % self.name

    def __str__(self):
        ent = '\n'.join(['  %s: %s' % (k, v) for k, v in self.entries.items()])
        return 'Report %s:\n%s' % (self.name, ent)
