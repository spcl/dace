""" Implementation of the performance instrumentation report. """

import json
import re
from typing import Dict, List


class InstrumentationReport(object):
    def __init__(self, filename: str):
        # Parse file
        self.name = re.match(r'.*report-(\d+)\.json', filename).groups()[0]
        with open(filename, 'r') as fp:
            self.entries: Dict[str, List[float]] = json.load(fp)

    def __repr__(self):
        return 'InstrumentationReport(name=%s)' % self.name

    def __str__(self):
        return 'Report %s:\n%s' % (self.name, self.entries)
