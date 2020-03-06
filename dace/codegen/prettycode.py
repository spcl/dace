""" Code I/O stream that automates indentation and mapping of code to SDFG 
    nodes. """

from six import StringIO
from dace.config import Config


class CodeIOStream(StringIO):
    """ Code I/O stream that automates indentation and mapping of code to SDFG 
        nodes. """
    def __init__(self, base_indentation=0):
        super(CodeIOStream, self).__init__()
        self._indent = 0
        self._spaces = int(Config.get('compiler', 'indentation_spaces'))

    def write(self, contents, sdfg=None, state_id=None, node_id=None):
        # Delete single trailing newline, as this will be implicitly inserted
        # anyway
        if contents:
            if contents[-1] == "\n":
                lines = contents[:-1].split("\n")
            else:
                lines = contents.split('\n')
        else:
            lines = contents

        # If SDFG/state/node location is given, annotate this line
        if sdfg is not None:
            location_identifier = '  ////__DACE:%s' % sdfg.name
            if state_id is not None:
                location_identifier += ':' + str(state_id)
                if node_id is not None:
                    if not isinstance(node_id, list):
                        node_id = [node_id]
                    for i, nid in enumerate(node_id):
                        if not isinstance(nid, int):
                            node_id[i] = sdfg.nodes()[state_id].node_id(nid)
                    location_identifier += ':' + ','.join(
                        [str(nid) for nid in node_id])
        else:
            location_identifier = ''

        # Write each line separately
        for line in lines:
            opening_braces = line.count('{')
            closing_braces = line.count('}')
            brace_balance = opening_braces - closing_braces

            # Write line and then change indentation
            if brace_balance < 0:
                self._indent += brace_balance

            codeline = self._indent * self._spaces * ' ' + line.strip()

            # Location identifier is written at character 81 and on, find out
            # how many spaces we need to add for that
            loc_spaces = max(80 - len(codeline), 2)

            if location_identifier != '':
                super(CodeIOStream, self).write(codeline + loc_spaces * ' ' +
                                                location_identifier + '\n')
            else:  # avoid ending spaces (useful for OpenCL and multiline macros)
                super(CodeIOStream, self).write(codeline + '\n')
            if brace_balance > 0:
                self._indent += brace_balance

            # If indentation failed, warn user
            if self._indent < -1:
                super(CodeIOStream, self).write(
                    '///WARNING: Indentation failure! This probably ' +
                    'indicates an error in the SDFG.\n')
                self._indent = 0
