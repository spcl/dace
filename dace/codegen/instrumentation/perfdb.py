import re


class PerfMetaInfo:
    """ Class dedicated to keep meta information about the generated code, in
        particular line numbers. """

    def __init__(self):
        self.nodes = dict()  # Maps nodes to their strings
        self.lines = dict()  # Maps nodes to their line number

    def add_node(self, node, string):
        self.nodes[node] = string

    def has_node(self, node):
        return node in self.nodes.keys()

    def resolve(self, codestr: str):
        """ Maps all entries in self.node to line numbers """
        index = 0
        line = 1
        print("self.nodes: %s\ntype: %s" % (self.nodes, type(self.nodes)))
        for key, value in self.nodes.items():
            pos = codestr.find(value, index)
            if pos == -1:
                # We will not accept this. This should only ever occur if some
                # part of the program pretty-prints code.
                assert False
            sublines = codestr.count('\n', index, pos)
            line += sublines
            index = pos
            # We store the current line back to self.lines
            self.lines[key] = line

    def analyze(self, vectorizer_output: str):
        """ Checks if a certain operation or a segment within a region of an
            operation was vectorized. """
        # We only match calls originating from ./src/cpu/*, but it might still
        # include some of the instrumentation. Consider running this on
        # non-instrumented code instead
        data = re.findall(
            r".*?src/cpu/(?P<file>[^:]*):(?P<line>[\d]*):(?P<col>[\d]*): (?P<msg>[^\n]*)",
            vectorizer_output)

        print("data is:\n%s" % data)

        print("Node information is\n%s\n" % self.nodes)
        print("Line information is\n%s\n" % self.lines)

        ret = dict(
        )  # We return a dict of node -> [(file, line, col, Message)]

        first = True
        tmp = (None, None)
        for key, value in self.lines.items():
            # We now find for each key the value of their respective start
            # (exception: MapExit, where the end counts)
            # Then, we associate the message to that key
            if not first:
                prevkey, prevval = tmp
                for file, line, col, message in data:
                    if int(prevval) <= int(line) and int(line) < int(value):
                        # Valid entry
                        if not (prevkey in ret.keys()):
                            ret[prevkey] = list()
                        ret[prevkey].append((file, line, col, message))
            else:
                first = False

            tmp = (key, value)

        # For the last entry:
        prevkey, prevval = tmp
        if prevkey is not None:
            for file, line, col, message in data:
                if int(prevval) <= int(line):
                    # Valid entry
                    if not (prevkey in ret.keys()):
                        ret[prevkey] = list()
                    ret[prevkey].append((file, line, col, message))

        print("ret:\n%s" % ret)

        return ret


# Singleton structures
class PerfMetaInfoStatic:
    info = PerfMetaInfo()
