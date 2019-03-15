import uuid


class AbstractSDFGNode:
    def __init__(self):
        self.uid = str(uuid.uuid4())
        self.label = None
        self.nodetype = "Unspecified"

    def set_nodetype(self, nodetype):
        self.nodetype = nodetype
        if self.label == None:
            self.label = nodetype

    def get_uid(self):
        return self.uid

    def set_label(self, label):
        self.label = label

    def get_label(self):
        return self.label

    def to_dot(self):
        dot = "\"" + self.uid + "\" ["
        if self.nodetype == "Map":
            dot += "shape=trapezium"
        if self.nodetype == "Unmap":
            dot += "shape=invtrapezium"
        if self.nodetype == "Array":
            dot += "shape=ellipse"
        if self.nodetype == "Tasklet":
            dot += "shape=octagon"
        if self.nodetype == "Confres":
            dot += "shape=invtriangle"
        if self.nodetype == "Stream":
            dot += "shape=ellipse, style=dashed"
        if self.nodetype == "StreamMap":
            dot += "shape=trapezium, style=dashed"
        if self.nodetype == "StreamUnmap":
            dot += "shape=invtrapezium, style=dashed"
        if self.nodetype == "Reduce":
            dot += "shape=invhouse"
        dot += ",label=\"" + str(self.label) + "\""
        dot += "];\n"
        return dot


class AbstractSDFGEdge:
    def __init__(self, tail, head):
        self.head = head
        self.tail = tail
        self.label = ""

    def get_head(self):
        return self.head

    def get_tail(self):
        return self.tail

    def set_label(self, label):
        self.label = label

    def get_label(self):
        return self.label

    def to_dot(self):
        dot = "\"" + self.tail.get_uid() + "\"" + " -> " + "\"" + \
              self.head.get_uid() + "\" [label=\"" + self.label + "\"];"
        return dot


class AbstractSDFG:
    def __init__(self):
        self.states = []
        self.interstate_edges = []
        self.nodes = []
        self.edges = []

    def to_dot(self):
        dot = "digraph G {\n"
        for node in self.nodes:
            dot += node.to_dot()
        for edge in self.edges:
            dot += edge.to_dot()
        for state in self.states:
            dot += state.to_dot()
        for insedge in self.interstate_edges:
            dot += insedge.to_dot()
        dot += "}\n"
        return dot

    def add_node(self, nodetype):
        node = AbstractSDFGNode()
        node.set_nodetype(nodetype)
        self.nodes.append(node)

    def add_edge(self, tailnode, headnode):
        t = self.find_node(tailnode)
        h = self.find_node(headnode)
        e = AbstractSDFGEdge(t, h)
        self.edges.append(e)

    def find_node(self, uid):
        for n in self.nodes:
            if n.get_uid() == uid:
                return n

    def find_edge(self, tail_uid, head_uid):
        for e in self.edges:
            if (e.get_tail().get_uid() == tail_uid) and \
               (e.get_head().get_uid() == head_uid):
                return e

    def delete_node(self, uid):
        for e in self.edges:
            t = e.get_tail()
            h = e.get_head()
            if (t.get_uid() == uid) or (h.get_uid() == uid):
                self.edges.remove(e)
        for n in self.nodes:
            if n.get_uid() == uid:
                self.nodes.remove(n)
