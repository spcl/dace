import copy
import html
from dace import data, memlet
from dace.graph import graph as gr, edges


def draw_edge_explicit(srcName, dstName, edge, sdfg, graph, **extraOpts):
    opts = {}
    if isinstance(edge.data, memlet.Memlet):
        if getattr(edge.data, '__label__', False):
            opts["label"] = edge.data.__label__(sdfg, graph)
        else:
            opts["label"] = str(edge.data)
        if edge.data.wcr is not None:
            opts['style'] = 'dashed'
    elif isinstance(edge.data, edges.InterstateEdge):
        opts.update(edge.data.dotOpts)
    # Unhandled properties
    elif edge.data != None:
        raise ValueError("Unhandled edge: " + str(edge.data))
    if extraOpts:
        opts.update(extraOpts)  # Custom options will overwrite default

    if isinstance(edge, gr.MultiConnectorEdge):
        sconn = '' if edge.src_conn is None else (':' + edge.src_conn)
        dconn = '' if edge.dst_conn is None else (':' + edge.dst_conn)
    else:
        sconn = ''
        dconn = ''

    return ("\"{}\"{sconn} -> \"{}\"{dconn}".format(
        srcName, dstName, sconn=sconn, dconn=dconn) + ((" [" + ", ".join(
            ["{}=\"{}\"".format(key, value)
             for key, value in opts.items()]) + "];") if opts else ";"))


def draw_edge(sdfg, graph, edge, **extraOpts):
    srcName = 's%d_%d' % (sdfg.node_id(graph), graph.node_id(edge.src))
    dstName = 's%d_%d' % (sdfg.node_id(graph), graph.node_id(edge.dst))

    return draw_edge_explicit(srcName, dstName, edge, sdfg, graph)


def draw_interstate_edge(sdfg, src_graph, dst_graph, edge, **extraOpts):
    srcName = 's%d_%d' % (sdfg.node_id(src_graph), src_graph.node_id(edge.src))
    dstName = 's%d_%d' % (sdfg.node_id(dst_graph), dst_graph.node_id(edge.dst))
    if isinstance(edge, gr.MultiConnectorEdge):
        if edge.src_conn is not None:
            srcName += '@' + edge.src_conn
        if edge.dst_conn is not None:
            dstName += '@' + edge.dst_conn

    return draw_edge_explicit(srcName, dstName, edge, sdfg, src_graph,
                              **extraOpts)


def draw_interstate_edge_by_name(srcName, dstName, edge, sdfg, src_graph,
                                 **extraOpts):
    return draw_edge_explicit(srcName, dstName, edge, sdfg, src_graph,
                              **extraOpts)


def draw_node(sdfg, graph, obj, **kwargs):
    name = 's%d_%d' % (sdfg.node_id(graph), graph.node_id(obj))
    if getattr(obj, '__label__', False):
        opts = {"label": obj.__label__(sdfg, graph)}
    else:
        opts = {"label": str(obj)}
    opts.update(kwargs)
    opts["label"] = "\"{}\"".format(opts["label"])

    if 'fillcolor' not in opts:
        opts['fillcolor'] = '"#ffffff"'
        if 'style' not in opts:
            opts['style'] = 'filled'
        else:
            opts['style'] = '"filled,%s"' % opts['style']

    ############################################
    if getattr(obj, 'in_connectors', False) != False and len(
            obj.in_connectors) + len(obj.out_connectors) > 0:
        # Header
        code = '{name} [label=<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="-4" CELLPADDING="0">'
        code = code.format(name=name)
        # Input connectors
        code += '<TR><TD BORDER="0"><TABLE BORDER="0" CELLBORDER="0" CELLSPACING="-8" CELLPADDING="0"><TR>'
        code += '<TD WIDTH="20"></TD>'
        connector_code = []
        for conn in sorted(obj.in_connectors):
            connector_code.append(
                '<TD PORT="{conn}" BORDER="1" CELLPADDING="1"><FONT POINT-SIZE="10">{conn}</FONT></TD>'.
                format(conn=conn))
        code += '<TD WIDTH="20"></TD>'.join(connector_code)
        code += '<TD WIDTH="20"></TD></TR></TABLE></TD></TR>'

        # Contents
        html_label = html.escape(opts['label'][1:-1])
        code += '<TR><TD BORDER="0" CELLPADDING="15" COLOR="black">{label}</TD></TR>'.format(
            label=html_label)

        # Output connectors
        code += '<TR><TD BORDER="0"><TABLE BORDER="0" CELLBORDER="0" CELLSPACING="-8" CELLPADDING="0"><TR>'
        code += '<TD WIDTH="20"></TD>'
        connector_code = []
        for conn in sorted(obj.out_connectors):
            connector_code.append(
                '<TD PORT="{conn}" BORDER="1" CELLPADDING="1"><FONT POINT-SIZE="10">{conn}</FONT></TD>'.
                format(conn=conn))
        code += '<TD WIDTH="20"></TD>'.join(connector_code)
        code += '<TD WIDTH="20"></TD></TR></TABLE></TD></TR>'

        # Footer
        code += '</TABLE>>'

        filtered_opts = {k: v for k, v in opts.items() if k != 'label'}
        if len(filtered_opts.items()) > 0:
            ostr = ", ".join([
                str(key) + "=" + str(val)
                for key, val in filtered_opts.items()
            ])
            code += ', ' + ostr
        code += '];\n'

        return code
    ############################################

    return "\"{}\" [{}];".format(
        name,
        ", ".join([str(key) + "=" + str(val) for key, val in opts.items()]))


def draw_invisible_node(name, **kwargs):
    opts = dict(label='\"\"', style="invisible")
    opts.update(kwargs)
    return "\"{}\" [{}];".format(
        name,
        ", ".join([str(key) + "=" + str(val) for key, val in opts.items()]))


def draw_graph(sdfg, graph, standalone=True):
    """ Creates a graphviz dot file from a networkx graph input.

        If standalone is set, return a full dot string including header and footer.
    """
    state_id = sdfg.node_id(graph)
    sdfg = copy.deepcopy(sdfg)
    graph = sdfg.nodes()[state_id]

    sdict = graph.scope_dict()
    sdict_children = graph.scope_dict(True)

    # Omit collapsed nodes out of nodes to draw
    def is_collapsed(node):
        scope = sdict[node]
        while scope is not None:
            if scope.is_collapsed:
                return True
            scope = sdict[scope]
        return False

    nodes_to_draw = set(
        node for node in graph.nodes() if not is_collapsed(node))

    # Collect edges to draw for collapsed nodes (we also need edges coming out of scope exits)
    nodes_for_edges = set()
    nodes_for_edges.update(nodes_to_draw)

    def add_exit_nodes(scope):
        for node in sdict_children[scope]:
            if node in sdict_children and node.is_collapsed:
                nodes_for_edges.add(graph.exit_nodes(node)[0])
            elif node in sdict_children:
                add_exit_nodes(node)

    add_exit_nodes(None)

    edges_to_draw = set(
        e for e in graph.edges()
        if e.src in nodes_for_edges and e.dst in nodes_for_edges)

    # Take care of scope entry connectors
    for node in nodes_to_draw:
        if node in sdict_children and node.is_collapsed:
            node._out_connectors.clear()

    # Take care of scope exit edges and connectors
    for e in edges_to_draw:
        if e.src in nodes_for_edges and e.src not in nodes_to_draw:
            newsrc = sdict[e.src]
            if newsrc is None:
                continue
            e._src = newsrc
            newsrc._out_connectors.add(e.src_conn)

    nodes = [x.draw_node(sdfg, graph) for x in nodes_to_draw]
    edges = [draw_edge(sdfg, graph, e) for e in edges_to_draw]

    if not standalone:
        return nodes, edges

    return "digraph DaCe {{\n    {}\n}}".format("\n    ".join(nodes + edges))
