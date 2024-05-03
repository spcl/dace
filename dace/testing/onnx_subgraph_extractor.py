"""
A tool that extracts a subgraph up to a given node from an onnx file.
"""

import collections
import argparse
import onnx
from onnx import helper

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        " A tool that extracts a subgraph up to a given node from an onnx file. "
    )
    parser.add_argument("input", help="path to the input onnx file")
    parser.add_argument("output", help="path to the output onnx file")

    parser.add_argument(
        "target",
        help=
        "the node to extract. The subgraph computing this node will be extracted"
    )
    args = parser.parse_args()

    input_model = onnx.load(args.input)

    def get_node_idx(name):
        cands = [
            i for i, n in enumerate(input_model.graph.node) if n.name == name
        ]
        if len(cands) != 1:
            raise ValueError(
                f"Expected 1 node with name {name}, found {len(cands)}")
        return cands[0]

    g_inputs = {p.name: p for p in input_model.graph.input}
    g_outputs = {p.name: p for p in input_model.graph.output}
    g_inits = {p.name: p for p in input_model.graph.initializer}
    g_vinfs = {p.name: p for p in input_model.graph.value_info}

    state = dict(inputs={}, vinfs={}, outputs={}, inits={})

    node_queue = collections.deque([get_node_idx(args.target)])
    added_nodes = set()
    while len(node_queue) > 0:
        idx = node_queue.popleft()
        if idx in added_nodes:
            continue
        added_nodes.add(idx)
        node = input_model.graph.node[idx]
        print(f"extracting {node.name}")

        for inp_name in node.input:
            if inp_name in set(state["inputs"]).union(state["vinfs"]).union(
                    state["inits"]):
                continue

            if inp_name in g_inputs:
                # copy this input
                state["inputs"][inp_name] = g_inputs[inp_name]
            elif inp_name in g_inits:
                state["inits"][inp_name] = g_inits[inp_name]
            elif inp_name in g_vinfs:
                # find the node that produces this, and copy add it to the queue
                cands = [
                    i for i, n in enumerate(input_model.graph.node)
                    if inp_name in n.output
                ]
                if len(cands) != 1:
                    raise ValueError(
                        f"Expected 1 node with input {inp_name}, found {len(cands)}"
                    )
                node_queue.append(cands[0])
            else:
                raise ValueError(
                    f"could not handle input {inp_name} of node {node.name}")

        for outp_name in node.output:
            # also copy the vinf
            if outp_name in g_vinfs:
                state["vinfs"][outp_name] = g_vinfs[outp_name]
            elif outp_name in g_outputs:
                state["outputs"][outp_name] = g_outputs[outp_name]

    output_graph = helper.make_graph(
        [input_model.graph.node[i] for i in sorted(added_nodes)],
        "subgraph",
        inputs=list(state["inputs"].values()),
        outputs=list(state["outputs"].values()),
        initializer=list(state["inits"].values()),
        value_info=list(state["vinfs"].values()))
    onnx.checker.check_graph(output_graph)
    output_model = helper.make_model(output_graph, producer_name="python-api")
    onnx.checker.check_model(output_model, full_check=True)
    onnx.save(output_model, args.output)
