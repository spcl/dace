import json
from argparse import ArgumentParser

def get_transformations(sdfg):
    # We lazy import DaCe, not to break cyclic imports, but to avoid any large
    # delays when booting in daemon mode.
    from dace.transformation.optimizer import SDFGOptimizer
    from dace.sdfg import SDFG

    sdfg_json = json.loads(sdfg)
    sdfg_object = SDFG.from_json(sdfg_json)
    optimizer = SDFGOptimizer(sdfg_object)
    matches = optimizer.get_pattern_matches()

    transformations = []
    for transformation in matches:
        parameters = []
        node_ids = []
        if transformation is not None:
            sdfg_id = transformation.sdfg_id
            state_id = transformation.state_id
            nodes = list(transformation.subgraph.values())
            for node in nodes:
                node_ids.append([sdfg_id, state_id, node])

        transformations.append({
            'label': transformation.__str__(),
            'parameters': parameters,
            'affected_nodes': node_ids,
            'children': [],
        })
        # if set(transformation.subgraph.values()) & viewed_nodes:
        #     pass

    return {
        'transformations': transformations,
    }

def run_daemon():
    from flask import Flask, request

    daemon = Flask('dace.transformation.interface.vscode')
    daemon.config['DEBUG'] = False

    @daemon.route('/', methods=['GET'])
    def root():
        return 'success!'

    @daemon.route('/transformations', methods=['POST'])
    def transformations():
        sdfg = request.get_json()
        transformation_json = get_transformations(sdfg)
        return transformation_json

    daemon.run()

if __name__ == '__main__':
    parser = ArgumentParser()

    '''
    parser.add_argument('-d',
                        '--daemon',
                        action='store_true',
                        help='Run as a daemon')
                        '''

    '''
    parser.add_argument('-p',
                        '--port',
                        action='store',
                        type='int',
                        help='The port to listen on')
                        '''

    parser.add_argument('-t',
                        '--transformations',
                        action='store_true',
                        help='Get applicable transformations for an SDFG')

    args = parser.parse_args()

    if (args.transformations):
        get_transformations(None)
    else:
        run_daemon()