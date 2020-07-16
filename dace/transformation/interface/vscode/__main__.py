import json
from argparse import ArgumentParser

def apply_transformation(sdfg, transformation):
    # We lazy import DaCe, not to break cyclic imports, but to avoid any large
    # delays when booting in daemon mode.
    from dace.transformation.pattern_matching import Transformation
    from dace.sdfg import SDFG

    sdfg_json = json.loads(sdfg)
    sdfg_object = SDFG.from_json(sdfg_json)

    revived_transformation = Transformation.from_json(transformation)
    revived_transformation.apply_pattern(sdfg_object)

    new_sdfg = sdfg_object.to_json()
    return {
        'sdfg': new_sdfg
    }

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
        transformations.append(transformation.to_json())

    return {
        'transformations': transformations,
    }

def run_daemon():
    from flask import Flask, request

    daemon = Flask('dace.transformation.interface.vscode')
    daemon.config['DEBUG'] = False

    @daemon.route('/', methods=['GET'])
    def _root():
        return 'success!'

    @daemon.route('/transformations', methods=['POST'])
    def _get_transformations():
        return get_transformations(request.get_json())

    @daemon.route('/apply_transformation', methods=['POST'])
    def _apply_transformation():
        request_json = request.get_json()
        return apply_transformation(request_json['sdfg'],
                                    request_json['transformation'])

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