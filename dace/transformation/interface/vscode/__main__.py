import json
from argparse import ArgumentParser

def reapply_history_until(sdfg_json, index):
    """
    Rewind a given SDFG back to a specific point in its history by reapplying
    all transformations until a given index in its history to its original
    state.
    :param sdfg_json:  The SDFG to rewind.
    :param index:      Index of the last history item to apply.
    """
    # We lazy import DaCe, not to break cyclic imports, but to avoid any large
    # delays when booting in daemon mode.
    from dace.sdfg import SDFG

    sdfg = SDFG.from_json(sdfg_json)

    original_sdfg = sdfg.orig_sdfg
    history = sdfg.transformation_hist

    for i in range(index + 1):
        transformation = history[i]
        # FIXME: The appending should happen with the call to apply the pattern.
        # The way it currently stands, the callee must make sure that
        # append_transformation is called before apply_pattern, because the
        # original SDFG may be saved incorrectly otherwise. This is not ideal
        # and needs to be fixed.
        original_sdfg.append_transformation(transformation)
        transformation.apply_pattern(original_sdfg)

    new_sdfg = original_sdfg.to_json()
    return {
        'sdfg': new_sdfg,
    }

def apply_transformation(sdfg_json, transformation):
    # We lazy import DaCe, not to break cyclic imports, but to avoid any large
    # delays when booting in daemon mode.
    from dace.transformation.pattern_matching import Transformation
    from dace.sdfg import SDFG

    sdfg = SDFG.from_json(sdfg_json)

    revived_transformation = Transformation.from_json(transformation)
    # FIXME: The appending should happen with the call to apply the pattern. The
    # way it currently stands, the callee must make sure that
    # append_transformation is called before apply_pattern, because the original
    # SDFG may be saved incorrectly otherwise. This is not ideal and needs to be
    # fixed.
    sdfg.append_transformation(revived_transformation)
    revived_transformation.apply_pattern(sdfg)

    new_sdfg = sdfg.to_json()
    return {
        'sdfg': new_sdfg,
    }

def get_transformations(sdfg_json):
    # We lazy import DaCe, not to break cyclic imports, but to avoid any large
    # delays when booting in daemon mode.
    from dace.transformation.optimizer import SDFGOptimizer
    from dace.sdfg import SDFG

    sdfg = SDFG.from_json(sdfg_json)
    optimizer = SDFGOptimizer(sdfg)
    matches = optimizer.get_pattern_matches()

    transformations = []
    docstrings = {}
    for transformation in matches:
        transformations.append(transformation.to_json())
        docstrings[type(transformation).__name__] = transformation.__doc__

    return {
        'transformations': transformations,
        'docstrings': docstrings,
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
        request_json = request.get_json()
        return get_transformations(request_json['sdfg'])

    @daemon.route('/apply_transformation', methods=['POST'])
    def _apply_transformation():
        request_json = request.get_json()
        return apply_transformation(request_json['sdfg'],
                                    request_json['transformation'])

    @daemon.route('/reapply_history_until', methods=['POST'])
    def _reapply_history_until():
        request_json = request.get_json()
        return reapply_history_until(request_json['sdfg'],
                                     request_json['index'])

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
