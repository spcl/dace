# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import json
import traceback
import sys
from argparse import ArgumentParser

def get_exception_message(exception):
    return '%s: %s' % (type(exception).__name__, exception)

def load_sdfg_from_json(json):
    # We lazy import SDFGs, not to break cyclic imports, but to avoid any large
    # delays when booting in daemon mode.
    from dace.sdfg import SDFG

    if 'error' in json:
        message = ''
        if ('message' in json['error']):
            message = json['error']['message']
        error = {
            'error': {
                'message': 'Invalid SDFG provided',
                'details': message,
            }
        }
        sdfg = None
    else:
        try:
            sdfg = SDFG.from_json(json)
            error = None
        except Exception as e:
            print(traceback.format_exc(), file=sys.stderr)
            sys.stderr.flush()
            error = {
                'error': {
                    'message': 'Failed to parse the provided SDFG',
                    'details': get_exception_message(e),
                },
            }
            sdfg = None
    return {
        'error': error,
        'sdfg': sdfg,
    }

def reapply_history_until(sdfg_json, index):
    """
    Rewind a given SDFG back to a specific point in its history by reapplying
    all transformations until a given index in its history to its original
    state.
    :param sdfg_json:  The SDFG to rewind.
    :param index:      Index of the last history item to apply.
    """
    loaded = load_sdfg_from_json(sdfg_json)
    if loaded['error'] is not None:
        return loaded['error']
    sdfg = loaded['sdfg']

    original_sdfg = sdfg.orig_sdfg
    history = sdfg.transformation_hist

    for i in range(index + 1):
        transformation = history[i]
        # FIXME: The appending should happen with the call to apply the pattern.
        # The way it currently stands, the callee must make sure that
        # append_transformation is called before apply_pattern, because the
        # original SDFG may be saved incorrectly otherwise. This is not ideal
        # and needs to be fixed.
        try:
            original_sdfg.append_transformation(transformation)
            transformation.apply_pattern(original_sdfg)
        except Exception as e:
            print(traceback.format_exc(), file=sys.stderr)
            sys.stderr.flush()
            return {
                'error': {
                    'message': 'Failed to play back the transformation history',
                    'details': get_exception_message(e),
                },
            }

    new_sdfg = original_sdfg.to_json()
    return {
        'sdfg': new_sdfg,
    }

def apply_transformation(sdfg_json, transformation):
    # We lazy import DaCe, not to break cyclic imports, but to avoid any large
    # delays when booting in daemon mode.
    from dace.transformation.pattern_matching import Transformation

    loaded = load_sdfg_from_json(sdfg_json)
    if loaded['error'] is not None:
        return loaded['error']
    sdfg = loaded['sdfg']

    try:
        revived_transformation = Transformation.from_json(transformation)
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        sys.stderr.flush()
        return {
            'error': {
                'message': 'Failed to parse the applied transformation',
                'details': get_exception_message(e),
            },
        }
    # FIXME: The appending should happen with the call to apply the pattern. The
    # way it currently stands, the callee must make sure that
    # append_transformation is called before apply_pattern, because the original
    # SDFG may be saved incorrectly otherwise. This is not ideal and needs to be
    # fixed.
    try:
        sdfg.append_transformation(revived_transformation)
        revived_transformation.apply_pattern(sdfg)
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        sys.stderr.flush()
        return {
            'error': {
                'message': 'Failed to apply the transformation to the SDFG',
                'details': get_exception_message(e),
            },
        }

    new_sdfg = sdfg.to_json()
    return {
        'sdfg': new_sdfg,
    }

def get_transformations(sdfg_json):
    # We lazy import DaCe, not to break cyclic imports, but to avoid any large
    # delays when booting in daemon mode.
    from dace.transformation.optimizer import SDFGOptimizer

    loaded = load_sdfg_from_json(sdfg_json)
    if loaded['error'] is not None:
        return loaded['error']
    sdfg = loaded['sdfg']

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
    from logging.config import dictConfig
    from flask import Flask, request

    # Move Flask's logging over to stdout, because stderr is used for error
    # reporting. This was taken from
    # https://stackoverflow.com/questions/56905756
    dictConfig({
        'version': 1,
        'formatters': {'default': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        }},
        'handlers': {'wsgi': {
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
            'formatter': 'default',
        }},
        'root': {
            'level': 'INFO',
            'handlers': ['wsgi'],
        }
    })

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
