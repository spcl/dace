from argparse import ArgumentParser

def get_transformations(sdfg):
    print("ok")

def run_daemon():
    from flask import Flask

    deamon = Flask('dace.transformation.interface.vscode')
    deamon.config['DEBUG'] = False

    @deamon.route('/', methods=['GET'])
    def root():
        return 'success!'

    deamon.run()

if __name__ == '__main__':
    parser = ArgumentParser()

    '''
    parser.add_argument('-d',
                        '--daemon',
                        action='store_true',
                        help='Run as a daemon')
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