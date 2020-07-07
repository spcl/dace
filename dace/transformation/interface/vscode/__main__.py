from flask import Flask

deamon = Flask('dace.transformation.interface.vscode')
deamon.config['DEBUG'] = True

@deamon.route('/', methods=['GET'])
def root():
    return 'success!'

if __name__ == '__main__':
    deamon.run()