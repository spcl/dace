""" SDFG visualizer that uses Flask, HTML5, and Javascript. """

import sys
import os

import dace

# Create Flask (Web interface) application
from flask import Flask, render_template, jsonify, request
app = Flask(__name__, static_url_path='', static_folder='client')

sdfg = None


@app.route('/', methods=['GET'])
def main():
    return render_template('sdfv.html', sdfg=sdfg.toJSON())


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('USAGE: sdfv <PATH TO SDFG FILE>')
        exit(1)
    if os.path.isdir(sys.argv[1]):
        filename = os.path.join(sys.argv[1], 'program.sdfg')
    else:
        filename = sys.argv[1]

    if not os.path.isfile(filename):
        print('SDFG file', filename, 'not found')
        exit(2)

    sdfg = dace.SDFG.from_file(filename)
    app.run(port=5799)
