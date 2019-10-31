""" SDFG visualizer that uses Flask, HTML5, and Javascript. """

import json
import sys
import os

import dace

# Create Flask (Web interface) application
from flask import Flask, render_template, jsonify, request
app = Flask(__name__, static_url_path='', static_folder='client')

sdfg_json = None


@app.route('/', methods=['GET'])
def main():
    return render_template('sdfv.html', sdfg=json.dumps(sdfg_json))


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

    # Open JSON file directly
    with open(filename, 'rb') as fp:
        firstbyte = fp.read(1)
        fp.seek(0)
        if firstbyte == b'{':
            sdfg_json = fp.read().decode('utf-8')

    # Load SDFG
    if sdfg_json is None:
        sdfg = dace.SDFG.from_file(filename)
        sdfg_json = sdfg.to_json()

    app.run(port=5799)
