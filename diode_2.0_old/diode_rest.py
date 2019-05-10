#!flask/bin/python

import dace
import parse
from flask import Flask, request, abort, make_response, jsonify

app = Flask(__name__)


@app.route('/dace/api/v1.0/compile', methods=['POST'])
def compile():
    if not request.json or not 'code' in request.json:
        print("abort")
        abort(400)
    code = request.json['code']
    print("Code is: " + code)
    statements = parse.parse(code, debug=False)
    print("After Parsing")
    print(str(statements))
    print("===============")
    statements.provide_parents()
    statements.specialize()
    print("After Specialization:")
    print(str(statements))
    print("===============")
    sdfg = statements.generate_code()
    sdfg.set_sourcecode(code, "matlab")
    json = sdfg.toJSON()
    return jsonify({"sdfg": json})


@app.route('/dace/api/v1.0/status', methods=['POST'])
def status():
    # just a kind of ping/pong to see if the server is running
    return "OK"


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
