""" SDFG visualizer that uses Flask, HTML5, and Javascript. """

import json
import sys
import os
import platform

import dace
import tempfile
import jinja2

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

    basepath = os.path.dirname(__file__)
    template_loader = jinja2.FileSystemLoader(
        searchpath=os.path.join(basepath, 'templates'))
    template_env = jinja2.Environment(loader=template_loader)
    template = template_env.get_template('sdfv.html')

    with tempfile.NamedTemporaryFile(
            dir=basepath, prefix='sdfv_tmp', suffix='.html',
            delete=False) as fp:
        fp.write(template.render(sdfg=json.dumps(sdfg_json)).encode('utf-8'))
        filename = fp.name

    system = platform.system()

    if system == 'Windows':
        os.system(filename)
    elif system == 'Darwin':
        os.system('open %s' % filename)
    else:
        os.system('xdg-open %s' % filename)

    print('Running in browser, press any key to exit...')
    sys.stdin.read(1)

    os.unlink(filename)
