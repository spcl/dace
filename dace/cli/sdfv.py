# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" SDFG visualizer that uses Flask, HTML5, and Javascript. """

import json
import tempfile
import sys
import os
import platform

import dace
import tempfile
import jinja2


def view(sdfg, filename=None):
    """View an sdfg in the system's HTML viewer
       :param sdfg: the sdfg to view, either as `dace.SDFG` object or a json string
       :param filename: the filename to write the HTML to. If `None`, a temporary file will be created.
    """
    if type(sdfg) is dace.SDFG:
        sdfg = dace.serialize.dumps(sdfg.to_json())

    basepath = os.path.join(os.path.dirname(os.path.realpath(dace.__file__)), 'viewer')
    template_loader = jinja2.FileSystemLoader(searchpath=os.path.join(basepath, 'templates'))
    template_env = jinja2.Environment(loader=template_loader)
    template = template_env.get_template('sdfv.html')

    html = template.render(sdfg=json.dumps(sdfg), dir=basepath + '/')

    if filename is None:
        fd, html_filename = tempfile.mkstemp(suffix=".sdfg.html")
    else:
        fd = None
        html_filename = filename + ".html"

    with open(html_filename, "w") as f:
        f.write(html)

    print("File saved at %s" % html_filename)

    system = platform.system()

    if system == 'Windows':
        os.system(html_filename)
    elif system == 'Darwin':
        os.system('open %s' % html_filename)
    else:
        os.system('xdg-open %s' % html_filename)

    if fd is not None:
        os.close(fd)


def main():
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

    sdfg_json = None

    # Open JSON file directly
    with open(filename, 'rb') as fp:
        firstbyte = fp.read(1)
        fp.seek(0)
        if firstbyte == b'{':
            sdfg_json = fp.read().decode('utf-8')

    # Load SDFG
    if sdfg_json is None:
        sdfg = dace.SDFG.from_file(filename)
        view(sdfg, filename)
    else:
        view(sdfg_json, filename)


if __name__ == '__main__':
    main()
