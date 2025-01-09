# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" SDFG visualizer that uses Jinja, HTML5, and Javascript. """

import json
import tempfile
import sys
import os
import platform
from typing import Optional, Union
import functools
import http.server
import threading

import dace
import tempfile


def partialclass(cls, *args, **kwds):
    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwds)

    return NewCls


def view(sdfg: dace.SDFG, filename: Optional[Union[str, int]] = None, verbose: bool = True, compress: bool = True):
    """
    View an sdfg in the system's HTML viewer

    :param sdfg: the sdfg to view, either as `dace.SDFG` object or a json string
    :param filename: the filename to write the HTML to. If `None`, a
                    temporary file will be created. If an integer,
                    the generated HTML and related sources will be
                    served using a basic web server on that port,
                    blocking the current thread.
    :param verbose: Be verbose.
    :param compress: Use compression for the temporary file.
    """
    # If vscode is open, try to open it inside vscode
    if filename is None:
        if (
            'VSCODE_IPC_HOOK' in os.environ
            or 'VSCODE_IPC_HOOK_CLI' in os.environ
            or 'VSCODE_GIT_IPC_HANDLE' in os.environ
        ):
            suffix = '.sdfgz' if compress else '.sdfg'
            fd, filename = tempfile.mkstemp(suffix=suffix)
            sdfg.save(filename, compress=compress)
            if platform.system() == 'Darwin':
                # Special case for MacOS
                os.system(f'open {filename}')
            else:
                os.system(f'code {filename}')
            os.close(fd)
            return

    try:
        import jinja2
    except (ImportError, ModuleNotFoundError):
        raise ImportError('SDFG.view() requires jinja2, please install by running `pip install jinja2`')

    if type(sdfg) is dace.SDFG:
        sdfg = dace.serialize.dumps(sdfg.to_json())

    if filename is None:
        fd, html_filename = tempfile.mkstemp(suffix=".sdfg.html")
    elif isinstance(filename, int):
        dirname = tempfile.mkdtemp()
        html_filename = os.path.join(dirname, "sdfg.html")
        fd = None
    else:
        fd = None
        html_filename = filename + ".html"

    basepath = os.path.join(os.path.dirname(os.path.realpath(dace.__file__)), 'viewer')
    template_loader = jinja2.FileSystemLoader(searchpath=os.path.join(basepath, 'templates'))
    template_env = jinja2.Environment(loader=template_loader)
    template = template_env.get_template('sdfv.html')

    # if we are serving, the base path should just be root
    html = template.render(sdfg=json.dumps(sdfg), dir="/" if isinstance(filename, int) else (basepath + '/'))

    with open(html_filename, "w") as f:
        f.write(html)

    if(verbose):
        print("File saved at %s" % html_filename)

    if fd is not None:
        os.close(fd)

    if isinstance(filename, int):
        # link in the web resources
        os.symlink(os.path.join(basepath, 'webclient'), os.path.join(dirname, 'webclient'))

        # start the web server
        handler = partialclass(http.server.SimpleHTTPRequestHandler, directory=dirname)
        httpd = http.server.HTTPServer(('localhost', filename), handler)
        if(verbose):
            print(f"Serving at localhost:{filename}, press enter to stop...")

        # start the server in a different thread
        def serve():
            httpd.serve_forever()

        thread = threading.Thread(target=serve)
        thread.start()

        # wait for user input
        input()

        # kill the server
        httpd.shutdown()
        httpd.server_close()
        thread.join()
        print("Server shutdown")
    else:
        system = platform.system()

        if system == 'Windows':
            os.system(html_filename)
        elif system == 'Darwin':
            os.system('open %s' % html_filename)
        else:
            os.system('xdg-open %s' % html_filename)


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
