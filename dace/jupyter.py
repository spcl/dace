# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Jupyter Notebook support for DaCe. """

import os
import urllib.request
import urllib.error


def _connected():
    try:
        urllib.request.urlopen('https://spcl.github.io/dace/webclient2/dist/sdfv.js', timeout=1)
        return True
    except urllib.error.URLError:
        return False


# From https://stackoverflow.com/a/39662359/6489142
def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def preamble():
    # Emit javascript headers for SDFG renderer
    sdfv_js_deps = ['sdfv.js']
    sdfv_css_deps = ['sdfv.css']

    result = ''

    # Rely on internet connection for Material icons
    result += '<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">'

    # Try to load dependencies from online sources
    if _connected():
        for dep in sdfv_js_deps:
            result += '<script src="https://spcl.github.io/dace/webclient2/dist/%s"></script>\n' % dep
        for dep in sdfv_css_deps:
            result += '<link href="https://spcl.github.io/dace/webclient2/%s" rel="stylesheet">\n' % dep
        return result

    # Load local dependencies
    root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dace', 'viewer', 'webclient')
    for dep in sdfv_js_deps:
        file = os.path.join(root_path, 'dist', dep)
        with open(file, 'r') as fp:
            result += '<script>%s</script>\n' % fp.read()
    for dep in sdfv_css_deps:
        file = os.path.join(root_path, dep)
        with open(file, 'r') as fp:
            result += '<style>%s</style>\n' % fp.read()

    # Run this code once
    return result


def enable():
    from IPython.display import display, HTML
    display(HTML(preamble()))


# Code that runs on "import dace"
if isnotebook():
    enable()
