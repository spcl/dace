# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
"""
Generates automatic reStructuredText documentation from a configuration schema YAML file.
"""

import os


def generate_docs():
    print('Generating documentation from schema')

    schema_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'source', 'config_schema.rst')

    with open(schema_path, 'w') as fp:
        fp.write('''
.. _config_schema:

Configuration entry reference
=============================

The following configuration entries are available in ``.dace.conf``, as part of the API, and as environment variables.
See :ref:`config` for more information on how to use the interface.
''')
        # TODO: Write configuration entries as table, organized by category
