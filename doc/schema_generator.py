# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
"""
Generates automatic reStructuredText documentation from a configuration schema YAML file.
"""

import os
from typing import Any, Dict, Iterator, TextIO, Tuple

import yaml


def generate_docs():
    print('Generating documentation from schema')

    schema_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'source', 'config_schema.rst')

    # Read metadata
    schema_yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dace', 'config_schema.yml')
    with open(schema_yaml_path, 'r') as fp:
        metadata = yaml.load(fp.read(), Loader=yaml.SafeLoader)

    # Write .rst file
    with open(schema_path, 'w') as fp:
        fp.write('''
.. _config_schema:

Configuration entry reference
=============================

The following configuration entries are available in ``.dace.conf``, as part of the API, and as environment variables.
See :ref:`config` for more information on how to use the interface.
''')
        # Write configuration entries as table, organized by category
        for name, element in _traverse_metadata(metadata, ''):
            # Skip top-level element
            if not name:
                continue

            _write_entry_doc(fp, name, element)


def _traverse_metadata(metadata: Dict[str, Any], top_name: str = '') -> Iterator[Tuple[str, Dict[str, Any]]]:
    yield top_name, metadata
    if 'required' in metadata:
        ancestor_name = f'{top_name}.' if top_name else ''
        subelems: Dict[str, Any] = metadata['required']
        for key, val in sorted(subelems.items(), key=_sortkey):
            yield from _traverse_metadata(val, ancestor_name + key)


def _sortkey(item):
    # Sort values by first displaying the top-level keys, then the ones who have children
    k, v = item
    if isinstance(v, dict) and 'required' in v:
        return k
    else:
        return 'aaaaa' + k


def _write_entry_doc(fp: TextIO, name: str, element: Dict[str, Any]) -> None:
    TITLE_CHARACTERS = '-^~'

    # Category
    if 'required' in element:
        # Make appropriate title
        title_level = min(name.count('.'), len(TITLE_CHARACTERS))
        title_char = TITLE_CHARACTERS[title_level]
        fp.write(f'{name}\n{title_char * len(name)}\n')
        fp.write(f'{element["description"]}\n\n')
        return

    # Indent description entry from second line onwards
    desc: str = element['description']
    desc = '\n'.join(' ' * 8 + line if i > 0 else line for i, line in enumerate(desc.splitlines()))

    # Configuration entry
    fp.write(f'''
``{name}``: {element['title']}

    * **Type**: ``{element['type']}``
    * **Description**: {desc}
    * **Default value**: {element['default']}
''')
    # Write platform-specific defaults
    for k, v in element.items():
        if k.startswith('default_'):
            platform_name = k[len('default_'):]
            fp.write(f'    * **Default value (on {platform_name})**: {v}\n')

    # Footer
    fp.write('\n\n')


if __name__ == '__main__':
    generate_docs()
