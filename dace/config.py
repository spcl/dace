# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import contextlib
import os
import platform
import tempfile
import threading
import io
from typing import Any, Dict, Optional
import yaml
import warnings


@contextlib.contextmanager
def set_temporary(*path, value):
    """
    Temporarily set configuration value at ``path`` to value, and reset it after the context manager exits.

    Example::

        print(Config.get("compiler", "build_type")
        with set_temporary("compiler", "build_type", value="Debug"):
            print(Config.get("compiler", "build_type")
        print(Config.get("compiler", "build_type")
    """
    old_value = Config.get(*path)
    Config.set(*path, value=value)
    try:
        yield Config
    finally:
        Config.set(*path, value=old_value)


@contextlib.contextmanager
def temporary_config():
    """
    Creates a context where all configuration options changed will be reset when the context exits.

    Example::

        with temporary_config():
            Config.set("testing", "serialization", value=True)
            Config.set("optimizer", "autooptimize", value=True)
            foo()
    """
    with tempfile.TemporaryFile(mode='w+t') as fp:
        Config.save(file=fp)
        try:
            yield Config
        finally:
            fp.seek(0)  # rewind to the beginning of the file.
            Config.load(file=fp)


def _env2bool(envval):
    """
    Converts an arbitrary value to boolean.

    :param envval: Arbitrary value.
    :return: True if the input value matches a valid TRUE
             value, or False otherwise.
    """
    return str(envval).lower() in ['true', '1', 'y', 'yes', 'on', 'verbose']


def _add_defaults(config, metadata):
    """ Add defaults to configuration from metadata.

        :return: True if configuration was modified, False otherwise.
    """
    osname = platform.system()
    modified = False
    for k, v in metadata.items():
        # Recursive call for fields inside the dictionary
        if v['type'] == 'dict':
            if k not in config:
                modified = True
                config[k] = {}
            modified |= _add_defaults(config[k], v['required'])
            continue
        # Empty list initialization (if no default is specified)
        elif v['type'] == 'list':
            if k not in config and 'default' not in v:
                modified = True
                config[k] = []
                continue
        # Key does not exist in configuration, add default value
        if k not in config:
            modified = True
            # Per-OS default
            if 'default_' + osname in v:
                config[k] = v['default_' + osname]
            else:
                config[k] = v['default']
    return modified


class _ConfigData(threading.local):
    """Thread local data storage for the configuration scheme.

    Note that DaCe on its own is not thread safe, however, the tests
    impose this requirement.
    """

    def __init__(self) -> None:
        self.default_filename = '.dace.conf'
        self._config = {}
        self._config_metadata = {}
        self._cfg_filename = None
        self._default_cfg_path = None
        self._metadata_filename = None
        self._initialize()

    def cfg_filename(self):
        return self._cfg_filename

    def _initialize(self):
        """Initialize `self`, loads the specified configuration file.
        The function is automatically called by the constructor.
        """

        # If already initialized, skip
        if self._config_metadata:
            return

        # Override default configuration file path
        if 'DACE_CONFIG' in os.environ:
            default_cfg_filename = os.environ['DACE_CONFIG']
        else:
            home = os.path.expanduser("~")
            default_cfg_filename = os.path.join(home, self.default_filename)

        self._default_cfg_path = default_cfg_filename

        dace_path = os.path.dirname(os.path.abspath(__file__))
        self._metadata_filename = os.path.join(dace_path, 'config_schema.yml')

        # Load configuration schema (for validation and defaults)
        self.load_schema(filename=None)

        # Priority order: current working directory, default configuration file (DACE_CONFIG), then ~/.dace.conf
        for filename in [self.default_filename, default_cfg_filename]:
            self._cfg_filename = filename
            try:
                if os.path.isfile(filename):
                    self.load()
                    break
            except (FileNotFoundError, PermissionError, OSError):
                # If any filesystem-related error happened during file load, move on to next candidate
                continue
        else:
            # None of the files were found, load defaults from metadata
            self._cfg_filename = None
            self._config = {}
            _add_defaults(self._config, self._config_metadata['required'])

        # Check for old configurations to update the file
        if 'execution' in self._config and self._cfg_filename:
            # Reset config to only nondefault ones
            self.save(all=False)

    def load(self, filename: Optional[str] = None, file: Optional[io.FileIO] = None):
        if file is not None:
            assert filename is None
            self._config = yaml.load(file.read(), Loader=yaml.SafeLoader)
        else:
            with open(filename if filename else self._cfg_filename, 'r') as f:
                self._config = yaml.load(f.read(), Loader=yaml.SafeLoader)

        if self._config is None:
            self._config = {}

        # Add defaults from metadata
        _add_defaults(self._config, self._config_metadata['required'])

    def load_schema(self, filename: Optional[str] = None):
        if filename is None:
            filename = self._metadata_filename
        with open(filename, 'r') as f:
            self._config_metadata = yaml.load(f.read(), Loader=yaml.SafeLoader)

    def save(self, path: Optional[str] = None, all: bool = False, file: Optional[io.FileIO] = None):
        if path is None and file is None:
            path = self._cfg_filename
            if path is None:
                # Try to create a new config file in reversed priority order, and if all else fails keep config in memory
                for filename in [self._default_cfg_path, self.default_filename]:
                    try:
                        self.save(path=filename, all=all)
                        self._cfg_filename = filename
                        return
                    except (FileNotFoundError, PermissionError, OSError):
                        # If any filesystem-related error happened during file save, move on to next candidate
                        continue

                warnings.warn('No DaCe configuration file was able to be saved')
                return

        # Write configuration file
        if file is not None:
            yaml.dump(self._config if all else self.nondefaults(), file, default_flow_style=False)
        else:
            with open(path, 'w') as f:
                yaml.dump(self._config if all else self.nondefaults(), f, default_flow_style=False)

    def get_metadata(self, *key_hierarchy):
        # Support for "a.b.c" in calls
        if len(key_hierarchy) == 1 and '.' in key_hierarchy[0]:
            key_hierarchy = key_hierarchy[0].split('.')

        # Traverse the key hierarchy
        current_conf = self._config_metadata
        for key in key_hierarchy:
            current_conf = current_conf['required'][key]
        return current_conf

    def get_default(self, *key_hierarchy):
        # Support for "a.b.c" in calls
        if len(key_hierarchy) == 1 and '.' in key_hierarchy[0]:
            key_hierarchy = key_hierarchy[0].split('.')

        # Traverse the key hierarchy
        current_conf = self._config_metadata
        for key in key_hierarchy:
            current_conf = current_conf['required'][key]
        if 'default_' + platform.system() in current_conf:
            return current_conf['default_' + platform.system()]
        return current_conf['default']

    def get(self, *key_hierarchy):
        # Support for "a.b.c" in calls
        if len(key_hierarchy) == 1 and '.' in key_hierarchy[0]:
            key_hierarchy = key_hierarchy[0].split('.')

        # Environment variable override
        # NOTE: will only work if a specific key is accessed!
        envvar = 'DACE_' + '_'.join(key_hierarchy)
        if envvar in os.environ:
            return os.environ[envvar]

        # Traverse the key hierarchy
        current_conf = self._config
        for key in key_hierarchy:
            current_conf = current_conf[key]

        return current_conf

    def get_bool(self, *key_hierarchy):
        res = self.get(*key_hierarchy)
        if isinstance(res, bool):
            return res
        return _env2bool(str(res))

    def append(self, *key_hierarchy, value, autosave):
        # Support for "a.b.c" in calls
        if len(key_hierarchy) == 1 and '.' in key_hierarchy[0]:
            key_hierarchy = key_hierarchy[0].split('.')

        # Traverse the key hierarchy up until the next to last element
        current_conf = self._config
        for key in key_hierarchy[:-1]:
            current_conf = current_conf[key]

        current_conf[key_hierarchy[-1]] += value
        if autosave:
            self.save()

        return current_conf[key_hierarchy[-1]]

    def set(self, *key_hierarchy, value, autosave):
        # Support for "a.b.c" in calls
        if len(key_hierarchy) == 1 and '.' in key_hierarchy[0]:
            key_hierarchy = key_hierarchy[0].split('.')

        # Traverse the key hierarchy up until the next to last element
        current_conf = self._config
        for key in key_hierarchy[:-1]:
            current_conf = current_conf[key]

        current_conf[key_hierarchy[-1]] = value
        if autosave:
            self.save()

    def nondefaults(self) -> Dict[str, Any]:
        current_conf = self._config
        defaults = self._config_metadata
        system_default_key = 'default_' + platform.system()

        def traverse(conf: Dict[str, Any], defaults: Dict[str, Any], result: Dict[str, Any]):
            for k, v in conf.items():
                if k not in defaults:  # Configuration entry no longer exists
                    continue
                elif 'required' in defaults[k]:  # Traverse further
                    internal = {}
                    traverse(v, defaults[k]['required'], internal)
                    if internal:
                        result[k] = internal
                elif system_default_key in defaults[k]:
                    if v != defaults[k][system_default_key]:
                        result[k] = v
                elif 'default' in defaults[k]:
                    if v != defaults[k]['default']:
                        result[k] = v

        output = {}
        traverse(current_conf, defaults['required'], output)
        return output


class Config(object):
    """Interface to the DaCe hierarchical configuration file.

    :note: The data is stored inside a thread local, aka. `threading.local`,
        variable. This means that in the beginning every thread is initialized
        with the _default_ setting.
    """

    _data = _ConfigData()

    @staticmethod
    def cfg_filename():
        """
        Returns the current configuration file path.
        """
        return Config._data.cfg_filename()

    @staticmethod
    def load(filename: Optional[str] = None, file: Optional[io.FileIO] = None):
        """
        Loads a configuration from an existing file.

        :param filename: The file to load. If unspecified,
                         uses default configuration file.
        :param file: Load the configuration from the file object.
        """
        return Config._data.load(filename=filename, file=file)

    @staticmethod
    def load_schema(filename: Optional[str] = None):
        """
        Loads a configuration schema from an existing file.

        :param filename: The file to load. If unspecified,
                         uses default schema file.
        """
        return Config._data.load(filename=filename)

    @staticmethod
    def save(path: Optional[str] = None, all: bool = False, file: Optional[io.FileIO] = None):
        """
        Saves the current configuration to a file.

        :param path: The file to save to. If unspecified,
                     uses default configuration file.
        :param all: If False, only saves non-default configuration entries.
                    Otherwise saves all entries.
        :param file: A file object to use directly.
        """
        return Config._data.save(path=path, all=all, file=file)

    @staticmethod
    def get_metadata(*key_hierarchy):
        """ Returns the configuration specification of a given entry
            from the schema.

            :param key_hierarchy: A tuple of strings leading to the
                                  configuration entry.
                                  For example: ('a', 'b', 'c') would be
                                  configuration entry c which is in the
                                  path a->b.
            :return: Configuration specification as a dictionary.
        """
        return Config._data.get_metadata(*key_hierarchy)

    @staticmethod
    def get_default(*key_hierarchy):
        """ Returns the default value of a given configuration entry.
            Takes into accound current operating system.

            :param key_hierarchy: A tuple of strings leading to the
                                  configuration entry.
                                  For example: ('a', 'b', 'c') would be
                                  configuration entry c which is in the
                                  path a->b.
            :return: Default configuration value.
        """
        return Config._data.get_default(*key_hierarchy)

    @staticmethod
    def get(*key_hierarchy):
        """
        Returns the current value of a given configuration entry.

        :param key_hierarchy: A tuple of strings leading to the
                                configuration entry.
                                For example: ('a', 'b', 'c') would be
                                configuration entry c which is in the
                                path a->b.
        :return: Configuration entry value.
        """
        return Config._data.get(*key_hierarchy)

    @staticmethod
    def get_bool(*key_hierarchy):
        """ Returns the current value of a given boolean configuration entry.
            This specialization allows more string types to be converted to
            boolean, e.g., due to environment variable overrides.

            :param key_hierarchy: A tuple of strings leading to the
                                  configuration entry.
                                  For example: ('a', 'b', 'c') would be
                                  configuration entry c which is in the
                                  path a->b.
            :return: Configuration entry value (as a boolean).
        """
        return Config._data.get_bool(*key_hierarchy)

    @staticmethod
    def append(*key_hierarchy, value=None, autosave=False):
        """
        Appends to the current value of a given configuration entry
        and sets it.

        :param key_hierarchy: A tuple of strings leading to the
                                configuration entry.
                                For example: ('a', 'b', 'c') would be
                                configuration entry c which is in the
                                path a->b.
        :param value: The value to append.
        :param autosave: If True, saves the configuration to the file
                            after modification.
        :return: Current configuration entry value.

        Examples::

            Config.append('compiler', 'cpu', 'args', value='-fPIC')
        """
        return Config._data.append(*key_hierarchy, value=value, autosave=autosave)

    @staticmethod
    def set(*key_hierarchy, value=None, autosave=False):
        """
        Sets the current value of a given configuration entry.

        :param key_hierarchy: A tuple of strings leading to the
                              configuration entry.
                              For example: ('a', 'b', 'c') would be
                              configuration entry c which is in the
                              path a->b.
        :param value: The value to set.
        :param autosave: If True, saves the configuration to the file
                         after modification.

        Examples::

            Config.set('profiling', value=True)
        """
        return Config._data.set(*key_hierarchy, value=value, autosave=autosave)

    def nondefaults(self) -> Dict[str, Any]:
        return Config._data.nondefaults()
